"""Исполнитель экспериментов для Linux/WSL без защитной изоляции.

Сгенерированный код сохраняет доступ к файлам и сети. Потребление памяти группы
процессов (RSS) измеряется периодически; это не жёсткий лимит через cgroups.
Намеренный выход кода из группы процессов не поддерживается.
Импортов из проекта нет: этот файл можно разместить в корне архива решения.
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Literal


@dataclass(frozen=True)
class ExecutionResult:
    status: Literal['success', 'error', 'timeout', 'memory_limit']
    stdout: str
    stderr: str
    exit_code: int | None
    duration_sec: float
    peak_memory_mb: float
    output_files: tuple[str, ...]
    missing_outputs: tuple[str, ...]
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error: str | None = None


_BOOTSTRAP = '''
import os, resource, runpy, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:int(sys.argv[2])])
script = sys.argv[1]
sys.argv = [script]
runpy.run_path(script, run_name='__main__')
'''


def _group_rss(group_id: int) -> int:
    pages = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[2]) == group_id:
                pages += int(fields[21])
        except (OSError, ValueError, IndexError):
            continue
    return pages * os.sysconf('SC_PAGE_SIZE')


def _kill_group(group_id: int) -> None:
    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_experiment(code: str, work_dir: str | Path, timeout_sec: float,
                   memory_limit_mb: int = 4096, cpu_limit: int = 1, *,
                   expected_outputs: tuple[str, ...] = (),
                   max_output_bytes: int = 65536,
                   cancel_event=None) -> ExecutionResult:
    """Запустить код текущим Python и вернуть диагностику и пути к результатам.

    Для каждого эксперимента нужна отдельная папка. Входные данные готовит
    вызывающий модуль. Пути ожидаемых выходных файлов должны быть относительными,
    а сами файлы до запуска должны отсутствовать. Проверяется только их наличие:
    формат и метрики вызывающий модуль должен проверять независимо.
    Неверные параметры вызывают ValueError; ошибки выполнения возвращаются
    в объекте результата. Ключи API из окружения не передаются, но это
    НЕ запрещает коду читать файлы с секретами.
    Дочерние процессы наследуют привязку к ядрам CPU и настройки числа потоков.
    Вызывающий модуль выделяет ресурсы с учётом оставшегося бюджета задачи.
    RSS измеряется примерно каждые 20 мс: общие страницы памяти могут учитываться
    повторно, а краткие скачки потребления — пропускаться. Для жёстких лимитов
    нужны ограничения на уровне ОС.
    """
    if sys.platform != 'linux':
        raise ValueError('Use Linux/WSL Python')
    if not isinstance(code, str) or not code.strip():
        raise ValueError('code must be a non-empty string')
    if (isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(timeout_sec) or timeout_sec <= 0):
        raise ValueError('timeout_sec must be finite and positive')
    for name, value in [('memory_limit_mb', memory_limit_mb),
                        ('cpu_limit', cpu_limit), ('max_output_bytes', max_output_bytes)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    try:
        root = Path(work_dir).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError('work_dir must be an existing accessible directory') from exc
    if not root.is_dir():
        raise ValueError('work_dir must be an existing directory')
    outputs = []
    for name in expected_outputs:
        rel = Path(name)
        path = root / rel
        if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(root):
            raise ValueError('Output paths must stay inside work_dir')
        if path.exists() or path.is_symlink():
            raise ValueError(f'Output already exists: {name}; use a fresh directory')
        outputs.append((name, path))
    started = time.monotonic()
    deadline = started + timeout_sec
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    truncated = {'stdout': False, 'stderr': False}
    status, error, peak, process = 'error', None, 0, None
    env = {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
           'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
           'PYTHONNOUSERSITE': '1'}
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env[key] = str(cpu_limit)
    try:
        with tempfile.TemporaryDirectory(prefix='.executor-', dir=root) as scratch:
            script = Path(scratch) / 'experiment.py'
            script.write_text(code, encoding='utf-8')
            with selectors.DefaultSelector() as selector:
                process = subprocess.Popen(
                    [sys.executable, '-u', '-c', _BOOTSTRAP, str(script), str(cpu_limit)],
                    cwd=root, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                try:
                    for name, pipe in [('stdout', process.stdout), ('stderr', process.stderr)]:
                        os.set_blocking(pipe.fileno(), False)
                        selector.register(pipe, selectors.EVENT_READ, name)
                    stopped, drain_deadline = False, None
                    while True:
                        now = time.monotonic()
                        if not stopped:
                            rss = _group_rss(process.pid)
                            peak = max(peak, rss)
                            if cancel_event is not None and cancel_event.is_set():
                                status, error = 'error', 'Experiment cancelled'
                                stopped = True
                            elif rss > memory_limit_mb * 1024 * 1024:
                                status, error = 'memory_limit', 'Process group exceeded sampled RSS limit'
                                stopped = True
                            elif process.poll() is not None:
                                status = 'success' if process.returncode == 0 else 'error'
                                if status == 'error':
                                    error = f'Python exited with code {process.returncode}'
                                stopped = True
                            elif now >= deadline:
                                status, error = 'timeout', 'Experiment exceeded wall-clock limit'
                                stopped = True
                            if stopped:
                                _kill_group(process.pid)
                                drain_deadline = now + 1.0
                        if stopped and (not selector.get_map() or now >= drain_deadline):
                            break
                        # Bound work per read so continuous output cannot starve supervision.
                        for key, _ in selector.select(timeout=.02):
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            target = buffers[key.data]
                            target.extend(chunk)
                            if len(target) > max_output_bytes:
                                del target[:-max_output_bytes]
                                truncated[key.data] = True
                finally:
                    _kill_group(process.pid)
                    process.wait(timeout=5)
                    process.stdout.close()
                    process.stderr.close()
    except (OSError, subprocess.SubprocessError) as exc:
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    found, missing = [], []
    for name, path in outputs:
        if path.is_file() and path.resolve().is_relative_to(root):
            found.append(str(path.resolve()))
        else:
            missing.append(name)
    if status == 'success' and missing:
        status, error = 'error', 'Expected outputs are missing or outside work_dir'
    return ExecutionResult(
        status=status, stdout=buffers['stdout'].decode('utf-8', errors='replace'),
        stderr=buffers['stderr'].decode('utf-8', errors='replace'),
        exit_code=None if process is None else process.returncode,
        duration_sec=time.monotonic() - started, peak_memory_mb=peak / (1024 * 1024),
        output_files=tuple(found), missing_outputs=tuple(missing),
        stdout_truncated=truncated['stdout'], stderr_truncated=truncated['stderr'], error=error)
