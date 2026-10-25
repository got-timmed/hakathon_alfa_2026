"""Проверки отказов и очистки ресурсов исполнителя, без сети."""
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.sandbox import run_experiment
import agent.sandbox as sandbox


@unittest.skipUnless(sys.platform == 'linux', 'Требуется Linux/WSL')
class FailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def execute(self, code, **options):
        settings = dict(timeout_sec=3, memory_limit_mb=256, cpu_limit=1)
        settings.update(options)
        return run_experiment(code, self.root, **settings)

    def test_python_failures(self):
        cases = [
            ('def broken(:', 'SyntaxError'),
            ('import nonexistent_hackathon_test_module', 'ModuleNotFoundError'),
            ("open('missing_directory/result.csv', 'w')", 'FileNotFoundError'),
            ("open('.', 'w')", 'IsADirectoryError'),
            ('input()', 'EOFError'),
        ]
        for code, message in cases:
            with self.subTest(message=message):
                result = self.execute(code)
                self.assertEqual(result.status, 'error', result)
                self.assertIn(message, result.stderr)
                self.assertFalse(list(self.root.glob('.executor-*')))

    def test_abrupt_exit(self):
        result = self.execute('import os, signal; os.kill(os.getpid(), signal.SIGKILL)')
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.exit_code, -signal.SIGKILL)

    def test_launch_failure(self):
        with patch.object(sandbox.subprocess, 'Popen', side_effect=OSError('test launch error')):
            result = self.execute('pass')
        self.assertEqual(result.status, 'error')
        self.assertIsNone(result.exit_code)
        self.assertIn('test launch error', result.error)
        self.assertFalse(list(self.root.glob('.executor-*')))

    def test_repeated_runs_no_fd_leak(self):
        before = len(os.listdir('/proc/self/fd'))
        for index in range(12):
            result = self.execute('print(1)' if index % 2 else 'raise ValueError()')
            self.assertEqual(result.status, 'success' if index % 2 else 'error')
        self.assertLessEqual(len(os.listdir('/proc/self/fd')), before)
        self.assertFalse(list(self.root.glob('.executor-*')))

    def test_child_memory_counted(self):
        code = """
import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; a=bytearray(80*1024*1024); time.sleep(10)'])
child.wait()
"""
        result = self.execute(code, memory_limit_mb=60)
        self.assertEqual(result.status, 'memory_limit', result)

    def test_multiple_children_stopped(self):
        code = """
import subprocess, sys, time
from pathlib import Path
children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']) for _ in range(3)]
Path('pids').write_text(','.join(str(p.pid) for p in children))
time.sleep(10)
"""
        result = self.execute(code, timeout_sec=.5)
        self.assertEqual(result.status, 'timeout')
        for pid in (self.root / 'pids').read_text().split(','):
            self.assert_stopped(int(pid))

    def assert_stopped(self, pid):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                stat = Path(f'/proc/{pid}/stat').read_text()
            except FileNotFoundError:
                return
            if stat[stat.rfind(')') + 2:].split()[0] == 'Z':
                return  # Остановленный процесс ожидает сбора родителем/init.
            time.sleep(.01)
        self.fail(f'Процесс {pid} продолжает работать')

    def test_interrupt_cleans_up_and_propagates(self):
        real_rss = sandbox._group_rss
        pids = []
        def interrupted(group_id):
            pids.append(group_id)
            real_rss(group_id)
            raise KeyboardInterrupt()
        with patch.object(sandbox, '_group_rss', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.execute('import time; time.sleep(10)')
        self.assert_stopped(pids[0])
        self.assertFalse(list(self.root.glob('.executor-*')))

    def test_output_symlink_loop_returns_error(self):
        code = "from pathlib import Path; Path('loop').symlink_to('loop'); Path('result').symlink_to('loop')"
        result = self.execute(code, expected_outputs=('result',))
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.missing_outputs, ('result',))

    def test_partial_output_does_not_make_failed_run_successful(self):
        result = self.execute("from pathlib import Path; Path('partial').write_text('partial'); raise RuntimeError()", expected_outputs=('partial',))
        self.assertEqual(result.status, 'error')
        self.assertEqual(len(result.output_files), 1)

    def test_binary_output_decodes_safely(self):
        result = self.execute("import os; os.write(1, b'\\xff\\xfe'); os.write(2, b'\\xff')")
        self.assertEqual(result.status, 'success')
        self.assertIn('\ufffd', result.stdout)

    def test_invalid_configuration_consistently_raises_value_error(self):
        for limit in (None, '3', True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.execute('pass', timeout_sec=limit)

    def test_missing_work_directory_is_configuration_error(self):
        with self.assertRaises(ValueError):
            run_experiment('pass', self.root / 'absent', 1)

    def test_tiny_budget(self):
        result = self.execute('import time; time.sleep(5)', timeout_sec=.001)
        self.assertEqual(result.status, 'timeout')
        self.assertLess(result.duration_sec, 2)


if __name__ == '__main__':
    unittest.main()
