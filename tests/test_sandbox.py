import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from agent.sandbox import run_experiment


@unittest.skipUnless(sys.platform == 'linux', 'Linux/WSL runner')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_code(self, code, **kwargs):
        options = dict(timeout_sec=5, memory_limit_mb=256, cpu_limit=1)
        options.update(kwargs)
        return run_experiment(code, self.root, **options)

    def test_success_and_artifact(self):
        result = self.run_code("from pathlib import Path\nprint('привет')\nPath('pred.csv').write_text('id,y\\n1,2\\n')", expected_outputs=('pred.csv',))
        self.assertEqual(result.status, 'success', result)
        self.assertEqual(result.stdout.strip(), 'привет')
        self.assertEqual(result.output_files, (str(self.root / 'pred.csv'),))

    def test_exception(self):
        result = self.run_code("raise RuntimeError('test failure')")
        self.assertEqual(result.status, 'error')
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('test failure', result.stderr)

    def test_timeout(self):
        result = self.run_code('while True: pass', timeout_sec=.2)
        self.assertEqual(result.status, 'timeout')
        self.assertLess(result.duration_sec, 3)

    def test_memory(self):
        result = self.run_code("import time\na=[]\nwhile True:\n a.append(bytearray(4*1024*1024))\n time.sleep(.02)", memory_limit_mb=40)
        self.assertEqual(result.status, 'memory_limit', result)

    def test_output_is_bounded_and_drained(self):
        result = self.run_code("import sys\nfor _ in range(500):\n print('x'*1000)\n print('y'*1000, file=sys.stderr)", max_output_bytes=1024)
        self.assertEqual(result.status, 'success', result)
        self.assertLessEqual(len(result.stdout), 1024)
        self.assertLessEqual(len(result.stderr), 1024)
        self.assertTrue(result.stdout_truncated and result.stderr_truncated)

    def test_credentials_and_cpu_affinity(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'TEST_SECRET', 'UNRELATED_SECRET': 'TEST_SECRET'}):
            result = self.run_code("import os\nassert 'OPENAI_API_KEY' not in os.environ\nassert 'UNRELATED_SECRET' not in os.environ\nassert len(os.sched_getaffinity(0)) == 1\nassert os.environ['OMP_NUM_THREADS'] == '1'")
        self.assertEqual(result.status, 'success', result)

    def test_missing_and_stale_artifacts(self):
        result = self.run_code('pass', expected_outputs=('missing.csv',))
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.missing_outputs, ('missing.csv',))
        (self.root / 'old.csv').write_text('old')
        with self.assertRaises(ValueError):
            self.run_code('pass', expected_outputs=('old.csv',))

    def test_escaping_output_rejected(self):
        with self.assertRaises(ValueError):
            self.run_code('pass', expected_outputs=('../outside.csv',))
        result = self.run_code("from pathlib import Path\nPath('outside').symlink_to('/etc/hosts')", expected_outputs=('outside',))
        self.assertEqual(result.status, 'error')
        self.assertFalse(result.output_files)

    def test_child_process_cleanup_on_timeout_and_normal_exit(self):
        for parent_tail, status in [('time.sleep(10)', 'timeout'), ('time.sleep(.15)', 'success')]:
            with self.subTest(parent_tail=parent_tail):
                code = ("import subprocess, sys, time\n"
                        "subprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(.8); Path('child_survived').touch()\"])\n" + parent_tail)
                result = self.run_code(code, timeout_sec=.4)
                self.assertEqual(result.status, status, result)
                time.sleep(.9)
                self.assertFalse((self.root / 'child_survived').exists())

    def test_invalid_limits(self):
        for options in [dict(timeout_sec=0), dict(timeout_sec=float('nan')),
                        dict(cpu_limit=0), dict(memory_limit_mb=-1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_code('pass', **options)

    def test_continuous_output_cannot_bypass_timeout(self):
        result = self.run_code("while True: print('x'*1000)", timeout_sec=.2, max_output_bytes=128)
        self.assertEqual(result.status, 'timeout')
        self.assertLessEqual(len(result.stdout), 128)

    def test_flat_package_import(self):
        import shutil
        source = Path(__file__).resolve().parents[1] / 'agent' / 'sandbox.py'
        shutil.copyfile(source, self.root / 'sandbox.py')
        result = self.run_code("from sandbox import ExecutionResult\nprint(ExecutionResult.__name__)")
        self.assertEqual(result.status, 'success', result)
        self.assertIn('ExecutionResult', result.stdout)


if __name__ == '__main__':
    unittest.main()
