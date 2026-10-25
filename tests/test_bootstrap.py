"""Recover the exact partial-venv/ensurepip failure without modifying system Python."""
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.bootstrap_venv import Bootstrap, GET_PIP_URL


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = self.root / 'venv'
        (self.target / 'bin').mkdir(parents=True)
        (self.target / 'pyvenv.cfg').write_text('fixture')
        (self.target / 'bin/python').write_text('fixture')
        self.marker = self.target / 'preserve.txt'
        self.marker.write_text('keep existing files')
        self.bootstrap = Bootstrap(self.target, self.root / 'logs')

    def result(self, code):
        return subprocess.CompletedProcess([],code,stdout='',stderr='ensurepip fixture error' if code else '')

    def test_healthy_environment_does_not_reseed_or_download(self):
        with patch.object(self.bootstrap,'verify_target'), patch.object(self.bootstrap,'pip_ready',return_value=True), \
             patch.object(self.bootstrap,'run') as run, patch('urllib.request.urlopen') as download:
            self.assertEqual(self.bootstrap.install(),self.target/'bin/python')
        run.assert_not_called(); download.assert_not_called()
        self.assertTrue(self.marker.exists())

    def test_partial_environment_ensurepip_failure_recovers_with_base_pip(self):
        with patch.object(self.bootstrap,'verify_target'), patch.object(self.bootstrap,'pip_ready',side_effect=[False,True]), \
             patch.object(self.bootstrap,'run',side_effect=[self.result(1),self.result(0)]) as run, \
             patch('urllib.request.urlopen') as download:
            self.bootstrap.install()
        self.assertEqual([call.args[0] for call in run.call_args_list],['ensurepip','base-pip-to-venv'])
        command=run.call_args_list[1].args[1]
        self.assertEqual(command[command.index('--python')+1],self.target/'bin/python')
        download.assert_not_called()
        self.assertEqual(self.marker.read_text(),'keep existing files')

    def test_missing_base_pip_uses_official_bootstrap_only_inside_verified_venv(self):
        with patch.object(self.bootstrap,'verify_target') as verify, \
             patch.object(self.bootstrap,'pip_ready',side_effect=[False,True]), \
             patch.object(self.bootstrap,'run',side_effect=[self.result(1),self.result(1),self.result(0)]) as run, \
             patch('urllib.request.urlopen',return_value=io.BytesIO(b'# fixture installer')) as download, patch('builtins.print'):
            self.bootstrap.install()
        verify.assert_called_once()
        download.assert_called_once_with(GET_PIP_URL,timeout=60)
        self.assertEqual(run.call_args_list[-1].args[1][0],self.target/'bin/python')
        self.assertTrue((self.bootstrap.logs/'download.json').is_file())
        self.assertTrue(self.marker.exists())

    def test_unverified_environment_cannot_install_or_download(self):
        with patch.object(self.bootstrap,'verify_target',side_effect=RuntimeError('not isolated')), \
             patch.object(self.bootstrap,'pip_ready') as pip, patch('urllib.request.urlopen') as download:
            with self.assertRaisesRegex(RuntimeError,'not isolated'): self.bootstrap.install()
        pip.assert_not_called(); download.assert_not_called()

    def test_network_failure_keeps_existing_directory(self):
        with patch.object(self.bootstrap,'verify_target'), patch.object(self.bootstrap,'pip_ready',return_value=False), \
             patch.object(self.bootstrap,'run',return_value=self.result(1)), \
             patch('urllib.request.urlopen',side_effect=OSError('network unavailable')), patch('builtins.print'):
            with self.assertRaisesRegex(OSError,'network unavailable'): self.bootstrap.install()
        self.assertEqual(self.marker.read_text(),'keep existing files')

    def test_real_creation_is_isolated_without_ensurepip(self):
        # This must also pass on servers where ensurepip is missing: that is the
        # failure first.sh now recovers from, not an extra server prerequisite.
        target=self.root/'real-venv'
        subprocess.run([sys.executable,'-m','venv','--without-pip',str(target)],check=True,capture_output=True)
        marker=target/'preserved.txt'; marker.write_text('partial creation')
        bootstrap=Bootstrap(target,self.root/'real-logs')
        with patch('urllib.request.urlopen',side_effect=AssertionError('offline test must not download')):
            bootstrap.verify_target()
            self.assertFalse(bootstrap.pip_ready('expected-missing-pip'))
        self.assertEqual(marker.read_text(),'partial creation')


if __name__ == '__main__':
    unittest.main()
