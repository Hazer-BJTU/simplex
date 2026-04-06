import os
import sys
import signal
import threading
import subprocess

from typing import List, Optional


class CommandProcess:
    def __init__(self, cmd: List[str] | str, shell: bool = True):
        self.cmd = cmd
        self.shell = shell
        self.proc: Optional[subprocess.Popen] = None
        self.output: List[str] = []

    def _capture_output(self):
        for line in iter(self.proc.stdout.readline, ''):
            if line:
                self.output.append(line.strip())

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.cmd,
            shell = self.shell,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            bufsize = 1,
            universal_newlines = True
        )
        threading.Thread(target=self._capture_output, daemon=True).start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout = 3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        return False
    
if __name__ == '__main__':
    pass
