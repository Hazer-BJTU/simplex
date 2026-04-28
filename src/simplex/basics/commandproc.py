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
        self._stop_event = threading.Event()

    def _capture_output(self):
        while not self._stop_event.is_set() and self.proc:
            try:
                line = self.proc.stdout.readline() # type: ignore
                if not line:
                    break
                self.output.append(line.strip())
            except Exception:
                break

    def __enter__(self):
        self._stop_event.clear()
        startupinfo = None
        creationflags = 0

        self.proc = subprocess.Popen(
            self.cmd,
            shell = self.shell,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            bufsize = 1,
            universal_newlines = True,
            close_fds = False,
            startupinfo = startupinfo,
            creationflags = creationflags
        )

        threading.Thread(target = self._capture_output, daemon = True).start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.proc:
            return False

        self._stop_event.set()

        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout = 3)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait()

        try:
            if self.proc.stdout:
                self.proc.stdout.close()
        except:
            pass

        return False
    
if __name__ == '__main__':
    pass
