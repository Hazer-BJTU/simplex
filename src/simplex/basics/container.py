import os
import uuid
import docker
import pexpect
import asyncio

from typing import Optional, List
from docker.errors import ImageNotFound, NotFound, DockerException

import simplex.basics.exception

from simplex.basics.exception import EnvironmentError, UnbuiltError


class ContainerManager:
    def __init__(
        self,
        name: str,
        container_id: Optional[str] = None,
        default_image: Optional[str] = None,
        auto_pull: bool = True,
        clean_on_finish: bool = True,
        mem_limit: str = '256m',
        cpu_ratio: float = 0.25,
        timeout: float = 10
    ) -> None:
        self.name = name
        self.container_id = container_id
        self.default_image = default_image
        self.auto_pull = auto_pull
        self.clean_on_finish = clean_on_finish
        self.mem_limit = mem_limit
        self.timeout = timeout
        
        self.cpu_period: int = 100000
        self.cpu_quota: int = int(self.cpu_period * cpu_ratio)
        self.initialized: bool = False

        self.command_end_flag: str = '<<<EXECUTION_COMPLETE>>>'
        self.output_buffer: List[str] = []

        self.client = None
        self.container = None
        self.terminal = None
        self.platform = os.name

    async def _read_buffer(self, pattern: str, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            timeout = self.timeout
        
        try:
            assert self.terminal is not None, 'failed to spawn terminal interaction'
            index = await self.terminal.expect([pattern, pexpect.EOF, pexpect.TIMEOUT], timeout = timeout, async_ = True) #type: ignore
            if index == 0:
                expected: str = ''
                if isinstance(self.terminal.before, bytes):
                    expected += self.terminal.before.decode('utf8', errors = 'ignore')
                if isinstance(self.terminal.after, bytes):
                    expected += self.terminal.after.decode('utf8', errors = 'ignore')
                expected = expected.strip()
                if expected:
                    self.output_buffer.append(expected)
                return True
            else:
                expected: str = ''
                if isinstance(self.terminal.before, bytes):
                    expected += self.terminal.before.decode('utf8', errors = 'ignore')
                expected = expected.strip()
                if expected:
                    self.output_buffer.append(expected)
                return False
        except Exception:
            raise

    async def build(self) -> None:
        if self.initialized or self.container is not None:
            return
        
        try:
            event_loop = asyncio.get_running_loop()
            self.client = docker.from_env()

            if self.container_id is not None:
                try:
                    self.container = self.client.containers.get(self.container_id)
                    if self.container is not None:
                        spawn_command: str = f"docker exec -it {self.container_id} /bin/bash"
                        self.terminal = pexpect.spawn(spawn_command)
                        successful = await self._read_buffer(r'[#$]')
                        assert successful, 'failed to spawn terminal interaction'
                        self.initialized = True
                        return
                except NotFound:
                    self.container = None
                except AssertionError:
                    raise

            if self.default_image is not None:
                try:
                    self.client.images.get(self.default_image)
                except ImageNotFound:
                    if not self.auto_pull:
                        raise
                    try:
                        await event_loop.run_in_executor(None, lambda: self.client.images.pull(self.default_image)) #type: ignore
                    except Exception:
                        raise
            
            self.container = await event_loop.run_in_executor(
                None, lambda: self.client.containers.run( #type: ignore
                    image = self.default_image, #type: ignore
                    command = "tail -f /dev/null",
                    detach = True,
                    tty = True,
                    mem_limit = self.mem_limit,
                    cpu_quota = self.cpu_quota,
                    name = f"{self.name}_{uuid.uuid4().hex[:7]}",
                    remove = False
                )
            )

            self.container_id = self.container.id
            assert self.container is not None, f'unable to initialize container'
            spawn_command: str = f"docker exec -it {self.container_id} /bin/bash"
            self.terminal = pexpect.spawn(spawn_command)
            successful = await self._read_buffer(r'[#$]')
            assert successful, 'failed to spawn terminal interaction'
            self.initialized = True
        except Exception as e:
            raise EnvironmentError(e)

    async def release(self) -> None:
        try:
            event_loop = asyncio.get_running_loop()

            if self.container is not None and self.clean_on_finish:
                await event_loop.run_in_executor(None, self.container.stop)
                await event_loop.run_in_executor(None, self.container.remove)
                self.container = None

            if self.client is not None:
                self.client.close()
                self.client = None

            if self.terminal is not None:
                self.terminal.close()
                self.terminal = None

            self.initialized = False
        except Exception as e:
            raise EnvironmentError(e)
        
    async def exec_bash_command(
        self, 
        command: str, 
        timeout: Optional[float] = None, 
        command_end_flag: Optional[str] = None
    ) -> None:
        if not self.initialized or self.container is None:
            raise UnbuiltError(self.__class__.__name__)

        if command_end_flag is None:
            command_end_flag = self.command_end_flag
        
        try:
            assert self.terminal is not None, 'failed to spawn terminal interaction'
            self.terminal.sendline(command)
            self.terminal.sendline(f": '{command_end_flag}'")
            await self._read_buffer(f": '{command_end_flag}'", timeout = timeout)
            await self._read_buffer(r'[#$]', timeout = timeout)
        except Exception:
            raise

    async def exec_run(
        self,
        command: str | List[str],
        timeout: Optional[float] = None
    ) -> str:
        if not self.initialized or self.container is None:
            raise UnbuiltError(self.__class__.__name__)

        if timeout is None:
            timeout = self.timeout

        try:
            event_loop = asyncio.get_running_loop()

            exec_return = await asyncio.wait_for(
                event_loop.run_in_executor(
                    None, lambda: self.container.exec_run( #type: ignore
                        cmd = command,
                        stdout = True,
                        stderr = True
                    )
                ),
                timeout = timeout
            )

            stdout = exec_return.output.decode('utf8', errors = 'ignore').strip()
            if exec_return.exit_code != 0:
                stderr = exec_return.output.output('utf8', errors = 'ignore').strip()
            else:
                stderr = ''

            results: str = ''
            if stdout:
                results += f"[STDOUT]: {stdout}\n"
            if stderr:
                results += f"[STDERR]: {stderr}"
            results = results.strip()
            return results if results != '' else '[NO OUTPUT]'
        except asyncio.TimeoutError:
            return f'[ERROR]: execution timeout after {timeout} seconds'
        except DockerException as e:
            return f'[ERROR]: container execution failed: {e}'
        except Exception as e:
            return f'[ERROR]: unexpected error during container execution: {e}'

if __name__ == '__main__':
    container = ContainerManager(
        'test',
        default_image = 'python:3.11-slim',
        cpu_ratio = 1.0
    )

    async def test() -> None:
        await container.build()
        print(container.output_buffer)
        await container.exec_bash_command("python -c 'import math; print(math.sqrt(math.pi))'")
        print(container.output_buffer)
        results = await container.exec_run(['python', '-c', 'import math; print(math.sqrt(math.pi))'])
        print(results)
        await container.release()

    asyncio.run(test())
