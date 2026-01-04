import os
import docker
import asyncio

from typing import Optional, List, Dict
from docker.errors import DockerException, ImageNotFound, APIError, NotFound

import simplex.basics.exception
import simplex.basics.dataclass
import simplex.tools.base

from simplex.basics.exception import EnvironmentError
from simplex.basics.dataclass import ToolCall, ToolReturn
from simplex.tools.base import ToolCollection


class PythonInterpreter(ToolCollection):
    def __init__(
        self, 
        rename: str = 'python_interpreter',
        time_limit: int = 10,
        use_container: bool = False,
        container_id: Optional[str] = None,
        default_image: Optional[str] = None,
        auto_pull: bool = True,
        mem_limit: str = '256m',
        cpu_quota: int = 25000
    ) -> None:
        super().__init__({rename: '_tool_python_interpreter'})

        self.name = rename
        self.time_limit = time_limit
        self.use_container = use_container
        self.container_id = container_id
        self.default_image = default_image
        self.auto_pull = auto_pull
        self.mem_limit = mem_limit
        self.cpu_quota = cpu_quota

        self.pyinterpreter_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "execute given python scripts and return program output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "a python script, e.g., 'import math; print(math.sqrt(math.sin(math.pi)))'",
                        }
                    },
                    "required": ["script"],
                },
            },
        }

        self.docker_client = None
        self.container = None

    async def build(self) -> None:
        if not self.use_container:
            return
        
        try:
            event_loop = asyncio.get_running_loop()
            self.docker_client = docker.from_env()

            if self.container_id is not None:
                try:
                    self.container = self.docker_client.containers.get(self.container_id)
                    return
                except NotFound:
                    pass                

            if self.default_image is not None:
                try:
                    self.docker_client.images.get(self.default_image)
                except ImageNotFound:
                    if self.auto_pull:
                        await event_loop.run_in_executor(
                            None,   
                            lambda: self.docker_client.images.pull(self.default_image) #type: ignore
                        )
                    else:
                        raise

                self.container = await event_loop.run_in_executor(
                    None,
                    lambda: self.docker_client.containers.run( #type: ignore
                        image = self.default_image, #type: ignore
                        command = "tail -f /dev/null && /bin/bash",
                        detach = True,
                        tty = True,
                        mem_limit = self.mem_limit,
                        cpu_quota = self.cpu_quota,
                        name = f"{self.name}_container",
                        remove = False
                    )
                )
                self.container_id = self.container.id

            assert self.container is not None, f'unable to initialize container for {self.name} tool'
        except DockerException as e:
            raise EnvironmentError(e)
        except AssertionError as e:
            raise EnvironmentError(e)
        except Exception as e:
            raise
    
    async def release(self) -> None:
        event_loop = asyncio.get_running_loop()
        try:
            if self.container is not None:
                await event_loop.run_in_executor(
                    None,
                    self.container.stop
                )
                await event_loop.run_in_executor(
                    None,
                    self.container.remove
                )
                self.container = None

            if self.docker_client is not None:
                self.docker_client.close()
                self.docker_client = None
        except DockerException as e:
            raise EnvironmentError(e)
        except Exception as e:
            raise
    
    def get_tools(self) -> List[Dict]:
        return [self.pyinterpreter_schema]
    
    def tools_descriptions(self) -> List[Dict]:
        return [
            {
                'tool_name': self.name,
                'description': 'execute given python scripts and return program output',
                'input': 'script: string',
                'output': 'program execution results'
            }
        ]
    
    async def dispatch(self, tool_call: ToolCall) -> ToolReturn:
        return await super().dispatch(tool_call)
    
    async def _execute_locally(self, script: str, **kwargs) -> str:
        process = await asyncio.create_subprocess_exec(
            'python', '-c', script, 
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout = self.time_limit
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f'[ERROR]: execution timeout after {self.time_limit} seconds'
        
        results: str = ''
        if stdout:
            results += f"[STDOUT]: {stdout.decode('utf8', errors='ignore').strip()}\n"
        if stderr:
            results += f"[STDERR]: {stderr.decode('utf8', errors='ignore').strip()}"
        
        return results if results != '' else '[NO OUTPUT]'
    
    async def _execute_container(self, script: str, **kwargs) -> str:
        try:
            event_loop = asyncio.get_running_loop()

            exec_return = await asyncio.wait_for(
                event_loop.run_in_executor(
                    None,
                    lambda: self.container.exec_run( #type: ignore
                        cmd = ['python', '-c', script],
                        stdout = True,
                        stderr = True,
                    )
                ),
                timeout=self.time_limit
            )

            stdout = exec_return.output.decode('utf8', errors='ignore').strip()
            stderr = exec_return.output.decode('utf8', errors='ignore').strip() if exec_return.exit_code != 0 else ''

        except asyncio.TimeoutError:
            return f'[ERROR]: execution timeout after {self.time_limit} seconds'
        except DockerException as e:
            return f'[ERROR]: Container execution failed: {e}'
        except Exception as e:
            return f'[ERROR]: Unexpected error during container execution: {e}'
        
        results: str = ''
        if stdout:
            results += f"[STDOUT]: {stdout}\n"
        if stderr:
            results += f"[STDERR]: {stderr}"
        
        return results if results != '' else '[NO OUTPUT]'
    
    async def _tool_python_interpreter(self, script: str, **kwargs) -> str:
        if self.use_container and self.container is not None:
            return await self._execute_container(script, **kwargs)
        else:
            return await self._execute_locally(script, **kwargs)

if __name__ == '__main__':
    tool = PythonInterpreter(use_container=True, container_id='bfe8ff78433ddcee3554d884f154284d635cf3b0411193727bd5a56c05d208f9')
    
    async def test():
        await tool.build()

        tool_return = await tool.dispatch(ToolCall('task#1', tool.name, {'script': 'import math; print(math.sqrt(math.pi))'}))

        print(tool_return)

        await tool.release()

    asyncio.run(test())
