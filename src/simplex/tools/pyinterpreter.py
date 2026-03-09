import os
import uuid
import asyncio

from typing import Optional, List, Dict, Callable, TYPE_CHECKING

import simplex.basics
import simplex.tools.base

from simplex.basics import (
    ModelInput,
    ContainerManager,
    UnbuiltError,
    EntityInitializationError
)
from simplex.tools.base import (
    ToolSchema,
    ToolCollection,
    load_schema,
    load_tool_definitions
)

if TYPE_CHECKING:
    import simplex.loop

    from simplex.loop import AgentLoop


class PythonInterpreter(ToolCollection):
    SCHEMA_FILE: str = 'schema_pyinterpreter'
    PYINTERPRETER: str = 'pyinterpreter'

    def __init__(
        self, 
        instance_id: str = uuid.uuid4().hex,
        rename: str = 'python_interpreter',
        use_container: bool = False,
        container_manager: Optional[ContainerManager] = None,
        exec_command: Callable[[str], List[str]] = lambda script: ['python', '-c', script],
        timeout: float = 10
    ) -> None:
        super().__init__(instance_id, { rename: '_tool_python_interpreter' })

        self.name = rename
        self.use_container = use_container
        self.container_manager = container_manager
        self.exec_command = exec_command
        self.timeout = timeout

        try:
            self.initialized: bool = False

            if self.use_container == True:
                assert self.container_manager is not None, "argument 'container_manager' must not be None when use_container is True"

            self.tool_definitions = load_tool_definitions(self.SCHEMA_FILE)
            self.pyinterpreter_schema = load_schema(self.SCHEMA_FILE, self.PYINTERPRETER, self.name)
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)

    async def build(self) -> None:
        try:
            if self.use_container and self.container_manager is not None:
                await self.container_manager.build()
            self.initialized = True
        except Exception:
            raise
    
    async def release(self) -> None:
        try:
            if self.use_container and self.container_manager is not None:
                await self.container_manager.release()
            self.initialized = False
        except Exception:
            raise
        
    async def reset(self) -> None:
        pass
        
    def get_names(self) -> List[str]:
        return [self.name]
    
    def get_tools(self) -> List[ToolSchema]:
        return [self.pyinterpreter_schema]
    
    def tools_descriptions(self) -> str:
        return self.tool_definitions
    
    def on_init_output(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        pass
    
    async def _execute_locally(self, script: str, **kwargs) -> str:
        process = await asyncio.create_subprocess_exec(
            *self.exec_command(script), 
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout = self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f'[ERROR]: execution timeout after {self.timeout} seconds'
        
        results: str = ''
        if stdout:
            results += f"[STDOUT]: {stdout.decode('utf8', errors = 'ignore').strip()}\n"
        if stderr:
            results += f"[STDERR]: {stderr.decode('utf8', errors = 'ignore').strip()}"
        results = results.strip()
        return results if results != '' else '[NO OUTPUT]'
    
    async def _execute_container(self, script: str, **kwargs) -> str:
        try:
            return await self.container_manager.exec_run( #type: ignore
                self.exec_command(script),
                timeout = self.timeout
            )
        except Exception:
            raise
    
    async def _tool_python_interpreter(self, script: str, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)

        if self.use_container and self.container_manager is not None:
            return await self._execute_container(script, **kwargs)
        else:
            return await self._execute_locally(script, **kwargs)

if __name__ == '__main__':
    pass
