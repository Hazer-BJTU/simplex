import os
import uuid
import subprocess
import asyncio

from typing import Optional, List

import simplex.io
import simplex.basics
import simplex.tools.base

from simplex.io import UserInputInterface
from simplex.basics import (
    UserNotify,
    UserResponse
)
from simplex.tools.base import (
    ToolSchema,
    ToolCollection,
    load_schema,
    load_tool_definitions
)

class SubprocessExecutorLocal(ToolCollection):
    SCHEMA_FILE: str = 'schema_bash'

    def __init__(
        self, 
        rename: str = 'bash',
        permission_required: bool = True,
        instance_id: Optional[str] = None
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex, {rename: '_tool_subprocess_execute'})

        self.name = rename
        self.permission_required = permission_required
        
        self.schema = load_schema(self.SCHEMA_FILE, 'subprocess_executor', self.name)
        self.tool_definitions = load_tool_definitions(self.SCHEMA_FILE)

        self.input_interface: Optional[UserInputInterface] = None

    async def build(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def reset(self) -> None:
        pass

    def clone(self) -> "SubprocessExecutorLocal":
        return SubprocessExecutorLocal(
            self.name,
            self.permission_required,
            uuid.uuid4().hex
        )

    async def bind_io(self, input_interface: UserInputInterface, **kwargs) -> None:
        self.input_interface = input_interface

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [self.schema]
    
    def tools_descriptions(self) -> str:
        return self.tool_definitions
    
    async def _tool_subprocess_execute(self, cwd: str, command: str, **kwargs) -> str:
        if self.input_interface and self.permission_required:
            response = await self.input_interface.notify_user(UserNotify('permission', f"Do you allow agent to execute command: '{command}' under '{cwd}'?"))
            if not response.permitted:
                return f"[ERROR]: Permission error! {response.reason}"
        
        try:
            result = subprocess.run(
                ['bash', '-c', command],
                cwd = cwd,
                capture_output = True,
                text = True,
                check = True
            )

            message = f"[STDOUT]: {result.stdout}"
            if result.stderr:
                message = message + '\n' + f"[STDERR]: {result.stderr}"
            return message
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"[ERROR] bash execution failed (return code: {e.returncode})\n"
                f"[COMMAND]: {command}\n"
                f"[CWD]: {cwd}\n"
                f"[STDOUT]: {e.stdout.strip()}\n"
                f"[STDERR]: {e.stderr.strip()}"
            )
            return error_msg

if __name__ == '__main__':
    pass
