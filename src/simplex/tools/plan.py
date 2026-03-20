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

class SequentialPlan(ToolCollection):
    SCHEMA_FILE: str = 'schema_plan'

    def __init__(
        self, 
        rename: str = 'make_plan',
        empty_on_reset: bool = True,
        instance_id: Optional[str] = None
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex, {rename: '_tool_sequential_plan'})

        self.name = rename
        self.empty_on_reset = empty_on_reset
        
        self.schema = load_schema(self.SCHEMA_FILE, 'make_plan', self.name)
        self.tool_definitions = load_tool_definitions(self.SCHEMA_FILE)

        self.content: str = ''

    async def build(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def reset(self) -> None:
        if self.empty_on_reset:
            self.content = ''

    def clone(self) -> "SequentialPlan":
        return SequentialPlan(
            self.name,
            self.empty_on_reset
        )

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [self.schema]
    
    def tools_descriptions(self) -> str:
        return self.tool_definitions
    
    async def _tool_sequential_plan(self, content: str, **kwargs) -> str:
        if self.content:
            response: str = f"[original plan]:\n{self.content}\n\n[new plan]:\n{content}"
        else:
            response: str = f"[new plan]:\n{content}"
        self.content = content
        return response

if __name__ == '__main__':
    pass
