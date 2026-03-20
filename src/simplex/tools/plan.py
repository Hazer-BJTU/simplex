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
    UserResponse,
    AgentLoopStateEdit,
    PromptTemplate
)
from simplex.tools.base import (
    ToolSchema,
    ToolCollection,
    load_schema,
    load_tool_definitions,
    load_tool_skill
)

class SequentialPlan(ToolCollection):
    SCHEMA_FILE: str = 'schema_plan'
    SKILL_FILE: str = 'skill_plan'

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
        
        # Load the skill instructions
        self.skill: str = load_tool_skill(self.SKILL_FILE, {'make_plan': f"`{self.name}`"})
        self.skill_added: bool = False

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
    
    def process_prompt(self, user_prompt: PromptTemplate, **kwargs) -> Optional[AgentLoopStateEdit]:
        if not self.skill_added:
            self.skill_added = True
            new_user_prompt = user_prompt + self.skill
            return AgentLoopStateEdit(user_prompt = new_user_prompt)
    
    async def _tool_sequential_plan(self, content: str, **kwargs) -> str:
        if self.content:
            response: str = f"[original plan]:\n{self.content}\n\n[new plan]:\n{content}"
        else:
            response: str = f"[new plan]:\n{content}"
        self.content = content
        return response

if __name__ == '__main__':
    pass
