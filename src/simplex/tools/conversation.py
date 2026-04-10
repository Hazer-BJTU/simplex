import os
import uuid
import asyncio

from typing import Optional, List

import simplex.io
import simplex.basics
import simplex.tools.base

from simplex.io import UserInputInterface
from simplex.basics import (
    UserNotify,
    PromptTemplate,
    AgentLoopStateEdit
)
from simplex.tools.base import (
    ToolSchema,
    ToolCollection,
    load_schema,
    load_tool_definitions,
    load_tool_skill
)

class InLoopConversation(ToolCollection):
    SCHEMA_FILE: str = 'schema_propose'
    SKILL_FILE: str = 'skill_propose'

    def __init__(
        self,
        rename = 'propose',
        instance_id: Optional[str] = None,
        add_skill: bool = True
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex, {rename: '_tool_propose'})

        self.name = rename
        self.add_skill = add_skill

        self.schema = load_schema(self.SCHEMA_FILE, 'propose', self.name)
        self.tool_definitions = load_tool_definitions(self.SCHEMA_FILE)

        # Load the skill instructions
        self.skill: str = load_tool_skill(self.SKILL_FILE, {'propose': f"`{self.name}`"})
        self.skill_added: bool = False
        self.input_interface: Optional[UserInputInterface] = None

    async def build(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def reset(self) -> None:
        pass

    def clone(self) -> "InLoopConversation":
        return InLoopConversation(
            self.name,
            uuid.uuid4().hex,
            self.add_skill
        )
    
    async def bind_io(self, input_interface: UserInputInterface, **kwargs) -> None:
        self.input_interface = input_interface
    
    def get_tool_schemas(self) -> List[ToolSchema]:
        return [self.schema]
    
    def tools_descriptions(self) -> str:
        return self.tool_definitions
    
    def process_prompt(self, user_prompt: PromptTemplate, **kwargs) -> Optional[AgentLoopStateEdit]:
        if not self.skill_added and self.add_skill:
            self.skill_added = True
            new_user_prompt = user_prompt + self.skill
            return AgentLoopStateEdit(user_prompt = new_user_prompt)
    
    async def _tool_propose(self, content: str, **kwargs) -> str:
        if self.input_interface:
            user_response = await self.input_interface.notify_user(UserNotify('conversation', content = content, title = 'Proposal'))
            return user_response.content
        
        return f"[ERROR]: User input interface is not online."

if __name__ == '__main__':
    pass
