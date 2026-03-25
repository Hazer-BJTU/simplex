import os
import uuid
import copy

from typing import Literal, Optional, Dict

import simplex.basics
import simplex.context.base

from simplex.basics import (
    ModelInput, 
    ToolSchema,
    AgentLoopStateEdit,
    PromptTemplate
)
from simplex.context.base import ContextPlugin, load_tool_skill


TASK_STATUES: Dict[Literal['derailed', 'blocked', 'steady', 'solid', 'perfect'], int] = {
    'derailed': 0,
    'blocked': 1,
    'steady': 2,
    'solid': 3,
    'perfect': 4
}

ACTION_EXPECTS: Dict[Literal['unknown', 'workaround', 'effective', 'precise', 'optimal'], int] = {
    'unknown': 0,
    'workaround': 1,
    'effective': 2,
    'precise': 3,
    'optimal': 4
}

class ActionSelfEvaluation(ContextPlugin):
    SKILL_FILE: str = 'skill_self_eval'

    def __init__(
        self, 
        instance_id: Optional[str] = None,
        add_skill: bool = True
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex)

        self.add_skill = add_skill

        self.skill: str = load_tool_skill(self.SKILL_FILE)
        self.skill_added: bool = False

    def process_prompt(self, user_prompt: PromptTemplate, **kwargs) -> Optional[AgentLoopStateEdit]:
        if not self.skill_added and self.add_skill:
            self.skill_added = True
            new_user_prompt = user_prompt + self.skill
            return AgentLoopStateEdit(user_prompt = new_user_prompt)

    def start_loop(self, model_input: ModelInput, **kwargs) -> AgentLoopStateEdit:
        new_model_input = copy.deepcopy(model_input)
        if new_model_input.tools:
            for schema in new_model_input.tools:
                schema.params.append(ToolSchema.Parameter(
                    field = 'task_status',
                    type = 'string',
                    description = 'An objective assessment of the steady-state health of the current task execution pipeline before the tool call you are about to make.',
                    required = True,
                    enum = ['derailed', 'blocked', 'steady', 'solid', 'perfect']
                ))
                schema.params.append(ToolSchema.Parameter(
                    field = 'action_quality',
                    type = 'string',
                    description = 'How valuable is the next action? Estimate the expected impact of the tool call you are about to make.',
                    required = True,
                    enum = ['unknown', 'workaround', 'effective', 'precise', 'optimal']
                ))
        
        return AgentLoopStateEdit(model_input = new_model_input)

if __name__ == '__main__':
    pass
