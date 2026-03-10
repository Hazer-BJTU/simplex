import os
import copy
import uuid

from dataclasses import asdict
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, TYPE_CHECKING

import simplex.basics

from simplex.basics import (
    ModelInput,
    ModelResponse,
    ToolReturn
)

if TYPE_CHECKING:
    import simplex.loop

    from simplex.loop import AgentLoop


class ContextPlugin(ABC):
    def __init__(
        self, 
        instance_id: str
    ) -> None:
        self.instance_id = instance_id

    @property
    def key(self) -> str:
        return self.instance_id

    @abstractmethod
    async def build(self) -> None:
        pass

    @abstractmethod
    async def release(self) -> None:
        pass

    @abstractmethod
    async def reset(self) -> None:
        pass

    @abstractmethod
    def on_start_procedure(self, agent: "AgentLoop") -> None:
        pass

    @abstractmethod
    def on_process_prompt(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        pass

    @abstractmethod
    def on_prompt_ready(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        pass

    @abstractmethod
    def on_model_response(self, model_response: ModelResponse, agent: "AgentLoop") -> None:
        pass

    @abstractmethod
    def on_tool_return(self, tool_return: List[ToolReturn], agent: "AgentLoop") -> None:
        pass

    @abstractmethod
    def on_final_answer(self, model_response: ModelResponse, agent: "AgentLoop") -> None:
        pass

class InitPromptContext(ContextPlugin):
    def __init__(
        self, 
        user_instruction: str,
        instance_id: str = uuid.uuid4().hex,
        system_prompt: str = 'You are a helpful assistant.',
        chat_history: Optional[List[Dict]] = None
    ) -> None:
        super().__init__(instance_id)

        self.message: List[Dict] = []
        self.message.append({'role': 'system', 'content': system_prompt})
        if chat_history is not None:
            self.message.extend(chat_history)
        self.message.append({'role': 'user', 'content': user_instruction})

    async def build(self) -> None:
        return

    async def release(self) -> None:
        return

    async def reset(self) -> None:
        return
    
    def on_start_procedure(self, agent: "AgentLoop") -> None:
        return

    def on_process_prompt(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        model_input.messages = copy.deepcopy(self.message)

    def on_prompt_ready(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        return
    
    def on_model_response(self, model_response: ModelResponse, agent: "AgentLoop") -> None:
        return
    
    def on_tool_return(self, tool_return: List[ToolReturn], agent: "AgentLoop") -> None:
        return
    
    def on_final_answer(self, model_response: ModelResponse, agent: "AgentLoop") -> None:
        return

class TrajectoryLogContext(ContextPlugin):
    def __init__(
        self, 
        instance_id: str = uuid.uuid4().hex,
        empty_on_reset: bool = True,
        line_width: int = 100
    ) -> None:
        super().__init__(instance_id)

        self.empty_on_reset = empty_on_reset
        self.line_width = line_width

        self.log: List[Dict] = []
        self.training_log: List[Dict] = []
        self.markdown: str = ''

    async def build(self) -> None:
        return
    
    async def release(self) -> None:
        return
    
    async def reset(self) -> None:
        if self.empty_on_reset:
            self.log = []
            self.training_log = []
            self.markdown = ''
        return

    def on_start_procedure(self, agent: "AgentLoop") -> None:
        return
    
    def on_process_prompt(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        return
    
    def on_prompt_ready(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        # log details
        self.log.append(asdict(model_input) | {'iter': 'initial_input'})
        
        # log markdown
        if model_input.tools:
            self.markdown += 'Tools available:\n---\n```yaml\n'
            for schema in model_input.tools:
                self.markdown += schema.human_readable_descriptions(self.line_width) + '\n\n'
            self.markdown += '```\n'
        return
    
    def on_model_response(self, model_response: ModelResponse, agent: "AgentLoop") -> None:
        # log details
        info: Dict = {'iter': agent.iter}
        if model_response.reasoning_content:
            info['reasoning_content'] = model_response.reasoning_content
        if model_response.tool_call:
            info['tool_call'] = [asdict(call) for call in model_response.tool_call]
        if model_response.extras:
            info['extras'] = model_response.extras
        if model_response.response:
            info['response'] = model_response.response
        self.log.append(info)

        # log markdown
        if model_response.reasoning_content:
            self.markdown += f"Reason #{agent.iter + 1}:\n---\n{model_response.reasoning_content}\n"
        if model_response.tool_call:
            self.markdown += f"Function Calling #{agent.iter + 1}:\n---\n```yaml\n"
            for tool_call in model_response.tool_call:
                self.markdown += tool_call.human_readable_descriptions(self.line_width) + '\n\n'
            self.markdown += '```\n'
        if model_response.response:
            self.markdown += f"Response #{agent.iter + 1}:\n---\n{model_response.response}\n"
        return
    
    def on_tool_return(self, tool_return: List[ToolReturn], agent: "AgentLoop") -> None:
        # log details
        for ret in tool_return:
            info: Dict = {'iter': agent.iter}
            info['tool_return'] = asdict(ret)
            self.log.append(info)

        # log markdown
        if tool_return:
            self.markdown += f"Tool Return #{agent.iter + 1}:\n---\n"
        for ret in tool_return:
            self.markdown += f"```\n{ret.content}\n```\n"
        return
    
    def on_final_answer(self, model_response: ModelResponse, agent: "AgentLoop") -> None:
        return

    @property
    def detailed(self) -> List[Dict]:
        return self.log
    
    @property
    def human_readable(self) -> str:
        return self.markdown
    
    @property
    def for_training(self) -> List[Dict]:
        return self.training_log

if __name__ == '__main__':
    pass
