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
    ToolReturn,
    PromptTemplate
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
        line_width: int = 150
    ) -> None:
        super().__init__(instance_id)

        self.empty_on_reset = empty_on_reset
        self.line_width = line_width

        self.log: List[Dict] = []
        self.training_log: List[Dict] = []
        self.markdown: PromptTemplate = PromptTemplate()

    async def build(self) -> None:
        return
    
    async def release(self) -> None:
        return
    
    async def reset(self) -> None:
        if self.empty_on_reset:
            self.log = []
            self.training_log = []
            self.markdown = PromptTemplate()
        return

    def on_start_procedure(self, agent: "AgentLoop") -> None:
        return
    
    def on_process_prompt(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        return
    
    def on_prompt_ready(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        # log details
        self.log.append(model_input.to_dict() | {'iter': 'initial_input'})
        
        # log markdown
        self.markdown.add_main_title('Initial states')
        if model_input.tools:
            self.markdown.add_block([schema.human_readable_descriptions(self.line_width) for schema in model_input.tools], 'Tools available', 'yaml')
        if model_input.messages:
            for message in model_input.messages:
                if 'role' in message and 'content' in message:
                    self.markdown.add_simple(message['content'], message['role'])
        return
    
    def on_model_response(self, model_response: ModelResponse, agent: "AgentLoop") -> None:
        # log details
        self.log.append(model_response.to_dict() | {'iter': agent.iter})

        # log markdown
        self.markdown.add_main_title(f"Agent iteration #{agent.iter}")
        if model_response.reasoning_content:
            self.markdown.add_simple(model_response.reasoning_content, "Reason content")
        if model_response.tool_call:
            self.markdown.add_block([tool_call.human_readable_descriptions(self.line_width) for tool_call in model_response.tool_call], 'Function calling')
        if model_response.response:
            self.markdown.add_simple(model_response.response, "Model Response")
        return
    
    def on_tool_return(self, tool_return: List[ToolReturn], agent: "AgentLoop") -> None:
        # log details
        self.log.append({'iter': agent.iter, 'tool_returns': [ret.to_dict() for ret in tool_return]})

        # log markdown
        if tool_return:
            self.markdown.add_block([ret.content for ret in tool_return], "Tool returns")
        return
    
    def on_final_answer(self, model_response: ModelResponse, agent: "AgentLoop") -> None:
        return

    @property
    def detailed(self) -> List[Dict]:
        return self.log
    
    @property
    def human_readable(self) -> str:
        return str(self.markdown)
    
    @property
    def for_training(self) -> List[Dict]:
        return self.training_log

if __name__ == '__main__':
    pass
