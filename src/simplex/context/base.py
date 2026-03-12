import os
import copy
import uuid

from abc import ABC
from dataclasses import asdict
from typing import Dict, List, Any

import simplex.basics

from simplex.basics import (
    ModelInput,
    ModelResponse,
    ToolReturn,
    PromptTemplate
)

class ContextPlugin(ABC):
    def __init__(self, instance_id: str) -> None:
        self.__instance_id = instance_id

    @property
    def key(self) -> str:
        return self.__instance_id
    
    async def build(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def reset(self) -> None:
        pass

    def clone(self) -> "ContextPlugin":
        return copy.deepcopy(self)
    
    def process_prompt(self, *args, **kwargs) -> Any:
        pass

    def start_loop(self, *args, **kwargs) -> Any:
        pass

    async def start_loop_async(self, *args, **kwargs) -> Any:
        pass

    def before_response(self, *args, **kwargs) -> Any:
        pass

    async def before_response_async(self, *args, **kwargs) -> Any:
        pass

    def after_response(self, *args, **kwargs) -> Any:
        pass

    async def after_response_async(self, *args, **kwargs) -> Any:
        pass

    def after_tool_call(self, *args, **kwargs) -> Any:
        pass

    async def after_tool_call_async(self, *args, **kwargs) -> Any:
        pass

    def after_final_response(self, *args, **kwargs) -> Any:
        pass

    async def after_final_response_async(self, *args, **kwargs) -> Any:
        pass

    def on_loop_end(self, *args, **kwargs) -> Any:
        pass

    async def on_loop_end_async(self, *args, **kwargs) -> Any:
        pass

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
    
    async def reset(self) -> None:
        if self.empty_on_reset:
            self.log = []
            self.training_log = []
            self.markdown = PromptTemplate()
        return

    async def start_loop_async(self, model_input: ModelInput, **kwargs) -> Any:
        # log details
        self.log.append(asdict(model_input) | {'iter': 'initial_input'})
        
        # log markdown
        self.markdown.add_main_title('Initial states')
        if model_input.tools:
            self.markdown.add_block([schema.human_readable_descriptions(self.line_width) for schema in model_input.tools], 'Tools available', 'yaml')
        if model_input.messages:
            for message in model_input.messages:
                if 'role' in message and 'content' in message:
                    self.markdown.add_simple(message['content'], message['role'])
        return
    
    async def after_final_response_async(self, iter: int, model_response: ModelResponse, **kwargs) -> Any:
        # log details
        self.log.append(asdict(model_response) | {'iter': iter})

        # log markdown
        self.markdown.add_main_title(f"Agent iteration #{iter}")
        if model_response.reasoning_content:
            self.markdown.add_simple(model_response.reasoning_content, "Reason content")
        if model_response.tool_call:
            self.markdown.add_block([tool_call.human_readable_descriptions(self.line_width) for tool_call in model_response.tool_call], 'Function calling')
        if model_response.response:
            self.markdown.add_simple(model_response.response, "Model Response")
        return
    
    async def after_tool_call_async(self, iter: int, tool_returns: List[ToolReturn], **kwargs) -> Any:
        # log details
        self.log.append({'iter': iter, 'tool_returns': [asdict(ret) for ret in tool_returns]})

        # log markdown
        if tool_returns:
            self.markdown.add_block([ret.content for ret in tool_returns], "Tool returns")
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
