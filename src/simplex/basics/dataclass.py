import os
import copy
import uuid
import hashlib
import textwrap
import numpy as np

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Literal, Any

import simplex.basics.prompt

from simplex.basics.prompt import PromptTemplate


@dataclass
class DocumentEntry:
    content: bytes
    digest: bytes = field(init = False)
    key: str = field(default_factory = lambda: uuid.uuid4().hex)
    file_identifier: str = 'unknown'
    extras: Optional[Dict] = None

    def __post_init__(self) -> None:
        self.digest = hashlib.blake2b(self.content, digest_size=32).digest()
    
    @property
    def str_content(self) -> str:
        try:
            return self.content.decode('utf8')
        except UnicodeDecodeError:
            raise

    @property
    def size(self) -> int:
        return len(self.content)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict
    extras: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)
    
    def human_readable_descriptions(self, text_width = 30) -> str:
        output: str = f"name: {self.name}\narguments:\n"
        for key, value in self.arguments.items():
            if len(str(value)) <= text_width:
                output += f"{key}: {value}\n"
            else:
                output += f"{key}:\n{value}\n"
        return output.strip()

@dataclass
class ToolReturn:
    content: str
    original_call: ToolCall
    extras: Optional[Dict] = None

    @property
    def id(self) -> str:
        return self.original_call.id
    
    @property
    def name(self) -> str:
        return self.original_call.name
    
    @property
    def arguments(self) -> Dict:
        return self.original_call.arguments
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
@dataclass
class ModelResponse:
    response: str = ''
    token_cost: int = 0
    reasoning_content: Optional[str] = None
    tool_call: Optional[List[ToolCall]] = None
    embedding: Optional[np.ndarray] = None
    extras: Optional[Dict] = None

    def to_dict(self) -> Dict:
        copied = copy.deepcopy(self)
        if copied.embedding:
            copied.embedding = None
        if copied.extras and 'original_call' in copied.extras:
            del copied.extras['original_call']
        return asdict(copied)
    
@dataclass
class ToolSchema:
    @dataclass
    class Parameter:
        field: str
        type: str
        description: str
        required: bool = True
        enum: Optional[List] = None
        extras: Optional[Dict] = None
        
        def to_dict(self) -> Dict:
            result = {
                'field': self.field,
                'type': self.type,
                'description': self.description,
                'enum': self.enum,
                'required': self.required
            }
            if self.extras is not None:
                result['extras'] = self.extras
            return result

    name: str
    description: str
    params: List[Parameter] = field(default_factory = list)
    extras: Optional[Dict] = None

    @property
    def param_dict(self) -> Dict:
        return { param.field: param.to_dict() for param in self.params }
    
    def human_readable_descriptions(self, text_width = 30) -> str:
        output: str = f"{self.name}:\n  description:\n"
        output += textwrap.fill(
            self.description, 
            width = text_width,
            initial_indent = '    ',
            subsequent_indent = '    ',
            replace_whitespace = False,
            drop_whitespace = True
        )
        output += '\n  parameters:\n'
        for param in self.params:
            output += f"    {param.field}:\n"
            output += f"      type: {param.type}\n"
            output += f"      description:\n"
            output += textwrap.fill(
                param.description,
                width = text_width,
                initial_indent = '        ',
                subsequent_indent = '        ',
                replace_whitespace = False,
                drop_whitespace = True
            ) + '\n'
            if param.enum:
                output += f"      enum: {str(param.enum)}\n"
            output += f"      required: {param.required}\n"
        return output.strip()
    
    def to_dict(self) -> Dict:
        result = {
            'name': self.name,
            'description': self.description,
            'parameters': self.param_dict
        }
        if self.extras is not None:
            result['extras'] = self.extras
        return result
    
@dataclass
class ModelInput:
    model: Optional[str] = None
    messages: Optional[List[Dict]] = None
    tools: Optional[List[ToolSchema]] = None
    input: Optional[str | List] = None
    extras: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        output_dict: Dict = {}
        if self.model is not None:
            output_dict['model'] = self.model
        if self.messages is not None:
            output_dict['messages'] = self.messages
        if self.tools is not None:
            output_dict['tools'] = self.tools
        if self.input is not None:
            output_dict['input'] = self.input
        if self.extras is not None:
            output_dict |= self.extras
        return output_dict
    
@dataclass
class LoopInformation:
    model_input: Optional[ModelInput] = None
    model_response: Optional[ModelResponse] = None
    tool_returns: List[ToolReturn] = field(default_factory = list)
    extras: Optional[Dict] = None

    def to_dict(self) -> Dict:
        result: Dict = {}
        if self.model_input:
            result['model_input'] = asdict(self.model_input)
        if self.model_response:
            result['model_response'] = asdict(self.model_response)
        if self.tool_returns:
            result['tool_returns'] = [asdict(ret) for ret in self.tool_returns]
        if self.extras:
            result['extras'] = self.extras
        return result
    
@dataclass
class AgentLoopStateEdit:
    """
    Dataclass representing the mutable state of AgentLoop that can be modified by lifecycle hooks
    
    This class encapsulates all loop variables that plugins/tools are allowed to modify
    during synchronous lifecycle hooks. Asynchronous hooks cannot modify the state directly.
    
    Attributes:
        system_prompt: System prompt template for the conversation
        user_prompt: User prompt template for the conversation
        model_input: Formatted input for the language model
        model_response: Raw response from the language model
        tool_returns: Results from executed tool calls
        exit_flag: Boolean flag to terminate loop early
    """

    system_prompt: Optional[PromptTemplate] = None
    user_prompt: Optional[PromptTemplate] = None
    model_input: Optional[ModelInput] = None
    model_response: Optional[ModelResponse] = None
    tool_returns: Optional[List[ToolReturn]] = None
    exit_flag: Optional[bool] = None

@dataclass
class UserMessage:
    system_prompt: PromptTemplate = field(default_factory = PromptTemplate)
    user_prompt: PromptTemplate = field(default_factory = PromptTemplate)
    quit: bool = False

@dataclass
class UserNotify:
    notify_type: Literal['unknown', 'permission', 'notify'] = 'permission'
    content: str = ''
    title: str = ''
    objects: Optional[List[Dict[str, Any]]] = None

@dataclass
class UserResponse:
    permitted: bool
    reason: str = ''
    # more in the future

if __name__ == '__main__':
    pass
