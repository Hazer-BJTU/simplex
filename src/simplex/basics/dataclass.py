import os
import uuid
import hashlib
import numpy as np

from typing import Dict, Optional, List
from dataclasses import dataclass, field


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
    
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict
    extras: Optional[Dict] = None

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

    
@dataclass
class ModelResponse:
    response: str = ''
    token_cost: int = 0
    reasoning_content: Optional[str] = None
    tool_call: Optional[List[ToolCall]] = None
    embedding: Optional[np.ndarray] = None
    extras: Optional[Dict] = None

@dataclass
class ModelInput:
    model: Optional[str] = None
    messages: Optional[List[Dict]] = None
    tools: Optional[List] = None
    input: Optional[str | List] = None
    extras: Optional[Dict] = None

    @property
    def dict(self) -> Dict:
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
class ToolSchema:
    @dataclass
    class Parameter:
        field: str
        type: str
        description: str
        required: bool = True
        extras: Optional[Dict] = None

        @property
        def dict(self) -> Dict:
            result = {
                'field': self.field,
                'type': self.type,
                'description': self.description,
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
        return { param.field: param.dict for param in self.params }

    @property
    def dict(self) -> Dict:
        result = {
            'name': self.name,
            'description': self.description,
            'parameters': self.param_dict
        }
        if self.extras is not None:
            result['extras'] = self.extras
        return result

if __name__ == '__main__':
    pass
