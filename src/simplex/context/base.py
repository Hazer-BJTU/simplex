import os
import copy
import uuid

from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import simplex.basics.dataclass
import simplex.basics.exception

from simplex.basics.dataclass import ModelInput


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
    def process_prompt(self, model_input: ModelInput) -> None:
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

    def process_prompt(self, model_input: ModelInput) -> None:
        model_input.messages = copy.deepcopy(self.message)

if __name__ == '__main__':
    pass
