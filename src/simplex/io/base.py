import os

from abc import ABC, abstractmethod
from typing import Optional, Any

import simplex.basics
import simplex.context

from simplex.basics import UserMessage, UserNotify, UserResponse
from simplex.context import ContextPlugin


class UserInputInterface(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    async def next_message(self) -> UserMessage:
        pass

    @abstractmethod
    async def notify_user(self, notify: UserNotify) -> UserResponse:
        pass

    @abstractmethod
    def get_input_plugin(self) -> Optional[ContextPlugin]:
        pass
    
class UserOutputInterface(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    async def push_message(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_output_plugin(self) -> Optional[ContextPlugin]:
        pass

if __name__ == '__main__':
    pass
