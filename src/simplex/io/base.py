import os

from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

import simplex.basics

from simplex.basics import UserMessage, UserNotify

if TYPE_CHECKING:
    import simplex.context

    from simplex.context import ContextPlugin


class UserInputInterface(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    async def next_message(self) -> UserMessage:
        pass

    @abstractmethod
    async def notify_user(self, notify: UserNotify) -> Any:
        pass

    @abstractmethod
    def get_input_plugin(self) -> Optional["ContextPlugin"]:
        pass
    
class UserOutputInterface(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    async def push_message(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_output_plugin(self) -> Optional["ContextPlugin"]:
        pass

if __name__ == '__main__':
    pass
