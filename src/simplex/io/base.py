import os

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import simplex.basics
import simplex.context

from simplex.basics import UserMessage
from simplex.context import ContextPlugin


class UserInputInterface(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    async def next_message(self) -> UserMessage:
        pass

    @abstractmethod
    def get_context_plugin(self) -> Optional[ContextPlugin]:
        pass
    
class UserOutputInterface(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_context_plugin(self) -> Optional[ContextPlugin]:
        pass        

if __name__ == '__main__':
    pass
