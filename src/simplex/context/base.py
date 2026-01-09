import os
import uuid

from abc import ABC, abstractmethod

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

if __name__ == '__main__':
    pass
