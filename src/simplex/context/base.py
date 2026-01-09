import os

from abc import ABC, abstractmethod

import simplex.basics.dataclass
import simplex.basics.exception

from simplex.basics.dataclass import ModelInput


class ContextPlugin(ABC):
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
