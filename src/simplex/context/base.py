import os

from abc import ABC, abstractmethod

import simplex.basics.dataclass
import simplex.basics.exception


class ContextPlugin(ABC):
    @abstractmethod
    async def build(self) -> None:
        pass

    @abstractmethod
    async def release(self) -> None:
        pass

if __name__ == '__main__':
    pass
