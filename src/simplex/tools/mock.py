import os
import uuid
import asyncio

from typing import Optional, List, Dict, Callable, TYPE_CHECKING

import simplex.basics
import simplex.tools.base

from simplex.basics import (
    ModelInput,
    ContainerManager,
    UnbuiltError,
    EntityInitializationError
)
from simplex.tools.base import ToolCollection

if TYPE_CHECKING:
    import simplex.loop

    from simplex.loop import AgentLoop


class MockCalculator(ToolCollection):
    def __init__(
        self, 
        instance_id: str, 
        rename: str = 'calculator'
    ) -> None:
        super().__init__(instance_id, { rename: '_tool_calculator' })

        self.name = rename

        self.schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Execute given python scripts and return program results. " \
                                "Remember to use 'print' function to output to stdout!",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "a python script, e.g., 'import math; print(math.sqrt(math.sin(math.pi)))'",
                        }
                    },
                    "required": ["script"],
                },
            }
        }

    async def build(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def reset(self) -> None:
        pass

    def get_names(self) -> List[str]:
        return [ self.name ]
    
    def get_tools(self) -> List[Dict]:
        return [  ]