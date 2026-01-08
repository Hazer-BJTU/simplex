import os

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import simplex.basics.dataclass
import simplex.basics.exception
import simplex.context.base
import simplex.models.base
import simplex.tools.base

from simplex.basics.dataclass import ToolCall, ToolReturn
from simplex.context.base import ContextPlugin
from simplex.models.base import ConversationModel
from simplex.tools.base import ToolCollection


class AgentLoop(ABC):
    def __init__(
        self,
        agent_model: ConversationModel,
        *args: Any
    ) -> None:
        self.agent_model = agent_model
        self.tools_list: List[ToolCollection] = []
        self.context_list: List[ContextPlugin] = []


if __name__ == '__main__':
    pass
