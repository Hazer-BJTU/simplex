import json
import uuid
import copy

from typing import List, Dict, Optional

import simplex.basics
import simplex.models.base

from simplex.basics import (
    ModelInput,
    ModelResponse,
    ToolCall,
    ToolReturn,
    RequestError,
    ParameterError
)
from simplex.models.base import ConversationModel, openai_compatiable_translate


class DeepSeekConversationModel(ConversationModel):
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Optional[Dict] = None, 
        default_generate_configs: Optional[Dict] = None, 
        instance_id: Optional[str] = None, 
    ) -> None:
        pass

if __name__ == '__main__':
    pass
