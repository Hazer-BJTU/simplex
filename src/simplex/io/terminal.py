import os
import rich

from rich.text import Text
from rich.console import Console
from typing import Optional, Dict, Any, TYPE_CHECKING

import simplex.io.base
import simplex.basics

from simplex.io.base import UserInputInterface, UserOutputInterface
from simplex.basics import UserMessage, UserNotify, PromptTemplate

if TYPE_CHECKING:
    import simplex.context

    from simplex.context import ContextPlugin


class RichTerminalInterface(UserInputInterface, UserOutputInterface):
    def __init__(
        self,
        name: str = 'interface',
        system_prompt: str = 'You are a helpful assistant.',
        style_set: Optional[Dict] = None
    ) -> None:
        super().__init__()

        self.name = name
        self.system_prompt = system_prompt
        self.style_set = {
            'rule_title': 'bold gold1',
            'rule_line': 'dark_orange',
            'symbol': 'bold dark_orange',
            'text': 'grey100',
            'text_weak': 'italic grey70',
            '$': 'bold grey100'
        }

        if style_set is not None:
            self.style_set = self.style_set | style_set

        self.console = Console()

    def _get_style(self, key: str) -> Any:
        return self.style_set.get(key, '')

    async def next_message(self) -> UserMessage:
        self.console.rule(
            Text("What's your requirement?", style = self._get_style('rule_title')), 
            style = self._get_style('rule_line')
        )

        instruction: str = self.console.input(
            (Text(f"[{self.name}]: ", style = self._get_style('symbol')) + 
             Text(f"enter your instruction here", style = self._get_style('text_weak')) +
             Text(f" ❯❯ ", style = self._get_style('$')))
        )

        if instruction.strip() == 'exit':
            return UserMessage(quit = True)
        else:
            return UserMessage(system_prompt = PromptTemplate(self.system_prompt), user_prompt = PromptTemplate(instruction))

    def get_input_plugin(self) -> Optional["ContextPlugin"]:
        return None
    
    def get_output_plugin(self) -> Optional["ContextPlugin"]:
        return None
    
    def push_message(self, *args, **kwargs) -> Any:
        pass

    async def notify_user(self, notify: UserNotify) -> Any:
        pass

if __name__ == '__main__':
    pass
