import os
import sys
import uuid
import rich
import asyncio

from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.spinner import Spinner
from rich.console import Console, Group
from rich.markdown import Markdown
from typing import Optional, Dict, Any, List

import simplex.io.base
import simplex.basics
import simplex.context

from simplex.basics import (
    UserMessage, 
    UserNotify, 
    PromptTemplate, 
    ModelResponse, 
    LogExceptionHandler,
    ToolReturn,
    UserResponse,
    SkillRetriever
)
from simplex.io.base import UserInputInterface, UserOutputInterface
from simplex.context import ContextPlugin


class RichTerminalExceptionHandler(LogExceptionHandler):
    def __init__(self, console: Console, style_set: Dict, instance_id: Optional[str] = None) -> None:
        super().__init__(instance_id, file = None)

        self.console = console
        self.style_set = style_set

    def _get_style(self, key: str) -> Any:
        return self.style_set.get(key, '')
    
    def handle_exception(self, exception: Exception) -> None:
        super().handle_exception(exception)
        text = Text(f"{type(exception).__name__}: ", style = self._get_style('text_explicit'))
        text.append(str(exception), style = self._get_style('text'))
        panel = Panel(
            text,
            border_style = self._get_style('box_line_explicit'),
            title = Text('Error', style = self._get_style('box_title_explicit')),
            title_align = 'left'
        )
        self.console.print(panel)

class RichTerminalOutputPlugin(ContextPlugin):
    def __init__(
        self, 
        console: Console, 
        style_set: Dict, 
        max_string: int = 80, 
        instance_id: Optional[str] = None
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex)

        self.console = console
        self.style_set = style_set
        self.max_string = max_string

        self.live_display: Optional[Live] = None

    def _get_style(self, key: str) -> Any:
        return self.style_set.get(key, '')

    async def before_response_async(self, *args, **kwargs) -> None:
        spinner = Spinner('dots', Text('LLM backend is inferencing...', style = self._get_style('text')))
        panel = Panel(
            spinner,
            border_style = self._get_style('border_line'),
            title = Text('Processing', style = self._get_style('box_title')),
            title_align = 'left'
        )
        self.live_display = Live(
            panel,
            console = self.console,
            refresh_per_second = 12,
            auto_refresh = True,
            transient = True
        )
        self.live_display.start()

    async def after_response_async(self, model_response: ModelResponse, *args, **kwargs) -> None:
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

        if model_response.reasoning_content or model_response.response:
            self.console.rule(
                Text("Response", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )
            if model_response.reasoning_content:
                panel = Panel(
                    Text(model_response.reasoning_content, style = self._get_style('text')),
                    border_style = self._get_style('box_line_explicit'),
                    title = Text('Reasoning', style = self._get_style('box_title_explicit')),
                    title_align = 'left'
                )
                self.console.print(panel)
            if model_response.response:
                panel = Panel(
                    Markdown(model_response.response),
                    border_style = self._get_style('box_line_explicit'),
                    title = Text('Answer', style = self._get_style('box_title_explicit')),
                    title_align = 'left'
                )
                self.console.print(panel)
        
        if model_response.tool_call:
            for call in model_response.tool_call:
                text = Text(f"⚒️  Using tool ", style = self._get_style('text'))
                text.append(f"{call.name}", style = self._get_style('text_explicit'))
                group = Group(text, Pretty(call.arguments, max_string = self.max_string))
                panel = Panel(
                    group,
                    border_style = self._get_style('box_line_weak'),
                    title = Text('Tool Call', style = self._get_style('box_title_weak')),
                    title_align = 'left'
                )
                self.console.print(panel)

    async def after_tool_call_async(self, tool_returns: List[ToolReturn], **kwargs) -> None:
        for ret in tool_returns:
            text = Text(f"⚙️  Tool ", style = self._get_style('text'))
            text.append(f"{ret.original_call.name} ", style = self._get_style('text_explicit'))
            text.append("returns: ", style = self._get_style('text'))
            text.append(f"{ret.content}", style = self._get_style('text'))
            panel = Panel(
                text,
                border_style = self._get_style('box_line_weak'),
                title = Text('Tool Return', style = self._get_style('box_title_weak')),
                title_align = 'left'
            )
            self.console.print(panel)

class RichTerminalInterface(UserInputInterface, UserOutputInterface):
    def __init__(
        self,
        name: str = 'interface',
        system_prompt: str = 'You are a helpful assistant.',
        retriever: Optional[SkillRetriever] = None,
        style_set: Optional[Dict] = None,
        max_string: int = 80
    ) -> None:
        super().__init__()

        self.name = name
        self.system_prompt = system_prompt
        self.retriever = retriever if retriever is not None else SkillRetriever()
        self.style_set = {
            'rule_title': 'not italic bold gold1',
            'rule_line': 'dark_orange',
            'name': 'not italic bold dark_orange',
            'user': 'not italic bold violet',
            'text': 'not italic not bold grey100',
            'text_weak': 'not italic not bold grey70',
            'text_explicit': 'not italic bold yellow1',
            '$': 'not italic bold grey100',
            'box_title': 'not italic bold gold1',
            'border_line': 'dark_orange',
            'box_title_weak': 'not italic bold violet',
            'box_line_weak': 'orchid',
            'box_title_explicit': 'not italic bold orange_red1',
            'box_line_explicit': 'gold1'
        }
        self.max_string = max_string

        if style_set is not None:
            self.style_set = self.style_set | style_set

        self.console = Console()
        self.skills_loaded: bool = False

    def _get_style(self, key: str) -> Any:
        return self.style_set.get(key, '')
    
    def _load_skills(self, instruction: str) -> Optional[PromptTemplate]:
        if self.skills_loaded:
            return None
        
        self.skills_loaded = True

        if self.retriever:
            retrieved = self.retriever.search(instruction)

            if not retrieved:
                return None
            
            self.console.rule(
                Text("Skills Useful", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )

            for idx, skill in enumerate(retrieved):
                panel = Panel(
                    Text(skill['description'], style = self._get_style('text')), 
                    border_style = self._get_style('box_line_weak'),
                    title = Text(f"{idx + 1}. {skill['title']}", style = self._get_style('box_title_weak')),
                    title_align = 'left'
                )
                self.console.print(panel)

            self.console.rule(
                Text("Loading Skills", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )

            text = Text("Load them? [use indices like ", style = self._get_style('text_weak'))
            text.append("1, 2, 3, ...", style = self._get_style('text_explicit'))
            text.append("] ", style = self._get_style('text_weak'))
            text.append("❯❯ ", style = self._get_style('$'))

            selected = self.console.input(text)
            selected = [max(int(part.strip()) - 1, 0) for part in selected.split(',') if part.strip().isdigit()]
            prompt = PromptTemplate().add_main_title('Skills that may be useful')
            for idx in selected:
                try:
                    prompt.add_simple(retrieved[idx]['content'])
                except Exception:
                    pass

            return prompt


    async def next_message(self) -> UserMessage:
        self.console.rule(
            Text("Agent Online", style = self._get_style('rule_title')), 
            style = self._get_style('rule_line')
        )

        text = Text(f"[{self.name}]: ", style = self._get_style('name'))
        text.append(f"your instruction", style = self._get_style('text_weak'))
        text.append(f" ❯❯ ", style = self._get_style('$'))
        instruction: str = self.console.input(text)

        if instruction.strip() == 'exit':
            return UserMessage(quit = True)
        else:
            if self.retriever:
                system_prompt = self.retriever.get_system_prompt()
            else:
                system_prompt = PromptTemplate(self.system_prompt)
            skills_prompt = self._load_skills(instruction)
            if skills_prompt:
                user_prompt = PromptTemplate(instruction) + skills_prompt
            else:
                user_prompt = PromptTemplate(instruction)
            return UserMessage(system_prompt = system_prompt, user_prompt = user_prompt)

    def get_input_plugin(self) -> Optional[ContextPlugin]:
        return None
    
    def get_output_plugin(self) -> Optional[ContextPlugin]:
        return RichTerminalOutputPlugin(self.console, self.style_set, self.max_string)
    
    def push_message(self, *args, **kwargs) -> Any:
        pass

    async def notify_user(self, notify: UserNotify) -> UserResponse:
        if notify.notify_type == 'permission':
            if notify.content:
                panel = Panel(
                    Text(notify.content, style = self._get_style('text')),
                    border_style = self._get_style('box_line_explicit'),
                    title = Text('Request', style=self._get_style('box_title_explicit')),
                    title_align = 'left'
                )
                self.console.print(panel)
            
            prompt_text = Text("Allow this operation? [", style = self._get_style('text_weak'))
            prompt_text.append("yes", style = self._get_style('text_explicit'))
            prompt_text.append("/", style = self._get_style('text_weak'))
            prompt_text.append("no", style = self._get_style('text_explicit'))
            prompt_text.append(" (or ", style = self._get_style('text_weak'))
            prompt_text.append("no: reason", style = self._get_style('text_explicit'))
            prompt_text.append(")] ", style = self._get_style('text_weak'))
            prompt_text.append("❯❯ ", style = self._get_style('$'))
            
            self.console.rule(
                Text("Permission Required", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )

            while True:
                response = self.console.input(prompt_text)
                response = response.strip().lower()
                
                if not response:
                    return UserResponse(permitted = False, reason = 'User denied the request without any explanations.')
                
                if response == 'yes':
                    return UserResponse(permitted = True)
                
                if response.startswith('no'):
                    if ':' in response:
                        _, reason = response.split(':', 1)
                        reason = reason.strip()
                        return UserResponse(permitted = False, reason = reason if reason else '')
                    else:
                        return UserResponse(permitted = False)
                
                text = Text("Invalid input. Please enter '", style = self._get_style('text_weak'))
                text.append("yes", style = self._get_style('text_explicit'))
                text.append("', '", style = self._get_style('text_weak'))
                text.append("no", style = self._get_style('text_explicit'))
                text.append("' or '", style = self._get_style('text_weak'))
                text.append("no: reason", style = self._get_style('text_explicit'))
                text.append("'.", style = self._get_style('text_weak'))
                self.console.print(text)
        
        return UserResponse(permitted = True)

    def get_exception_handler(self) -> RichTerminalExceptionHandler:
        return RichTerminalExceptionHandler(self.console, self.style_set)

if __name__ == '__main__':
    pass
