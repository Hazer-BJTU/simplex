import os
import sys
import uuid
import rich
import asyncio

from rich import box
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.spinner import Spinner
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.table import Table
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
    """
    An exception handler that displays errors in a rich terminal format.
    
    This class extends LogExceptionHandler to provide visual error display in the terminal
    using rich formatting. It catches exceptions that occur during agent execution and
    displays them in styled panels with clear error messages.
    
    The handler maintains the logging functionality of the parent class while adding
    visual presentation features appropriate for terminal interfaces.
    
    Attributes:
        console (Console): Rich console instance for output rendering
        style_set (Dict): Dictionary of styling configurations for error display
    """
    def __init__(self, console: Console, style_set: Dict, instance_id: Optional[str] = None) -> None:
        """
        Initialize the RichTerminalExceptionHandler with console and styling configuration.
        
        Args:
            console: Rich Console instance to use for error output rendering
            style_set: Dictionary containing style definitions for error display
            instance_id: Optional unique identifier for this exception handler instance
        """
        super().__init__(instance_id, file = None)

        self.console = console
        self.style_set = style_set

    def _get_style(self, key: str) -> Any:
        """
        Retrieve a style definition by key from the style set.
        
        Args:
            key: The style key to look up
            
        Returns:
            The style definition string, or empty string if key not found
        """
        return self.style_set.get(key, '')
    
    def handle_exception(self, exception: Exception) -> None:
        """
        Handle and display an exception in the terminal with rich formatting.
        
        This method logs the exception using the parent class functionality and then
        displays it in a visually distinct panel in the terminal with appropriate styling.
        
        Args:
            exception: The exception to handle and display
        """
        super().handle_exception(exception)
        # Create styled text showing the exception type and message
        text = Text(f"{type(exception).__name__}: ", style = self._get_style('text_explicit'))
        text.append(str(exception), style = self._get_style('text'))
        # Display the exception in a styled panel
        panel = Panel(
            text,
            border_style = self._get_style('box_line_explicit'),
            title = Text('Error', style = self._get_style('box_title_explicit')),
            title_align = 'left'
        )
        self.console.print(panel)

class RichTerminalOutputPlugin(ContextPlugin):
    """
    A context plugin that provides rich terminal output formatting for agent responses.
    
    This plugin integrates with the agent loop to display formatted output including:
    - Loading spinners during LLM processing
    - Structured display of agent responses (reasoning and answers)
    - Tool call visualization with arguments
    - Tool return values with clear labeling
    
    The plugin uses the rich library to create visually appealing terminal output
    with panels, rules, and colored text based on the provided style configuration.
    
    Attributes:
        console (Console): Rich console instance for output rendering
        style_set (Dict): Dictionary of styling configurations for different UI elements
        max_string (int): Maximum length for displaying strings before truncation
        live_display (Optional[Live]): Active Live display instance for spinner animation
    """
    def __init__(
        self, 
        console: Console, 
        style_set: Dict, 
        max_string: int = 80, 
        max_text: int = 2000,
        instance_id: Optional[str] = None
    ) -> None:
        """
        Initialize the RichTerminalOutputPlugin with console and styling configuration.
        
        Args:
            console: Rich Console instance to use for output rendering
            style_set: Dictionary containing style definitions for various UI elements
            max_string: Maximum character length before truncating displayed strings
            instance_id: Optional unique identifier for this plugin instance
        """
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex)

        self.console = console
        self.style_set = style_set
        self.max_string = max_string
        self.max_text = max_text

        self.live_display: Optional[Live] = None

    def _get_style(self, key: str) -> Any:
        """
        Retrieve a style definition by key from the style set.
        
        Args:
            key: The style key to look up
            
        Returns:
            The style definition string, or empty string if key not found
        """
        return self.style_set.get(key, '')
    
    async def release(self) -> None:
        """
        Close live animations.

        Args:
            None

        Returns:
            None
        """
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

    async def before_response_async(self, *args, **kwargs) -> None:
        """
        Display a loading spinner before the LLM generates a response.
        
        This method is called asynchronously before the language model responds,
        showing a visual indicator that processing is happening. It creates and starts
        a Live display with a spinner animation that continues until the response arrives.
        """
        # Create a spinner with descriptive text
        spinner = Spinner('dots', Text('LLM backend is inferencing...', style = self._get_style('text')))
        # Wrap the spinner in a panel with styling
        panel = Panel(
            spinner,
            border_style = self._get_style('border_line'),
            title = Text('Processing', style = self._get_style('box_title')),
            title_align = 'left'
        )
        # Create a Live display instance to show the spinner
        self.live_display = Live(
            panel,
            console = self.console,
            refresh_per_second = 12,  # Update spinner 12 times per second
            auto_refresh = True,      # Automatically refresh the display
            transient = True          # Remove the display when stopped
        )
        self.live_display.start()

    async def after_response_async(self, model_response: ModelResponse, *args, **kwargs) -> None:
        """
        Display the model response after it has been received.
        
        This method handles displaying the agent's response including:
        - Reasoning content (if present)
        - Final answer/response
        - Tool calls that were made
        
        It stops the loading spinner and presents the content in formatted panels.
        
        Args:
            model_response: The ModelResponse object containing the agent's output
        """
        # Stop the spinner since response has arrived
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

        # Display reasoning and/or response content if available
        if model_response.reasoning_content or model_response.response:
            self.console.rule(
                Text("Response", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )
            # Display reasoning content in a separate panel
            if model_response.reasoning_content:
                panel = Panel(
                    Text(model_response.reasoning_content, style = self._get_style('text')), 
                    border_style = self._get_style('box_line_explicit'),
                    title = Text('Reasoning', style = self._get_style('box_title_explicit')),
                    title_align = 'left'
                )
                self.console.print(panel)
            # Display the final response/answer in a separate panel
            if model_response.response:
                panel = Panel(
                    Markdown(model_response.response),  # Render response as markdown
                    border_style = self._get_style('box_line_explicit'),
                    title = Text('Answer', style = self._get_style('box_title_explicit')),
                    title_align = 'left'
                )
                self.console.print(panel)
        
        # Display any tool calls that were made during the response
        if model_response.tool_call:
            for call in model_response.tool_call:
                # Create formatted text showing the tool being used
                text = Text(f"⚒️  Using tool ", style = self._get_style('text'))
                text.append(f"{call.name}", style = self._get_style('text_explicit'))
                # Show tool arguments with truncation if too long
                group = Group(text, Pretty(call.arguments, max_string = self.max_string))
                panel = Panel(
                    group,
                    border_style = self._get_style('box_line_weak'),
                    title = Text('Tool Call', style = self._get_style('box_title_weak')),
                    title_align = 'left'
                )
                self.console.print(panel)

    async def after_tool_call_async(self, tool_returns: List[ToolReturn], **kwargs) -> None:
        """
        Display the results of tool calls after they have been executed.
        
        This method formats and shows the return values from tools that were called
        by the agent, making it clear which tool returned what information.
        
        Args:
            tool_returns: List of ToolReturn objects containing the results from tool executions
        """
        for ret in tool_returns:
            # Create formatted text showing the tool name and its return value
            text = Text(f"⚙️  Tool ", style = self._get_style('text'))
            text.append(f"{ret.original_call.name} ", style = self._get_style('text_explicit'))
            text.append("returns:\n", style = self._get_style('text'))
            text.append(f"{ret.content}", style = self._get_style('text'))
            text.truncate(self.max_text, overflow = 'ellipsis')
            # Display the return value in a styled panel
            panel = Panel(
                text,
                border_style = self._get_style('box_line_weak'),
                title = Text('Tool Return', style = self._get_style('box_title_weak')),
                title_align = 'left'
            )
            self.console.print(panel)

class RichTerminalInterface(UserInputInterface, UserOutputInterface):
    """
    A rich terminal interface that handles user input/output for the agent framework.
    
    This class implements both UserInputInterface and UserOutputInterface to provide
    a complete terminal-based interaction layer for the agent system. It features:
    - Styled terminal output using the rich library
    - Skill retrieval and selection capabilities
    - Permission request handling
    - Integration with the agent loop system
    
    The interface manages two key aspects of agent interaction:
    1. Input: Capturing user instructions and handling permission requests
    2. Output: Displaying agent responses, skills, and notifications with rich formatting
    
    Attributes:
        name (str): Name identifier for the interface instance
        system_prompt (str): Default system prompt for the conversation
        retriever (SkillRetriever): Component for retrieving relevant skills based on user input
        style_set (Dict): Dictionary of styling configurations for different UI elements
        max_string (int): Maximum length for displaying strings before truncation
        console (Console): Rich console instance for formatted output
        skills_loaded (bool): Flag tracking whether skills have been loaded for current session
    """
    def __init__(
        self,
        name: str = 'interface',
        system_prompt: str = 'You are a helpful assistant.',
        retriever: Optional[SkillRetriever] = None,
        style_set: Optional[Dict] = None,
        max_string: int = 80,
        max_text: int = 2000
    ) -> None:
        """
        Initialize the RichTerminalInterface with configuration options.
        
        Args:
            name: Name identifier for this interface instance (displayed in prompts)
            system_prompt: Default system prompt template for the conversation
            retriever: SkillRetriever instance for fetching relevant skills (creates default if None)
            style_set: Custom styling dictionary to override default styles
            max_string: Maximum string length before truncation in displays
        """
        super().__init__()

        self.name = name
        self.system_prompt = system_prompt
        self.retriever = retriever if retriever is not None else SkillRetriever()
        # Default styling configuration for various UI elements
        self.style_set = {
            'rule_title': 'not italic bold gold1',      # Style for rule titles (e.g., "Skills Useful")
            'rule_line': 'dark_orange',                 # Style for rule lines separating sections
            'name': 'not italic bold dark_orange',      # Style for interface name display
            'user': 'not italic bold violet',           # Style for user-related text
            'text': 'not italic not bold grey100',      # Style for general text content
            'text_weak': 'not italic not bold grey70',  # Style for secondary/weak text
            'text_explicit': 'not italic bold yellow1', # Style for emphasized/explicit text
            '$': 'not italic bold grey100',             # Style for prompt symbols (e.g., ❯❯)
            'box_title': 'not italic bold gold1',       # Style for panel box titles
            'border_line': 'dark_orange',               # Style for panel borders
            'box_title_weak': 'not italic bold violet', # Style for weak panel titles
            'box_line_weak': 'orchid',                  # Style for weak panel borders
            'box_title_explicit': 'not italic bold orange_red1', # Style for explicit panel titles
            'box_line_explicit': 'gold1'                # Style for explicit panel borders
        }
        self.max_string = max_string
        self.max_text = max_text

        # Override default styles if custom style set provided
        if style_set is not None:
            self.style_set = self.style_set | style_set

        self.console = Console()
        self.skills_loaded: bool = False

    def _get_style(self, key: str) -> Any:
        """
        Retrieve a style definition by key from the style set.
        
        Args:
            key: The style key to look up
            
        Returns:
            The style definition string, or empty string if key not found
        """
        return self.style_set.get(key, '')
    
    def _load_skills(self, instruction: str) -> Optional[PromptTemplate]:
        """
        Load and present relevant skills to the user based on their instruction.
        
        This method retrieves skills that might be useful for the given instruction,
        displays them in a formatted way, and allows the user to select which ones
        to incorporate into the current agent session.
        
        Args:
            instruction: The user's instruction that determines which skills to retrieve
            
        Returns:
            PromptTemplate containing selected skills, or None if no skills were selected
        """
        if self.skills_loaded:
            return None
        
        self.skills_loaded = True

        if self.retriever:
            retrieved = self.retriever.search(instruction)

            if not retrieved:
                return None
            
            # Display the header for useful skills section
            self.console.rule(
                Text("Skills Useful", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )

            # Display each retrieved skill in a formatted panel
            for idx, skill in enumerate(retrieved):
                panel = Panel(
                    Text(skill['description'], style = self._get_style('text')), 
                    border_style = self._get_style('box_line_weak'),
                    title = Text(f"{idx + 1}. {skill['title']}", style = self._get_style('box_title_weak')),
                    title_align = 'left'
                )
                self.console.print(panel)

            # Prompt user to select which skills to load
            self.console.rule(
                Text("Loading Skills", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )

            # Create styled input prompt asking for skill indices
            text = Text("Load them? [use indices like ", style = self._get_style('text_weak'))
            text.append("1, 2, 3, ...", style = self._get_style('text_explicit'))
            text.append("] ", style = self._get_style('text_weak'))
            text.append("❯❯ ", style = self._get_style('$'))

            selected = self.console.input(text)
            # Parse the comma-separated indices provided by user
            selected = [max(int(part.strip()) - 1, 0) for part in selected.split(',') if part.strip().isdigit()]
            
            # Create a prompt template containing the selected skills
            prompt = PromptTemplate().add_main_title('Skills that may be useful')
            for idx in selected:
                try:
                    prompt.add_simple(retrieved[idx]['content'])
                except Exception:
                    pass

            return prompt


    async def next_message(self) -> UserMessage:
        """
        Asynchronously capture the next user message/Instruction.
        
        This method displays the interface status, prompts for user input,
        processes the input (including skill loading), and returns a structured
        UserMessage containing the necessary components for the agent loop.
        
        Returns:
            UserMessage containing either a quit signal or system/user prompts
        """
        # Display that the agent interface is ready for input
        self.console.rule(
            Text("Agent Online", style = self._get_style('rule_title')), 
            style = self._get_style('rule_line')
        )

        # Create styled input prompt with interface name
        text = Text(f"[{self.name}]: ", style = self._get_style('name'))
        text.append(f"your instruction", style = self._get_style('text_weak'))
        text.append(f" ❯❯ ", style = self._get_style('$'))
        instruction: str = self.console.input(text)

        # Check if user wants to exit the interface
        if instruction.strip() == 'exit':
            return UserMessage(quit = True)
        else:
            # Determine the system prompt to use (from retriever or default)
            if self.retriever:
                system_prompt = self.retriever.get_system_prompt()
            else:
                system_prompt = PromptTemplate(self.system_prompt)
            
            # Attempt to load relevant skills based on the instruction
            skills_prompt = self._load_skills(instruction)
            if skills_prompt:
                # Combine user instruction with selected skills
                user_prompt = PromptTemplate(instruction) + skills_prompt
            else:
                # Use just the user instruction without additional skills
                user_prompt = PromptTemplate(instruction)
            return UserMessage(system_prompt = system_prompt, user_prompt = user_prompt)

    def get_input_plugin(self) -> Optional[ContextPlugin]:
        """
        Retrieve the input plugin for this interface.
        
        This interface doesn't provide an input plugin, so returns None.
        
        Returns:
            None, as this interface doesn't implement additional input functionality
        """
        return None
    
    def get_output_plugin(self) -> Optional[ContextPlugin]:
        """
        Retrieve the output plugin for this interface.
        
        Creates and returns a RichTerminalOutputPlugin instance that handles
        formatted output display for the agent's responses.
        
        Returns:
            RichTerminalOutputPlugin instance configured with this interface's settings
        """
        return RichTerminalOutputPlugin(self.console, self.style_set, self.max_string, self.max_text)

    async def push_message(self, notify: UserNotify) -> None:
        if notify.notify_type == 'notify':
            if notify.title:
                title = Text(notify.title, style = self._get_style('box_title_explicit'))
            else:
                title = Text('Notice', style = self._get_style('box_title_explicit'))

            if notify.content:
                panel = Panel(
                    Text(notify.content, style = self._get_style('text')),
                    border_style = self._get_style('box_line_explicit'),
                    title = title,
                    title_align = 'left'
                )
                self.console.print(panel)

            if notify.objects:
                if notify.object_display_type == 'table':
                    table = Table(title = title, border_style = self._get_style('box_line_explicit'), expand = True, box = box.HORIZONTALS)
                    first_item = notify.objects[0]
                    for k, _ in first_item.items():
                        table.add_column(Text(str(k), style = self._get_style('text_explicit')), justify = 'left')
                    for item in notify.objects:
                        entries = []
                        for v in item.values():
                            entries.append(Text(str(v), style = self._get_style('text')))
                        table.add_row(*entries)
                    self.console.print(table)

    async def notify_user(self, notify: UserNotify) -> UserResponse:
        """
        Handle user notifications, particularly permission requests.
        
        This method processes different types of notifications from the agent,
        with special handling for permission requests that require user approval.
        
        Args:
            notify: UserNotify object containing notification details
            
        Returns:
            UserResponse indicating the user's decision/approval
        """
        if notify.title:
            title = Text(notify.title, style = self._get_style('box_title_explicit'))
        else:
            title = Text('Request', style = self._get_style('box_title_explicit'))

        if notify.notify_type == 'permission':
            if notify.content:
                # Display the permission request content in a styled panel
                panel = Panel(
                    Text(notify.content, style = self._get_style('text')),
                    border_style = self._get_style('box_line_explicit'),
                    title = title,
                    title_align = 'left'
                )
                self.console.print(panel)
            
            # Create styled prompt asking for permission
            prompt_text = Text("Allow this operation? [", style = self._get_style('text_weak'))
            prompt_text.append("yes", style = self._get_style('text_explicit'))
            prompt_text.append("/", style = self._get_style('text_weak'))
            prompt_text.append("no", style = self._get_style('text_explicit'))
            prompt_text.append(" (or ", style = self._get_style('text_weak'))
            prompt_text.append("no: reason", style = self._get_style('text_explicit'))
            prompt_text.append(")] ", style = self._get_style('text_weak'))
            prompt_text.append("❯❯ ", style = self._get_style('$'))
            
            # Display permission request header
            self.console.rule(
                Text("Permission Required", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )

            # Keep prompting until valid response received
            while True:
                response = self.console.input(prompt_text).strip()
                
                # Handle empty response (defaults to denial)
                if not response:
                    return UserResponse(permitted = False, reason = 'User denied the request without any explanations.')
                
                # Handle positive response
                if response.lower().startswith('yes'):
                    return UserResponse(permitted = True)
                
                # Handle negative response with optional reason
                if response.lower().startswith('no'):
                    if ':' in response:
                        _, reason = response.split(':', 1)
                        reason = reason.strip()
                        return UserResponse(permitted = False, reason = reason if reason else '')
                    else:
                        return UserResponse(permitted = False)
                
                # Invalid input - show error message and continue loop
                text = Text("Invalid input. Please enter '", style = self._get_style('text_weak'))
                text.append("yes", style = self._get_style('text_explicit'))
                text.append("', '", style = self._get_style('text_weak'))
                text.append("no", style = self._get_style('text_explicit'))
                text.append("' or '", style = self._get_style('text_weak'))
                text.append("no: reason", style = self._get_style('text_explicit'))
                text.append("'.", style = self._get_style('text_weak'))
                self.console.print(text)

        elif notify.notify_type == 'conversation':
            if notify.content:
                # Display the conversational content in a styled panel
                panel = Panel(
                    Markdown(notify.content),
                    border_style = self._get_style('box_line_explicit'),
                    title = title,
                    title_align = 'left'
                )
                self.console.print(panel)

            # Create styled prompt asking for advises
            prompt_text = Text("Enter your ", style = self._get_style('text_weak'))
            prompt_text.append("opinion ", style = self._get_style('text_explicit'))
            prompt_text.append("or ", style = self._get_style('text_weak'))
            prompt_text.append("choice ", style = self._get_style('text_explicit'))
            prompt_text.append("❯❯ ", style = self._get_style('$'))

            # Display conversation header
            self.console.rule(
                Text("Feedback Required", style = self._get_style('rule_title')), 
                style = self._get_style('rule_line')
            )

            response = self.console.input(prompt_text).strip()

            return UserResponse(content = response)

    def get_exception_handler(self) -> RichTerminalExceptionHandler:
        """
        Get the exception handler for this interface.
        
        Creates and returns a RichTerminalExceptionHandler instance that provides
        rich-formatted error display in the terminal.
        
        Returns:
            RichTerminalExceptionHandler instance configured with this interface's console and styles
        """
        return RichTerminalExceptionHandler(self.console, self.style_set)

if __name__ == '__main__':
    pass
