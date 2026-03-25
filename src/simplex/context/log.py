import os
import uuid

from typing import Any, List, Dict, Optional

import simplex.basics
import simplex.context.base

from simplex.basics import (
    ModelResponse, 
    PromptTemplate,
    LoopInformation,
    ModelInput,
    ToolReturn
)
from simplex.context.base import ContextPlugin


class TrajectoryLogContext(ContextPlugin):
    """
    Context plugin for logging and tracking agent loop trajectory information.
    
    This plugin records the complete lifecycle of an agent loop, including:
    - Initial model input states
    - Per-iteration model responses and tool calls
    - Tool execution returns
    - Human-readable markdown formatting of the entire trajectory
    
    The logged data is available in multiple formats (raw list, markdown string, dictionary)
    for different use cases (debugging, reporting, serialization).
    """

    def __init__(
        self, 
        instance_id: Optional[str] = None,
        empty_on_reset: bool = True,
        line_width: int = 150,
        delta: bool = True
    ) -> None:
        """
        Initialize a TrajectoryLogContext instance.
        
        Args:
            instance_id: Unique identifier for this log context instance (auto-generated if not provided)
            empty_on_reset: If True, clears all logged data when reset() is called (default: True)
            line_width: Maximum line width for formatting human-readable markdown content (default: 150)
        
        Attributes initialized:
            log: List to store raw LoopInformation objects for each loop iteration
            markdown: PromptTemplate instance for building human-readable log output
            loopinfo: Current LoopInformation object being populated during iteration
        """

        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex)

        self.empty_on_reset = empty_on_reset
        self.line_width = line_width
        self.delta = delta

        self.log: List[LoopInformation] = []
        self.markdown: PromptTemplate = PromptTemplate()

        self.loopinfo: LoopInformation = LoopInformation()
    
    async def reset(self) -> None:
        """
        Reset the trajectory log context to its initial state (async).
        
        If empty_on_reset is True, clears all logged data (log list, markdown content, 
        and current loopinfo object) to prepare for a new agent loop.
        
        Overrides the base ContextPlugin.reset() method.
        """

        if self.empty_on_reset:
            self.log = []
            self.markdown = PromptTemplate()
            self.loopinfo = LoopInformation()

    async def start_loop_async(self, model_input: ModelInput, system_prompt: PromptTemplate, user_prompt: PromptTemplate, **kwargs) -> Any:
        """
        Async lifecycle hook for agent loop startup - logs initial loop state.
        
        Records the initial model input (tools, messages) to both the raw log list
        and the human-readable markdown template.
        
        Overrides the base ContextPlugin.start_loop_async() method.
        
        Args:
            model_input: Initial input data for the agent loop containing tools and messages
            **kwargs: Additional keyword arguments (unused in this implementation)
        
        Returns:
            None
        """

        # Log details.
        # self.log.append(LoopInformation(model_input = model_input))
        
        # Log markdown.
        self.markdown.add_main_title('Initial states')
        if model_input.tools:
            self.markdown.add_block([schema.human_readable_descriptions(self.line_width) for schema in model_input.tools], 'Tools available', 'yaml')
        # if model_input.messages:
        #     for message in model_input.messages:
        #         if 'role' in message and 'content' in message:
        #             self.markdown.add_simple(message['content'], message['role'])
        self.markdown.add_simple(str(system_prompt), 'System')
        self.markdown.add_simple(str(user_prompt), 'User')

    async def before_response_async(self, iter: int, model_input: ModelInput, **kwargs) -> Any:
        """
        Async lifecycle hook before sending model request - logs model input.
        
        Records the initial model input (tools, messages) within a request to the raw log.
        
        Overrides the base ContextPlugin.before_response_async() method.
        
        Args:
            model_input: Initial input data for the agent loop containing tools and messages
            **kwargs: Additional keyword arguments (unused in this implementation)
        
        Returns:
            None
        """

        # Log details.
        self.loopinfo = LoopInformation(model_input = model_input)
    
    async def after_response_async(self, iter: int, model_response: ModelResponse, **kwargs) -> Any:
        """
        Async lifecycle hook after model response generation - logs iteration response data.
        
        Records the model's response (reasoning, tool calls, text response) for the current
        loop iteration to both the raw loopinfo object and markdown template.
        
        Overrides the base ContextPlugin.after_response_async() method.
        
        Args:
            iter: Current loop iteration number (1-based)
            model_response: Response data from the model containing reasoning and tool calls
            **kwargs: Additional keyword arguments (unused in this implementation)
        
        Returns:
            None
        """

        # Log details.
        self.loopinfo.model_response = model_response

        # Log markdown.
        self.markdown.add_main_title(f"Agent iteration #{iter}")
        if model_response.reasoning_content:
            self.markdown.add_simple(model_response.reasoning_content, "Reason content")
        if model_response.tool_call:
            self.markdown.add_block([tool_call.human_readable_descriptions(self.line_width) for tool_call in model_response.tool_call], 'Function calling')
        if model_response.response:
            self.markdown.add_simple(model_response.response, "Model Response")
    
    async def after_tool_call_async(self, tool_returns: List[ToolReturn], **kwargs) -> Any:
        """
        Async lifecycle hook after tool execution - logs tool return data.
        
        Records the results from executed tools to the current loopinfo object and
        adds them to the markdown template for human readability.
        
        Overrides the base ContextPlugin.after_tool_call_async() method.
        
        Args:
            tool_returns: List of ToolReturn objects containing results from executed tools
            **kwargs: Additional keyword arguments (unused in this implementation)
        
        Returns:
            None
        """

        # Log details.
        self.loopinfo.tool_returns = tool_returns

        # Log markdown.
        if tool_returns:
            self.markdown.add_block([ret.content for ret in tool_returns], "Tool returns")
    
    async def on_loop_end_async(self, **kwargs) -> Any:
        """
        Async lifecycle hook at loop iteration end - finalizes iteration logging.
        
        Adds the fully populated loopinfo object (containing model response and tool returns)
        to the main log list to complete the record for this iteration.
        
        Overrides the base ContextPlugin.on_loop_end_async() method.
        
        Args:
            **kwargs: Additional keyword arguments (unused in this implementation)
        
        Returns:
            None
        """

        # Log details.
        self.log.append(self.loopinfo)

    async def after_final_response_async(self, *args, **kwargs) -> Any:
        """
        Async lifecycle hook called after final response generation.
        
        After final response is made, the 'on_loop_end / on_loop_end_async' hook will never be called.
        As a result, we need to append temporary loop info to log list.
        """

        # Log details.
        self.log.append(self.loopinfo)

    @property
    def detailed(self) -> List[LoopInformation]:
        """
        Get the raw detailed log of all loop iterations.
        
        Returns:
            List of LoopInformation objects containing complete trajectory data
            (model inputs, responses, tool returns) for each iteration
        """
        return self.log
    
    @property
    def human_readable(self) -> str:
        """
        Get a human-readable markdown string of the entire trajectory log.
        
        Returns:
            Formatted markdown string with sectioned information for initial state,
            each iteration's response, and tool returns
        """
        return str(self.markdown)
    
    @property
    def dictionary(self) -> List[Dict]:
        """
        Get the trajectory log as a list of dictionaries (serializable format).
        
        Converts each LoopInformation object in the log to a dictionary for
        easy serialization, storage, or API responses.
        
        Returns:
            List of dictionaries representing the complete trajectory log
        """
        return [info.to_dict() for info in self.log]
    

if __name__ == '__main__':
    pass
