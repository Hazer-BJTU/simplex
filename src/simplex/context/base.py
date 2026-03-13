import os
import copy
import uuid

from abc import ABC
from dataclasses import asdict
from typing import Dict, List, Any

import simplex.basics

from simplex.basics import (
    ModelInput,
    ModelResponse,
    ToolReturn,
    PromptTemplate,
    LoopInformation
)

class ContextPlugin(ABC):
    """
    Abstract base class for managing a context plugin with lifecycle management.
    
    Provides a standardized interface for context plugin, including:
    - Async context manager support (build/release lifecycle)
    - Lifecycle hooks for agent loop integration
    - Cloning and reset capabilities
    
    This class is designed to be subclassed to implement specific context plugins.
    """

    def __init__(self, instance_id: str) -> None:
        """
        Initialize a ContextPlugin instance.
        
        Args:
            instance_id: Unique identifier for this context plugin instance
        """
        super().__init__()
        self.__instance_id = instance_id

    @property
    def key(self) -> str:
        """
        Get the unique identifier for this context plugin instance.
        
        Returns:
            The instance ID set during initialization (read-only property)
        """
        return self.__instance_id
    
    async def __aenter__(self):
        """
        Async context manager entry point.
        
        Executes the build method to initialize resources when entering the context.
        
        Returns:
            The initialized ToolCollection instance
        """
        await self.build()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Async context manager exit point.
        
        Executes the release method to clean up resources when exiting the context,
        regardless of whether an exception occurred.
        
        Args:
            exc_type: Type of exception (if any) that caused the context to exit
            exc: Exception instance (if any)
            tb: Traceback object (if any)
            
        Returns:
            False to propagate any exceptions that occurred
        """
        await self.release()
        return False
    
    async def build(self) -> None:
        """
        Initialize resources for the context plugin (async).
        
        This method is called when entering the async context manager.
        Subclasses should override this to implement resource initialization
        (e.g. connecting to external services, loading models).
        """
        pass

    async def release(self) -> None:
        """
        Clean up resources for the context plugin (async).
        
        This method is called when exiting the async context manager.
        Subclasses should override this to implement resource cleanup
        (e.g. closing connections, releasing memory).
        """
        pass

    async def reset(self) -> None:
        """
        Reset the context plugin to its initial state (async).
        
        Subclasses should override this to implement state reset logic,
        allowing the context plugin to be reused without reinitialization.
        """
        pass

    def clone(self) -> "ContextPlugin":
        """
        Create a deep copy of the ToolCollection instance.
        
        Returns:
            A new ToolCollection instance with identical state
        """
        return copy.deepcopy(self)
    
    def process_prompt(self, *args, **kwargs) -> Any:
        """
        Lifecycle hook for prompt preprocessing (sync).
        
        Called before the agent loop starts to process/modify prompts.
        Subclasses should override this to implement custom prompt processing logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any processed result (return value is context-dependent)
        """
        pass

    def start_loop(self, *args, **kwargs) -> Any:
        """
        Lifecycle hook for agent loop startup (sync).
        
        Called at the beginning of the agent loop before iterations start.
        Subclasses should override this to implement loop initialization logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any startup result (return value is context-dependent)
        """
        pass

    async def start_loop_async(self, *args, **kwargs) -> Any:
        """
        Async lifecycle hook for agent loop startup.
        
        Async version of start_loop, called first to avoid state modification conflicts.
        Subclasses should override this to implement async loop initialization logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any startup result (return value is context-dependent)
        """
        pass

    def before_response(self, *args, **kwargs) -> Any:
        """
        Lifecycle hook called before model response generation (sync).
        
        Subclasses should override this to implement pre-response logic
        (e.g. modifying model input, logging).
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any pre-processing result (return value is context-dependent)
        """
        pass

    async def before_response_async(self, *args, **kwargs) -> Any:
        """
        Async lifecycle hook called before model response generation.
        
        Async version of before_response, called first to avoid state modification conflicts.
        Subclasses should override this to implement async pre-response logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any pre-processing result (return value is context-dependent)
        """
        pass

    def after_response(self, *args, **kwargs) -> Any:
        """
        Lifecycle hook called after model response generation (sync).
        
        Subclasses should override this to implement post-response logic
        (e.g. processing model output, logging).
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any post-processing result (return value is context-dependent)
        """
        pass

    async def after_response_async(self, *args, **kwargs) -> Any:
        """
        Async lifecycle hook called after model response generation.
        
        Async version of after_response, called first to avoid state modification conflicts.
        Subclasses should override this to implement async post-response logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any post-processing result (return value is context-dependent)
        """
        pass

    def after_tool_call(self, *args, **kwargs) -> Any:
        """
        Lifecycle hook called after tool execution (sync).
        
        Subclasses should override this to implement post-tool-execution logic
        (e.g. processing tool results, updating state).
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any post-processing result (return value is context-dependent)
        """
        pass

    async def after_tool_call_async(self, *args, **kwargs) -> Any:
        """
        Async lifecycle hook called after tool execution.
        
        Async version of after_tool_call, called first to avoid state modification conflicts.
        Subclasses should override this to implement async post-tool-execution logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any post-processing result (return value is context-dependent)
        """
        pass

    def after_final_response(self, *args, **kwargs) -> Any:
        """
        Lifecycle hook called after final response generation (sync).
        
        Called when the agent loop produces a final answer (no more tool calls).
        Subclasses should override this to implement final response processing logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any post-processing result (return value is context-dependent)
        """
        pass

    async def after_final_response_async(self, *args, **kwargs) -> Any:
        """
        Async lifecycle hook called after final response generation.
        
        Async version of after_final_response, called first to avoid state modification conflicts.
        Subclasses should override this to implement async final response processing logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any post-processing result (return value is context-dependent)
        """
        pass

    def on_loop_end(self, *args, **kwargs) -> Any:
        """
        Lifecycle hook called at the end of each loop iteration (sync).
        
        Subclasses should override this to implement per-iteration cleanup/processing logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any iteration result (return value is context-dependent)
        """
        pass

    async def on_loop_end_async(self, *args, **kwargs) -> Any:
        """
        Async lifecycle hook called at the end of each loop iteration.
        
        Async version of on_loop_end, called first to avoid state modification conflicts.
        Subclasses should override this to implement async per-iteration logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any iteration result (return value is context-dependent)
        """
        pass

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
        instance_id: str = uuid.uuid4().hex,
        empty_on_reset: bool = True,
        line_width: int = 150
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

        super().__init__(instance_id)

        self.empty_on_reset = empty_on_reset
        self.line_width = line_width

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
        return

    async def start_loop_async(self, model_input: ModelInput, **kwargs) -> Any:
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
        self.log.append(LoopInformation(model_input = model_input))
        
        # Log markdown.
        self.markdown.add_main_title('Initial states')
        if model_input.tools:
            self.markdown.add_block([schema.human_readable_descriptions(self.line_width) for schema in model_input.tools], 'Tools available', 'yaml')
        if model_input.messages:
            for message in model_input.messages:
                if 'role' in message and 'content' in message:
                    self.markdown.add_simple(message['content'], message['role'])
        return
    
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
        self.loopinfo = LoopInformation(model_response = model_response)

        # Log markdown.
        self.markdown.add_main_title(f"Agent iteration #{iter}")
        if model_response.reasoning_content:
            self.markdown.add_simple(model_response.reasoning_content, "Reason content")
        if model_response.tool_call:
            self.markdown.add_block([tool_call.human_readable_descriptions(self.line_width) for tool_call in model_response.tool_call], 'Function calling')
        if model_response.response:
            self.markdown.add_simple(model_response.response, "Model Response")
        return
    
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
        return
    
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
        return

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
