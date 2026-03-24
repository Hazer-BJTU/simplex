import os
import copy
import uuid

from abc import ABC
from typing import Dict, List, Any, Optional

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
    
    async def bind_io(self, *args, **kwargs) -> Any:
        """
        Bind user input and output interfaces to the context plugin
        
        This method allows dynamic assignment of I/O interfaces, which can be used
        by plugins/tools during lifecycle hooks to interact with users.
        
        Args:
            input_interface: UserInputInterface instance for receiving input
            output_interface: UserOutputInterface instance for sending output
            
        Returns:
            None
        """
        pass
    
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

    def on_exit(self, *args, **kwargs) -> Any:
        """
        Lifecycle hook called at the end of the whole loop (sync).
        
        Subclasses should override this to implement post-loop cleanup/processing logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any result (return value is context-dependent)
        """
        pass

    async def on_exit_async(self, *args, **kwargs) -> Any:
        """
        Async lifecycle hook called at the end of the whole loop (async).
        
        Async version of on_exit, called first to avoid state modification conflicts.
        Subclasses should override this to implement async per-iteration logic.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments (typically contains loop state)
            
        Returns:
            Any result (return value is context-dependent)
        """
        pass

if __name__ == '__main__':
    pass
