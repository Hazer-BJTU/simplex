import os
import re
import copy
import yaml
import inspect

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

import simplex.basics

from simplex.basics import (
    ToolCall,
    ToolReturn,
    ToolSchema,
    ParameterError,
    ImplementationError
)

MODULE_PATH: Path = Path(__file__).resolve().parent
SCHEMA_DIR: Path = MODULE_PATH / 'schema'
SKILL_DIR: Path = MODULE_PATH / 'skill'

class ToolCollection(ABC):
    """
    Abstract base class for managing a collection of tools with lifecycle management.
    
    Provides a standardized interface for tool collections, including:
    - Instance identification and tool name mapping
    - Async context manager support (build/release lifecycle)
    - Tool dispatch mechanism with validation
    - Lifecycle hooks for agent loop integration
    - Cloning and reset capabilities
    
    This class is designed to be subclassed to implement specific tool collections,
    requiring concrete implementations for tool metadata retrieval methods.
    """

    def __init__(self, instance_id: str, name_mapping: Dict) -> None:
        """
        Initialize a ToolCollection instance.
        
        Args:
            instance_id: Unique identifier for this tool collection instance
            name_mapping: Dictionary mapping external tool names to internal method names
                          (e.g. {"external_tool_name": "internal_method_name"})
        """
        super().__init__()
        self.__instance_id = instance_id
        self.__name_mapping: Dict = name_mapping

    @property
    def key(self) -> str:
        """
        Get the unique identifier for this tool collection instance.
        
        Returns:
            The instance ID set during initialization (read-only property)
        """
        return self.__instance_id

    async def __call__(self, tool_call: ToolCall) -> ToolReturn:
        """
        Enable callable interface for tool execution (async).
        
        This method serves as a convenience wrapper around the dispatch method,
        allowing the tool collection instance to be called directly like a function.
        
        Args:
            tool_call: ToolCall object containing the tool name and arguments
            
        Returns:
            ToolReturn object with the execution result
        """
        return await self._dispatch(tool_call)
    
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
        Initialize resources for the tool collection (async).
        
        This method is called when entering the async context manager.
        Subclasses should override this to implement resource initialization
        (e.g. connecting to external services, loading models).
        """
        pass

    async def release(self) -> None:
        """
        Clean up resources for the tool collection (async).
        
        This method is called when exiting the async context manager.
        Subclasses should override this to implement resource cleanup
        (e.g. closing connections, releasing memory).
        """
        pass

    async def reset(self) -> None:
        """
        Reset the tool collection to its initial state (async).
        
        Subclasses should override this to implement state reset logic,
        allowing the tool collection to be reused without reinitialization.
        """
        pass

    def clone(self) -> "ToolCollection":
        """
        Create a deep copy of the ToolCollection instance.
        
        Returns:
            A new ToolCollection instance with identical state
        """
        return copy.deepcopy(self)
    
    async def bind_io(self, *args, **kwargs) -> Any:
        """
        Bind user input and output interfaces to the tool collection
        
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

    # @abstractmethod
    # def get_names(self) -> List[str]:
    #     pass
    # 
    # @abstractmethod
    # def get_tools(self) -> List[ToolSchema]:
    #     pass

    @abstractmethod
    def get_tool_schemas(self) -> List[ToolSchema]:
        pass

    @abstractmethod
    def tools_descriptions(self) -> str:
        """
        Get combined human-readable descriptions of all tools (abstract method).
        
        Subclasses must implement this to return a formatted string containing
        descriptions of all tools, typically used in system prompts.
        
        Returns:
            String containing descriptions of all available tools
        """
        pass
    
    async def _dispatch(self, tool_call: ToolCall) -> ToolReturn:
        """
        Dispatch a tool call to the appropriate internal method (async).
        
        Validates the tool call, maps the external tool name to internal method name,
        verifies the method exists and is a coroutine function, then executes it
        with the provided arguments.
        
        Args:
            tool_call: ToolCall object containing the tool name and arguments
            
        Returns:
            ToolReturn object with the execution result
            
        Raises:
            ParameterError: If the tool name is not in the name mapping or the method doesn't exist
            ImplementationError: If the target method is not a coroutine function
            Exception: Any exception raised by the target method (propagated)
        """

        function_name: str = tool_call.name
        arguments: Dict = tool_call.arguments

        try:
            member_name: str = self.__name_mapping[function_name]
        except KeyError:
            raise ParameterError(
                'dispatch',
                'tool_call',
                f'tool_call.name should be one of {str(self.__name_mapping)}',
                type_hint='ToolCall',
                class_name=self.__class__.__name__
            )
        
        if not hasattr(self, member_name):
            raise ParameterError(
                'dispatch',
                'tool_call',
                f'undefined tool_call.name = {function_name} or member {member_name}',
                type_hint='ToolCall',
                class_name=self.__class__.__name__
            )
        
        target_function = getattr(self, member_name)
        if not callable(target_function) or not inspect.iscoroutinefunction(target_function):
            raise ImplementationError(
                target_function.__name__ if hasattr(target_function, '__name__') else 'unknown_function',
                'target member should be a coroutine function',
                class_name=self.__class__.__name__
            )

        try:
            result_text: str = await target_function(**arguments)
        except Exception:
            raise
        return ToolReturn(result_text, tool_call)

def load_tool_definitions(file_name: str) -> str:
    try:
        with open(SCHEMA_DIR / f"{file_name}.yml", 'r', encoding = 'utf8') as schema_file:
            content: str = schema_file.read()
        return content.strip()
    except Exception:
        raise

def load_schema(file_name: str, tool_name: str, rename: Optional[str] = None) -> ToolSchema:
    try:
        with open(SCHEMA_DIR / f"{file_name}.yml", 'r', encoding = 'utf8') as schema_file:
            content = yaml.safe_load(schema_file)
        tool_schema: Dict = content.get(tool_name, {})
        if rename is not None:
            name: str = rename
        else:
            name: str = tool_schema.get('name', tool_name)
        description: str = tool_schema.get('description', '')
        params: List[Dict] = tool_schema.get('params', [])
        extras: Dict = tool_schema.get('extras', {})
        formated_params: List[ToolSchema.Parameter] = []
        for param in params:
            param_field: str = param.get('field', 'unknown')
            param_type: str = param.get('type', 'string')
            param_description: str = param.get('description', '')
            param_required: bool = param.get('required', True)
            param_extras: Dict = param.get('extras', {})
            formated_params.append(ToolSchema.Parameter(
                field = param_field,
                type = param_type,
                description = param_description,
                required = param_required,
                extras = param_extras
            ))
        return ToolSchema(
            name = name,
            description = description,
            params = formated_params,
            extras = extras
        )
    except Exception:
        raise

def load_tool_skill(file_name: str, replace_mapping: Optional[Dict[str, str]] = None) -> str:
    try:
        with open(SKILL_DIR / f"{file_name}.md", 'r', encoding = 'utf8') as skill_file:
            content = skill_file.read()
        
        if replace_mapping:
            pattern = r'%(\w+)%'
            def replace_func(match):
                key = match.group(1)
                return replace_mapping.get(key, match.group(0))            
            return re.sub(pattern, replace_func, content) # type: ignore
        else:
            return content
    except Exception:
        raise

# def to_openai_function_calling_schema(tool_schema: ToolSchema) -> Dict:
#     properties: Dict = {
#         param.field: {
#             'type': param.type,
#             'description': param.description
#         }
#         for param in tool_schema.params
#     }
#     required: List = [
#         param.field
#         for param in tool_schema.params
#         if param.required
#     ]
#
#     function_body: Dict = {
#         'name': tool_schema.name,
#         'description': tool_schema.description,
#         'parameters': {
#             'type': 'object',
#             'properties': properties,
#             'required': required
#         }
#     }
#
#     return {
#         'type': 'function',
#         'function': function_body
#     }

if __name__ == '__main__':
    pass
