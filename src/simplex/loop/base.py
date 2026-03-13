import os
import sys
import uuid
import copy
import inspect
import asyncio

from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Coroutine, List, Dict, Optional, Any

import simplex.basics
import simplex.context
import simplex.models
import simplex.tools

from simplex.basics import (
    ModelInput,
    ModelResponse,
    ToolCall,
    ToolReturn,
    ToolSchema,
    ConflictError,
    EntityInitializationError,
    PromptTemplate,
    RequestError,
    Notice
)
from simplex.context import ContextPlugin
from simplex.models import ConversationModel
from simplex.tools import ToolCollection


class ExceptionHandler(ABC):
    """
    Handles exceptions generated during the agent loop execution.
    
    This class is designed to process exceptions that occur in the context of
    sequential/asynchronous method calls of components such as ContextPlugin and
    ToolCollection. It can extract and handle exceptions from a result list
    (similar to the return value of asyncio.gather), as well as process individual exceptions.
    """

    def __init__(self, instance_id) -> None:
        super().__init__()
        self.__instance_id = instance_id

    @property
    def key(self) -> str:
        return self.__instance_id

    @abstractmethod
    def handle_exception(self, exception: Exception) -> None:
        """
        Process a single exception instance (to be implemented by subclasses).

        This abstract method must be overridden by any subclass of ExceptionHandler.
        It defines the specific logic for handling an individual exception.

        Args:
            exception: The single Exception instance to be processed
        """
        pass

    def __call__(self, *args: Exception | List) -> None:
        for arg in args:
            if isinstance(arg, Exception):
                self.handle_exception(arg)
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, Exception):
                        self.handle_exception(item)

    def clone(self) -> "ExceptionHandler":
        return copy.deepcopy(self)

class LogExceptionHandler(ExceptionHandler):
    def __init__(self, instance_id = uuid.uuid4().hex) -> None:
        super().__init__(instance_id)
        self.content: str = ''

    def handle_exception(self, exception: Exception) -> None:
        """
        Process a single exception instance with timestamped logging.

        This concrete implementation formats the exception with current timestamp,
        prints the error message to standard error stream (stderr), and appends
        the formatted message to the instance's `content` attribute for persistent
        storage/audit purposes.

        Args:
            exception: The single Exception instance to be processed and logged
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{current_time}] {type(exception).__name__}: {exception}\n"
        print(message, file = sys.stderr)
        self.content += message

    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"LogExceptionHandler(key={repr(self.key)}, content={repr(self.content)})"

async def call_coroutine_functions(
    target_list: List[Any],
    names: str | List[str],
    arguments: Dict[str, Any] | List[Dict[str, Any]],
    exception_handler: ExceptionHandler
) -> List[Any]:
    """
    Asynchronously call coroutine functions on multiple target objects.

    This function dynamically invokes specified coroutine methods on a list of target objects,
    handles input parameter normalization (supports single/multiple names/arguments),
    validates the existence and type of target methods, executes all coroutines concurrently,
    and passes the results (including exceptions) to the specified exception handler.

    Args:
        target_list: List of target objects on which coroutine functions will be called
        names: Single method name (str) or list of method names (one per target in target_list)
        arguments: Single dict of keyword arguments (applied to all targets) or list of dicts
                   (one argument dict per target in target_list)
        exception_handler: Instance of ExceptionHandler (or compatible callable) to process
                           exceptions from the async gather results

    Returns:
        List of results from asyncio.gather (includes return values or exceptions for each call)

    Raises:
        AssertionError: If the length of names/arguments list doesn't match target_list length
        AttributeError: Indirectly (via Notice exception) if target lacks the specified attribute
        TypeError: Indirectly (via Notice exception) if target attribute is not a coroutine function

    Notes:
        - Uses asyncio.gather with return_exceptions=True to ensure all coroutines complete
        - Creates deep copies of the argument dict when applying a single dict to multiple targets
        - Validates that target attributes are callable coroutine functions before execution
    """
    async def call_exception(exception: Exception) -> Exception:
        return exception
    
    num_targets = len(target_list)
    
    # Normalize names: convert single string to list of same name for all targets
    if isinstance(names, list):
        assert len(names) == num_targets, 'names list length must match target list length'
        names_used = names
    elif isinstance(names, str):
        names_used = [names] * num_targets
    
    # Normalize arguments: convert single dict to deep-copied list for all targets
    if isinstance(arguments, list):
        assert len(arguments) == num_targets, 'arguments list length must match target list length'
        arguments_used = arguments
    elif isinstance(arguments, dict):
        arguments_used = [ copy.deepcopy(arguments) for _ in range(num_targets) ]

    call_list: List = []
    for name, argument, target in zip(names_used, arguments_used, target_list):
        if not hasattr(target, name):
            call_list.append(call_exception(Notice(f"{repr(target)} doesn't have attribute: {name}")))
            continue
        
        target_function = getattr(target, name)
        if not callable(target_function) or not inspect.iscoroutinefunction(target_function):
            call_list.append(call_exception(Notice(f"{repr(target)} doesn't have a coroutine function named: {name}")))
            continue
        
        call_list.append(target_function(**argument))

    # Execute all coroutines concurrently (return exceptions instead of raising)
    result_list = await asyncio.gather(*call_list, return_exceptions = True)
    # Pass results (including exceptions) to exception handler for processing
    exception_handler(result_list)
    return result_list

def call_functions(
    target_list: List[Any],
    names: str | List[str],
    arguments: Dict[str, Any] | List[Dict[str, Any]],
    exception_handler: ExceptionHandler
) -> List[Any]:
    """
    Synchronously call methods on multiple target objects with error handling.

    This function dynamically invokes specified methods on a list of target objects,
    normalizes input parameters (supports single/multiple names/arguments), validates
    the existence and callability of target methods, catches exceptions during execution,
    and passes all results (including exceptions/Notices) to the specified exception handler.

    Unlike the async version, this function executes method calls sequentially and
    synchronously, without using asyncio.

    Args:
        target_list: List of target objects on which methods will be called
        names: Single method name (str) applied to all targets, or list of method names
               (one name per target in target_list, must match length)
        arguments: Single dict of keyword arguments (deep-copied for each target) or list
                   of dicts (one argument dict per target, must match target_list length)
        exception_handler: Instance of ExceptionHandler to process exceptions/Notices
                           in the result list

    Returns:
        List of results from method calls:
        - Return value of the method if execution succeeds
        - Notice instance if target lacks the attribute/method
        - Exception instance if method execution raises an error

    Raises:
        AssertionError: If length of names/arguments list does not match target_list length
    """
    num_targets = len(target_list)
    
    # Normalize names: convert single string to list of same name for all targets
    if isinstance(names, list):
        assert len(names) == num_targets, 'names list length must match target list length'
        names_used = names
    elif isinstance(names, str):
        names_used = [names] * num_targets
    
    # Normalize arguments: convert single dict to deep-copied list for all targets
    if isinstance(arguments, list):
        assert len(arguments) == num_targets, 'arguments list length must match target list length'
        arguments_used = arguments
    elif isinstance(arguments, dict):
        arguments_used = [ copy.deepcopy(arguments) for _ in range(num_targets) ]

    result_list: List[Any] = []
    for name, argument, target in zip(names_used, arguments_used, target_list):
        if not hasattr(target, name):
            result_list.append(Notice(f"{repr(target)} doesn't have attribute: {name}"))
            continue
        
        target_function = getattr(target, name)
        if not callable(target_function):
            result_list.append(Notice(f"{repr(target)} doesn't have a method named: {name}"))
            continue
        
        try:
            result = target_function(**argument)
        except Exception as e:
            result = e

        result_list.append(result)

    # Pass results (including exceptions) to exception handler for processing
    exception_handler(result_list)
    return result_list

# ------------------------------ #
# Loop actions definitions       #
# ------------------------------ #
"""Literal type defining all valid loop action names for AgentLoop lifecycle events"""
LoopAction = Literal[
    'build',                      # Initialize resources (async)
    'release',                    # Clean up resources (async)
    'reset',                      # Reset loop state (async)
    'clone',                      # Create copy of instance (sync)
    'process_prompt',             # Preprocess prompt templates (sync)
    'start_loop',                 # Execute before main loop starts (sync)
    'start_loop_async',           # Execute before main loop starts (async)
    'before_response',            # Run before model generation (sync)
    'before_response_async',      # Run before model generation (async)
    'after_response',             # Run after model response (sync)
    'after_response_async',       # Run after model response (async)
    'after_tool_call',            # Run after tool execution (sync)
    'after_tool_call_async',      # Run after tool execution (async)
    'after_final_response',       # Run after final response (sync)
    'after_final_response_async', # Run after final response (async)
    'on_loop_end',                # Run at end of iteration (sync)
    'on_loop_end_async'           # Run at end of iteration (async)
]

@dataclass
class LoopStateEdit:
    """
    Dataclass representing the mutable state of AgentLoop that can be modified by lifecycle hooks
    
    This class encapsulates all loop variables that plugins/tools are allowed to modify
    during synchronous lifecycle hooks. Asynchronous hooks cannot modify the state directly.
    
    Attributes:
        system_prompt: System prompt template for the conversation
        user_prompt: User prompt template for the conversation
        model_input: Formatted input for the language model
        model_response: Raw response from the language model
        tool_returns: Results from executed tool calls
        exit_flag: Boolean flag to terminate loop early
    """

    system_prompt: Optional[PromptTemplate] = None
    user_prompt: Optional[PromptTemplate] = None
    model_input: Optional[ModelInput] = None
    model_response: Optional[ModelResponse] = None
    tool_returns: Optional[List[ToolReturn]] = None
    exit_flag: Optional[bool] = None

class AgentLoop:
    """
    Core class managing the lifecycle and execution loop of an AI agent
    
    This class coordinates model interactions, tool execution, and plugin/tool lifecycle
    management through a structured loop with synchronous and asynchronous hooks.
    
    Attributes:
        __model: Underlying conversation model for generating responses
        __exception_handler: Handler for processing exceptions during execution
        __tools: Collection of tool sets available to the agent
        __contexts: Context plugins modifying agent behavior/state
        __instances: Registry of all tools/contexts by unique key
        __tool_mapping: Map of tool names to their parent collections
        __tool_schemas: Combined schema definitions for all available tools
        __iter: Current iteration count in the main loop
        __system_prompt: System prompt template for the conversation
        __user_prompt: User prompt template for the conversation
        __model_input: Formatted input for the language model
        __model_response: Raw response from the language model
        __tool_returns: Results from executed tool calls
        __exit_flag: Mutable flag to terminate loop early (list for mutability)
    """

    def __init__(
        self,
        model: ConversationModel,
        exception_handler: ExceptionHandler,
        *args: ToolCollection | ContextPlugin
    ) -> None:
        """
        Initialize AgentLoop instance
        
        Args:
            model: Conversation model instance for generating responses
            exception_handler: Handler for processing exceptions (defaults to logging)
            *args: Variable list of ToolCollection/ContextPlugin instances to register
            
        Raises:
            EntityInitializationError: If initialization fails (duplicate keys/tools)
            ConflictError: If duplicate instance keys or tool names are detected
        """

        # Core dependencies
        self.__model = model
        self.__exception_handler = exception_handler

        # Registry for tools and context plugins
        self.__tools: List[ToolCollection] = []
        self.__contexts: List[ContextPlugin] = []
        self.__instances: Dict[str, ToolCollection | ContextPlugin] = {}
        
        # Tool mapping for quick lookup and schema storage
        self.__tool_mapping: Dict[str, ToolCollection] = {}
        self.__tool_schemas: List[ToolSchema] = []

        # Loop state management
        self.__iter: int = 0
        self.__system_prompt: PromptTemplate = PromptTemplate()
        self.__user_prompt: PromptTemplate = PromptTemplate()
        self.__model_input: ModelInput = ModelInput()
        self.__model_response: ModelResponse = ModelResponse()
        self.__tool_returns: List[ToolReturn] = []
        self.__exit_flag: List[bool] = [False]

        try:
            # Register all provided tool/context instances
            for instance in args:
                if instance.key in self.__instances:
                    raise ConflictError(f"duplicated instance key: {instance.key}")
                self.__instances[instance.key] = instance
                
                if isinstance(instance, ToolCollection):
                    self.__tools.append(instance)
                elif isinstance(instance, ContextPlugin):
                    self.__contexts.append(instance)

            for tool in self.__tools:
                schemas = tool.get_tool_schemas()
                for schema in schemas:
                    if schema.name in self.__tool_mapping:
                        raise ConflictError(f"duplicated tool name \'{schema.name}\' from collection: {repr(tool)}")
                    self.__tool_mapping[schema.name] = tool
                self.__tool_schemas.extend(schemas)
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)

    def __getitem__(self, instance_key: str) -> ToolCollection | ContextPlugin:
        """
        Access registered instances by key (dict-like access)
        
        Args:
            instance_key: Unique key of the tool/context instance
            
        Returns:
            The requested ToolCollection or ContextPlugin instance
        """
        return self.__instances[instance_key]

    @property
    def _instance_list(self) -> List[ToolCollection | ContextPlugin]:
        """
        Get list of all registered tool/context instances
        
        Returns:
            List of all ToolCollection/ContextPlugin instances
        """
        return list(self.__instances.values())
    
    @property
    def _captured_states(self) -> Dict[str, Any]:
        """
        Capture current loop state for lifecycle hook arguments
        
        Returns:
            Dictionary containing current loop state variables
        """
        captured_states = {
            'iter': self.__iter,
            'system_prompt': self.__system_prompt,
            'user_prompt': self.__user_prompt,
            'model_input': self.__model_input,
            'model_response': self.__model_response,
            'tool_returns': self.__tool_returns,
            'exit_flag': self.__exit_flag
        }
        return captured_states
    
    def _call_sequential(self, name: LoopAction) -> List[Any]:
        """
        Execute synchronous lifecycle hooks across all instances
        
        Iterates through all registered instances and attempts to call the specified
        method if it exists and is callable. Catches exceptions and returns them
        in the results list.
        
        Args:
            name: Name of the lifecycle method to execute
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            List of results/errors from method execution
        """

        captured: Dict[str, Any] = self._captured_states
        results: List[Any] = []
        for instance in self._instance_list:
            if not hasattr(instance, name):
                results.append(Notice(f"{repr(instance)} doesn't have attribute: {name}"))
                continue

            target = getattr(instance, name)
            
            if not callable(target):
                results.append(Notice(f"{repr(instance)} doesn't have a method named: {name}"))
                continue

            try:
                result = target(**copy.deepcopy(captured))
            except Exception as e:
                result = e
            results.append(result)

        # Pass results to exception handler
        self.__exception_handler(results)

        for result in results:
            if isinstance(result, LoopStateEdit):
                if result.system_prompt:
                    self.__system_prompt = result.system_prompt
                if result.user_prompt:
                    self.__user_prompt = result.user_prompt
                if result.model_input:
                    self.__model_input = result.model_input
                if result.model_response:
                    self.__model_response = result.model_response
                if result.tool_returns:
                    self.__tool_returns = result.tool_returns

        return results
    
    async def _call_async(self, name: LoopAction) -> List[Any]:
        """
        Execute asynchronous lifecycle hooks across all instances
        
        Similar to _call_sequential but for async methods, validating coroutine functions
        and executing them concurrently with asyncio.gather.
        
        Args:
            name: Name of the async lifecycle method to execute
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            List of results/errors from async method execution
        """

        async def _return_exception(exception: Exception) -> Exception:
            # Helper to return exceptions as async results
            return exception
        
        captured: Dict[str, Any] = self._captured_states
        tasks: List[Coroutine[Any, Any, Any]] = []
        for instance in self._instance_list:
            if not hasattr(instance, name):
                tasks.append(_return_exception(Notice(f"{repr(instance)} doesn't have attribute: {name}")))
                continue

            target = getattr(instance, name)

            if not callable(target) or not inspect.iscoroutinefunction(target):
                tasks.append(_return_exception(Notice(f"{repr(instance)} doesn't have a coroutine function named: {name}")))
                continue

            tasks.append(target(**copy.deepcopy(captured)))

        # Execute all async tasks concurrently (return exceptions instead of raising)
        results: List[Any] = await asyncio.gather(*tasks, return_exceptions = True)
        # Pass results to exception handler
        self.__exception_handler(results)
        return results
    
    async def build(self) -> None:
        # Trigger async build lifecycle hook for all instances
        await self._call_async('build')

    async def release(self) -> None:
        # Trigger async release lifecycle hook for all instances
        await self._call_async('release')

    async def reset(self) -> None:
        """
        Reset loop state to initial conditions and trigger reset hook
        
        Resets iteration counter, prompts, model I/O, tool returns, and exit flag,
        then calls reset lifecycle method on all instances.
        """
        self.__iter = 0
        self.__system_prompt = PromptTemplate()
        self.__user_prompt = PromptTemplate()
        self.__model_input = ModelInput()
        self.__model_response = ModelResponse()
        self.__tool_returns = []
        self.__exit_flag = [False]
        await self._call_async('reset')

    def clone(self) -> "AgentLoop":
        """
        Create a deep copy of the AgentLoop instance
        
        Clones the model, exception handler, and all registered instances via
        their clone methods, then initializes a new AgentLoop with these clones.
        
        Returns:
            New AgentLoop instance with cloned dependencies
        """
        return AgentLoop(
            self.__model.clone(),
            self.__exception_handler.clone(),
            *self._call_sequential('clone')
        )
    
    async def complete(
        self,
        system: Optional[PromptTemplate] = None,
        user: Optional[PromptTemplate] = None,
        history: Optional[List[Dict]] = None, # standard openai message list
        max_iteration: int = 30,
        timeout: float = 120,
        max_retry: int = 5,
        keep_original_system: bool = False
    ) -> ModelResponse:
        """
        Execute main agent loop to generate a complete response
        
        Orchestrates the full agent workflow: prompt processing, model generation,
        tool execution, and lifecycle hook management with retry logic and timeout.
        
        Args:
            system: System prompt template for the conversation
            user: User prompt template for the current request
            history: List of OpenAI-style message dicts (role/content pairs)
            max_iteration: Maximum number of loop iterations (prevents infinite loops)
            timeout: Timeout in seconds for model generation requests
            max_retry: Maximum retry attempts for failed model requests
            keep_original_system: Preserve original system prompt from history
            
        Returns:
            Final ModelResponse from the agent loop
            
        Raises:
            RuntimeError: If model generation fails after max_retry attempts
            Exception: For fatal errors during execution
        """
        
        async def tool_not_exists(original_call: ToolCall):
            """
            Create error response for non-existent tool calls
            
            Args:
                original_call: The tool call request for non-existent tool
                
            Returns:
                ToolReturn with error message about missing tool
            """
            return ToolReturn(content = f'[ERROR]: Tool {original_call.name} not exists. Please try again.', original_call = original_call)
        
        # Initialize prompt templates
        self.__system_prompt = system if system is not None else PromptTemplate()
        self.__user_prompt = user if user is not None else PromptTemplate()
        
        # Default to empty history if None provided
        if history is None:
            history = []

        # Process system prompt from history
        if len(history) > 0 and history[0].get('role', '') == 'system':
            if keep_original_system:
                # Append new system prompt to original if requested
                self.__system_prompt = history[0].get('content', '') + self.__system_prompt
            # Remove original system prompt to avoid duplication
            history.pop(0)

        # Preprocess prompts through lifecycle hook
        self._call_sequential('process_prompt')

        # Rebuild message history with processed prompts
        history.insert(0, {'role': 'system', 'content': str(self.__system_prompt)})
        # Append new user prompt for multi-turn chat completion
        history.append({'role': 'user', 'content': str(self.__user_prompt)})

        # Prepare model input with messages and tool schemas
        self.__model_input = ModelInput(messages = history, tools = self.__tool_schemas)

        # Execute pre-loop lifecycle hooks (async first to avoid state modification)
        await self._call_async('start_loop_async')
        self._call_sequential('start_loop')

        # Main iteration loop
        for curr_iter in range(max_iteration):
            self.__iter = curr_iter
            
            # Pre-response lifecycle hooks
            await self._call_async('before_response_async')
            self._call_sequential('before_response')
            
            # Model generation with retry logic
            for attempt in range(max_retry + 1):
                try:
                    self.__model_response = await asyncio.wait_for(self.__model.generate(model_input = self.__model_input), timeout = timeout)
                except (asyncio.TimeoutError, RequestError) as e:
                    self.__exception_handler(e, Notice(f"model endpoint retry attempt [{attempt}/{max_retry}]"))
                    if attempt == max_retry:
                        # Raise error after final failed attempt
                        raise RuntimeError(f"failed to receive from model endpoint after {max_retry} attempts")
                    await asyncio.sleep(0.5)
                    continue
                except Exception as e:
                    self.__exception_handler(e, Notice(f"loop quit due to unexpected error"))
                    raise e  # Fatal error, re-raise to terminate loop
                # Exit retry loop on successful generation
                break
            
            # Post-response lifecycle hooks
            await self._call_async('after_response_async')
            self._call_sequential('after_response')
            
            # Process tool calls if present in response
            if self.__model_response.tool_call is not None and len(self.__model_response.tool_call) > 0:
                tasks: List[Coroutine[Any, Any, ToolReturn]] = []
                for call in self.__model_response.tool_call:
                    if call.name in self.__tool_mapping:
                        tasks.append(self.__tool_mapping[call.name](call))
                    else:
                        tasks.append(tool_not_exists(call))
                
                # Execute all tool calls concurrently
                tool_return_with_exceptions = await asyncio.gather(*tasks, return_exceptions = True)
                self.__tool_returns = []
                for idx, ret in enumerate(tool_return_with_exceptions):
                    if isinstance(ret, TypeError):
                        # May be parameter mismatch errors
                        original_call: ToolCall = self.__model_response.tool_call[idx]
                        error_message = (
                            f"[ERROR]: Parameter error of tool call '{original_call.name}'. "
                            f"Please double-check the parameter requirements. {str(ret)}"
                        )
                        self.__tool_returns.append(ToolReturn(error_message, original_call))
                    elif isinstance(ret, Exception):
                        # Handle other tool execution exceptions
                        original_call: ToolCall = self.__model_response.tool_call[idx]
                        error_message = (
                            f"[ERROR]: An exception has occurred during tool call '{original_call.name}': {str(ret)}."
                        )
                        self.__tool_returns.append(ToolReturn(error_message, original_call))
                    elif isinstance(ret, ToolReturn):
                        # Valid tool return - add to list
                        self.__tool_returns.append(ret)

                # Notify exception handler of tool execution results
                self.__exception_handler(tool_return_with_exceptions)

                # Post-tool-execution lifecycle hooks
                await self._call_async('after_tool_call_async')
                self._call_sequential('after_tool_call')

                # Update model input for next iteration
                self.__model_input = self.__model.tool_return_integrate(self.__model_input, self.__model_response, self.__tool_returns)

            elif self.__model_response.response is not None and len(self.__model_response.response) > 0:
                # Post-final-response lifecycle hooks
                await self._call_async('after_final_response_async')
                self._call_sequential('after_final_response')
                break # Terminate loop (final answer generated)
            
            # End-of-iteration lifecycle hooks
            await self._call_async('on_loop_end_async')
            self._call_sequential('on_loop_end')

            # Check for early exit flag
            if self.__exit_flag[0]:
                break
        
        # Return final model response
        return self.__model_response

    async def __aenter__(self):
        # Async context manager entry point - build resources
        await self.build()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        # Async context manager exit point - release resources
        await self.release()
        return False

if __name__ == '__main__':
    pass
