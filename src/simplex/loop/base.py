import os
import sys
import uuid
import copy
import inspect
import asyncio

from datetime import datetime
from abc import ABC, abstractmethod
from typing import TypeAlias, List, Dict, Optional, Any

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
            elif isinstance(arg, List):
                for item in arg:
                    if isinstance(item, Exception):
                        self.handle_exception(item)

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

'''
class AgentLoopBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __getitem__(self, instance_key: str) -> ToolCollection | ContextPlugin:
        pass

    @abstractmethod
    async def build(self) -> None:
        pass

    @abstractmethod
    async def release(self) -> None:
        pass

    @abstractmethod
    async def reset(self) -> None:
        pass

    @abstractmethod
    def clone(self) -> "AgentLoopBase":
        pass

    @abstractmethod
    async def complete(self, *args, **kwargs) -> Optional[ModelResponse]:
        pass

    async def __aenter__(self):
        await self.build()
        return self
    
    async def __exit__(self, exc_type, exc, tb):
        await self.release()
        return False
'''
        
class AgentLoop:
    """
    Core loop controller for agent execution with state management.
    
    This class manages the full lifecycle of an agent, including initialization,
    state transitions (build/release/reset), prompt processing, model interaction,
    tool execution, and exception handling. It follows an async context manager
    pattern for resource management.

    Class Attributes:
        STATE_BUILD: State for initializing resources (tools/contexts)
        STATE_RELEASE: State for cleaning up resources
        STATE_RESET: State for resetting agent state (e.g., iteration counter)
        STATE_ON_START_PROCEDURE: State before prompt initialization
        STATE_ON_PROCESS_PROMPT: State for prompt template processing
        STATE_ON_PROMPT_READY: State when model input is fully prepared
        STATE_ON_MODEL_RESPONSE: State after receiving model response
        STATE_ON_TOOL_RETURN: State after tool execution completes
        STATE_ON_FINAL_ANSWER: State when final answer is generated
    """

    STATE_BUILD: str = 'build'       # async, all instances, no parameters
    STATE_RELEASE: str = 'release'   # async, all instances, no parameters
    STATE_RESET: str = 'reset'       # async, all instances, no parameters
    STATE_ON_START_PROCEDURE: str = 'on_start_procedure' # sync, all instances,  params: agent: AgentLoop
    STATE_ON_PROCESS_PROMPT: str = 'on_process_prompt'   # sync, all instances,  params: system: PromptTemplate, user: PromptTemplate, agent: AgentLoop
    STATE_ON_PROMPT_READY: str = 'on_prompt_ready'       # sync, all instances,  params: model_input: ModelInput, agent: AgentLoop
    STATE_ON_MODEL_RESPONSE: str = 'on_model_response'   # async, contexts only, params: model_input: ModelInput, model_response: ModelResponse, agent: AgentLoop
    STATE_ON_TOOL_RETURN: str = 'on_tool_return'         # async, contexts only, params: tool_return: List[ToolReturn], agent: AgentLoop
    STATE_ON_FINAL_ANSWER: str = 'on_final_answer'       # async, contexts only, params: model_response: ModelResponse, agent: AgentLoop

    def __init__(
        self,
        model: ConversationModel,
        exception_handler: ExceptionHandler = LogExceptionHandler(),
        *args: ToolCollection | ContextPlugin
    ) -> None:
        """
        Initialize the AgentLoop with model and plugin/tool instances.

        Args:
            model: Conversation model for generating responses
            exception_handler: Handler for processing exceptions (defaults to LogExceptionHandler)
            *args: Variable list of ToolCollection/ContextPlugin instances

        Raises:
            EntityInitializationError: If initialization fails (e.g., duplicate keys/names)
            ConflictError: If duplicate instance keys or tool names are detected
        """

        self.__model = model
        self.__exception_handler = exception_handler

        self.__iter: int = 0
        self.__tools: List[ToolCollection] = []
        self.__contexts: List[ContextPlugin] = []
        self.__instances: Dict[str, ToolCollection | ContextPlugin] = {}
        self.__tool_mapping: Dict[str, ToolCollection] = {}
        self.__tool_schemas: List[ToolSchema] = []

        try:
            for instance in args:
                if instance.key in self.__instances:
                    raise ConflictError(f"duplicated instance key: {instance.key}")
                self.__instances[instance.key] = instance
                
                if isinstance(instance, ToolCollection):
                    self.__tools.append(instance)
                elif isinstance(instance, ContextPlugin):
                    self.__contexts.append(instance)

            for tool in self.__tools:
                self.__tool_schemas.extend(tool.get_tools())
                for name in tool.get_names():
                    if name in self.__tool_mapping:
                        raise ConflictError(f"duplicated tool name \'{name}\' from collection: {repr(tool)}")
                    self.__tool_mapping[name] = tool
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)

    def __getitem__(self, instance_key: str) -> ToolCollection | ContextPlugin:
        """"
        Get a registered instance by key (tools/contexts).

        Args:
            instance_key: Unique key of the instance to retrieve

        Returns:
            Registered ToolCollection or ContextPlugin instance

        Raises:
            KeyError: If the instance key does not exist
        """
        return self.__instances[instance_key]
    
    @property
    def iter(self) -> int:
        # Current iteration count of the agent loop (read-only).
        return self.__iter

    @property
    def instances(self) -> Dict[str, ToolCollection | ContextPlugin]:
        # Dictionary of all registered instances (key -> instance).
        return self.__instances
    
    @property
    def instance_list(self) -> List[ToolCollection | ContextPlugin]:
        # Ordered list of instances (tools first, then contexts).
        return self.__tools + self.__contexts
    
    async def build(self) -> None:
        """
        Initialize all registered instances (tools/contexts) asynchronously.
        
        Calls the 'build' method on all instances via coroutine execution,
        with exception handling enabled.
        """
        await call_coroutine_functions(self.instance_list, self.STATE_BUILD, {}, self.__exception_handler)

    async def release(self) -> None:
        """
        Clean up all registered instances (tools/contexts) asynchronously.
        
        Calls the 'release' method on all instances via coroutine execution,
        with exception handling enabled.
        """
        await call_coroutine_functions(self.instance_list, self.STATE_RELEASE, {}, self.__exception_handler)

    async def reset(self) -> None:
        """
        Reset agent loop state and reinitialize instances.
        
        Resets the iteration counter to 0 and calls the 'reset' method on
        all instances asynchronously.
        """
        self.__iter = 0
        await call_coroutine_functions(self.instance_list, self.STATE_RESET, {}, self.__exception_handler)

    def clone(self) -> "AgentLoop":
        """
        Create a deep copy of the AgentLoop instance.
        
        Clones the model, exception handler, and all registered instances,
        preserving the same configuration but with independent state.

        Returns:
            New AgentLoop instance with cloned dependencies
        """
        new_model = self.__model.clone()
        new_handler = self.__exception_handler.clone()
        new_instances = [instance.clone() for instance in self.instance_list]
        return AgentLoop(new_model, new_handler, *new_instances)
    
    async def complete(
        self,
        system: Optional[str | PromptTemplate] = None,
        user: Optional[str | PromptTemplate] = None, 
        history: Optional[List[Dict]] = None,  
        max_iteration: int = 30,
        timeout: float = 120,
        max_retry: int = 5
    ) -> Optional[ModelResponse]:
        """
        Execute the full agent loop to generate a final response.
        
        Handles prompt initialization, model interaction, tool execution,
        and context updates in a loop until a final answer is generated or
        maximum iterations are reached.

        Args:
            system: System prompt (template or string)
            user: User prompt (template or string)
            history: Predefined conversation history (skips prompt initialization if provided)
            max_iteration: Maximum number of loop iterations (default: 30)
            timeout: Timeout (seconds) for model generation per attempt (default: 120)
            max_retry: Maximum retry attempts for model requests (default: 5)

        Returns:
            Final model response (None if loop exits without valid response)

        Raises:
            RuntimeError: If model requests fail after max_retry attempts
            Exception: Unhandled exceptions during loop execution (propagated)
        """

        async def tool_not_exists(original_call: ToolCall):
            """
            Generate error response for non-existent tool calls.
            
            Args:
                original_call: Tool call request for non-existent tool
            
            Returns:
                ToolReturn with error message
            """
            return ToolReturn(
                content = f'[ERROR]: Tool {original_call.name} not exists. Please try again.',
                original_call = original_call
            )

        # Step 1: Pre-prompt initialization hook (all instances)
        call_functions(self.instance_list, self.STATE_ON_START_PROCEDURE, {'agent': self}, self.__exception_handler)

        # Step 2: Initialize model input (with/without history)
        if history is None:
            # Generate prompt templates (system/user) (all instances)
            system_generated, user_generated = PromptTemplate(), PromptTemplate()
            call_functions(self.instance_list, self.STATE_ON_PROCESS_PROMPT, {'system': system_generated, 'user': user_generated, 'agent': self}, self.__exception_handler)
            model_input = ModelInput(
                messages = [
                    # If prompts are provided through arguments, use them in precedence.
                    {'role': 'system', 'content': str(system_generated) if system is None else str(system)},
                    {'role': 'user', 'content': str(user_generated) if user is None else str(user)}
                ], 
                tools = self.__tool_schemas
            )
        else:
            # Use existing history (skip prompt generation)
            model_input = ModelInput(messages = history, tools = self.__tool_schemas)

        model_response: Optional[ModelResponse] = None
        # Step 3: Post-prompt initialization hook (all instances)
        call_functions(self.instance_list, self.STATE_ON_PROMPT_READY, {'model_input': model_input, 'agent': self}, self.__exception_handler) # tools + contexts
        
        # Step 4: Main agent loop
        for current_iter in range(max_iteration):
            self.__iter = current_iter

            # Step 4.1: Model generation with retry/timeout
            model_response = None
            for attempt in range(max_retry):
                try:
                    model_response = await asyncio.wait_for(self.__model.generate(model_input = model_input), timeout = timeout)
                    break
                except asyncio.TimeoutError as e:
                    self.__exception_handler(e, Notice(f"model endpoint retry attempt [{attempt + 1}/{max_retry}]"))
                except RequestError as e:
                    self.__exception_handler(e, Notice(f"model endpoint retry attempt [{attempt + 1}/{max_retry}]"))
                except Exception as e:
                    self.__exception_handler(e, Notice(f"agent loop quit"))
                    return None
            
            assert model_response is not None, f"failed to receive from model endpoint after {max_retry} attempts"
            # Step 4.2: Post-model-response hook (contexts only)
            await call_coroutine_functions(self.__contexts, self.STATE_ON_MODEL_RESPONSE, {'model_input': model_input, 'model_response': model_response, 'agent': self}, self.__exception_handler) # contexts only

            # Step 4.3: Handle tool calls (if any)
            if model_response.tool_call is not None and len(model_response.tool_call) > 0:
                tasks: List = []
                for call in model_response.tool_call:
                    if call.name in self.__tool_mapping:
                        tasks.append(self.__tool_mapping[call.name](call))
                    else:
                        tasks.append(tool_not_exists(call))

                # Execute tool calls concurrently (capture exceptions)
                tool_return = await asyncio.gather(*tasks, return_exceptions = True)

                # Process tool return results (normalize to ToolReturn)
                pure_tool_return: List[ToolReturn] = []
                for idx, ret in enumerate(tool_return):
                    if isinstance(ret, TypeError):
                        # May be arguments mismatch
                        original_call: ToolCall = model_response.tool_call[idx]
                        error_message = (
                            f"[ERROR]: Parameter error of tool call '{original_call.name}'. "
                            f"Please double-check the parameter requirements. {str(ret)}"
                        )
                        pure_tool_return.append(ToolReturn(error_message, original_call))
                    elif isinstance(ret, Exception):
                        # Other exceptions
                        original_call: ToolCall = model_response.tool_call[idx]
                        error_message = (
                            f"[ERROR]: An exception has occurred during tool call '{original_call.name}': {str(ret)}."
                        )
                        pure_tool_return.append(ToolReturn(error_message, original_call))
                    elif isinstance(ret, ToolReturn):
                        pure_tool_return.append(ret)

                # Notify exception handler of tool execution results
                self.__exception_handler(tool_return)
                # Step 4.4: Post-tool-return hook (contexts only)
                await call_coroutine_functions(self.__contexts, self.STATE_ON_TOOL_RETURN, {'tool_return': pure_tool_return, 'agent': self}, self.__exception_handler) # contexts only
                # Update model input for next iteration
                model_input = self.__model.tool_return_integrate(model_input, model_response, pure_tool_return)
            
            elif model_response.response and len(model_response.response) > 0:
                # Step 4.5: Handle final answer (no tool calls)
                await call_coroutine_functions(self.__contexts, self.STATE_ON_FINAL_ANSWER, {'model_response': model_response, 'agent': self}, self.__exception_handler) # contexts only
                break # Terminate loop (final answer generated)
        
        return model_response
    
    async def __aenter__(self):
        """
        Async context manager entry: initialize resources (build).
        
        Returns:
            Self (AgentLoop instance)
        """
        await self.build()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        """
        Async context manager exit: clean up resources (release).
        
        Args:
            exc_type: Type of exception (if any)
            exc: Exception instance (if any)
            tb: Traceback object (if any)
        
        Returns:
            False: Allow exceptions to propagate (standard behavior)
        """
        await self.release()
        return False

if __name__ == '__main__':
    pass
