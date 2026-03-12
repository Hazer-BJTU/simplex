import os
import sys
import uuid
import copy
import inspect
import asyncio

from datetime import datetime
from abc import ABC, abstractmethod
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

LoopAction = Literal[
    'build',
    'release',
    'reset',
    'clone',
    'process_prompt',
    'start_loop',
    'start_loop_async',
    'before_response',
    'before_response_async',
    'after_response',
    'after_response_async',
    'after_tool_call',
    'after_tool_call_async',
    'after_final_response',
    'after_final_response_async',
    'on_loop_end',
    'on_loop_end_async'
]

class AgentLoop:
    def __init__(
        self,
        model: ConversationModel,
        exception_handler: ExceptionHandler = LogExceptionHandler(),
        *args: ToolCollection | ContextPlugin
    ) -> None:
        self.__model = model
        self.__exception_handler = exception_handler

        self.__tools: List[ToolCollection] = []
        self.__contexts: List[ContextPlugin] = []
        self.__instances: Dict[str, ToolCollection | ContextPlugin] = {}
        self.__tool_mapping: Dict[str, ToolCollection] = {}
        self.__tool_schemas: List[ToolSchema] = []

        self.__iter: int = 0
        self.__system_prompt: PromptTemplate = PromptTemplate()
        self.__user_prompt: PromptTemplate = PromptTemplate()
        self.__model_input: ModelInput = ModelInput()
        self.__model_response: ModelResponse = ModelResponse()
        self.__tool_returns: List[ToolReturn] = []
        self.__exit_flag: List[bool] = [False]

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
                schemas = tool.get_tools()
                for schema in schemas:
                    if schema.name in self.__tool_mapping:
                        raise ConflictError(f"duplicated tool name \'{schema.name}\' from collection: {repr(tool)}")
                    self.__tool_mapping[schema.name] = tool
                self.__tool_schemas.extend(schemas)
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)

    def __getitem__(self, instance_key: str) -> ToolCollection | ContextPlugin:
        return self.__instances[instance_key]

    @property
    def _instance_list(self) -> List[ToolCollection | ContextPlugin]:
        return list(self.__instances.values())
    
    def _call_sequential(self, name: LoopAction, *args, **kwargs) -> List[Any]:
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
                result = target(*args, **kwargs)
            except Exception as e:
                result = e
            results.append(result)

        self.__exception_handler(results)
        return results
    
    async def _call_async(self, name: LoopAction, *args, **kwargs) -> List[Any]:
        async def _return_exception(exception: Exception) -> Exception:
            return exception
        
        tasks: List[Coroutine[Any, Any, Any]] = []
        for instance in self._instance_list:
            if not hasattr(instance, name):
                tasks.append(_return_exception(Notice(f"{repr(instance)} doesn't have attribute: {name}")))
                continue

            target = getattr(instance, name)

            if not callable(target) or not inspect.iscoroutinefunction(target):
                tasks.append(_return_exception(Notice(f"{repr(instance)} doesn't have a coroutine function named: {name}")))
                continue

            tasks.append(target(*args, **kwargs))

        results: List[Any] = await asyncio.gather(*tasks, return_exceptions = True)
        self.__exception_handler(results)
        return results
    
    async def build(self) -> None:
        await self._call_async('build')

    async def release(self) -> None:
        await self._call_async('release')

    async def reset(self) -> None:
        self.__iter = 0
        self.__system_prompt = PromptTemplate()
        self.__user_prompt = PromptTemplate()
        self.__model_input = ModelInput()
        self.__model_response = ModelResponse()
        self.__tool_returns = []
        self.__exit_flag = [False]
        await self._call_async('reset')

    def clone(self) -> "AgentLoop":
        return AgentLoop(
            self.__model.clone(),
            self.__exception_handler.clone(),
            *self._call_sequential('clone')
        )
    
    async def complete(
        self,
        system: PromptTemplate = PromptTemplate(),
        user: PromptTemplate = PromptTemplate(),
        history: Optional[List[Dict]] = None, # standard openai message list
        max_iteration: int = 30,
        timeout: float = 120,
        max_retry: int = 5,
        keep_original_system: bool = False
    ) -> ModelResponse:
        def _capture() -> Dict:
            return {
                'iter': self.__iter, # read only
                'system_prompt': self.__system_prompt,
                'user_prompt': self.__user_prompt,
                'model_input': self.__model_input,
                'model_response': self.__model_response,
                'tool_returns': self.__tool_returns,
                'exit_flag': self.__exit_flag
            }
        
        async def tool_not_exists(original_call: ToolCall):
            return ToolReturn(content = f'[ERROR]: Tool {original_call.name} not exists. Please try again.', original_call = original_call)
        
        self.__system_prompt = system
        self.__user_prompt = user
        
        if history is None:
            history = []

        if len(history) > 0 and history[0].get('role', '') == 'system':
            if keep_original_system:
                self.__system_prompt = history[0].get('content', '') + self.__system_prompt
            history.pop(0) # remove original system prompt

        self._call_sequential('process_prompt', **_capture())

        history.insert(0, {'role': 'system', 'content': str(self.__system_prompt)}) # reinsert processed system prompt
        history.append({'role': 'user', 'content': str(self.__user_prompt)}) # append new user prompt for multi-turn chat completion

        self.__model_input = ModelInput(messages = history, tools = self.__tool_schemas)

        await self._call_async('start_loop_async', **_capture())  # async always comes first, should not modify captured attributes!
        self._call_sequential('start_loop', **_capture())         # modify attributes during sequential call

        for curr_iter in range(max_iteration):
            self.__iter = curr_iter

            await self._call_async('before_response_async', **_capture())
            self._call_sequential('before_response', **_capture())

            for attempt in range(max_retry + 1):
                try:
                    self.__model_response = await asyncio.wait_for(self.__model.generate(model_input = self.__model_input), timeout = timeout)
                except (asyncio.TimeoutError, RequestError) as e:
                    self.__exception_handler(e, Notice(f"model endpoint retry attempt [{attempt}/{max_retry}]"))
                    if attempt == max_retry:
                        raise RuntimeError(f"failed to receive from model endpoint after {max_retry} attempts")
                    await asyncio.sleep(0.5)
                    continue
                except Exception as e:
                    self.__exception_handler(e, Notice(f"loop quit due to unexpected error"))
                    raise e # fatal error, quit
                break
            
            await self._call_async('after_response_async', **_capture())
            self._call_sequential('after_response', **_capture())

            if self.__model_response.tool_call is not None and len(self.__model_response.tool_call) > 0:
                tasks: List = [Coroutine[Any, Any, ToolReturn]]
                for call in self.__model_response.tool_call:
                    if call.name in self.__tool_mapping:
                        tasks.append(self.__tool_mapping[call.name](call))
                    else:
                        tasks.append(tool_not_exists(call))

                tool_return_with_exceptions = await asyncio.gather(*tasks, return_exceptions = True)
                self.__tool_returns = []
                for idx, ret in enumerate(tool_return_with_exceptions):
                    if isinstance(ret, TypeError):
                        # May be arguments mismatch
                        original_call: ToolCall = self.__model_response.tool_call[idx]
                        error_message = (
                            f"[ERROR]: Parameter error of tool call '{original_call.name}'. "
                            f"Please double-check the parameter requirements. {str(ret)}"
                        )
                        self.__tool_returns.append(ToolReturn(error_message, original_call))
                    elif isinstance(ret, Exception):
                        # Other exceptions
                        original_call: ToolCall = self.__model_response.tool_call[idx]
                        error_message = (
                            f"[ERROR]: An exception has occurred during tool call '{original_call.name}': {str(ret)}."
                        )
                        self.__tool_returns.append(ToolReturn(error_message, original_call))
                    elif isinstance(ret, ToolReturn):
                        self.__tool_returns.append(ret)

                # Notify exception handler of tool execution results
                self.__exception_handler(tool_return_with_exceptions)

                await self._call_async('after_tool_call_async', **_capture())
                self._call_sequential('after_tool_call', **_capture())

                # Update model input for next iteration
                self.__model_input = self.__model.tool_return_integrate(self.__model_input, self.__model_response, self.__tool_returns)

            elif self.__model_response.response is not None and len(self.__model_response.response) > 0:
                await self._call_async('after_final_response_async', **_capture())
                self._call_sequential('after_final_response', **_capture())
                break # Terminate loop (final answer generated)

            await self._call_async('on_loop_end_async', **_capture())
            self._call_sequential('on_loop_end', **_capture())

            if self.__exit_flag[0]:
                break

        return self.__model_response

    async def __aenter__(self):
        await self.build()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.release()
        return False

if __name__ == '__main__':
    pass
