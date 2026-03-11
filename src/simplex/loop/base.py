import os
import sys
import uuid
import copy
import inspect
import asyncio

from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

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
    RuntimeError,
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
class AgentLoop:
    def __init__(
        self,
        agent_model: ConversationModel,
        *args: Any
    ) -> None:
        self.agent_model = agent_model
        self.tools_list: List[ToolCollection] = []
        self.context_list: List[ContextPlugin] = []
        self.instance_dict: Dict[str, Any] = {}

        self.iter: int = 0

        try:
            for instance in args:
                if isinstance(instance, ToolCollection):
                    self.tools_list.append(instance)
                    if instance.key in self.instance_dict:
                        raise ConflictError(f'duplicated instance key: {instance.key}')
                    self.instance_dict[instance.key] = instance
                elif isinstance(instance, ContextPlugin):
                    self.context_list.append(instance)
                    if instance.key in self.instance_dict:
                        raise ConflictError(f'duplicated instance key: {instance.key}')
                    self.instance_dict[instance.key] = instance
                else:
                    pass
            
            self.tool_mapping: Dict[str, ToolCollection] = {}
            self.tool_schemas: List[ToolSchema] = []

            for tool in self.tools_list:
                self.tool_schemas.extend(tool.get_tools())
                names = tool.get_names()
                for name in names:
                    if name in self.tool_mapping:
                        raise ConflictError(f'duplicated tool name: {name}')
                    self.tool_mapping[name] = tool
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)

    def __getitem__(self, instance_key: str) -> Any:
        return self.instance_dict.get(instance_key, None)
    
    @property
    def _all_instances(self) -> List[ToolCollection | ContextPlugin]:
        return self.tools_list + self.context_list

    async def build(self) -> None:
        try:
            await asyncio.gather(*call_coroutine_functions(self._all_instances, 'build'))
        except Exception:
            raise

    async def release(self) -> None:
        try:
            await asyncio.gather(*call_coroutine_functions(self._all_instances, 'release'))
        except Exception:
            raise
    
    async def reset(self) -> None:
        try:
            await asyncio.gather(*call_coroutine_functions(self._all_instances, 'reset'))
        except Exception:
            raise
    
    async def procedure(self, max_iteration: int = 30) -> None:
        async def tool_not_exists(original_call: ToolCall):
            return ToolReturn(
                content = f'[ERROR]: Tool {original_call.name} not exists. Please try again.',
                original_call = original_call
            )
        
        async def tool_exception(original_call: ToolCall, e: Exception):
            return ToolReturn(
                content = f'[ERROR]: An exception has occurred during tool call: {e}. Please try again.',
                original_call = original_call
            )
        
        initial_prompt = ModelInput(messages = [], tools = self.tool_schemas)

        call_functions(self.context_list, 'on_start_procedure', agent = self)
        call_functions(self.context_list, 'on_process_prompt', model_input = initial_prompt, agent = self)
        call_functions(self.tools_list, 'on_init_output', model_input = initial_prompt, agent = self)
        call_functions(self.context_list, 'on_prompt_ready', model_input = initial_prompt, agent = self)

        input: ModelInput = copy.deepcopy(initial_prompt)
        output: Optional[ModelResponse] = None
        for self.iter in range(max_iteration):
            output = await self.agent_model.generate(input)
            call_functions(self.context_list, 'on_model_response', model_response = output, agent = self)
            
            if output.tool_call is not None and len(output.tool_call) > 0:
                tool_call_tasks: List = []
                for tool_call in output.tool_call:
                    if tool_call.name in self.tool_mapping:
                        try:
                            dispatched = self.tool_mapping[tool_call.name](tool_call)
                        except Exception as e:
                            dispatched = tool_exception(tool_call, e)
                        tool_call_tasks.append(dispatched)
                    else:
                        tool_call_tasks.append(tool_not_exists(tool_call))
                tool_returns = await asyncio.gather(*tool_call_tasks, return_exceptions = True)
                tool_returns_no_exception: List[ToolReturn] = []
                for idx, tool_return in enumerate(tool_returns):
                    if isinstance(tool_return, TypeError):
                        original_call: ToolCall = output.tool_call[idx]
                        error_message: str = f"[ERROR]: Parameter error of tool call \'{original_call.name}\'. " \
                                             f"Please double-check the parameter requirements. {tool_return}"
                        tool_returns_no_exception.append(ToolReturn(error_message, original_call))
                    elif isinstance(tool_return, Exception):
                        original_call: ToolCall = output.tool_call[idx]
                        error_message: str = f"[ERROR]: An exception has occurred during tool call \'{original_call.name}\': {tool_return}."
                        tool_returns_no_exception.append(ToolReturn(error_message, original_call))
                    else:
                        tool_returns_no_exception.append(tool_return) #type: ignore
                call_functions(self.context_list, 'on_tool_return', tool_return = tool_returns_no_exception, agent = self)
                input = self.agent_model.tool_return_integrate(input, output, tool_returns_no_exception)

            if output.response != '':
                call_functions(self.context_list, 'on_final_answer', model_response = output, agent = self)
                if output.tool_call is None or len(output.tool_call) == 0:
                    break

    async def __aenter__(self):
        await self.build()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.release()
        return False
'''

if __name__ == '__main__':
    pass
