import os
import uuid
import copy
import inspect
import asyncio

from abc import ABC, abstractmethod
from typing import Literal, Coroutine, List, Dict, Optional, Any

import simplex.basics
import simplex.context
import simplex.models
import simplex.tools
import simplex.io

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
    MaxRetriesExceeded,
    Notice,
    UserNotify,
    UnbuiltError,
    LoopInformation,
    AgentLoopStateEdit,
    UserMessage,
    ExceptionHandler,
    async_retry_timeout
)
from simplex.context import ContextPlugin
from simplex.models import ConversationModel
from simplex.tools import ToolCollection
from simplex.io import UserInputInterface, UserOutputInterface


# ------------------------------ #
# Adapter definitions            #
# ------------------------------ #
class AgentLoopAdapter(ABC):
    """
    The AgentLoopAdapter class is the base class of AgentLoop, which defines a series of interfaces 
    for adapting to UserLoop. In fact, this means that AgentLoop itself can be regarded as its own Adapter. 
    The role of the Adapter is to add sampling logic to AgentLoop. 

    The UserLoop class can accept an AgentLoopAdapter (e.g., the ParallelSampleAdapter that wraps a raw AgentLoop, 
    which is used to implement diverse agent trajectory sampling logic) as the core logic for handling agent loops.
    
    Also see:
        class AgentLoop
    """
    
    def __init__(self) -> None:
        super().__init__()
    
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
    def clone(self) -> "AgentLoopAdapter":
        pass

    @abstractmethod
    async def bind_io(self, input_interface: UserInputInterface, output_interface: UserOutputInterface) -> None:
        pass

    @abstractmethod
    async def complete(
        self,
        system: Optional[PromptTemplate] = None,
        user: Optional[PromptTemplate] = None,
        history: Optional[List[Dict]] = None, # standard openai message list
        **kwargs
    ) -> ModelInput:
        pass

    async def __aenter__(self):
        await self.build()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.release()
        return False

# ------------------------------ #
# UserLoop definitions           #
# ------------------------------ #
class UserLoop:
    """
    Manages the user interaction loop for an AI agent conversation session.
    
    This class orchestrates the high-level conversation flow between a user and an AI agent.
    It handles receiving user messages through an input interface, delegating processing
    to an AgentLoopAdapter, and optionally maintaining conversation history across turns.
    
    The UserLoop serves as the outermost control layer, wrapping the agent's internal
    processing loop (AgentLoop/AgentLoopAdapter) with user-facing I/O management.
    
    Attributes:
        __input_interface: Interface for receiving user messages (UserInputInterface)
        __output_interface: Interface for sending responses to user (UserOutputInterface)
        __agent_loop: The underlying agent loop adapter handling model interactions
        __keep_history: Whether to preserve conversation history between turns
        __complete_configs: Additional configuration parameters passed to complete()
    
    Example:
        ```python
        async with UserLoop(input_if, output_if, agent_loop) as loop:
            await loop.serve()
        ```
    
    Also see:
        class AgentLoop
        class AgentLoopAdapter
    """

    def __init__(
        self,
        input_interface: UserInputInterface,
        output_interface: UserOutputInterface,
        agent_loop: AgentLoopAdapter,
        keep_history: bool = True,
        complete_configs: Optional[Dict] = None
    ) -> None:
        self.__input_interface = input_interface
        self.__output_interface = output_interface
        self.__agent_loop = agent_loop
        self.__keep_history = keep_history

        self.__complete_configs: Dict = complete_configs if complete_configs is not None else {}

    async def _do_serve(self, loop: AgentLoopAdapter) -> None:
        # Initialize history list
        history: List[Dict] = []
        while True:
            # Get next user message
            user_message: UserMessage = await self.__input_interface.next_message()
            
            if user_message.quit:
                break
            
            # Do complete
            try:
                finalized_input: ModelInput = await loop.complete(
                    system = user_message.system_prompt,
                    user = user_message.user_prompt,
                    history = history,
                    **self.__complete_configs
                )
            except Exception as e:
                await self.__output_interface.push_message(UserNotify('notify', title = 'Warning', content = f"User loop quit due to exception: {e}"))
                break

            # Assign message list to history if 'keep_history'
            if self.__keep_history and finalized_input.messages:
                history = copy.deepcopy(finalized_input.messages)

    async def serve(self) -> None:
        # Bind i/o interface BEFORE 'build()'
        await self.__agent_loop.bind_io(self.__input_interface, self.__output_interface)
        async with self.__agent_loop as loop: # 'build()' is called here
            await self._do_serve(loop)

# ------------------------------ #
# AgentLoop definitions          #
# ------------------------------ #
"""Literal type defining all valid loop action names for AgentLoop lifecycle events"""
AgentLoopAction = Literal[
    'build',                      # Initialize resources (async)
    'release',                    # Clean up resources (async)
    'reset',                      # Reset loop state (async)
    'clone',                      # Create copy of instance (sync)
    'bind_io',                    # Bind input/output interfaces (async)
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
    'on_loop_end_async',          # Run at end of iteration (async)
    'on_exit',                    # Run at end of whole loop (sync)
    'on_exit_async'               # Run at end of whole loop (async)
]

class AgentLoop(AgentLoopAdapter):
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
        super().__init__()

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
        self.__exit_flag: bool = False

        # Rehearsal mode parameters
        self.__perform_rehearsal: bool = False
        self.__rehearsal_list: List[LoopInformation] = []
        self.__rehearsal_iter: int = 0

        # Object state managemnt
        self.__initialized: bool = False

        try:
            self.add_instance(*args)
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)
        
    def add_instance(self, *args: ToolCollection | ContextPlugin) -> None:
        """
        Register instances and build tool-mapping based on schemas.

        Args:
            *args: Variable list of ToolCollection/ContextPlugin instances to register

        Returns:
            None
        """

        if self.__initialized:
            raise RuntimeError(f"{self.__class__.__name__}: unable to add instances after \'build\' is called")
        
        # Register all provided tool/context instances
        for instance in args:
            if instance.key in self.__instances:
                raise ConflictError(f"duplicated instance key: {instance.key}")
            self.__instances[instance.key] = instance

            if isinstance(instance, ToolCollection):
                self.__tools.append(instance)
                schemas = instance.get_tool_schemas()
                for schema in schemas:
                    if schema.name in self.__tool_mapping:
                        raise ConflictError(f"duplicated tool name \'{schema.name}\' from collection: {repr(instance)}")
                    self.__tool_mapping[schema.name] = instance
                self.__tool_schemas.extend(schemas)

            elif isinstance(instance, ContextPlugin):
                self.__contexts.append(instance)
            
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
    
    def _call_sequential(self, name: AgentLoopAction, params: Optional[Dict] = None) -> List[Any]:
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

        results: List[Any] = []
        for instance in self._instance_list:
            captured: Dict[str, Any] = self._captured_states
            
            if not hasattr(instance, name):
                results.append(Notice(f"{repr(instance)} doesn't have attribute: {name}"))
                continue

            target = getattr(instance, name)
            
            if not callable(target):
                results.append(Notice(f"{repr(instance)} doesn't have a method named: {name}"))
                continue

            try:
                if params is not None:
                    result = target(**params)
                else:
                    result = target(**copy.deepcopy(captured))

                if isinstance(result, AgentLoopStateEdit):
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
                    if result.exit_flag:
                        self.__exit_flag = result.exit_flag
            except Exception as e:
                result = e
            results.append(result)

        # Pass results to exception handler
        self.__exception_handler(results)

        return results
    
    async def _call_async(self, name: AgentLoopAction, params: Optional[Dict] = None) -> List[Any]:
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
            
            if params is not None:
                tasks.append(target(**params))
            else:
                tasks.append(target(**copy.deepcopy(captured)))

        # Execute all async tasks concurrently (return exceptions instead of raising)
        results: List[Any] = await asyncio.gather(*tasks, return_exceptions = True)
        # Pass results to exception handler
        self.__exception_handler(results)
        return results
    
    async def build(self) -> None:
        # Trigger async build lifecycle hook for all instances
        await self._call_async('build', params = {})
        self.__initialized = True

    async def release(self) -> None:
        # Trigger async release lifecycle hook for all instances
        await self._call_async('release', params = {})
        self.__initialized = False

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
        self.__exit_flag = False
        await self._call_async('reset', params = {})

    def clone(self) -> "AgentLoop":
        """
        Create a deep copy of the AgentLoop instance
        
        Clones the model, exception handler, and all registered instances via
        their clone methods, then initializes a new AgentLoop with these clones.
        
        Returns:
            New AgentLoop instance with cloned dependencies
        """

        if not self.__initialized:
            raise UnbuiltError(self.__class__.__name__)

        return AgentLoop(
            self.__model.clone(),
            self.__exception_handler.clone(),
            *self._call_sequential('clone', params = {})
        )
    
    async def bind_io(
        self,
        input_interface: UserInputInterface,
        output_interface: UserOutputInterface
    ) -> None:
        """
        Bind user input and output interfaces to the loop
        
        This method allows dynamic assignment of I/O interfaces, which can be used
        by plugins/tools during lifecycle hooks to interact with users.
        
        Args:
            input_interface: UserInputInterface instance for receiving input
            output_interface: UserOutputInterface instance for sending output
            
        Returns:
            None
        """

        if self.__initialized:
            raise RuntimeError(f"{self.__class__.__name__}: unable to add instances after \'build\' is called")

        await self._call_async('bind_io', params = {
            'input_interface': input_interface,
            'output_interface': output_interface
        })
        input_plugin = input_interface.get_input_plugin()
        output_plugin = output_interface.get_output_plugin()
        if input_plugin:
            self.add_instance(input_plugin)
        if output_plugin:
            self.add_instance(output_plugin)
    
    async def complete(
        self,
        system: Optional[PromptTemplate] = None,
        user: Optional[PromptTemplate] = None,
        history: Optional[List[Dict]] = None, # standard openai message list
        max_iteration: int = 50,
        timeout: float = 600,
        max_retry: int = 5,
        keep_original_system: bool = False,
        rehearsal_mode: bool = False,
        rehearsal_list: Optional[List[LoopInformation]] = None, 
        **kwargs
    ) -> ModelInput:
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
            rehearsal_mode: Perform an established sequence of actions again.
            rehearsal_list: The established action sequence for rehearsal_mode.
            
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
        
        if not self.__initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        # Initialize prompt templates
        self.__system_prompt = system if system is not None else PromptTemplate()
        self.__user_prompt = user if user is not None else PromptTemplate()

        # Initialize rehearsal settings
        if rehearsal_mode and rehearsal_list:
            self.__perform_rehearsal = True
            self.__rehearsal_list = copy.deepcopy(rehearsal_list)
            self.__rehearsal_iter = 0
        
        # Default to empty history if None provided
        if history is None:
            used_history = []
        else:
            used_history = copy.deepcopy(history)

        # Process system prompt from history
        if len(used_history) > 0 and used_history[0].get('role', '') == 'system':
            if keep_original_system:
                # Append new system prompt to original if requested
                self.__system_prompt = used_history[0].get('content', '') + self.__system_prompt
            # Remove original system prompt to avoid duplication
            used_history.pop(0)

        # Preprocess prompts through lifecycle hook
        self._call_sequential('process_prompt')

        # Rebuild message history with processed prompts
        used_history.insert(0, {'role': 'system', 'content': str(self.__system_prompt)})
        # Append new user prompt for multi-turn chat completion
        used_history.append({'role': 'user', 'content': str(self.__user_prompt)})

        # Prepare model input with messages and tool schemas
        self.__model_input = ModelInput(messages = used_history, tools = self.__tool_schemas)

        # Execute pre-loop lifecycle hooks (async first to avoid state modification)
        await self._call_async('start_loop_async')
        self._call_sequential('start_loop')

        # Main iteration loop
        for curr_iter in range(max_iteration):
            self.__iter = curr_iter
            
            # Pre-response lifecycle hooks
            await self._call_async('before_response_async')
            self._call_sequential('before_response')
            
            response_already_got: bool = False

            if self.__perform_rehearsal:
                rehearsal_model_output: Optional[ModelResponse] = None
                # Find the first `loop_info` object where the `model_response` field is not None.
                while self.__rehearsal_iter < len(self.__rehearsal_list):
                    loop_info = self.__rehearsal_list[self.__rehearsal_iter]
                    if loop_info.model_response:
                        rehearsal_model_output = copy.deepcopy(loop_info.model_response)
                        self.__rehearsal_iter += 1
                        break
                    else:
                        self.__rehearsal_iter += 1
                
                if rehearsal_model_output:
                    self.__model_response = rehearsal_model_output
                    response_already_got = True
                else:
                    self.__perform_rehearsal = False

            if not response_already_got: # If the `model_response` has already been obtained, skip the backend request process.
                # Model generation with retry logic
                try:
                    if hasattr(self.__model.generate, 'retry_already_handled'):
                        self.__model_response = await self.__model.generate(model_input = self.__model_input)
                    else:
                        self.__model_response = await async_retry_timeout(
                            max_retry, 
                            timeout, 
                            retry_exceptions = (asyncio.TimeoutError, RequestError),
                            on_retry = lambda e, attempt: self.__exception_handler(Notice(f"model endpoint retry attempt [{attempt + 1}/{max_retry}]: {e}"))
                        )(self.__model.generate)(model_input = self.__model_input)
                except MaxRetriesExceeded:
                    e = RuntimeError(f"failed to receive from model endpoint after {max_retry} attempts")
                    self.__exception_handler(e, Notice(f"loop quit due to maximum retry exceeded"))
                    raise e
                except Exception as e:
                    self.__exception_handler(e, Notice(f"loop quit due to unexpected error"))
                    raise e
            
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

                # Update model input for output
                self.__model_input = self.__model.final_response_integrate(self.__model_input, self.__model_response)
                break # Terminate loop (final answer generated)
            
            # End-of-iteration lifecycle hooks
            await self._call_async('on_loop_end_async')
            self._call_sequential('on_loop_end')

            # Check for early exit flag
            if self.__exit_flag:
                break
        
        await self._call_async('on_exit_async')
        self._call_sequential('on_exit')
        # Return final model response
        return self.__model_input

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
