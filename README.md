# SimpleX - A Modular Python AI Agent SDK

SimpleX is a flexible and extensible Python SDK for building AI agent systems. It provides a modular architecture that allows developers to easily customize and extend every component of an agent system, including models, tools, context plugins, and I/O interfaces.

## Features

- **Modular Architecture**: Clean separation of concerns with pluggable components
- **Tool Collections**: Easily define and manage sets of tools for your agents
- **Context Plugins**: Inject custom behavior and state management into agent loops
- **Model Abstraction**: Support for multiple LLM backends with a unified interface
- **Rich I/O Interfaces**: Built-in terminal interface with rich formatting
- **Lifecycle Hooks**: Comprehensive hooks for customizing agent behavior at every stage
- **Async-First**: Full async support for efficient concurrent operations

## Installation

### Prerequisites

- Python 3.10 or higher
- `uv` package manager (recommended) or `pip`

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd simplex

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

### Dependencies

The project uses the following key dependencies:
- `openai` - OpenAI API client for LLM interactions
- `rich` - Rich text and beautiful formatting in the terminal
- `pyyaml` - YAML parsing for configuration files

## Quick Start

Here's a minimal example to get you started with SimpleX:

```python
import os
import asyncio

from simplex.models import QwenConversationModel
from simplex.context import TrajectoryLogContext
from simplex.loop import AgentLoop, UserLoop
from simplex.tools import SubprocessExecutorLocal
from simplex.io import RichTerminalInterface

async def main():
    # Initialize the conversation model
    model = QwenConversationModel(
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=os.getenv('API_KEY'),
        qwen_model='qwen-plus',
        enable_thinking=False
    )
    
    # Create the terminal interface
    interface = RichTerminalInterface(model.qwen_model)
    
    # Set up the agent loop with tools and context plugins
    loop = AgentLoop(
        model,
        interface.get_exception_handler(),
        TrajectoryLogContext(instance_id='log'),
        SubprocessExecutorLocal()
    )
    
    # Run the user loop
    await UserLoop(
        interface, 
        interface, 
        loop, 
        complete_configs={'max_iteration': 30}
    ).serve()

if __name__ == '__main__':
    asyncio.run(main())
```

## Project Structure

```
simplex/
├── src/
│   └── simplex/
│       ├── basics/          # Core data structures and utilities
│       │   ├── dataclass.py # Basic data classes (ToolCall, ToolReturn, etc.)
│       │   ├── prompt.py    # Prompt template utilities
│       │   ├── client.py    # WebSocket client for external communication
│       │   └── exception.py # Custom exception classes
│       ├── models/          # Model implementations
│       │   ├── base.py      # Abstract base classes for models
│       │   ├── qwen.py      # Qwen model implementation
│       │   └── mock.py      # Mock model for testing
│       ├── tools/           # Tool collections
│       │   ├── base.py      # ToolCollection base class
│       │   ├── edit.py      # File editing tools
│       │   ├── plan.py      # Planning tools
│       │   ├── pyinterpreter.py  # Python interpreter tool
│       │   └── pysublocal.py     # Subprocess execution tool
│       ├── context/         # Context plugins
│       │   ├── base.py      # ContextPlugin base class
│       │   └── tokenc.py    # Token cost counter
│       ├── loop/            # Loop implementations
│       │   ├── base.py      # AgentLoop and UserLoop
│       │   └── adapter.py   # AgentLoopAdapter implementations
│       └── io/              # I/O interfaces
│           ├── base.py      # Abstract I/O interfaces
│           └── terminal.py  # Rich terminal interface
├── unitest/                 # Test files
└── pyproject.toml          # Project configuration
```

## Core Concepts

### Architecture Overview

SimpleX follows a layered architecture where each component has a specific responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                        UserLoop                              │
│  (Manages user interaction and conversation history)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      AgentLoop                               │
│  (Core agent logic with lifecycle management)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Model     │  │ToolCollection│  │   ContextPlugin    │  │
│  │(LLM Backend)│  │  (Tools)    │  │ (State/Behavior)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    I/O Interfaces                            │
│  UserInputInterface  │  UserOutputInterface                  │
│  (Receive input)     │  (Send responses)                     │
└─────────────────────────────────────────────────────────────┘
```

### AgentLoop

The `AgentLoop` is the core component that manages the agent's execution lifecycle. It handles:

- **Model Interaction**: Communicating with the LLM backend
- **Tool Execution**: Dispatching tool calls and collecting results
- **State Management**: Maintaining loop state across iterations
- **Lifecycle Hooks**: Providing extensibility points for custom behavior

#### AgentLoop Lifecycle

The agent loop follows a structured execution flow:

1. **build**: Initialize resources (called once when entering async context)
2. **process_prompt**: Preprocess system and user prompts
3. **start_loop**: Execute before main loop begins
4. **Loop iterations**:
   - **before_response**: Prepare for model generation
   - **Model Generation**: Call the LLM backend
   - **after_response**: Process model response
   - **Tool Execution** (if tool calls present):
     - Execute tools concurrently
     - **after_tool_call**: Process tool results
   - **after_final_response**: Handle final response (if no tool calls)
   - **on_loop_end**: End of iteration cleanup
5. **on_exit**: Final cleanup before loop terminates
6. **release**: Release resources (called when exiting async context)

#### Example: Basic AgentLoop Usage

```python
from simplex.loop import AgentLoop
from simplex.models import QwenConversationModel
from simplex.tools import SubprocessExecutorLocal
from simplex.context import TrajectoryLogContext
from simplex.basics import LogExceptionHandler

async def run_agent():
    model = QwenConversationModel(
        base_url='https://api.example.com/v1',
        api_key='your-api-key',
        qwen_model='model-name'
    )
    
    # Create agent loop with tools and context plugins
    agent = AgentLoop(
        model,
        LogExceptionHandler(instance_id='errors'),
        TrajectoryLogContext(instance_id='trajectory'),
        SubprocessExecutorLocal()
    )
    
    # Use as async context manager
    async with agent:
        result = await agent.complete(
            system="You are a helpful assistant.",
            user="What is the capital of France?",
            max_iteration=10,
            timeout=60
        )
    
    return result
```

### UserLoop

The `UserLoop` wraps the `AgentLoop` to handle user interaction:

- Receives user messages through `UserInputInterface`
- Passes messages to `AgentLoop` for processing
- Maintains conversation history across turns
- Sends responses through `UserOutputInterface`

#### Example: UserLoop with Custom Configurations

```python
from simplex.loop import UserLoop, AgentLoop
from simplex.io import RichTerminalInterface

async def interactive_session():
    interface = RichTerminalInterface('model-name')
    
    agent = AgentLoop(
        model,
        interface.get_exception_handler(),
        # ... tools and context plugins
    )
    
    # Configure UserLoop with custom settings
    user_loop = UserLoop(
        input_interface=interface,
        output_interface=interface,
        agent_loop=agent,
        keep_history=True,  # Maintain conversation history
        complete_configs={
            'max_iteration': 50,
            'timeout': 120,
            'max_retry': 3
        }
    )
    
    await user_loop.serve()
```

### I/O Interfaces

SimpleX provides abstract interfaces for user interaction:

#### UserInputInterface

```python
from simplex.io import UserInputInterface
from simplex.basics import UserMessage, UserNotify, UserResponse

class CustomInputInterface(UserInputInterface):
    async def next_message(self) -> UserMessage:
        """Receive the next message from the user."""
        # Implement custom input logic
        content = await get_user_input()
        return UserMessage(
            user_prompt=content,
            system_prompt=None,
            quit=False
        )
    
    async def notify_user(self, notify: UserNotify) -> UserResponse:
        """Send a notification to the user and get response."""
        # Handle user notifications (e.g., permission requests)
        return UserResponse(response=True)
    
    def get_input_plugin(self):
        """Return an optional ContextPlugin for input handling."""
        return None
```

#### UserOutputInterface

```python
from simplex.io import UserOutputInterface
from simplex.basics import UserNotify

class CustomOutputInterface(UserOutputInterface):
    async def push_message(self, notify: UserNotify):
        """Send output to the user."""
        # Implement custom output logic
        await send_to_user(notify.content)
    
    def get_output_plugin(self):
        """Return an optional ContextPlugin for output handling."""
        return None
```

### AgentLoopAdapter

The `AgentLoopAdapter` abstract class allows you to wrap `AgentLoop` with additional functionality. This is useful for:

- **Parallel Sampling**: Running multiple trajectories simultaneously
- **Custom Execution Patterns**: Implementing specialized loop behaviors
- **Monitoring and Logging**: Adding observability layers

#### Built-in Adapter: ParallelSampleAdapter

```python
from simplex.loop.adapter import ParallelSampleAdapter
from simplex.basics import PromptTemplate

async def parallel_sampling_example():
    # Create base agent loop
    agent = AgentLoop(model, exception_handler, tools...)
    
    # Wrap with parallel sampling adapter
    adapter = ParallelSampleAdapter(agent)
    
    async with adapter:
        # Sample 3 trajectories for each prompt
        results = await adapter.sample_trajectories(
            prompts=[
                PromptTemplate("Explain quantum computing"),
                PromptTemplate("Explain machine learning")
            ],
            num_trajectories=3,
            max_iteration=20
        )
    
    # results is List[List[ModelInput]]
    # Each prompt gets num_trajectories results
    for prompt_idx, trajectory_results in enumerate(results):
        print(f"Prompt {prompt_idx}: {len(trajectory_results)} trajectories")
```
## Customization Guide

### Creating Custom ToolCollections

`ToolCollection` is the base class for defining groups of tools that your agent can use. Each tool collection manages a set of related tools and handles their execution.

#### ToolCollection Base Class

```python
from simplex.tools import ToolCollection
from simplex.basics import ToolCall, ToolReturn, ToolSchema

class MyCustomTools(ToolCollection):
    def __init__(self, instance_id: str = 'my_tools'):
        # Define name mapping: external tool name -> internal method name
        name_mapping = {
            'calculate': '_tool_calculate',
            'lookup': '_tool_lookup'
        }
        super().__init__(instance_id, name_mapping)
        
        # Define tool schemas
        self.schemas = [
            ToolSchema(
                name='calculate',
                description='Perform mathematical calculations',
                params=[
                    ToolSchema.Parameter(
                        field='expression',
                        type='string',
                        description='Mathematical expression to evaluate',
                        required=True
                    )
                ]
            ),
            ToolSchema(
                name='lookup',
                description='Look up information in a database',
                params=[
                    ToolSchema.Parameter(
                        field='query',
                        type='string',
                        description='Search query',
                        required=True
                    )
                ]
            )
        ]
    
    def get_tool_schemas(self) -> list:
        """Return list of tool schemas for this collection."""
        return self.schemas
    
    def tools_descriptions(self) -> str:
        """Return human-readable description of all tools."""
        return "Calculate: Evaluate math expressions.\nLookup: Search database."
    
    # Tool implementation methods (must be async)
    async def _tool_calculate(self, expression: str) -> str:
        """Execute the calculate tool."""
        try:
            result = eval(expression)  # Note: Use safe eval in production!
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    async def _tool_lookup(self, query: str) -> str:
        """Execute the lookup tool."""
        # Implement your lookup logic here
        return f"Results for: {query}"
```

#### Using Schemas from YAML Files

You can define tool schemas in YAML files for better organization:

```yaml
# src/simplex/tools/schema/my_tools.yml
calculate:
  name: calculate
  description: Perform mathematical calculations
  params:
    - field: expression
      type: string
      description: Mathematical expression to evaluate
      required: true

lookup:
  name: lookup
  description: Look up information in a database
  params:
    - field: query
      type: string
      description: Search query
      required: true
```

Then load them in your ToolCollection:

```python
from simplex.tools.base import load_schema, load_tool_definitions

class MyCustomTools(ToolCollection):
    SCHEMA_FILE = 'my_tools'
    
    def __init__(self, instance_id: str = 'my_tools'):
        name_mapping = {
            'calculate': '_tool_calculate',
            'lookup': '_tool_lookup'
        }
        super().__init__(instance_id, name_mapping)
        
        # Load schemas from YAML
        self.calculate_schema = load_schema(self.SCHEMA_FILE, 'calculate')
        self.lookup_schema = load_schema(self.SCHEMA_FILE, 'lookup')
        
    def get_tool_schemas(self) -> list:
        return [self.calculate_schema, self.lookup_schema]
    
    def tools_descriptions(self) -> str:
        return load_tool_definitions(self.SCHEMA_FILE)
```

#### Lifecycle Hooks in ToolCollections

ToolCollections can implement lifecycle hooks to manage resources:

```python
class DatabaseTools(ToolCollection):
    def __init__(self, db_config: dict, instance_id: str = 'db_tools'):
        super().__init__(instance_id, {...})
        self.db_config = db_config
        self.connection = None
    
    async def build(self) -> None:
        """Initialize database connection."""
        self.connection = await create_connection(self.db_config)
    
    async def release(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
    
    async def reset(self) -> None:
        """Reset state for new conversation."""
        # Clear any cached data
        pass
    
    def after_tool_call(self, tool_returns, **kwargs):
        """Process tool results after execution."""
        # Log tool usage, update metrics, etc.
        pass
```

#### Example: Weather Tool Collection

```python
import aiohttp
from simplex.tools import ToolCollection
from simplex.basics import ToolSchema

class WeatherTools(ToolCollection):
    def __init__(self, api_key: str, instance_id: str = 'weather'):
        name_mapping = {
            'get_weather': '_tool_get_weather',
            'get_forecast': '_tool_get_forecast'
        }
        super().__init__(instance_id, name_mapping)
        
        self.api_key = api_key
        self.base_url = "https://api.weatherapi.com/v1"
        
        self.schemas = [
            ToolSchema(
                name='get_weather',
                description='Get current weather for a location',
                params=[
                    ToolSchema.Parameter('location', 'string', 'City name', True)
                ]
            )
        ]
    
    def get_tool_schemas(self) -> list:
        return self.schemas
    
    def tools_descriptions(self) -> str:
        return "Weather tools for current conditions and forecasts."
    
    async def _tool_get_weather(self, location: str) -> str:
        """Fetch current weather data."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/current.json"
            params = {'key': self.api_key, 'q': location}
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                return f"Weather in {location}: {data['current']['temp_c']}°C"
```

### Creating Custom ContextPlugins

`ContextPlugin` allows you to inject custom behavior and state management into the agent loop. Unlike tools, context plugins don't define callable tools but can modify loop state and behavior.

#### ContextPlugin Base Class

```python
from simplex.context import ContextPlugin
from simplex.basics import AgentLoopStateEdit, PromptTemplate

class MyContextPlugin(ContextPlugin):
    def __init__(self, instance_id: str = 'my_context'):
        super().__init__(instance_id)
        self.call_count = 0
    
    def process_prompt(
        self, 
        system_prompt: PromptTemplate, 
        user_prompt: PromptTemplate,
        **kwargs
    ) -> AgentLoopStateEdit:
        """Modify prompts before the loop starts."""
        # Add custom instructions to the system prompt
        enhanced_system = system_prompt + "\n\nAdditional instructions here."
        
        return AgentLoopStateEdit(system_prompt=enhanced_system)
    
    def before_response(self, iter: int, **kwargs):
        """Called before each model response."""
        self.call_count += 1
        print(f"Starting iteration {iter}")
    
    async def after_response_async(self, model_response, **kwargs):
        """Process model response asynchronously."""
        # Log, analyze, or modify the response
        pass
    
    def after_final_response(self, model_input, **kwargs):
        """Called when the agent produces a final answer."""
        print(f"Agent completed after {self.call_count} iterations")
```

#### Built-in Context Plugins

##### TrajectoryLogContext

Records the complete agent trajectory:

```python
from simplex.context import TrajectoryLogContext

log_context = TrajectoryLogContext(
    instance_id='trajectory',
    empty_on_reset=True,  # Clear logs on reset
    line_width=150,       # Formatting width
    delta=True            # Only show changes
)

# After agent execution, access the logs
print(log_context.human_readable)  # Markdown formatted log
print(log_context.dictionary)       # Structured data
```

##### TokenCostCounter

Tracks token usage and costs:

```python
from simplex.context import TokenCostCounter

cost_counter = TokenCostCounter(
    instance_id='costs',
    price_input=0.001,    # $ per 1K input tokens
    price_output=0.002    # $ per 1K output tokens
)

# After execution
print(f"Total cost: ${cost_counter.total_cost}")
print(f"Input tokens: {cost_counter.total_input_tokens}")
```

#### Example: Custom Metrics Plugin

```python
import time
from simplex.context import ContextPlugin
from simplex.basics import AgentLoopStateEdit

class MetricsPlugin(ContextPlugin):
    """Track performance metrics during agent execution."""
    
    def __init__(self, instance_id: str = 'metrics'):
        super().__init__(instance_id)
        self.start_time = None
        self.iteration_times = []
        self.tool_usage = {}
    
    async def start_loop_async(self, **kwargs):
        """Record loop start time."""
        self.start_time = time.time()
        self.iteration_times = []
        self.tool_usage = {}
    
    def before_response(self, iter: int, **kwargs):
        """Record iteration start."""
        self.iteration_times.append({'start': time.time()})
    
    def after_response(self, iter: int, **kwargs):
        """Record iteration end."""
        if iter < len(self.iteration_times):
            self.iteration_times[iter]['end'] = time.time()
    
    def after_tool_call(self, tool_returns, **kwargs):
        """Track tool usage."""
        for ret in tool_returns:
            tool_name = ret.original_call.name
            self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
    
    def on_exit(self, **kwargs):
        """Calculate final metrics."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n=== Metrics ===")
        print(f"Total time: {total_time:.2f}s")
        print(f"Iterations: {len(self.iteration_times)}")
        print(f"Tool usage: {self.tool_usage}")
        
        if self.iteration_times:
            avg_iter = sum(
                t['end'] - t['start'] 
                for t in self.iteration_times 
                if 'end' in t and 'start' in t
            ) / len(self.iteration_times)
            print(f"Avg iteration time: {avg_iter:.2f}s")
```

#### Example: Dynamic Prompt Enhancement

```python
from simplex.context import ContextPlugin
from simplex.basics import AgentLoopStateEdit, PromptTemplate

class SkillInjectorPlugin(ContextPlugin):
    """Dynamically inject skills based on user query."""
    
    def __init__(
        self, 
        skill_library: dict, 
        instance_id: str = 'skill_injector'
    ):
        super().__init__(instance_id)
        self.skill_library = skill_library
        self.injected_skills = set()
    
    def process_prompt(
        self, 
        user_prompt: PromptTemplate, 
        **kwargs
    ) -> AgentLoopStateEdit:
        """Inject relevant skills into the prompt."""
        user_text = str(user_prompt).lower()
        
        # Check which skills are relevant
        relevant_skills = []
        for skill_name, skill_content in self.skill_library.items():
            if skill_name in user_text and skill_name not in self.injected_skills:
                relevant_skills.append(skill_content)
                self.injected_skills.add(skill_name)
        
        if relevant_skills:
            enhanced_prompt = user_prompt + "\n\n" + "\n\n".join(relevant_skills)
            return AgentLoopStateEdit(user_prompt=enhanced_prompt)
        
        return None
    
    async def reset(self):
        """Clear injected skills for new conversation."""
        self.injected_skills.clear()
```
### Creating Custom Models

SimpleX provides a model abstraction layer that allows you to integrate different LLM backends. The `ConversationModel` base class defines the interface for chat-based models.

#### ConversationModel Base Class

```python
from simplex.models import ConversationModel
from simplex.basics import ModelInput, ModelResponse, ToolReturn

class CustomConversationModel(ConversationModel):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        client_configs: dict = None,
        generate_configs: dict = None
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            client_configs=client_configs or {},
            default_generate_configs=generate_configs or {}
        )
        self.model_name = model_name
    
    def clone(self) -> "CustomConversationModel":
        """Create a copy of this model instance."""
        return CustomConversationModel(
            base_url=self._base_url,
            api_key=self._api_key,
            model_name=self.model_name,
            client_configs=self._client_configs,
            generate_configs=self._default_generate_configs
        )
    
    async def generate(self, model_input: ModelInput) -> ModelResponse:
        """
        Generate a response from the model.
        
        This method must be implemented to call the LLM backend.
        """
        # Implement your model API call here
        # Use self.client (OpenAI async client) or implement custom HTTP calls
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=model_input.messages,
            tools=model_input.tools if model_input.tools else None,
            **self._default_generate_configs
        )
        
        # Parse and return the response
        return self._parse_response(response)
    
    async def batch_response(
        self, 
        inputs: List[ModelInput]
    ) -> List[ModelResponse]:
        """Generate responses for multiple inputs concurrently."""
        import asyncio
        return await asyncio.gather(*[self.generate(inp) for inp in inputs])
    
    def tool_return_integrate(
        self,
        input: ModelInput,
        response: ModelResponse,
        tool_return: List[ToolReturn],
        **kwargs
    ) -> ModelInput:
        """
        Integrate tool execution results into the message history.
        
        Called after tools are executed to prepare the next model input.
        """
        messages = list(input.messages)
        
        # Add the assistant's tool call message
        messages.append({
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': call.id,
                    'type': 'function',
                    'function': {
                        'name': call.name,
                        'arguments': call.arguments
                    }
                }
                for call in response.tool_call
            ]
        })
        
        # Add tool results
        for ret in tool_return:
            messages.append({
                'role': 'tool',
                'tool_call_id': ret.original_call.id,
                'content': ret.content
            })
        
        return ModelInput(messages=messages, tools=input.tools)
    
    def final_response_integrate(
        self,
        input: ModelInput,
        response: ModelResponse,
        **kwargs
    ) -> ModelInput:
        """
        Integrate the final response into message history.
        
        Called when the agent produces a final answer (no tool calls).
        """
        messages = list(input.messages)
        messages.append({
            'role': 'assistant',
            'content': response.response
        })
        return ModelInput(messages=messages)
```

#### Example: OpenAI-Compatible Model

```python
from simplex.models.base import ConversationModel, openai_compatiable_translate
from simplex.basics import ModelInput, ModelResponse, ToolCall

class OpenAICompatibleModel(ConversationModel):
    """Model implementation for OpenAI-compatible APIs."""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            generate_configs={
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
        self.model_name = model_name
    
    def clone(self) -> "OpenAICompatibleModel":
        return OpenAICompatibleModel(
            base_url=self._base_url,
            api_key=self._api_key,
            model_name=self.model_name,
            temperature=self._default_generate_configs.get('temperature', 0.7),
            max_tokens=self._default_generate_configs.get('max_tokens', 4096)
        )
    
    async def generate(self, model_input: ModelInput) -> ModelResponse:
        # Convert ModelInput to OpenAI format
        request_dict = openai_compatiable_translate(model_input)
        request_dict['model'] = self.model_name
        
        # Add generate configs
        request_dict.update(self._default_generate_configs)
        
        # Make the API call
        response = await self.client.chat.completions.create(**request_dict)
        
        # Parse response
        choice = response.choices[0]
        
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments
                )
                for tc in choice.message.tool_calls
            ]
        
        return ModelResponse(
            response=choice.message.content,
            tool_call=tool_calls
        )
    
    async def batch_response(
        self, 
        inputs: List[ModelInput]
    ) -> List[ModelResponse]:
        import asyncio
        return await asyncio.gather(*[self.generate(inp) for inp in inputs])
    
    def tool_return_integrate(
        self, 
        input: ModelInput, 
        response: ModelResponse, 
        tool_return: List[ToolReturn],
        **kwargs
    ) -> ModelInput:
        messages = list(input.messages)
        
        # Add assistant message with tool calls
        if response.tool_call:
            messages.append({
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': tc.id,
                        'type': 'function',
                        'function': {
                            'name': tc.name,
                            'arguments': tc.arguments
                        }
                    }
                    for tc in response.tool_call
                ]
            })
        
        # Add tool results
        for ret in tool_return:
            messages.append({
                'role': 'tool',
                'tool_call_id': ret.original_call.id,
                'content': ret.content
            })
        
        return ModelInput(messages=messages, tools=input.tools)
    
    def final_response_integrate(
        self, 
        input: ModelInput, 
        response: ModelResponse,
        **kwargs
    ) -> ModelInput:
        messages = list(input.messages)
        messages.append({
            'role': 'assistant',
            'content': response.response
        })
        return ModelInput(messages=messages)
```

#### Using the Mock Model for Testing

```python
from simplex.models import MockConversationModel
from simplex.basics import ModelResponse, ToolCall

# Create mock model with predefined responses
mock_model = MockConversationModel(
    expected_responses=[
        ModelResponse(tool_call=[
            ToolCall('call_1', 'search', {'query': 'python asyncio'})
        ]),
        ModelResponse(response="Based on the search results, asyncio is..."),
        ModelResponse(tool_call=[
            ToolCall('call_2', 'make_plan', {'content': 'Plan: ...'})
        ]),
        ModelResponse(response="Here's the final answer.")
    ]
)

# Use in tests
async def test_agent():
    agent = AgentLoop(mock_model, exception_handler, tools...)
    async with agent:
        result = await agent.complete(user="Explain asyncio")
```

## Advanced Topics

### Accessing AgentLoop Instances

You can access registered tools and context plugins using dictionary-style access:

```python
# Create agent loop with named instances
agent = AgentLoop(
    model,
    exception_handler,
    TrajectoryLogContext(instance_id='trajectory'),
    SubprocessExecutorLocal(),
    MetricsPlugin(instance_id='metrics')
)

# Access instances by their key
trajectory_log = agent['trajectory']
metrics = agent['metrics']

# Use instance data after execution
async with agent:
    await agent.complete(user="...")

print(agent['trajectory'].human_readable)
print(agent['metrics'].iteration_times)
```

### Modifying Loop State with AgentLoopStateEdit

Context plugins and tools can modify the loop state by returning `AgentLoopStateEdit`:

```python
from simplex.basics import AgentLoopStateEdit

class StateModifierPlugin(ContextPlugin):
    def after_response(
        self,
        model_response,
        exit_flag,
        **kwargs
    ) -> AgentLoopStateEdit:
        """Modify loop state based on response."""
        
        # Force exit after certain condition
        if len(model_response.response or '') > 1000:
            return AgentLoopStateEdit(exit_flag=True)
        
        return None
```

Available state modifications:
- `system_prompt`: Modify the system prompt
- `user_prompt`: Modify the user prompt
- `model_input`: Replace the model input entirely
- `model_response`: Modify the model response
- `tool_returns`: Modify tool execution results
- `exit_flag`: Set to `True` to exit the loop early

### Creating Custom AgentLoopAdapter

Implement your own adapter for specialized execution patterns:

```python
from simplex.loop.base import AgentLoopAdapter
from simplex.basics import PromptTemplate, ModelInput

class RetryAdapter(AgentLoopAdapter):
    """Adapter that retries on failed responses."""
    
    def __init__(self, agent_loop: AgentLoop, max_retries: int = 3):
        self.agent_loop = agent_loop
        self.max_retries = max_retries
    
    async def build(self) -> None:
        await self.agent_loop.build()
    
    async def release(self) -> None:
        await self.agent_loop.release()
    
    async def reset(self) -> None:
        await self.agent_loop.reset()
    
    def clone(self) -> "RetryAdapter":
        return RetryAdapter(self.agent_loop.clone(), self.max_retries)
    
    async def bind_io(self, input_interface, output_interface) -> None:
        await self.agent_loop.bind_io(input_interface, output_interface)
    
    async def complete(
        self,
        system: PromptTemplate = None,
        user: PromptTemplate = None,
        history: list = None,
        **kwargs
    ) -> ModelInput:
        """Execute with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await self.agent_loop.complete(
                    system=system,
                    user=user,
                    history=history,
                    **kwargs
                )
                # Check if result is valid
                if result.messages and result.messages[-1].get('content'):
                    return result
            except Exception as e:
                last_error = e
            
            # Reset for retry
            await self.agent_loop.reset()
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")
```

## Future Features and Roadmap

### AgentLoopAdapter Extensions

The `AgentLoopAdapter` pattern is designed to support advanced multi-agent scenarios:

```python
# Planned: Hierarchical Agent Architecture
class HierarchicalAdapter(AgentLoopAdapter):
    """
    Adapter for hierarchical agent systems.
    
    A supervisor agent delegates tasks to specialized worker agents.
    """
    
    def __init__(
        self, 
        supervisor: AgentLoop,
        workers: Dict[str, AgentLoop]
    ):
        self.supervisor = supervisor
        self.workers = workers
    
    async def complete(self, user, **kwargs) -> ModelInput:
        # Supervisor analyzes and delegates
        supervisor_result = await self.supervisor.complete(user=user, **kwargs)
        
        # Extract delegation decisions
        delegations = self._parse_delegations(supervisor_result)
        
        # Execute worker agents
        worker_results = await asyncio.gather(*[
            self.workers[w].complete(user=task)
            for w, task in delegations
        ])
        
        # Aggregate results
        return await self.supervisor.complete(
            user=self._aggregate_results(worker_results)
        )
```

### Pipe-like I/O Interface for Multi-Agent Systems

A planned feature for composing agents in a pipeline fashion:

```python
# Planned: Pipe Interface
class PipeInterface:
    """
    Enables connecting multiple agents in a pipeline.
    
    Output from one agent becomes input to the next.
    """
    
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
    
    async def send(self, message):
        await self.output_queue.put(message)
    
    async def receive(self):
        return await self.input_queue.get()

# Usage: Agent composition
agent1 = AgentLoop(model1, ...)
agent2 = AgentLoop(model2, ...)

# Connect agents
pipe1 = PipeInterface()
pipe2 = PipeInterface()

# Agent1 writes to pipe1, reads from input
# Agent2 reads from pipe1, writes to pipe2
```

### Better Skill Management

Planned improvements for skill definition and injection:

```python
# Planned: SkillRegistry
class SkillRegistry:
    """
    Centralized management of agent skills.
    
    Skills are modular capabilities that can be:
    - Dynamically loaded based on context
    - Shared across agents
    - Versioned and updated
    """
    
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
    
    def register(self, name: str, skill: Skill):
        """Register a new skill."""
        self.skills[name] = skill
    
    def get_relevant_skills(self, context: str) -> List[Skill]:
        """Retrieve skills relevant to the current context."""
        return [
            skill for name, skill in self.skills.items()
            if skill.is_relevant(context)
        ]
    
    def to_prompt_section(self, skills: List[Skill]) -> str:
        """Convert skills to a prompt section."""
        return "\n\n".join(skill.to_prompt() for skill in skills)

# Usage
registry = SkillRegistry()
registry.register("web_search", WebSearchSkill())
registry.register("code_interpreter", CodeInterpreterSkill())

# Auto-inject relevant skills
class SkillInjectionPlugin(ContextPlugin):
    def __init__(self, registry: SkillRegistry):
        self.registry = registry
    
    def process_prompt(self, user_prompt, **kwargs):
        skills = self.registry.get_relevant_skills(str(user_prompt))
        if skills:
            skill_prompt = self.registry.to_prompt_section(skills)
            return AgentLoopStateEdit(
                user_prompt=user_prompt + f"\n\n{skill_prompt}"
            )
```

### Planned: Streaming Response Support

```python
# Planned: Streaming support
class StreamingModel(ConversationModel):
    async def generate_stream(
        self, 
        model_input: ModelInput
    ) -> AsyncIterator[str]:
        """Stream response chunks instead of returning complete response."""
        async for chunk in self.client.chat.completions.create(
            model=self.model_name,
            messages=model_input.messages,
            stream=True
        ):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### Planned: Memory and State Management

```python
# Planned: Memory systems
class ConversationMemory:
    """
    Persistent memory across conversations.
    
    Supports:
    - Short-term: Recent messages
    - Long-term: Important facts
    - Semantic: Vector-based retrieval
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.short_term: List[Dict] = []
        self.long_term: List[Dict] = []
    
    async def add_message(self, message: Dict):
        """Add message to memory."""
        self.short_term.append(message)
        await self._maybe_consolidate()
    
    async def get_relevant(self, query: str) -> List[Dict]:
        """Retrieve relevant memories."""
        # Semantic search through stored memories
        pass
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd simplex

# Install development dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff format .
uv run ruff check .
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
