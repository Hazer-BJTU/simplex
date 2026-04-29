# Simplex — Python LLM-Based Code Agent SDK

**Simplex** is a lightweight, extensible Python framework for building AI-powered code agents. It provides a modular architecture where you compose AI models, tools, context plugins, and I/O interfaces into a cohesive agent loop — all driven by a structured lifecycle with synchronous and asynchronous hooks.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [The Agent Loop Lifecycle](#the-agent-loop-lifecycle)
  - [Models](#models)
  - [Tools](#tools)
  - [Context Plugins](#context-plugins)
  - [I/O Interfaces](#io-interfaces)
  - [Data Classes](#data-classes)
- [API Reference](#api-reference)
  - [simplex.models](#simplexmodels)
  - [simplex.loop](#simplexloop)
  - [simplex.tools](#simplextools)
  - [simplex.context](#simplexcontext)
  - [simplex.io](#simplexio)
  - [simplex.basics](#simplexbasics)
  - [simplex.agents](#simplexagents)
- [The `simplex_tool_server`](#the-simplex_tool_server)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)
  - [Parallel Trajectory Sampling](#parallel-trajectory-sampling)
  - [Rehearsal Mode](#rehearsal-mode)
  - [Building Custom Tools](#building-custom-tools)
  - [Building Custom Context Plugins](#building-custom-context-plugins)
- [Logging & Trajectory Output](#logging--trajectory-output)
- [Configuration Reference](#configuration-reference)

---

## Overview

Simplex is designed around a simple but powerful idea: an AI agent is just a **loop** that calls an LLM, executes tools, and manages context. The framework formalizes this loop with:

- **Pluggable AI models** — DeepSeek, Qwen, or any OpenAI-compatible API
- **8 built-in file editing tools** — view, search, edit, undo via WebSocket
- **Context management** — token counting, context window clipping, trajectory logging
- **Rich terminal UI** — spinners, syntax-highlighted blocks, permission prompts
- **Lifecycle hooks** — 22 hook points for plugins to intercept every phase of the loop

### Design Philosophy

- **Composition over inheritance**: Build agents by combining `ToolCollection` and `ContextPlugin` instances with an `AgentLoop`.
- **Explicit lifecycle**: Every component participates in a well-defined lifecycle (`build` → `start_loop` → `before_response` → … → `release`).
- **Async-first**: All I/O and model calls are asynchronous, with thread-pool support for parallel sampling.
- **User-in-the-loop**: Tools can request user permission before executing potentially dangerous operations.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     UserLoop                         │
│  Orchestrates multi-turn conversation with history   │
├─────────────────────────────────────────────────────┤
│                AgentLoopAdapter                      │
│  (AgentLoop itself, or ParallelSampleAdapter)        │
├─────────────────────────────────────────────────────┤
│                    AgentLoop                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Model   │  │  Tools   │  │ Context Plugins  │   │
│  │(generate)│  │(execute) │  │(lifecycle hooks) │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────┤
│                  I/O Interfaces                      │
│  UserInputInterface  │  UserOutputInterface           │
│  (RichTerminalInterface combines both)               │
└─────────────────────────────────────────────────────┘
```

**Data flow**: User message → `UserLoop` → `AgentLoop.complete()` → model generates → tool calls executed → response integrated → loop continues until final answer.

---

## Installation

### Prerequisites

- Python ≥ 3.11
- `uv` (recommended) or `pip`

### Install from source

```bash
git clone <repository-url>
cd simplex
uv sync
```

### Dependencies

Core dependencies include `openai`, `rich`, `websockets`, `pyyaml`, `numpy`, and `rank-bm25`. See `pyproject.toml` for the full list.

### The `simplex_tool_server`

The file-editing tools (`EditTools`) require an external WebSocket server process. Build it from the `cpptools` directory (C++ code) and ensure the `simplex_tool_server` binary is on your `PATH`.

```bash
simplex_tool_server -p 9002 -c 20
```

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `-p, --port` | Port number (required) | — |
| `-j, --jobs` | Async worker processes | 1 |
| `-n, --head-n` | Lines for file preview | 200 |
| `-s, --history` | Undo history entries per file | 15 |
| `-c, --concurrent` | Threads for concurrent search | 4 |
| `-m, --max-result` | Max response bytes | 24576 |

---

## Quick Start

Below is a complete working example adapted from `examples/simple_code_agent_loop.py`:

```python
import os
import asyncio
from pathlib import Path
from simplex.basics import WebsocketClient, CommandProcess
from simplex.models import DeepSeekConversationModel
from simplex.context import TrajectoryLogContext, TokenCostCounter, RollContextClipper
from simplex.loop import AgentLoop, UserLoop
from simplex.tools import EditTools, SubprocessExecutorLocal, SequentialPlan, InLoopConversation
from simplex.io import RichTerminalInterface

EDIT_TOOL_SERVER_PORT = 9002

async def main(workspace_dir: Path):
    # 1. Create the AI model
    model = DeepSeekConversationModel(
        base_url='https://api.deepseek.com/beta',
        api_key=os.getenv('API_KEY'),
        model='deepseek-v4-pro',
        default_generate_configs={'temperature': 0.4},
        enable_thinking=True
    )

    # 2. Create a rich terminal interface
    interface = RichTerminalInterface(model.model)

    # 3. Build the agent loop with tools & contexts
    loop = AgentLoop(
        model,
        interface.get_exception_handler(),
        TrajectoryLogContext(instance_id='log'),
        EditTools(
            base_dir=workspace_dir,
            client=WebsocketClient(EDIT_TOOL_SERVER_PORT, 'localhost'),
            permission_required=True,
            add_skill=True
        ),
        SubprocessExecutorLocal(permission_required=True),
        SequentialPlan(add_skill=True),
        RollContextClipper(
            max_context_tokens=256000,
            threshold_ratio=0.65,
            keep_fc_msgs=120
        ),
        TokenCostCounter(),
        InLoopConversation()
    )

    # 4. Start the user interaction loop
    await UserLoop(
        input_interface=interface,
        output_interface=interface,
        agent_loop=loop,
        complete_configs={
            'max_iteration': 100,
            'timeout': 600,
            'max_retry': 5
        }
    ).serve()

if __name__ == '__main__':
    with CommandProcess(f"simplex_tool_server -p {EDIT_TOOL_SERVER_PORT} -c 20") as proc:
        asyncio.run(main(Path.cwd()))
```

Run it:

```bash
export API_KEY="your-deepseek-api-key"
python examples/simple_code_agent_loop.py --workspace /path/to/your/project
```

---

## Core Concepts

### The Agent Loop Lifecycle

Every `AgentLoop` execution follows a precise sequence of lifecycle hooks. Both `ToolCollection` and `ContextPlugin` instances can implement any of these hooks to intercept and modify the loop's behavior.

```
build()                            — Initialize resources (called once)
  └─ bind_io()                     — Bind user I/O interfaces
       └─ complete()               — One full task execution
            ├─ process_prompt()    — Modify prompts before loop
            ├─ start_loop_async()  — Async pre-loop setup
            ├─ start_loop()        — Sync pre-loop setup
            └─ For each iteration:
                 ├─ before_response_async()
                 ├─ before_response()
                 ├─ model.generate()        ← LLM call
                 ├─ after_response_async()
                 ├─ after_response()
                 ├─ [if tool calls]
                 │    ├─ tool execution      ← Concurrent tool dispatch
                 │    ├─ after_tool_call_async()
                 │    └─ after_tool_call()
                 ├─ [if final response]
                 │    ├─ after_final_response_async()
                 │    └─ after_final_response()
                 ├─ on_loop_end_async()
                 └─ on_loop_end()
            ├─ on_exit_async()
            └─ on_exit()
release()                            — Clean up resources
```

Hooks return `AgentLoopStateEdit` to modify loop state (prompts, model input, response, tool returns, exit flag).

### Models

Models are the "brain" of the agent. All models extend `BaseModel` → `ConversationModel` and implement:

| Method | Description |
|--------|-------------|
| `generate(model_input)` | Send input to LLM, return `ModelResponse` |
| `tool_return_integrate(input, response, returns)` | Merge tool results into message history |
| `final_response_integrate(input, response)` | Merge final text response into message history |
| `build()` / `release()` / `reset()` / `clone()` | Lifecycle management |

**Built-in models:**

- `DeepSeekConversationModel` — DeepSeek API with chain-of-thought reasoning support
- `QwenConversationModel` — Qwen (Alibaba Cloud) API with streaming + thinking mode
- `MockConversationModel` — For testing; returns canned responses

### Tools

Tools are callable capabilities exposed to the LLM via function-calling schemas. Each tool collection extends `ToolCollection` and provides:

- `get_tool_schemas()` — Returns `List[ToolSchema]` describing available functions
- `_tool_<name>()` — Internal method that implements the tool logic (dispatched by name)
- Optional lifecycle hooks (e.g., `process_prompt()` to inject skill instructions)

**Tool execution flow:** LLM response contains `ToolCall` objects → `AgentLoop` dispatches each call to the matching `ToolCollection` → returns `ToolReturn` → results integrated back into conversation.

### Context Plugins

Context plugins observe and modify the agent loop. They extend `ContextPlugin` and implement lifecycle hooks. Unlike tools, they are not exposed to the LLM — they work behind the scenes.

### I/O Interfaces

- **`UserInputInterface`** — Gets user messages (`next_message()`) and handles permission requests (`notify_user()`)
- **`UserOutputInterface`** — Pushes agent responses and notifications to the user (`push_message()`)

`RichTerminalInterface` implements both, providing a rich terminal UI with spinners, syntax highlighting, and formatted panels.

### Data Classes

| Class | Purpose |
|-------|---------|
| `ModelInput` | Carries `messages`, `tools`, `model` name to the LLM |
| `ModelResponse` | Holds `response` text, `tool_call` list, `reasoning_content`, `token_cost` |
| `ToolCall` | A single function call: `id`, `name`, `arguments` |
| `ToolReturn` | Result of a tool execution: `content`, references original `ToolCall` |
| `ToolSchema` | Describes a tool: `name`, `description`, list of `Parameter` objects |
| `PromptTemplate` | Buildable markdown prompt with title/block/simple helpers |
| `LoopInformation` | Snapshot of one loop iteration: input, response, tool returns |

---

## API Reference

### `simplex.models`

#### `BaseModel`

Abstract base for all models. Not used directly.

```python
class BaseModel(ABC):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        client_configs: Dict,
        default_generate_configs: Dict,
        instance_id: str,
        disable_openai_backend: bool = False
    )
```

| Property | Type | Description |
|----------|------|-------------|
| `key` | `str` | Unique instance identifier |
| `client` | `AsyncOpenAI \| None` | OpenAI async client |

#### `ConversationModel(BaseModel)`

Adds conversation-specific methods. Subclassed by all chat models.

```python
@abstractmethod
async def generate(self, model_input: ModelInput) -> ModelResponse

@abstractmethod
def tool_return_integrate(
    self, input: ModelInput, response: ModelResponse,
    tool_return: List[ToolReturn], **kwargs
) -> ModelInput

@abstractmethod
def final_response_integrate(
    self, input: ModelInput, response: ModelResponse, **kwargs
) -> ModelInput
```

#### `DeepSeekConversationModel`

```python
DeepSeekConversationModel(
    base_url: str,              # e.g. 'https://api.deepseek.com/beta'
    api_key: str,               # API key (load from env)
    client_configs: Optional[Dict] = None,
    default_generate_configs: Optional[Dict] = None,  # e.g. {'temperature': 0.4}
    instance_id: Optional[str] = None,
    model: str = 'deepseek-reasoner',
    enable_thinking: bool = True   # Enable chain-of-thought reasoning
)
```

#### `QwenConversationModel`

```python
QwenConversationModel(
    base_url: str,
    api_key: str,
    client_configs: Optional[Dict] = None,
    default_generate_configs: Optional[Dict] = None,
    instance_id: Optional[str] = None,
    qwen_model: str = 'qwen-coder-plus',
    enable_thinking: bool = True,
    thinking_budget: int = 1024
)
```

Uses streaming responses and reconstructs tool calls from streamed chunks.

---

### `simplex.loop`

#### `AgentLoop(AgentLoopAdapter)`

The core agent execution engine.

```python
AgentLoop(
    model: ConversationModel,
    exception_handler: ExceptionHandler,
    *args: ToolCollection | ContextPlugin   # Tools and context plugins
)
```

| Method | Description |
|--------|-------------|
| `add_instance(*args)` | Register additional tools/contexts (before `build()`) |
| `__getitem__(key)` | Access registered instance by `instance_id` (e.g. `loop['log']`) |
| `complete(system, user, history, ...)` | Execute one full task completion cycle |
| `build()` / `release()` / `reset()` / `clone()` | Lifecycle management |
| `bind_io(input_interface, output_interface)` | Bind user I/O and register I/O plugins |

**`complete()` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system` | `Optional[PromptTemplate]` | `None` | System prompt |
| `user` | `Optional[PromptTemplate]` | `None` | User prompt for this task |
| `history` | `Optional[List[Dict]]` | `None` | OpenAI-format message history |
| `max_iteration` | `int` | `50` | Max tool-calling rounds |
| `timeout` | `float` | `600` | Seconds timeout per model request |
| `max_retry` | `int` | `5` | Max retries for failed model requests |
| `keep_original_system` | `bool` | `False` | Preserve system prompt from history |
| `rehearsal_mode` | `bool` | `False` | Replay a previous trajectory |
| `rehearsal_list` | `Optional[List[LoopInformation]]` | `None` | Trajectory to replay |

#### `UserLoop`

Orchestrates multi-turn conversations between user and agent.

```python
UserLoop(
    input_interface: UserInputInterface,
    output_interface: UserOutputInterface,
    agent_loop: AgentLoopAdapter,
    keep_history: bool = True,
    complete_configs: Optional[Dict] = None
)
```

| Method | Description |
|--------|-------------|
| `serve()` | Start the interactive conversation loop |

`complete_configs` is forwarded to `AgentLoop.complete()` as keyword arguments.

#### `AgentLoopAdapter(ABC)`

Abstract base for wrapping `AgentLoop` with additional logic. `AgentLoop` itself is an `AgentLoopAdapter`. The `ParallelSampleAdapter` (in `simplex.loop.adapter`) wraps an `AgentLoop` for parallel trajectory sampling.

#### `ParallelSampleAdapter`

```python
ParallelSampleAdapter(agent_loop: AgentLoop)

async def sample_trajectories(
    prompts: Union[PromptTemplate, List[PromptTemplate]],
    num_trajectories: int,
    system: Optional[PromptTemplate] = None,
    history: Optional[List[dict]] = None,
    max_iteration: int = 30,
    timeout: float = 120,
    max_retry: int = 5,
    keep_original_system: bool = False
) -> List[List[ModelResponse]]
```

Runs multiple `AgentLoop` clones in parallel using a `ThreadPoolExecutor`. Returns one list of responses per prompt.

---

### `simplex.tools`

#### `ToolCollection(ABC)`

Base class for all tool collections.

```python
ToolCollection(instance_id: str, name_mapping: Dict)
```

`name_mapping` maps external tool names (exposed to LLM) to internal method names (e.g. `{"view_file_content": "_tool_view_file_content"}`).

| Method | Description |
|--------|-------------|
| `get_tool_schemas() -> List[ToolSchema]` | Return tool definitions for the LLM |
| `tools_descriptions() -> str` | Human-readable tool descriptions |
| `__call__(tool_call: ToolCall) -> ToolReturn` | Dispatch a tool call internally |

Also supports all lifecycle hooks: `build`, `release`, `reset`, `clone`, `bind_io`, `process_prompt`, `start_loop`, `start_loop_async`, etc.

#### `EditTools(ToolCollection)`

Provides 8 file-editing operations via WebSocket to `simplex_tool_server`:

| Operation | Description |
|-----------|-------------|
| `view_workspace` | Get current workspace file tree |
| `show_details` | Show directory/file details |
| `view_file_content` | Read file content with line range |
| `edit_file_content` | Replace or insert text in files |
| `str_replace_edit` | String-based find-and-replace |
| `undo` | Undo previous edit |
| `search` | Search by identifier, definition, or pattern |
| `operate_filesystem` | Create, remove, or rename files |

```python
EditTools(
    base_dir: str | Path,             # Agent's working directory
    client: WebsocketClient,          # Connection to simplex_tool_server
    permission_required: bool = True, # Require user approval for operations
    instance_id: Optional[str] = None,
    rename_mapping: Dict[EditOperation, str] = {...},  # Customize tool names
    add_skill: bool = True            # Auto-inject skill instructions into prompt
)
```

#### `SubprocessExecutorLocal(ToolCollection)`

Executes bash commands locally with optional user permission.

```python
SubprocessExecutorLocal(
    rename: str = 'bash',
    permission_required: bool = True,
    instance_id: Optional[str] = None
)
```

#### `SequentialPlan(ToolCollection)`

Provides the `make_plan` tool for the LLM to create, update, and review structured task plans.

```python
SequentialPlan(
    rename: str = 'make_plan',
    empty_on_reset: bool = True,
    instance_id: Optional[str] = None,
    add_skill: bool = True
)
```

The plan tool supports three `edit_type` modes: `replace`, `append`, and `check_only`.

#### `InLoopConversation(ToolCollection)`

Provides the `propose` tool — lets the agent pause and ask the user a question mid-execution.

```python
InLoopConversation(
    rename: str = 'propose',
    instance_id: Optional[str] = None,
    add_skill: bool = True
)
```

#### `PythonInterpreter(ToolCollection)`

Executes Python code snippets, locally or in a Docker container.

```python
PythonInterpreter(
    instance_id: Optional[str] = None,
    rename: str = 'python_interpreter',
    use_container: bool = False,
    container_manager: Optional[ContainerManager] = None,
    exec_command: Callable[[str], List[str]] = lambda script: ['python', '-c', script],
    timeout: float = 10
)
```

#### `MockCalculator(ToolCollection)`

A simple mock tool for testing. Provides basic arithmetic operations without external dependencies.

---

### `simplex.context`

#### `ContextPlugin(ABC)`

Base class for all context plugins. Implements the same lifecycle hooks as `ToolCollection` but is not exposed to the LLM.

```python
ContextPlugin(instance_id: str)
```

#### `TrajectoryLogContext(ContextPlugin)`

Logs the complete agent trajectory — system prompt, user prompt, each iteration's model response, tool calls, and tool returns.

```python
TrajectoryLogContext(
    instance_id: Optional[str] = None,
    empty_on_reset: bool = True,
    line_width: int = 150,
    delta: bool = True
)
```

| Property | Type | Description |
|----------|------|-------------|
| `log` | `List[LoopInformation]` | Raw structured log of all iterations |
| `markdown` | `PromptTemplate` | Human-readable markdown representation |
| `detailed` | `List[LoopInformation]` | Alias for `log` (used in example for pickle) |
| `human_readable` | `str` | String form of `markdown` |

#### `TokenCostCounter(ContextPlugin)`

Tracks token consumption across all model interactions.

```python
TokenCostCounter(
    instance_id: Optional[str] = None,
    empty_on_reset: bool = True
)
```

| Property | Type | Description |
|----------|------|-------------|
| `token_total` | `int` | Cumulative tokens consumed |
| `token_max` | `int` | Peak tokens in a single response |
| `input_token_total` | `int` | Cumulative prompt tokens |
| `output_token_total` | `int` | Cumulative completion tokens |
| `iterations` | `int` | Number of completed model calls |
| `token_cost` | `List[Dict]` | Formatted token statistics |
| `token_cost_formatted` | `List[Dict]` | Human-readable (e.g. `1.2k`, `3.5m`) |

#### `RollContextClipper(ContextPlugin)`

Prevents context window overflow by clipping old function-calling messages when token usage exceeds a threshold.

```python
RollContextClipper(
    instance_id: Optional[str] = None,
    max_context_tokens: int = 128000,
    threshold_ratio: float = 0.65,
    keep_fc_msgs: int = 50,
    identify_function: Callable = identify_openai_function_calling
)
```

**How it works:** After each iteration (`on_loop_end`), if `current_max_tokens > threshold_ratio * max_context_tokens`, it iteratively removes function-calling rounds (assistant tool_call + corresponding tool responses) while preserving at most `keep_fc_msgs` messages.

#### `ActionSelfEvaluation(ContextPlugin)`

Injects self-evaluation parameters (`task_status` and `action_quality`) into every tool schema, encouraging the LLM to reflect on its actions.

```python
ActionSelfEvaluation(
    instance_id: Optional[str] = None,
    add_skill: bool = True
)
```

---

### `simplex.io`

#### `UserInputInterface(ABC)`

```python
@abstractmethod
async def next_message(self) -> UserMessage

@abstractmethod
async def notify_user(self, notify: UserNotify) -> UserResponse

@abstractmethod
def get_input_plugin(self) -> Optional[ContextPlugin]
```

#### `UserOutputInterface(ABC)`

```python
@abstractmethod
async def push_message(self, notify: UserNotify) -> Any

@abstractmethod
def get_output_plugin(self) -> Optional[ContextPlugin]
```

#### `RichTerminalInterface`

Implements both `UserInputInterface` and `UserOutputInterface` with a rich terminal UI.

```python
RichTerminalInterface(
    model_name: str = 'unknown',
    style: str = 'default',
    base_system_prompt: Optional[PromptTemplate] = None,
    skill_retriever: Optional[SkillRetriever] = None
)
```

| Method | Description |
|--------|-------------|
| `get_exception_handler()` | Returns `RichTerminalExceptionHandler` for styled error display |
| `get_input_plugin()` | Returns `RichTerminalInputPlugin` (handles `/skills` command) |
| `get_output_plugin()` | Returns `RichTerminalOutputPlugin` (spinners, formatted output) |

---

### `simplex.basics`

#### `PromptTemplate`

A builder for structured markdown prompts.

```python
PromptTemplate(content: str = '')

# Methods (all return self for chaining):
.add_main_title(title)         # "# Title"
.add_sub_title(title)          # "## Title"
.add_simple(text, title)       # Plain text with optional heading
.add_block(text, title, block) # Fenced code block (e.g. block='yaml')
```

Supports `+`, `+=`, `str()`, and `repr()`.

#### `SkillRetriever`

BM25-based skill search for injecting relevant skill instructions into prompts.

```python
SkillRetriever(top_k: int = 5, path: Path = SKILLS_PATH)

search(query: str, top_k: Optional[int] = None) -> List[Dict]
get_more(top_k: int) -> List[Dict]
get_system_prompt(path: Optional[Path] = None) -> PromptTemplate
```

#### `WebsocketClient`

Async WebSocket client for communicating with `simplex_tool_server`.

```python
WebsocketClient(
    port: int,
    host: str = 'localhost',
    max_queue_size: int = 0,
    max_retry: int = 5,
    await_timeout: float = 1
)
```

| Method | Description |
|--------|-------------|
| `build()` | Start the WebSocket connection task |
| `release()` | Stop the connection |
| `exchange(data)` | Send JSON, wait for response (correlates by UUID) |

Supports `async with` context manager.

#### `WebsocketClientSync`

Synchronous variant using threading. Same API but blocking.

#### `CommandProcess`

Context manager for launching and managing a subprocess.

```python
CommandProcess(cmd: List[str] | str, shell: bool = True)

with CommandProcess("simplex_tool_server -p 9002") as proc:
    ...  # Server runs in background, terminated on exit
```

Automatically kills the process group on context exit (SIGTERM → SIGKILL escalation).

#### `ContainerManager`

Docker container management utility (used by `PythonInterpreter` when `use_container=True`).

#### Exception Types

All in `simplex.basics.exception`:

| Exception | Purpose |
|-----------|---------|
| `EntityInitializationError` | Model/tool/context init failure |
| `RequestError` | LLM API request failure |
| `ParameterError` | Invalid parameter (with function name, param, type hint) |
| `ImplementationError` | Method implementation error |
| `EnvironmentError` | Environment setup/teardown failure |
| `UnbuiltError` | Method called before `build()` |
| `ConflictError` | Duplicate instance key or tool name |
| `MaxRetriesExceeded` | All retry attempts exhausted |
| `Notice` | Non-fatal notification (logged but not raised) |

#### `ExceptionHandler` / `LogExceptionHandler`

```python
LogExceptionHandler(
    instance_id: Optional[str] = None,
    content: str = '',
    file: Any = sys.stderr
)
```

Logs exceptions with timestamps. `RichTerminalExceptionHandler` extends this for styled terminal output.

#### `async_retry_timeout`

Utility decorator for retry-with-timeout logic.

```python
async_retry_timeout(
    max_retry: int,
    timeout: float,
    retry_exceptions: tuple = (asyncio.TimeoutError, RequestError),
    on_retry: Callable = None
)
```

Used internally by `AgentLoop.complete()` for resilient model generation.

---

### `simplex.agents`

Convenience functions for creating common agent configurations.

#### `get_standard_coder_agent()`

```python
get_standard_coder_agent(
    model: ConversationModel,
    work_dir: str | Path,
    exception_handler: Optional[ExceptionHandler] = None,
    edit_tools_port: int = 9002,
    edit_tools_host: str = 'localhost',
    log: str = 'log',
    token_counter: str = 'token_counter'
) -> AgentLoopAdapter
```

Returns a pre-configured `AgentLoop` with `EditTools`, `TrajectoryLogContext`, and `TokenCostCounter`. A minimal starting point that you can extend.

---

## The `simplex_tool_server`

The `EditTools` module communicates with an external C++ WebSocket server that performs all file system operations. This separation ensures that file operations are sandboxed and auditable.

**Starting the server** (from the example):

```python
from simplex.basics import CommandProcess

with CommandProcess(f"simplex_tool_server -p 9002 -c 20") as proc:
    asyncio.run(main(...))
```

The server manages:
- Working directory scoping (set via `set_working_dir` message)
- File content viewing with line range
- Directory structure enumeration
- Text editing (replace, insert)
- String-based find-and-replace
- Undo history per file
- Pattern/identifier/definition search
- File system operations (create, remove, rename)
- Workspace refresh

---

## Error Handling

The framework uses a hierarchical exception system:

```
Exception
 └─ CustomException
      ├─ EntityInitializationError
      ├─ RequestError
      ├─ ParameterError
      ├─ ImplementationError
      ├─ EnvironmentError
      ├─ UnbuiltError
      ├─ ConflictError
      ├─ MaxRetriesExceeded
      └─ Notice
```

**In the AgentLoop:** The `ExceptionHandler` receives results from lifecycle hooks and tool executions. Non-fatal issues (like a plugin missing a hook) produce `Notice` exceptions. Fatal errors (like model failures after max retries) raise `RuntimeError`.

**In tools:** `UnbuiltError` is raised if a tool method is called before `build()`. `RequestError` wraps external service failures.

---

## Advanced Usage

### Parallel Trajectory Sampling

Sample multiple response trajectories for the same prompt:

```python
from simplex.loop.adapter import ParallelSampleAdapter

adapter = ParallelSampleAdapter(agent_loop)
results = await adapter.sample_trajectories(
    prompts=PromptTemplate("Write a Python function to..."),
    num_trajectories=5,
    system=system_prompt,
    max_iteration=50,
    timeout=300
)
# results: [[ModelResponse, ...]] — 5 trajectories
```

Each trajectory runs in its own thread with a cloned `AgentLoop`.

### Rehearsal Mode

Replay a previously logged trajectory exactly:

```python
log: TrajectoryLogContext = loop['log']
# ... first run ...

# Replay the same trajectory
await loop.complete(
    system=...,
    user=...,
    rehearsal_mode=True,
    rehearsal_list=log.log  # List[LoopInformation] from previous run
)
```

In rehearsal mode, model responses are taken from the recorded trajectory instead of calling the LLM. Tool calls are still executed.

### Building Custom Tools

```python
from simplex.tools import ToolCollection, load_schema, load_tool_definitions

class MyCustomTool(ToolCollection):
    SCHEMA_FILE = 'schema_my_tool'  # YAML file in simplex/tools/schema/

    def __init__(self, instance_id=None):
        super().__init__(
            instance_id or uuid.uuid4().hex,
            {'my_tool': '_tool_my_tool'}
        )
        self.schema = load_schema(self.SCHEMA_FILE, 'my_tool', 'my_tool')

    def get_tool_schemas(self):
        return [self.schema]

    async def _tool_my_tool(self, param1: str, param2: int, **kwargs) -> str:
        # Your tool logic here
        return f"Result: {param1} x {param2}"

# Register with AgentLoop
loop = AgentLoop(model, handler, MyCustomTool(), ...)
```

Tool schemas are defined as YAML files in `src/simplex/tools/schema/`.

### Building Custom Context Plugins

```python
from simplex.context import ContextPlugin
from simplex.basics import AgentLoopStateEdit

class MyPlugin(ContextPlugin):
    def __init__(self, instance_id=None):
        super().__init__(instance_id or uuid.uuid4().hex)

    async def before_response_async(self, model_input, **kwargs):
        # Modify model input before each LLM call
        print(f"About to call LLM with {len(model_input.messages)} messages")

    def on_loop_end(self, **kwargs):
        # Potentially modify loop state
        return AgentLoopStateEdit(exit_flag=should_stop_early())
```

---

## Logging & Trajectory Output

After a session, you can export the trajectory in multiple formats:

```python
log: TrajectoryLogContext = loop['log']

# Structured data (pickle)
with open('trajectory.pkl', 'wb') as f:
    pickle.dump(log.detailed, f)

# Human-readable markdown
with open('trajectory.md', 'w') as f:
    f.write(log.human_readable)
```

The markdown includes:
- Initial system and user prompts
- Per-iteration reasoning content, tool calls (with arguments), model responses
- Tool return values
- Token cost summary (via `TokenCostCounter`)

---

## Configuration Reference

### AgentLoop Constructor Parameters

| Component | Key Parameters |
|-----------|---------------|
| `DeepSeekConversationModel` | `base_url`, `api_key`, `model`, `temperature`, `enable_thinking` |
| `EditTools` | `base_dir`, `client` (WebsocketClient), `permission_required`, `add_skill` |
| `SubprocessExecutorLocal` | `permission_required` |
| `SequentialPlan` | `add_skill` |
| `RollContextClipper` | `max_context_tokens` (256000), `threshold_ratio` (0.65), `keep_fc_msgs` (120) |
| `TokenCostCounter` | (no required params) |
| `InLoopConversation` | `add_skill` |
| `TrajectoryLogContext` | `instance_id`, `line_width` |

### UserLoop / complete() Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iteration` | 100 | Max tool-calling rounds per task |
| `timeout` | 600 | Seconds per model request |
| `max_retry` | 5 | Retries for failed requests |
| `keep_history` | `True` | Maintain conversation history across turns |

---

## License

See [LICENSE](LICENSE) file for details.
