#!/usr/bin/env python3
"""
Simple Code Agent Loop Example
A minimal implementation of an AI agent loop using the Simplex SDK,
integrated with code editing tools, subprocess execution, and a CLI interface.
"""
import os
import pickle
import pathlib
import asyncio
import argparse

from pathlib import Path

# Core Simplex SDK imports
import simplex.basics
import simplex.models
import simplex.context
import simplex.loop
import simplex.tools
import simplex.io

# Key component imports (explicit for clarity)
from simplex.basics import WebsocketClient, CommandProcess
from simplex.models import DeepSeekConversationModel
from simplex.context import TrajectoryLogContext, TokenCostCounter, RollContextClipper
from simplex.loop import AgentLoop, UserLoop
from simplex.tools import EditTools, SubprocessExecutorLocal, SequentialPlan
from simplex.io import RichTerminalInterface

# --------------------------
# Configuration Constants
# --------------------------
# Port for the simplex_tool_server (code editing service)
EDIT_TOOL_SERVER_PORT: int = 9002
OUTPUT_PATH: Path = Path('output')

async def main(base_dir: Path | None = None) -> None:
    """
    Main asynchronous entry point for the agent application.
    Initializes the AI model, user interface, agent loop, and starts the service.

    Args:
        base_dir: Optional workspace directory for file operations.
                  If not provided, the current working directory is used.
    """
    # Determine workspace directory (defaults to current working directory)
    workspace_dir = base_dir.absolute() if base_dir is not None else Path.cwd()
    # 1. Initialize the DeepSeek AI conversation model
    model = DeepSeekConversationModel(
        # Model API endpoint
        base_url = 'https://api.deepseek.com/beta',
        # API key loaded from environment variable
        api_key = os.getenv('API_KEY'),
        # Specific DeepSeek model to use
        model = 'deepseek-reasoner',
        # Default generation parameters for LLM responses
        default_generate_configs = {
            'temperature': 0.4  # Lower = more deterministic, higher = more creative
        },
        # Enable chain-of-thought reasoning for the model
        enable_thinking = True
    )

    # 2. Initialize rich CLI interface for user input/output
    # Provides formatted terminal interaction with the agent
    interface = RichTerminalInterface(model.model)

    # 3. Build the core AgentLoop with all required components
    loop = AgentLoop(
        # AI language model instance
        model,
        # Exception handler from the user interface (show errors through CLI)
        interface.get_exception_handler(),
        # Context: Logs agent interaction trajectories to files
        TrajectoryLogContext(instance_id = 'log'),
        # Code editing tool with workspace and WebSocket connection
        EditTools(
            base_dir = workspace_dir,  # Agent's working directory for file operations
            client = WebsocketClient(EDIT_TOOL_SERVER_PORT, 'localhost'),
            permission_required = True,       # Require user approval for unsafe file operations
            add_skill = True                  # Register tool capabilities with the LLM
        ),
        # Local shell command executor (with user permission control)
        SubprocessExecutorLocal(permission_required = True),
        # Sequential planning tool for multi-step task execution
        SequentialPlan(add_skill = True),
        # Context window management to prevent token overflow
        RollContextClipper(
            max_context_tokens = 128000,    # Model's maximum context window size
            threshold_ratio = 0.65,         # Trigger clipping at 65% of max context usage
            keep_fc_msgs = 60               # Preserve up to 60 function call messages after clipping
        ),
        # Tracks token consumption and notifies the user
        TokenCostCounter()
    )

    # 4. Start the user-agent interaction loop
    # Manages back-and-forth conversation between user and agent
    await UserLoop(
        input_interface = interface,   # Use CLI for user input
        output_interface = interface,  # Use CLI for agent output
        agent_loop = loop,             # Bind the configured agent loop
        # Runtime execution constraints
        complete_configs = {
            'max_iteration': 100,  # Max 100 tool-calling rounds per task
            'timeout': 600,        # Max 10 minutes (600s) for model response
            'max_retry': 5         # Max 5 retries for failed model requests
        }
    ).serve()

    log: TrajectoryLogContext = loop['log'] # Get log by 'instance_id'
    pickle.dump(log.detailed, OUTPUT_PATH / 'simple_code_agent_loop.pkl')
    with open(OUTPUT_PATH / 'simple_code_agent_loop.md', 'w', encoding = 'utf8') as file:
        file.write(log.human_readable)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Simple Code Agent Loop with configurable workspace')
    parser.add_argument('--workspace', type = str, default = None, help = 'Workspace base directory for file operations (default: current directory)')
    args = parser.parse_args()
    
    try:
        # Start the external code editing tool server as a background process
        # This server handles file system operations for the agent
        with CommandProcess(f"simplex_tool_server -p {EDIT_TOOL_SERVER_PORT} -c 20") as proc:
            # Run the async main function with optional workspace directory
            asyncio.run(main(base_dir = Path(args.workspace)))
    except Exception as startup_err:
        # Propagate any errors during server/agent startup
        raise RuntimeError("Failed to start agent or tool server") from startup_err

# --------------------------
# simplex_tool_server Reference
# --------------------------
# simplex_tool_server [options]:
# -h [ --help ]                    Show command line help
# -p [ --port ] arg                Port number for the server (REQUIRED)
# -j [ --jobs ] arg (=1)           Number of async worker processes (default: 1)
# -n [ --head-n ] arg (=200)       Lines for file preview (default: 200)
# -s [ --history ] arg (=15)       Undo history entries per file (default: 15)
# -c [ --concurrent ] arg (=4)     Threads for concurrent search (default: 4)
# -m [ --max-result ] arg (=24576) Max response bytes (default: 24576)
