import asyncio
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import simplex.basics
import simplex.context
import simplex.models
import simplex.tools

from simplex.basics import PromptTemplate, ModelResponse
from .base import AgentLoop


class ParallelSampleAdapter:
    """
    Adapter class for parallel trajectory sampling with AgentLoop instances
    
    This class encapsulates the behavior of AgentLoop to enable parallel execution
    of multiple trajectories for the same or different prompts. It provides an
    interface consistent with AgentLoop while managing multiple concurrent executions.
    
    The adapter allows specifying the number of trajectories to sample in parallel,
    enabling efficient exploration of different response paths or testing of various
    prompts simultaneously.
    
    Attributes:
        __agent_loop: Base AgentLoop instance to use as template for cloning
        __executor: ThreadPoolExecutor for managing parallel executions
    """

    def __init__(self, agent_loop: AgentLoop) -> None:
        """
        Initialize ParallelSampleAdapter instance
        
        Args:
            agent_loop: Base AgentLoop instance to use as template for parallel executions
        """
        self.__agent_loop = agent_loop
        # Use thread pool executor for parallel sampling
        self.__executor = ThreadPoolExecutor()
    def _run_loop_complete(self, loop_instance, system, user, history, max_iteration, timeout, max_retry, keep_original_system):
        """
        Internal helper method to run the async complete method in a new event loop
        
        Args:
            loop_instance: Cloned AgentLoop instance to execute
            system: System prompt template
            user: User prompt template
            history: Message history
            max_iteration: Max iterations for the loop
            timeout: Timeout for model generation
            max_retry: Max retry attempts
            keep_original_system: Whether to preserve original system prompt from history
            
        Returns:
            Result from the complete method
        """
        import asyncio
        
        # Create a new event loop for this thread if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Current loop is closed")
        except RuntimeError:
            # No event loop in this thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            loop_instance.complete(
                system=system,
                user=user,
                history=history,
                max_iteration=max_iteration,
                timeout=timeout,
                max_retry=max_retry,
                keep_original_system=keep_original_system
            )
        )

    async def sample_trajectories(
        self,
        prompts: Union[PromptTemplate, List[PromptTemplate]],
        num_trajectories: int,
        system: Optional[PromptTemplate] = None,
        history: Optional[List[dict]] = None,
        max_iteration: int = 30,
        timeout: float = 120,
        max_retry: int = 5,
        keep_original_system: bool = False
    ) -> List[List[ModelResponse]]:
        """
        Execute parallel trajectory sampling for given prompts
        
        Runs multiple AgentLoop instances in parallel to generate different response
        trajectories for the same or different prompts. Each trajectory represents
        an independent execution path that may produce different results based on
        model stochasticity or tool interactions.
        
        Args:
            prompts: Single prompt template or list of prompt templates to sample
            num_trajectories: Number of parallel trajectories to execute per prompt
            system: System prompt template for all trajectories (optional)
            history: Message history to prepend to conversations (optional)
            max_iteration: Maximum iterations per trajectory (passed to AgentLoop.complete)
            timeout: Timeout for model generation per trajectory (passed to AgentLoop.complete)
            max_retry: Maximum retry attempts per trajectory (passed to AgentLoop.complete)
            keep_original_system: Whether to preserve original system prompt from history
            
        Returns:
            List of lists containing ModelResponse objects for each prompt's trajectories
            Format: [[responses_for_prompt_1], [responses_for_prompt_2], ...]
            
        Raises:
            RuntimeError: If underlying AgentLoop execution fails
        """
        # Convert single prompt to list for uniform processing
        if isinstance(prompts, PromptTemplate):
            prompts = [prompts]
        
        all_results = []
        
        # Execute trajectories for each prompt
        for prompt in prompts:
            # Create tasks for parallel execution
            tasks = []
            for _ in range(num_trajectories):
                # Clone the base agent loop for each trajectory to ensure isolation
                cloned_loop = self.__agent_loop.clone()
                
                # Schedule the complete method execution in the thread pool
                task = asyncio.get_event_loop().run_in_executor(
                    self.__executor,
                    self._run_loop_complete,
                    cloned_loop,
                    system,
                    prompt,
                    history,
                    max_iteration,
                    timeout,
                    max_retry,
                    keep_original_system
                )
                tasks.append(task)
            
            # Wait for all trajectories to complete for this prompt
            try:
                prompt_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for any exceptions in the results
                processed_results = []
                for result in prompt_results:
                    if isinstance(result, Exception):
                        # Re-raise exceptions to maintain consistency with AgentLoop behavior
                        raise result
                    processed_results.append(result)
                
                all_results.append(processed_results)
            except Exception as e:
                # Clean up the executor on error
                self.__executor.shutdown(wait=False)
                raise e
        
        return all_results

    async def close(self) -> None:
        """
        Close the adapter and clean up resources
        
        Shuts down the internal thread pool executor gracefully.
        """
        self.__executor.shutdown(wait=True)

    async def __aenter__(self):
        """
        Async context manager entry point
        
        Returns:
            Self for use in async with statements
        """
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Async context manager exit point
        
        Cleans up resources when exiting the context.
        
        Args:
            exc_type: Exception type if raised in context
            exc: Exception instance if raised in context
            tb: Traceback if exception occurred
        """
        await self.close()
        return False
