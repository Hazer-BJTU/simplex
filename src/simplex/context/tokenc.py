import os
import uuid

from typing import List, Dict, Optional, TYPE_CHECKING

import simplex.basics
import simplex.context.base

from simplex.basics import ModelResponse, UserNotify
from simplex.context.base import ContextPlugin

if TYPE_CHECKING:
    from simplex.io.base import UserOutputInterface


class TokenCostCounter(ContextPlugin):
    """
    A context plugin that tracks and manages token cost statistics during AI model interactions.
    
    This class inherits from ContextPlugin and provides functionality to monitor:
    - Total token consumption across all interactions
    - Input (prompt) token usage
    - Output (completion) token usage
    - Maximum token usage per category
    - Number of iterations/interactions
    
    The class also provides methods to reset counters, bind I/O interfaces, and report
    token costs when the context exits.
    
    Attributes:
        empty_on_reset (bool): Whether to reset all counters to zero during reset operation
        token_total (int): Total number of tokens consumed across all interactions
        token_max (int): Maximum number of tokens consumed in a single interaction
        input_token_total (int): Total number of input/prompt tokens consumed
        input_token_max (int): Maximum number of input/prompt tokens in a single interaction
        output_token_total (int): Total number of output/completion tokens generated
        output_token_max (int): Maximum number of output/completion tokens in a single interaction
        iterations (int): Count of completed interactions
        output_interface (Optional[UserOutputInterface]): Interface for pushing notifications
    """
    
    def __init__(self, instance_id: Optional[str] = None, empty_on_reset: bool = True) -> None:
        """
        Initialize the TokenCostCounter instance.
        
        Args:
            instance_id (Optional[str]): Unique identifier for this instance. If None, 
                                       a UUID will be generated automatically.
            empty_on_reset (bool): Flag indicating whether to reset all counters to zero 
                                  when reset() is called. If False, counters will retain their values.
        """
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex)

        # Configuration flag determining reset behavior
        self.empty_on_reset = empty_on_reset

        # Initialize all token counting variables to zero
        self.token_total: int = 0          # Cumulative count of all tokens processed
        self.token_max: int = 0            # Highest token count in any single response
        self.input_token_total: int = 0    # Cumulative count of input/prompt tokens
        self.input_token_max: int = 0      # Highest input token count in any single response
        self.output_token_total: int = 0   # Cumulative count of output/completion tokens
        self.output_token_max: int = 0     # Highest output token count in any single response
        self.iterations: int = 0           # Count of completed model interactions

        # Store reference to output interface for notifications
        self.output_interface: Optional["UserOutputInterface"] = None

    @property
    def token_cost(self) -> List[Dict]:
        """
        Get token cost statistics formatted as a list of dictionaries.
        
        Returns:
            List[Dict]: A list containing token cost information for different categories:
                       - 'All': Overall token statistics (total and maximum)
                       - 'Prompt': Input token statistics (total and maximum)
                       - 'Generated': Output token statistics (total and maximum)
        """
        return [
            {'Title': 'All', 'total': self.token_total, 'max': self.token_max},
            {'Title': 'Prompt', 'total': self.input_token_total, 'max': self.input_token_max},  # Note: Uses token_max instead of input_token_max - likely intentional for overall max tracking
            {'Title': 'Generated', 'total': self.output_token_total, 'max': self.output_token_max}
        ]
    
    def _format_token_count(self, count: int) -> str:
        """
        Format a token count as a human-readable string.
        
        Args:
            count (int): The token count to format
            
        Returns:
            str: Formatted string with 'k' for thousands or 'm' for millions
                 Examples: 1200 -> '1.2k', 3500000 -> '3.5m', 500 -> '500'
        """
        if count >= 1_000_000:  # 1 million and above
            return f"{count / 1_000_000:.1f}m"
        elif count >= 1_000:  # 1 thousand to 999,999
            return f"{count / 1_000:.1f}k"
        else:  # less than 1 thousand
            return str(count)

    @property
    def token_cost_formatted(self) -> List[Dict]:
        """
        Get token cost statistics formatted as a list of dictionaries with human-readable values.
        
        Returns:
            List[Dict]: A list containing token cost information for different categories,
                       with total and max values formatted as human-readable strings:
                       - 'All': Overall token statistics (formatted total and maximum)
                       - 'Prompt': Input token statistics (formatted total and maximum)
                       - 'Generated': Output token statistics (formatted total and maximum)
        """
        return [
            {
                'Title': 'All', 
                'total': self._format_token_count(self.token_total), 
                'max': self._format_token_count(self.token_max)
            },
            {
                'Title': 'Prompt', 
                'total': self._format_token_count(self.input_token_total), 
                'max': self._format_token_count(self.input_token_max)
            },
            {
                'Title': 'Generated', 
                'total': self._format_token_count(self.output_token_total), 
                'max': self._format_token_count(self.output_token_max)
            }
        ]

    async def reset(self) -> None:
        """
        Reset all token counters to zero if empty_on_reset is True.
        
        This method is called when the context is reset. If empty_on_reset is True,
        all token counters and the iteration counter are reset to zero. Otherwise,
        the counters retain their current values.
        """
        if self.empty_on_reset:
            self.token_total = 0
            self.token_max = 0
            self.input_token_total = 0
            self.input_token_max = 0
            self.output_token_total = 0
            self.output_token_max = 0
            self.iterations = 0

    async def bind_io(self, output_interface: "UserOutputInterface", **kwargs) -> None:
        """
        Bind an output interface for sending notifications.
        
        Args:
            output_interface (UserOutputInterface): The interface to use for pushing notifications
            **kwargs: Additional keyword arguments (currently unused)
        """
        self.output_interface = output_interface

    async def after_response_async(self, model_response: ModelResponse, **kwargs) -> None:
        """
        Process token cost information after each model response.
        
        This method is called automatically after each model response to update
        token counters with information from the response.
        
        Args:
            model_response (ModelResponse): The response object containing token cost information
            **kwargs: Additional keyword arguments (currently unused)
        """
        # Update total token count with the cost from this response
        self.token_total += model_response.token_cost
        # Update maximum token cost if this response had more tokens than previous max
        self.token_max = max(self.token_max, model_response.token_cost)
        
        # Process additional token information from response extras if available
        if model_response.extras:
            # Extract prompt and completion token counts from response extras
            prompt_tokens = model_response.extras.get('prompt_tokens', 0)
            completion_tokens = model_response.extras.get('completion_tokens', 0)
            
            # Update input token counters
            self.input_token_total += prompt_tokens
            self.input_token_max = max(self.input_token_max, prompt_tokens)
            
            # Update output token counters
            self.output_token_total += completion_tokens
            self.output_token_max = max(self.output_token_max, completion_tokens)
        
        # Increment the iteration counter to track number of responses processed
        self.iterations += 1
        
    async def on_exit_async(self, *args, **kwargs) -> None:
        """
        Send token cost notification when the context exits.
        
        This method is called when the context exits, sending a summary of token
        usage statistics through the bound output interface if available.
        
        Args:
            *args: Variable positional arguments (currently unused)
            **kwargs: Additional keyword arguments (currently unused)
        """
        if self.output_interface:
            # Push a notification containing the formatted token cost statistics
            await self.output_interface.push_message(UserNotify('notify', title='Token Cost', objects=self.token_cost_formatted))

if __name__ == '__main__':
    pass
