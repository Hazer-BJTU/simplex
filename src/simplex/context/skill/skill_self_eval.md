# Action Self-Evaluation Skill

## Overview
You may notice that in each function calling schema two extra parameters are required: `task_status` and `action_quality`.
This skill enables you to self-assess your current task state and the expected quality of your next action. 
Before making any tool call, you must provide two critical evaluations that help monitor and optimize your performance.

## Parameters

### 1. task_status
**Purpose:** Provide an objective assessment of the steady-state health of your current task execution pipeline.

**Values (from worst to best):**

| Value | Score | Meaning |
|-------|-------|---------|
| `derailed` | 0 | The task has completely gone off track. Especially when the same or similar error occurs TWICE or more.                                                   |
| `blocked`  | 1 | You've encountered a clear obstacle preventing progress. You know what's blocking you but need to find a way around it.                                   |
| `steady`   | 2 | There are no obvious obstacles or errors, but the correctness of the overall exploration and its potential final outcome remain uncertain.                |
| `solid`    | 3 | Making good, reliable progress with high confidence. Or you have successfully overcome an obstacle and verified the correctness of the method.            |
| `perfect`  | 4 | Everything is working optimally. All test results form a perfect closed loop, and you are just one final step away from successfully completing the task. |

### 2. action_quality
**Purpose:** Estimate how valuable your next action will be in advancing toward the goal.

**Values (from worst to best):**

| Value | Score | Meaning |
|-------|-------|---------|
| `unknown`    | 0 | You have no confidence in this action. Usually refers to the exploration and planning phase at the very beginning of a task, or the attempt following an encountered obstacle.     |
| `workaround` | 1 | This action may be redundant, adding unnecessary informational burden, or you stay skeptical about the solution or answer to the issues or obstacles encountered.                  |
| `effective`  | 2 | This action will definitely make progress. It's a straightforward, correct approach to the immediate need. This action will deliver crucial information to help resolve the issue. |
| `precise`    | 3 | This action exactly addresses the core need with minimal overhead. Especially when you have fully identified the root cause of the problem and successfully fixed it.              |
| `optimal`    | 4 | Given their limitations, this solution or answer is superior to all previous actions taken for the same issue or obstacle.                                                         |

## Examples

### **Scenario**: The user asks you to fetch and summarize the latest news about AI. 

### Example 1: Steady Progress with a Straightforward Action
- **Context**: After successfully retrieving the news articles, you now need to summarize them. The API returned clean data, and you’ve validated the structure.
- **task_status**: `steady` → No obstacles, but the final summary quality is still uncertain.
- **action_quality**: `effective` → Calling the summarization tool will definitely produce a summary; it’s the correct next step.

### Example 2: Encountering and Overcoming an Error
- **Context**: The news API returned a 429 rate‑limit error. You wait and retry with exponential backoff; the second attempt succeeds.
- **task_status**: `solid` → You encountered a clear obstacle (rate limit) and resolved it successfully, verifying that the retry logic works.
- **action_quality**: `precise` → The retry action exactly addresses the root cause (rate limiting) with minimal overhead, and it solved the problem.

### Example 3: Repeated Failure – Derailed
- **Context**: You’ve tried three different search APIs to get stock prices, but each returns authentication errors despite using valid credentials. You’ve run out of alternatives.
- **task_status**: `derailed` → The same type of error occurs repeatedly, and no workaround is in sight.
- **action_quality**: `workaround` → Your next action (e.g., asking the user for alternative credentials) depends on external factors and adds informational overhead.

### Example 4: Final Step – Perfect Task Status
- **Context**: You have generated a summary, formatted it, and run all validation checks. The only thing left is to output the final answer to the user.
- **task_status**: `perfect` → Everything is optimal; all test results form a perfect closed loop, and you are one step away from completion.
- **action_quality**: `optimal` → The output action is the most direct and superior way to deliver the final result compared to any earlier approach.

### Example 5: Blocked with a Clear Obstacle
- **Context**: You need to read a PDF, but the file is password‑protected. You know the password is required and that you don’t have it.
- **task_status**: `blocked` → You are stuck at a clear barrier and know exactly what’s missing.
- **action_quality**: `workaround` → Your next action (e.g., asking the user for the password) depends on external factors and adds informational overhead.

## Best Practices

1. **Be honest** - These evaluations are for your own guidance and system monitoring. Inaccurate self-assessments reduce their value.
2. **Update frequently** - Re-evaluate `task_status` after each significant step or when circumstances change.
3. **Calibrate over time** - Learn from outcomes. If you frequently rate actions as `optimal` but they fail, adjust your calibration.
4. **Use scores for reflection** - If you find yourself using `derailed` or `unknown` repeatedly, it may indicate you need to step back and reconsider your overall strategy.
