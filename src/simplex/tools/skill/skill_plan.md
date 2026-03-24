# Plan Management Instructions

## Tool Overview
The %make_plan% tool is a planning and checklist management tool designed to help agents track task progress, record statuses, and communicate intentions clearly to users. It supports editing plans/checklists via natural language and returns a comparison between the old and new plans after each operation, preventing information loss due to context collapse.

## Core Purpose
- Record task plans and checklists at key nodes to avoid forgetting details when context is collapsed.
- Intuitively show users your intentions, actions, and progress through clear plans.
- Provide guidance when tasks are tricky or you are unsure about the next step (via check_only mode).

## Important Notes
- The content should be concise and specific; avoid vague descriptions (e.g., "do the task" is not acceptable; specify the task details).
- Always select the correct edit_type: use "replace" for initial plans, "append" for updates, and "check_only" for reviews.
- After each use, check the tool’s return (old vs. new plan) to ensure the operation is correct.
- Do not skip using the tool at key nodes—this is crucial to avoid context loss and ensure user understanding.
