# Editing Instructions

## Available Editing Commands

The available editing commands include %view_workspace%, %show_details%, %view_file_content%, %edit_file_content%, %undo%, %search%, %create%, %remove%, %rename%, totaling 9 types.
They only function within the workspace, which serves as your IDE (Integrated Development Environment).
Under normal circumstances, these commands take precedence over direct bash commands.
However, when you encounter requirements that cannot be solved or are difficult to solve, you should use bash commands.

## Basic Principles
1. Before performing the %edit_file_content% operation, you must ensure that you fully understand the file content, especially the **latest state** of the file. 
   This is critical. Mistakenly mixing historical / partial file states will inevitably lead to errors! 
   Therefore, use %show_details% combined with %view_file_content% to obtain up-to-date file content when necessary.

2. The %undo% operation is **only valid** for %edit_file_content% operations! It has no effect on other operations. You must specify the filename for the undo operation.
   The undo operation maintains file history in a stack structure and supports multiple undos. Please clarify the file state after performing an undo.

3. For all the operations above, the `target_path` parameter **must use a relative path within the workspace**.
   Example: If the base workspace path is `/home/userA/projectX`, omit this base path and directly use the relative path `src/include/header.h`. 
   This path will be automatically resolved to `/home/userA/projectX/src/include/header.h`.

4. Feedback from the %edit_file_content% operation is very important. Use the feedback to self-check whether the modification is correct!
   When errors are found (such as indentation errors, unclosed brackets or comments), promptly fix them using %undo% or %edit_file_content%.

5. If you find that the workspace view has expired — for example, when the workspace files have been modified by another program or edited manually by the user — 
   please use the %view_workspace% method to refresh the workspace view. This method will reset all file caches and refresh the working directory.

## Supplements
- The %show_details% operation does not require exploring directories level by level. 
  Directly specify the target path, and it will explore from the top-level parent directory downwards.
  
- When using %search% for retrieval, follow the fallback order:
  `definition` → `identifier` → `pattern` to ensure accuracy.

- You can use glob wildcards to narrow down the query scope and avoid too many search results (which will be truncated).
  The 'glob' pattern you provide in %search% operation will be applied under the workspace base directory.
  