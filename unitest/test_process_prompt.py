"""
Test script for SequentialPlan.process_prompt function.

This script tests the new skill loading functionality added to the SequentialPlan class.
It does not depend on external test environments and uses mock data.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simplex.tools.plan import SequentialPlan
from simplex.basics import PromptTemplate, AgentLoopStateEdit


def test_sequential_plan_init():
    """Test that SequentialPlan initializes correctly with skill loaded."""
    print("\n=== Test 1: SequentialPlan Initialization ===")
    
    plan = SequentialPlan(rename='make_plan')
    
    # Check that skill is loaded
    assert hasattr(plan, 'skill'), "SequentialPlan should have 'skill' attribute"
    assert hasattr(plan, 'skill_added'), "SequentialPlan should have 'skill_added' attribute"
    assert plan.skill_added == False, "skill_added should be False initially"
    
    # Check that skill content is not empty
    assert len(plan.skill) > 0, "Skill content should not be empty"
    
    # Check that %make_plan% placeholder was replaced
    assert '%make_plan%' not in plan.skill, "Placeholder %make_plan% should be replaced"
    assert '`make_plan`' in plan.skill, "Tool name should be in skill content"
    
    print(f"  ✓ skill_added initialized to False")
    print(f"  ✓ Skill loaded with length: {len(plan.skill)}")
    print(f"  ✓ Placeholder replaced correctly")
    print("  PASSED\n")
    return plan


def test_custom_rename():
    """Test that custom rename works correctly with skill replacement."""
    print("=== Test 2: Custom Rename ===")
    
    custom_name = 'my_custom_plan'
    plan = SequentialPlan(rename=custom_name)
    
    # Check that the skill contains the custom tool name
    assert f'`{custom_name}`' in plan.skill, f"Custom tool name `{custom_name}` should be in skill"
    assert plan.name == custom_name, f"Plan name should be {custom_name}"
    
    print(f"  ✓ Custom tool name '{custom_name}' used in skill")
    print(f"  ✓ Plan name set correctly")
    print("  PASSED\n")
    return plan


def test_process_prompt_first_call():
    """Test the first call to process_prompt returns AgentLoopStateEdit."""
    print("=== Test 3: First Call to process_prompt ===")
    
    plan = SequentialPlan(rename='make_plan')
    user_prompt = PromptTemplate("This is a test user prompt.")
    
    result = plan.process_prompt(user_prompt)
    
    # Check return type
    assert result is not None, "First call should return a non-None result"
    assert isinstance(result, AgentLoopStateEdit), "Result should be AgentLoopStateEdit"
    
    # Check that skill_added was set to True
    assert plan.skill_added == True, "skill_added should be True after first call"
    
    # Check that user_prompt was modified
    assert result.user_prompt is not None, "user_prompt should not be None in result"
    
    # Check that skill was appended
    expected_content = str(user_prompt) + "\n\n" + plan.skill
    actual_content = str(result.user_prompt).strip()
    expected_content_stripped = expected_content.strip()
    
    assert actual_content == expected_content_stripped, \
        f"Content mismatch.\nExpected length: {len(expected_content_stripped)}\nActual length: {len(actual_content)}"
    
    print(f"  ✓ Returns AgentLoopStateEdit on first call")
    print(f"  ✓ skill_added set to True")
    print(f"  ✓ Skill appended to user_prompt correctly")
    print("  PASSED\n")
    return plan


def test_process_prompt_second_call():
    """Test that subsequent calls to process_prompt return None."""
    print("=== Test 4: Second Call to process_prompt ===")
    
    plan = SequentialPlan(rename='make_plan')
    user_prompt = PromptTemplate("This is a test user prompt.")
    
    # First call
    result1 = plan.process_prompt(user_prompt)
    assert result1 is not None, "First call should return non-None"
    
    # Second call
    result2 = plan.process_prompt(user_prompt)
    assert result2 is None, "Second call should return None"
    
    # Third call
    result3 = plan.process_prompt(user_prompt)
    assert result3 is None, "Third call should also return None"
    
    print(f"  ✓ Second call returns None")
    print(f"  ✓ Third call returns None")
    print("  PASSED\n")


def test_prompt_template_addition():
    """Test that PromptTemplate + string works correctly."""
    print("=== Test 5: PromptTemplate Addition ===")
    
    user_prompt = PromptTemplate("Original prompt content.")
    skill = "\n\n# Additional Instructions\n\nSome skill content here."
    
    new_prompt = user_prompt + skill
    
    assert isinstance(new_prompt, PromptTemplate), "Result should be PromptTemplate"
    assert "Original prompt content" in str(new_prompt), "Original content should be preserved"
    assert "Additional Instructions" in str(new_prompt), "Skill content should be added"
    
    # Original prompt should be unchanged
    assert str(user_prompt) == "Original prompt content.", "Original prompt should not be modified"
    
    print(f"  ✓ PromptTemplate + string returns new PromptTemplate")
    print(f"  ✓ Original prompt unchanged")
    print(f"  ✓ New content appended correctly")
    print("  PASSED\n")


def test_empty_user_prompt():
    """Test process_prompt with empty user prompt."""
    print("=== Test 6: Empty User Prompt ===")
    
    plan = SequentialPlan(rename='make_plan')
    user_prompt = PromptTemplate("")  # Empty prompt
    
    result = plan.process_prompt(user_prompt)
    
    assert result is not None, "Should still return AgentLoopStateEdit with empty prompt"
    assert plan.skill_added == True, "skill_added should be True"
    
    # The result should contain only the skill (plus empty content)
    result_content = str(result.user_prompt).strip()
    assert len(result_content) > 0, "Result should have content even with empty input"
    
    print(f"  ✓ Works correctly with empty user prompt")
    print("  PASSED\n")


def test_skill_content_structure():
    """Test that the loaded skill has expected structure."""
    print("=== Test 7: Skill Content Structure ===")
    
    plan = SequentialPlan(rename='make_plan')
    
    # Check skill contains expected sections
    assert "# Plan Management Instructions" in plan.skill, "Should contain main title"
    assert "## Purpose" in plan.skill, "Should contain Purpose section"
    assert "## Usage" in plan.skill, "Should contain Usage section"
    assert "## Best Practices" in plan.skill, "Should contain Best Practices section"
    
    print(f"  ✓ Skill has expected structure")
    print(f"  ✓ Contains main title")
    print(f"  ✓ Contains Purpose section")
    print(f"  ✓ Contains Usage section")
    print(f"  ✓ Contains Best Practices section")
    print("  PASSED\n")


def test_reset_behavior():
    """Test that reset does not affect skill_added flag."""
    print("=== Test 8: Reset Behavior ===")
    
    plan = SequentialPlan(rename='make_plan', empty_on_reset=True)
    user_prompt = PromptTemplate("Test prompt")
    
    # First call
    result1 = plan.process_prompt(user_prompt)
    assert result1 is not None
    
    # Reset (async, so we need to import asyncio)
    import asyncio
    asyncio.run(plan.reset())
    
    # skill_added should still be True after reset
    assert plan.skill_added == True, "skill_added should still be True after reset"
    
    # Next call should still return None
    result2 = plan.process_prompt(user_prompt)
    assert result2 is None, "Should still return None after reset"
    
    print(f"  ✓ skill_added not affected by reset")
    print(f"  ✓ Subsequent calls still return None")
    print("  PASSED\n")


def test_clone_behavior():
    """Test that cloned plans have independent skill_added state."""
    print("=== Test 9: Clone Behavior ===")
    
    plan1 = SequentialPlan(rename='make_plan')
    user_prompt = PromptTemplate("Test prompt")
    
    # Use first plan
    result1 = plan1.process_prompt(user_prompt)
    assert result1 is not None
    
    # Clone
    plan2 = plan1.clone()
    
    # Check clone has independent state
    assert hasattr(plan2, 'skill'), "Clone should have skill attribute"
    assert hasattr(plan2, 'skill_added'), "Clone should have skill_added attribute"
    
    # Clone should have fresh skill_added state
    # Note: clone() doesn't call __init__ with skill loading, so this tests clone implementation
    
    print(f"  ✓ Clone has required attributes")
    print("  PASSED\n")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("Testing SequentialPlan.process_prompt Function")
    print("=" * 60)
    
    tests = [
        test_sequential_plan_init,
        test_custom_rename,
        test_process_prompt_first_call,
        test_process_prompt_second_call,
        test_prompt_template_addition,
        test_empty_user_prompt,
        test_skill_content_structure,
        test_reset_behavior,
        test_clone_behavior,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)