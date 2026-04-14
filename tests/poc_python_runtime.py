"""PoC: Verify PythonRuntimeTool stateful execution.

This script tests the core integration of cave-agent's runtime into nanobot
WITHOUT requiring a running nanobot instance or LLM API key.

Run:
    cd /path/to/nanobot
    pip install 'cave-agent[all]'
    python tests/poc_python_runtime.py
"""

import asyncio
import sys


async def test_basic_stateful_execution():
    """Variables persist across execute() calls."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    tool = PythonRuntimeTool()

    r1 = await tool.execute(code="x = 42\nprint(f'Set x = {x}')")
    assert "Set x = 42" in r1, f"Expected 'Set x = 42', got: {r1}"

    r2 = await tool.execute(code="print(f'x is still {x}')")
    assert "x is still 42" in r2, f"Expected 'x is still 42', got: {r2}"

    print("  PASS: variables persist across calls")


async def test_inject_variable():
    """Injected variables are accessible in code."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    tool = PythonRuntimeTool()
    tool.inject_variable("secret", "!dlrow ,olleH", "A reversed message")

    r = await tool.execute(code="print(secret[::-1])")
    assert "Hello, world!" in r, f"Expected 'Hello, world!', got: {r}"

    print("  PASS: injected variables accessible")


async def test_inject_function():
    """Injected functions can be called from code."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    def add(a: int, b: int) -> int:
        return a + b

    tool = PythonRuntimeTool()
    tool.inject_function(add, "Add two numbers")

    r = await tool.execute(code="result = add(3, 5)\nprint(f'3 + 5 = {result}')")
    assert "3 + 5 = 8" in r, f"Expected '3 + 5 = 8', got: {r}"

    print("  PASS: injected functions callable")


async def test_complex_objects():
    """Complex objects (dict, list) persist and are manipulable."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    tool = PythonRuntimeTool()
    tool.inject_variable(
        "data", {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}, "User data"
    )

    r1 = await tool.execute(code="names = [u['name'] for u in data['users']]\nprint(names)")
    assert "Alice" in r1 and "Bob" in r1, f"Expected names, got: {r1}"

    r2 = await tool.execute(
        code="data['users'].append({'name': 'Charlie', 'age': 35})\nprint(len(data['users']))"
    )
    assert "3" in r2, f"Expected 3 users, got: {r2}"

    r3 = await tool.execute(code="print(data['users'][-1]['name'])")
    assert "Charlie" in r3, f"Expected Charlie, got: {r3}"

    print("  PASS: complex objects persist and mutable")


async def test_error_handling():
    """Errors are returned as readable messages, runtime stays alive."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    tool = PythonRuntimeTool()

    r1 = await tool.execute(code="1/0")
    assert "Error" in r1, f"Expected error, got: {r1}"

    r2 = await tool.execute(code="print('still alive')")
    assert "still alive" in r2, f"Runtime died after error, got: {r2}"

    print("  PASS: errors don't kill runtime")


async def test_reset():
    """reset() clears all state."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    tool = PythonRuntimeTool()

    await tool.execute(code="z = 999")
    await tool.reset()

    r = await tool.execute(code="try:\n    print(z)\nexcept NameError:\n    print('z is gone')")
    assert "z is gone" in r, f"Expected z gone after reset, got: {r}"

    print("  PASS: reset clears state")


async def test_retrieve():
    """retrieve() gets variable values for injected variables."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    tool = PythonRuntimeTool()
    tool.inject_variable("answer", 42, "The answer")

    val = await tool.retrieve("answer")
    assert val == 42, f"Expected 42, got: {val}"

    print("  PASS: retrieve returns injected values")


async def test_describe_namespace():
    """describe_namespace() returns readable description."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    tool = PythonRuntimeTool()
    tool.inject_function(greet, "Greet someone")
    tool.inject_variable("config", {"debug": True}, "App config")

    desc = tool.describe_namespace()
    assert "greet" in desc, f"Expected 'greet' in description, got: {desc}"
    assert "config" in desc, f"Expected 'config' in description, got: {desc}"

    print("  PASS: describe_namespace includes injected items")


async def test_tool_schema():
    """Tool has valid JSON schema for nanobot's ToolRegistry."""
    from nanobot.agent.tools.python_runtime import PythonRuntimeTool

    tool = PythonRuntimeTool()
    schema = tool.to_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "python"
    assert "code" in schema["function"]["parameters"]["properties"]
    assert "code" in schema["function"]["parameters"]["required"]

    print("  PASS: tool schema is valid")


async def main():
    print("=" * 60)
    print("PythonRuntimeTool PoC — Stateful Execution Tests")
    print("=" * 60)

    try:
        from cave_agent.runtime import IPythonRuntime  # noqa: F401
    except ImportError:
        print("\nERROR: cave-agent not installed!")
        print("Run: pip install 'cave-agent[all]'")
        sys.exit(1)

    tests = [
        test_tool_schema,
        test_basic_stateful_execution,
        test_inject_variable,
        test_inject_function,
        test_complex_objects,
        test_error_handling,
        test_reset,
        test_retrieve,
        test_describe_namespace,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed:
        sys.exit(1)
    print("\nAll tests passed! PythonRuntimeTool works correctly.")


if __name__ == "__main__":
    asyncio.run(main())
