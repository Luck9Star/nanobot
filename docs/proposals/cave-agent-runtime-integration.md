# Proposal: Integrate cave-agent Python Primitives Runtime into nanobot

> Status: **PoC** | Branch: `poc/cave-agent-runtime` | Date: 2026-04-14

## 1. Background & Motivation

### The Problem

nanobot uses **JSON Schema-based tool calling** (OpenAI function calling protocol). Each `ExecTool` invocation spawns an **independent subprocess** — variables, imports, and function definitions do NOT persist across calls. This severely limits nanobot's ability to handle complex, multi-step Python workflows.

### The Solution

[cave-agent](https://github.com/acodercat/cave-agent) provides a **stateful Python runtime** backed by IPython/IPyKernel:

- **Variable/Function/Type injection**: Inject any Python object into the runtime namespace
- **State persistence**: Variables, imports, and definitions persist across calls (Jupyter-like)
- **Complex object manipulation**: LLM can directly operate on DataFrames, DB connections, ML models, etc.
- **AST-based security**: `SecurityChecker` with fine-grained rules (ImportRule, FunctionRule, AttributeRule, RegexRule)
- **Two backends**: `IPythonRuntime` (in-process) and `IPyKernelRuntime` (process-isolated)

### Key Insight (from cave-agent paper)

LLMs are naturally good at generating code. JSON Schema tool calling is an artificial constraint. Letting LLMs generate Python code to directly manipulate objects yields **significantly higher success rates** than JSON tool calling.

## 2. Architecture

### Current nanobot Flow

```
LLM → JSON tool_calls → ToolRegistry.execute() → Tool.execute(**kwargs)
                          {"name":"exec",
                           "args":{"command":"pip install X"}}
                                               → subprocess (stateless)
```

### Proposed Flow

```
LLM → JSON tool_calls → ToolRegistry.execute()
                          {"name":"python",
                           "args":{"code":"import pandas as pd\ndf = pd.read_csv('data.csv')"}}
                                               → PythonRuntimeTool.execute(code)
                                                   → cave-agent IPythonRuntime.execute(code)
                                                   → stateful IPython namespace
                                                   → variables persist across calls!
```

### Component Diagram

```
nanobot AgentLoop
    │
    ├── ExecTool              (existing, subprocess, stateless)
    ├── ReadFileTool          (existing)
    ├── WriteFileTool         (existing)
    │   ...
    │
    └── PythonRuntimeTool     (NEW ← cave-agent)
            │
            ├── IPythonRuntime / IPyKernelRuntime
            │     ├── IPythonExecutor (stateful execution)
            │     ├── inject_into_namespace() (Variable/Function/Type)
            │     └── execute(code) → ExecutionResult
            │
            ├── SecurityChecker (cave-agent's AST security validation)
            └── Namespace descriptor (generates prompt text for LLM)
```

## 3. Integration Design

### 3.1 New Tool: `PythonRuntimeTool`

Located at: `nanobot/agent/tools/python_runtime.py`

**Core API:**
```python
class PythonRuntimeTool(Tool):
    async def execute(self, code: str) -> str:
        """Execute Python code in stateful runtime, return output."""
        
    def inject_variable(self, name: str, value: Any, description: str = ""):
        """Inject a Python object into the namespace."""
        
    def inject_function(self, func: Callable, description: str = ""):
        """Inject a callable into the namespace."""
        
    def inject_type(self, cls: type, description: str = ""):
        """Inject a type/class into the namespace."""
        
    def describe_namespace(self) -> str:
        """Generate namespace description for system prompt injection."""
        
    async def retrieve(self, name: str) -> Any:
        """Retrieve a variable value from the namespace."""
        
    async def reset(self):
        """Reset the runtime (clear all state)."""
```

**Tool Properties:**
- `name`: `"python"`
- `exclusive`: `True` (modifies namespace state, cannot run concurrently)
- `read_only`: `False`
- `concurrency_safe`: `False`

### 3.2 Configuration

Add to `nanobot/config/schema.py`:

```python
@dataclass
class PythonRuntimeConfig:
    enable: bool = True
    backend: str = "ipython"         # "ipython" or "ipykernel"
    max_output_chars: int = 10_000
    security_rules: list | None = None
```

### 3.3 System Prompt Injection

Modify `nanobot/agent/context.py` to inject namespace descriptions into the system prompt:

```
<python_runtime>
Available Python objects (stateful, persistent across calls):

Functions:
  add(a: int, b: int) -> int — Add two numbers
  process_data(df: DataFrame) -> dict — Process a DataFrame

Variables:
  secret (str): A reversed message
  data (DataFrame): Input dataset with columns [name, age, score]

Types:
  AnalysisResult (dataclass): mean (float), count (int), status (str)
</python_runtime>
```

### 3.4 Registration in AgentLoop

Modify `nanobot/agent/loop.py` `_register_default_tools()`:

```python
if python_runtime_config.enable:
    python_tool = PythonRuntimeTool(
        backend=python_runtime_config.backend,
        security_rules=python_runtime_config.security_rules,
        max_output_chars=python_runtime_config.max_output_chars,
    )
    self.tools.register(python_tool)
```

## 4. Runtime Backend Selection

| Scenario | Backend | Reason |
|----------|---------|--------|
| Development/Debug | `IPythonRuntime` | Zero overhead, direct debugging |
| Production | `IPyKernelRuntime` | Process isolation, crash-safe |
| Untrusted code | `IPyKernelRuntime` + bubblewrap | Double isolation |

## 5. Key Challenges

### 5.1 IPyKernel Async Lifecycle

`IPyKernelRuntime` requires `async with` for kernel startup/shutdown. This must be handled in `AgentLoop.__init__()` or lazily on first use.

### 5.2 Prompt Budget

Namespace descriptions consume context tokens. Need to:
- Truncate long descriptions
- Compact when context is full (nanobot already has microcompact)

### 5.3 Error Recovery

If the IPython kernel crashes, need to:
- Detect crash
- Auto-restart kernel
- Re-inject all registered primitives

### 5.4 Security

cave-agent's `SecurityChecker` provides AST-level validation. Combined with nanobot's existing `_guard_command()` and bubblewrap sandbox, this gives defense-in-depth.

## 6. Implementation Plan

### Phase 1: PoC (Current)
- [x] Create `PythonRuntimeTool` with basic IPythonRuntime backend
- [x] Register in `AgentLoop._register_default_tools()`
- [x] Test: basic stateful execution (variable persists across calls)
- [x] Test: inject Variable/Function and verify LLM can use them

### Phase 2: Integration
- [ ] Add `PythonRuntimeConfig` to nanobot config schema
- [ ] Inject namespace descriptions into system prompt via `ContextBuilder`
- [ ] Handle IPyKernel async lifecycle
- [ ] Add error recovery (kernel crash detection + restart)

### Phase 3: Production
- [ ] Full security rules configuration
- [ ] Context budget management for namespace descriptions
- [ ] Integration tests
- [ ] Documentation

## 7. Dependencies

```
# Already available via cave-agent
cave-agent>=0.7.0
ipython
# Optional (for IPyKernelRuntime)
ipykernel
dill
```

## 8. References

- cave-agent repo: https://github.com/acodercat/cave-agent
- cave-agent paper: https://arxiv.org/abs/2601.01569
- nanobot repo: https://github.com/HKUDS/nanobot
- Fork: https://github.com/Luck9Star/nanobot
