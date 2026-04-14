"""PythonRuntimeTool — stateful Python code execution powered by cave-agent's runtime.

This tool gives nanobot a Jupyter-like execution environment where variables,
imports, and function definitions persist across calls. It wraps cave-agent's
IPythonRuntime or IPyKernelRuntime as a nanobot Tool.

Architecture:
    LLM → tool_call("python", {"code": "..."}) → PythonRuntimeTool → cave-agent Runtime

Key advantages over nanobot's existing ExecTool (subprocess):
    - State persistence: variables/imports survive across tool calls
    - Object injection: arbitrary Python objects can be injected into the namespace
    - Direct manipulation: LLM generates Python code to operate on complex objects
    - AST security: fine-grained security rules (ImportRule, FunctionRule, etc.)
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Lazy import helpers — cave-agent may not be installed, so we import
# conditionally and give clear error messages.
# ---------------------------------------------------------------------------

_RUNTIME_IMPORT_ERROR = None
_IPythonRuntime = None
_IPyKernelRuntime = None
_Variable = None
_Function = None
_Type = None
_SecurityChecker = None


def _ensure_cave_agent() -> None:
    """Lazy-import cave-agent core types; raise clear error if missing."""
    global _RUNTIME_IMPORT_ERROR, _IPythonRuntime, _IPyKernelRuntime
    global _Variable, _Function, _Type, _SecurityChecker

    if _IPythonRuntime is not None:
        return

    try:
        from cave_agent.runtime.ipython_runtime import IPythonRuntime
        from cave_agent.runtime.primitives import Variable, Function, Type
        from cave_agent.security import SecurityChecker

        _IPythonRuntime = IPythonRuntime
        _IPyKernelRuntime = None
        _Variable = Variable
        _Function = Function
        _Type = Type
        _SecurityChecker = SecurityChecker
        _RUNTIME_IMPORT_ERROR = None
    except ImportError as exc:
        _RUNTIME_IMPORT_ERROR = ImportError(
            "cave-agent is required for PythonRuntimeTool. "
            "Install it with: pip install 'cave-agent[all]'"
        )
        raise _RUNTIME_IMPORT_ERROR from exc


def _get_ipykernel_runtime() -> type:
    try:
        from cave_agent.runtime.ipykernel_runtime import IPyKernelRuntime

        return IPyKernelRuntime
    except ImportError:
        raise ImportError(
            "IPyKernelRuntime requires ipykernel and dill. "
            "Install with: pip install 'cave-agent[ipykernel]'"
        )


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


@tool_parameters(
    tool_parameters_schema(
        code=StringSchema(
            "Python code to execute in the stateful runtime. "
            "Variables, imports, and functions persist across calls. "
            "Use print() to see output.",
        ),
        required=["code"],
    )
)
class PythonRuntimeTool(Tool):
    """Stateful Python code execution tool backed by cave-agent's runtime.

    Unlike ExecTool (which spawns an independent subprocess each call),
    this tool maintains a persistent IPython namespace where all variables,
    imports, and definitions survive across invocations.

    Usage from nanobot's LLM:
        tool_call("python", {"code": "import pandas as pd\\ndf = pd.read_csv('data.csv')\\nprint(df.head())"})
        # df persists! Next call can reference it directly:
        tool_call("python", {"code": "result = df[df['age'] > 30]\\nprint(result.shape)"})
    """

    _MAX_OUTPUT = 10_000
    _MAX_TRACEBACK = 2_000

    def __init__(
        self,
        *,
        backend: str = "ipython",
        security_rules: list | None = None,
        max_output_chars: int = 10_000,
        inject_functions: list[dict] | None = None,
        inject_variables: list[dict] | None = None,
        inject_types: list[dict] | None = None,
    ):
        """
        Args:
            backend: "ipython" (in-process, zero-overhead) or
                     "ipykernel" (separate process, crash-safe).
            security_rules: cave-agent SecurityRule instances.
            max_output_chars: Truncate output beyond this limit.
            inject_functions: List of {"func": callable, "description": str} to pre-inject.
            inject_variables: List of {"name": str, "value": Any, "description": str} to pre-inject.
            inject_types: List of {"cls": type, "description": str} to pre-inject.
        """
        _ensure_cave_agent()

        self._backend = backend
        self._security_rules = security_rules
        self._max_output = max_output_chars
        self._inject_functions = inject_functions or []
        self._inject_variables = inject_variables or []
        self._inject_types = inject_types or []
        self._runtime: Any = None  # initialized lazily in _ensure_runtime()
        self._started = False

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "python"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a STATEFUL runtime. "
            "Variables, imports, and functions persist across calls — "
            "like a Jupyter notebook where each call is a new cell. "
            "Use print() to see output. "
            "Prefer this over exec for data analysis, object manipulation, "
            "and multi-step Python workflows."
        )

    @property
    def exclusive(self) -> bool:
        return True  # modifies namespace state

    @property
    def read_only(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Runtime lifecycle
    # ------------------------------------------------------------------

    async def _ensure_runtime(self) -> Any:
        """Lazily initialize the cave-agent runtime."""
        if self._runtime is not None:
            return self._runtime

        _ensure_cave_agent()

        checker = None
        if self._security_rules:
            checker = _SecurityChecker(self._security_rules)

        if self._backend == "ipykernel":
            IPyKernelRuntime = _get_ipykernel_runtime()
            self._runtime = IPyKernelRuntime(security_checker=checker)
            await self._runtime._executor.start()
        else:
            self._runtime = _IPythonRuntime(security_checker=checker)

        # Pre-inject configured objects
        for item in self._inject_variables:
            self._runtime.inject_variable(
                _Variable(item["name"], item["value"], item.get("description", ""))
            )
        for item in self._inject_functions:
            self._runtime.inject_function(
                _Function(item["func"], description=item.get("description", ""))
            )
        for item in self._inject_types:
            self._runtime.inject_type(_Type(item["cls"], description=item.get("description", "")))

        self._started = True
        logger.info("PythonRuntimeTool initialized (backend={})", self._backend)
        return self._runtime

    async def reset(self) -> None:
        """Reset the runtime, clearing all state."""
        if self._runtime is not None:
            await self._runtime.reset()
            logger.info("PythonRuntimeTool reset")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, code: str, **kwargs: Any) -> str:
        """Execute Python code in the stateful runtime.

        Args:
            code: Python code to execute.

        Returns:
            stdout output or error message.
        """
        runtime = await self._ensure_runtime()

        try:
            result = await runtime.execute(code)
        except Exception as exc:
            # If kernel crashed, attempt reset and report
            logger.warning("PythonRuntime execution error: {}", exc)
            return self._format_error(exc)

        if result.success:
            output = result.stdout or "(no output)"
        else:
            output = self._format_error(result.error)

        if len(output) > self._max_output:
            half = self._max_output // 2
            output = (
                output[:half]
                + f"\n\n... ({len(output) - self._max_output:,} chars truncated) ...\n\n"
                + output[-half:]
            )

        return output

    # ------------------------------------------------------------------
    # Namespace management (public API for nanobot integration)
    # ------------------------------------------------------------------

    def inject_variable(self, name: str, value: Any, description: str = "") -> None:
        """Inject a Python object into the runtime namespace.

        The object will be available as a global variable in subsequent code.
        If the runtime hasn't started yet, the injection is deferred.
        """
        if self._runtime is not None:
            self._runtime.inject_variable(_Variable(name, value, description))
        else:
            self._inject_variables.append(
                {"name": name, "value": value, "description": description}
            )

    def inject_function(self, func: Any, description: str = "") -> None:
        """Inject a callable into the runtime namespace."""
        if self._runtime is not None:
            self._runtime.inject_function(_Function(func, description=description))
        else:
            self._inject_functions.append({"func": func, "description": description})

    def inject_type(self, cls: type, description: str = "") -> None:
        """Inject a type/class into the runtime namespace."""
        if self._runtime is not None:
            self._runtime.inject_type(_Type(cls, description=description))
        else:
            self._inject_types.append({"cls": cls, "description": description})

    async def retrieve(self, name: str) -> Any:
        """Retrieve a variable value from the namespace."""
        runtime = await self._ensure_runtime()
        return await runtime.retrieve(name)

    def describe_namespace(self) -> str:
        """Generate a text description of the namespace for system prompt injection.

        Returns a formatted string listing available functions, variables, and types.
        """
        if self._runtime is None:
            return self._describe_pending_injections()
        return self._describe_active_runtime()

    def _describe_pending_injections(self) -> str:
        """Describe objects that will be injected on first use."""
        parts = []
        if self._inject_functions:
            lines = []
            for item in self._inject_functions:
                func = item["func"]
                desc = item.get("description", "") or (func.__doc__ or "").strip().split("\n")[0]
                sig = getattr(func, "__annotations__", {})
                params = ", ".join(
                    f"{k}: {v.__name__ if hasattr(v, '__name__') else v}"
                    for k, v in sig.items()
                    if k != "return"
                )
                lines.append(f"  {func.__name__}({params}) — {desc}")
            parts.append("Functions:\n" + "\n".join(lines))
        if self._inject_variables:
            lines = []
            for item in self._inject_variables:
                vtype = type(item["value"]).__name__
                desc = item.get("description", "")
                lines.append(f"  {item['name']} ({vtype}): {desc}")
            parts.append("Variables:\n" + "\n".join(lines))
        if self._inject_types:
            lines = []
            for item in self._inject_types:
                desc = item.get("description", "")
                lines.append(f"  {item['cls'].__name__}: {desc}")
            parts.append("Types:\n" + "\n".join(lines))
        return "\n\n".join(parts) if parts else "Empty namespace"

    def _describe_active_runtime(self) -> str:
        """Describe objects currently in the active runtime."""
        parts = []
        funcs = self._runtime.describe_functions()
        vars_ = self._runtime.describe_variables()
        types = self._runtime.describe_types()
        if funcs and "No functions" not in funcs:
            parts.append(f"Functions:\n{funcs}")
        if vars_ and "No variables" not in vars_:
            parts.append(f"Variables:\n{vars_}")
        if types and "No types" not in types:
            parts.append(f"Types:\n{types}")
        return "\n\n".join(parts) if parts else "Empty namespace"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_error(self, exc: BaseException | None) -> str:
        """Format an error for LLM consumption."""
        if exc is None:
            return "Error: unknown execution failure"
        name = type(exc).__name__
        msg = str(exc)
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_str = "".join(tb)
        if len(tb_str) > self._MAX_TRACEBACK:
            tb_str = tb_str[: self._MAX_TRACEBACK] + "\n... (traceback truncated)"
        return f"Error: {name}: {msg}\n\nTraceback:\n{tb_str}"
