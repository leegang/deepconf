"""
# Minimal Tool and ToolManager for TIR mode
# 目标：提供一个 PythonExecTool 的最小安全执行实现（非完全安全，生产请使用容器化/专用沙箱）
"""

import os
import tempfile
import subprocess
import shutil
import time
from typing import Any, Dict
import dataclasses

# Only available on Unix-like systems
try:
    import resource
except Exception:
    resource = None

@dataclasses.dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str = ""

class Tool:
    """Abstract Tool: subclass并实现 execute()"""
    name: str = "base_tool"

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError

class ToolManager:
    def __init__(self):
        self.tools = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(False, error=f"Unknown tool: {tool_name}")
        try:
            return self.tools[tool_name].execute(args)
        except Exception as e:
            return ToolResult(False, error=str(e))

# --------------------
# PythonExecTool: 最小沙箱实现
# --------------------
class PythonExecTool(Tool):
    name = "python_exec"

    def __init__(self, cpu_time_limit_s: int = 2, memory_limit_bytes: int = 200 * 1024 * 1024):
        """
        cpu_time_limit_s: CPU seconds limit (RLIMIT_CPU)
        memory_limit_bytes: address space limit (RLIMIT_AS)
        注意：此为最小限制机制，仅在 Unix-like 平台并且 Python 进程拥有相应权限时生效。
        生产请使用容器/进程隔离。
        """
        self.cpu_time_limit_s = int(cpu_time_limit_s)
        self.memory_limit_bytes = int(memory_limit_bytes)

    def _preexec(self):
        # Called in child process (Unix)
        if resource is None:
            return
        try:
            # CPU time (seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_time_limit_s, self.cpu_time_limit_s + 1))
            # Address space / virtual memory
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit_bytes, self.memory_limit_bytes))
            # file size limit (optional small)
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
            # create new session
            os.setsid()
        except Exception:
            # best-effort; if not permitted, continue
            pass

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        code = args.get("code", "")
        if not isinstance(code, str) or code.strip() == "":
            return ToolResult(False, error="No code provided")

        # Run in a temporary working dir to avoid side-effects
        tmpdir = tempfile.mkdtemp(prefix="deepconf_python_exec_")
        try:
            # create a small file to run for nicer tracebacks
            run_file = os.path.join(tmpdir, "run.py")
            with open(run_file, "w", encoding="utf-8") as f:
                f.write(code)

            # Very restrictive environment
            env = {}

            # Use python3 -u to avoid buffering
            cmd = ["python3", "-u", run_file]

            start = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.cpu_time_limit_s + 1,
                    preexec_fn=self._preexec if resource is not None else None,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(False, error="Timeout")
            except Exception as e:
                return ToolResult(False, error=f"Execution failed: {e}")

            elapsed = time.time() - start
            result = {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "time_s": elapsed,
            }
            success = proc.returncode == 0
            return ToolResult(success, output=result, error="" if success else "Non-zero return code")
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
