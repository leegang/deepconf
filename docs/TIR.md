---

## Tool-Integrated Reasoning (TIR) - Preview

This repository includes an experimental preview of Tool-Integrated Reasoning (TIR) to allow the agent to call external tools during multi-step reasoning sessions.

Files added in branch `feat/tir-multistep`:
- `deepconf/tir.py` - Tool abstraction, ToolManager, and a minimal PythonExecTool that runs code inside a temporary directory with basic resource limits (Unix-only best-effort sandbox).
- `examples/tir_sample.py` - Minimal example showing how to register the PythonExecTool and execute a simple tool call.
- `tests/test_tir.py` - Basic tests for success and timeout behavior.

Usage notes and safety:
- TIR is disabled by default; production use should run Python execution inside hardened sandboxes or containers (Docker, Firecracker, gVisor, etc.).
- The provided PythonExecTool uses `resource.setrlimit` on Unix-like systems and a temporary working directory, but it is not sufficient for untrusted code.

---

Please review the files and let me know if you want the PR to include more tools, stricter sandboxing (containerized), or integration directly into the agent main loop.
