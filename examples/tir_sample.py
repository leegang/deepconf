"""
Minimal example demonstrating usage of the TIR ToolManager with PythonExecTool.
Run: python3 examples/tir_sample.py
"""

from deepconf.tir import ToolManager, PythonExecTool
import os


def main():
    tool_manager = ToolManager()
    tool_manager.register(PythonExecTool(cpu_time_limit_s=2, memory_limit_bytes=100 * 1024 * 1024))

    # A simple planner that instructs a python_exec tool call
    action = {
        "type": "tool_call",
        "tool_name": "python_exec",
        "args": {"code": "print(sum([i for i in range(10)]))"},
    }

    # Execute
    if action["type"] == "tool_call":
        res = tool_manager.execute(action["tool_name"], action.get("args", {}))
        print("Tool execution success:", res.success)
        print("stdout:\n", res.output.get("stdout") if res.output else None)
        print("stderr:\n", res.output.get("stderr") if res.output else None)


if __name__ == "__main__":
    main()
