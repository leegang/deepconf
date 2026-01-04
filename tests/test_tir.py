import time
from deepconf.tir import ToolManager, PythonExecTool


def test_python_exec_success():
    tm = ToolManager()
    tm.register(PythonExecTool(cpu_time_limit_s=2, memory_limit_bytes=100 * 1024 * 1024))
    res = tm.execute("python_exec", {"code": "print(1+2)"})
    assert res.success is True
    assert "3" in res.output.get("stdout", "")


def test_python_exec_timeout():
    tm = ToolManager()
    # small timeout to force a timeout
    tm.register(PythonExecTool(cpu_time_limit_s=1, memory_limit_bytes=100 * 1024 * 1024))
    res = tm.execute("python_exec", {"code": "import time\ntime.sleep(3)\nprint(1)"})
    assert res.success is False
    assert res.error == "Timeout"
