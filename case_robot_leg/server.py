"""
robot_server.py - 机器人腿部结构优化 MCP Server
================================================
与 src/mymcp/server.py（Maxwell Server）完全对称，
只注册 robot_leg 的两个工具。
"""
import asyncio
import io
import sys
import os

# server.py lives in case_robot_leg/.  Add ../src to sys.path so `mymcp` resolves.
_src_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from mcp.server import FastMCP
from mymcp.tool.robot_leg import validate_robot_design, run_robot_simulation
from mymcp.mcp_adapter import MCPOpenAIAdapter


class _JSONRPCStdoutFilter:
    """stdout 过滤器：只允许 JSONRPC 消息通过，其余重定向到 stderr。"""

    def __init__(self, original_stdout, fallback_stream):
        self._original = original_stdout
        self._fallback = fallback_stream
        self._buffer = getattr(original_stdout, "buffer", None)

    @property
    def buffer(self):
        return self._buffer

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", "strict")

    @property
    def newlines(self):
        return getattr(self._original, "newlines", None)

    def write(self, data: str) -> int:
        if not data:
            return 0
        if data.strip().startswith("{"):
            return self._original.write(data)
        return len(data)

    def flush(self):
        self._original.flush()
        self._fallback.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._original, name)


async def main():
    original_stdout = sys.stdout
    sys.stdout = _JSONRPCStdoutFilter(original_stdout, sys.stderr)

    server = FastMCP("robot_leg", log_level="DEBUG")

    server.add_tool(
        validate_robot_design,
        name="validate_robot_design",
        description=(
            "检查一组机器人腿部设计参数是否满足全部物理约束（7 类约束）。"
            "参数：m(前腿长,米), n(后腿长,米), alpha(后脚角,度), beta(前脚角,度), DIST_BETTERY(电池距离,米)。"
        ),
    )
    server.add_tool(
        run_robot_simulation,
        name="run_robot_simulation",
        description=(
            "运行 SolidWorks + Adams 仿真流水线，返回机器人位移、速度等性能指标和 fitness。"
            "参数：m(前腿长,米), n(后腿长,米), alpha(后脚角,度), beta(前脚角,度), DIST_BETTERY(电池距离,米)。"
            "fitness 越小越好（= -位移X + 5*振幅Y）。"
        ),
    )

    adapter = MCPOpenAIAdapter()
    tools = await server.list_tools()
    print("Robot Leg MCP 工具列表：", file=sys.stderr)
    print(tools, file=sys.stderr)

    tool_schemas = adapter.convert_to_tool_schema(tools)
    print("\n工具 schema：", file=sys.stderr)
    print(tool_schemas, file=sys.stderr)

    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
