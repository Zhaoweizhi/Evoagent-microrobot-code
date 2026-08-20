import asyncio
import io
import sys
import os
from typing import Callable, Optional

# 确保能找到 mymcp 模块
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from mcp.server import FastMCP

from mymcp.tool import (validate_maxwell_design,
                        run_maxwell_simulation)
from mymcp.mcp_adapter import MCPOpenAIAdapter


class _JSONRPCStdoutFilter(io.TextIOWrapper):
    """
    stdout 过滤器：只允许以 '{' 开头的行（JSONRPC 消息）通过，
    其余内容（如 PyAEDT INFO/WARNING）重定向到 stderr。
    """

    def __init__(self, original_stdout, fallback_stream):
        # 获取原始 stdout 的 buffer
        self._original = original_stdout
        self._fallback = fallback_stream
        self._buffer = getattr(original_stdout, 'buffer', None)

    @property
    def buffer(self):
        return self._buffer

    @property
    def encoding(self):
        return getattr(self._original, 'encoding', 'utf-8')

    @property
    def errors(self):
        return getattr(self._original, 'errors', 'strict')

    @property
    def newlines(self):
        return getattr(self._original, 'newlines', None)

    def write(self, data: str) -> int:
        if not data:
            return 0
        # 只有以 '{' 开头的行才是 JSONRPC 消息，其余重定向到 stderr
        if data.strip().startswith('{'):
            return self._original.write(data)
        else:
            # 非 JSON 内容写到 stderr（静默丢弃或输出到 stderr）
            # 这里选择静默丢弃，避免刷屏
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
        pass  # 不关闭原始流

    def __getattr__(self, name):
        # 代理其他属性到原始 stdout
        return getattr(self._original, name)


class MCPServer:
    """MCP server"""

    def __init__(self, name: str = "demo", log_level: str = "DEBUG"):
        self.server = FastMCP(name=name, log_level=log_level)

    def register_tool(self,
                      tool: Callable,
                      name: Optional[str] = None,
                      description: Optional[str] = None):
        """注册工具
        
        Args:
            tool (Callable): 工具函数
            name (str, optional): 工具名称，可选
            description (str, optional): 工具描述，可选
        """
        self.server.add_tool(tool, name=name, description=description)

    async def run(self, transport: Optional[str] = "stdio"):
        """运行MCP服务器
        
        Args:
            transport (str, optional): 传输方式，可选值为 "stdio" 或 "sse"，默认值为 "stdio"。
        """
        if transport == "stdio":
            await self.server.run_stdio_async()
        else:
            raise ValueError(f"Unsupported transport: {transport}")


async def main():
    # 在启动服务器前，用过滤器替换 sys.stdout，防止 PyAEDT 输出污染 JSONRPC 通道
    original_stdout = sys.stdout
    sys.stdout = _JSONRPCStdoutFilter(original_stdout, sys.stderr)

    mymcp = MCPServer()
    mymcp.register_tool(validate_maxwell_design,
                        name="validate_maxwell_design",
                        description="检查一组 Maxwell 几何参数是否满足全部前置约束。")
    mymcp.register_tool(run_maxwell_simulation,
                        name="run_maxwell_simulation",
                        description="调用 PyAEDT/Maxwell 跑一次仿真，返回 fitness 及关键指标。")

    mcp_adapter = MCPOpenAIAdapter()
    tools = await mymcp.server.list_tools()
    print("MCP工具列表：", file=sys.stderr)
    print(tools, file=sys.stderr)

    tool_schemas = mcp_adapter.convert_to_tool_schema(tools)
    print("\n工具schema：", file=sys.stderr)
    print(tool_schemas, file=sys.stderr)

    # 运行MCP服务器
    await mymcp.run(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
