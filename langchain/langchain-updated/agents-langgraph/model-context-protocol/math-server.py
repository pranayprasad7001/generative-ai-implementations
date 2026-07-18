from mcp.server.fastmcp import FastMCP

math = FastMCP("math")

@math.tool(name="add", description="Add two numbers")
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@math.tool(name="subtract", description="Subtract two numbers")
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

@math.tool(name="multiply", description="Multiply two numbers")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    math.run(transport="stdio")

# To test this stdio server directly via terminal, run:
# echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 1, "b": 2}}}' | python mathserver.py