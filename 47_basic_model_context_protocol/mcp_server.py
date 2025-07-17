# mcp dev mcp_server.py:mcp_server This is for stdio... No need to run this in CLI
# python mcp_server.py is for sse... --. where as before running mcp_agent_sse.py, we have to start the server in cli by using the command python mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP(name="Greet MCP Server")



@mcp_server.tool()
async def greet(name: str) -> str:
    """
    Greet the user with the given name.

    Args:
        name (str): The name of the user to greet

    Returns:
        (str): The final greeting message
    """

    return f"Hello {name}. Welcome to MCP World!"

def run_server():
    mcp_server.run(transport='stdio')#


if __name__ == "__main__":
    run_server()