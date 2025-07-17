from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import os
import time
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import asyncio
from pydantic_ai.mcp import MCPServer
from pydantic_ai.mcp import ToolDefinition

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_pydantic_ai()
logfire.instrument_openai()


class FilteringMCPServer(MCPServerStdio):
    async def list_tools(self) -> list[ToolDefinition]:
        tools = await super().list_tools()
        print([t.name for t in tools])
        filter_value = [
            "list_directory",
            "list_allowed_directories",
            "read_file",
            "read_multiple_files",
        ]
        return [t for t in tools if t.name in filter_value]


fs_server = FilteringMCPServer(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "--mount",
        "type=bind,src=D:/DataFiles/mcp-data,dst=/sample/",
        "mcp/filesystem",
        "/sample",
    ],
)
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_REASONING_MODEL"),
        provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
    ),
    system_prompt="You are an AI agent to perform operations on filesystem using tools.",
    mcp_servers=[fs_server],
)


async def main():
    queries = [
        "list the allowed directories",
        "Read the files and list them.",
        "What is my #1 favorite book by reading the files",
        "Look at my favorite songs by reading files. Suggest one new song that I might like.",
    ]
    history = []
    async with agent.run_mcp_servers():
        for query in queries:
            result = await agent.run(query, message_history=history)
            history = result.all_messages()
            print(result.data)


if __name__ == "__main__":
    asyncio.run(main())