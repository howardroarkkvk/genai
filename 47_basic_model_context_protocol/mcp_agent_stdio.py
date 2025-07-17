from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import os
import time
from dotenv import load_dotenv
import logfire
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import asyncio
from openai import AsyncOpenAI

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_pydantic_ai()
logfire.instrument_openai()

greet_mcp_server_stdio = MCPServerStdio(
    command="python",
    args=["D:/genai/47_model_context_protocol/mcp_server.py"],
)
print(os.environ["GITHUB_TOKEN"])
print(os.environ["GITHUB_MODEL"])
client = AsyncOpenAI(
    api_key=os.environ['GITHUB_TOKEN'], base_url="https://models.github.ai/inference"
)
#os.getenv("GITHUB_MODEL")
agent = Agent(
    model=OpenAIModel(
        model_name=os.environ["GITHUB_MODEL"],
        provider=OpenAIProvider(openai_client=client),
    ),
    mcp_servers=[greet_mcp_server_stdio],
)


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run("greet the user Algorithmica")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())