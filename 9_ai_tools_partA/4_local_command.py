from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import os
import logfire
import subprocess

load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()

gemini_model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])

system_prompt=""" You are helpful assistant
use 'who_am_i' tool to find the active user of my system
"""
agent=Agent(model=gemini_model,system_prompt=system_prompt)


@agent.tool_plain
def who_am_i()->str:
    result=subprocess.run("whoami",capture_output=True,text=True)
    # result=subprocess.check_output('whoami',text=True)
    print(result)
    return result.stdout
    # return result

output=agent.run_sync("who is the active user of my system")
print(output.data)