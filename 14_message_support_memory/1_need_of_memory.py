from dotenv import load_dotenv
import logfire
import os
from rich.console import Console
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel





load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()
model = model=GroqModel(model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY"))
agent = Agent(model=model,system_prompt="You are helpful assistant.")


def main():
    console= Console(width=15)
    console.log("Welcome to the vamshi's Bot how may i assist you",style='cyan')
    console.print(locals())
    console.print("foo",style='white on blue')
    # console.print("Welcome to vamshi's Bot how may i assist you",style='cyan')
    while True:
        user_message=input("Enter the message: >>")
        if user_message=='q':
            break

        result=agent.run_sync(user_message)
        console.print(result.data,style='white on blue')
            


if __name__=='__main__':
    main()

