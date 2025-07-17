from dotenv import load_dotenv
import logfire
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console


load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()
model = model=OpenAIModel(model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
agent = Agent(model=model,system_prompt="You are helpful assistant.")

def main():
    console=Console()
    console.print("Welcome to vamshi's chat bot how may i assist you today", style='cyan', end ='/n/n')
    messages=[]
    while True:
        input_message=input(" Please enter the user prompt as part of input message:>>")
        if input_message=='q':
            print(messages)
            print(result.all_messages())
            break
        console.print()
        result=agent.run_sync(input_message,message_history=messages)
        console.print(result.data,style='cyan',end='\n\n')
        print(result.new_messages())
        messages+=result.new_messages()
        print(result.usage())
        # print(result.new_messages())
        

if __name__ == '__main__':
    main()



