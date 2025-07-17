from llm import *
from pydantic import BaseModel
from typing import Annotated
from enum import Enum
from pydantic_ai import Agent

class RequestClassEnum(str,Enum):
    OR='Order'
    OT='Other'


class RequestClass(BaseModel):
    ticket_class:Annotated[RequestClassEnum,...,'The category of the query']



model=build_model(model_name=settings.llm_gaurd_rails.name,
                  api_key=settings.llm_gaurd_rails.api_key,
                  base_url='',
                  temperature=settings.llm_gaurd_rails.temperature,
                  max_tokens=settings.llm_gaurd_rails.max_tokens
                  )

input_guard_rail_agent=Agent(model=model,system_prompt=settings.llm_gaurd_rails.prompt,output_type=RequestClass)

user_messages = [
    "What is the status of order 001?",
    "What is the status of order 002?",
    "please update the shipping address of order 001: 10 Fifth Avenue, LosAngels, California, 10005, USA",
    "Escalate to human for my order 001 since it is not resolved",
    "Escalate to human for my order 001 since customer is unhappy",
    "how are you?",
    "what do you do?",
]

def main():
    for user_message in user_messages:
        result=input_guard_rail_agent.run_sync(user_message)
        print(result.data.ticket_class.value)

if __name__=='__main__':
    main()
