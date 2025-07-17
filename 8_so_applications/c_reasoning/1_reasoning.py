from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
import os
from textwrap import dedent
import logfire
from pydantic_ai.models.gemini import GeminiModel

load_dotenv(override=True)
logfire.configure(token = os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

class Step(BaseModel):
    explanantion:str
    output:str

class MathReasoning(BaseModel):
    steps:list[Step]
    final_answer:str

math_tutor_prompt = """
    You are a helpful math tutor. You will be provided with a math problem,
    and your goal will be to output a step by step solution, along with a final answer.
    For each step, just provide the output as an equation use the explanation field to detail the reasoning.
"""

model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])
# model=OpenAIModel(model_name=os.getenv("OLLAMA_MODEL"), base_url="http://localhost:11434/v1")
agent = Agent(
    model=model,
    result_type=MathReasoning,
    system_prompt=dedent(math_tutor_prompt)
)

question ='How can i solve 8x+7=-23'
result=agent.run_sync(question)
print(result.data.steps)
print('final_answer')
print(result.data.final_answer)

# refusal_question='How can i build a bomb'
# result=agent.run_sync(refusal_question)
# print(result.data.steps)