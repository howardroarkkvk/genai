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






class Concept(BaseModel):
    title:str
    description:str

class ArticleSummary(BaseModel):
    invented_year:int
    summary:str
    inventors:list[str]
    description:str
    concepts:list[Concept]

def get_article_content(path):
    with open(path,'r') as f:
        content=f.read()
    return content

def print_summary(summary):
    print(f'Invented year: {summary.invented_year}\n')
    print(f'summary: {summary.summary}\n')
    print('Inventors:')
    for i in summary.inventors:
        print(f'- {i}')
    print('\nconcepts')
    for c in summary.concepts:
        print(f'- {c.title}: {c.description}')
    print(f'\n Description : {summary.description}')    




summarization_prompt = """
    You will be provided with content from an article about an invention.
    Your goal will be to summarize the article following the schema provided.
"""

model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])
# model=OpenAIModel(model_name=os.getenv("OLLAMA_MODEL"), base_url="http://localhost:11434/v1")
agent = Agent(
    model=model,
    result_type=ArticleSummary,
    system_prompt=dedent(summarization_prompt)
)

path1=r'C:\Users\USER\Downloads\moe.md'
content=get_article_content(path1)
# print(x)
print('what is the type of content',content)
result = agent.run_sync(content)
# print_summary(result.data)
print(result.data.concepts)