from typing import List
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from textwrap import dedent
import logfire
from pydantic_ai.models.gemini import GeminiModel
import requests
from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown

load_dotenv(override=True)
logfire.configure(token = os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

def scrape_and_convert_to_md(url):
    content=requests.get(url).text
    soup=BeautifulSoup(content,'lxml')
    md=convert_to_markdown(str(soup))
    return md


class ProjectInformation(BaseModel):
    name:str=Field(description='Name of the project')
    tagline:str=Field(description='what this project is about?')
    benefits:List[str]=Field(description='A list of main benefits of the project including 3-5 words to summarize each one.')


model=OpenAIModel(model_name=os.getenv("OLLAMA_MODEL"), base_url="http://localhost:11434/v1")
system_prompt = """
        You are an intelligent research agent. 
        Analyze user request carefully and provide structured responses.
"""
agent=Agent(model=model,result_type=ProjectInformation,system_prompt=dedent(system_prompt))

url="https://playwright.dev/"
page_content=scrape_and_convert_to_md(url)
resp=agent.run_sync(f"extract the informatin from following web page {page_content}")
print(resp.data)