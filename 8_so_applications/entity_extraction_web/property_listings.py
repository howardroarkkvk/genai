import datetime
from typing import List 
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from textwrap import dedent
import logfire
from pydantic_ai.models.gemini import GeminiModel
from html_to_markdown import convert_to_markdown
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


load_dotenv(override=True)
logfire.configure(token = os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()


system_prompt = """
        You are an intelligent research agent. 
        Analyze user request carefully and provide structured responses.
    """




def scrape_and_convert_to_md(url):
    with sync_playwright() as playwright:
        browser=playwright.firefox.launch(headless=True)
        context=browser.new_context(viewport={"width":1920,"height":1080})
        page=context.new_page()
        page.goto(url)
        content=page.content()
        soup=BeautifulSoup(content,'lxml')
        md=convert_to_markdown(str(soup))
    return md


class Address(BaseModel):
    street:str
    city:str
    state:str
    zip_code:str

class PropertyFeatures(BaseModel):
    bedrooms:int
    bathrooms:int
    square_footage:float
    lot_size:float


class AdditionalInfo(BaseModel):
    price:float
    listing_agent:str
    last_updated:datetime.date

class Property(BaseModel):
    address:Address
    info:AdditionalInfo
    type:str
    mls_id:int
    features:PropertyFeatures
    garage_spaces:int



gemini_model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])

url='https://www.redfin.com/VA/Sterling/47516-Anchorage-Cir-20165/home/11931811'
x=scrape_and_convert_to_md(url) 
agent=Agent(model=gemini_model,result_type=Property,system_prompt=system_prompt)
resp=agent.run_sync(x)
print(resp.data)