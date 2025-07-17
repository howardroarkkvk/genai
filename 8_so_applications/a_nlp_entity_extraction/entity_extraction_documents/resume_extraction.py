from typing import List
from datetime import date
from pydantic import BaseModel,Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire
import os
from dotenv import load_dotenv
#pip install -qU langchain_community pypdf
from langchain_community.document_loaders.pdf import PyPDFLoader

load_dotenv(override=True)
logfire.configure(token = os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

class Job(BaseModel):
      
    """ A job or position extracted from resume """

    position:str=Field(...,description="name of the position")
    company:str = Field(...,description='Company Name')
    start_date: str = Field(None, description="Start date of the job")
    end_date: str = Field(None, description="End date of the job or 'Present'")
    top_keywords:List[str]=Field(None,description ="List of max. top 10 keywords, skills and technologies used for the job",)


class Degree(BaseModel):

    """ Degree or other type of education extracted from the resume """
    name: str = Field(..., description="Name of the degree and field of study")
    institution: str = Field(None, description="University name")
    start_date: str = Field(None, description="Start date of the studies")
    end_date: str = Field(None, description="End date of the studies")

class Resume(BaseModel):
    """Resume data extracted from the resume"""

    name: str = Field(..., description="The name of the person")
    email: str = Field(None, description="Email address of the person")
    phone: str = Field(None, description="Phone number of the person")
    location: str = Field(None, description="Current residence of the person")
    websites: str = Field(None, description="Website like LinkedIn, GitHub, Behance, etc.")
    work_experience: List[Job] = None
    education: List[Degree] = None
    skills: List[str] = Field(None, description="List of core skills and technologies")
    languages: List[str] = Field( None, description="List of languages spoken by the person")
    hobbies: List[str] = Field(None, description="Hobbies and interests of the person")


def get_text_from_pdf(path):
    pdf_loader=PyPDFLoader(path)
    docs=pdf_loader.load()
    return docs[0].page_content


model=OpenAIModel(model_name=os.getenv('OPENAI_CHAT_MODEL'),api_key=os.environ['OPENAI_API_KEY'])
agent=Agent(model=model,result_type=Resume)

path=r'C:\Users\USER\Documents\PL\resume.pdf'
resume_text=get_text_from_pdf(path)
print('resume_text',resume_text)
msg=agent.run_sync(resume_text)
print(msg.data)


