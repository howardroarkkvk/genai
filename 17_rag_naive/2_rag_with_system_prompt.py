from pydantic_ai import Agent,RunContext
from pydantic_ai.models.openai import OpenAIModel
from pathlib import Path
import os
from rich.console import Console
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import time

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_openai()



@dataclass
class KnowledgeDeps:
    kb_path:str

system_prompt="""
You are a helpful and knowledgeable assistant for the luxury fashion store H&M.
Your role is to provide detailed information and assistance about the store and its products.

Follow these guidelines:
- ALWAYS search the knowledge base using the search_knowledge_base tool to answer user questions.
- Provide accurate product and policy information based ONLY on the information retrieved from the knowledge base. Never make assumptions or provide information not present in the knowledge base.
- Structure your responses in a clear, concise and professional manner, maintaining our premium brand standards
- Highlight unique features, materials, and care instructions when relevant.
- If information is not found in the knowledge base, politely acknowledge this.
"""

model_name=OpenAIModel(model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
rag_agent=Agent(model=model_name,deps_type=KnowledgeDeps,system_prompt=system_prompt)

@rag_agent.system_prompt
def add_context_system_prompt(ctx:RunContext[KnowledgeDeps])->str:
    kb='<context>/n'
    for file in Path(ctx.deps.kb_path).glob('*.txt'):
        content=file.read_text(encoding='utf-8')
        kb+=content
        kb+='/n'
    kb+='</context>'        
    return kb



def main():
    console=Console()
    console.print("Welcome to Vamshi's programming Infro",style='cyan',end='/n')
    path=os.path.expanduser(r"D:\DataFiles")
    deps=KnowledgeDeps(kb_path=path)
    while True:
        user_input=input("Please enter what you want to search: >>")
        if user_input=='q':
            break
        response=rag_agent.run_sync(user_input,deps=deps)
        console.print(response.data,style='cyan')
 

if __name__=='__main__':
    main()


# How long does international shipping take?
# Show leather jackets
# How should i contact you?