import os
from pathlib import Path
from rich.console import Console
from dataclasses import dataclass
# from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from pydantic_ai import RunContext
from dotenv import load_dotenv
import logfire
import time


# def scrubbing_callback(m: logfire.ScrubMatch):
#     if (
#         m.path == ('attributes', 'all_messages', 2, 'parts', 0, 'content')
#         and m.pattern_match.group(0) == 'Auth'
#     ):
#         return m.value

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))#,scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback)
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

# model_name=GroqModel(model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY"))
model_name=OpenAIModel(model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
rag_agent=Agent(model=model_name,deps_type=KnowledgeDeps,system_prompt=system_prompt)




@rag_agent.tool
def search_knowledge_base(ctx: RunContext[KnowledgeDeps],search_query:str)->str:
    """Search the knowledge base to retrieve information about Harvest & Mill, the store and its products."""
   
    kb=''
    for file in Path(ctx.deps.kb_path).glob('*.txt'):
        content=file.read_text(encoding='utf-8')
        kb+=content
        kb+='\n'
        print(kb)
    return kb
    


def main():
    console=Console()
    path1=os.path.expanduser(r'D:\DataFiles')
    deps=KnowledgeDeps(kb_path=path1)
    console.print("Welcome to Vamshi's programming course:",style='cyan',end='\n')
    while True:
        user_message=input('>>')
        if user_message=='q':
            break
        result=rag_agent.run_sync(user_message,deps=deps)
        console.print(result.data,style='cyan')




if __name__=='__main__':
    main()


# How long does international shipping take?
# Show leather jackets