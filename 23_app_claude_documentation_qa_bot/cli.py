from rich.console import Console
from dotenv import load_dotenv
from prompts import get_prompt
from rag import *
import logfire
import os

def main():
    load_dotenv (override=True)
    logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
    logfire.instrument_openai

    src_dir=r'D:\DataFiles'
    kb_dir=r'kb_json'
    vector_db_dir=r'faiss_qa_bot'
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
    top_k=2
    llm_name='ollama:llama3.2'
    system_prompt=get_prompt()
    print(system_prompt)
    rag=DocumentationRAG(src_dir,kb_dir,vector_db_dir,embedding_model,top_k,llm_name,system_prompt)
    console=Console()

    console.print("Welcome to Claude Documentation Bot.  Ask me anything on claude documentation.",
        style="cyan",
        end="\n\n",)
    
    while True:
        user_input=input(">>")
        if user_input=='q':
            break

        console.print()
        context=rag.retrievecontent(user_input)
        console.print(context,style='green',end='\n\n')

        answer=rag.execute(user_input)
        console.print(answer,style='cyan',end='\n\n')





if __name__=='__main__':
    main()


# supported llm models
# embedding models supported by claude
# rate limits of models
# pricing of models
# multi-modal models supported
# free tier models supported
# text classification example