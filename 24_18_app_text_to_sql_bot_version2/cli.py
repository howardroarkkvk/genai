

from prompts import get_prompt
from rich.console import Console
from dotenv import load_dotenv
import logfire
import os
from rag_1 import TexttoSqlRAG

def main():

    src_dir=r'D:\DataFiles\text_to_sql'
    kb_dir='kb'
    vector_db_dir='index'
    llm_name="groq:llama-3.3-70b-versatile"
    system_prompt=get_prompt()
    top_k=2
    embedding_model_name=os.getenv('HF_EMBEDDINGS_MODEL')
    rag=TexttoSqlRAG(src_dir,kb_dir,vector_db_dir,embedding_model_name,top_k,llm_name,system_prompt)
    load_dotenv(override=True)
    logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
    logfire.instrument_openai()
    console=Console()
    console.print("Welcome to TextToSql Bot.  How can i help you.",
        style="cyan",
        end="\n\n",)
    
    while True:
        user_input=input('>>')
        if user_input=='q':
            break
        console.print()
        # context = rag.retreiveContext(user_input)
        # console.print(context, style="green", end="\n\n")
        answer = rag.execute(user_input)
        console.print(answer, style="cyan", end="\n\n")
 



if __name__=='__main__':
    main()


# How many albums are there?
# How many distinct genres are there?
# What is the total number of invoice lines?
