

from dotenv import load_dotenv
from rich.console import Console
import os
from semantic_searcher import SemanticSearcher

def main():
    load_dotenv(override=True)
    main_dir=r'D:\DataFiles'
    print(main_dir)
    vector_db_dir=r'faiss_ss'
    kb_dir=r'kb_json'
    embeddings_model_name=os.getenv('HF_EMBEDDINGS_MODEL')
    print(embeddings_model_name)
    retriever_top_k=2
    ss=SemanticSearcher(main_dir,kb_dir,vector_db_dir,embeddings_model_name,retriever_top_k)
    console=Console()
    console.print("Welcome to SemanticSearch On Claude Documentation.  Ask questions and get semantically closest results.",
        style="cyan",
        end="\n\n",
    )
    while True:
        user_input=input("Please enter the input >>")
        if user_input=='q':
            break
        # result=ss.retriever(query=user_input)
        result=ss.retrievercontent(query=user_input)
        console.print(result,style='cyan',end='\n\n')


if __name__=='__main__':
    main()


# supported llm models
# embedding models
# rate limits of models
# pricing of models
# multi-modal models supported
# free tier models supported
# text classification example