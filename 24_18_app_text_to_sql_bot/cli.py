from rich.console import Console
from dotenv import load_dotenv
from rag import TexttoSqlRAG
from prompts import get_prompt




def main():
    src_path=r'D:\DataFiles\text_to_sql'
    db_path=r'db\chinook.sqlite'
    system_prompt=get_prompt()
    llm_name = "groq:llama-3.3-70b-versatile"
    rag=TexttoSqlRAG(src_dir=src_path,db_path=db_path,llm_name=llm_name,system_prompt=system_prompt)
    console=Console()
    console.print( "Welcome to TextToSql Bot.  How can i help you.",style="cyan",end="\n\n",)
    while True:
        user_input=input('>>')
        if user_input=='q':
            break
        reponse=rag.execute(user_input)
        console.print(reponse,style='cyan',end='\n\n')






if __name__=='__main__':
    main()



# How many employees are there?
# What are the names and salaries of employees in the Marketing department?
# What are the names and hire dates of employees in the Engineering department, ordered by their salary?
# What is the average salary of employees hired in 2022?

# How many albums are there?
# How many distinct genres are there?
# What is the total number of invoice lines?\
# What are the names and hire dates of employees in the marketing department, ordered by their salary?