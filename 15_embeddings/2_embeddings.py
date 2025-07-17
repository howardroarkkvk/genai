from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import time

load_dotenv(override=True)
time.sleep(2)

embedding_model=OpenAIEmbeddings(model=os.getenv('OPENAI_EMBEDDING_MODEL'),api_key=os.getenv('OPENAI_API_KEY'))
res=embedding_model.embed_query("king")
print(len(res))
print(res)