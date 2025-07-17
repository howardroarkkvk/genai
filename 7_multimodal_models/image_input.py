import base64
from pydantic import BaseModel,Field
from typing import List
from dotenv import load_dotenv
import logfire
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from textwrap import dedent

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()


class ImageLoaderBase64:
    

    def __init__(self,user_prompt:str,image_file_path:str):
        with open(image_file_path,'rb') as image_file:
            base64_image=base64.b64encode(image_file.read()).decode("utf-8")
            # print('this is base64 image ',base64_image)
        
        self.encoded_message_With_image=[{"type":"text","text":f"{user_prompt}"},
                                         {"type":"image_url","image_url":{"url":f"data:image/png;base64,{base64_image}"}}]
        #"detail": "high"

open_model=OpenAIModel(model_name=os.environ['OPENAI_REASONING_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))
# open_model=OpenAIModel(model_name=os.environ['GEMINI_CHAT_MODEL'],api_key=os.getenv('GEMINI_API_KEY'))


system_prompt= """ You are  smart vision agent with  text extraction skills """

agent=Agent(model=open_model,system_prompt=dedent(system_prompt))

user_prompt="Extract the text from image"
image_file_path=os.path.expanduser(r'~\Downloads\All-I-want-to-know-is-where-i-am-going-to-die.png')
print(image_file_path)
image_request_1=ImageLoaderBase64(user_prompt,image_file_path)
# print(image_request_1.encoded_message_With_image)

resp=agent.run_sync(image_request_1.encoded_message_With_image)
print(resp.data)
