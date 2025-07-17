from typing import List
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
import os
import base64
from textwrap import dedent
import logfire
from pydantic_ai.models.gemini import GeminiModel


load_dotenv(override=True)
logfire.configure(token = os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()


class Address(BaseModel):
    name:str=Field(description='the name of the person and organization')
    address_line:str=Field(description='the local delivery infomration such as street, building number,PO Box or apartment portion of the postal address')
    city:str=Field(description='the city portion of the address')
    state_province_code:str=Field(description='the code for address us states')
    postal_code:int=Field(description='the postal code portion of the address')

class Product(BaseModel):
    product_description:str=Field(description='the description of the product or service')
    count:int=Field(description='the number of units bought for the product')
    unit_item_price:float=Field(description='price per unit')
    product_total_price:float=Field(description='the total prices , number of units * unit_price')

class TotalBill(BaseModel):
    total:float=Field(description='the total amount before tax , discount and delivery')
    discount_amount:float=Field(description='discount amount is total amount * discount%')
    tax_amount:float=Field(description='tax amount is tax_percentage * (total - discount_amount). If discount_amount is 0, then its tax_percentage * total')
    delivery_charges:float=Field(description='the cost of shipping the products')
    final_total:float=Field(description='the total price or balance after removing tax, adding delivery and tax from total')


class Invoice(BaseModel):
    invoice_number:str=Field(description='extraction of relevant info from the invoice')
    billing_address:Address=Field(description='where the bill for the product or service is sent to so that it is paid by the customer')
    product:List[Product]=Field(description='the different product details present in the bill')
    total_bill:TotalBill=Field(description='the details of total amount , discount and tax')


class ImageLoaderBase64:
    
    def __init__(self,user_prompt:str,image_file_path:str):
        # print(image_file_path)
        # print(user_prompt)
        with open(image_file_path,"rb") as image_file:
            base64_image=base64.b64encode(image_file.read()).decode("utf-8")

        # self.encoded_message_with_image=[{"type":"text","text":f'{user_prompt}'},
                                        #  {"type":'image_url','image_url':{"url":f'data:image/png;base64,{base64_image}',"detail":"high"}}]
        
        self.encoded_message_with_image = [{"type": "text", "text": f"{user_prompt}"},{"type": "image_url","image_url": {
                    "url": f"data:image/png;base64,{base64_image}","detail": "high"
                }
            }
        ]


# gemini_model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])
model=OpenAIModel(model_name=os.getenv("OLLAMA_MODEL"), base_url="http://localhost:11434/v1")
system_prompt = """
    You are an assistant that extracts details from invoices.
"""
agent=Agent(model=model,result_type=Invoice,system_prompt=dedent(system_prompt))

image_path_1=os.path.expanduser(r"~\Downloads\invoice.png")
print(image_path_1)
user_prompt_1='extract total amount for the provided image'

image_req_1=ImageLoaderBase64(user_prompt_1,image_path_1)
# print(image_req_1.encoded_message_with_image)
resp=agent.run_sync(image_req_1.encoded_message_with_image)
print(resp.data)