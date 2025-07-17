from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import logfire
import os

load_dotenv(override=True)

class AddressInfo(BaseModel):
    first_name: str
    last_name: str
    street: str
    house_number: str
    postal_code: str
    city: str
    state: str
    country: str

class Addresses(BaseModel):
    adresses:List[AddressInfo]

openai_model=OpenAIModel(model_name=os.getenv('OPENAI_CHAT_MODEL'),api_key=os.environ['OPENAI_API_KEY'])

agent=Agent(model=openai_model,result_type=Addresses)

resp=agent.run_sync("During my recent travels, I had the pleasure of visiting several fascinating locations. "
    "My journey began at the office of Dr. Elena Martinez, 142B Elm Street, San Francisco, "
    "CA 94107, USA. Her office, nestled in the bustling heart of the city, was a hub of "
    "innovation and creativity. Next, I made my way to the historic residence of Mr. Hans "
    "Gruber located at 3. Stock, Goethestrasse 22, 8001 Zürich, Switzerland. The old building, "
    "with its classic Swiss architecture, stood as a testament to the city’s rich cultural "
    "heritage. My adventure continued at the tranquil countryside home of Satoshi Nakamoto, "
    "2-15-5, Sakura-cho, Musashino-shi, Tokyo-to 180-0003, Japan. Their home was surrounded by "
    "beautiful cherry blossoms, creating a picturesque scene straight out of a postcard. In "
    "Europe, I visited the charming villa of Mme. Catherine Dubois, 15 Rue de la République, "
    "69002 Lyon, France. The cobblestone streets and historic buildings of Lyon provided a "
    "perfect backdrop to her elegant home. Finally, my journey concluded at the modern apartment "
    "of Mr. David Johnson, Apt 7B, 34 Queen Street, Toronto, ON M5H 2Y4, Canada. The sleek "
    "design of the apartment building mirrored the contemporary vibe of the city itself.")

print(resp.data)