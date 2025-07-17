from dotenv import load_dotenv
from openai import OpenAI
import os

# load_dotenv is function which loads all the environment variables present in the .env file for the session and override value is Ture implies that
# if some env variable changes this override will replace the value of changed variable to new value as we have provided override as True
load_dotenv(override=True)

# client instance we have created using open ai key, this is like the api call we make to the open ai model...using the api key
# get_env is function in os package which takes one of the parameter which is key in string format and returns the value as string
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# once the connection is established in the previous step, using the model and by sending messages, create will create the model response.
completion=client.chat.completions.create(
    model=os.getenv("OPENAI_CHAT_MODEL"),#
                               messages=[
                                {"role":"system","content":"You are a helpful assistant"},
                                         {"role":"user","content":"what is value investing?"}]

                                )

# completion contains chocices which has message to be displayed.
print(completion.choices[0].message.content)





# print(completion.choices[0].message)
# print(completion.choices[0].message.content)

# completion=client.chat.completions.create(model=os.getenv("OPENAI_CHAT_MODEL"),
#                                messages=[{"role":"system","content":"You are a helpful assistant"},
#                                          {"role":"user","content":"what is recurssion in programming"}]

#                                 )
# print(completion.choices[0].message)
# print(completion.choices[0].message.content)