from dotenv import load_dotenv
import logfire
import os


load_dotenv(override=True)
print(os.getenv("LOGFIRE_TOKEN"))

logfire.configure(token='SGtFQXq0qRhrb6xKjhYt0pGkWSVHY395z3cv4X10jV6C') #

logfire.info("Hello {name}!",name='Algorithmica')