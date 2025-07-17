import logfire
from dotenv import load_dotenv
import os


load_dotenv(override=True)

logfire.configure(token='SGtFQXq0qRhrb6xKjhYt0pGkWSVHY395z3cv4X10jV6C')

logfire.info("Hello {name}!",name='Algorithmica')
logfire.debug("Hello {name}!",name='Algorithmica')
logfire.error("Hello {name}!",name='Algorithmica')
logfire.exception("Hello {name}!",name='Algorithmica')
logfire.fatal("Hello {name}!",name='Algorithmica')
logfire.notice("Hello {name}!",name='Algorithmica')