import asyncio



class CustomerDB:
    """This is fake database for example purpose
    In reality, you'd be connecting to an external Db
    (e.g. postgres sql) to get the info about the customers
    """

    @classmethod
    async def customer_name(cls,*,id:int):
        if id == 123:
            return 'John'
        return None
    
    @classmethod
    async def customer_balance(cls,*,id:int,include_pending:bool):
        if id == 123:
            return 123.45
        else:
            return ValueError("Customer Not Found")

    



