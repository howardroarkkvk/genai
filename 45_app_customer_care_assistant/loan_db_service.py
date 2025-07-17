
import asyncio
# Fake database for Loan Data 

class LoanDB:
    """ This is the fake loan database for example purpose.
    In reality,you'd be connecting to an external datbase.
    (e.g postgresql) to manage loan information
    """

    @classmethod
    async def customer_name(cls,*,id:int)->str:
        if id==123:
            return 'John'
        
    @classmethod
    async def loan_status(cls,*,id:int)->str:
        """Fetch the loan status of a customer by ID"""
        if id==123:
            return 'Active'
        elif id==124:
            return 'Paidoff'
        elif id == 125:
            return 'Defaulted'
        else:
            return None
        
    @classmethod
    async def cancel_loan(cls,*,id:int)->str:
        """cancel a loan for a customer"""    
        if id==123:
            """Fake logic for cancelling a loan"""
            return f'Loan for the customer id {id} is cancelled'
        else:
            raise ValueError(f'Customer with ID {id} doesnt have active loan account')
        
    @classmethod
    async def add_loan(cls,*,id:int, amount:float, interest_rate:float)->str:
        if id == 123:
            """Fake logic for adding a loan"""
            return f'Loan amount  of {amount} with an interest rate {interest_rate} is added for an id {id}'
        else:
            raise ValueError(f'Customer id {id} cannot be found to add a loan')
        
    @classmethod
    async def loan_balance(cls,*,id:int)->float|None:
        if id==123:
            return 5000.0
        elif id ==124:
            return 0.0
        else:
            raise ValueError(f'customer with id {id} not found for loan exists')

if __name__=='__main__':
    print(asyncio.run( LoanDB.customer_name(id=123)))


        





