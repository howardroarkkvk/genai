from pydantic import BaseModel
from enum import Enum
from typing import List
from datetime import datetime,timedelta

# contains class which helps in manuipulating data just like we do it in a database....
class OrderStatus(str,Enum):
    PENDING='pending'
    CONFIRMED='confirmed'
    SHIPPED='shipped'
    DELIVERED='delivered'
    CANCELLED='cancelled'

class ReturnReason(str,Enum):
    WRONG_SIZE='wrong_size'
    WRONG_COLOR='wrong_color'
    NOT_AS_DESCRIBED='not_as_described'
    CHANGED_MIND='changed_mind'
    DAMAGED='damaged'


class EscalationReason(str,Enum):
    COMPLEX_REQUEST='complex_request'
    CUSTOMER_DISSATISFIED='customer_dissatisfied'
    CUSTOMER_REQUEST='customer_request'
    CANNOT_RESOLVE_SITUATION='cannot_resolve_situation'


class OrderItem(BaseModel):
    product_code:str
    name:str
    size:str
    color:str
    quantity:int
    price:float

class Address(BaseModel):
    street:str
    city:str
    postal_code:str
    country:str

class Order(BaseModel):
    order_id:str
    status:OrderStatus
    items:List[OrderItem]
    total_amount:float
    created_at:datetime
    shipping_address:Address

    @property
    def can_modify(self):
        return self.status in [OrderStatus.PENDING,OrderStatus.CONFIRMED]
    

    @property
    def can_return(self):
        if self.status!=OrderStatus.DELIVERED:
            return False
        elif ((datetime.now()-self.created_at).days)<=30:
            return True
        else:
            return False

class OrderDBService:
    def __init__(self):
        self.orders={'001':Order(order_id='001',status=OrderStatus.CONFIRMED,items=[OrderItem(product_code='LJ001',name='Classic Noir Biker Jacker',size='M',color='Black',quantity=1,price=1499)],total_amount=1499,created_at=datetime.now()-timedelta(seconds=30),shipping_address=Address(street="123 Avenue des Champs-Élysées",city="Paris",postal_code="75008",country="France")),
                     '002':Order(order_id='002',status=OrderStatus.PENDING,items=[OrderItem(product_code='BG001',name='Leather Tote',size='ONE',color='Black',quantity=1,price=899)],total_amount=899,created_at=datetime.now()-timedelta(days=15),shipping_address=Address(street="10 Fifth Avenue",city="New York",postal_code="10011",country="USA"))}
        


    # we are updating order based on the order id , if order id is present in the list of orders the db has, it will change the status of the order
    def update_order_status(self,order_id:str,status:OrderStatus):
        if order_id in self.orders.keys():
            self.orders[order_id].status=status
            return True
        return False
    
    def update_shipping_address(self,order_id:str,new_address:Address):
        order=self.orders.get(order_id)
        if order and order.can_modify:
            self.orders[order_id].shipping_address=new_address
            return True
        return False
    
    def get_order(self,order_id:str):
        return self.orders[order_id] # self.orders.get(order_id)
    

    def get_orders(self):
        return self.orders
    
if __name__=='__main__':
    service=OrderDBService()
    # print(service.get_orders())
    print(service.get_order(order_id='001').can_modify)
    print(service.get_order(order_id='002').can_return)
    # print(service.update_order_status('001',OrderStatus.SHIPPED))
    # print(service.get_orders())
    # print(service.update_order_status('003',OrderStatus.PENDING))
    # print(service.get_orders())
    # print(service.update_shipping_address('002',Address(
    #             street="789 Broadway",
    #             city="Los Angeles",
    #             postal_code="90001",
    #             country="USA",
    #         )))
    
    # print(service.get_orders())
    # print(service.get_order('001'))





