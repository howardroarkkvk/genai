
from enum import Enum
from pydantic import BaseModel
from datetime import datetime
class OrderStatus(str,Enum):
    PENDING='pending'
    CONFIRMED='confirmed'
    SHIPPED='shipped'
    DELIVERED='delivered'
    CANCELLED='cancelled'

class Order(BaseModel):
    id:int
    status:OrderStatus

def create_order(order:Order):
    return {'order_id':order.id,'stuats':order.status.value}

order1=Order(id=101,status=OrderStatus.PENDING)
print(order1)

print(create_order(order1))

if OrderStatus.PENDING=='pending':
    print(True)

print(datetime.now())
print(order1.id)
print(order1.status)

# enums are defined as name value parirs, they are accessed using the <classname>.<enum_name>.<name> or <classname>.<enum_name>.<value>
# print(list(OrderStatus))
# print(OrderStatus('pending'))
# print(OrderStatus.PENDING.value)
# print(OrderStatus.PENDING.name)
# print(OrderStatus.PENDING)
# print(OrderStatus.PENDING.lower())
# print(OrderStatus('pending'))
# print(OrderStatus['PENDING'])
