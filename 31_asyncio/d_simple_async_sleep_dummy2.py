from asyncio import events
import asyncio
import time
import threading


async def hello():
    print(threading.get_ident())
    await asyncio.sleep(2)
    

t=threading.Thread(target=asyncio.run(hello()))
t1=threading.Thread(target=asyncio.run(hello()))
print(t)
print(t1)







# st=time.perf_counter()
# run(add())
# et=time.perf_counter()
# print(f'Elapsed time is : {et-st:.6f}')