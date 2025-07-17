from asyncio import events
import asyncio
import time
import threading


async def dummy(value):
    print('thread',threading.get_ident())
    await asyncio.sleep(1)
    return value * 10


async def approach():
    async_tasks=[dummy(i) for i in range(10)]
    print(async_tasks)
    values=await asyncio.gather(*async_tasks)
    print(values)

st=time.perf_counter()
asyncio.run(approach())
et=time.perf_counter()
print(f'Time Elapsed : {et-st:.6f}')

    









# st=time.perf_counter()
# run(add())
# et=time.perf_counter()
# print(f'Elapsed time is : {et-st:.6f}')