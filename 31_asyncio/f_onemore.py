import asyncio
import time


async def add():
    await asyncio.sleep(1)
    print(time.asctime())
    return 10


async def custom1():
    coroutine_taks=[add() for i in range(10)]
    print(coroutine_taks)
    values=await asyncio.gather(*coroutine_taks)
    print( sum(values))


st=time.perf_counter()
asyncio.run(custom1())
et=time.perf_counter()
print(f'Elasped time is :{et-st}')

