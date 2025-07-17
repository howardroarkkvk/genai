
import asyncio
import threading
import time

async def dummy(value):
    print(threading.get_ident())
    await asyncio.sleep(1)
    return value*10


async def dummy_caller():
    total=0    
    for i in range(10):
        total+=await dummy(i)
    print(total)

start_time=time.perf_counter()
asyncio.run(dummy_caller())
end_time=time.perf_counter()
print(f'Total execution time of program is {end_time-start_time:.6f}')



