
import asyncio

import time

async def dummy(value):
    await asyncio.sleep(1)
    return value*10


async def dummy_caller():
    total=0    
    tasks=[dummy(i) for i in range(10)]
    x=await asyncio.gather(*tasks)
    print(sum(x))


start_time=time.perf_counter()
asyncio.run(dummy_caller())
end_time=time.perf_counter()
print(f'Total execution time of program is {end_time-start_time:.6f}')



