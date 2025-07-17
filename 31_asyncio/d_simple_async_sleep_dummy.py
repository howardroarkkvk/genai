from asyncio import events
import asyncio
import time




async def add():
    tasks=[asyncio.sleep(i) for i in range(10)]
    print('tasks in add func',tasks)
    await asyncio.gather(*tasks)




def run(fn):
    loop=events.new_event_loop()
    loop.set_debug(True)
    print(loop.get_debug())
    print(fn)
    loop.run_until_complete(fn)
    loop.close()



st=time.perf_counter()
run(add())
et=time.perf_counter()
print(f'Elapsed time is : {et-st:.6f}')