import threading
import time
import asyncio


async def task(thread_name,duration):
    st=time.perf_counter()
    print('start Time is :',time.asctime())
    print(f'thread started {thread_name}')
    print(threading.get_ident())
    await asyncio.sleep(duration)
    print(f'thread ended {thread_name}')
    et=time.perf_counter()
    print(f'elapsed time : {et-st}')


# t1=threading.Thread(target=task , args=['thread-1',1])
# t2=threading.Thread(target=task , args=['thread-2',2])
# t3=threading.Thread(target=task , args=['thread-3',3])

# t1.run()
# t2.run()
# t3.run()
async def run_1():
    coroutines_1=[task(str(i),1)  for i in range(10)]
    await asyncio.gather(*coroutines_1)

asyncio.run(run_1())
