import time
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from pathlib import Path
import time

# MyEventHandler is inherited from FileSystemEventHandler and on_any_event is the method, which we can override while using it....
class MyEventHandler(FileSystemEventHandler):
    def on_any_event(self,event):
        print(event)

observer=Observer()

event_handler=MyEventHandler()
path=r'D:\watchdog'
watch_dog_dir=Path(path)
print(watch_dog_dir)
observer.schedule(event_handler,watch_dog_dir,recursive=True)
observer.start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Key board interupped")
    observer.stop()
    observer.join()
finally:
    observer.stop()
    observer.join()

