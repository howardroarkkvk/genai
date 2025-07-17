

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
import time
from datetime import datetime,timedelta
from pathlib import Path
import os


class MyPatternMatchingEventHandler(PatternMatchingEventHandler):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.last_modified=datetime.now()

    def on_created(self,event):
        if event.is_directory:
            print(f"Directory got created : {event.src_path}")
        else:
            print(f"File is created : {event.src_path}")

    def on_deleted(self,event):
        print(f"file got deleted {event.src_path}")

    def on_modified(self,event):
        if datetime.now()-self.last_modified>timedelta(seconds=1):
            return
        self.last_modified=datetime.now()
        print(f"File modified is :{event.src_path}")

    def on_moved(self, event):
        print(f"moved from {event.src_path} to destination is : {event.dest_path}")
        return super().on_moved(event)
    

class Watcher:

    def __init__(self,watch_dir,event_handler):
        self.watch_dir=watch_dir
        self.event_handler=event_handler
        self.observer=Observer()

    def run(self):
        print("Watchdog observation started >>>")
        self.observer.schedule(event_handler=self.event_handler,path=self.watch_dir,recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Key board interupption happend")
        finally:
            self.observer.stop()
            self.observer.join()
            print("End of Observations >>>")



if __name__=='__main__':
    dir=r'D:\genai'
    print(dir)
    # Path(dir).resolve().parent/"9_ai_tools_partA"
    watch_dir=os.path.join(dir,'9_ai_tools_partA')
    print(watch_dir)
    event_handler=MyPatternMatchingEventHandler(patterns=['*.py'],ignore_patterns=["tmp"],ignore_directories=True)
    watcher=Watcher(watch_dir=watch_dir,event_handler=event_handler)
    watcher.run()



        


    
