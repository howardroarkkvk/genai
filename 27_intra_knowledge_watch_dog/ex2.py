from watchdog.events import FileSystemEvent,FileSystemEventHandler
from watchdog.observers import Observer
from datetime import datetime,timedelta
import time
from pathlib import Path

class MyEventHandler(FileSystemEventHandler):
    
    def __init__(self):
        self.last_modified=datetime.now()
        print(self.last_modified)

    
    def on_created(self, event):
        if event.is_directory:
            print("Directory is created")
            print(f"Directory is created: {event.src_path}")
        else:
            print(f"file created: {event.src_path}")

    def on_deleted(self,event):
        if event.is_directory:
            return
        print(f"file deleted : {event.src_path}")
    

    def on_modified(self,event):
        if datetime.now()-self.last_modified<timedelta(seconds=1):
            return
        else:
            self.last_modified=datetime.now()
            print(datetime.now())
            print(f"file modified :{event.src_path}")

    def on_moved(self,event):
        print(f"file moved from :{event.src_path} to {event.dest_path}")



class Watcher:
    def __init__(self,watch_dir,event_handler):
        self.watch_dir=watch_dir
        self.event_handler=event_handler
        self.observer=Observer()

    def run(self):
        print(f"watch running for {self.watch_dir}")
        self.observer.schedule(event_handler=self.event_handler,path=self.watch_dir,recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("key board interuption happend")

        finally:
            self.observer.stop()
            self.observer.join()
            print("watch dog terminated here")


if __name__=='__main__':
    dir=Path('D:/').resolve().parent/"watchdog"
    print(dir)
    eventhandler=MyEventHandler()
    watcher=Watcher(dir,event_handler=eventhandler)
    watcher.run()






        

        