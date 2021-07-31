import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

if __name__ == "__main__":
    my_event_handler = PatternMatchingEventHandler(['*.avi'], None, True, True)

    def on_created(event):
        print(f"hey, {event.src_path} has been created!")

    my_event_handler.on_created = on_created

    my_observer = Observer()
    my_observer.schedule(my_event_handler, "video/swings/", recursive=False)

    my_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()

