class Observable():
    def __init__(self, events):
        self._events = {event: set() for event in events}

    def subscribe(self, event, observer):
        self._events[event].add(observer)

    def unsubscribe(self, event, observer):
        self._events[event].remove(observer)

    def dispatch(self, event, *args):
        for observer in self._events[event]:
            observer(*args)
