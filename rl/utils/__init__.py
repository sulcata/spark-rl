from .circular_queue import CircularQueue
from .observable import Observable
from .transition import Transition

def coerce_callable(x):
    return x if callable(x) else lambda _: x
