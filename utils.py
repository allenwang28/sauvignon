import heapq
import queue
import threading

from monarch.actor_mesh import Actor, current_rank, endpoint


class PriorityQueue:
    def __init__(self, maxsize=0):
        self._queue = []
        self._index = 0
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def put(self, item, priority, timeout=None):
        with self._not_full:
            if self._maxsize > 0 and len(self._queue) >= self._maxsize:
                if not self._not_full.wait(timeout=timeout):
                    raise queue.Full("PriorityQueue is full")
            heapq.heappush(self._queue, (-priority, self._index, item))
            self._index += 1
            self._not_empty.notify()

    def get(self, timeout=None):
        with self._not_empty:
            while not self._queue:
                if not self._not_empty.wait(timeout=timeout):
                    raise queue.Empty
            return heapq.heappop(self._queue)[-1]

    def empty(self):
        with self._lock:
            return len(self._queue) == 0

    def full(self):
        with self._lock:
            return 0 < self._maxsize <= len(self._queue)


# Entities and their verbs
class Generator(Actor):
    def __init__(self):
        self.weights = None
        self.rank = current_rank()["gpus"]

    @endpoint
    def generate(self, prompt: str) -> str:
        return f"g[r={self.rank}-w{self.weights}-p{prompt}]"

    @endpoint
    def update_weights(self, weights):
        self.weights = weights


class Learner(Actor):
    def __init__(self):
        self.weights = 0
        self.rank = current_rank()["gpus"]

    @endpoint
    def step(self, batch):
        print("")
        self.weights += 1

    @endpoint
    def get_weights(self):
        return self.weights


class Scorer(Actor):
    def __init__(self):
        self.rank = current_rank()["gpus"]

    @endpoint
    def score(self, generation: str) -> str:
        return f"s(r{self.rank}={generation})"


class PolicyStore:
    def __init__(self):
        self.weights = None
        self.version = 0
        self.lock = threading.Lock()

    def get_latest_version(self) -> int | None:
        with self.lock:
            return self.version

    def get_latest_weights(self):
        with self.lock:
            return self.weights

    def publish_weights(self, version: int, weights):
        with self.lock:
            self.weights = weights
            self.version = version
