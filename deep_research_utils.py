import heapq
import queue
import random
import threading
from dataclasses import dataclass

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


_ACTIONS = ["browse", "code", "stop"]


@dataclass
class DeepResearchRequest:
    initial_prompt: str
    events: list[str] = None
    policies: list[int] = None
    latest_turn: int = 0
    action: str | None = None  # see _ACTIONS
    # How do we support arbitrary tools / actions?
    coder_id: int | None = None
    browser_id: int | None = None
    browser_id: int | None = None
    generator_id: int | None = None

    processed_browser_requests: int = 0
    processed_code_requests: int = 0
    score: int = 0

    def __post_init__(self):
        if self.events is None:
            self.events = []
        if self.policies is None:
            self.policies = []


# Entities and their verbs
class DeepResearchGenerator(Actor):
    def __init__(self):
        self.weights = None
        self.version = None
        self.rank = current_rank()["gpus"]

    @endpoint
    def generate(self, data: DeepResearchRequest) -> DeepResearchRequest:
        action = random.choice(_ACTIONS)
        data.action = action
        data.events.append(f"generation(weights={self.weights},action={action})")
        data.policies.append(self.version)
        data.generator_id = self.rank
        data.latest_turn += 1
        return data

    @endpoint
    def update_weights(self, weights, version):
        self.weights = weights
        self.version = version


class Learner(Actor):
    def __init__(self):
        self.weights = 0
        self.rank = current_rank()["gpus"]

    @endpoint
    def step(self, batch):
        self.weights += 1

    @endpoint
    def get_weights(self):
        return self.weights


class Browser(Actor):
    def __init__(self):
        self.rank = current_rank()["gpus"]

    @endpoint
    def step(self, data: DeepResearchRequest) -> DeepResearchRequest:
        data.events.append(f"browser({self.rank},{data.action})")
        data.processed_browser_requests += 1
        data.browser_id = self.rank
        return data


class CodeExecutor(Actor):
    def __init__(self):
        self.rank = current_rank()["gpus"]

    @endpoint
    def step(self, data: DeepResearchRequest) -> DeepResearchRequest:
        data.events.append(f"coder({self.rank})-call-{data.action}")
        data.processed_code_requests += 1
        data.coder_id = self.rank
        return data


class Scorer(Actor):
    def __init__(self):
        self.rank = current_rank()["gpus"]

    @endpoint
    def score(self, data: DeepResearchRequest) -> DeepResearchRequest:
        data.score = self.rank
        return data
