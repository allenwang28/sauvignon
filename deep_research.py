"""Deep Research proof of concept.

This module implements a distributed RL workflow that enables agents to perform research
tasks involving web browsing and code execution. The system orchestrates multiple components:

1. Generators: Policy models that produce actions based on the current state
2. Browsers: Agents that can navigate and extract information from the web
3. Code Executors: Agents that can write and execute code
4. Scorers: Models that evaluate the quality of completed tasks
5. Learner: Updates policy weights based on collected experiences

The system uses a loop-based approach where control flow, queues, and threading are
exposed as first-class concerns. This design allows for fine-grained control over
the orchestration of components while enabling complex multi-turn interactions.

Key components:
- Request routing system that directs tasks to appropriate executors
- Priority queues that manage task scheduling based on importance
- Distributed policy store for weight synchronization across generators
- Multi-turn interaction flow where agents can perform sequences of actions

"""

import pprint

# pyre-unsafe
import queue
import random
import threading
import time
from dataclasses import dataclass

from deep_research_utils import (
    Browser,
    CodeExecutor,
    DeepResearchGenerator,
    DeepResearchRequest,
    Environment,
    Learner,
    PolicyStore,
    PriorityQueue,
    Scorer,
)
from monarch.proc_mesh import proc_mesh


@dataclass
class DeepResearchConfig:
    # Entity knobs
    num_generators: int = 2
    num_scorers: int = 4
    num_browsers: int = 1
    num_coders: int = 1

    # Trainer knobs
    num_train_steps: int = 5
    train_batch_size: int = 4
    num_envs: int = 4

    # Parallelism knobs
    gp: int = 4  # generator parallelism
    lp: int = 8  # learner parallelism
    sp: int = 2  # scorer parallelism


config = DeepResearchConfig()


# Main control loops
def prompt_generator(
    generator_queues: list[PriorityQueue], stop_event: threading.Event
):
    """
    Generates initial research requests and distributes them to generator queues.

    This function serves as the entry point for new tasks in the system. It:
    1. Creates DeepResearchRequest objects with initial prompts
    2. Distributes them across generator queues in round-robin fashion
    3. Sets appropriate priority based on the turn number (0 for initial prompts)

    In a production system, this would typically pull from a dataset, user input,
    or task queue rather than generating synthetic prompts.

    Args:
        generator_queues: List of priority queues for each generator
        stop_event: Threading event to signal when to stop generating prompts
    """
    prompt_id = 0
    queue_id = 0

    while not stop_event.is_set():
        try:
            # Here, you would replace prompt with an actual torch data loader
            # or some other text iterator...
            request = DeepResearchRequest(initial_prompt=str(prompt_id), latest_turn=0)
            generator_queue = generator_queues[queue_id]

            queue_id += 1
            if queue_id == config.num_generators:
                queue_id = 0

            print(
                "[PromptGenerator] putting request {} to generator {}".format(
                    request, queue_id
                )
            )
            try:
                generator_queue.put(request, priority=request.latest_turn, timeout=0.5)
                prompt_id += 1
            except queue.Full:
                print("[PromptGenerator] Queue is full, retrying...")

        except Exception as e:
            print(f"Prompt generator error: {e}")
            break

    print("[PromptGenerator] Shutting down.")


def generate_loop(
    generator_id: int,
    generator: DeepResearchGenerator,
    generator_queue: queue.Queue,
    executor_queue: queue.Queue,
    store: PolicyStore,
    stop_event: threading.Event,
):
    """
    Main loop for policy generation.

    This function manages a single generator instance that:
    1. Synchronizes with the latest policy weights from the store
    2. Pulls prompts from its dedicated queue
    3. Generates responses based on those prompts
    4. Sends generations to be scored

    The generator handles both initial prompts and multi-turn interactions where
    previous responses are fed back after scoring for continued conversation.

    Args:
        generator_id: Unique identifier for this generator
        generator: The DeepResearchGenerator instance that produces responses
        generator_queue: Queue containing prompts to process
        executor_queue: Queue to put generated responses into
        store: PolicyStore for synchronizing model weights
        stop_event: Signal to terminate the loop
    """
    version = None
    print(f"[Generator-{generator_id}] Starting loop")
    while store.get_latest_version() is None and not stop_event.is_set():
        print(f"[Generator-{generator_id}] Waiting to sync weights...")
        time.sleep(1)

    while not stop_event.is_set():
        latest_version = store.get_latest_version()
        if latest_version != version:
            print(f"[Generator-{generator_id}] Updating weights to {latest_version}")
            generator.update_weights.call(
                weights=store.get_latest_weights(), version=latest_version
            ).get()
            version = latest_version

        try:
            data: DeepResearchRequest = generator_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        time.sleep(random.uniform(0.5, 1))  # simulate inference
        generation = generator.generate.call(data).get()
        # generation returns a ValueMesh which we need to gather
        # imagine that Monarch adds an API for this later, like generation.to_tensor()
        # creates a tensor of size [num_generators, ...]
        generation = generation._values[0]
        print(
            f"[Generator-{generator_id}] generated {generation} with version {version} on turn {generation.latest_turn}."
        )
        try:
            executor_queue.put(generation, timeout=0.5)
        except queue.Full:
            print(f"[Generator-{generator_id}] Generation queue is full, retrying...")

    print(f"[Generator-{generator_id}] Shutting down.")


def router_loop(
    executor_queue: queue.Queue,
    browser_queues: list[queue.Queue],
    coder_queues: list[queue.Queue],
    scorer_queue: queue.Queue,
    stop_event: threading.Event,
):
    """
    Routes generated actions to appropriate executor queues.

    This function:
    1. Takes actions from the executor queue
    2. Determines the appropriate destination based on action type
    3. Routes to browser, coder, or scorer queues accordingly
    4. Maintains round-robin distribution for load balancing

    Args:
        executor_queue: Queue containing actions from generators
        browser_queues: List of queues for browser executors
        coder_queues: List of queues for code executors
        scorer_queue: Queue for completed tasks ready for scoring
        stop_event: Signal to terminate the loop
    """
    coder_id = 0
    browser_id = 0
    while not stop_event.is_set():
        try:
            data = executor_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        dest = None
        if data.action == "stop":
            print(f"[Router] Sending {data} to scorer queue")
            dest = scorer_queue
        elif data.action == "code":
            print(f"[Router] Sending {data} to coder queue")
            if data.coder_id:
                dest = coder_queues[data.coder_id]
            else:
                dest = coder_queues[coder_id]
                coder_id += 1
                if coder_id == config.num_coders:
                    coder_id = 0
        elif data.action == "browse":
            print(f"[Router] Sending {data} to browser queue")
            if data.browser_id:
                dest = browser_queues[data.browser_id]
            else:
                dest = browser_queues[browser_id]
                browser_id += 1
                if browser_id == config.num_browsers:
                    browser_id = 0
        else:
            raise ValueError(f"Unknown action: {data.action}")

        try:
            dest.put(data, timeout=0.5)
        except queue.Full:
            print("[Router] Browser queue is full, retrying...")


def env_loop(
    env: Environment,
    env_queue: queue.Queue,
    generation_queues: list[PriorityQueue],
    stop_event: threading.Event,
):
    """
    Executes environment actions and return results to generators.

    This function:
    1. Takes browsing requests from its queue
    2. Executes the browsing action using the browser agent
    3. Returns results back to the appropriate generator queue

    Args:
        browser: The Browser instance that performs web actions
        browser_queue: Queue containing browsing requests
        generation_queues: List of queues to return results to generators
        stop_event: Signal to terminate the loop
    """
    while not stop_event.is_set():
        try:
            data: DeepResearchRequest = env_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        data = env.step.call_one(data).get()
        try:
            generation_queues[data.generator_id].put(
                data, priority=data.latest_turn, timeout=0.5
            )
        except queue.Full:
            print("[Router] Browser queue is full, retrying...")


def score_loop(
    scorer_id: int,
    scorer: Scorer,
    scorer_queue: queue.Queue,
    experience_queue: queue.Queue,
    stop_event: threading.Event,
):
    """
    Main loop for scoring generated responses.

    This function:
    1. Takes generations from the generation queue
    2. Scores them using the scorer model
    3. Either:
       a. Sends the scored experience to the learner if we've reached the max turns
       b. Returns the scored response to the generator for continued interaction

    This is a critical component for multi-turn RL as it determines whether to continue
    a conversation or consider it complete and ready for learning.

    Args:
        scorer_id: Unique identifier for this scorer
        scorer: The Scorer instance that evaluates responses
        scorer_queue: Queue containing completed tasks to score
        experience_queue: Queue to put completed experiences into for training
        stop_event: Signal to terminate the loop
    """
    while not stop_event.is_set():
        try:
            data: DeepResearchRequest = scorer_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        scored = scorer.score.call(data).get()
        # again doing a gather here, but imagine Monarch provides primitives for this.
        scored = scored._values[0]
        try:
            print(f"[Scorer-{scorer_id}] Scored {scored}, pushing to learner")
            experience_queue.put(scored, timeout=0.5)
        except queue.Full:
            print(f"[Scorer-{scorer_id}] Experience queue is full, retrying...")

    print(f"[Scorer-{scorer_id}] Shutting down.")


def train_loop(
    experience_queue: queue.Queue,
    learner: Learner,
    store: PolicyStore,
    stop_event: threading.Event,
):
    """
    Main loop for training the policy model.

    This function:
    1. Publishes initial policy weights to the store
    2. Collects batches of experiences from the experience queue
    3. Trains the policy on these experiences
    4. Publishes updated weights for generators to use
    5. Tracks staleness of experiences (how many versions old they are)

    The training loop continues until the configured number of training steps
    is reached, at which point it signals the system to shut down.

    Args:
        experience_queue: Queue containing scored experiences for training
        learner: The Learner instance that updates the policy
        store: PolicyStore for publishing updated model weights
        stop_event: Signal to terminate all system loops when training completes
    """
    for step in range(config.num_train_steps):
        weights = learner.get_weights.call().get()
        # again doing a gather here, but imagine Monarch provides primitives for this.
        weights = weights._values[0]
        store.publish_weights(
            version=step,
            weights=weights,
        )
        print(f"[Trainer] Published weights for version {step}")
        batch = []
        stalenesses = []
        while len(batch) < config.train_batch_size:
            data = experience_queue.get()
            batch.append(data)
            staleness = step - sum(data.policies) / len(data.policies)
            stalenesses.append(staleness)
        learner.step.call(batch).get()
        time.sleep(0.2)
        print(
            f"[Trainer,step={step}] Trained policy {step+1} on batch ({pprint.pformat(batch)}). Staleness: {stalenesses}"
        )
    print("[Trainer] Finished training. Shutting down.")
    stop_event.set()


def main():
    print("Config: {}".format(config))

    # We could possibly push all of this into an "Orchestrator" class too,
    # but this is a later implementation detail.
    generation_procs = [
        proc_mesh(gpus=config.gp).get() for _ in range(config.num_generators)
    ]
    scorer_procs = [proc_mesh(gpus=config.sp).get() for _ in range(config.num_scorers)]
    learn_proc = proc_mesh(gpus=config.lp).get()
    browser_procs = [proc_mesh(gpus=1).get() for _ in range(config.num_browsers)]
    coder_procs = [proc_mesh(gpus=1).get() for _ in range(config.num_coders)]

    generators = [
        p.spawn("generator", DeepResearchGenerator).get() for p in generation_procs
    ]
    scorers = [p.spawn("scorer", Scorer).get() for p in scorer_procs]
    learner = learn_proc.spawn("learner", Learner).get()
    browsers = [p.spawn("browser", Browser).get() for p in browser_procs]
    coders = [p.spawn("coder", CodeExecutor).get() for p in coder_procs]

    # probably worth introducing some kind of load balance primitives...
    generator_queues = [PriorityQueue(maxsize=4) for _ in range(config.num_generators)]
    experience_queue = queue.Queue()
    executor_queue = queue.Queue()
    browser_queues = [queue.Queue() for _ in range(config.num_browsers)]
    coder_queues = [queue.Queue() for _ in range(config.num_coders)]
    scorer_queue = queue.Queue()
    store = PolicyStore()
    stop_event = threading.Event()

    threads = (
        [
            threading.Thread(
                target=prompt_generator, args=(generator_queues, stop_event)
            ),
            threading.Thread(
                target=train_loop,
                args=(experience_queue, learner, store, stop_event),
            ),
        ]
        + [
            threading.Thread(
                target=score_loop,
                args=(
                    i,
                    scorers[i],
                    scorer_queue,
                    experience_queue,
                    stop_event,
                ),
            )
            for i in range(config.num_scorers)
        ]
        + [
            threading.Thread(
                target=generate_loop,
                args=(
                    i,
                    generators[i],
                    generator_queues[i],
                    executor_queue,
                    store,
                    stop_event,
                ),
            )
            for i in range(config.num_generators)
        ]
        + [
            threading.Thread(
                target=router_loop,
                args=(
                    executor_queue,
                    browser_queues,
                    coder_queues,
                    scorer_queue,
                    stop_event,
                ),
            )
        ]
        + [
            threading.Thread(
                target=env_loop,
                args=(
                    browsers[i],
                    browser_queues[i],
                    generator_queues,
                    stop_event,
                ),
            )
            for i in range(config.num_browsers)
        ]
        + [
            threading.Thread(
                target=env_loop,
                args=(
                    coders[i],
                    coder_queues[i],
                    generator_queues,
                    stop_event,
                ),
            )
            for i in range(config.num_coders)
        ]
    )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("[System] all tasks completed and shut down.")


if __name__ == "__main__":
    main()
