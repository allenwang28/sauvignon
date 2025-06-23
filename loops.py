"""
Expresses the skeleton of an asynchronous RL loop.

This module demonstrates an asynchronous RL workflow using a
"loop-based" approach. It showcases control flow, queues, and threading to orchestrate
a distributed RL system where orchestration and control flow are exposed as first-class
concerns directly to the user.

The system consists of:
1. Multiple generators (policies) that produce responses based on prompts
2. Multiple scorers that evaluate generations and provide feedback
3. A learner that trains on experiences collected from the system
4. A multi-turn setup where generators can have multiple interactions with the environment

The way a user is intended to interact with this system is by starting with toy entities
that implement the above components to get a sense of the control flow.


Key components:
- Prompt queue: Manages raw text data and allows policies to interact with prior steps
- Generation queue: Collects outputs from generators to be scored
- Experience queue: Collects scored experiences for training
- PolicyStore: Manages versioned policy weights for synchronization
"""

# pyre-unsafe
import queue
import random
import threading
import time
from dataclasses import dataclass

from monarch.actor_mesh import ValueMesh
from monarch.proc_mesh import proc_mesh
from utils import Generator, Learner, PolicyStore, PriorityQueue, Scorer


@dataclass
class Config:
    # Throughput knobs
    num_generators: int = 2
    num_scorers: int = 4

    # Environment
    num_multi_turn_steps: int = 2

    # Trainer knobs
    num_train_steps: int = 5
    train_batch_size: int = 2

    # Parallelism knobs
    gp: int = 1  # generator parallelism
    lp: int = 1  # learner parallelism
    sp: int = 1  # scorer parallelism


config = Config()


# Main control loops
def prompt_generator(prompt_queues: list[PriorityQueue], stop_event: threading.Event):
    """
    Generates prompts and puts them into the prompt queues.

    This function continuously generates prompts and distributes them across
    the available prompt queues in a round-robin fashion. Each prompt is assigned
    a unique ID and is placed in the queue with a priority of 0 (lowest priority).
    Higher priority values (e.g., 10) indicate tasks that should be processed first.

    In a real implementation, this would typically pull from a dataset or environment
    and feed initial prompts into the system to start the RL loop.

    Args:
        prompt_queues: A list of PriorityQueue objects to distribute prompts to
        stop_event: A threading.Event that signals when to stop generating prompts
    """
    prompt_id = 0
    queue_id = 0

    while not stop_event.is_set():
        try:
            # Here, you would replace prompt with an actual torch data loader
            # or some other text iterator...
            prompt = str(prompt_id)
            prompt_queue = prompt_queues[queue_id]

            queue_id += 1
            if queue_id == config.num_generators:
                queue_id = 0

            print(
                "[PromptGenerator] putting prompt {} to generator {}".format(
                    prompt, queue_id
                )
            )
            try:
                prompt_queue.put((prompt, 0), priority=0, timeout=0.5)
                prompt_id += 1
            except queue.Full:
                print("[PromptGenerator] Queue is full, retrying...")

        except Exception as e:
            print(f"Prompt generator error: {e}")
            break

    print("[PromptGenerator] Shutting down.")


def generate_loop(
    generator_id: int,
    generator: Generator,
    prompt_queue: queue.Queue,
    generation_queue: queue.Queue,
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
        generator: The Generator instance that produces responses
        prompt_queue: Queue containing prompts to process
        generation_queue: Queue to put generated responses into
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
            print(
                f"[Generator-{generator_id}] Updating weights to {store.get_latest_weights()}"
            )
            generator.update_weights.call(store.get_latest_weights()).get()
            version = latest_version

        try:
            prompt, turn_number = prompt_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        time.sleep(random.uniform(0.5, 1))  # simulate inference

        generation = generator.generate.call(prompt).get()
        # generation returns a ValueMesh which we need to gather
        # imagine that Monarch adds an API for this later, like generation.to_tensor()
        # creates a tensor of size [num_generators, ...]
        generation = "".join([g for g in generation._values])
        print(
            f"[Generator-{generator_id}] generated {generation} with version {version} on turn {turn_number}."
        )
        try:
            generation_queue.put(
                (generator_id, generation, version, turn_number), timeout=0.5
            )
        except queue.Full:
            print(f"[Generator-{generator_id}] Generation queue is full, retrying...")

    print(f"[Generator-{generator_id}] Shutting down.")


def score_loop(
    scorer_id: int,
    scorer: Scorer,
    generation_queue: queue.Queue,
    prompt_queues: list[queue.Queue],
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
        generation_queue: Queue containing generations to score
        prompt_queues: List of queues to return scored responses to for continued interaction
        experience_queue: Queue to put completed experiences into for training
        stop_event: Signal to terminate the loop
    """
    while not stop_event.is_set():
        try:
            generation_id, generation, version, turn_number = generation_queue.get(
                timeout=0.5
            )
        except queue.Empty:
            continue

        scored = scorer.score.call(generation).get()
        # again doing a gather here, but imagine Monarch provides primitives for this.
        scored = "".join([s for s in scored._values])
        if turn_number >= config.num_multi_turn_steps:
            try:
                print(f"[Scorer-{scorer_id}] Scored {scored}, pushing to learner")
                experience_queue.put((scored, version), timeout=0.5)
            except queue.Full:
                print(f"[Scorer-{scorer_id}] Experience queue is full, retrying...")
        else:
            try:
                print(f"[Scorer-{scorer_id}] Scored {scored}, returning to generator")
                prompt_queues[generation_id].put(
                    (scored, turn_number + 1), priority=turn_number + 1, timeout=0.5
                )
            except queue.Full:
                print(f"[Scorer-{scorer_id}] Prompt queue is full, retrying...")

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
        weights = [w for w in weights._values]
        store.publish_weights(
            version=step,
            weights=weights,
        )
        print(f"[Trainer] Published weights for version {step}")
        batch = []
        staleness = []
        while len(batch) < config.train_batch_size:
            scored, version = experience_queue.get()
            batch.append(scored)
            staleness.append(step - version)
        learner.step.call(batch).get()
        time.sleep(0.2)
        print(
            f"[Trainer,step={step}] Trained policy {step+1} on batch ({batch}). Staleness: {staleness}"
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
    generators = [p.spawn("generator", Generator).get() for p in generation_procs]
    scorers = [p.spawn("scorer", Scorer).get() for p in scorer_procs]
    learner = learn_proc.spawn("learner", Learner).get()

    prompt_queues = [PriorityQueue(maxsize=8) for _ in range(config.num_generators)]
    generation_queue = queue.Queue()
    experience_queue = queue.Queue()
    store = PolicyStore()
    stop_event = threading.Event()

    threads = (
        [
            threading.Thread(target=prompt_generator, args=(prompt_queues, stop_event)),
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
                    generation_queue,
                    prompt_queues,
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
                    prompt_queues[i],
                    generation_queue,
                    store,
                    stop_event,
                ),
            )
            for i in range(config.num_generators)
        ]
    )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("[System] all tasks completed and shut down.")


if __name__ == "__main__":
    main()


# TODO:
# - replace call_one with call
# - turn a ValueMesh into some type of handle?
# - add sharding
# - show a simple environment abstraction
