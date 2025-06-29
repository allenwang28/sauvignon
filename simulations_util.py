from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DeepResearchRequest:
    initial_prompt: str
    rank: int = 0
    events: List[str] = field(default_factory=list)
    latest_turn: int = 0
    action: Optional[str] = None
    score: int = 0

    def summary(self):
        return {
            "prompt": self.initial_prompt,
            "rank": self.rank,
            "turn": self.latest_turn,
            "events": list(self.events),
            "action": self.action,
            "score": self.score,
        }

@dataclass
class SimulationConfig:
    # Shared config
    num_coding_verifiers: int = 2
    num_math_verifiers: int = 2
    num_llm_judge_judgers: int = 1
    batch_size: int = 4
    max_steps: int = 4
    coding_verifier_step_low: float = 0.2
    coding_verifier_step_high: float = 0.5
    math_verifier_step_low: float = 0.3
    math_verifier_step_high: float = 0.7
    llm_judge_judger_step_low: float = 0.1
    llm_judge_judger_step_high: float = 0.3
    trace_output: str = "trace.json"
    
    # Pipe-specific config
    num_policys: int = 2
    num_preprocess: int = 1
    num_prompt_loaders: int = 1
    policy_step_low: float = 0.5
    policy_step_high: float = 1.0
    preprocess_step_low: float = 0.05
    preprocess_step_high: float = 0.2
    
    # Rollout-specific config
    num_envs: int = 4
