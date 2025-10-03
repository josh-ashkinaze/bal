"""
equilibrium_min.py

Simple, readable implementation of Equilibrium Neutrality.

- Two agents alternate proposing edits to a statement.
- If an agent's best change is tiny (<= epsilon thresholds), that agent is "satisfied".
- If BOTH are satisfied in the same round, we have equilibrium.
- If a proposal is too large (> max thresholds), it's rejected and the turn cedes.
- Semantic distance = SBERT cosine distance. Text diff = simple word Jaccard.

Requirements:
  pip install sentence-transformers litellm numpy

Environment:
  export OPENAI_API_KEY="sk-..."  # or provider key for your LiteLLM model

Run:
  python equilibrium_min.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from litellm import completion

from dotenv import load_dotenv
import csv
import json
from datetime import datetime

load_dotenv("../src/.env")


# ---------------------------
# Diff metrics (simple, transparent)
# ---------------------------

def text_diff(a: str, b: str) -> float:
    """Word-level Jaccard distance in [0,1].

    Args:
      a: Original text.
      b: Candidate text.

    Returns:
      0 means identical word set, 1 means disjoint word sets.
    """
    A = {t.strip(".,;:!?()[]\"'").lower() for t in a.split() if t.strip()}
    B = {t.strip(".,;:!?()[]\"'").lower() for t in b.split() if t.strip()}
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return 1.0 - (inter / union if union else 1.0)


class SBERT:
    """Tiny wrapper for SBERT embeddings + cosine distance with a small cache."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, np.ndarray] = {}

    def embed(self, s: str) -> np.ndarray:
        """Return normalized embedding, cached."""
        if s in self.cache:
            return self.cache[s]
        vec = self.model.encode([s], normalize_embeddings=True)[0]
        self.cache[s] = vec
        return vec

    def cosine_distance(self, a: str, b: str) -> float:
        """Cosine distance (1 - similarity) in ~[0,1] for normalized vectors."""
        va, vb = self.embed(a), self.embed(b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-8
        cos_sim = float(np.dot(va, vb) / denom)
        # clamp to [-1,1] then convert to distance
        cos_sim = max(min(cos_sim, 1.0), -1.0)
        return 1.0 - cos_sim


# ---------------------------
# Data containers
# ---------------------------

@dataclass
class Thresholds:
    """Satisfaction and anti-troll thresholds."""
    epsilon_text: float = 0.02   # small word change ⇒ satisfied
    epsilon_sem: float = 0.12    # small semantic change ⇒ satisfied
    max_text: float = 0.35       # too large text change ⇒ reject
    max_sem: float = 0.45        # too large semantic change ⇒ reject


@dataclass
class Step:
    """One turn in the game."""
    round_no: int
    mover: str
    applied: bool
    reason: str
    text_diff: float
    sem_diff: float
    statement_after: str
    justification: str


@dataclass
class Result:
    """Final outcome."""
    statement: str
    equilibrium: bool
    rounds: int
    steps: List[Step]
    satisfied: Dict[str, bool]

    def to_json(self) -> str:
        return json.dumps(
            {
                "statement": self.statement,
                "equilibrium": self.equilibrium,
                "rounds": self.rounds,
                "satisfied": self.satisfied,
                "steps": [asdict(s) for s in self.steps],
            },
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------
# Agents
# ---------------------------

class Agent:
    """Base agent. Override `propose` to return (proposal, justification)."""

    def __init__(self, name: str):
        self.name = name

    def propose(self, statement: str, history: List[str]) -> Tuple[str, str]:
        return statement, "(no change)"


class LLMProxyAgent(Agent):
    """LiteLLM-backed agent with a short, straightforward prompt."""

    def __init__(self, name: str, model: str, system_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 300):
        super().__init__(name)
        self.model = model
        self.system_prompt = system_prompt.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _parse_json(s: str) -> Tuple[str, str]:
        """Parse {"proposal": "...", "justification": "..."}; fall back gracefully."""
        s = s.strip()
        if not (s.startswith("{") and s.endswith("}")):
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j > i:
                s = s[i:j + 1]
        try:
            obj = json.loads(s)
            return (obj.get("proposal", "").strip(), obj.get("justification", "").strip())
        except Exception:
            return s, ""

    def propose(self, statement: str, history: List[str]) -> Tuple[str, str]:
        last3 = "\n".join(history[-3:]) if history else "(none)"
        user = (
            "Propose ONE minimally edited version of the statement that preserves factual meaning "
            "and advances your constituency's framing.\n\n"
            f"CURRENT STATEMENT:\n{statement}\n\n"
            f"RECENT HISTORY:\n{last3}\n\n"
            "GUIDANCE:\n- Keep edits small.\n- Do not change factual meaning.\n\n"
            "Return ONLY JSON (no extra text):\n"
            '{ "proposal": "<edited statement>", "justification": "<one sentence reason>" }'
        )
        resp = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            # Keep it simple: ask for a JSON object; no schemas, no retries
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = resp["choices"][0]["message"]["content"]
        proposal, justification = self._parse_json(content)
        # Fallback: if proposal came back empty, do no change.
        return (proposal or statement, justification)


# ---------------------------
# Orchestrator
# ---------------------------

class EquilibriumNeutralizer:
    """Alternating-edit game until equilibrium, cycle, or max rounds."""

    def __init__(self, agents: List[Agent], thresholds: Thresholds,
                 sbert: SBERT, max_rounds: int = 10, order: Tuple[int, int] = (0, 1),
                 detect_cycles: bool = True):
        if len(agents) != 2:
            raise ValueError("This minimal version supports exactly 2 agents.")
        self.agents = agents
        self.t = thresholds
        self.sbert = sbert
        self.max_rounds = max_rounds
        self.order = order
        self.detect_cycles = detect_cycles

    @staticmethod
    def _fp(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def _is_satisfied(self, dt: float, ds: float) -> bool:
        return dt <= self.t.epsilon_text and ds <= self.t.epsilon_sem

    def _over_max(self, dt: float, ds: float) -> bool:
        return dt > self.t.max_text or ds > self.t.max_sem

    def run(self, initial: str) -> Result:
        s = initial
        history = [s]
        seen = {self._fp(s)} if self.detect_cycles else set()
        steps: List[Step] = []
        satisfied = {self.agents[0].name: False, self.agents[1].name: False}

        for r in range(1, self.max_rounds + 1):
            for k in satisfied:
                satisfied[k] = False

            for idx in self.order:
                agent = self.agents[idx]
                proposal, justification = agent.propose(s, history)

                dt = text_diff(s, proposal)
                ds = self.sbert.cosine_distance(s, proposal)

                if self._over_max(dt, ds):
                    steps.append(Step(r, agent.name, False,
                                      "rejected: exceeds max thresholds",
                                      dt, ds, s, justification))
                    continue

                if self._is_satisfied(dt, ds):
                    satisfied[agent.name] = True
                    steps.append(Step(r, agent.name, False,
                                      "satisfied: proposed change <= epsilon",
                                      dt, ds, s, justification))
                    continue

                # Apply acceptable substantive edit
                prev = s
                s = proposal
                history.append(s)
                steps.append(Step(r, agent.name, True, "applied", dt, ds, s, justification))

                # Optional cycle detection
                if self.detect_cycles:
                    fp = self._fp(s)
                    if fp in seen:
                        return Result(prev, False, r, steps, satisfied.copy())
                    seen.add(fp)

            if all(satisfied.values()):
                return Result(s, True, r, steps, satisfied.copy())

        return Result(s, False, self.max_rounds, steps, satisfied.copy())


# ---------------------------
# Minimal demo
# ---------------------------

def _print_result(res: Result) -> None:
    print("\n=== EQUILIBRIUM RESULT ===")
    print(f"Equilibrium: {res.equilibrium} | Rounds: {res.rounds}\n")
    print("Final Statement:\n", res.statement)
    print("\nLog:")
    for st in res.steps:
        tag = "APPLIED" if st.applied else "—"
        print(f"[Round {st.round_no:02d}] {st.mover:12s} {tag:7s} "
              f"(text={st.text_diff:.3f}, sem={st.sem_diff:.3f}) :: {st.reason}")
    print("\nSatisfied:", res.satisfied)


if __name__ == "__main__":
    # Two simple LLM proxies (tweak prompts/models to taste)
    liberal = LLMProxyAgent(
        name="Liberal",
        model="gpt-4o-mini",
        system_prompt="You prefer language from a liberal viewpoint. Keep edits minimal; preserve facts.",
        temperature=0.2,
        max_tokens=300,
    )
    conservative = LLMProxyAgent(
        name="Conservative",
        model="gpt-4o-mini",
        system_prompt="You prefer language from a conservative viewpoint. Keep edits minimal; preserve facts.",
        temperature=0.2,
        max_tokens=300,
    )

    thresholds = Thresholds(
        epsilon_text=0.02,  # <=2% word set change counts as satisfied
        epsilon_sem=0.12,   # SBERT cosine distance threshold
        max_text=0.35,      # >35% text diff is rejected
        max_sem=0.45,       # >45% semantic distance is rejected
    )

    sbert = SBERT("sentence-transformers/all-MiniLM-L6-v2")

    engine = EquilibriumNeutralizer(
        agents=[liberal, conservative],
        thresholds=thresholds,
        sbert=sbert,
        max_rounds=10,
        order=(0, 1),
        detect_cycles=True,
    )

    initial_statement = "Voters should be required to present valid photographic identification verifying their identity in order to vote in any election, subject to exceptions which may be established by law"
    result = engine.run(initial_statement)
    print(result.to_json())
    print(initial_statement)
    _print_result(result)

    with open("equilibrium_log.json", "w", encoding="utf-8") as f:
        f.write(result.to_json())
    print("\nSaved: equilibrium_log.json")
