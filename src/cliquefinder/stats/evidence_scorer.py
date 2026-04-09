"""LLM-based evidence quality scorer for INDRA edges.

PROTOTYPE scorer derived from experimental v5_no_grounding variant.
Achieves 73/96 parsed (~76%) accuracy on a 100-record held-out INDRA benchmark
(95% CI overlaps substantially with parametric belief; not statistically
distinguished at current N).

Known limitations:
- Score grid is discrete (6 values: 0.05/0.20/0.35/0.50/0.65/0.80/0.95).
  Apparent "polarization" is partly an artifact of the scoring rubric.
- Parse failures return score=0.5 with verdict=None (not true uncertainty).
- Has not been validated against blinded random-sample evaluation.
- Multi-evidence aggregation strategies untested at scale.
- Not benchmarked against simpler NER/regex alternatives.

Use cases:
- Post-hoc inspection of specific edges flagged by other pipeline steps.
- Debugging why edges pass/fail in discovery.
- NOT as a primary filter until validated.

Usage:
    scorer = EvidenceScorer(model_name="gemma-moe")
    score = scorer.score_evidence(
        subject="MTOR",
        stmt_type="Activation",
        obj="RPS6KB1",
        evidence_text="mTOR phosphorylates and activates S6K1.",
    )
    # → 0.95 (high-confidence correct extraction)

Can be called from discovery_bridge.get_targets() as an optional
evidence-level filter, after provenance-based reliability scoring.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal


# Reuse the validated v5_no_grounding prompt design
SYSTEM_PROMPT = """\
You judge whether a biomedical text-mining extraction is correct. Compare the
claim against the evidence. The examples show the judgment standard through
contrasting pairs — attend to what distinguishes correct from incorrect cases.

Output JSON: {"verdict": "correct" or "incorrect", "confidence": "high" | "medium" | "low"}\
"""


# Contrastive example pairs — same design as v5_no_grounding
CONTRASTIVE_EXAMPLES = [
    # Complex with explicit signal (correct)
    {
        "claim": "Actin [Complex] CDK9",
        "evidence": "Actin was found to interact with Cdk9, a catalytic subunit of P-TEFb, in elongation complexes.",
        "verdict": "correct", "confidence": "high",
        "reason": "Evidence says 'in elongation complexes' — Complex relationship supported.",
    },
    # Complex without signal (incorrect)
    {
        "claim": "AKT [Complex] CASP3",
        "evidence": "Akt and caspase-3 expression interact to regulate proliferation and apoptosis.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Text says 'interact' without complex formation.",
    },
    # Activity (correct)
    {
        "claim": "AXIN [Activation] JNK",
        "evidence": "Dvl and Axin can activate the mitogen-activated protein kinase JNK via distinct mechanisms.",
        "verdict": "correct", "confidence": "high",
        "reason": "Text says 'activate' — activity state change matches Activation.",
    },
    # Expression mis-labeled as Activation (incorrect)
    {
        "claim": "ERK [Activation] MMP2",
        "evidence": "Activation of ERK leads to increases in the expression of matrix metalloproteinase-2.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Text describes expression increase — should be IncreaseAmount, not Activation.",
    },
    # Logical inversion (correct)
    {
        "claim": "AGER [Activation] MMP2",
        "evidence": "RAGE blockade reduced MMP-2 activity to control level.",
        "verdict": "correct", "confidence": "high",
        "reason": "Logical inversion: blocking RAGE reduces MMP-2 activity, so RAGE activates MMP-2.",
    },
    # Second inversion pattern (correct)
    {
        "claim": "TP53 [Inhibition] MDM2",
        "evidence": "TP53 knockdown increased MDM2 protein levels in these cells.",
        "verdict": "correct", "confidence": "high",
        "reason": "Logical inversion: knockdown of TP53 increases MDM2, so TP53 normally decreases MDM2.",
    },
    # Indirect chain (incorrect)
    {
        "claim": "P70S6K [Activation] RPS6",
        "evidence": "Ghrelin strongly activated mTOR, P70S6K, and S6 in parallel.",
        "verdict": "incorrect", "confidence": "medium",
        "reason": "Text shows ghrelin activating multiple targets in parallel, not P70S6K acting on RPS6.",
    },
    # Direct statement (correct)
    {
        "claim": "MTOR [Activation] RPS6KB1",
        "evidence": "mTOR phosphorylates and activates S6K1, leading to increased ribosomal biogenesis.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct activity change: 'mTOR activates S6K1'.",
    },
]


# Parsing
JSON_VERDICT_PATTERN = re.compile(
    r'\{[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?\}',
    re.IGNORECASE,
)


def _render_example(ex: dict) -> tuple[str, str]:
    user = f"CLAIM: {ex['claim']}\nEVIDENCE: \"{ex['evidence']}\""
    assistant = (
        f"Reason: {ex['reason']}\n"
        f'{{"verdict": "{ex["verdict"]}", "confidence": "{ex["confidence"]}"}}'
    )
    return user, assistant


def _extract_verdict(text: str) -> tuple[str | None, str | None]:
    matches = JSON_VERDICT_PATTERN.findall(text)
    if matches:
        v, c = matches[-1]
        return v.lower(), c.lower()
    return None, None


def _verdict_to_score(verdict: str | None, confidence: str | None) -> float:
    if verdict is None:
        return 0.5
    grid = {
        ("correct", "high"): 0.95,
        ("correct", "medium"): 0.80,
        ("correct", "low"): 0.65,
        ("incorrect", "low"): 0.35,
        ("incorrect", "medium"): 0.20,
        ("incorrect", "high"): 0.05,
    }
    return grid.get((verdict, confidence or "medium"), 0.50)


@dataclass
class ScoringResult:
    score: float
    verdict: str | None
    confidence: str | None
    raw_text: str
    tokens: int


class EvidenceScorer:
    """Score biomedical evidence extractions via LLM.

    Designed to be called from pipeline code as an optional evidence-level
    quality filter. Caches scores by (subject, stmt_type, object, evidence_text)
    to avoid redundant LLM calls.
    """

    def __init__(
        self,
        model_name: str = "gemma-moe",
        max_tokens: int = 1500,
    ):
        """Initialize scorer.

        Args:
            model_name: One of the models registered in model_client.LOCAL_MODELS
                        or a Claude model name. Default "gemma-moe" (fast, accurate).
            max_tokens: Generation limit per call.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = None
        self._prebuilt_examples = None

    def _ensure_client(self):
        if self._client is None:
            # Defer import to avoid hard dependency on experimental code
            import sys
            from pathlib import Path
            exp_path = Path(__file__).resolve().parents[3] / "experiments" / "belief_benchmark"
            if str(exp_path) not in sys.path:
                sys.path.insert(0, str(exp_path.parent.parent))
            from experiments.belief_benchmark.model_client import ModelClient
            self._client = ModelClient(self.model_name)

        if self._prebuilt_examples is None:
            self._prebuilt_examples = []
            for ex in CONTRASTIVE_EXAMPLES:
                user, assistant = _render_example(ex)
                self._prebuilt_examples.append({"role": "user", "content": user})
                self._prebuilt_examples.append({"role": "assistant", "content": assistant})

    @lru_cache(maxsize=10000)
    def score_evidence(
        self,
        subject: str,
        stmt_type: str,
        obj: str,
        evidence_text: str,
    ) -> ScoringResult:
        """Score a single evidence mention.

        Returns:
            ScoringResult with score in [0, 1] (higher = more likely correct).

        Cached on (subject, stmt_type, obj, evidence_text) tuple.
        """
        self._ensure_client()

        messages = list(self._prebuilt_examples)
        claim = f"{subject} [{stmt_type}] {obj}"
        messages.append({
            "role": "user",
            "content": f'CLAIM: {claim}\nEVIDENCE: "{evidence_text}"',
        })

        try:
            response = self._client.call(
                system=SYSTEM_PROMPT,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            verdict, confidence = _extract_verdict(response.raw_text)
            return ScoringResult(
                score=_verdict_to_score(verdict, confidence),
                verdict=verdict,
                confidence=confidence,
                raw_text=response.raw_text,
                tokens=response.tokens,
            )
        except Exception as e:
            return ScoringResult(
                score=0.5,
                verdict=None,
                confidence=None,
                raw_text=f"error: {e}",
                tokens=0,
            )

    def score_edge(
        self,
        source: str,
        target: str,
        stmt_type: str,
        evidences: list[dict],
        aggregation: Literal["max", "mean", "any_correct"] = "max",
    ) -> dict:
        """Score an edge by scoring each evidence and aggregating.

        Args:
            source, target, stmt_type: edge claim
            evidences: list of dicts with 'text' field (evidence sentences)
            aggregation:
              "max" — edge score = max evidence score (optimistic)
              "mean" — edge score = mean evidence score (conservative)
              "any_correct" — 1.0 if any evidence scored correct, else max

        Returns:
            {"edge_score": float, "per_evidence": [ScoringResult, ...]}
        """
        results = []
        for ev in evidences:
            text = ev.get("text") or ev.get("evidence_text", "")
            if not text:
                continue
            result = self.score_evidence(source, stmt_type, target, text)
            results.append(result)

        if not results:
            return {"edge_score": 0.5, "per_evidence": [], "n_evidence": 0}

        scores = [r.score for r in results]
        if aggregation == "max":
            edge_score = max(scores)
        elif aggregation == "mean":
            edge_score = sum(scores) / len(scores)
        elif aggregation == "any_correct":
            # Require STRICT > 0.5 to exclude parse failures (score=0.5)
            edge_score = 1.0 if any(s > 0.5 for s in scores) else max(scores)
        else:
            edge_score = max(scores)

        return {
            "edge_score": edge_score,
            "per_evidence": [
                {"score": r.score, "verdict": r.verdict, "confidence": r.confidence}
                for r in results
            ],
            "n_evidence": len(results),
        }
