"""Keep empty claim extraction distinct from empty verdict generation (#3629)."""

import asyncio

import pytest

from deepeval.metrics import TurnFaithfulnessMetric
from deepeval.metrics.turn_faithfulness.schema import (
    Claims,
    FaithfulnessScoreReason,
    Truths,
    Verdicts,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


class StageJudge(DeepEvalBaseLLM):
    def __init__(self, stage, json_output):
        self.stage = stage
        self.json_output = json_output
        self.calls = []
        super().__init__()

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self, *args, **kwargs):
        return "extraction-stage-stub"

    def generate(self, prompt, schema=None, **kwargs):
        self.calls.append(schema)
        if schema is Truths:
            result = Truths(truths=["HTTPS uses TLS."])
        elif schema is Claims:
            result = Claims(
                claims=[] if self.stage == "claims" else ["HTTPS uses TLS."]
            )
        elif schema is Verdicts:
            result = Verdicts(
                verdicts=(
                    []
                    if self.stage == "verdicts"
                    else [{"verdict": "yes", "reason": "Supported by context."}]
                )
            )
        elif schema is FaithfulnessScoreReason:
            result = FaithfulnessScoreReason(reason="Stub aggregate reason.")
        else:
            raise AssertionError(f"Unexpected schema: {schema}")
        return result.model_dump_json() if self.json_output else result

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("include_reason", [False, True])
@pytest.mark.parametrize("stage", ["claims", "verdicts", "none"])
def test_faithfulness_reports_the_empty_stage(
    async_mode, json_output, include_reason, stage
):
    judge = StageJudge(stage, json_output)
    metric = TurnFaithfulnessMetric(
        model=judge,
        eval_mode="llm",
        async_mode=async_mode,
        include_reason=include_reason,
    )
    case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="What does HTTPS use?"),
            Turn(
                role="assistant",
                content="HTTPS uses TLS.",
                retrieval_context=["HTTPS uses TLS."],
            ),
        ]
    )
    metric.measure(case, _show_indicator=False)

    # Preserve the PR's policy: this test concerns the diagnosis, not scoring.
    assert metric.score == 1.0
    assert metric.is_successful()
    assert (Verdicts in judge.calls) == (stage != "claims")
    if not include_reason:
        assert metric.reason is None
        assert FaithfulnessScoreReason not in judge.calls
        assert "Reason: None" in metric.verbose_logs
    elif stage == "claims":
        assert "No claims were extracted" in metric.verbose_logs
    elif stage == "verdicts":
        assert "HTTPS uses TLS." in metric.verbose_logs
        assert "No claims were extracted" not in metric.verbose_logs
        assert "no verdicts" in metric.verbose_logs.lower()
    else:
        assert "No claims were extracted" not in metric.verbose_logs


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("include_reason", [False, True])
@pytest.mark.parametrize("claim_count", [None, 0, 1])
def test_empty_verdict_helpers_preserve_claim_count(
    async_mode, include_reason, claim_count
):
    metric = TurnFaithfulnessMetric(
        model=StageJudge("verdicts", False),
        eval_mode="llm",
        include_reason=include_reason,
    )
    # Omit the new argument for callers that don't know the claim count.
    kwargs = {} if claim_count is None else {"claim_count": claim_count}
    if async_mode:
        score, reason = asyncio.run(
            metric._a_get_interaction_score_and_reason([], False, **kwargs)
        )
    else:
        score, reason = metric._get_interaction_score_and_reason(
            [], False, **kwargs
        )
    assert score == 1.0
    if not include_reason:
        assert reason is None
    elif claim_count == 0:
        assert "No claims were extracted" in reason
    else:
        assert "no verdicts" in reason
        assert "No claims were extracted" not in reason
        if claim_count == 1:
            assert "1 extracted claim(s)" in reason
    assert metric.model.calls == []
