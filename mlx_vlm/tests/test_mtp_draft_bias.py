"""Static constraints apply to draft seeds and successors without target mutation."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.sample_utils import make_logits_processors
from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel
from mlx_vlm.speculative.mtp import _DraftBiasSampler


def bias(token):
    return make_logits_processors(logit_bias={token: -10000.0})


@pytest.mark.parametrize("greedy", [True, False])
def test_seed_and_successors_exclude_suppressed_token(greedy):
    # The raw head always prefers EOS=2. Both the cached seed and every
    # subsequent proposal must instead choose token 1.
    sampler = _DraftBiasSampler(lambda x: mx.argmax(x, axis=-1), [bias(2)])
    drafter = SimpleNamespace(
        _lm_head_fn=lambda h: mx.array([[[0.0, 5.0, 10.0]]]),
        _input_embed=object(),
        _forward_token=lambda tok, hidden, dtype: hidden,
        _draft_round=0,
        _greedy_token=lambda h: mx.array([[2]]),
    )
    hidden = mx.zeros((1, 1, 3))
    Qwen3_5MTPDraftModel._set_seed_from_hidden(drafter, hidden, sampler, greedy)
    tokens = Qwen3_5MTPDraftModel.draft_block(
        drafter, 0, hidden, None, 4, sampler, greedy=greedy
    )
    assert tokens.tolist() == [[1, 1, 1]]


def test_row_filter_and_rowwise_drafting_keep_request_bias():
    sampler = _DraftBiasSampler(lambda x: mx.argmax(x, axis=-1), [bias(2), bias(1)])
    logits = mx.array([[[0.0, 5.0, 10.0]], [[0.0, 10.0, 5.0]]])
    assert sampler.sample_draft(logits, greedy=True).tolist() == [[1], [2]]
    assert sampler.select_rows([1]).sample_draft(logits[1:], greedy=True).tolist() == [
        [2]
    ]
    # Normal/target invocation must retain its original distribution.
    assert sampler(logits).tolist() == [[2], [1]]


def test_plain_greedy_keeps_fused_argmax():
    def no_projection(hidden):
        raise AssertionError("unconstrained greedy must retain fused argmax")

    drafter = SimpleNamespace(
        _lm_head_fn=no_projection,
        _greedy_token=lambda hidden: mx.array([[2]]),
    )
    Qwen3_5MTPDraftModel._set_seed_from_hidden(
        drafter, mx.zeros((1, 1, 3)), lambda x: x, True
    )
    assert drafter._seed_token.tolist() == [[2]]


def test_only_static_bias_is_marked_draft_safe():
    processors = make_logits_processors(logit_bias={2: -10000}, repetition_penalty=1.1)
    assert [getattr(p, "draft_safe", False) for p in processors] == [True, False]


@pytest.mark.parametrize("transition", [False, True])
def test_real_qwen_proposals_follow_target_bias(monkeypatch, transition):
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.tests.test_mtp_processors import tiny_target_and_draft

    mx.random.seed(42)
    target, draft = tiny_target_and_draft()
    proposals = []
    original = Qwen3_5MTPDraftModel.draft_block

    def capture(self, *args, **kwargs):
        tokens = original(self, *args, **kwargs)
        proposals.extend(tokens.reshape(-1).tolist())
        return tokens

    monkeypatch.setattr(Qwen3_5MTPDraftModel, "draft_block", capture)
    processors = make_logits_processors(
        logit_bias={i: -10000 for i in range(32) if i != 7}
    )
    prompt = PromptProcessingBatch(
        model=target,
        uids=[0],
        input_ids=[[1, 6, 7]],
        max_tokens=[12],
        inputs_embeds=None,
        prompt_kwargs={},
        logits_processors=[processors],
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=3,
        greedy_sampling=True,
    )
    batch = prompt.generate(
        lambda x: mx.argmax(x, axis=-1), lambda t: False, compute_logprobs=False
    )
    output = [r.token for r in batch.next()]
    output.extend(r.token for r in batch.next())
    if transition:
        batch = batch.to_autoregressive()
        batch.enable_mtp_repromotion(draft, draft_block_size=3)
        output.extend(r.token for r in batch.next())
        batch = batch.to_speculative_mtp()
        assert batch is not None
    while len(batch):
        output.extend(r.token for r in batch.next() if r.token is not None)
    assert output == [7] * 12
    assert proposals and set(proposals) == {7}
    # The final round may stop at the output budget before accepting its tail.
    assert draft.accept_lens[:-1] == draft.draft_lens[:-1]
    assert sum(draft.accept_lens) > 0
    batch.release_drafter()


def test_real_qwen_compaction_keeps_per_request_bias(monkeypatch):
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.tests.test_mtp_processors import tiny_target_and_draft

    mx.random.seed(42)
    target, draft = tiny_target_and_draft()
    proposal_batches = []
    original = Qwen3_5MTPDraftModel.draft_block

    def capture(self, *args, **kwargs):
        tokens = original(self, *args, **kwargs)
        proposal_batches.append(tokens.tolist())
        return tokens

    monkeypatch.setattr(Qwen3_5MTPDraftModel, "draft_block", capture)
    processors = [
        make_logits_processors(logit_bias={i: -10000 for i in range(32) if i != token})
        for token in (7, 9)
    ]
    prompt = PromptProcessingBatch(
        model=target,
        uids=[10, 20],
        input_ids=[[1, 6], [1, 8]],
        max_tokens=[5, 11],
        inputs_embeds=None,
        prompt_kwargs={},
        logits_processors=processors,
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=3,
        greedy_sampling=True,
    )
    batch = prompt.generate(
        lambda x: mx.argmax(x, axis=-1),
        lambda token: False,
        compute_logprobs=False,
    )
    output = {10: [], 20: []}
    while len(batch):
        for response in batch.next():
            if response.token is not None:
                output[response.uid].append(response.token)

    assert output == {10: [7] * 5, 20: [9] * 11}
    assert any(len(rows) == 1 for rows in proposal_batches)
    for rows in proposal_batches:
        expected = (7, 9) if len(rows) == 2 else (9,)
        assert [set(row) for row in rows] == [{token} for token in expected]
    batch.release_drafter()
