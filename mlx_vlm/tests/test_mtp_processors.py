"""Stateful constraints must follow committed MTP prefixes, not draft suffixes."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.speculative import mtp


def walk(
    monkeypatch,
    drafts,
    targets,
    *,
    uniform=False,
    budgets=None,
    forced=None,
    observer=None,
    stop=None,
):
    consumed = [[] for _ in drafts]
    monkeypatch.setattr(mtp, "_mtp_logits_from_hidden", lambda lm, hidden: hidden)
    logits = mx.full((len(drafts), len(drafts[0]) + 1, 16), -100.0)
    for row, tokens in enumerate(targets):
        for pos, token in enumerate(tokens):
            logits[row, pos, token] = 100.0

    def process(row, previous, values):
        consumed[row].append(previous)
        return values

    result = mtp._speculative_walk_batch_processed(
        SimpleNamespace(),
        logits,
        mx.array(drafts),
        lambda values: mx.argmax(values, axis=-1),
        budgets or [10] * len(drafts),
        active_idx=list(range(len(drafts))),
        first_bonus=[1] * len(drafts),
        process_logits=process,
        uniform_acceptance=uniform,
        forced_token_provider=forced,
        token_observer=observer,
        stop_check=stop,
    )
    return result, consumed


def test_rejected_suffix_never_reaches_processor(monkeypatch):
    result, consumed = walk(monkeypatch, [[2, 3, 4]], [[2, 9, 4, 5]])
    assert result == ([1], [[2, 9]])
    assert consumed == [[1, 2]]


def test_all_accepted_bonus_and_budget(monkeypatch):
    result, consumed = walk(monkeypatch, [[2, 3]], [[2, 3, 4]])
    assert result == ([2], [[2, 3, 4]])
    assert consumed == [[1, 2, 3]]
    result, consumed = walk(monkeypatch, [[2, 3]], [[2, 3, 4]], budgets=[1])
    assert result == ([0], [[2]])
    assert consumed == [[1]]


@pytest.mark.parametrize("uniform", [False, True])
def test_rows_do_not_consume_discarded_uniform_suffix(monkeypatch, uniform):
    result, consumed = walk(
        monkeypatch, [[2, 3], [2, 3]], [[2, 3, 4], [9, 3, 4]], uniform=uniform
    )
    assert result == (([0, 0], [[2], [9]]) if uniform else ([2, 0], [[2, 3, 4], [9]]))
    assert consumed == ([[1], [1]] if uniform else [[1, 2, 3], [1]])


def test_thinking_boundary_observed_once_and_stops_before_next_mask(monkeypatch):
    seen = []

    def observe(row, token):
        seen.append((row, token))
        return token == 3

    result, consumed = walk(monkeypatch, [[2, 3, 4]], [[2, 3, 4, 5]], observer=observe)
    assert result == ([1], [[2, 3]])
    assert seen == [(0, 2), (0, 3)]
    assert consumed == [[1, 2]]


def test_forced_token_and_eos_are_round_boundaries(monkeypatch):
    result, consumed = walk(monkeypatch, [[2, 3]], [[2, 3, 4]], forced=lambda row: 7)
    assert result == ([0], [[7]])
    assert consumed == [[1]]
    result, consumed = walk(
        monkeypatch, [[2, 3]], [[2, 3, 4]], stop=lambda row, token: token == 2
    )
    assert result == ([0], [[2]])
    assert consumed == [[1]]


class SequenceProcessor:
    """Strict state machine: a duplicate or discarded consumption is an error."""

    def __init__(self, sequence):
        self.sequence = sequence
        self.seen = []
        self.started = False

    def __call__(self, tokens, logits):
        if self.started:
            return self.process_last_token(int(tokens[-1]), logits)
        self.started = True
        return self.mask(logits)

    def mask(self, logits):
        values = mx.full(logits.shape, -float("inf"))
        values[:, self.sequence[len(self.seen) % len(self.sequence)]] = 0
        return values

    def process_last_token(self, token, logits):
        if not self.started:
            self.started = True
            return self.mask(logits)
        assert token == self.sequence[len(self.seen) % len(self.sequence)]
        self.seen.append(token)
        return self.mask(logits)


def tiny_target_and_draft():
    from mlx_vlm.models.qwen3_5.language import LanguageModel, TextConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp import (
        ModelConfig,
        Qwen3_5MTPDraftModel,
    )

    config = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=1,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        head_dim=8,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 0, 0],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    config.mtp_num_hidden_layers = 1
    target = LanguageModel(config)
    target.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
    )
    return target, Qwen3_5MTPDraftModel(ModelConfig(text_config=config, block_size=3))


@pytest.mark.parametrize("transition", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
def test_real_qwen_layers_constraints_survive_mtp_ar_mtp(transition, greedy):
    from mlx_vlm.generate.ar import PromptProcessingBatch, SpeculativeGenerationBatch

    mx.random.seed(42)
    target, draft = tiny_target_and_draft()
    processor = SequenceProcessor([2, 3, 4, 5])
    prompt = PromptProcessingBatch(
        model=target,
        uids=[0],
        input_ids=[[1, 6, 7]],
        max_tokens=[12],
        inputs_embeds=None,
        prompt_kwargs={},
        logits_processors=[[processor]],
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=3,
        greedy_sampling=greedy,
    )
    sampler = (
        (lambda x: mx.argmax(x, axis=-1))
        if greedy
        else (lambda x: mx.random.categorical(x))
    )
    batch = prompt.generate(sampler, lambda token: False, compute_logprobs=False)
    assert isinstance(batch, SpeculativeGenerationBatch)
    emitted = [r.token for r in batch.next()]
    emitted.extend(r.token for r in batch.next())
    cache_identity = batch.prompt_cache
    if transition:
        batch = batch.to_autoregressive()
        assert batch.prompt_cache is cache_identity
        batch.enable_mtp_repromotion(draft, draft_block_size=3)
        emitted.extend(r.token for r in batch.next())
        batch = batch.to_speculative_mtp()
        assert isinstance(batch, SpeculativeGenerationBatch)
        assert batch.prompt_cache is cache_identity
    while len(batch):
        emitted.extend(r.token for r in batch.next() if r.token is not None)
    assert emitted == [2, 3, 4, 5] * 3
    assert processor.seen == emitted[:-1]
    assert draft.draft_lens
    batch.release_drafter()


def test_real_llguidance_json_with_mtp():
    import json
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from mlx_vlm.generate.ar import PromptProcessingBatch, SpeculativeGenerationBatch
    from mlx_vlm.structured import build_json_schema_logits_processor

    chars = list(dict.fromkeys('{}[]":,Ġtruefalsnok0123456789XY'))
    vocab = {char: idx for idx, char in enumerate(chars)}
    vocab["<eos>"] = 31
    tokenizer_impl = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer_impl.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_impl.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl, eos_token="<eos>"
    )
    processor = build_json_schema_logits_processor(
        tokenizer,
        {
            "type": "object",
            "properties": {"ok": {"const": True}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )
    target, draft = tiny_target_and_draft()
    prompt = PromptProcessingBatch(
        model=target,
        uids=[0],
        input_ids=[[1, 6]],
        max_tokens=[64],
        inputs_embeds=None,
        prompt_kwargs={},
        logits_processors=[[processor]],
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=3,
        greedy_sampling=True,
    )
    batch = prompt.generate(
        lambda x: mx.argmax(x, axis=-1), lambda t: t == 31, compute_logprobs=False
    )
    assert isinstance(batch, SpeculativeGenerationBatch)
    output = []
    while len(batch):
        output.extend(r.token for r in batch.next() if r.token is not None)
    assert output[-1] == 31
    assert json.loads(tokenizer.decode(output, skip_special_tokens=True)) == {
        "ok": True
    }
    assert draft.draft_lens
    batch.release_drafter()


def test_processor_round_failure_aborts_cache_transaction(monkeypatch):
    from unittest.mock import Mock

    target, draft = tiny_target_and_draft()
    verify = mtp._MTPVerifyResult(mx.zeros((1, 3, 16)), {})
    abort = Mock()
    monkeypatch.setattr(verify, "abort", abort)
    monkeypatch.setattr(mtp, "_mtp_verify_target", lambda *args, **kwargs: verify)
    monkeypatch.setattr(
        mtp, "_mtp_draft_block_active", lambda *args, **kwargs: mx.array([[2, 3]])
    )

    def fail(*args):
        raise ValueError("invalid grammar state")

    rounds = mtp._mtp_rounds_batch(
        target,
        draft,
        [],
        mx.zeros((1, 1, 16)),
        {},
        first_bonus=mx.array([1]),
        max_tokens=8,
        sampler=lambda x: mx.argmax(x, axis=-1),
        greedy_sampling=True,
        process_logits=fail,
    )
    with pytest.raises(ValueError, match="invalid grammar state"):
        next(rounds)
    abort.assert_called_once()


def test_real_qwen_rows_finish_independently():
    from mlx_vlm.generate.ar import PromptProcessingBatch

    target, draft = tiny_target_and_draft()
    processors = [SequenceProcessor([2, 3, 4]), SequenceProcessor([5, 6, 7])]
    prompt = PromptProcessingBatch(
        model=target,
        uids=[10, 20],
        input_ids=[[1, 6], [1, 7]],
        max_tokens=[5, 11],
        inputs_embeds=None,
        prompt_kwargs={},
        logits_processors=[[p] for p in processors],
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=3,
        greedy_sampling=True,
    )
    batch = prompt.generate(
        lambda x: mx.argmax(x, axis=-1), lambda token: False, compute_logprobs=False
    )
    outputs = {10: [], 20: []}
    while len(batch):
        for response in batch.next():
            if response.token is not None:
                outputs[response.uid].append(response.token)
    assert outputs[10] == [2, 3, 4, 2, 3]
    assert outputs[20] == ([5, 6, 7] * 4)[:11]
    assert processors[0].seen == outputs[10][:-1]
    assert processors[1].seen == outputs[20][:-1]
    batch.release_drafter()


@pytest.mark.parametrize("kind", ["mtp", "dflash", "eagle3"])
def test_server_accepts_processors_only_for_supported_drafter(kind):
    from unittest.mock import Mock
    from mlx_vlm.server.generation import GenerationArguments, ResponseGenerator

    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.wait_until_ready = lambda: None
    gen.draft_model = object()
    gen.draft_kind = kind
    gen._preprocess_request = Mock(side_effect=RuntimeError("reached preprocessing"))
    expected = RuntimeError if kind == "mtp" else ValueError
    message = "reached preprocessing" if kind == "mtp" else "Structured response_format"
    with pytest.raises(expected, match=message):
        gen.generate("test", args=GenerationArguments(logits_processors=[object()]))


def test_thinking_budget_and_processor_keep_full_forced_sequence():
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.structured import ThinkingAwareLogitsProcessor
    from mlx_vlm.utils import ThinkingBudgetCriteria

    tokenizer = SimpleNamespace(
        encode=lambda text, **kw: [{"<think>": 19, "</think>": 21, "\n": 20}[text]]
    )
    inner = SequenceProcessor([2, 3, 4])
    processor = ThinkingAwareLogitsProcessor(inner, tokenizer, enable_thinking=True)

    def always_think(tokens, logits):
        # Make the randomly initialized target keep thinking until the budget
        # forces a close. The real wrapper then activates the answer grammar.
        logits = mx.full(logits.shape, -float("inf"))
        logits[:, 8] = 0
        return logits

    criteria = ThinkingBudgetCriteria(
        tokenizer,
        3,
        enable_thinking=True,
        thinking_start_token="<think>",
        prompt_preopens_thinking=True,
    )
    target, draft = tiny_target_and_draft()
    prompt = PromptProcessingBatch(
        model=target,
        uids=[0],
        input_ids=[[1, 6]],
        max_tokens=[12],
        inputs_embeds=None,
        prompt_kwargs={},
        logits_processors=[[always_think, processor]],
        thinking_budget_criteria=[criteria],
        draft_model=draft,
        draft_kind="mtp",
        draft_block_size=3,
        greedy_sampling=True,
    )
    batch = prompt.generate(
        lambda x: mx.argmax(x, axis=-1), lambda token: False, compute_logprobs=False
    )
    output = []
    while len(batch):
        output.extend(r.token for r in batch.next() if r.token is not None)
    assert output == [8, 8, 8, 8, 20, 21, 2, 3, 4, 2, 3, 4]
    assert criteria.thinking_token_count == 5  # four thoughts and the forced newline
    assert inner.seen == [2, 3, 4, 2, 3]
    batch.release_drafter()


@pytest.mark.parametrize("cadence", [1, 4])
def test_immediate_processor_yield_resumes_due_prefill(cadence):
    from unittest.mock import Mock
    from mlx_vlm.generate.ar import BatchGenerator

    class Decode:
        logits_processors = [[SimpleNamespace(requires_immediate_decode_yield=True)]]
        prompt_cache = []

        def __len__(self):
            return 1

        next = Mock(return_value=["token"])

    gen = BatchGenerator.__new__(BatchGenerator)
    gen._generation_batch = Decode()
    gen._prompt_batch = SimpleNamespace(needs_processing=lambda: True)
    gen._unprocessed_sequences = []
    gen._gen_tokens_counter = gen._steps_counter = 0
    gen._decode_prefill_cadence_step = 0
    gen._prefill_schedule_interval = cadence
    gen._cache_eval_interval = 0
    gen.completion_batch_size = 4
    gen._sync_generation_head_residency = lambda: None
    gen._advance_prompt_batch = Mock()
    gen._prompt_time_counter = 0
    for _ in range(cadence):
        assert gen._next() == ([], ["token"])
        gen._advance_prompt_batch.assert_not_called()
    gen._next()
    gen._advance_prompt_batch.assert_called_once()
    assert gen._generation_batch.next.call_count == cadence
    gen._prompt_batch = None
