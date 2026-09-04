import queue
from types import SimpleNamespace

import pytest
import torch

import pocket_tts.models.tts_model as tts_model_module
from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models.tts_model import TTSModel, _is_safetensors_source


def test_generate_audio_stream_uses_prepared_chunk_text(monkeypatch):
    calls = []

    def fake_split_into_best_sentences(
        tokenizer, text_to_generate, max_tokens, pad_with_spaces_for_short_inputs, remove_semicolons
    ):
        assert text_to_generate == "hi"
        assert pad_with_spaces_for_short_inputs is True
        return ["hi"]

    def fake_generate_audio_stream_short_text(**kwargs):
        calls.append(kwargs)
        yield torch.tensor([0.0])

    monkeypatch.setattr(
        tts_model_module, "split_into_best_sentences", fake_split_into_best_sentences
    )
    model = SimpleNamespace(
        flow_lm=SimpleNamespace(conditioner=SimpleNamespace(tokenizer=object())),
        model_recommended_frames_after_eos=None,
        pad_with_spaces_for_short_inputs=True,
        remove_semicolons=False,
        _generate_audio_stream_short_text=fake_generate_audio_stream_short_text,
    )

    chunks = list(TTSModel.generate_audio_stream(model, {}, "hi"))

    assert len(chunks) == 1
    assert torch.equal(chunks[0], torch.tensor([0.0]))
    assert calls[0]["text_to_generate"] == "        Hi."
    assert calls[0]["frames_after_eos"] == 5


def test_generate_audio_stream_teacher_forcing_uses_previous_chunk_audio(monkeypatch):
    calls = []

    def fake_split_into_best_sentences(
        tokenizer, text_to_generate, max_tokens, pad_with_spaces_for_short_inputs, remove_semicolons
    ):
        return ["chunk one", "chunk two"]

    def fake_generate_audio_stream_short_text(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            yield torch.tensor([1.0, 2.0])
        else:
            yield torch.tensor([3.0])

    monkeypatch.setattr(
        tts_model_module, "split_into_best_sentences", fake_split_into_best_sentences
    )
    model = SimpleNamespace(
        flow_lm=SimpleNamespace(conditioner=SimpleNamespace(tokenizer=object())),
        model_recommended_frames_after_eos=None,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
        sample_rate=24000,
        _generate_audio_stream_short_text=fake_generate_audio_stream_short_text,
    )

    chunks = list(
        TTSModel.generate_audio_stream(
            model,
            {},
            "chunk one. chunk two.",
            crossfade_duration=0.0,
            chunk_conditioning="teacher_forcing",
        )
    )

    assert len(calls) == 2
    # First chunk has no predecessor to force against.
    assert calls[0]["teacher_force_audio"] is None
    assert calls[0]["text_to_generate"] == "Chunk one."
    # Second chunk is conditioned on the first chunk's own generated audio,
    # and its text includes the first chunk's text as read-ahead context.
    assert calls[1]["teacher_force_audio"] is not None
    assert torch.equal(calls[1]["teacher_force_audio"], torch.tensor([1.0, 2.0]))
    assert calls[1]["text_to_generate"] == "Chunk one chunk two."
    # Output excludes the replayed prefix - only each chunk's own new audio.
    assert [c.tolist() for c in chunks] == [[1.0, 2.0], [3.0]]


def test_generate_audio_stream_rejects_unknown_chunk_conditioning():
    model = SimpleNamespace(model_recommended_frames_after_eos=None)
    with pytest.raises(ValueError, match="chunk_conditioning"):
        list(
            TTSModel.generate_audio_stream(
                model, {}, "hi", chunk_conditioning="not-a-real-mode"
            )
        )


def test_generate_dialogue_stream_empty_turns_yields_nothing():
    assert list(TTSModel.generate_dialogue_stream(SimpleNamespace(), [])) == []


def test_generate_dialogue_stream_single_turn_skips_stitching():
    calls = []

    def fake_generate_audio_stream(**kwargs):
        calls.append(kwargs)
        yield torch.tensor([1.0, 2.0])

    model = SimpleNamespace(sample_rate=24000, generate_audio_stream=fake_generate_audio_stream)

    chunks = list(
        TTSModel.generate_dialogue_stream(model, [({"speaker": "a"}, "hi")])
    )

    # A single turn has no boundary to stitch, so it's passed straight
    # through to generate_audio_stream with no extra silence/crossfade work.
    assert len(calls) == 1
    assert calls[0]["model_state"] == {"speaker": "a"}
    assert calls[0]["text_to_generate"] == "hi"
    assert [c.tolist() for c in chunks] == [[1.0, 2.0]]


def test_generate_dialogue_stream_inserts_silence_at_speaker_change():
    calls = []

    def fake_generate_audio_stream(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            yield torch.full((100,), 1.0)
        else:
            yield torch.full((100,), 2.0)

    model = SimpleNamespace(sample_rate=24000, generate_audio_stream=fake_generate_audio_stream)

    chunks = list(
        TTSModel.generate_dialogue_stream(
            model,
            [({"speaker": "a"}, "hi"), ({"speaker": "b"}, "there")],
            crossfade_duration=0.0,
            turn_silence_duration=0.01,
        )
    )

    assert len(calls) == 2
    assert calls[0]["model_state"] == {"speaker": "a"}
    assert calls[1]["model_state"] == {"speaker": "b"}

    audio = torch.cat(chunks)
    # 100 samples of turn one, a 0.01s (240-sample) silence gap, then 100
    # samples of turn two - a real gap at the speaker change, not a raw cut.
    assert audio.shape[0] == 100 + 240 + 100
    assert torch.all(audio[:100] == 1.0)
    assert torch.all(audio[100:340] == 0.0)
    assert torch.all(audio[340:] == 2.0)


def test_generate_reports_autoregressive_errors_before_decoder_done():
    error = RuntimeError("generation failed")

    def raise_generation(*args, **kwargs):
        raise error

    model = SimpleNamespace(
        _flow_lm_current_end=lambda model_state: 0,
        _expand_kv_cache=lambda model_state, sequence_length: None,
        _run_flow_lm_and_increment_step=lambda model_state, text_tokens: None,
        _autoregressive_generation=raise_generation,
    )
    latents_queue = queue.Queue()
    result_queue = queue.Queue()

    TTSModel._generate(
        model,
        model_state={},
        prepared=TokenizedText(torch.zeros((1, 1), dtype=torch.long)),
        max_gen_len=1,
        frames_after_eos=1,
        latents_queue=latents_queue,
        result_queue=result_queue,
    )

    kind, value = result_queue.get(timeout=1)
    assert kind == "error"
    assert value is error
    assert latents_queue.get(timeout=1) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("voice.safetensors", True),
        ("hf://owner/repo/voices/voice.safetensors@abcdef", True),
        ("https://example.com/voice.safetensors?download=1", True),
        ("https://example.com/voice.wav?format=safetensors", False),
    ],
)
def test_is_safetensors_source_handles_revisions_and_query_strings(source, expected):
    assert _is_safetensors_source(source) is expected
