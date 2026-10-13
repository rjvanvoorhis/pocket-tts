"""Stitching independently generated audio chunks into one stream.

Each sentence chunk (and each dialogue turn) is generated with a freshly reset decoder state, which
leaves an audible discontinuity at every boundary. These helpers join the chunks with either a
gapless Hann-windowed crossfade or a short silent gap bounded by fades.
"""

import logging
import math

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# A chunk boundary from split_into_best_sentences() can land on different
# kinds of punctuation depending on how the text had to be broken up: a full
# stop most of the time, but a comma/semicolon/colon when an oversized
# sentence was sub-split, or nothing at all when even that fallback failed
# and a run-on clause was hard-split mid-word. Only the first is a natural
# breath point, so it gets the whole configured pause; the others get a
# fraction of it.
_SILENCE_MULTIPLIER_BY_BOUNDARY_KIND = {"sentence": 1.0, "clause": 0.4, "word": 0.15}


def _classify_chunk_boundary(chunk_text: str) -> str:
    """Classify the punctuation `chunk_text` ends on: "sentence" (., !, ?,
    possibly followed by closing quotes/brackets), "clause" (, ; :), or
    "word" (no trailing punctuation at all).
    """
    stripped = chunk_text.rstrip().rstrip("\"'”’)]")
    if not stripped:
        return "word"
    last = stripped[-1]
    if last in ".!?":
        return "sentence"
    if last in ",;:":
        return "clause"
    return "word"


def _tap(stream, sink: list):
    """Pass an audio-chunk stream through unchanged while collecting its
    pieces into `sink`, so the caller has the full chunk audio available
    once the stream is exhausted (used for "teacher_forcing" conditioning).
    """
    for piece in stream:
        sink.append(piece)
        yield piece


def _make_hann_fade_window(n: int, device, dtype) -> torch.Tensor:
    """Create a Hann-windowed fade curve from 0 to 1.

    This gives a smooth, perceptually better fade than linear, with no
    abrupt changes in curvature at the endpoints.
    """
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    # Hann window shape: 0.5 * (1 - cos(pi * i / (n - 1))) for i in 0..n-1
    # This naturally goes from 0 at i=0 to 1 at i=n-1
    i = torch.arange(n, device=device, dtype=dtype)
    return 0.5 * (1 - torch.cos(math.pi * i / (n - 1)))


def _smooth_spectral_envelope(
    tail: torch.Tensor, head: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth spectral envelope at chunk boundary using STFT analysis.

    Analyzes the magnitude spectrum near the boundary and blends them to reduce
    the "noise quality" jump that results from decoder state reset.

    Args:
        tail: Last chunk's audio (will use tail[-n:])
        head: Next chunk's audio (will use head[:n])
        n: Number of samples to analyze/blend at boundary

    Returns:
        (smoothed_tail, smoothed_head): Spectrally smoothed boundary segments,
        each guaranteed to have shape [n]
    """
    # Use a modest FFT size for boundary analysis
    fft_size = 2048
    if n < 256:
        # Not enough samples for meaningful STFT; return unchanged
        return tail[-n:], head[:n]

    try:
        # Extract boundary regions - guaranteed to be size n
        tail_seg = tail[-n:]
        head_seg = head[:n]

        # Use FFT on the full segment (pad if smaller than fft_size, truncate if larger)
        if n < fft_size:
            tail_padded = F.pad(tail_seg, (0, fft_size - n))
            head_padded = F.pad(head_seg, (0, fft_size - n))
        else:
            # Take center portion for larger segments
            start = max(0, (n - fft_size) // 2)
            tail_padded = tail_seg[start : start + fft_size]
            head_padded = head_seg[start : start + fft_size]

        # Compute FFT and get magnitude/phase
        tail_fft = torch.fft.rfft(tail_padded, n=fft_size)
        head_fft = torch.fft.rfft(head_padded, n=fft_size)

        tail_mag = torch.abs(tail_fft)
        head_mag = torch.abs(head_fft)
        head_phase = torch.angle(head_fft)

        # Blend magnitudes smoothly: head's phase with blended magnitude
        # This reduces spectral discontinuity while preserving prosody
        fade = torch.linspace(0, 1, tail_mag.shape[0], device=tail.device, dtype=tail.dtype)
        blended_mag = tail_mag * (1 - fade) + head_mag * fade

        # Reconstruct: use blended magnitude + head's phase
        blended_fft = blended_mag * torch.exp(1j * head_phase)
        blended_time = torch.fft.irfft(blended_fft, n=fft_size)

        # Blend: weight the spectrally smoothed version into the original head
        # The longer the crossfade, the more blending. Use head_seg as base and blend in
        smoothed_head = head_seg.clone()

        # Apply spectral envelope from blended spectrum at the start of the segment
        if blended_time.shape[0] >= n:
            # Blend with the start of the reconstructed audio
            spectral_component = blended_time[:n]
        else:
            # If reconstruction is shorter, pad it
            spectral_component = F.pad(blended_time, (0, n - blended_time.shape[0]))

        # Weight the spectral component in, scaled by envelope
        blend_weight = (
            0.3  # Balance between original and spectrally smoothed (30% smooth, 70% original)
        )
        smoothed_head = (1 - blend_weight) * head_seg + blend_weight * spectral_component

        smoothed_tail = tail_seg  # Keep tail unchanged, only smooth the head

        return smoothed_tail, smoothed_head
    except Exception as e:
        # Spectral smoothing is a best-effort enhancement; fall back if it fails
        logger.debug(f"Spectral envelope smoothing failed: {e}, using original audio")
        return tail[-n:], head[:n]


# How much of a chunk's own natural trailing/leading near-silence (e.g. from
# frames_after_eos padding) to strip before inserting the boundary's own
# silence gap, so the gap between sentences has a consistent, configured
# length rather than the model's trailing silence stacking on top of it.
_MAX_SILENCE_TRIM_S = 0.3
# Samples at or below this absolute amplitude (audio is float32 in [-1, 1])
# count as "near silence" for trimming purposes.
_SILENCE_TRIM_AMPLITUDE_THRESHOLD = 0.01


def _trim_trailing_near_silence(audio: torch.Tensor, max_trim: int) -> torch.Tensor:
    """Strip near-silent samples from the end of `audio`, looking back at
    most `max_trim` samples. If the whole lookback window is near-silent, all
    of it is trimmed.
    """
    if max_trim <= 0 or audio.shape[0] == 0:
        return audio
    window = audio[-max_trim:] if audio.shape[0] > max_trim else audio
    above = (window.abs() > _SILENCE_TRIM_AMPLITUDE_THRESHOLD).nonzero(as_tuple=True)[0]
    cut = window.shape[0] if above.numel() == 0 else window.shape[0] - 1 - above[-1].item()
    return audio[: audio.shape[0] - cut] if cut > 0 else audio


def _trim_leading_near_silence(audio: torch.Tensor, max_trim: int) -> torch.Tensor:
    """Strip near-silent samples from the start of `audio`, looking ahead at
    most `max_trim` samples. Mirror of `_trim_trailing_near_silence`.
    """
    if max_trim <= 0 or audio.shape[0] == 0:
        return audio
    window = audio[:max_trim] if audio.shape[0] > max_trim else audio
    above = (window.abs() > _SILENCE_TRIM_AMPLITUDE_THRESHOLD).nonzero(as_tuple=True)[0]
    cut = window.shape[0] if above.numel() == 0 else above[0].item()
    return audio[cut:] if cut > 0 else audio


def _boundary_transition(
    tail: torch.Tensor,
    head: torch.Tensor,
    fade_samples: int,
    silence_samples: int,
    trim_samples: int = 0,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Compute the audio to emit at a chunk boundary, and the residual tail to
    carry forward as the start of the *next* boundary's lookback window.

    If `silence_samples` is 0, this is a gapless overlap-add: the trailing
    `fade_samples` of `tail` are crossfaded into the leading `fade_samples` of
    `head` using a Hann window, after spectral envelope smoothing to reduce
    the "noise quality" jump that a decoder state reset otherwise produces.

    If `silence_samples` is > 0, a true silent gap is inserted instead: any
    pre-existing near-silence at the end of `tail` / start of `head` is
    trimmed first (up to `trim_samples`, so the gap length stays consistent
    regardless of how much trailing silence the model happened to generate),
    then `tail` is faded down to silence, followed by `silence_samples` of
    silence, followed by `head` fading up from silence. Fading into real
    silence on both sides is inherently discontinuity-free (no unrelated
    spectra are blended together), so spectral smoothing isn't needed in this
    mode.
    """
    if silence_samples > 0 and trim_samples > 0:
        tail = _trim_trailing_near_silence(tail, trim_samples)
        head = _trim_leading_near_silence(head, trim_samples)

    n = min(fade_samples, tail.shape[0], head.shape[0])
    pieces: list[torch.Tensor] = []
    if tail.shape[0] > n:
        pieces.append(tail[:-n] if n > 0 else tail)

    if silence_samples > 0:
        if n > 0:
            fade_out = _make_hann_fade_window(n, device=tail.device, dtype=tail.dtype)
            pieces.append(tail[-n:] * (1 - fade_out))
        pieces.append(torch.zeros(silence_samples, device=head.device, dtype=head.dtype))
        if n > 0:
            fade_in = _make_hann_fade_window(n, device=head.device, dtype=head.dtype)
            pieces.append(head[:n] * fade_in)
        new_tail = head[n:] if head.shape[0] > n else None
    elif n > 0:
        tail_smooth, head_smooth = _smooth_spectral_envelope(tail, head, n)
        fade_in = _make_hann_fade_window(n, device=head.device, dtype=head.dtype)
        pieces.append(tail_smooth[-n:] * (1 - fade_in) + head_smooth[:n] * fade_in)
        new_tail = head[n:] if head.shape[0] > n else None
    else:
        new_tail = head

    return pieces, new_tail


def _crossfade_concatenated_streams(
    streams, crossfade_samples: int, silence_samples: int | list[int] = 0, trim_samples: int = 0
):
    """Concatenate a sequence of audio-chunk generators into a single stream.

    Chunks within a sub-stream are passed through unchanged. Only the boundary
    between one sub-stream and the next is treated specially, via
    `_boundary_transition`: either a gapless Hann-windowed crossfade (with
    spectral envelope smoothing) or a true silent gap bounded by fades to/from
    silence, depending on whether that boundary's silence amount is 0.

    `silence_samples` can be a single value applied to every boundary, or a
    list with one entry per boundary (i.e. `len(list(streams)) - 1` entries)
    for a per-boundary pause length. `trim_samples` bounds how much
    pre-existing near-silence around each boundary is trimmed before the gap
    is inserted - see `_boundary_transition`.
    """
    tail = None  # trailing samples produced so far, held back for the next boundary
    # The lookback buffer needs to be large enough for both the crossfade
    # fade window and the silence-trim search window; the smaller of the two
    # boundary transitions ("full window arrived" vs. "stream ended early")
    # still fires correctly on the same amount of buffered audio.
    lookback_samples = max(crossfade_samples, trim_samples)

    for boundary_idx, stream in enumerate(streams):
        boundary_silence = (
            silence_samples[boundary_idx - 1]
            if isinstance(silence_samples, (list, tuple))
            else silence_samples
        )
        # Even with crossfade and trimming both disabled, a nonzero silence
        # gap still needs to probe at least one sample past the boundary to
        # trigger `_boundary_transition`.
        needed = 0
        if tail is not None:
            needed = lookback_samples or (1 if boundary_silence > 0 else 0)
        boundary_chunks = []
        boundary_len = 0

        for piece in stream:
            if piece.shape[0] == 0:
                continue

            if needed > 0:
                boundary_chunks.append(piece)
                boundary_len += piece.shape[0]
                if boundary_len < needed:
                    continue
                head = torch.cat(boundary_chunks)
                pieces, tail = _boundary_transition(
                    tail, head, crossfade_samples, boundary_silence, trim_samples
                )
                yield from pieces
                needed = 0
                continue

            tail = piece if tail is None else torch.cat([tail, piece])
            if tail.shape[0] > lookback_samples:
                cut = tail.shape[0] - lookback_samples
                yield tail[:cut]
                tail = tail[cut:]

        if needed > 0 and boundary_chunks:
            # The sub-stream ended before enough samples arrived to fill a full
            # boundary window; transition with however much we got.
            head = torch.cat(boundary_chunks)
            pieces, tail = _boundary_transition(
                tail, head, crossfade_samples, boundary_silence, trim_samples
            )
            yield from pieces

    if tail is not None and tail.shape[0] > 0:
        yield tail
