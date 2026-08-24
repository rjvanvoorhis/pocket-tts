DEFAULT_LANGUAGE = "english"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_LSD_DECODE_STEPS = 1
DEFAULT_NOISE_CLAMP = None
DEFAULT_EOS_THRESHOLD = -4.0
DEFAULT_FRAMES_AFTER_EOS = None
# Duration, in seconds, crossfaded across the boundary between consecutive
# sentence chunks in generate_audio_stream() to mask the decoder-state reset
# that happens at each chunk boundary. Also used as the fade-to/from-silence
# duration when DEFAULT_SILENCE_DURATION_S > 0.
DEFAULT_CROSSFADE_DURATION_S = 0.1
# Duration, in seconds, of true silence inserted between consecutive sentence
# chunks in generate_audio_stream(), in place of directly crossfading one
# chunk's audio into the next. Without a gap, chunk boundaries can sound like
# the next sentence starts abruptly. Set to 0 to fall back to a gapless
# crossfade (the old behavior).
DEFAULT_SILENCE_DURATION_S = 0.15
# Duration, in seconds, of true silence inserted between turns in
# generate_dialogue_stream(), analogous to DEFAULT_SILENCE_DURATION_S but for
# speaker-turn boundaries rather than same-speaker sentence boundaries.
# Conversational back-and-forth paces differently from narration, so this is
# kept independently tunable rather than reusing DEFAULT_SILENCE_DURATION_S -
# starts at the same value as a first guess, tune by ear from there.
DEFAULT_DIALOGUE_SILENCE_DURATION_S = 0.15
# Overridable via `serve --max-tokens` / the /tts `max_tokens` form field, and
# the `generate --max-tokens` CLI option. 50 is a conservative default; some
# models (e.g. english_2026-04) reportedly tolerate bigger chunks, but the
# practical per-model ceiling (where generation quality degrades) hasn't been
# measured yet - it requires listening to samples, not just checking for
# crashes/warnings.
MAX_TOKEN_PER_CHUNK = 50
# Where the server persists voice profiles created via POST /voices
DEFAULT_VOICES_DIR = "./data/voices"

DEFAULT_TEXT_FOR_LANGUAGE = {
    "english": (
        "Hello world. I am Kyutai's Pocket TTS. "
        "I'm fast enough to run on small CPUs. "
        "I hope you'll like me."
    ),
    "french": (
        "Bonjour le monde. Je suis le TTS de poche de Kyutai. "
        "Je suis assez rapide pour fonctionner sur de petits CPU. "
        "J'espère que vous m'aimerez."
    ),
    "german": (
        "Hallo Welt. Ich bin Pocket TTS von Kyutai. "
        "Ich bin schnell genug, um auch auf kleinen CPUs zu laufen. "
        "Ich hoffe, ich gefalle dir."
    ),
    "portuguese": (
        "Olá mundo. Eu sou o Pocket TTS da Kyutai. "
        "Sou rápido o suficiente para rodar em CPUs pequenas. "
        "Espero que você goste de mim."
    ),
    "italian": (
        "Ciao mondo. Sono il Pocket TTS di Kyutai. "
        "Sono abbastanza veloce da funzionare su piccole CPU. "
        "Spero che ti piacerò."
    ),
    "spanish": (
        "Hola mundo. Soy el Pocket TTS de Kyutai. "
        "Soy lo suficientemente rápido para funcionar en pequeñas CPU. "
        "Espero que te guste."
    ),
}

DEFAULT_VOICE_FOR_LANGUAGE = {
    "italian": "giovanni",
    "spanish": "lola",
    "german": "juergen",
    "portuguese": "rafael",
    "french": "estelle",
}
DEFAULT_VOICE_FALLBACK = "alba"


def get_default_text_for_language(language: str | None) -> str:
    for key, text in DEFAULT_TEXT_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return text
    return DEFAULT_TEXT_FOR_LANGUAGE[DEFAULT_LANGUAGE]


def get_default_voice_for_language(language: str | None) -> str:
    for key, voice in DEFAULT_VOICE_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return voice
    return DEFAULT_VOICE_FALLBACK
