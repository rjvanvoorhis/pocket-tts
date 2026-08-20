import io
import logging
import os
import re
import sys
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue

import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from safetensors import safe_open
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_VOICES_DIR,
    MAX_TOKEN_PER_CHUNK,
    get_default_text_for_language,
    get_default_voice_for_language,
)
from pocket_tts.models.tts_model import TTSModel, export_model_state
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool",
    pretty_exceptions_show_locals=False,
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance, one per worker process. With `batch_size > 1`,
# uvicorn spawns multiple worker processes (each a full copy of this module),
# so each worker ends up with its own independent `tts_model` here.
tts_model: TTSModel | None = None

# Directory where cloned voice profiles (.safetensors) are persisted, set by
# the lifespan startup hook below.
VOICES_DIR: Path = Path(DEFAULT_VOICES_DIR)

# Server-wide default for max_tokens (tokens per generated chunk before a
# sentence gets split further), overridable per-request on /tts. Set by the
# lifespan startup hook below.
DEFAULT_MAX_TOKENS_PER_CHUNK: int = MAX_TOKEN_PER_CHUNK

_VOICE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

# serve() can't load the model directly and hand it to worker processes: with
# `batch_size > 1`, uvicorn spawns separate worker processes that each import
# this module fresh, so model loading has to happen per-worker in the
# lifespan hook below. serve() passes its CLI options through via env vars,
# which subprocesses inherit automatically.
_ENV_LANGUAGE = "POCKET_TTS_LANGUAGE"
_ENV_CONFIG = "POCKET_TTS_CONFIG"
_ENV_QUANTIZE = "POCKET_TTS_QUANTIZE"
_ENV_VOICES_DIR = "POCKET_TTS_VOICES_DIR"
_ENV_MAX_TOKENS = "POCKET_TTS_MAX_TOKENS"
_ENV_QUIET = "POCKET_TTS_QUIET"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, VOICES_DIR, DEFAULT_MAX_TOKENS_PER_CHUNK

    quiet = os.environ.get(_ENV_QUIET) == "1"
    with enable_logging("pocket_tts", logging.ERROR if quiet else logging.INFO):
        language = os.environ.get(_ENV_LANGUAGE) or None
        config = os.environ.get(_ENV_CONFIG) or None
        quantize = os.environ.get(_ENV_QUANTIZE) == "1"
        voices_dir = os.environ.get(_ENV_VOICES_DIR, DEFAULT_VOICES_DIR)
        DEFAULT_MAX_TOKENS_PER_CHUNK = int(
            os.environ.get(_ENV_MAX_TOKENS) or MAX_TOKEN_PER_CHUNK
        )

        logger.info("Loading model instance (pid=%d)...", os.getpid())
        tts_model = TTSModel.load_model(
            language=language, config=config, quantize=quantize
        )
        VOICES_DIR = Path(voices_dir).expanduser()
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Model instance loaded (pid=%d).", os.getpid())

        yield


web_app = FastAPI(
    title="Kyutai Pocket TTS API",
    description="Text-to-Speech generation API",
    version="1.0.0",
    lifespan=lifespan,
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    content = static_path.read_text(encoding='utf-8')
    # Replace the placeholder with the actual default text prompt
    print(str(tts_model.origin))
    content = content.replace(
        "DEFAULT_TEXT_PROMPT", get_default_text_for_language(str(tts_model.origin))
    )
    return content


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


def write_to_queue(queue, text_to_generate, model_state, max_tokens: int):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state, text_to_generate=text_to_generate, max_tokens=max_tokens
    )
    stream_audio_chunks(
        FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate
    )


def generate_data_with_state(text_to_generate: str, model_state: dict, max_tokens: int):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(
        target=write_to_queue, args=(queue, text_to_generate, model_state, max_tokens)
    )
    thread.start()

    # Yield data as it becomes available
    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

    thread.join()


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    max_tokens: int | None = Form(
        None,
        description="Max tokens per generated chunk. Long sentences get split at "
        "commas/semicolons/colons once they exceed this. Defaults to the "
        "server's --max-tokens setting.",
    ),
):
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional built-in voice name (e.g., "alba"), or voice URL (http://, https://, or hf://).
            Can point to a voice profile's .safetensors data, e.g. http://localhost:8000/voices/<id>/data
        voice_wav: Optional uploaded voice file (mutually exclusive with voice_url)
        max_tokens: Optional override for the server's --max-tokens setting, for this request only.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS_PER_CHUNK
    elif max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be positive")

    if voice_url is None and voice_wav is None:
        voice_url = get_default_voice_for_language(str(tts_model.origin))

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(
            status_code=400, detail="Cannot provide both voice_url and voice_wav"
        )

    # Use the appropriate model state
    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in _ORIGINS_OF_PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400,
                detail="voice_url must start with http://, https://, or hf://",
            )
        model_state = tts_model._cached_get_state_for_audio_prompt(voice_url)
        logging.warning("Using voice from URL: %s", voice_url)
    elif voice_wav is not None:
        # Use uploaded voice file - preserve extension for format detection
        suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name

        # Close the file before reading it back (required on Windows)
        try:
            model_state = tts_model.get_state_for_audio_prompt(
                Path(temp_file_path), truncate=True
            )
        finally:
            os.unlink(temp_file_path)
    else:
        raise HTTPException(status_code=500, detail="This should never happen.")

    return StreamingResponse(
        generate_data_with_state(text, model_state, max_tokens),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


# ------------------------------------------------------
# Voice profile endpoints
# ------------------------------------------------------


class VoiceRecord(BaseModel):
    id: str
    name: str


def _read_voice_record(path: Path) -> VoiceRecord | None:
    try:
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
    except Exception:
        logger.warning("Skipping unreadable voice profile: %s", path)
        return None
    return VoiceRecord(
        id=metadata.get("id", path.stem),
        name=metadata.get("name", path.stem),
    )


@web_app.post("/voices", response_model=VoiceRecord, status_code=201)
def create_voice(
    name: str = Form(...),
    voice_wav: UploadFile = File(...),
):
    """Clone a voice from an audio sample and save it as a reusable profile."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = voice_wav.file.read()
        temp_file.write(content)
        temp_file.flush()
        temp_file_path = temp_file.name

    # Close the file before reading it back (required on Windows)
    try:
        model_state = tts_model.get_state_for_audio_prompt(
            Path(temp_file_path), truncate=True
        )
    finally:
        os.unlink(temp_file_path)

    voice_id = uuid.uuid4().hex
    dest_path = VOICES_DIR / f"{voice_id}.safetensors"
    export_model_state(model_state, dest_path, metadata={"id": voice_id, "name": name})

    return VoiceRecord(id=voice_id, name=name)


@web_app.get("/voices", response_model=list[VoiceRecord])
def list_voices():
    """List all voice profiles available on this server."""
    records = (
        _read_voice_record(path) for path in sorted(VOICES_DIR.glob("*.safetensors"))
    )
    return [record for record in records if record is not None]


@web_app.get("/voices/{voice_id}", response_model=VoiceRecord)
def get_voice(voice_id: str):
    """Look up a single voice profile by id."""
    if not _VOICE_ID_PATTERN.match(voice_id):
        raise HTTPException(status_code=404, detail="Voice not found")

    path = VOICES_DIR / f"{voice_id}.safetensors"
    record = _read_voice_record(path) if path.exists() else None
    if record is None:
        raise HTTPException(status_code=404, detail="Voice not found")
    return record


@web_app.get("/voices/{voice_id}/data")
def get_voice_data(voice_id: str):
    """Download the raw .safetensors file for a voice profile."""
    if not _VOICE_ID_PATTERN.match(voice_id):
        raise HTTPException(status_code=404, detail="Voice not found")

    path = VOICES_DIR / f"{voice_id}.safetensors"
    record = _read_voice_record(path) if path.exists() else None
    if record is None:
        raise HTTPException(status_code=404, detail="Voice not found")

    return FileResponse(
        path=path, media_type="application/octet-stream", filename=path.name
    )


@web_app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str):
    """Delete a voice profile."""
    if not _VOICE_ID_PATTERN.match(voice_id):
        raise HTTPException(status_code=404, detail="Voice not found")

    path = VOICES_DIR / f"{voice_id}.safetensors"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Voice not found")

    try:
        path.unlink()
        logger.info("Deleted voice profile: %s", voice_id)
        return {"status": "deleted", "id": voice_id}
    except Exception as e:
        logger.error("Error deleting voice profile: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete voice")


@cli_app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help="Language for the TTS model. "
            "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'german_24l', 'portuguese', 'italian', 'spanish'."
            " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
    voices_dir: Annotated[
        str,
        typer.Option(help="Directory to store voice profiles created via POST /voices"),
    ] = DEFAULT_VOICES_DIR,
    max_tokens: Annotated[
        int,
        typer.Option(
            help="Default max tokens per generated chunk. Text is split into "
            "sentences and packed into chunks up to this size before generation; "
            "a single sentence longer than this still gets generated as one "
            "oversized chunk (after trying to sub-split it on commas/semicolons), "
            "which can skip words. Overridable per-request via the /tts "
            "'max_tokens' form field."
        ),
    ] = MAX_TOKEN_PER_CHUNK,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Number of independent model instances to run, each in its own "
            "worker process (like running `serve` this many times). Each instance "
            "uses ~450MB memory (~234MB with --quantize) and can handle one "
            "request at a time, so this many requests can be processed "
            "concurrently. Default is 1 (a single process, no concurrency)."
        ),
    ] = 1,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Disable logging output")
    ] = False,
):
    """Start the FastAPI server."""

    if reload and batch_size > 1:
        raise typer.BadParameter(
            "--reload and --batch-size > 1 cannot be used together."
        )

    # The model itself can't be loaded here and shared with worker processes:
    # each worker (when batch_size > 1) is a separate process that imports
    # this module fresh, so loading happens per-worker in the `lifespan`
    # startup hook instead. Pass the options through via env vars, which
    # worker subprocesses inherit.
    os.environ[_ENV_LANGUAGE] = language or ""
    os.environ[_ENV_CONFIG] = config or ""
    os.environ[_ENV_QUANTIZE] = "1" if quantize else "0"
    os.environ[_ENV_VOICES_DIR] = voices_dir
    os.environ[_ENV_MAX_TOKENS] = str(max_tokens)
    os.environ[_ENV_QUIET] = "1" if quiet else "0"

    uvicorn.run(
        "pocket_tts.main:web_app",
        host=host,
        port=port,
        reload=reload,
        workers=batch_size if batch_size > 1 else None,
    )


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[str, typer.Option(help="Text to generate")] = None,
    voice: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to audio conditioning file (voice to clone). "
                "Defaults to a built-in voice chosen from the language: "
                "'giovanni' for italian, 'lola' for spanish, 'juergen' for german, "
                "'rafael' for portuguese, 'estelle' for french, 'alba' otherwise."
            ),
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Disable logging output")
    ] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'spanish_24l',"
                "'german_24l', 'portuguese_24l', 'italian_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview. They're not the final "
                "models for those languages."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float | None,
        typer.Option(
            help="Temperature for generation. Defaults to the model's recommended "
            "value from its config (0.3 for the English model, 0.7 otherwise)."
        ),
    ] = None,
    noise_clamp: Annotated[
        float, typer.Option(help="Noise clamp value")
    ] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[
        float, typer.Option(help="EOS threshold")
    ] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens per chunk.")
    ] = MAX_TOKEN_PER_CHUNK,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Generate speech using Kyutai Pocket TTS."""
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if text is None:
            text = get_default_text_for_language(language)
        if text == "-":
            # Read text from stdin
            text = sys.stdin.read()

        if not text.strip():
            logger.error("No input received from stdin.")
            raise typer.Exit(code=1)
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temperature,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
        )
        tts_model.to(device)

        if voice is None:
            voice = get_default_voice_for_language(language)
        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
        )

        stream_audio_chunks(
            output_path, audio_chunks, tts_model.config.mimi.sample_rate
        )

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


# ----------------------------------------------
# export audio to safetensors CLI implementation
# ----------------------------------------------


@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Disable logging output")
    ] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'german_24l','spanish_24l',"
                " 'portuguese_24l', 'italian_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
):
    """Convert and save audio to .safetensors file"""

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(language=language, config=config)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=audio_path, truncate=True
        )
        export_model_state(model_state, export_path)


if __name__ == "__main__":
    cli_app()
