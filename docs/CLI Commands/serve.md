# Serve

The `serve` command starts a FastAPI web server that provides both a web interface and HTTP API for text-to-speech generation.

## Basic Usage

```bash
uvx pocket-tts serve
# or if installed manually:
pocket-tts serve
```

This starts a server on `http://localhost:8000` with the default voice model.

## Command Options

- `--host HOST`: Host to bind to (default: "localhost")
- `--port PORT`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--language`: Language for the TTS model, one of `'english_2026-01'`, `'english_2026-04'`, `'english'`, `'french_24l'`, `'german_24l'`, `'portuguese_24l'`, `'italian_24l'`, `'spanish_24l'` (default: `english`, which is the same model as `'english_2026-04'`). Incompatible with `--config`. The "24l" variants are bigger models, not distilled yet and here only as preview.
- `--config`: Path to a custom config .yaml. Incompatible with `--language`.
- `--quantize`: Use int8 quantization for the model (default: False). This can reduce memory usage and increase speed, with minimal impact on audio quality.
- `--batch-size`: Number of independent model instances to run, each in its own worker process, like running `serve` this many times (default: 1). Each instance uses ~450MB of memory (~234MB with `--quantize`) and can handle one request at a time, so this many requests can be processed concurrently. Incompatible with `--reload`.
- `--max-tokens`: Default max tokens per generated chunk (default: 50). Text is split into sentences and packed into chunks up to this size; a single sentence longer than this still generates as one oversized chunk (after trying to sub-split it on commas/semicolons/colons), which can skip words. Overridable per-request via the `/tts` endpoint's `max_tokens` form field.
- `--chunk-conditioning`: Default strategy for generating chunks after the first, when text needs multiple chunks (default: `independent`). `independent` generates each chunk from scratch off the voice prompt (cheaper); `teacher-forcing` re-processes each chunk alongside the previous one, forcing the decoder to replay the previous chunk's own audio as context instead of resetting, at roughly double the generation work per chunk - see [Chunk Stitching](#chunk-stitching) below. Overridable per-request via the `/tts` endpoint's `chunk_conditioning` form field.
- `--quiet` / `-q`: Disable logging output.
## Examples

### Basic Server

```bash
# Start with default settings
pocket-tts serve

# Custom host and port
pocket-tts serve --host "localhost" --port 8080
```

### Concurrent Requests

To handle multiple requests at once (e.g. processing many chapters of a book in parallel), run several model instances with `--batch-size`:

```bash
# Load 2 model instances (~900MB memory) and handle 2 requests concurrently
pocket-tts serve --batch-size 2
```

Clients keep hitting the same host/port as usual — the server transparently spreads incoming requests across the instances.

`--batch-size` relies on uvicorn's built-in multi-worker mode, which forks worker processes on Linux (the deployment target, e.g. via the provided `Dockerfile`) — reliable and battle-tested. On Windows, the same feature uses a `spawn` + socket-sharing workaround that can be flaky (a worker occasionally fails to start and gets silently respawned, which can leave you running fewer live workers than requested without any error). Prefer testing `--batch-size > 1` in Docker or on Linux; on Windows, `--batch-size 1` (the default) is unaffected.

### Chunk Stitching

Long text gets split into sentence chunks (sized by `--max-tokens`) that are each generated independently by default, which can produce audible variation in intonation from chunk to chunk. `--chunk-conditioning teacher-forcing` instead re-generates each chunk together with the text of the chunk before it, forcing the decoder to reproduce that previous chunk's own already-generated audio for that span (rather than sampling it again) before letting it freely generate the new chunk's audio. This gives the model real prior context to continue from without ever re-deriving the voice conditioning from generated audio (which would drift, like a copy of a copy) - each chunk still always starts from the original voice prompt. The tradeoff is roughly double the generation work per chunk, since each chunk's text gets processed twice (once as new content, once as forced context for the next chunk).

```bash
pocket-tts serve --chunk-conditioning teacher-forcing
```

This is exploratory - it hasn't been validated for perceptual quality across a wide range of text, so try it on your own material and see whether the tradeoff is worth it.

### Custom Language
To select the default language model, pass `--language`:
```bash
pocket-tts serve --language french_24l
```

### Custom Model Config

If you'd like to override the paths from which the models are loaded, you can provide a custom YAML configuration.

Copy one of the files in `pocket_tts/config` (for example `pocket_tts/config/english.yaml`) and change `weights_path`, `weights_path_without_voice_cloning:`, and `tokenizer_path:` to the paths of the models you want to load.

Then, use the --config option to point to your newly created config.

```bash
# Use a different config
pocket-tts serve --config "C://pocket-tts/my_config.yaml"
```

## Web Interface

Once the server is running, navigate to `http://localhost:8000` to access the web interface.

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model.
