# AI Image Restoration API

REST API for image enhancement using Real-ESRGAN (super resolution), GFPGAN (face restoration), and OpenCV FastNlMeans (denoising). Built with FastAPI, containerised with Docker.

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/sumitjh/ai-image-restoration)** — try super resolution, face restoration, and denoising in the browser, no setup required.

## Models

| ID | Name | Description |
|----|------|-------------|
| `sr` | Real-ESRGAN | Super resolution — 2× or 4× upscaling |
| `face` | GFPGAN | Face restoration and enhancement |
| `denoise` | FastNlMeans | Color image denoising (strength 1–30) |

## Quick Start (Docker)

### Build

```bash
docker build -t ai-image-restoration .
```

Build takes ~5 minutes on first run (downloads PyTorch + model weights). Subsequent builds use cached layers and are much faster.

### Run

```bash
docker run -p 8000:8000 ai-image-restoration
```

The API is now available at `http://localhost:8000`.

### Verify

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{"status": "ok", "models": ["real-esrgan", "gfpgan", "fastNlMeans"]}
```

## API Endpoints

### GET /api/v1/health

Health check.

### GET /api/v1/models

List available models with descriptions and parameters.

### POST /api/v1/enhance/sr

Super resolution upscaling.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `file` | image | — | — | Input image (JPEG/PNG) |
| `scale` | int | 4 | 2–4 | Upscale factor |

```bash
curl -X POST http://localhost:8000/api/v1/enhance/sr \
  -F "file=@input.jpg" \
  -F "scale=4" \
  --output output_sr.png
```

Response headers include:
- `X-Processing-Time-Ms` — inference time in milliseconds
- `X-PSNR` — PSNR in dB vs bicubic baseline (higher = better)
- `X-Scale` — scale factor applied

### POST /api/v1/enhance/face

Face restoration and enhancement.

```bash
curl -X POST http://localhost:8000/api/v1/enhance/face \
  -F "file=@portrait.jpg" \
  --output output_face.png
```

### POST /api/v1/enhance/denoise

Color image denoising using OpenCV Non-Local Means.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `file` | image | — | — | Input image (JPEG/PNG) |
| `strength` | int | 10 | 1–30 | Filter strength (higher = more smoothing) |

```bash
curl -X POST http://localhost:8000/api/v1/enhance/denoise \
  -F "file=@noisy.jpg" \
  -F "strength=10" \
  --output output_denoised.png
```

**Strength guide:**
- 1–5: Light denoising, preserves fine texture
- 10–15: Balanced — good for typical camera noise
- 20–30: Heavy smoothing — use for very noisy images

## Local Development (without Docker)

Requires conda with the `cv` environment (Python 3.10, torch 2.0.1, numpy 1.26.4).

```bash
conda activate cv
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

## Project Structure

```
ai-image-restoration/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── models/
│   │   ├── sr_model.py      # Real-ESRGAN (singleton)
│   │   ├── face_model.py    # GFPGAN (singleton)
│   │   └── denoise_model.py # OpenCV FastNlMeans
│   └── utils.py         # Image I/O, validation, PSNR
├── tests/
├── Dockerfile
└── requirements.txt
```

## Notes

- Models load on first request and are cached in memory (singleton pattern) — first call has 2–5s overhead, subsequent calls are fast.
- PSNR is computed by comparing the SR output against a bicubic upscale of the same input, giving a meaningful quality delta over the baseline.
- Docker image uses CPU-only PyTorch. For GPU inference, replace the torch wheel URL in the Dockerfile with the CUDA variant.
