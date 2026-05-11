from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io
import time

from app.models import enhance_sr, enhance_face
from app.utils import validate_image, load_image, image_to_bytes, compute_psnr

app = FastAPI(
    title="AI Image Restoration API",
    description="Super resolution and face restoration using Real-ESRGAN and GFPGAN",
    version="1.0.0",
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "models": ["real-esrgan", "gfpgan"]}


@app.get("/api/v1/models")
def list_models():
    return {
        "models": [
            {"id": "sr", "name": "Real-ESRGAN", "description": "Super resolution (2x, 4x)", "scales": [2, 4]},
            {"id": "face", "name": "GFPGAN", "description": "Face restoration and enhancement"},
        ]
    }


@app.post("/api/v1/enhance/sr")
async def enhance_super_resolution(
    file: UploadFile = File(...),
    scale: int = Query(default=4, ge=2, le=4),
):
    contents = await file.read()
    valid, error = validate_image(contents)
    if not valid:
        raise HTTPException(status_code=400, detail=error)

    image = load_image(contents)
    start = time.time()
    enhanced = enhance_sr(image, scale=scale)
    elapsed = int((time.time() - start) * 1000)

    psnr = compute_psnr(image, enhanced[:image.shape[0], :image.shape[1]])
    result_bytes = image_to_bytes(enhanced)

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/png",
        headers={
            "X-Processing-Time-Ms": str(elapsed),
            "X-PSNR": f"{psnr:.2f}",
            "X-Scale": str(scale),
        }
    )


@app.post("/api/v1/enhance/face")
async def enhance_face_restoration(
    file: UploadFile = File(...),
):
    contents = await file.read()
    valid, error = validate_image(contents)
    if not valid:
        raise HTTPException(status_code=400, detail=error)

    image = load_image(contents)
    start = time.time()
    enhanced = enhance_face(image)
    elapsed = int((time.time() - start) * 1000)

    result_bytes = image_to_bytes(enhanced)

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/png",
        headers={"X-Processing-Time-Ms": str(elapsed)},
    )
