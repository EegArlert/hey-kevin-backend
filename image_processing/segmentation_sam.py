import os
import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Load environment variables
load_dotenv()

app = FastAPI()

# Directories
image_dir = "/app/images"
model_dir = "/app/sam"  # Adjust your path

# Ensure directories exist
os.makedirs(image_dir, exist_ok=True)

# Serve images statically
app.mount("/images", StaticFiles(directory=image_dir), name="images")

# Load SAM model
sam_checkpoint = os.path.join(model_dir, "sam_vit_h_4b8939.pth")  # Adjust to your model
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Apply a segmentation mask as an overlay on an image."""
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    overlayed = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    overlayed[mask_resized == 0] = image[mask_resized == 0]
    return overlayed

def generate_sam_masks(image_path):
    """Load an image and generate segmentation masks using SAM."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)
    return masks

@app.post("/test")
async def test_api(request: Request):
    print(f"[DEBUG] Received test request")
    return {"message": "Test successful"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[DEBUG] Incoming request: {request.method} {request.url}")
    print(f"[DEBUG] Headers: {request.headers}")
    response = await call_next(request)
    print(f"[DEBUG] Response Status: {response.status_code}")
    return response

@app.post("/segmentsam")
async def segment(file: UploadFile = File(...)):
    try:
        print("[INFO] Received request at /segment")
        print("[INFO] Received request at /segment ")
        # Read and save the image
        contents = await file.read()
        img_path = os.path.join(image_dir, file.filename)
        with open(img_path, "wb") as f:
            f.write(contents)

        # Load and process image
        img = cv2.imread(img_path)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Failed to load image"})

        print("[INFO] Image received. Running SAM segmentation...")

        # Generate segmentation masks
        masks = generate_sam_masks(img_path)

        if not masks:
            return JSONResponse(content={"message": "No segmentation masks found", "segmented_image": None})

        # Overlay masks on the image
        segmented_img = img.copy()
        for mask_data in masks:
            segmented_img = overlay_mask(segmented_img, mask_data['segmentation'])

        # Save segmented image
        segmented_img_path = os.path.join(image_dir, "segmented.jpg")
        cv2.imwrite(segmented_img_path, segmented_img)

        print("[INFO] Segmented image saved successfully.")

        return JSONResponse(content={
            "message": "Segmentation completed.",
            "segmented_image_path": f"/images/segmented.jpg"
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
