import os
import cv2
import json
import uuid
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import sys
sys.path.append("..")
from middleware import api_key_middleware
from ultralytics import SAM
from image_processing.background_processor import process_bing_and_gpt

# Loads .env locally
load_dotenv()

app = FastAPI()
api_key_middleware(app)

image_dir = "/app/images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)

# Serve images statically at "/images"
app.mount("/images", StaticFiles(directory="/app/images"), name="images")

model = SAM("mobile_sam.pt")

def overlay_mask_with_outline(image, mask, color=(0, 255, 0), alpha=0.5, outline_thickness=5):
    # Convert to binary mask
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Draw the colored region only where mask is active
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color

    # Apply blending only to masked area
    overlay[mask_resized > 0] = cv2.addWeighted(
        image[mask_resized > 0], 1 - alpha, colored_mask[mask_resized > 0], alpha, 0
    )

    # === Find contours and draw a thick outline ===
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=outline_thickness)

    return overlay

@app.post("/segment")
async def segment(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())
        print(f"current session ID {session_id}")
        # Read image file
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print("[ERROR] Image decoding failed!")
            return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

        print("[DEBUG] Original image shape:", img.shape)

        img_h, img_w, _ = img.shape  # Get image dimensions

        # Define a bounding box around the center (adjust size as needed)
        box_size = 700
        x_min = max(0, img_w // 2 - box_size)
        y_min = max(0, img_h // 2 - box_size)
        x_max = min(img_w, img_w // 2 + box_size)
        y_max = min(img_h, img_h // 2 + box_size)

        # Run segmentation on this region
        results = model(img, bboxes=[x_min, y_min, x_max, y_max])

        print("[DEBUG] SAM model finished processing.")

        result = results[0]
        if result.masks is None:
            print("[INFO] No segmentation masks found.")
            return JSONResponse(content={"message": "No segmentation masks found", "segmented_image": None})

        masks = result.masks.data.cpu().numpy()
        json_masks = json.dumps({'nums': masks.tolist()})
        print("converting to json_masks successful")

        segmented_img = img.copy()
        for mask in masks:
            segmented_img = overlay_mask_with_outline(segmented_img, mask)

        segmented_img_path = "/app/images/segmented.jpg"
        cv2.imwrite(segmented_img_path, segmented_img)

        print("[INFO] Segmented image saved successfully.")

        # Trigger background task for Bing + GPT
        print("Background task being added")
        background_tasks.add_task(process_bing_and_gpt, segmented_img_path, session_id)
        print("Background task finished")

        return JSONResponse(content={
            "message": "Segmentation completed.",
            "mask": json_masks,
            "segmented_image_path": os.getenv("LOCAL_HOST_IMAGE_PATH"),
            "session_id": session_id
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)}) 
    

    
# Method to check the status of session_id. Return gpt api response if found.
@app.get("/status/{session_id}")
async def get_status(session_id: str):
    try:
        file_path = f"/app/status/{session_id}.json"
        if not os.path.exists(file_path):
            return JSONResponse(status_code=202, content={"status":"processing"})
        
        with open(file_path, "r") as f:
            result = json.load(f)
            return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})