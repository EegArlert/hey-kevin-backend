import os
import cv2
import numpy as np
import uvicorn
import json
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from image_processing.middleware import api_key_middleware
from ultralytics import SAM

# Loads .env locally
load_dotenv()

app = FastAPI()
api_key_middleware(app)

image_dir = "/app/images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)

# Serve images statically at "/images"
app.mount("/images", StaticFiles(directory="/app/images"), name="images")

# Load YOLO model
model = SAM("mobile_sam.pt")

def overlay_mask_with_outline(image, mask, color=(0, 255, 0), alpha=0.5, outline_thickness=5):
    # Convert to binary mask
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # === Draw the colored region only where mask is active ===
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
async def segment(file: UploadFile = File(...)):
    try:
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
        box_size = 700  # Change this to control the bounding box size
        x_min = max(0, img_w // 2 - box_size)
        y_min = max(0, img_h // 2 - box_size)
        x_max = min(img_w, img_w // 2 + box_size)
        y_max = min(img_h, img_h // 2 + box_size)

        # Run segmentation on this region
        results = model(img, bboxes=[x_min, y_min, x_max, y_max])

        print("[DEBUG] SAM model finished processing.")

        # Ensure masks exist
        result = results[0]
        if result.masks is None:
            print("[INFO] No segmentation masks found.")
            return JSONResponse(content={"message": "No segmentation masks found", "segmented_image": None})

        # Get segmentation mask
        masks = result.masks.data.cpu().numpy()
        # print(masks)
        
        json_masks = json.dumps({'nums': masks.tolist()})
        # json_masks = pd.Series(masks).to_json(orient='values')
        print(json_masks)
        print("converting to json_masks successful")

        # Apply mask to the image
        segmented_img = img.copy()
        for mask in masks:
            segmented_img = overlay_mask_with_outline(segmented_img, mask)
            
        print ([mask.astype(int).tolist() for mask in masks])

        # Save segmented image
        segmented_img_path = "/app/images/segmented.jpg"
        cv2.imwrite(segmented_img_path, segmented_img)

        print("[INFO] Segmented image saved successfully.")

        # Return processed image path
        return JSONResponse(content={
            "message": "Segmentation completed.",
            "mask": json_masks
            # "segmented_image_path": str(os.getenv("IMAGE_URL"))
        })
 
    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)