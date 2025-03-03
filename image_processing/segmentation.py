import os
import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

image_dir = "/app/images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)

# Serve images statically at "/images"
app.mount("/images", StaticFiles(directory="/app/images"), name="images")

# Load YOLO model
model = YOLO("yolo11x-seg.pt")

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Apply a segmentation mask as a transparent overlay on an image"""
    # Convert to binary mask
    mask = (mask * 255).astype(np.uint8)  
    # Resize mask
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)  
    colored_mask = np.zeros_like(image)
     # Apply color
    colored_mask[:, :] = color
    # Blend mask with image
    overlayed = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)  
    # Keep non-mask areas unchanged
    overlayed[mask_resized == 0] = image[mask_resized == 0]  
    print(np.size(overlayed))
    return overlayed

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

        print("[INFO] Image received. Running YOLO segmentation...")

        # Run YOLO segmentation
        predictions = model.predict(img, conf=0.2, save=False)

        # Extract bounding boxes & masks
        result = predictions[0]
        boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None

        print(f"[INFO] Detected {len(boxes)} objects")

        if not boxes:
            print("[INFO] No object detected.")
            return JSONResponse(content={"message": "No object detected", "cropped_image": None})

        # Get image center
        img_h, img_w, _ = img.shape
        img_center = np.array([img_w / 2, img_h / 2])

        # Find the box closest to the center
        def box_center(box):
            return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        closest_idx = min(range(len(boxes)), key=lambda i: np.linalg.norm(box_center(boxes[i]) - img_center))
        closest_box = boxes[closest_idx]
        mask = masks[closest_idx] if masks is not None else None

        print(f"[INFO] Closest object bounding box: {closest_box}")

        # Convert bbox to integers
        x1, y1, x2, y2 = map(int, closest_box)
        
        box_width = x2 - x1
        box_height = y2 - y1
        margin_percentage = 0.2  # Adjust this value as needed

        x1 = max(0, int(x1 - box_width * margin_percentage))
        y1 = max(0, int(y1 - box_height * margin_percentage))
        x2 = min(img_w, int(x2 + box_width * margin_percentage))
        y2 = min(img_h, int(y2 + box_height * margin_percentage))

        # Resize the mask to match the **original image dimensions** BEFORE cropping
        if mask is not None:
            print("[INFO] Resizing mask to match image dimensions before cropping...")
            mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            # Ensure cropped area is within valid bounds
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

            # Crop the image and mask
            cropped_img = img[y1:y2, x1:x2]
            mask_cropped = mask_resized[y1:y2, x1:x2]

            print("[INFO] Applying segmentation mask...")
            cropped_img = overlay_mask(cropped_img, mask_cropped)
            # noncropped_img = overlay_mask(img, mask)
        else:
            print("[INFO] No mask found, returning only cropped image.")
            cropped_img = img[y1:y2, x1:x2]

        # Save cropped image
        cropped_img_path = "/app/images/cropped_segmented.jpg"
        cv2.imwrite(cropped_img_path, cropped_img)

        print("[INFO] Cropped image saved successfully.")

        # return JSONResponse(content={
        #     "message": "Segmentation completed.",
        #     "cropped_image_path": cropped_img_path
        # })
        return JSONResponse(content={
            "message": "Segmentation completed.",
            "cropped_image_path": f"{IMAGE_URL}"
            # "cropped_image_path": cropped_img_path
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)