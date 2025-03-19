import os
import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import SAM
import torch

print("[DEBUG] Torch Device:", torch.cuda.is_available())

# Loads .env locally
load_dotenv()

app = FastAPI()

image_dir = "/app/images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)

# Serve images statically at "/images"
app.mount("/images", StaticFiles(directory="/app/images"), name="images")

# Load YOLO model
model = SAM("mobile_sam.pt")

# def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
#     """Apply a segmentation mask as a transparent overlay on an image"""
#     # Convert to binary mask
#     mask = (mask > 0.5).astype(np.uint8) * 255
#     # Resize mask
#     mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)  
#     colored_mask = np.zeros_like(image)
#      # Apply color
#     colored_mask[:, :] = color
#     # Blend mask with image
#     overlayed = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)  
#     # Keep non-mask areas unchanged
#     overlayed[mask_resized == 0] = image[mask_resized == 0]  
#     print(np.size(overlayed))
#     return overlayed

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Apply a segmentation mask as an overlay on an image."""
    mask = (mask > 0.5).astype(np.uint8) * 255  # Convert mask to binary
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure mask is properly applied
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    overlayed = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    overlayed[mask_resized == 0] = image[mask_resized == 0]

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

        print("[DEBUG] Original image shape:", img.shape)

        img_h, img_w, _ = img.shape  # Get image dimensions

        # Define a bounding box around the center (adjust size as needed)
        box_size = 900  # Change this to control the bounding box size
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

        # Apply mask to the image
        segmented_img = img.copy()
        for mask in masks:
            segmented_img = overlay_mask(segmented_img, mask)

        # Save segmented image
        segmented_img_path = "/app/images/segmented.jpg"
        cv2.imwrite(segmented_img_path, segmented_img)

        print("[INFO] Segmented image saved successfully.")

        # Return processed image path
        return JSONResponse(content={
            "message": "Segmentation completed.",
            "segmented_image_path": str(os.getenv("IMAGE_URL"))
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
# @app.post("/segment")
# async def segment(file: UploadFile = File(...)):
#     try:
#         # Read image file
#         contents = await file.read()
#         print("[DEBUG] Received image file, size:", len(contents))
#         np_arr = np.frombuffer(contents, np.uint8)
#         print("[DEBUG] Converted to numpy array, shape:", np_arr.shape)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if img is None:
#             print("[ERROR] Image decoding failed!")
#             return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

#         print("[DEBUG] Image successfully decoded, shape:", img.shape)

#         # Run YOLO segmentation
#         # Reduce image size to 25%
#         scale_factor = 0.25  # Resize to 25% of original size
#         new_width = int(img.shape[1] * scale_factor)
#         new_height = int(img.shape[0] * scale_factor)

#         img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
#         print("[DEBUG] Resized image shape:", img_resized.shape)
#         # predictions = model.predict(img, conf=0.9, save=False)
#         # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
#         # print("[DEBUG] Resized image to:", img.shape)

#         # Run SAM segmentation
#         print("[DEBUG] Running SAM model...")
#         predictions = model(img)  # Make sure this works
#         # Extract bounding boxes & masks
#         # result = predictions[0]
#         # boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
#         # masks = result.masks.data.cpu().numpy() if result.masks is not None else None

#         # print(f"[INFO] Detected {len(boxes)} objects")

#         # if not boxes:
#         #     print("[INFO] No object detected.")
#         #     return JSONResponse(content={"message": "No object detected", "cropped_image": None})
        
#         # Get the first prediction result
#         result = predictions[0]

#         # Ensure masks exist
#         if result.masks is None:
#             print("[INFO] No segmentation masks found.")
#             return JSONResponse(content={"message": "No segmentation masks found", "segmented_image": None})

#         # Get all masks
#         masks = result.masks.data.cpu().numpy()

#         # Apply all masks to the image
#         segmented_img = img.copy()
#         for mask in masks:
#             segmented_img = overlay_mask(segmented_img, mask)

#         # Save segmented image
#         segmented_img_path = "/app/images/segmented.jpg"
#         cv2.imwrite(segmented_img_path, segmented_img)

#         print("[INFO] Segmented image saved successfully.")

#         # Return the processed image path
#         return JSONResponse(content={
#             "message": "Segmentation completed.",
#             "segmented_image_path": str(os.getenv("IMAGE_URL"))
#         })


#         # Get image center
#         img_h, img_w, _ = img.shape
#         img_center = np.array([img_w / 2, img_h / 2])

#         # Find the box closest to the center
#         # Find the most centered object
#         def distance_from_center(box):
#             box_x_center = (box[0] + box[2]) / 2
#             box_y_center = (box[1] + box[3]) / 2
#             return np.linalg.norm(np.array([box_x_center, box_y_center]) - img_center)

#         # Select the most centered object
#         centered_idx = min(range(len(boxes)), key=lambda i: distance_from_center(boxes[i]))

#         # Get the bounding box and mask for the centered object
#         centered_box = boxes[centered_idx]
#         mask = masks[centered_idx] if masks is not None else None

#         print(f"[INFO] Most centered object bounding box: {centered_box}")

#         # Convert bbox to integers
#         x1, y1, x2, y2 = map(int, centered_box)
        
#         box_width = x2 - x1
#         box_height = y2 - y1
#         margin_percentage = 0.2  # Adjust this value as needed

#         x1 = max(0, int(x1 - box_width * margin_percentage))
#         y1 = max(0, int(y1 - box_height * margin_percentage))
#         x2 = min(img_w, int(x2 + box_width * margin_percentage))
#         y2 = min(img_h, int(y2 + box_height * margin_percentage))

#         # Resize the mask to match the **original image dimensions** BEFORE cropping
#         if mask is not None:
#             print("[INFO] Resizing mask to match image dimensions before cropping...")
#             mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

#             # Ensure cropped area is within valid bounds
#             x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

#             # Crop the image and mask
#             cropped_img = img[y1:y2, x1:x2]
#             mask_cropped = mask_resized[y1:y2, x1:x2]

#             print("[INFO] Applying segmentation mask...")
#             cropped_img = overlay_mask(cropped_img, mask_cropped)
#             # noncropped_img = overlay_mask(img, mask)
#         else:
#             print("[INFO] No mask found, returning only cropped image.")
#             cropped_img = img[y1:y2, x1:x2]

#         # Save cropped image
#         cropped_img_path = "/app/images/cropped_segmented.jpg"
#         cv2.imwrite(cropped_img_path, cropped_img)

#         print("[INFO] Cropped image saved successfully.")
        
#         return JSONResponse(content={
#             "message": "Segmentation completed.",
#             "cropped_image_path": str(os.getenv("IMAGE_URL"))
#             # "cropped_image_path": cropped_img_path
#         })

#     except Exception as e:
#         print("[ERROR]", str(e))
#         return JSONResponse(status_code=500, content={"error": str(e)})

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)