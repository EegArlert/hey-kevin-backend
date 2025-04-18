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

def compress_img(image_path):
    img = cv2.imread(image_path)

    max_size_mb=1.0
    scale_factor = 0.4  # Adjust as needed to get desired size
    quality = 95  # keeps the return quality of the image high

    #SEE: https://stackoverflow.com/questions/66311867/how-to-scale-down-an-image-to-1mb
    #SEE: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga8ac397bd09e48851665edbe12aa28f25
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(image_path, resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    file_size = os.path.getsize(image_path) / (1024 * 1024)

    print(f"Image resized to {width}x{height} and compressed to {file_size:.2f} MB")

    if file_size > max_size_mb:
        print(f"Warning: Image size exceeds {max_size_mb}MB.")
        #in which case reduced quality or scale would help
        #need to add error handling to this

def blur_and_outline(image, mask, outline_color=(0, 255, 255), outline_thickness=3, blur_strength=45, darken_factor=0.4):
    # Prep mask
    mask_bin = (mask > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask_bin, (image.shape[1], image.shape[0]))
    mask_3d = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)

    # Blur background
    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

    # Darken background
    darkened = (blurred * darken_factor).astype(np.uint8)

    # Combine sharp subject with dark blurry background
    combined = (image * mask_3d + darkened * (1 - mask_3d)).astype(np.uint8)

    # Draw outline on top
    outlined_img = combined.copy()
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outlined_img, contours, -1, outline_color, thickness=outline_thickness)

    return outlined_img

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
            segmented_img = blur_and_outline(segmented_img, mask)
            

        # Creating a unique image path for each session
        filename_segmented = f"{session_id}-segmented.jpg"
        filename_reduced = f"{session_id}-reduced.jpg"
        segmented_img_path = f"/app/images/{filename_segmented}"
        reduced_img_size_path = f"/app/images/{filename_reduced}"
        
        
        # Segmenting the image
        cv2.imwrite(segmented_img_path, segmented_img)
        print("[INFO] Segmented image successfully saved")
        
        # Reducing the image size function
        cv2.imwrite(reduced_img_size_path, img)
        compress_img(reduced_img_size_path)

        print("[INFO] Compressed image successfully saved .")

        # Trigger background task for Bing + GPT
        print("Background task being added")
        background_tasks.add_task(process_bing_and_gpt, reduced_img_size_path, session_id)
        print("Background task finished")

        return JSONResponse(content={
            "message": "Segmentation completed.",
            "mask": json_masks,
            "segmented_image_path": f"{os.getenv('LOCALHOST_IMAGE_PATH_SEG')}/images/{filename_segmented}",
            "reduced_image_size_path": f"{os.getenv('LOCALHOST_IMAGE_PATH_RED')}/images/{filename_reduced}",
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