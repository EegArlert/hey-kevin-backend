import os
import requests
import json
import uuid
from fastapi.responses import JSONResponse
from fastapi import FastAPI

def process_bing_and_gpt(reduced_image_size_path, session_id):
    print("[BACKGROUND] Task started with session_id:", session_id)
    try:
        print("[BACKGROUND] Starting Bing + GPT processing...")

        # Bing API req
        bing_url = os.getenv("BING_LOCALHOST_PATH")
        
        headers = {"x-api-key": os.getenv("FEED_SERVER_API_KEY", "")}
        with open(reduced_image_size_path, "rb") as img_file:
            files = {"file": ("reduced.jpg", img_file.read(), "image/jpeg")}

        response = requests.post(bing_url, files=files, headers=headers)
        
        if response.status_code != 200:
            print(f"[BING ERROR] Status {response.status_code}: {response.text}")
            return

        bing_data = response.json()
        print("[BING] Visual Search completed.")

        # Manipulate Bing API resp retrieve best match label
        query_label = None
        if isinstance(bing_data.get("query"), list):
            query_label = bing_data.get("query")[0]

        if not query_label:
            print("[BING] No label found in result.")
            return

        print(f"[BING] Detected object: {query_label}")

        # Forward label to gpt service api
        gpt_url = os.getenv("OPENAI_LOCALHOST_PATH")
        payload = {"label": query_label}
        headers = {
            "Content-Type": "application/json",
            "x-api-key": os.getenv("FEED_SERVER_API_KEY", "")
        }
        print("[DEBUG] Using API Key:", headers["x-api-key"])
        gpt_response = requests.post(gpt_url, data=json.dumps(payload), headers=headers)

        if gpt_response.status_code != 200:
            print(f"[GPT ERROR] Status {gpt_response.status_code}: {gpt_response.text}")
            return

        gpt_data = gpt_response.json()
        print("[GPT] Comments received:")

        for style, comment in gpt_data.items():
            print(f"  - {style}: {comment}")
            
        result = {
            "message":  "Bing + GPT completed.",
            "label": query_label,
            "comments": gpt_data
        }

        # Save to file using session ID
        os.makedirs("/app/status", exist_ok=True)
        with open(f"/app/status/{session_id}.json", "w") as f:
            json.dump(result, f)

        print(f"[BACKGROUND] Results saved to /app/status/{session_id}.json")
            

    except Exception as e:
        print(f"[BACKGROUND ERROR] {str(e)}")
