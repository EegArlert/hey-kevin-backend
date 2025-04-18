from fastapi import Request, HTTPException
import os

def api_key_middleware(app):
    allowed_keys = os.getenv("SERVER_API_KEYS", "").split(",")

    @app.middleware("http")
    async def verify_api_key(request: Request, call_next):
        path = request.url.path

        # Don't require API key for static files (like segmented image)
        if path.startswith("/images") or path.startswith("/status"):
            return await call_next(request)

        token = request.headers.get("x-api-key")
        if token not in allowed_keys:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")

        return await call_next(request)

    return app

