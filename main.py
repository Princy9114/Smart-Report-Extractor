from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.response import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file():
    return
