import asyncio
import json
import os
from pathlib import Path

import numpy as np
import websockets
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from vosk import Model, KaldiRecognizer
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Configuration
MODEL_PATH = os.getenv("VOSK_MODEL_PATH", r"vosk-model-small-en-us-0.15")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

templates = Jinja2Templates(directory="templates")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Vosk
print("🔄 Loading Vosk model...")
model = Model(MODEL_PATH)

# Initialize Gemini
client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = "gemini-2.5-flash"

app = FastAPI(title="VoiceScribe AI")

# Mount static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    text: str
    mode: str
    prompt: str


class AnalysisResponse(BaseModel):
    result: str


# Prompt templates
PROMPT_TEMPLATES = {
    "summary": """Provide a comprehensive yet concise summary of the following transcript.
Capture the main points, key insights, and overall context. Use clear paragraphs.

Transcript:
{text}

Summary:""",
    "bullet": """Convert the following transcript into well-organized bullet points.
Extract key information, main ideas, and important details. Use nested bullets for sub-points if needed.

Transcript:
{text}

Key Points:""",
    "concise": """Rewrite the following transcript to be extremely concise and to-the-point.
Remove all filler words (um, uh, like, you know), repetitions, and unnecessary details.
Keep only the essential information.

Transcript:
{text}

Concise Version:""",
    "action": """Analyze the following transcript and extract all action items, tasks, deadlines, and decisions.
Format as a clear checklist with checkboxes. Include who should do what if mentioned.

Transcript:
{text}

Action Items:""",
}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")

    rec = KaldiRecognizer(model, 16000)

    try:
        while True:
            data = await websocket.receive_bytes()

            # Convert bytes to numpy array then to bytes for Vosk
            audio_data = np.frombuffer(data, dtype=np.int16).tobytes()

            if rec.AcceptWaveform(audio_data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    await websocket.send_json({"type": "final", "text": text})
            else:
                partial = json.loads(rec.PartialResult())
                text = partial.get("partial", "")
                if text:
                    await websocket.send_json({"type": "partial", "text": text})

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🔌 Client disconnected")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze transcript with Gemini AI"""
    try:
        if request.mode in PROMPT_TEMPLATES:
            prompt = PROMPT_TEMPLATES[request.mode].format(text=request.text)
        else:
            prompt = request.prompt

        print(f"🤖 Gemini analyzing with mode: {request.mode}")
        print(f"🤖 Prompt length: {len(prompt)}")

        response = client.models.generate_content(model=gemini_model, contents=prompt)

        print(f"🤖 Response type: {type(response)}")
        print(f"🤖 Response attributes: {dir(response)}")

        # Extract text from response
        result_text = response.text if hasattr(response, "text") else str(response)

        return AnalysisResponse(result=result_text)

    except Exception as e:
        import traceback

        print(f"❌ Full error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting VoiceScribe AI...")
    print(f"📍 Vosk Model: {MODEL_PATH}")

    port = int(os.environ.get("PORT", 8000))

    print(f"🌐 Server running on 0.0.0.0:{port}")

    uvicorn.run(app, host="0.0.0.0", port=port)
