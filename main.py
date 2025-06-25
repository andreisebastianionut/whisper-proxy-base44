from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import tempfile
from typing import Dict
import uvicorn

# Creează aplicația FastAPI
app = FastAPI(title="Whisper Proxy API", version="1.0.0")

# Configurează CORS pentru a permite apeluri din base44
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # În producție, specifică doar domeniile necesare
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurează cheia API OpenAI din variabila de mediu
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY nu este setată în variabilele de mediu!")

@app.get("/")
async def root():
    """Endpoint de test pentru a verifica dacă API-ul funcționează"""
    return {"message": "Whisper Proxy API funcționează!", "status": "OK"}

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Transcrie un fișier audio folosind OpenAI Whisper
    
    Args:
        audio_file: Fișierul audio de transcris (wav, mp3, m4a, etc.)
    
    Returns:
        Dict cu transcrierea sau eroarea
    """
    
    # Verifică dacă fișierul a fost încărcat
    if not audio_file:
        raise HTTPException(status_code=400, detail="Nu a fost furnizat niciun fișier audio")
    
    # Verifică tipul fișierului
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/m4a", "audio/webm"]
    if audio_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipul fișierului nu este suportat. Tipuri permise: {', '.join(allowed_types)}"
        )
    
    try:
        # Citește conținutul fișierului
        audio_content = await audio_file.read()
        
        # Creează un fișier temporar pentru a-l trimite la OpenAI
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        try:
            # Transcrie folosind OpenAI Whisper
            with open(temp_file_path, "rb") as audio:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language="ro"  # Setează limba română
                )
            
            # Returnează rezultatul
            return {
                "success": True,
                "transcription": transcript.text,
                "filename": audio_file.filename
            }
            
        finally:
            # Șterge fișierul temporar
            os.unlink(temp_file_path)
            
    except Exception as e:
        # Gestionează erorile
        raise HTTPException(
            status_code=500, 
            detail=f"Eroare la transcrierea audio: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Endpoint pentru verificarea stării aplicației"""
    return {"status": "healthy", "service": "whisper-proxy"}

# Pentru rularea locală (opțional)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
