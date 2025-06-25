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

# Configurează cheia API OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY nu este setată în variabilele de mediu!")

# Setează cheia API pentru openai (versiunea 1.3.8)
openai.api_key = openai_api_key

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
    
    # Debug info
    print(f"Fișier primit: {audio_file.filename}")
    print(f"Content-Type: {audio_file.content_type}")
    print(f"Dimensiune: {audio_file.size if hasattr(audio_file, 'size') else 'necunoscută'}")
    
    # Verifică tipul fișierului - verificare mai relaxată pentru base44
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/m4a", "audio/webm", "application/octet-stream", "audio/wave"]
    
    # Acceptă și fișiere fără content-type sau cu content-type generic
    if audio_file.content_type and audio_file.content_type not in allowed_types:
        # Verifică și extensia fișierului ca backup
        filename = audio_file.filename or ""
        allowed_extensions = [".wav", ".mp3", ".m4a", ".webm", ".mp4"]
        
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"Tipul fișierului nu este suportat. Content-type: {audio_file.content_type}, Filename: {filename}"
            )
    
    try:
        # Citește conținutul fișierului
        audio_content = await audio_file.read()
        
        # Creează un fișier temporar pentru a-l trimite la OpenAI
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        try:
            # Transcrie folosind OpenAI Whisper (sintaxa pentru v1.3.8)
            with open(temp_file_path, "rb") as audio:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio,
                    language="ro"  # Setează limba română
                )
                
                # În versiunea 1.3.8, response este un dicționar
                transcript_text = response.get("text", "") if isinstance(response, dict) else str(response)
            
            # Returnează rezultatul
            return {
                "success": True,
                "transcription": transcript_text,
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

@app.get("/transcribe")
async def transcribe_info():
    """Informații despre endpoint-ul de transcriere (pentru accesul din browser)"""
    return {
        "message": "Endpoint pentru transcrierea audio",
        "method": "POST",
        "required": "Fișier audio în format multipart/form-data",
        "supported_formats": ["wav", "mp3", "m4a", "webm"],
        "example": "Folosiți POST cu un fișier audio pentru a obține transcrierea"
    }

@app.get("/test-openai")
async def test_openai_connection():
    """Test pentru a verifica conectivitatea cu OpenAI"""
    try:
        # Testează conexiunea cu OpenAI (versiunea 1.3.8)
        models = openai.Model.list()
        return {
            "status": "success",
            "message": "Conexiunea cu OpenAI funcționează",
            "api_key_present": bool(openai_api_key),
            "openai_configured": bool(openai.api_key)
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Eroare la conectarea cu OpenAI: {str(e)}",
            "api_key_present": bool(openai_api_key),
            "openai_configured": bool(openai.api_key)
        }

@app.get("/health")
async def health_check():
    """Endpoint pentru verificarea stării aplicației"""
    return {"status": "healthy", "service": "whisper-proxy"}

# Pentru rularea locală (opțional)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
