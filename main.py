import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY ortam değişkeni tanımlı değil.")

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY)

# Sohbet modeli olmayanları ele
EXCLUDE = ("whisper", "guard", "tts", "orpheus", "playai", "embed", "safeguard")
# Bu sırayla tercih et (eşleşmeyenler en sona gider)
PREFER = ("gpt-oss-120b", "qwen", "gpt-oss-20b", "allam")

_models_cache = None


def get_models() -> list[str]:
    """Hesabın gerçekten erişebildiği modelleri Groq'tan çeker."""
    global _models_cache
    if _models_cache is None:
        ids = [m.id for m in client.models.list().data]
        ids = [i for i in ids if not any(x in i.lower() for x in EXCLUDE)]
        ids.sort(
            key=lambda i: next(
                (n for n, p in enumerate(PREFER) if p in i.lower()), len(PREFER)
            )
        )
        _models_cache = ids
    return _models_cache


def ask_groq(system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
    errors = []
    try:
        models = get_models()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model listesi alınamadı (API anahtarını kontrol et): {e}",
        )

    for model_name in models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            errors.append(f"{model_name}: {str(e)[:150]}")

    global _models_cache
    _models_cache = None  # bir sonraki istekte listeyi yenile
    raise HTTPException(
        status_code=502,
        detail="Hiçbir model çalışmadı. " + " | ".join(errors),
    )


class TextRequest(BaseModel):
    text: str


@app.post("/api/humanize")
def humanize_text(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    system_prompt = (
        "Sen profesyonel bir metin insanlaştırıcısısın. Verilen yapay zeka "
        "metinlerindeki robotik kalıpları, aşırı resmi dili ve tekdüze cümle "
        "yapılarını boz. Cümle uzunluklarını çeşitlendir ve daha doğal, akıcı "
        "bir insan anlatımıyla yeniden yaz. Anlam bütünlüğünü koru. "
        "Sadece yeniden yazılmış metni ver, başka açıklama ekleme."
    )
    return {"result": ask_groq(system_prompt, req.text, temperature=0.9)}


@app.post("/api/detect")
def detect_ai(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    system_prompt = (
        "Sen çok katı ve keskin bir AI Dedektörüsün. Metnin yapay zeka olup "
        "olmadığını analiz et. Yanıtını KESİNLİKLE şu geçerli JSON formatında "
        'ver, başka hiçbir açıklama ekleme: {"ai_score": 90, "human_score": '
        '10, "comment": "kısa yorum"}. ai_score ve human_score toplamı 100 olmalı.'
    )
    raw = ask_groq(system_prompt, req.text, temperature=0.0)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise HTTPException(status_code=502, detail=f"Model JSON döndürmedi: {raw[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"JSON okunamadı: {raw[:200]}")


@app.get("/api/models")
def list_models():
    return get_models()


app.mount("/", StaticFiles(directory="static", html=True), name="static")