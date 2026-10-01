import os
import uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from groq import Groq
from tavily import TavilyClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Load API keys securely from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY or TAVILY_API_KEY in environment variables.")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Dynamically select an active text generation model from your Groq account
ACTIVE_LLM_MODEL = None
try:
    available_models = [m.id for m in groq_client.models.list().data]
    preferred_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "mixtral-8x7b-32768"
    ]
    for pref in preferred_models:
        if pref in available_models:
            ACTIVE_LLM_MODEL = pref
            break

    if not ACTIVE_LLM_MODEL:
        text_models = [m for m in available_models if "whisper" not in m and "safeguard" not in m]
        if text_models:
            ACTIVE_LLM_MODEL = text_models[0]

    print(f"--> Atom's Cognitive Engine initialized using model: [{ACTIVE_LLM_MODEL}]")
except Exception as err:
    print(f"Warning: Could not fetch active model list dynamically: {err}")
    ACTIVE_LLM_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------
# VECTOR MEMORY (Qdrant + Local Embeddings)
# ---------------------------------------------------------
print("Loading Local Embedding Model (SentenceTransformer)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

qdrant_client = QdrantClient(path="./atom_vector_db")

if not qdrant_client.collection_exists("atom_knowledge"):
    qdrant_client.create_collection(
        collection_name="atom_knowledge",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
print("Atom's Memory Initialized Successfully.")

# ---------------------------------------------------------
# FASTAPI LIFECYCLE HANDLER
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Clean shutdown of vector DB connection
    qdrant_client.close()

app = FastAPI(title="Atom Brain & Ears Server", lifespan=lifespan)

# ---------------------------------------------------------
# CORE PROCESSING ENDPOINT
# ---------------------------------------------------------
@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        temp_audio_path = "temp_incoming.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(await file.read())

        # --- STEP 1: THE EARS (Speech-to-Text via Whisper-V3) ---
        print("\n[EARS] Transcribing audio clip...")
        with open(temp_audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_audio_path, audio_file.read()),
                model="whisper-large-v3",
                temperature=0.0,
                prompt="No, I mean that as soon as you sit down one place now, where you come de go to? Hospital self, you de go, walking about."
            )

        raw_speech = transcription.text.strip()
        print(f"[EARS Raw]: \"{raw_speech}\"")

        if not raw_speech:
            return {"status": "error", "message": "Could not hear any spoken words clearly."}

        # --- STEP 1.5: UNIVERSAL ACCENT & PHONETIC NORMALIZATION PASS ---
        print("[EARS] Normalizing phonetic transcript for regional accents...")
        clean_prompt = (
            "You are an expert transcript corrector for global speech recognition and West African English/Pidgin. "
            "Fix speech-to-text phonetic mishearings, accent distortions, and mangled words in the transcript "
            "while retaining the speaker's original intent, language style, and slang. "
            "Output ONLY the corrected transcript text and nothing else.\n\n"
            f"Raw Transcript: \"{raw_speech}\"\n"
            "Corrected Transcript:"
        )

        # Uses dynamically chosen ACTIVE_LLM_MODEL instead of hardcoded model name
        norm_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": clean_prompt}],
            model=ACTIVE_LLM_MODEL,
            temperature=0.1
        )

        user_speech = norm_response.choices[0].message.content.strip()
        print(f"[EARS Cleaned]: \"{user_speech}\"")

        # --- STEP 2: THE BRAIN - MEMORY SEARCH (Qdrant Vector DB) ---
        print("[BRAIN] Searching internal memory...")
        query_vector = encoder.encode(user_speech).tolist()

        query_response = qdrant_client.query_points(
            collection_name="atom_knowledge",
            query=query_vector,
            limit=1
        )
        search_results = query_response.points

        memory_found = False
        context = ""
        source = ""

        if search_results and search_results[0].score > 0.80:
            memory_found = True
            context = search_results[0].payload["text"]
            source = "internal_memory"
            print(f"[BRAIN] Found memory match (Score: {search_results[0].score:.2f})")
        else:
            # --- STEP 3: THE BRAIN - LIVE WEB SEARCH (Tavily) ---
            print("[BRAIN] No prior memory found. Querying Live Web (Tavily)...")
            search_response = tavily_client.search(query=user_speech, search_depth="basic", max_results=2)
            context = " ".join([res['content'] for res in search_response.get('results', [])])
            source = "live_web_search"

            # Store new fact into Atom's Vector Memory for future recall
            qdrant_client.upsert(
                collection_name="atom_knowledge",
                points=[PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_vector,
                    payload={"text": context, "query": user_speech}
                )]
            )
            print("[BRAIN] Saved new findings to Qdrant memory.")

        # --- STEP 4: COGNITIVE REASONING ---
        system_instruction = (
            "You are Atom, an intelligent voice assistant. "
            f"Context provided: {context}. "
            "Formulate a direct, concise, conversational spoken response based on this context. "
            "Keep answers clear and ready for text-to-speech engine."
        )

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_speech}
            ],
            model=ACTIVE_LLM_MODEL
        )

        final_answer = response.choices[0].message.content

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return {
            "status": "success",
            "raw_heard_text": raw_speech,
            "heard_text": user_speech,
            "response": final_answer,
            "information_source": source,
            "model_used": ACTIVE_LLM_MODEL
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)