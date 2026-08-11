"""
main.py  –  v2.0
================
FastAPI backend for the Vernacular Voice-First Anti-Doping RAG application.
"""

import os
import re
import time
import pickle
import sqlite3 
import logging
from contextlib import asynccontextmanager, contextmanager
from io import BytesIO
from typing import Optional
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    import psycopg2
    from psycopg2.extras import DictCursor
    from psycopg2.pool import ThreadedConnectionPool
    HAS_PG = True
except ImportError:
    HAS_PG = False



# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
CHUNKS_PATH      = os.getenv("CHUNKS_PATH",      "chunks.pkl")
SQLITE_DB_PATH   = os.getenv("SQLITE_DB_PATH",   "chats_v2.db")
DATABASE_URL     = os.getenv("DATABASE_URL")
IS_POSTGRES      = HAS_PG and DATABASE_URL is not None and (DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://"))
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL        = "llama-3.3-70b-versatile"
WHISPER_MODEL    = "whisper-large-v3"
TOP_K_RETRIEVAL  = 3    # FAISS chunks retrieved per query
MEMORY_WINDOW    = 10   # Context messages fed into LLM
HISTORY_LIMIT    = 20   # Default /history page size

# Startup timestamp (for /health uptime reporting)
_START_TIME = time.time()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    faiss_index:  Optional[faiss.Index]            = None
    chunks:       Optional[list[str]]              = None
    embedder:     Optional[SentenceTransformer]    = None
    groq_client:  Optional[Groq]                   = None
    pg_pool:      Optional[ThreadedConnectionPool] = None
    use_postgres: bool                             = IS_POSTGRES  # can flip to False at runtime

state = AppState()


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class TextQueryRequest(BaseModel):
    query: str

class VerifyResponse(BaseModel):
    chat_id:      int
    user_query:   str
    ai_response:  str
    audio_base64: Optional[str] = None
    language:     str = "hindi"
    risk_level:   str = "unknown"

class ChatSession(BaseModel):
    id: int
    title: str
    created_at: str

class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    risk_level: str
    language: str
    timestamp: str

class ChatDetailResponse(BaseModel):
    chat_id: int
    title: str
    messages: list[MessageItem]


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_start = time.time()
    log.info("⏳ Starting production-grade system initialization...")

    # 1. DB connection pool setup & migrations
    db_start = time.time()
    if state.use_postgres:
        url = DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        try:
            state.pg_pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=url)
            log.info("✅ PostgreSQL Connection Pool initialized successfully")
        except Exception as e:
            log.warning(
                f"⚠️  PostgreSQL unreachable — falling back to SQLite.\n"
                f"    Reason: {e}\n"
                f"    This is expected on HF Spaces free tier (no outbound IPv6)."
            )
            state.use_postgres = False
            state.pg_pool = None

    try:
        init_db()
        with db_session() as conn:
            if state.use_postgres:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            else:
                conn.execute("SELECT 1").fetchone()
        db_init_time = time.time() - db_start
        db_type = "PostgreSQL" if state.use_postgres else "SQLite"
        log.info(f"✅ Database initialised ({db_type}) & connectivity verified in {db_init_time:.4f}s")
    except Exception as e:
        log.critical(f"💥 Database startup check failed: {e}")
        raise RuntimeError(f"Database Startup Failure: {e}")

    # 2. FAISS index load
    faiss_start = time.time()
    log.info("⏳ Loading FAISS index…")
    if not os.path.exists(FAISS_INDEX_PATH):
        log.critical(f"❌ FAISS index not found at '{FAISS_INDEX_PATH}'!")
        raise RuntimeError(f"FAISS index file missing: {FAISS_INDEX_PATH}")
    
    try:
        state.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        faiss_load_time = time.time() - faiss_start
        log.info(f"✅ FAISS loaded – {state.faiss_index.ntotal} vectors in {faiss_load_time:.4f}s")
    except Exception as e:
        log.critical(f"💥 Failed to load FAISS index: {e}")
        raise RuntimeError(f"FAISS Index Load Failure: {e}")

    # 3. Chunks load
    chunks_start = time.time()
    log.info("⏳ Loading text chunks…")
    if not os.path.exists(CHUNKS_PATH):
        log.critical(f"❌ Chunks file not found at '{CHUNKS_PATH}'!")
        raise RuntimeError(f"Chunks file missing: {CHUNKS_PATH}")
    
    try:
        with open(CHUNKS_PATH, "rb") as f:
            state.chunks = pickle.load(f)
        chunks_load_time = time.time() - chunks_start
        log.info(f"✅ {len(state.chunks)} chunks loaded in {chunks_load_time:.4f}s")
    except Exception as e:
        log.critical(f"💥 Failed to load text chunks: {e}")
        raise RuntimeError(f"Chunks Load Failure: {e}")

    # 4. Integrity Check
    if len(state.chunks) != state.faiss_index.ntotal:
        err_msg = f"FAISS integrity check failed: Chunks count ({len(state.chunks)}) does not match FAISS index total ({state.faiss_index.ntotal})!"
        log.critical(f"💥 {err_msg}")
        raise RuntimeError(err_msg)
    log.info("✅ FAISS integrity validation check passed")

    # 5. Model load
    model_start = time.time()
    log.info("⏳ Loading embedding model…")
    try:
        state.embedder = SentenceTransformer(EMBEDDING_MODEL)
        model_load_time = time.time() - model_start
        log.info(f"✅ Embedding model ready in {model_load_time:.4f}s")
    except Exception as e:
        log.critical(f"💥 Failed to load embedding model: {e}")
        raise RuntimeError(f"Model Load Failure: {e}")

    # 6. Groq client connection check
    if not GROQ_API_KEY:
        log.warning("⚠️ GROQ_API_KEY environment variable is not set! LLM queries will fail.")
    else:
        log.info("⏳ Connecting to Groq…")
        try:
            state.groq_client = Groq(api_key=GROQ_API_KEY)
            log.info("✅ Groq client ready")
        except Exception as e:
            log.error(f"❌ Groq client setup failed: {e}")

    total_startup = time.time() - startup_start
    log.info(f"🚀 V-Shield System fully initialized in {total_startup:.2f}s")

    yield

    # Clean up pg pool on shutdown
    if state.use_postgres and state.pg_pool is not None:
        log.info("⏳ Closing PostgreSQL Connection Pool...")
        try:
            state.pg_pool.closeall()
            log.info("✅ PostgreSQL Connection Pool closed")
        except Exception as e:
            log.error(f"Failed to close connection pool: {e}")

    log.info("👋 Shutting down…")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseConnectionError(Exception):
    """Custom exception raised when database connectivity is down."""
    pass

app = FastAPI(
    title="V-Shield: Vernacular Anti-Doping Assistant",
    lifespan=lifespan,
)

# Hardened CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://localhost:5173",
    "http://localhost:3000"
]
env_origins = os.getenv("ALLOWED_ORIGINS")
if env_origins:
    ALLOWED_ORIGINS.extend(env_origins.split(","))
else:
    # If in dev (no cloud DB URL configured), default to "*"
    if not os.getenv("DATABASE_URL"):
        ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

from fastapi import Request

@app.exception_handler(DatabaseConnectionError)
async def database_connection_error_handler(request: Request, exc: DatabaseConnectionError):
    log.error(f"🚨 Graceful Degradation: Database connection failure: {exc}")
    return JSONResponse(
        status_code=503,
        content={"error": "database temporarily unavailable"}
    )


# ══════════════════════════════════════════════════════════════════════════════
# SQLITE
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def db_session():
    conn = None
    try:
        if state.use_postgres:
            if state.pg_pool is None:
                raise RuntimeError("PostgreSQL Connection Pool is not initialized")
            conn = state.pg_pool.getconn()
            yield conn
            conn.commit()
        else:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
    except Exception as e:
        if conn and state.use_postgres:
            try:
                conn.rollback()
            except Exception as rollback_err:
                log.error(f"PostgreSQL rollback failed: {rollback_err}")
        log.error(f"❌ Database Session Error: {e}")
        raise DatabaseConnectionError(str(e))
    finally:
        if conn:
            if state.use_postgres:
                try:
                    state.pg_pool.putconn(conn)
                except Exception as put_err:
                    log.error(f"PostgreSQL putconn failed: {put_err}")
            else:
                try:
                    conn.close()
                except Exception as close_err:
                    log.error(f"SQLite close failed: {close_err}")

def init_db():
    with db_session() as conn:
        if state.use_postgres:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id         SERIAL PRIMARY KEY,
                    title      TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id         SERIAL PRIMARY KEY,
                    chat_id    INTEGER NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    language   TEXT NOT NULL DEFAULT 'hindi',
                    risk_level TEXT NOT NULL DEFAULT 'unknown',
                    timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
                )
            """)
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    title      TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id    INTEGER NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    language   TEXT NOT NULL DEFAULT 'hindi',
                    risk_level TEXT NOT NULL DEFAULT 'unknown',
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
                )
            """)

def create_new_chat(title: str = "New Chat") -> int:
    with db_session() as conn:
        if state.use_postgres:
            cur = conn.cursor()
            cur.execute("INSERT INTO chats (title) VALUES (%s) RETURNING id", (title,))
            lastrowid = cur.fetchone()[0]
            return lastrowid
        else:
            cursor = conn.execute("INSERT INTO chats (title) VALUES (?)", (title,))
            return cursor.lastrowid

def update_chat_title(chat_id: int, title: str):
    with db_session() as conn:
        if state.use_postgres:
            cur = conn.cursor()
            cur.execute("UPDATE chats SET title = %s WHERE id = %s", (title, chat_id))
        else:
            conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))

def save_message(chat_id: int, role: str, content: str, language: str = "hindi", risk_level: str = "unknown"):
    with db_session() as conn:
        if state.use_postgres:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO messages (chat_id, role, content, language, risk_level) VALUES (%s, %s, %s, %s, %s)",
                (chat_id, role, content, language, risk_level)
            )
        else:
            conn.execute(
                "INSERT INTO messages (chat_id, role, content, language, risk_level) VALUES (?, ?, ?, ?, ?)",
                (chat_id, role, content, language, risk_level)
            )

def fetch_chat_context(chat_id: int, limit: int = MEMORY_WINDOW) -> list[dict]:
    with db_session() as conn:
        if state.use_postgres:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute(
                "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY id DESC LIMIT %s",
                (chat_id, limit),
            )
            rows = cur.fetchall()
        else:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def fetch_all_chats() -> list[dict]:
    with db_session() as conn:
        if state.use_postgres:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute("SELECT id, title, created_at FROM chats ORDER BY created_at DESC")
            rows = cur.fetchall()
        else:
            rows = conn.execute("SELECT id, title, created_at FROM chats ORDER BY created_at DESC").fetchall()
    res = []
    for r in rows:
        created_at_val = r["created_at"]
        if not isinstance(created_at_val, str):
            created_at_val = created_at_val.strftime("%Y-%m-%d %H:%M:%S")
        res.append({"id": r["id"], "title": r["title"], "created_at": created_at_val})
    return res

def fetch_chat_detail(chat_id: int) -> Optional[dict]:
    with db_session() as conn:
        if state.use_postgres:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute("SELECT id, title, created_at FROM chats WHERE id = %s", (chat_id,))
            chat = cur.fetchone()
            if not chat: return None
            cur.execute(
                "SELECT id, role, content, risk_level, language, timestamp FROM messages WHERE chat_id = %s ORDER BY id ASC",
                (chat_id,)
            )
            messages = cur.fetchall()
        else:
            chat = conn.execute("SELECT id, title, created_at FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not chat: return None
            messages = conn.execute(
                "SELECT id, role, content, risk_level, language, timestamp FROM messages WHERE chat_id = ? ORDER BY id ASC",
                (chat_id,)
            ).fetchall()
            
    formatted_messages = []
    for m in messages:
        timestamp_val = m["timestamp"]
        if not isinstance(timestamp_val, str):
            timestamp_val = timestamp_val.strftime("%Y-%m-%d %H:%M:%S")
        formatted_messages.append({
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "risk_level": m["risk_level"],
            "language": m["language"],
            "timestamp": timestamp_val
        })
        
    created_at_val = chat["created_at"]
    if not isinstance(created_at_val, str):
        created_at_val = created_at_val.strftime("%Y-%m-%d %H:%M:%S")
        
    return {
        "chat_id": chat["id"],
        "title": chat["title"],
        "messages": formatted_messages
    }

def delete_chat_session(chat_id: int) -> bool:
    try:
        with db_session() as conn:
            if state.use_postgres:
                cur = conn.cursor()
                cur.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
                rowcount = cur.rowcount
                return rowcount > 0
            else:
                cursor = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                return cursor.rowcount > 0
    except Exception as e:
        log.error(f"❌ Deletion error: {e}")
        return False

def get_db_row_count() -> int:
    try:
        with db_session() as conn:
            if state.use_postgres:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM messages")
                return cur.fetchone()[0]
            else:
                return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    except: return 0


# ══════════════════════════════════════════════════════════════════════════════
# LOGIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_HINGLISH_KEYWORDS = {"kya", "hai", "nahi", "karo", "bhai", "aur", "le", "se", "mein", "ko", "pe", "ke", "ki", "ka", "ho", "tum", "main", "yeh", "woh", "kuch", "safe", "lena", "poocha", "doping", "supplement"}

def strip_devanagari(text: str) -> str:
    """Removes pure Devanagari characters from the string."""
    return re.sub(r'[\u0900-\u097F]+', '', text).strip()

def detect_language(text: str) -> str:
    if any('\u0900' <= c <= '\u097F' for c in text): return "hindi"
    tokens = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    if len(tokens & _HINGLISH_KEYWORDS) >= 2: return "hindi"
    return "english"

def parse_risk_level(response_text: str) -> str:
    start_text = response_text[:20].strip()
    if re.search(r'BANNED|❌', start_text, re.IGNORECASE): return "banned"
    if re.search(r'CAUTION|⚠️', start_text, re.IGNORECASE): return "caution"
    if re.search(r'SAFE|✅', start_text, re.IGNORECASE): return "safe"
    if re.search(r'UNKNOWN|❓', start_text, re.IGNORECASE): return "unknown"
    return "caution"


# ══════════════════════════════════════════════════════════════════════════════
# STT / TTS
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """
    Transcribes speech audio using Groq Whisper-large-v3.

    IMPORTANT: Do NOT set language='hi' here.
    Forcing a language causes Whisper to hallucinate or incorrectly transcribe
    when the user speaks English. Leaving it unset enables true multilingual
    auto-detection — it correctly handles Hindi, Hinglish, and English in the
    same session without any configuration changes.
    """
    if not state.groq_client: return "Server busy."
    try:
        transcription = state.groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=WHISPER_MODEL,
            response_format="text",
            # language is intentionally omitted → Whisper auto-detects
            # Supports Hindi, Hinglish (romanised) and English seamlessly
        )
        transcribed = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
        log.info(f"🎙️ Whisper transcribed: '{transcribed}'")
        return transcribed
    except Exception as e:
        log.error(f"❌ Whisper error: {e}")
        return "Audio samajh nahi aaya, please type your question."


_EMOJI_RE = re.compile(r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FE0F✅❌⚠️❓🏋️🎙️]+", flags=re.UNICODE)

@lru_cache(maxsize=128)
def generate_tts(text: str, language: str = "hi") -> Optional[bytes]:
    try:
        from gtts import gTTS
        clean = _EMOJI_RE.sub("", text).strip()
        if not clean: return None
        tts = gTTS(text=clean, lang=language, slow=False)
        fp  = BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        log.error(f"❌ TTS error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# RAG & LLM
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context_with_timing(query: str, top_k: int = TOP_K_RETRIEVAL) -> tuple[str, float, float]:
    """Returns context string, embedding time, and search time."""
    if state.faiss_index is None or state.embedder is None:
        return "", 0.0, 0.0
    
    # 1. Embedding Time
    embed_start = time.time()
    query_vec = state.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    embed_time = time.time() - embed_start
    
    # 2. FAISS Search Time
    search_start = time.time()
    _, indices = state.faiss_index.search(query_vec, top_k)
    retrieved = [state.chunks[idx] for idx in indices[0] if 0 <= idx < len(state.chunks)]
    search_time = time.time() - search_start
    
    return "\n\n".join(retrieved), embed_time, search_time

def retrieve_context(query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
    context, _, _ = retrieve_context_with_timing(query, top_k)
    return context

SYSTEM_PROMPT = """
You are a senior anti-doping mentor for road athletes. Your goal is to provide safety guidance on supplements and medications.

STRICT INSTRUCTION: YOUR KEYBOARD HAS NO HINDI LETTERS.
1. RESPONSE LANGUAGE & SCRIPT:
   - If user asks in English -> Response: English.
   - If user asks in Hindi/Hinglish -> Response: **Hinglish (Hindi words in English/Latin script)**.
   - **MANDATORY**: NEVER use Devanagari script (e.g. नमस्ते). ONLY use Latin letters (e.g. Namaste).
   - If you accidentally output a Hindi character, the system will break. Use ROMAN script only.

2. CONTEXT & MEMORY:
   - You will be provided with "KNOWLEDGE CONTEXT" (scientific facts) and "CONVERSATION HISTORY".
   - Use history to understand follow-up questions (e.g., "it", "them").
   - If the user's current query references something previously discussed, use that context.

3. DOMAIN: Answer ONLY about anti-doping, supplements, and prohibited substances.
   - If asked about something else, politely decline in the same language/script style.

4. STRUCTURE:
   - Every answer MUST start with a risk tag: SAFE ✅, CAUTION ⚠️, BANNED ❌, or UNKNOWN ❓.
   - Keep responses concise: exactly 3 sentences.
   - Sentence 1: Risk tag + direct answer.
   - Sentence 2: Scientific/factual reason based on the provided context.
   - Sentence 3: Practical advice.

5. NO HALLUCINATION: If the context doesn't mention a substance and it's not in your memory, say UNKNOWN ❓.
""".strip()

def call_llm(user_query: str, knowledge_context: str, past_messages: list[dict], lang: str = "hindi") -> str:
    if not state.groq_client: return "CAUTION ⚠️ – Server busy."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in past_messages: messages.append({"role": msg["role"], "content": msg["content"]})
    if knowledge_context:
        messages.append({"role": "system", "content": f"KNOWLEDGE CONTEXT: {knowledge_context}"})
    messages.append({"role": "user", "content": user_query})
    try:
        completion = state.groq_client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.4, max_tokens=400)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"❌ LLM error: {e}")
        return "CAUTION ⚠️ – Error processing request."


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE & ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def run_text_pipeline(query: str, chat_id: int) -> dict:
    pipeline_start = time.time()
    
    lang              = detect_language(query)
    
    # 1. DB context retrieval timing
    db_fetch_start = time.time()
    past_messages     = fetch_chat_context(chat_id)
    db_fetch_time  = time.time() - db_fetch_start
    
    # 2. FAISS context retrieval + embedding timing
    knowledge_context, embed_time, search_time = retrieve_context_with_timing(query)
    
    # 3. LLM latency profiling
    llm_start         = time.time()
    ai_response       = call_llm(query, knowledge_context, past_messages, lang=lang)
    llm_time          = time.time() - llm_start
    
    # Filter Devanagari leakage
    if lang == "hindi":
        ai_response = strip_devanagari(ai_response)
        if not ai_response: # Fallback if everything was Devanagari
            ai_response = "CAUTION ⚠️ – Mujhse Hindi script mein baat na karein, English letters use karein. Please check with NADA certified products."

    risk_level        = parse_risk_level(ai_response)
    
    # 4. DB save latency profiling
    db_save_start = time.time()
    save_message(chat_id, "user", query, language=lang)
    save_message(chat_id, "assistant", ai_response, language=lang, risk_level=risk_level)
    if len(past_messages) == 0:
        new_title = query[:30] + "..." if len(query) > 30 else query
        update_chat_title(chat_id, new_title)
    db_save_time = time.time() - db_save_start
        
    total_time = time.time() - pipeline_start
    log.info(
        f"📋 --- RAG TELEMETRY TRACE ---\n"
        f"  Question: '{query}'\n"
        f"  ├─ Embedding Generation Time : {embed_time:.4f}s\n"
        f"  ├─ FAISS Search Time         : {search_time:.4f}s\n"
        f"  ├─ LLM Reasoning Time        : {llm_time:.4f}s\n"
        f"  ├─ Database Operations Time  : {db_fetch_time + db_save_time:.4f}s (fetch: {db_fetch_time:.4f}s, save: {db_save_time:.4f}s)\n"
        f"  └─ Total Request Latency     : {total_time:.4f}s\n"
        f"------------------------------"
    )
    return {"chat_id": chat_id, "user_query": query, "ai_response": ai_response, "language": lang, "risk_level": risk_level}

@app.post("/verify")
async def verify(chat_id: int = Query(...), audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    # Guard: reject suspiciously small blobs (< 1 KB usually means empty / mic error)
    if len(audio_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Audio recording too short or empty. Please try again.")

    # Pass the real filename so Whisper can identify the codec (webm / mp4 / ogg)
    filename = audio.filename or "recording.webm"
    user_query = transcribe_audio(audio_bytes, filename=filename)

    result = run_text_pipeline(user_query, chat_id)
    tts_lang = "hi" if result["language"] == "hindi" else "en"
    tts_bytes = generate_tts(result["ai_response"], language=tts_lang)
    return JSONResponse({**result, "audio_base64": tts_bytes.hex() if tts_bytes else None})

@app.post("/verify-text")
async def verify_text(payload: TextQueryRequest, chat_id: int = Query(...)):
    result = run_text_pipeline(payload.query, chat_id)
    return JSONResponse(result)

@app.post("/chats")
async def create_chat():
    return {"chat_id": create_new_chat(), "title": "New Chat"}

@app.get("/chats", response_model=list[ChatSession])
async def list_chats():
    return fetch_all_chats()

@app.get("/chats/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: int):
    detail = fetch_chat_detail(chat_id)
    if not detail: raise HTTPException(status_code=404, detail="Chat not found")
    return detail

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int):
    if delete_chat_session(chat_id): return {"deleted": True}
    raise HTTPException(status_code=404, detail="Chat not found")

@app.get("/api/tts")
async def get_api_tts(text: str = Query(...), lang: str = Query("hi")):
    """
    Returns audio bytes for the given text and language.
    Language defaults to 'hi' to ensure natural Hinglish pronunciation.
    """
    tts_lang = "hi" if lang == "hindi" else "en"
    audio_bytes = generate_tts(text, language=tts_lang)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="TTS generation failed")
    return Response(content=audio_bytes, media_type="audio/mpeg")

@app.get("/health")
async def health():
    return {"status": "ok" if (state.faiss_index and state.groq_client) else "degraded", "uptime_seconds": int(time.time() - _START_TIME), "db_row_count": get_db_row_count()}

@app.get("/check-model")
async def check_model():
    if state.embedder is not None:
        return {"status": "healthy", "model_name": EMBEDDING_MODEL}
    return JSONResponse(status_code=503, content={"status": "unhealthy", "error": "Embedding model not initialized"})

@app.get("/check-faiss")
async def check_faiss():
    if state.faiss_index is not None and state.chunks is not None:
        return {
            "status": "healthy",
            "vectors": state.faiss_index.ntotal,
            "chunks_loaded": len(state.chunks)
        }
    return JSONResponse(status_code=503, content={"status": "unhealthy", "error": "FAISS index or chunks not loaded"})

@app.get("/check-db")
async def check_db():
    try:
        with db_session() as conn:
            if state.use_postgres:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
                db_type = "postgresql"
            else:
                conn.execute("SELECT 1").fetchone()
                db_type = "sqlite"
        return {"status": "healthy", "database_type": db_type}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

# Serve frontend static files
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
else:
    @app.get("/")
    async def serve_root_fallback():
        if os.path.exists("index.html"):
            return FileResponse("index.html")
        return {"status": "ok", "message": "V-Shield API is running. Build frontend to view the UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)