#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          YT GUARDIAN v2.0 — TEK DOSYA MODERASYON & ANALİZ SİSTEMİ          ║
║  Kanal: @ShmirchikArt | NLP·BART·RL·Graf·Bayes·HMM·Oyun Kuramı·Stilometri  ║
║  Lokal AI (Ollama phi4:14b) | ROCm GPU | SQLite + ChromaDB | Selenium FF    ║
╚══════════════════════════════════════════════════════════════════════════════╝

KURULUM (Ubuntu):
  pip install flask flask-socketio selenium yt-dlp requests numpy scipy
      scikit-learn torch transformers sentence-transformers spacy langdetect
      bertopic umap-learn hdbscan networkx python-louvain hmmlearn chromadb
      ollama pillow flask-cors eventlet
  python -m spacy download xx_ent_wiki_sm
  pip install fasttext          # veya: pip install fasttext-wheel
  ollama pull phi4:14b

KONFİGÜRASYON:
  yt_guardian_config.json dosyası oluşturun (örnek aşağıda):
  {
    "yt_email": "",
    "yt_password": "",
    "channel_url": "https://www.youtube.com/@ShmirchikArt/streams",
    "channel_handle": "@ShmirchikArt",
    "db_path": "yt_guardian.db",
    "chroma_path": "./chromadb_data",
    "data_dir": "./yt_data",
    "ollama_model": "phi4:14b",
    "ollama_host": "http://localhost:11434",
    "flask_port": 5000,
    "flask_secret": "yt_guardian_secret_2024",
    "date_from": "2023-01-01",
    "date_to": "2026-12-31",
    "similarity_threshold": 0.65,
    "bot_threshold": 0.70,
    "hate_threshold": 0.65,
    "stalker_threshold": 0.55,
    "device": "auto",
    "allow_destructive_actions": false,
    "require_env_credentials": true
  }

GÜVENLİK NOTU:
  YouTube kimlik bilgilerini config dosyasına düz metin olarak yazmayın.
  Tercih edilen yöntem:
    export YT_EMAIL="mail@domain.com"
    export YT_PASSWORD="..."
"""

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import os, sys, re, json, time, math, hashlib, threading, logging, unicodedata
import sqlite3, subprocess, collections, random, copy, traceback, argparse
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter, deque
from typing import Optional, List, Dict, Tuple, Any
from functools import lru_cache

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import entropy as scipy_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS

import networkx as nx
try:
    import community as community_louvain        # python-louvain
except ImportError:
    community_louvain = None

try:
    import hmmlearn.hmm as hmmlearn_hmm
    HMMLEARN_OK = True
except ImportError:
    HMMLEARN_OK = False

try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except ImportError:
    SBERT_OK = False

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import spacy
    SPACY_OK = True
except ImportError:
    SPACY_OK = False

try:
    from langdetect import detect as langdetect_detect, DetectorFactory
    DetectorFactory.seed = 42
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    import fasttext
    FASTTEXT_OK = True
except ImportError:
    FASTTEXT_OK = False

try:
    from bertopic import BERTopic
    BERTOPIC_OK = True
except ImportError:
    BERTOPIC_OK = False

try:
    import ollama as ollama_sdk
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

try:
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options as FFOptions
    from selenium.webdriver.firefox.service import Service as FFService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import (NoSuchElementException, TimeoutException,
                                             StaleElementReferenceException, WebDriverException)
    SELENIUM_OK = True
except ImportError:
    SELENIUM_OK = False

try:
    from flask import Flask, render_template_string, request, jsonify, g
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
    FLASK_OK = True
except ImportError:
    FLASK_OK = False

import requests as http_requests
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — KONFİGÜRASYON & SABITLER
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("yt_guardian.log")]
)
log = logging.getLogger("YTGuardian")

DEFAULT_CONFIG = {
    "yt_email":           os.environ.get("YT_EMAIL", ""),
    "yt_password":        os.environ.get("YT_PASSWORD", ""),
    "channel_url":        "https://www.youtube.com/@ShmirchikArt/streams",
    "channel_handle":     "@ShmirchikArt",
    "db_path":            "yt_guardian.db",
    "chroma_path":        "./chromadb_data",
    "data_dir":           "./yt_data",
    "ollama_model":       "phi4:14b",
    "ollama_host":        "http://localhost:11434",
    "flask_port":         5000,
    "flask_secret":       os.environ.get("FLASK_SECRET", ""),
    "date_from":          "2023-01-01",
    "date_to":            "2026-12-31",
    "similarity_threshold": 0.65,
    "bot_threshold":      0.70,
    "hate_threshold":     0.65,
    "stalker_threshold":  0.55,
    "device":             "auto",
    "fasttext_model":     "lid.176.bin",
    "retrain_threshold":  500,
    "allow_destructive_actions": False,   # Silme/ban gibi işlemler için güvenlik kapısı
    "require_env_credentials":  True,     # Kimlik bilgisi env'den gelmeli
}

def load_config(config_file: str = "yt_guardian_config.json") -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if Path(config_file).exists():
        with open(config_file, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    # Env override
    env_map = {
        "yt_email": "YT_EMAIL",
        "yt_password": "YT_PASSWORD",
        "channel_url": "YT_CHANNEL_URL",
        "channel_handle": "YT_CHANNEL_HANDLE",
        "db_path": "YT_DB_PATH",
        "flask_secret": "FLASK_SECRET",
        "ollama_model": "OLLAMA_MODEL",
        "ollama_host": "OLLAMA_HOST",
    }
    for key, env_key in env_map.items():
        env_val = os.environ.get(env_key, "")
        if env_val:
            cfg[key] = env_val
    # Güvenlik: require_env_credentials açıksa dosyadaki düz metin credentialları yok say
    if cfg.get("require_env_credentials", True):
        if not os.environ.get("YT_EMAIL"):
            cfg["yt_email"] = ""
        if not os.environ.get("YT_PASSWORD"):
            cfg["yt_password"] = ""
    return cfg

CONFIG = load_config()

COLOR_MAP = {
    "GREEN":   "#2ECC71",
    "YELLOW":  "#F1C40F",
    "ORANGE":  "#E67E22",
    "RED":     "#E74C3C",
    "BLUE":    "#3498DB",
    "PURPLE":  "#9B59B6",
    "CRIMSON": "#8B0000",
}

THREAT_LABELS_ZEROSHOT = [
    "antisemitic content",
    "hate speech against Jewish people",
    "islamophobic content",
    "white supremacist content",
    "groyper movement content",
    "harassment and stalking behavior",
    "identity impersonation",
    "coordinated bot attack",
    "neutral friendly message",
    "spam content",
]

BOT_ZEROSHOT_LABELS = ["human-like conversation", "spam or bot-like message"]

ACTION_NAMES = {0: "HUMAN", 1: "BOT", 2: "HATER", 3: "STALKER", 4: "IMPERSONATOR", 5: "COORDINATED"}

def destructive_actions_enabled() -> bool:
    """Silme/ban gibi yıkıcı işlemler için global güvenlik anahtarı."""
    return bool(CONFIG.get("allow_destructive_actions", False))

def build_deletion_candidates(limit: int = 300) -> List[dict]:
    """
    Mevcut analizlerden hareketle moderatör inceleme kuyruğu oluşturur.
    Not: Bu fonksiyon otomatik silme yapmaz, sadece öneri üretir.
    """
    sql = """
    SELECT
      m.id, m.video_id, m.author, m.message, m.timestamp,
      COALESCE(u.threat_score,0) AS threat_score,
      COALESCE(u.hate_score,0)   AS hate_score,
      COALESCE(u.bot_prob,0)     AS bot_prob,
      COALESCE(u.stalker_score,0) AS stalker_score,
      COALESCE(u.threat_level,'GREEN') AS threat_level
    FROM messages m
    LEFT JOIN users u ON u.author = m.author
    WHERE m.deleted = 0
    ORDER BY (COALESCE(u.threat_score,0)*0.45 +
              COALESCE(u.hate_score,0)*0.30 +
              COALESCE(u.bot_prob,0)*0.15 +
              COALESCE(u.stalker_score,0)*0.10) DESC, m.timestamp DESC
    LIMIT ?
    """
    rows = db_exec(sql, (int(limit),), fetch="all") or []
    out = []
    for r in rows:
        d = dict(r)
        score = (float(d.get("threat_score", 0))*0.45 +
                 float(d.get("hate_score", 0))*0.30 +
                 float(d.get("bot_prob", 0))*0.15 +
                 float(d.get("stalker_score", 0))*0.10)
        decision = "REVIEW"
        if score >= 0.80:
            decision = "HIGH_RISK_REVIEW"
        elif score >= 0.65:
            decision = "MEDIUM_RISK_REVIEW"
        d["delete_candidate_score"] = round(score, 4)
        d["decision"] = decision
        out.append(d)
    return out

# Oyun kuramı ödül matrisi  [moderatör_idx][aktör_idx] = (mod_payoff, aktör_payoff)
PAYOFF_MATRIX = np.array([
    [(-1,-5), (3,-3), (5,-4), (4,-3)],   # BAN
    [( 1, 0), (-1,-1),(1,-2), (0,-1)],   # WARN
    [( 2, 2), (-3, 3),(-4,4), (-3,3)],  # IGNORE
    [( 1, 0), (2,-1), (3,-2), (2,-1)],  # MONITOR
], dtype=object)

MOD_ACTIONS   = ["BAN", "WARN", "IGNORE", "MONITOR"]
ACTOR_ACTIONS = ["BEHAVE", "TROLL", "IMPERSONATE", "FLOOD"]

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — VERİTABANI KATMANI (SQLite + ChromaDB)
# ═══════════════════════════════════════════════════════════════════════════════
DB_PATH = CONFIG["db_path"]
_db_lock = threading.Lock()

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id            TEXT PRIMARY KEY,
            video_id      TEXT NOT NULL,
            title         TEXT,
            video_date    TEXT,
            author        TEXT NOT NULL,
            author_cid    TEXT,
            message       TEXT NOT NULL,
            timestamp     INTEGER,
            lang          TEXT,
            script_type   TEXT,
            source_type   TEXT,
            is_live       INTEGER DEFAULT 0,
            deleted       INTEGER DEFAULT 0,
            created_at    INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(author, message, content='messages', content_rowid='rowid');
        CREATE TABLE IF NOT EXISTS user_profiles (
            author            TEXT PRIMARY KEY,
            author_cid        TEXT,
            msg_count         INTEGER DEFAULT 0,
            human_score       REAL DEFAULT 0.5,
            bot_prob          REAL DEFAULT 0.0,
            hate_score        REAL DEFAULT 0.0,
            stalker_score     REAL DEFAULT 0.0,
            impersonator_prob REAL DEFAULT 0.0,
            identity_vector   TEXT DEFAULT '{}',
            cluster_id        INTEGER DEFAULT -1,
            threat_level      TEXT DEFAULT 'GREEN',
            threat_score      REAL DEFAULT 0.0,
            tfidf_vector      TEXT DEFAULT '{}',
            ngram_fingerprint TEXT DEFAULT '{}',
            typo_fingerprint  TEXT DEFAULT '{}',
            pos_profile       TEXT DEFAULT '{}',
            account_created   TEXT,
            subscriber_count  INTEGER DEFAULT 0,
            is_new_account    INTEGER DEFAULT 0,
            hmm_state         TEXT DEFAULT 'NORMAL',
            q_state           TEXT DEFAULT '00000',
            game_strategy     TEXT DEFAULT 'BEHAVE',
            ollama_summary    TEXT,
            first_seen        INTEGER,
            last_seen         INTEGER,
            updated_at        INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS identity_links (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_a      TEXT NOT NULL,
            user_b      TEXT NOT NULL,
            sim_score   REAL,
            method      TEXT,
            confidence  REAL,
            created_at  INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS graph_clusters (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id    INTEGER,
            members       TEXT,
            algorithm     TEXT,
            created_at    INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS game_history (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            author           TEXT,
            moderator_action TEXT,
            actor_action     TEXT,
            payoff_m         REAL,
            payoff_a         REAL,
            timestamp        INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS training_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name  TEXT,
            version     INTEGER,
            accuracy    REAL,
            f1_score    REAL,
            trained_at  INTEGER DEFAULT (strftime('%s','now')),
            notes       TEXT
        );
        CREATE TABLE IF NOT EXISTS dataset (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_id      TEXT,
            author      TEXT,
            message     TEXT,
            label       TEXT,
            confirmed   INTEGER DEFAULT 0,
            created_at  INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS scraped_videos (
            video_id    TEXT PRIMARY KEY,
            title       TEXT,
            video_date  TEXT,
            source_type TEXT,
            scraped_at  INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE INDEX IF NOT EXISTS idx_msg_author   ON messages(author);
        CREATE INDEX IF NOT EXISTS idx_msg_video    ON messages(video_id);
        CREATE INDEX IF NOT EXISTS idx_msg_ts       ON messages(timestamp);
        CREATE INDEX IF NOT EXISTS idx_up_threat    ON user_profiles(threat_level);
        """)
        _run_schema_migrations(conn)
    log.info("✅ SQLite veritabanı hazır: %s", DB_PATH)

def _table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows} if rows else set()

def _ensure_column(conn: sqlite3.Connection, table_name: str, col_name: str, col_sql: str):
    cols = _table_columns(conn, table_name)
    if col_name not in cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_sql}")

def _run_schema_migrations(conn: sqlite3.Connection):
    """
    Eski DB dosyaları için geriye dönük uyumluluk migrasyonları.
    no such column hatalarını önlemek için index/table kolonlarını doğrular.
    """
    # identity_links eski şemadan geliyorsa eksik kolonları tamamla
    if _table_columns(conn, "identity_links"):
        _ensure_column(conn, "identity_links", "user_a", "TEXT")
        _ensure_column(conn, "identity_links", "user_b", "TEXT")
        _ensure_column(conn, "identity_links", "sim_score", "REAL")
        _ensure_column(conn, "identity_links", "method", "TEXT")
        _ensure_column(conn, "identity_links", "confidence", "REAL")
        _ensure_column(conn, "identity_links", "created_at", "INTEGER DEFAULT (strftime('%s','now'))")
    else:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS identity_links (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_a      TEXT NOT NULL,
            user_b      TEXT NOT NULL,
            sim_score   REAL,
            method      TEXT,
            confidence  REAL,
            created_at  INTEGER DEFAULT (strftime('%s','now'))
        );
        """)

    # index sadece ilgili kolonlar gerçekten varsa oluştur
    id_cols = _table_columns(conn, "identity_links")
    if {"user_a", "user_b"}.issubset(id_cols):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_link_ab ON identity_links(user_a, user_b)")

def db_exec(sql: str, params: tuple = (), fetch: str = None):
    with _db_lock:
        with get_db() as conn:
            cur = conn.execute(sql, params)
            if fetch == "one":  return cur.fetchone()
            if fetch == "all":  return cur.fetchall()
            return cur.lastrowid

def upsert_message(msg: dict):
    sql = """INSERT OR IGNORE INTO messages
        (id,video_id,title,video_date,author,author_cid,message,timestamp,lang,
         script_type,source_type,is_live)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
    db_exec(sql, (
        msg["msg_id"], msg.get("video_id",""), msg.get("title",""),
        msg.get("video_date",""), msg["author"], msg.get("author_channel_id",""),
        msg["message"], msg.get("timestamp_utc",0), msg.get("lang_detected",""),
        msg.get("script",""), msg.get("source_type","comment"),
        int(msg.get("is_live", False))
    ))
    db_exec("INSERT OR IGNORE INTO messages_fts(rowid,author,message) "
            "SELECT rowid,author,message FROM messages WHERE id=?", (msg["msg_id"],))

def get_user_messages(author: str) -> List[Dict]:
    rows = db_exec("SELECT * FROM messages WHERE author=? AND deleted=0 ORDER BY timestamp",
                   (author,), fetch="all")
    return [dict(r) for r in rows] if rows else []

def get_all_users() -> List[Dict]:
    rows = db_exec("SELECT * FROM user_profiles ORDER BY threat_score DESC", fetch="all")
    return [dict(r) for r in rows] if rows else []

def upsert_user_profile(author: str, updates: dict):
    existing = db_exec("SELECT author FROM user_profiles WHERE author=?", (author,), fetch="one")
    if not existing:
        db_exec("INSERT OR IGNORE INTO user_profiles(author) VALUES(?)", (author,))
    set_clause = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [author]
    db_exec(f"UPDATE user_profiles SET {set_clause}, updated_at=strftime('%s','now') WHERE author=?", tuple(vals))

# ChromaDB
_chroma_client = None
_chroma_msg_col = None
_chroma_user_col = None

def init_chroma():
    global _chroma_client, _chroma_msg_col, _chroma_user_col
    if not CHROMA_OK:
        log.warning("ChromaDB yüklü değil, vektör araması devre dışı")
        return
    try:
        Path(CONFIG["chroma_path"]).mkdir(parents=True, exist_ok=True)
        _chroma_client   = chromadb.PersistentClient(path=CONFIG["chroma_path"])
        _chroma_msg_col  = _chroma_client.get_or_create_collection("message_embeddings",  metadata={"hnsw:space":"cosine"})
        _chroma_user_col = _chroma_client.get_or_create_collection("user_profiles",        metadata={"hnsw:space":"cosine"})
        log.info("✅ ChromaDB hazır: %s", CONFIG["chroma_path"])
    except Exception as e:
        log.warning("ChromaDB başlatılamadı: %s", e)

def chroma_upsert_msg(msg_id: str, embedding: List[float], metadata: dict):
    if _chroma_msg_col is None: return
    try:
        _chroma_msg_col.upsert(ids=[msg_id], embeddings=[embedding],
                               metadatas=[{k: str(v)[:500] for k,v in metadata.items()}])
    except Exception as e:
        log.debug("Chroma upsert hata: %s", e)

def chroma_search_similar(embedding: List[float], n: int = 10) -> List[dict]:
    if _chroma_msg_col is None: return []
    try:
        r = _chroma_msg_col.query(query_embeddings=[embedding], n_results=n)
        return [{"id": r["ids"][0][i], "distance": r["distances"][0][i],
                 "meta": r["metadatas"][0][i]} for i in range(len(r["ids"][0]))]
    except:
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — MODEL YÖNETİMİ (Lazy Loading)
# ═══════════════════════════════════════════════════════════════════════════════
_models = {}
_model_lock = threading.Lock()

def _get_device():
    cfg_dev = CONFIG.get("device","auto")
    if cfg_dev != "auto": return cfg_dev
    if TORCH_OK and torch.cuda.is_available():    return "cuda"
    if TORCH_OK:
        try:
            if torch.version.hip:                 return "cuda"  # ROCm
        except: pass
    return "cpu"

DEVICE = _get_device()

def get_sbert():
    with _model_lock:
        if "sbert" not in _models:
            if not SBERT_OK:
                log.warning("sentence-transformers yüklü değil"); return None
            try:
                _models["sbert"] = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
                log.info("✅ SBERT yüklendi (%s)", DEVICE)
            except Exception as e:
                log.error("SBERT yüklenemedi: %s", e); return None
        return _models["sbert"]

def get_bart_zeroshot():
    with _model_lock:
        if "bart" not in _models:
            if not TRANSFORMERS_OK:
                log.warning("transformers yüklü değil"); return None
            try:
                _models["bart"] = hf_pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if DEVICE in ("cuda","mps") else -1
                )
                log.info("✅ BART zero-shot yüklendi (%s)", DEVICE)
            except Exception as e:
                log.error("BART yüklenemedi: %s", e); return None
        return _models["bart"]

def get_spacy():
    with _model_lock:
        if "spacy" not in _models:
            if not SPACY_OK:
                log.warning("spacy yüklü değil"); return None
            try:
                import spacy as _spacy
                _models["spacy"] = _spacy.load("xx_ent_wiki_sm")
                log.info("✅ spaCy xx_ent_wiki_sm yüklendi")
            except Exception as e:
                log.warning("spaCy modeli yüklenemedi: %s", e); return None
        return _models["spacy"]

def get_fasttext():
    with _model_lock:
        if "fasttext" not in _models:
            if not FASTTEXT_OK:
                return None
            model_path = CONFIG.get("fasttext_model","lid.176.bin")
            if not Path(model_path).exists():
                log.warning("fasttext modeli bulunamadı: %s", model_path)
                return None
            try:
                _models["fasttext"] = fasttext.load_model(model_path)
                log.info("✅ fasttext yüklendi")
            except Exception as e:
                log.warning("fasttext yüklenemedi: %s", e); return None
        return _models["fasttext"]

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — NORMALİZASYON & DİL TESPİTİ (Katman 0)
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_RE = {
    "Hebrew":    re.compile(r"[\u0590-\u05FF]"),
    "Arabic":    re.compile(r"[\u0600-\u06FF]"),
    "Cyrillic":  re.compile(r"[\u0400-\u04FF]"),
    "Devanagari":re.compile(r"[\u0900-\u097F]"),
    "CJK":       re.compile(r"[\u4E00-\u9FFF]"),
}
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA9F\U00002300-\U000023FF]+",
    flags=re.UNICODE
)

def detect_script(text: str) -> str:
    for name, pat in SCRIPT_RE.items():
        if pat.search(text): return name
    return "Latin"

def detect_language(text: str) -> Tuple[str, float]:
    ft = get_fasttext()
    if ft:
        try:
            labels, probs = ft.predict(text.replace("\n"," "), k=1)
            lang = labels[0].replace("__label__","")
            return lang, float(probs[0])
        except: pass
    if LANGDETECT_OK:
        try:
            return langdetect_detect(text), 0.75
        except: pass
    return "und", 0.0

def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = re.sub(r"&amp;","&", text); text = re.sub(r"&lt;","<", text)
    text = re.sub(r"&gt;",">", text); text = re.sub(r"&quot;",'"', text)
    return text.strip()

def normalize_username(name: str) -> str:
    return unicodedata.normalize("NFKC", name).lower().strip()

def build_msg_id(video_id: str, author: str, timestamp: int, message: str) -> str:
    raw = f"{video_id}|{author}|{timestamp}|{message}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def extract_emojis(text: str) -> List[str]:
    return EMOJI_RE.findall(text)

def process_raw_message(raw: dict) -> Optional[dict]:
    """Ham veriyi standart mesaj formatına dönüştür"""
    try:
        message = normalize_text(raw.get("message","") or raw.get("text","") or "")
        if not message: return None
        author  = (raw.get("author","") or raw.get("username","")).strip()
        if not author: return None
        ts = int(raw.get("timestamp_utc", raw.get("timestamp",0)) or 0)
        video_id = raw.get("video_id","")
        lang, conf = detect_language(message[:200])
        return {
            "msg_id":            build_msg_id(video_id, author, ts, message),
            "video_id":          video_id,
            "title":             raw.get("title",""),
            "video_date":        raw.get("video_date",""),
            "author":            author,
            "author_channel_id": raw.get("author_channel_id", raw.get("author_cid","")),
            "message":           message,
            "timestamp_utc":     ts,
            "timestamp_iso":     datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "",
            "lang_detected":     lang,
            "lang_confidence":   conf,
            "script":            detect_script(message),
            "source_type":       raw.get("source_type","comment"),
            "emojis":            extract_emojis(message),
            "is_live":           raw.get("is_live", False),
        }
    except Exception as e:
        log.debug("Mesaj işleme hatası: %s", e)
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — YOUTUBE SCRAPER (yt-dlp + Selenium)
# ═══════════════════════════════════════════════════════════════════════════════
_selenium_driver = None
_selenium_lock = threading.Lock()

def create_firefox_driver(headless: bool = False) -> Optional[Any]:
    if not SELENIUM_OK: return None
    try:
        opts = FFOptions()
        if headless: opts.add_argument("--headless")
        opts.set_preference("dom.webnotifications.enabled", False)
        opts.set_preference("media.volume_scale", "0.0")
        driver = webdriver.Firefox(options=opts)
        driver.set_page_load_timeout(60)
        log.info("✅ Firefox WebDriver başlatıldı")
        return driver
    except Exception as e:
        log.error("Firefox başlatılamadı: %s", e)
        return None

def youtube_login(driver, email: str, password: str) -> bool:
    if not driver: return False
    try:
        driver.get("https://accounts.google.com/signin/v2/identifier?service=youtube")
        wait = WebDriverWait(driver, 20)
        email_field = wait.until(EC.presence_of_element_located((By.NAME, "identifier")))
        email_field.clear(); email_field.send_keys(email); email_field.send_keys(Keys.RETURN)
        time.sleep(2)
        pwd_field = wait.until(EC.presence_of_element_located((By.NAME, "Passwd")))
        pwd_field.clear(); pwd_field.send_keys(password); pwd_field.send_keys(Keys.RETURN)
        time.sleep(4)
        if "youtube.com" in driver.current_url or "accounts.google.com/signin/oauth" in driver.current_url:
            driver.get("https://www.youtube.com")
            time.sleep(2)
        log.info("✅ YouTube girişi başarılı: %s", email)
        return True
    except Exception as e:
        log.error("YouTube girişi başarısız: %s", e)
        return False

def get_channel_video_ids_ytdlp(channel_url: str, date_from: str = "2023-01-01",
                                  date_to: str = "2026-12-31") -> List[Dict]:
    """yt-dlp ile kanal videolarını listele"""
    Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist", "--no-download",
        "--print", "%(id)s\t%(title)s\t%(upload_date)s\t%(url)s",
        "--dateafter",  date_from.replace("-",""),
        "--datebefore", date_to.replace("-",""),
        "--ignore-errors",
        channel_url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip(): continue
            parts = line.split("\t")
            if len(parts) >= 2:
                vid_id = parts[0].strip()
                title  = parts[1].strip() if len(parts)>1 else ""
                date   = parts[2].strip() if len(parts)>2 else ""
                if vid_id and len(vid_id) == 11:
                    videos.append({"video_id": vid_id, "title": title,
                                   "video_date": date, "source_type": "stream"})
        log.info("✅ %d video bulundu", len(videos))
        return videos
    except Exception as e:
        log.error("yt-dlp video listesi hatası: %s", e)
        return []

def scrape_comments_ytdlp(video_id: str, title: str = "", video_date: str = "",
                           source_type: str = "comment") -> List[Dict]:
    """yt-dlp ile video yorumlarını çek"""
    out_dir = Path(CONFIG["data_dir"]) / "comments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{video_id}.json"
    if out_file.exists():
        log.info("  [cache] %s zaten mevcut", video_id)
        try:
            with open(out_file,"r",encoding="utf-8") as f:
                return json.load(f)
        except: pass
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--write-comments", "--skip-download",
        "--no-warnings", "--quiet",
        "-o", str(out_dir / f"{video_id}.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception as e:
        log.warning("yt-dlp yorum çekme hatası %s: %s", video_id, e)

    # yt-dlp .info.json içindeki yorumları oku
    info_file = out_dir / f"{video_id}.info.json"
    messages = []
    if info_file.exists():
        try:
            with open(info_file,"r",encoding="utf-8") as f:
                info = json.load(f)
            comments = info.get("comments", [])
            for c in comments:
                author = c.get("author","")
                text   = c.get("text","")
                ts     = int(c.get("timestamp",0) or 0)
                cid    = c.get("author_id","")
                if author and text:
                    raw = {"video_id": video_id, "title": title, "video_date": video_date,
                           "author": author, "author_channel_id": cid,
                           "message": text, "timestamp_utc": ts, "source_type": source_type}
                    msg = process_raw_message(raw)
                    if msg: messages.append(msg)
        except Exception as e:
            log.warning("JSON okuma hatası %s: %s", video_id, e)
    if messages:
        with open(out_file,"w",encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False)
    log.info("  %s → %d yorum", video_id, len(messages))
    return messages

def scrape_live_chat_ytdlp(video_id: str, title: str = "", video_date: str = "") -> List[Dict]:
    """yt-dlp ile live chat replay çek"""
    out_dir = Path(CONFIG["data_dir"]) / "live_chats"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"{video_id}_chat.json"
    if cache.exists():
        try:
            with open(cache,"r",encoding="utf-8") as f:
                return json.load(f)
        except: pass
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--write-subs", "--sub-format","json3",
        "--skip-download", "--no-warnings", "--quiet",
        "--sub-langs","live_chat",
        "-o", str(out_dir / f"{video_id}.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
    except Exception as e:
        log.warning("Live chat çekme hatası %s: %s", video_id, e)
    chat_file = out_dir / f"{video_id}.live_chat.json"
    messages = []
    if chat_file.exists():
        try:
            with open(chat_file,"r",encoding="utf-8") as f:
                chat_data = json.load(f)
            events = chat_data.get("events",[])
            for ev in events:
                segs = ev.get("segs",[])
                text = "".join(s.get("utf8","") for s in segs).strip()
                if not text: continue
                author = ev.get("authorName","")
                ts_ms  = int(ev.get("tOffsetMs", ev.get("videoOffsetTimeMsec",0)) or 0)
                cid    = ev.get("authorExternalChannelId","")
                raw = {"video_id": video_id, "title": title, "video_date": video_date,
                       "author": author, "author_channel_id": cid,
                       "message": text, "timestamp_utc": ts_ms // 1000,
                       "source_type": "replay_chat", "is_live": False}
                msg = process_raw_message(raw)
                if msg: messages.append(msg)
        except Exception as e:
            log.warning("Live chat JSON hatası %s: %s", video_id, e)
    if messages:
        with open(cache,"w",encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False)
    return messages

def fetch_live_stream_chat_selenium(driver, video_url: str,
                                     video_id: str, video_title: str) -> List[Dict]:
    """Selenium ile CANLI yayın chat mesajlarını gerçek zamanlı çek"""
    if not driver: return []
    messages = []
    try:
        driver.get(video_url)
        time.sleep(3)
        chat_items = driver.find_elements(By.CSS_SELECTOR,
            "yt-live-chat-text-message-renderer, yt-live-chat-paid-message-renderer")
        now_ts = int(time.time())
        for item in chat_items:
            try:
                author_el = item.find_element(By.ID, "author-name")
                msg_el    = item.find_element(By.ID, "message")
                author = author_el.text.strip()
                text   = msg_el.text.strip()
                if author and text:
                    raw = {"video_id": video_id, "title": video_title,
                           "author": author, "message": text,
                           "timestamp_utc": now_ts, "source_type": "live", "is_live": True}
                    msg = process_raw_message(raw)
                    if msg: messages.append(msg)
            except: pass
    except Exception as e:
        log.warning("Selenium live chat hatası: %s", e)
    return messages

def full_scrape_channel(emit_progress=None):
    """Tüm kanal videolarını tara, yorumları ve chatleri çek, DB'ye kaydet"""
    videos = get_channel_video_ids_ytdlp(
        CONFIG["channel_url"],
        CONFIG.get("date_from","2023-01-01"),
        CONFIG.get("date_to","2026-12-31")
    )
    if not videos:
        log.warning("Video bulunamadı")
        return 0
    total_msgs = 0
    for i, vid in enumerate(videos):
        vid_id  = vid["video_id"]
        title   = vid["title"]
        v_date  = vid["video_date"]
        s_type  = vid.get("source_type","comment")
        if emit_progress:
            emit_progress({"step": i+1, "total": len(videos),
                           "video_id": vid_id, "title": title})
        # Yorumlar
        comments = scrape_comments_ytdlp(vid_id, title, v_date, s_type)
        # Canlı yayın replays
        chats = scrape_live_chat_ytdlp(vid_id, title, v_date)
        all_msgs = comments + chats
        for msg in all_msgs:
            upsert_message(msg)
            total_msgs += 1
        db_exec("INSERT OR REPLACE INTO scraped_videos(video_id,title,video_date,source_type)"
                " VALUES(?,?,?,?)", (vid_id, title, v_date, s_type))
        log.info("[%d/%d] %s — %d mesaj (toplam: %d)",
                 i+1, len(videos), vid_id, len(all_msgs), total_msgs)
    return total_msgs

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — NLP PİPELİNE (Katman 1-2)
# ═══════════════════════════════════════════════════════════════════════════════
_global_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3),
                                 analyzer="word", sublinear_tf=True)
_tfidf_fitted = False
_tfidf_lock = threading.Lock()

def fit_tfidf(all_texts: List[str]):
    global _tfidf_fitted
    with _tfidf_lock:
        if all_texts:
            _global_tfidf.fit(all_texts)
            _tfidf_fitted = True

def get_tfidf_vector(text: str) -> np.ndarray:
    with _tfidf_lock:
        if not _tfidf_fitted:
            return np.zeros(100)
        try:
            v = _global_tfidf.transform([text])
            return v.toarray()[0]
        except:
            return np.zeros(_global_tfidf.max_features or 100)

def embed_text(text: str) -> Optional[List[float]]:
    model = get_sbert()
    if model is None: return None
    try:
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    except Exception as e:
        log.debug("Embedding hatası: %s", e); return None

def embed_texts_batch(texts: List[str]) -> Optional[np.ndarray]:
    model = get_sbert()
    if model is None: return None
    try:
        embs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return embs
    except Exception as e:
        log.debug("Batch embedding hatası: %s", e); return None

def ngram_fingerprint(text: str, n_range: Tuple = (2,3)) -> Counter:
    """Karakter n-gram frekans sayacı"""
    text = text.lower()
    fp = Counter()
    for n in range(n_range[0], n_range[1]+1):
        for i in range(len(text)-n+1):
            fp[text[i:i+n]] += 1
    return fp

def jaccard_ngram_sim(fp_a: Counter, fp_b: Counter) -> float:
    set_a = set(fp_a.keys()); set_b = set(fp_b.keys())
    inter = len(set_a & set_b); union = len(set_a | set_b)
    return inter / union if union else 0.0

def pos_profile(text: str) -> dict:
    nlp = get_spacy()
    if not nlp: return {}
    doc = nlp(text[:1000])
    counts = Counter(t.pos_ for t in doc)
    total = max(len(doc), 1)
    return {pos: cnt/total for pos, cnt in counts.items()}

def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def punctuation_density(text: str) -> float:
    if not text: return 0.0
    return sum(1 for c in text if c in ".,!?;:'\"()[]{}") / max(len(text),1)

def lexical_diversity(tokens: List[str]) -> float:
    if not tokens: return 0.0
    return len(set(tokens)) / len(tokens)

def shannon_entropy(text: str) -> float:
    if not text: return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((cnt/total)*math.log2(cnt/total+1e-12) for cnt in freq.values())

def typo_fingerprint(messages: List[str]) -> dict:
    text = " ".join(messages)
    tokens = text.split()
    return {
        "double_letters":     len(re.findall(r"(\w)\1{2,}", text)),
        "uppercase_ratio":    uppercase_ratio(text),
        "punctuation_density":punctuation_density(text),
        "ellipsis_use":       text.count("...") / max(len(messages),1),
        "avg_msg_length":     sum(len(m) for m in messages) / max(len(messages),1),
        "emoji_density":      len(extract_emojis(text)) / max(len(text),1),
        "exclamation_rate":   text.count("!") / max(len(text),1),
        "question_rate":      text.count("?") / max(len(text),1),
    }

def burrows_delta(vectors: List[np.ndarray], user_a_idx: int, user_b_idx: int) -> float:
    """Burrows Delta stilometrik mesafe"""
    arr = np.array(vectors)
    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0) + 1e-12
    z_a = (arr[user_a_idx] - mu) / sigma
    z_b = (arr[user_b_idx] - mu) / sigma
    return float(np.mean(np.abs(z_a - z_b)))

def cosine_delta(vec_a: np.ndarray, vec_b: np.ndarray, mean_vec: np.ndarray,
                 std_vec: np.ndarray) -> float:
    """Cosine Delta"""
    z_a = (vec_a - mean_vec) / (std_vec + 1e-12)
    z_b = (vec_b - mean_vec) / (std_vec + 1e-12)
    norm_a = np.linalg.norm(z_a); norm_b = np.linalg.norm(z_b)
    if norm_a < 1e-9 or norm_b < 1e-9: return 1.0
    return 1.0 - np.dot(z_a, z_b) / (norm_a * norm_b)

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p = p / p.sum(); q = q / q.sum()
    m = (p + q) / 2
    return float(0.5 * scipy_entropy(p, m) + 0.5 * scipy_entropy(q, m))

def compute_composite_similarity(emb_sim: float, ngram_sim: float,
                                  typo_sim: float, time_sim: float,
                                  topic_sim: float,
                                  weights: List[float] = None) -> float:
    if weights is None: weights = [0.35, 0.25, 0.15, 0.15, 0.10]
    return sum(w*s for w,s in zip(weights,
               [emb_sim, ngram_sim, typo_sim, time_sim, topic_sim]))

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8 — BOT TESPİTİ (Katman 3)
# ═══════════════════════════════════════════════════════════════════════════════
def burstiness(timestamps: List[int]) -> float:
    if len(timestamps) < 3: return 0.0
    diffs = np.diff(sorted(timestamps)).astype(float)
    mu = np.mean(diffs); sigma = np.std(diffs)
    if mu + sigma < 1e-9: return 0.0
    return float((sigma - mu) / (sigma + mu))

def hawkes_intensity(t: float, history: List[float],
                     mu: float = 0.1, alpha: float = 0.5, beta: float = 1.0) -> float:
    """Hawkes process intensity λ(t)"""
    kernel = sum(alpha * math.exp(-beta*(t-ti)) for ti in history if ti < t)
    return mu + kernel

def hawkes_stalker_score(user_ts: List[int], target_ts: List[int], delta: int = 60) -> float:
    """Kullanıcının hedef mesajlarından sonra gelen mesaj oranı"""
    if not user_ts or not target_ts: return 0.0
    responses = 0
    for u_t in user_ts:
        for tgt_t in target_ts:
            if 0 < u_t - tgt_t <= delta:
                responses += 1
                break
    return responses / max(len(user_ts), 1)

def repetition_score(messages: List[str]) -> float:
    if len(messages) < 2: return 0.0
    pairs = [(messages[i], messages[i+1]) for i in range(len(messages)-1)]
    sims = []
    for a, b in pairs[:50]:
        tokens_a = set(a.lower().split()); tokens_b = set(b.lower().split())
        inter = len(tokens_a & tokens_b); union = len(tokens_a | tokens_b)
        sims.append(inter/union if union else 0)
    return float(np.mean(sims)) if sims else 0.0

def heuristic_bot_score(messages: List[str], timestamps: List[int]) -> float:
    if not messages: return 0.0
    all_text = " ".join(messages)
    tokens = all_text.lower().split()
    D = lexical_diversity(tokens)
    H = shannon_entropy(all_text) / 4.5   # normalize by max ~4.5
    L = min(1.0, sum(len(m) for m in messages)/max(len(messages),1) / 80)
    Q = sum(1 for m in messages if "?" in m) / max(len(messages),1)
    P = punctuation_density(all_text)
    E = len(extract_emojis(all_text)) / max(len(all_text),1) * 100
    U = uppercase_ratio(all_text)
    R = repetition_score(messages)
    score = 1 - (0.28*D + 0.18*min(1,H) + 0.12*min(1,L) +
                 0.10*Q + 0.10*P + 0.07*min(1,E) +
                 0.05*(1-U) + 0.10*(1-R))
    return max(0.0, min(1.0, float(score)))

def bart_classify(text: str, labels: List[str], hypothesis: str = "This text is {}.") -> Dict[str,float]:
    bart = get_bart_zeroshot()
    if not bart:
        return {l: 1/len(labels) for l in labels}
    try:
        result = bart(text[:512], candidate_labels=labels, hypothesis_template=hypothesis)
        return dict(zip(result["labels"], result["scores"]))
    except Exception as e:
        log.debug("BART sınıflandırma hatası: %s", e)
        return {l: 1/len(labels) for l in labels}

def compute_bot_score(author: str, messages: List[str], timestamps: List[int]) -> float:
    if not messages: return 0.0
    heuristic = heuristic_bot_score(messages, timestamps)
    bart_scores = bart_classify(" ".join(messages[:5]), BOT_ZEROSHOT_LABELS)
    bart_bot = bart_scores.get("spam or bot-like message", 0.5)
    return 0.55 * bart_bot + 0.45 * heuristic

def detect_co_entry(events: List[Tuple[str,int]], delta_sec: int = 300) -> List[Tuple]:
    co = []
    for i, (ua, ta) in enumerate(events):
        for ub, tb in events[i+1:]:
            if abs(ta - tb) <= delta_sec and ua != ub:
                co.append((ua, ub, abs(ta-tb)))
    return co

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 9 — NEFRET SÖYLEMİ & KİMLİK ÖRTÜsü TESPİTİ (Katman 4-5)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_hate_scores(text: str) -> Dict[str, float]:
    scores = bart_classify(text, THREAT_LABELS_ZEROSHOT)
    antisem = max(scores.get("antisemitic content",0),
                  scores.get("hate speech against Jewish people",0))
    groyper  = scores.get("groyper movement content", 0)
    hate     = max(scores.get("islamophobic content",0),
                   scores.get("white supremacist content",0)) + groyper * 0.5
    stalker  = scores.get("harassment and stalking behavior", 0)
    imperso  = scores.get("identity impersonation", 0)
    bot_sig  = scores.get("coordinated bot attack", 0)
    neutral  = scores.get("neutral friendly message", 0)
    overall_hate = max(antisem, hate * 0.8)
    return {
        "antisemitism": round(antisem, 4),
        "hate_general": round(hate, 4),
        "groyper":      round(groyper, 4),
        "stalker_sig":  round(stalker, 4),
        "impersonation":round(imperso, 4),
        "bot_signal":   round(bot_sig, 4),
        "neutral":      round(neutral, 4),
        "overall":      round(overall_hate, 4),
    }

def persona_masking_score(candidate: str, candidate_msgs: List[str],
                           known_users: Dict[str, List[str]]) -> Tuple[float, str]:
    """
    Kimlik örtüsü: Aday kullanıcının mesajları başka bir kullanıcıya
    benziyor mu? İsim benzerliği + stilometri.
    """
    candidate_norm = normalize_username(candidate)
    best_sim = 0.0; best_match = ""
    cand_emb = embed_text(" ".join(candidate_msgs[:20]) or "no text")

    for known_user, known_msgs in known_users.items():
        if known_user == candidate: continue
        known_norm = normalize_username(known_user)
        # İsim benzerliği
        name_sim = 0.0
        for i in range(min(len(candidate_norm), len(known_norm))):
            if i < len(candidate_norm) and i < len(known_norm):
                if candidate_norm[i] == known_norm[i]: name_sim += 1
        name_sim = name_sim / max(len(candidate_norm), len(known_norm), 1)

        # Embedding benzerliği
        emb_sim = 0.0
        if cand_emb and known_msgs:
            known_emb = embed_text(" ".join(known_msgs[:20]))
            if known_emb:
                emb_sim = float(1 - cosine_dist(cand_emb, known_emb))

        combined = 0.4 * name_sim + 0.6 * emb_sim
        if combined > best_sim:
            best_sim = combined; best_match = known_user
    return best_sim, best_match

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 10 — KONU MODELLEMESİ (Katman 6)
# ═══════════════════════════════════════════════════════════════════════════════
_bertopic_model = None

def fit_topic_model(documents: List[str], n_topics: int = 20) -> Tuple[Any, List[int]]:
    global _bertopic_model
    if not BERTOPIC_OK or len(documents) < 50:
        log.warning("BERTopic kullanılamıyor, LDA yedek kullanılıyor")
        return _lda_topic_model(documents, n_topics)
    try:
        from bertopic import BERTopic
        _bertopic_model = BERTopic(nr_topics=n_topics, language="multilingual",
                                    verbose=False, calculate_probabilities=True)
        topics, probs = _bertopic_model.fit_transform(documents)
        return _bertopic_model, topics
    except Exception as e:
        log.warning("BERTopic hatası: %s. LDA yedek.", e)
        return _lda_topic_model(documents, n_topics)

def _lda_topic_model(documents: List[str], n_topics: int = 20):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(max_features=2000, ngram_range=(1,2))
    try:
        dtm = vec.fit_transform(documents)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        topic_assignments = lda.transform(dtm).argmax(axis=1).tolist()
        return lda, topic_assignments
    except Exception as e:
        log.warning("LDA hatası: %s", e)
        return None, [0]*len(documents)

def get_user_topic_vector(user_msgs: List[str]) -> np.ndarray:
    if not user_msgs or _bertopic_model is None:
        return np.zeros(20)
    try:
        _, probs = _bertopic_model.transform(user_msgs)
        return np.mean(probs, axis=0)
    except:
        return np.zeros(20)

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 11 — ZAMANSAL ANALİZ (Katman 7)
# ═══════════════════════════════════════════════════════════════════════════════
def temporal_fingerprint(timestamps: List[int]) -> dict:
    if not timestamps: return {}
    ts_arr = np.array(sorted(timestamps))
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts_arr]
    hours = [dt.hour for dt in dts]
    days  = [dt.weekday() for dt in dts]
    diffs = np.diff(ts_arr).astype(float) if len(ts_arr) > 1 else np.array([0.0])
    return {
        "peak_hour":       int(Counter(hours).most_common(1)[0][0]) if hours else 0,
        "active_days":     list(set(days)),
        "mean_interval":   float(np.mean(diffs)),
        "std_interval":    float(np.std(diffs)),
        "burstiness":      burstiness(list(ts_arr)),
        "min_interval":    float(np.min(diffs)) if len(diffs) else 0,
        "max_interval":    float(np.max(diffs)) if len(diffs) else 0,
    }

def time_similarity(tfp_a: dict, tfp_b: dict) -> float:
    if not tfp_a or not tfp_b: return 0.0
    hour_sim = 1 - abs(tfp_a.get("peak_hour",0) - tfp_b.get("peak_hour",0)) / 24.0
    day_a = set(tfp_a.get("active_days",[])); day_b = set(tfp_b.get("active_days",[]))
    day_sim = len(day_a&day_b)/max(len(day_a|day_b),1)
    burst_sim = 1 - abs(tfp_a.get("burstiness",0)-tfp_b.get("burstiness",0))/2
    return float(0.4*hour_sim + 0.3*day_sim + 0.3*burst_sim)

def changepoint_detection_simple(values: List[float]) -> List[int]:
    """Basit CUSUM tabanlı changepoint detection"""
    if len(values) < 4: return []
    arr = np.array(values, dtype=float)
    mu = np.mean(arr); sigma = np.std(arr)+1e-9
    cusum = np.cumsum((arr - mu)/sigma)
    threshold = 3.0
    changepoints = []
    for i in range(1, len(cusum)-1):
        if abs(cusum[i]) > threshold and abs(cusum[i]) > abs(cusum[i-1]) and abs(cusum[i]) > abs(cusum[i+1]):
            changepoints.append(i)
    return changepoints

def ks_test_temporal(ts_a: List[int], ts_b: List[int]) -> Tuple[float, float]:
    if len(ts_a)<3 or len(ts_b)<3: return 0.0, 1.0
    diffs_a = np.diff(sorted(ts_a)).astype(float)
    diffs_b = np.diff(sorted(ts_b)).astype(float)
    stat, pval = stats.ks_2samp(diffs_a, diffs_b)
    return float(stat), float(pval)

def pearson_correlation_activity(ts_a: List[int], ts_b: List[int],
                                   bin_size: int = 3600) -> float:
    if not ts_a or not ts_b: return 0.0
    all_ts = ts_a + ts_b
    t_min = min(all_ts); t_max = max(all_ts)
    if t_max == t_min: return 0.0
    n_bins = max(10, (t_max-t_min)//bin_size + 1)
    bins_a = np.zeros(n_bins); bins_b = np.zeros(n_bins)
    for t in ts_a:
        idx = min(int((t-t_min)//bin_size), n_bins-1); bins_a[idx]+=1
    for t in ts_b:
        idx = min(int((t-t_min)//bin_size), n_bins-1); bins_b[idx]+=1
    if bins_a.std()<1e-9 or bins_b.std()<1e-9: return 0.0
    corr = np.corrcoef(bins_a, bins_b)[0,1]
    return float(corr) if not np.isnan(corr) else 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 12 — GRAF KÜMELEMESİ & İLİŞKİ AĞI (Katman 8)
# ═══════════════════════════════════════════════════════════════════════════════
def build_similarity_graph(users: List[str],
                            sim_matrix: np.ndarray,
                            threshold: float = None) -> nx.Graph:
    if threshold is None: threshold = CONFIG.get("similarity_threshold", 0.65)
    G = nx.Graph()
    G.add_nodes_from(users)
    n = len(users)
    for i in range(n):
        for j in range(i+1, n):
            sim = float(sim_matrix[i,j])
            if sim >= threshold:
                G.add_edge(users[i], users[j], weight=sim)
    return G

def dbscan_cluster(sim_matrix: np.ndarray, eps: float = None) -> np.ndarray:
    if eps is None: eps = 1 - CONFIG.get("similarity_threshold",0.65)
    dist_matrix = 1 - np.clip(sim_matrix, 0, 1)
    db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    return labels

def spectral_cluster(sim_matrix: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    n = sim_matrix.shape[0]
    if n < n_clusters: n_clusters = max(2, n//2)
    try:
        sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                                 random_state=42)
        return sc.fit_predict(np.clip(sim_matrix, 0, 1))
    except Exception as e:
        log.warning("Spectral clustering hatası: %s", e)
        return np.zeros(n, dtype=int)

def louvain_cluster(G: nx.Graph) -> Dict[str, int]:
    if community_louvain is None:
        return {n: 0 for n in G.nodes()}
    try:
        partition = community_louvain.best_partition(G, weight="weight")
        return partition
    except Exception as e:
        log.warning("Louvain hatası: %s", e)
        return {n: 0 for n in G.nodes()}

def compute_pagerank(G: nx.Graph) -> Dict[str, float]:
    if len(G.nodes()) == 0: return {}
    try:
        return nx.pagerank(G, weight="weight", alpha=0.85)
    except:
        return {n: 1/max(len(G.nodes()),1) for n in G.nodes()}

def mutual_information_activity(ts_a: List[int], ts_b: List[int],
                                  bin_size: int = 3600) -> float:
    """Aktivite zaman serisi mutual information"""
    if not ts_a or not ts_b: return 0.0
    all_ts = ts_a + ts_b; t_min = min(all_ts); t_max = max(all_ts)
    if t_max == t_min: return 0.0
    n_bins = max(10, (t_max-t_min)//bin_size + 1)
    bins_a = np.zeros(n_bins, dtype=int); bins_b = np.zeros(n_bins, dtype=int)
    for t in ts_a:
        idx = min(int((t-t_min)//bin_size), n_bins-1); bins_a[idx]+=1
    for t in ts_b:
        idx = min(int((t-t_min)//bin_size), n_bins-1); bins_b[idx]+=1
    bins_a = (bins_a > 0).astype(int); bins_b = (bins_b > 0).astype(int)
    from sklearn.metrics import mutual_info_score
    try:
        return float(mutual_info_score(bins_a, bins_b))
    except:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 13 — Q-LEARNING & DQN (Katman 9)
# ═══════════════════════════════════════════════════════════════════════════════
class QTable:
    """Tabular Q-Learning — 5 boyutlu durum uzayı, 6 eylem"""
    def __init__(self, state_dims: Tuple = (10,10,10,10,10), n_actions: int = 6,
                  alpha: float = 0.15, gamma: float = 0.90, epsilon: float = 0.1):
        self.q = np.zeros((*state_dims, n_actions))
        self.alpha = alpha; self.gamma = gamma
        self.epsilon = epsilon; self.n_actions = n_actions
        self.step = 0

    def encode_state(self, count: int, rep: float, diversity: float,
                      human_score: float, burstiness_val: float) -> tuple:
        s1 = min(9, count//5)
        s2 = min(9, int(rep*10))
        s3 = min(9, int(diversity*10))
        s4 = min(9, int(human_score*10))
        s5 = min(9, int((burstiness_val+1)/2*10))
        return (s1, s2, s3, s4, s5)

    def act(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        return int(np.argmax(self.q[state]))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        self.step += 1
        self.epsilon = max(0.01, self.epsilon * 0.9999)
        best_next = np.max(self.q[next_state])
        self.q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q[state][action]
        )

    def save(self, path: str = "qtable.npy"):
        np.save(path, self.q)

    def load(self, path: str = "qtable.npy"):
        if Path(path).exists():
            self.q = np.load(path)

_qtable = QTable()

if TORCH_OK:
    import torch.nn as nn
    import torch.optim as optim

    class DQN(nn.Module):
        def __init__(self, state_dim: int = 64, n_actions: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256), nn.ReLU(),
                nn.Linear(256, 128),       nn.ReLU(),
                nn.Linear(128, 64),        nn.ReLU(),
                nn.Linear(64, n_actions)
            )
        def forward(self, x): return self.net(x)

    class DQNAgent:
        def __init__(self, state_dim: int = 64, n_actions: int = 6,
                      lr: float = 1e-4, gamma: float = 0.90, epsilon: float = 0.1):
            self.device = torch.device(DEVICE if DEVICE != "cpu" else "cpu")
            self.n_actions = n_actions; self.gamma = gamma
            self.epsilon = epsilon; self.step = 0
            self.policy_net = DQN(state_dim, n_actions).to(self.device)
            self.target_net = DQN(state_dim, n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.memory = deque(maxlen=10000)
            self.batch_size = 64
            self.update_target_every = 100

        def build_state(self, user_profile: dict) -> torch.Tensor:
            feats = [
                float(user_profile.get("msg_count",0)/100),
                float(user_profile.get("bot_prob",0)),
                float(user_profile.get("hate_score",0)),
                float(user_profile.get("stalker_score",0)),
                float(user_profile.get("human_score",0.5)),
                float(user_profile.get("impersonator_prob",0)),
                float(user_profile.get("threat_score",0)),
            ]
            feats = feats + [0.0]*(64-len(feats))
            return torch.tensor(feats[:64], dtype=torch.float32, device=self.device)

        def act(self, state: torch.Tensor) -> int:
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions-1)
            with torch.no_grad():
                q_vals = self.policy_net(state.unsqueeze(0))
                return int(q_vals.argmax().item())

        def remember(self, s, a, r, s_next, done):
            self.memory.append((s, a, r, s_next, done))

        def train_step(self):
            if len(self.memory) < self.batch_size: return
            batch = random.sample(self.memory, self.batch_size)
            states  = torch.stack([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch], device=self.device)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
            next_states = torch.stack([b[3] for b in batch])
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad(); loss.backward()
            self.optimizer.step()
            self.step += 1
            if self.step % self.update_target_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = max(0.01, self.epsilon * 0.9999)

    _dqn_agent = DQNAgent()
else:
    _dqn_agent = None

def rl_recommend_action(user_profile: dict) -> Tuple[int, str]:
    """Q-Learning veya DQN ile eylem önerisi"""
    count = int(user_profile.get("msg_count", 0))
    rep   = float(user_profile.get("bot_prob", 0))
    div   = float(user_profile.get("human_score", 0.5))
    hs    = float(user_profile.get("human_score", 0.5))
    burst = float(user_profile.get("threat_score", 0))
    state = _qtable.encode_state(count, rep, div, hs, burst)
    action = _qtable.act(state)
    return action, ACTION_NAMES.get(action, "UNKNOWN")

def rl_reward_update(user_profile: dict, action: int, reward: float,
                      next_profile: dict):
    count_n = int(next_profile.get("msg_count",0))
    rep_n   = float(next_profile.get("bot_prob",0))
    div_n   = float(next_profile.get("human_score",0.5))
    hs_n    = float(next_profile.get("human_score",0.5))
    burst_n = float(next_profile.get("threat_score",0))
    count = int(user_profile.get("msg_count",0))
    rep   = float(user_profile.get("bot_prob",0))
    div   = float(user_profile.get("human_score",0.5))
    hs    = float(user_profile.get("human_score",0.5))
    burst = float(user_profile.get("threat_score",0))
    state = _qtable.encode_state(count, rep, div, hs, burst)
    next_state = _qtable.encode_state(count_n, rep_n, div_n, hs_n, burst_n)
    _qtable.update(state, action, reward, next_state)
    if _dqn_agent:
        s  = _dqn_agent.build_state(user_profile)
        sn = _dqn_agent.build_state(next_profile)
        _dqn_agent.remember(s, action, reward, sn, False)
        _dqn_agent.train_step()

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 14 — OYUN KURAMI (Katman 10)
# ═══════════════════════════════════════════════════════════════════════════════
def find_nash_equilibria() -> List[Tuple[int,int]]:
    nash = []
    n_mod = len(MOD_ACTIONS); n_act = len(ACTOR_ACTIONS)
    for i in range(n_mod):
        for j in range(n_act):
            pair = PAYOFF_MATRIX[i,j]
            mod_payoff = pair[0]; act_payoff = pair[1]
            mod_best = mod_payoff == max(PAYOFF_MATRIX[k,j][0] for k in range(n_mod))
            act_best = act_payoff == max(PAYOFF_MATRIX[i,l][1] for l in range(n_act))
            if mod_best and act_best:
                nash.append((i, j, MOD_ACTIONS[i], ACTOR_ACTIONS[j]))
    return nash

def grim_trigger(history: List[str], current_action: str) -> str:
    """Grim Trigger stratejisi"""
    bad_actions = {"TROLL","FLOOD","IMPERSONATE","COORDINATE"}
    if any(a in bad_actions for a in history):
        return "BAN"
    return "MONITOR"

def bayesian_belief_update(prior: Dict[str,float], likelihood: Dict[str,float]) -> Dict[str,float]:
    """P(θ|mesaj) ∝ P(mesaj|θ)·P(θ)"""
    posterior = {}
    denom = sum(likelihood.get(k,1e-9)*prior.get(k,0.25) for k in prior)
    for k in prior:
        posterior[k] = (likelihood.get(k,1e-9) * prior.get(k,0.25)) / max(denom, 1e-12)
    return posterior

def compute_game_score(user_history: List[str],
                        threat_score: float,
                        prior: Dict[str,float] = None) -> dict:
    if prior is None:
        prior = {"BOT":0.15,"HATER":0.15,"STALKER":0.10,"NORMAL":0.60}
    likelihood = {
        "BOT":    threat_score * 0.8,
        "HATER":  threat_score * 0.9,
        "STALKER":threat_score * 0.6,
        "NORMAL": max(0.01, 1 - threat_score)
    }
    posterior = bayesian_belief_update(prior, likelihood)
    mod_action = grim_trigger(user_history, "")
    best_action_idx = MOD_ACTIONS.index(mod_action) if mod_action in MOD_ACTIONS else 3
    nash = find_nash_equilibria()
    return {
        "posterior": posterior,
        "recommended_mod_action": mod_action,
        "nash_equilibria": nash,
        "dominant_type": max(posterior, key=posterior.get),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 15 — BAYES/MARKOV/KALMAN (Katman 11)
# ═══════════════════════════════════════════════════════════════════════════════
class KalmanThreatFilter:
    """Kullanıcı tehdit skorunu zamanla takip eden Kalman filtresi"""
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.x = 0.0   # Tahmin edilen tehdit skoru
        self.P = 1.0   # Kovaryans
        self.Q = process_noise
        self.R = measurement_noise
        self.F = 1.0; self.H = 1.0

    def predict(self):
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q

    def update(self, z: float):
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.x = self.x + K * (z - self.H * self.x)
        self.P = (1 - K * self.H) * self.P
        return self.x

_kalman_filters: Dict[str, KalmanThreatFilter] = {}

def kalman_update_user(author: str, new_score: float) -> float:
    if author not in _kalman_filters:
        _kalman_filters[author] = KalmanThreatFilter()
    kf = _kalman_filters[author]
    kf.predict()
    return kf.update(new_score)

def hmm_state_sequence(threat_scores: List[float]) -> List[str]:
    """Basit 3-durumlu HMM ile durum dizisi tahmini"""
    if not HMMLEARN_OK or len(threat_scores) < 3:
        states = ["NORMAL" if s < 0.3 else "SUSPICIOUS" if s < 0.6 else "ATTACKING"
                  for s in threat_scores]
        return states
    try:
        obs = np.array(threat_scores).reshape(-1,1)
        model = hmmlearn_hmm.GaussianHMM(n_components=3, covariance_type="diag",
                                          n_iter=100, random_state=42)
        model.fit(obs)
        hidden_states = model.predict(obs)
        # En düşük ortalama = NORMAL, en yüksek = ATTACKING
        means = [model.means_[i][0] for i in range(3)]
        order = sorted(range(3), key=lambda x: means[x])
        state_names = {order[0]:"NORMAL", order[1]:"LURKING", order[2]:"ATTACKING"}
        return [state_names.get(s,"NORMAL") for s in hidden_states]
    except Exception as e:
        log.debug("HMM hatası: %s", e)
        return ["NORMAL"]*len(threat_scores)

def naive_bayes_predict(texts: List[str], labels: List[str],
                         new_text: str) -> Dict[str,float]:
    """Basit Naive Bayes metin sınıflandırma"""
    if len(texts) < 10 or len(set(labels)) < 2:
        return {"NORMAL":0.5,"THREAT":0.5}
    try:
        vec = TfidfVectorizer(max_features=500)
        X = vec.fit_transform(texts)
        nb = ComplementNB()
        nb.fit(X, labels)
        xnew = vec.transform([new_text])
        proba = nb.predict_proba(xnew)[0]
        return dict(zip(nb.classes_, proba))
    except Exception as e:
        log.debug("NB hatası: %s", e)
        return {"NORMAL":0.5,"THREAT":0.5}

def theorem_router_local(user_profile: dict) -> str:
    """Ollama tabanlı teorem yönlendirici (Ollama mevcut değilse kural tabanlı)"""
    ts = float(user_profile.get("threat_score",0))
    msg_count = int(user_profile.get("msg_count",0))
    if msg_count > 20 and ts > 0.5:
        return "HMM"
    elif ts > 0.6:
        return "BayesianUpdate"
    elif ts < 0.2 and msg_count > 5:
        return "MarkovChain"
    else:
        return "KalmanFilter"

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 16 — OLLAMA ENTEGRASYONU (Sadece yorum analizi için)
# ═══════════════════════════════════════════════════════════════════════════════
def ollama_analyze_comments(author: str, messages: List[str], task: str = "threat_analysis") -> dict:
    """Ollama phi4:14b ile yorum analizi — SADECE bu işlev için kullanılır"""
    if not OLLAMA_OK:
        return {"error": "Ollama SDK yüklü değil", "summary": "", "threat_indicators": []}
    model_name = CONFIG.get("ollama_model","phi4:14b")
    context = "\n".join([f"- {m[:200]}" for m in messages[:15]])
    prompt = f"""
Kullanıcı: {author}
Görev: {task}

Mesajlar:
{context}

Bu mesajları analiz et. JSON formatında döndür (başka bir şey yazma):
{{
  "summary": "kısa özet",
  "threat_indicators": ["liste"],
  "identity_clues": ["liste"],
  "language_patterns": ["liste"],
  "recommended_action": "BAN|WARN|MONITOR|IGNORE",
  "confidence": 0.0
}}
"""
    try:
        response = ollama_sdk.chat(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            options={"temperature":0.1}
        )
        raw = response["message"]["content"].strip()
        # JSON çıkar
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"summary": raw, "threat_indicators": [], "identity_clues": [],
                "language_patterns": [], "recommended_action": "MONITOR", "confidence": 0.5}
    except Exception as e:
        log.warning("Ollama hatası: %s", e)
        return {"error": str(e), "summary": "", "threat_indicators": [],
                "recommended_action": "MONITOR", "confidence": 0.0}

def ollama_theorem_router(user_profile: dict) -> str:
    """Ollama ile teorem yönlendirici"""
    if not OLLAMA_OK: return theorem_router_local(user_profile)
    model_name = CONFIG.get("ollama_model","phi4:14b")
    summary = json.dumps({
        "msg_count": user_profile.get("msg_count",0),
        "bot_prob": user_profile.get("bot_prob",0),
        "hate_score": user_profile.get("hate_score",0),
        "stalker_score": user_profile.get("stalker_score",0),
        "threat_level": user_profile.get("threat_level","GREEN")
    }, ensure_ascii=False)
    prompt = f"""
Kullanıcı profili: {summary}

Hangi istatistiksel yöntem en uygun? Sadece yöntem adını döndür:
- Zaman serisi tutarsızlığı → MarkovChain
- İnanç güncelleme → BayesianUpdate
- Ani davranış değişimi → ChangePoint
- Grup koordinasyonu → GraphClustering
- Kimlik belirsizliği → GaussianMixture
- Tehdit skoru takibi → KalmanFilter
"""
    try:
        resp = ollama_sdk.chat(model=model_name,
                               messages=[{"role":"user","content":prompt}],
                               options={"temperature":0.0})
        text = resp["message"]["content"].strip().split("\n")[0]
        valid = ["MarkovChain","BayesianUpdate","ChangePoint","GraphClustering",
                 "GaussianMixture","KalmanFilter","HMM"]
        for v in valid:
            if v.lower() in text.lower():
                return v
        return "KalmanFilter"
    except:
        return theorem_router_local(user_profile)

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 17 — KULLANICI HESAP ANALİZİ (Selenium)
# ═══════════════════════════════════════════════════════════════════════════════
_account_cache: Dict[str, dict] = {}

def inspect_user_account_selenium(driver, channel_id: str) -> dict:
    """Selenium ile kullanıcı hesap bilgilerini çek"""
    if not driver or not channel_id: return {}
    if channel_id in _account_cache: return _account_cache[channel_id]
    result = {"channel_id": channel_id, "account_created": "", "subscriber_count": 0,
              "is_new_account": False, "channel_url": "", "about_text": ""}
    try:
        channel_url = f"https://www.youtube.com/channel/{channel_id}/about"
        driver.get(channel_url)
        time.sleep(3)
        # Katılma tarihi
        try:
            join_el = driver.find_element(By.XPATH,
                "//*[contains(text(),'Joined') or contains(text(),'Katıldı')]")
            result["account_created"] = join_el.text.strip()
        except: pass
        # Abone sayısı
        try:
            sub_el = driver.find_element(By.XPATH,
                "//*[@id='subscriber-count' or contains(@class,'subscriber')]")
            result["subscriber_count"] = parse_subscriber_count(sub_el.text)
        except: pass
        # Hakkında
        try:
            about_el = driver.find_element(By.XPATH, "//*[@id='description-container']")
            result["about_text"] = about_el.text[:500]
        except: pass
        result["channel_url"] = channel_url
        # Yeni hesap tespiti (6 ay içinde açılmış)
        if result.get("account_created"):
            result["is_new_account"] = _is_new_account(result["account_created"])
        _account_cache[channel_id] = result
        time.sleep(1)
    except Exception as e:
        log.debug("Hesap inceleme hatası %s: %s", channel_id, e)
    return result

def parse_subscriber_count(text: str) -> int:
    text = text.replace(",","").replace(".","").strip().lower()
    match = re.search(r'([\d.]+)\s*([kmb]?)', text)
    if not match: return 0
    num = float(match.group(1))
    suffix = match.group(2)
    if suffix == "k": num *= 1000
    elif suffix == "m": num *= 1_000_000
    elif suffix == "b": num *= 1_000_000_000
    return int(num)

def _is_new_account(created_text: str, months_threshold: int = 6) -> bool:
    now = datetime.now()
    patterns = [
        r"(\w+ \d{1,2}, \d{4})",      # Jan 15, 2024
        r"(\d{1,2} \w+ \d{4})",        # 15 Jan 2024
        r"(\d{4}-\d{2}-\d{2})",        # 2024-01-15
    ]
    for pat in patterns:
        m = re.search(pat, created_text)
        if m:
            try:
                for fmt in ["%b %d, %Y","%d %b %Y","%B %d, %Y","%Y-%m-%d"]:
                    try:
                        dt = datetime.strptime(m.group(1), fmt)
                        diff_months = (now.year - dt.year)*12 + (now.month - dt.month)
                        return diff_months <= months_threshold
                    except: pass
            except: pass
    return False

def analyze_linked_accounts(users_info: List[dict]) -> List[Tuple[str,str,float]]:
    """Yeni hesap + birbiriyle ilişkili hesap tespiti"""
    pairs = []
    new_accounts = [u for u in users_info if u.get("is_new_account",False)]
    for u in new_accounts:
        log.info("🆕 Yeni hesap tespiti: %s (oluşturma: %s)",
                 u.get("author",""), u.get("account_created",""))
    # Birlikte yeni açılmış hesapları ilişkilendir
    for i in range(len(new_accounts)):
        for j in range(i+1, len(new_accounts)):
            ua = new_accounts[i]; ub = new_accounts[j]
            # Aynı dönemde açılmış → şüpheli
            pairs.append((ua.get("author",""), ub.get("author",""), 0.7))
    return pairs

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 18 — YORUM SİLME (Selenium — Moderatör İşlevi)
# ═══════════════════════════════════════════════════════════════════════════════
def delete_comment_selenium(driver, video_id: str, comment_text: str,
                             author: str, max_scroll: int = 30) -> bool:
    """Moderatör olarak yorumu sil"""
    if not driver: return False
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        driver.get(url); time.sleep(3)
        # Yorumlara kaydır
        for _ in range(5):
            driver.execute_script("window.scrollBy(0,500)")
            time.sleep(1)
        found = False
        for scroll_n in range(max_scroll):
            comments = driver.find_elements(By.CSS_SELECTOR,
                "ytd-comment-renderer, ytd-comment-thread-renderer")
            for cmt in comments:
                try:
                    author_el = cmt.find_element(By.CSS_SELECTOR, "#author-text")
                    msg_el    = cmt.find_element(By.CSS_SELECTOR, "#content-text")
                    if (author.lower() in author_el.text.lower() and
                            comment_text[:40].lower() in msg_el.text.lower()):
                        # 3 nokta menüsünü bul
                        menu_btn = cmt.find_element(By.CSS_SELECTOR,
                            "yt-icon-button.dropdown-trigger, #action-menu")
                        driver.execute_script("arguments[0].click()", menu_btn)
                        time.sleep(1)
                        # "Kaldır" / "Remove" seçeneği
                        for selector in ["[aria-label*='Remove']","[aria-label*='Kaldır']",
                                         ".yt-simple-endpoint[role='menuitem']"]:
                            try:
                                btns = driver.find_elements(By.CSS_SELECTOR, selector)
                                for btn in btns:
                                    txt = btn.text.lower()
                                    if "remove" in txt or "kaldır" in txt or "sil" in txt:
                                        btn.click(); time.sleep(1)
                                        # Onayla
                                        try:
                                            confirm = WebDriverWait(driver,5).until(
                                                EC.element_to_be_clickable((By.CSS_SELECTOR,
                                                    "yt-button-renderer[dialog-confirm] button")))
                                            confirm.click()
                                        except: pass
                                        # DB'de işaretle
                                        db_exec("UPDATE messages SET deleted=1 WHERE "
                                                "author=? AND message LIKE ?",
                                                (author, f"%{comment_text[:40]}%"))
                                        log.info("✅ Yorum silindi: @%s — %s",
                                                 author, comment_text[:50])
                                        found = True
                                        return True
                            except: pass
                except StaleElementReferenceException: pass
                except: pass
            if found: break
            driver.execute_script("window.scrollBy(0,1000)")
            time.sleep(1.5)
        log.warning("Yorum bulunamadı: @%s — %s", author, comment_text[:50])
        return False
    except Exception as e:
        log.error("Yorum silme hatası: %s", e)
        return False

def delete_live_chat_message_selenium(driver, video_id: str, author: str,
                                       message_text: str) -> bool:
    """Canlı yayın moderasyon: chat mesajı sil"""
    if not driver: return False
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        driver.get(url); time.sleep(3)
        chat_frames = driver.find_elements(By.TAG_NAME, "iframe")
        for frame in chat_frames:
            if "live_chat" in (frame.get_attribute("src") or ""):
                driver.switch_to.frame(frame); break
        time.sleep(2)
        items = driver.find_elements(By.CSS_SELECTOR,
            "yt-live-chat-text-message-renderer")
        for item in items:
            try:
                aut = item.find_element(By.ID,"author-name").text
                msg = item.find_element(By.ID,"message").text
                if aut.lower() == author.lower() and message_text[:20].lower() in msg.lower():
                    driver.execute_script("arguments[0].scrollIntoView()", item)
                    ActionChains(driver).context_click(item).perform()
                    time.sleep(0.5)
                    try:
                        remove_btn = WebDriverWait(driver,3).until(
                            EC.presence_of_element_located((By.XPATH,
                                "//*[contains(text(),'Remove') or contains(text(),'Kaldır')]")))
                        remove_btn.click()
                        log.info("✅ Canlı chat mesajı silindi: @%s",author)
                        driver.switch_to.default_content()
                        return True
                    except: pass
            except: pass
        driver.switch_to.default_content()
        return False
    except Exception as e:
        log.error("Canlı chat silme hatası: %s", e)
        try: driver.switch_to.default_content()
        except: pass
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 19 — ANA ANALİZ PİPELİNE
# ═══════════════════════════════════════════════════════════════════════════════
_pipeline_lock = threading.Lock()

def calculate_threat_level(profile: dict) -> dict:
    composite = (
        0.30 * float(profile.get("hate_score",0)) +
        0.25 * float(profile.get("bot_prob",0)) +
        0.20 * float(profile.get("stalker_score",0)) +
        0.15 * float(profile.get("impersonator_prob",0)) +
        0.10 * (1 - float(profile.get("human_score",0.5)))
    )
    composite = max(0.0, min(1.0, composite))
    if composite >= 0.85:   level = "CRIMSON"
    elif composite >= 0.70: level = "RED"
    elif composite >= 0.50: level = "ORANGE"
    elif composite >= 0.25: level = "YELLOW"
    else:                   level = "GREEN"
    return {"score": composite, "level": level, "color": COLOR_MAP[level]}

def analyze_user_full(author: str, run_ollama: bool = True) -> dict:
    """Kullanıcı için tam analiz çalıştır"""
    msgs_rows = get_user_messages(author)
    if not msgs_rows:
        return {"author": author, "error": "Mesaj bulunamadı", "threat_level": "GREEN"}

    messages   = [r["message"] for r in msgs_rows]
    timestamps = [int(r["timestamp"] or 0) for r in msgs_rows]

    # --- Katman 1-2: NLP & Stilometri ---
    all_text = " ".join(messages)
    tfidf_vec = get_tfidf_vector(all_text)
    ngram_fp  = ngram_fingerprint(all_text)
    typo_fp   = typo_fingerprint(messages)
    pos_fp    = pos_profile(all_text)
    emb       = embed_text(all_text[:2000])

    # --- Katman 3: Bot Tespiti ---
    bot_score = compute_bot_score(author, messages, timestamps)
    burst     = burstiness(timestamps)
    human_score = max(0.0, 1.0 - bot_score)

    # --- Katman 4: Nefret Söylemi ---
    sample_text = " ".join(messages[:10])[:1000]
    hate_result = compute_hate_scores(sample_text)
    hate_score  = hate_result["overall"]

    # --- Katman 7: Zamansal ---
    time_fp = temporal_fingerprint(timestamps)

    # --- Katman 3: Stalker ---
    # (Stalker skoru başka kullanıcılarla karşılaştırma gerektirir)
    # Hawkes ile kendi mesaj patlamasından hesapla
    stalker_score = max(0.0, min(1.0, abs(burst) * hate_score))

    # --- Katman 14: Bayes/Kalman ---
    kalman_score = kalman_update_user(author, hate_score * 0.5 + bot_score * 0.5)

    # --- Profil güncelle ---
    existing = db_exec("SELECT * FROM user_profiles WHERE author=?", (author,), fetch="one")
    threat_composite = calculate_threat_level({
        "hate_score": hate_score, "bot_prob": bot_score,
        "stalker_score": stalker_score, "impersonator_prob": 0.0,
        "human_score": human_score
    })

    profile_data = {
        "msg_count":        len(messages),
        "human_score":      round(human_score, 4),
        "bot_prob":         round(bot_score, 4),
        "hate_score":       round(hate_score, 4),
        "stalker_score":    round(stalker_score, 4),
        "tfidf_vector":     json.dumps(tfidf_vec.tolist()[:50]),
        "ngram_fingerprint":json.dumps(dict(list(ngram_fp.most_common(30)))),
        "typo_fingerprint": json.dumps(typo_fp),
        "pos_profile":      json.dumps(pos_fp),
        "identity_vector":  json.dumps(hate_result),
        "threat_level":     threat_composite["level"],
        "threat_score":     round(threat_composite["score"], 4),
        "first_seen":       min(timestamps) if timestamps else 0,
        "last_seen":        max(timestamps) if timestamps else 0,
    }

    # --- HMM durum dizisi ---
    threat_history = []
    if existing:
        prev = json.loads(existing["identity_vector"] or "{}") if existing["identity_vector"] else {}
        threat_history = [float(existing.get("threat_score",0))]
    threat_history.append(threat_composite["score"])
    hmm_states = hmm_state_sequence(threat_history)
    profile_data["hmm_state"] = hmm_states[-1] if hmm_states else "NORMAL"

    # --- Oyun Kuramı ---
    game_res = compute_game_score([], threat_composite["score"])
    profile_data["game_strategy"] = game_res.get("dominant_type","NORMAL")

    # --- Ollama (sadece yorum analizi için) ---
    if run_ollama and OLLAMA_OK and hate_score > 0.3:
        ollama_res = ollama_analyze_comments(author, messages, "threat_analysis")
        profile_data["ollama_summary"] = ollama_res.get("summary","")

    upsert_user_profile(author, profile_data)

    # ChromaDB'ye embedding kaydet
    if emb:
        chroma_upsert_msg(f"user_{author}", emb,
                          {"type":"user","author":author,
                           "threat":threat_composite["level"]})

    # Dataset güncelle (self-feeding)
    if threat_composite["score"] >= 0.85:
        label = "CRIMSON"
        db_exec("INSERT INTO dataset(msg_id,author,message,label,confirmed) VALUES(?,?,?,?,1)",
                (f"user_{author}", author, sample_text[:500], label))
    elif threat_composite["score"] >= 0.50:
        label = _infer_label(hate_result, bot_score)
        db_exec("INSERT INTO dataset(msg_id,author,message,label,confirmed) VALUES(?,?,?,?,0)",
                (f"user_{author}", author, sample_text[:500], label))

    return {
        "author": author, "msg_count": len(messages),
        "bot_prob": bot_score, "hate_score": hate_score,
        "stalker_score": stalker_score, "human_score": human_score,
        "threat_level": threat_composite["level"],
        "threat_score": threat_composite["score"],
        "threat_color": threat_composite["color"],
        "hmm_state": profile_data["hmm_state"],
        "game_strategy": profile_data["game_strategy"],
        "hate_breakdown": hate_result,
        "temporal": time_fp,
        "typo": typo_fp,
        "ollama_summary": profile_data.get("ollama_summary",""),
        "recommended_action": game_res.get("recommended_mod_action","MONITOR"),
    }

def _infer_label(hate_result: dict, bot_score: float) -> str:
    if hate_result.get("antisemitism",0) > 0.5: return "ANTISEMITE"
    if hate_result.get("groyper",0) > 0.5: return "GROYPER"
    if hate_result.get("hate_general",0) > 0.5: return "HATER"
    if bot_score > 0.7: return "BOT"
    if hate_result.get("stalker_sig",0) > 0.5: return "STALKER"
    return "SUSPICIOUS"

def build_global_similarity_matrix(users: List[str]) -> Tuple[List[str], np.ndarray]:
    """Tüm kullanıcılar için N×N benzerlik matrisi"""
    n = len(users)
    if n == 0: return [], np.zeros((0,0))
    # Embedding al
    user_texts = []
    user_ngrams = []
    user_times  = []
    for author in users:
        msgs_rows = get_user_messages(author)
        messages  = [r["message"] for r in msgs_rows]
        timestamps= [int(r["timestamp"] or 0) for r in msgs_rows]
        text = " ".join(messages[:50])
        user_texts.append(text or "empty")
        user_ngrams.append(ngram_fingerprint(text))
        user_times.append(temporal_fingerprint(timestamps))

    embeddings = embed_texts_batch(user_texts)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i][j] = 1.0; continue
            # Embedding sim
            emb_sim = 0.0
            if embeddings is not None:
                emb_sim = float(1 - cosine_dist(embeddings[i], embeddings[j]))
                emb_sim = max(0, emb_sim)
            # N-gram sim
            ng_sim = jaccard_ngram_sim(user_ngrams[i], user_ngrams[j])
            # Temporal sim
            tm_sim = time_similarity(user_times[i], user_times[j])
            composite = compute_composite_similarity(emb_sim, ng_sim, 0.0, tm_sim, 0.0)
            sim_matrix[i][j] = composite; sim_matrix[j][i] = composite
    return users, sim_matrix

def run_full_clustering(users: List[str] = None) -> dict:
    """Graf kümeleme, DBSCAN, Louvain, PageRank"""
    if users is None:
        rows = db_exec("SELECT author FROM user_profiles", fetch="all")
        users = [r["author"] for r in rows] if rows else []
    if len(users) < 3:
        return {"error":"Yeterli kullanıcı yok","clusters":{},"graph_data":{}}

    user_list, sim_matrix = build_global_similarity_matrix(users)
    # DBSCAN
    db_labels  = dbscan_cluster(sim_matrix)
    # Graf
    G = build_similarity_graph(user_list, sim_matrix)
    # Louvain
    louvain_partition = louvain_cluster(G)
    # PageRank
    pagerank = compute_pagerank(G)
    # Kimlik bağlantıları kaydet
    threshold = CONFIG.get("similarity_threshold", 0.65)
    n = len(user_list)
    for i in range(n):
        for j in range(i+1, n):
            sim = float(sim_matrix[i][j])
            if sim >= threshold:
                db_exec("INSERT OR IGNORE INTO identity_links(user_a,user_b,sim_score,method,confidence)"
                        " VALUES(?,?,?,?,?)",
                        (user_list[i],user_list[j],sim,"combined",sim))

    # Graf verisi (D3.js için)
    graph_data = {
        "nodes": [{"id": u, "group": int(louvain_partition.get(u,0)),
                   "pagerank": round(float(pagerank.get(u,0)),4)} for u in user_list],
        "links": [{"source": u, "target": v,
                   "value": round(float(G[u][v]["weight"]),3)}
                  for u,v in G.edges()]
    }
    # DB'ye küme kaydet
    clusters = {}
    for user, cid in louvain_partition.items():
        clusters.setdefault(cid, []).append(user)
    for cid, members in clusters.items():
        db_exec("INSERT INTO graph_clusters(cluster_id,members,algorithm) VALUES(?,?,?)",
                (cid, json.dumps(members), "louvain"))
    return {"clusters": clusters, "graph_data": graph_data,
            "dbscan_labels": dict(zip(user_list, db_labels.tolist())),
            "pagerank": {u: round(float(v),4) for u,v in pagerank.items()}}

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 20 — GERÇEK ZAMANLI CANLI YAYIN MONİTÖRÜ (Katman 14)
# ═══════════════════════════════════════════════════════════════════════════════
_live_monitor_active = False
_live_monitor_thread = None
_live_video_id = None
_socketio_instance = None

def start_live_monitor(video_id: str, driver, socketio):
    global _live_monitor_active, _live_monitor_thread, _live_video_id, _socketio_instance
    _live_video_id = video_id; _socketio_instance = socketio
    _live_monitor_active = True
    _live_monitor_thread = threading.Thread(target=_live_monitor_loop,
                                             args=(video_id,driver,socketio), daemon=True)
    _live_monitor_thread.start()
    log.info("⚡ Canlı yayın monitörü başlatıldı: %s", video_id)

def stop_live_monitor():
    global _live_monitor_active
    _live_monitor_active = False
    log.info("⚡ Canlı yayın monitörü durduruldu")

def _live_monitor_loop(video_id: str, driver, socketio):
    seen_ids = set()
    poll_interval = 5
    while _live_monitor_active:
        try:
            msgs = fetch_live_stream_chat_selenium(driver, f"https://www.youtube.com/watch?v={video_id}",
                                                    video_id, "")
            new_msgs = [m for m in msgs if m["msg_id"] not in seen_ids]
            for msg in new_msgs:
                seen_ids.add(msg["msg_id"])
                upsert_message(msg)
                # Hızlı analiz
                author = msg["author"]; text = msg["message"]
                hate = compute_hate_scores(text[:500])
                bot_score = heuristic_bot_score([text], [msg.get("timestamp_utc",0)])
                threat = calculate_threat_level({
                    "hate_score": hate["overall"], "bot_prob": bot_score,
                    "stalker_score": 0, "impersonator_prob": 0,
                    "human_score": max(0, 1-bot_score)
                })
                alert = {
                    "type": "live_message",
                    "author": author, "message": text[:200],
                    "threat_level": threat["level"],
                    "threat_score": round(threat["score"],3),
                    "threat_color": threat["color"],
                    "video_id": video_id,
                    "msg_id": msg["msg_id"],
                    "timestamp": int(time.time())
                }
                if socketio:
                    try: socketio.emit("live_alert", alert, namespace="/ws")
                    except: pass
                if threat["level"] in ("RED","CRIMSON","ORANGE"):
                    log.warning("🚨 CANLI UYARI [%s] @%s: %s",
                                threat["level"], author, text[:80])
        except Exception as e:
            log.debug("Canlı monitör hata: %s", e)
        time.sleep(poll_interval)

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 21 — SELF-FEEDING DATASET (Katman 22)
# ═══════════════════════════════════════════════════════════════════════════════
def check_retrain_needed() -> bool:
    row = db_exec("SELECT COUNT(*) as cnt FROM dataset WHERE confirmed=1", fetch="one")
    total = row["cnt"] if row else 0
    row2 = db_exec("SELECT MAX(trained_at) as last FROM training_log", fetch="one")
    last = row2["last"] if row2 and row2["last"] else 0
    new_since = db_exec("SELECT COUNT(*) as cnt FROM dataset WHERE confirmed=1 AND created_at > ?",
                        (last,), fetch="one")
    cnt = new_since["cnt"] if new_since else 0
    return cnt >= CONFIG.get("retrain_threshold", 500)

def approve_dataset_item(item_id: int, corrected_label: str = None):
    if corrected_label:
        db_exec("UPDATE dataset SET confirmed=1, label=? WHERE id=?",
                (corrected_label, item_id))
    else:
        db_exec("UPDATE dataset SET confirmed=1 WHERE id=?", (item_id,))

def retrain_naive_bayes() -> dict:
    """Onaylı dataset ile Naive Bayes modelini yeniden eğit"""
    rows = db_exec("SELECT message,label FROM dataset WHERE confirmed=1 LIMIT 5000", fetch="all")
    if not rows or len(rows) < 50:
        return {"error":"Yetersiz veri","count":len(rows) if rows else 0}
    texts  = [r["message"] for r in rows]
    labels = [r["label"]  for r in rows]
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(texts)
    nb = ComplementNB()
    nb.fit(X, labels)
    # Basit cross-val accuracy
    from sklearn.model_selection import cross_val_score
    try:
        scores = cross_val_score(nb, X, labels, cv=min(5,len(set(labels))), scoring="f1_macro")
        acc = float(np.mean(scores))
    except: acc = 0.0
    db_exec("INSERT INTO training_log(model_name,version,accuracy,f1_score,notes) VALUES(?,?,?,?,?)",
            ("naive_bayes",1,acc,acc,f"dataset_size={len(texts)}"))
    log.info("✅ Naive Bayes yeniden eğitildi. F1: %.3f", acc)
    return {"success":True,"f1":acc,"dataset_size":len(texts)}

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 22 — FLASK WEB PANELİ (Katman 15)
# ═══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YT Guardian v2.0 — @ShmirchikArt</title>
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--text:#c9d1d9;--text2:#8b949e;--accent:#58a6ff;--green:#2ECC71;--yellow:#F1C40F;--orange:#E67E22;--red:#E74C3C;--crimson:#8B0000;--blue:#3498DB;--purple:#9B59B6}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;font-size:13px}
#app{display:flex;height:100vh;overflow:hidden}
#sidebar{width:205px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:10px 0}
#sidebar h1{font-size:13px;font-weight:700;color:var(--accent);padding:10px 15px 15px;border-bottom:1px solid var(--border);line-height:1.4}
.nav-item{padding:9px 15px;cursor:pointer;display:flex;align-items:center;gap:8px;color:var(--text2);transition:.15s}
.nav-item:hover,.nav-item.active{background:var(--bg3);color:var(--text)}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#topbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:8px 15px;display:flex;align-items:center;gap:10px}
.input,.select{background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px}
.input{width:260px}.btn{background:var(--accent);color:#000;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600}
.btn:hover{opacity:.85}.btn-red{background:var(--red);color:#fff}.btn-green{background:var(--green);color:#000}.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text)}
#content{flex:1;overflow-y:auto;padding:15px}.card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:15px;margin-bottom:12px}.card h3{font-size:13px;font-weight:600;margin-bottom:10px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}.stat-box{background:var(--bg3);border-radius:6px;padding:12px;text-align:center}.stat-box .val{font-size:22px;font-weight:700}.stat-box .lbl{font-size:11px;color:var(--text2);margin-top:3px}
.user-table{width:100%;border-collapse:collapse}.user-table th{background:var(--bg3);padding:8px 10px;text-align:left;font-size:11px;color:var(--text2);border-bottom:1px solid var(--border)}.user-table td{padding:7px 10px;border-bottom:1px solid var(--border);font-size:12px;vertical-align:middle}
.threat-badge{padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;color:#000}.t-GREEN{background:#2ECC71}.t-YELLOW{background:#F1C40F}.t-ORANGE{background:#E67E22}.t-RED{background:#E74C3C;color:#fff}.t-CRIMSON{background:#8B0000;color:#fff}.t-BLUE{background:#3498DB}.t-PURPLE{background:#9B59B6;color:#fff}
.msg-item{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:10px 12px;margin-bottom:8px}.meta{font-size:11px;color:var(--text2);display:flex;gap:8px;align-items:center;margin-bottom:5px}.text{line-height:1.45}.pagination{display:flex;gap:6px;margin-top:10px}.pagination button{padding:4px 10px;background:var(--bg3);border:1px solid var(--border);color:var(--text);border-radius:4px;cursor:pointer}.pagination .active{background:var(--accent);color:#000}
#graph-container{width:100%;height:500px;background:var(--bg2);border-radius:8px;border:1px solid var(--border)}
.modal{display:flex;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:1000;align-items:center;justify-content:center}.modal-box{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:20px;width:600px;max-height:80vh;overflow-y:auto;position:relative}.modal-close{position:absolute;top:12px;right:15px;cursor:pointer}
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h1>🛡️ YT Guardian<br><span style="color:var(--text2);font-weight:400">@ShmirchikArt</span></h1>
    <div v-for="item in navItems" :key="item.key" class="nav-item" :class="{active: activeTab===item.key}" @click="changeTab(item.key)"><span>{{ item.icon }}</span>{{ item.label }}</div>
  </div>
  <div id="main">
    <div id="topbar">
      <input class="input" v-model="globalSearchQ" placeholder="🔍 Kullanıcı adı veya mesaj ara..." @input="runGlobalSearch">
      <select class="select" v-model="globalSearchMode"><option value="text">Metin</option><option value="user">Kullanıcı</option><option value="semantic">Semantik</option></select>
      <div style="margin-left:auto;display:flex;align-items:center;gap:8px"><span v-if="liveRunning" style="color:var(--green)">● Canlı</span><span style="color:var(--text2)">{{ statusMsg }}</span><button class="btn" @click="startScrape">▶ Tara</button></div>
    </div>
    <div id="content">
      <template v-if="activeTab==='dashboard'">
        <div class="stat-grid"><div class="stat-box" v-for="s in statCards" :key="s.key"><div class="val" :style="{color:s.color||'var(--accent)'}">{{ dashboard[s.key] ?? 0 }}</div><div class="lbl">{{ s.label }}</div></div></div>
        <div class="card" style="margin-top:12px"><h3>Son Uyarılar <button class="btn btn-outline" style="float:right;font-size:10px" @click="alerts=[]">Temizle</button></h3><div v-if="alerts.length===0" style="color:var(--text2)">Uyarı yok</div><div v-for="a in alerts" :key="a.id" class="msg-item"><div class="meta"><span class="threat-badge" :class="'t-'+(a.threat_level||'GREEN')">{{ a.threat_level }}</span><strong>@{{ a.author }}</strong><span>{{ fmtTs(a.timestamp) }}</span></div><div class="text">{{ a.message }}</div></div></div>
        <div class="card"><h3>Tehdit Dağılımı</h3><canvas id="threat-chart" height="90"></canvas></div>
      </template>

      <template v-else-if="activeTab==='users'">
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px"><input class="input" v-model="userFilter" placeholder="Kullanıcı ara..." style="width:200px"><select class="select" v-model="threatFilter"><option value="">Tüm Seviyeler</option><option v-for="lvl in ['CRIMSON','RED','ORANGE','YELLOW','GREEN']" :key="lvl" :value="lvl">{{ lvl }}</option></select><button class="btn" @click="analyzeAllUsers">⚡ Tüm Analiz</button><button class="btn" @click="runClustering">🕸️ Kümeleme</button><span style="margin-left:auto;color:var(--text2)">{{ filteredUsers.length }} kullanıcı</span></div>
        <table class="user-table"><thead><tr><th>Kullanıcı</th><th>Mesaj</th><th>Tehdit</th><th>Bot%</th><th>Nefret%</th><th>Stalker%</th><th>HMM</th><th>Durum</th></tr></thead><tbody><tr v-for="u in pagedUsers" :key="u.author"><td><a href="#" @click.prevent="showUser(u.author)" style="color:var(--accent)">@{{ u.author }}</a></td><td>{{ u.msg_count||0 }}</td><td><span class="threat-badge" :class="'t-'+(u.threat_level||'GREEN')">{{ u.threat_level||'GREEN' }}</span></td><td>{{ pct(u.bot_prob) }}</td><td>{{ pct(u.hate_prob) }}</td><td>{{ pct(u.stalker_score) }}</td><td>{{ u.hmm_state || '-' }}</td><td>{{ u.status || '-' }}</td></tr></tbody></table>
        <div class="pagination"><button v-for="p in totalUserPages" :key="p" :class="{active:userPage===p}" @click="userPage=p">{{ p }}</button></div>
      </template>

      <template v-else-if="activeTab==='messages'">
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px"><input class="input" v-model="msgQuery" placeholder="Mesaj ara" style="flex:1;min-width:220px" @input="loadMessages(1)"><input class="input" v-model="msgAuthor" placeholder="Kullanıcı" style="width:180px" @input="loadMessages(1)"><select class="select" v-model="msgSource" @change="loadMessages(1)"><option value="">Tüm Kaynaklar</option><option value="comment">Yorum</option><option value="replay_chat">Replay Chat</option><option value="live">Canlı</option></select><button class="btn btn-outline" @click="loadDeleteCandidates">🧪 Silme Adayları</button></div>
        <div v-if="deleteCandidates.length" class="card"><h3>Silme Adayları</h3><div v-for="c in deleteCandidates" :key="c.id||c.message" class="msg-item"><div class="meta"><span class="threat-badge" :class="'t-'+(c.threat_level||'RED')">{{ c.threat_level||'RED' }}</span><span>@{{ c.author }}</span></div><div class="text">{{ c.message }}</div></div></div>
        <div v-for="m in messages" :key="m.id||m.timestamp+m.author" class="msg-item"><div class="meta"><span style="color:var(--accent)">@{{ m.author }}</span><span>{{ m.video_id }}</span><span class="threat-badge" :class="'t-'+(m.threat_level||'GREEN')">{{ m.threat_level||'GREEN' }}</span><span>{{ fmtTs(m.timestamp) }}</span></div><div class="text">{{ m.message }}</div></div>
        <div class="pagination"><button v-for="p in totalMsgPages" :key="p" :class="{active:msgPage===p}" @click="loadMessages(p)">{{ p }}</button></div>
      </template>

      <template v-else-if="activeTab==='graph'"><div style="display:flex;gap:8px;margin-bottom:10px"><button class="btn" @click="loadGraph">🔄 Grafiği Yükle</button><button class="btn btn-outline" @click="runClustering">⚙️ Kümeleri Yenile</button></div><div id="graph-container"></div></template>

      <template v-else-if="activeTab==='live'"><div class="card"><h3>Canlı Yayın Monitörü</h3><div style="display:flex;gap:8px"><input class="input" v-model="liveVideoId" placeholder="Video ID (11 karakter)"><button class="btn btn-green" @click="startLive">▶ Başlat</button><button class="btn btn-red" @click="stopLive">⏹ Durdur</button></div><div style="margin-top:10px"><div v-for="a in liveMessages" :key="a.timestamp+a.author" class="msg-item"><div class="meta"><span class="threat-badge" :class="'t-'+(a.threat_level||'GREEN')">{{ a.threat_level }}</span><span>@{{ a.author }}</span></div><div class="text">{{ a.message }}</div></div></div></div></template>

      <template v-else-if="activeTab==='search'"><div class="card"><h3>Gelişmiş Arama</h3><div style="display:flex;gap:8px;flex-wrap:wrap"><input class="input" v-model="advQuery" style="flex:1" placeholder="Arama terimi..."><select class="select" v-model="advMode"><option value="text">Tam Metin (FTS5)</option><option value="user">Kullanıcı</option><option value="semantic">Semantik Benzerlik</option><option value="pattern">N-gram Pattern</option></select><button class="btn" @click="advancedSearch">🔍 Ara</button></div><div style="margin-top:10px"><div v-for="u in searchResults.users||[]" :key="u.author" class="msg-item"><div class="meta"><a href="#" @click.prevent="showUser(u.author)" style="color:var(--accent)">@{{ u.author }}</a><span class="threat-badge" :class="'t-'+(u.threat_level||'GREEN')">{{ u.threat_level }}</span></div></div><div v-for="m in searchResults.messages||[]" :key="m.author+m.timestamp" class="msg-item"><div class="meta"><span>@{{ m.author }}</span></div><div class="text">{{ m.message }}</div></div></div></div></template>

      <template v-else-if="activeTab==='stats'"><div class="card"><h3>Zaman Serisi — Tehdit Skorları</h3><canvas id="timeline-chart" height="120"></canvas></div><div class="card"><h3>Kimlik Eşleşmeleri</h3><div v-for="l in identityLinks" :key="l.user_a+l.user_b" class="msg-item"><div class="meta"><span>{{ l.user_a }}</span><span>⟷</span><span>{{ l.user_b }}</span><span>{{ pct(l.sim_score) }}</span></div></div></div><div class="card"><h3>Oyun Kuramı — Nash Dengesi</h3><div v-for="e in equilibria" :key="e.join('-')" class="msg-item"><div class="text">{{ e[2] }} / {{ e[3] }} — Payoff: {{ e[0] }} / {{ e[1] }}</div></div></div></template>

      <template v-else-if="activeTab==='dataset'"><div style="display:flex;gap:8px;margin-bottom:10px"><button class="btn" @click="loadPendingDataset">⏳ Onay Bekleyenleri Yükle</button><button class="btn btn-red" @click="retrainModel">🔄 Modeli Yeniden Eğit</button></div><div v-for="item in datasetItems" :key="item.id" class="msg-item"><div class="meta"><strong>@{{ item.author }}</strong><span class="threat-badge" :class="'t-'+(item.label||'GREEN')">{{ item.label }}</span></div><div class="text">{{ item.message }}</div><div style="margin-top:8px;display:flex;gap:5px"><button class="btn btn-green" @click="approveDataset(item.id,null)">✓</button><select class="select" v-model="item._label" style="padding:3px 6px"><option>ANTISEMITE</option><option>GROYPER</option><option>HATER</option><option>BOT</option><option>STALKER</option><option>NORMAL</option></select><button class="btn btn-outline" @click="approveDataset(item.id,item._label)">✓ Etiketle</button></div></div></template>

      <template v-else-if="activeTab==='settings'"><div class="card"><h3>Sistem Durumu</h3><div v-for="(v,k) in systemStatus" :key="k" class="msg-item"><div class="meta"><span>{{ k }}</span><span>{{ v }}</span></div></div></div><div class="card"><h3>YouTube Giriş</h3><div style="display:flex;flex-direction:column;gap:8px;max-width:360px"><input class="input" v-model="ytEmail" placeholder="E-posta" style="width:100%"><input class="input" v-model="ytPass" type="password" placeholder="Şifre" style="width:100%"><button class="btn" @click="ytLogin">🔑 Giriş Yap</button><span :style="{color:loginOk?'var(--green)':'var(--red)'}">{{ loginStatus }}</span></div></div></template>
    </div>
  </div>

  <div v-if="modalOpen" class="modal" @click.self="modalOpen=false"><div class="modal-box"><span class="modal-close" @click="modalOpen=false">✕</span><h2>@{{ modalUser?.author || modalUser?.name || 'Kullanıcı' }}</h2><pre style="white-space:pre-wrap">{{ JSON.stringify(modalUser, null, 2) }}</pre></div></div>
</div>

<script>
const { createApp } = Vue;
createApp({
  data(){return {navItems:[{key:'dashboard',label:'Dashboard',icon:'📊'},{key:'users',label:'Kullanıcılar',icon:'👥'},{key:'messages',label:'Mesajlar',icon:'💬'},{key:'graph',label:'İlişki Ağı',icon:'🔗'},{key:'live',label:'Canlı Yayın',icon:'⚡'},{key:'search',label:'Arama',icon:'🔍'},{key:'stats',label:'İstatistikler',icon:'📈'},{key:'dataset',label:'Dataset',icon:'🗃️'},{key:'settings',label:'Ayarlar',icon:'⚙️'}],activeTab:'dashboard',statusMsg:'Hazır',globalSearchQ:'',globalSearchMode:'text',dashboard:{},alerts:[],statCards:[{key:'total_messages',label:'Toplam Mesaj'},{key:'total_users',label:'Kullanıcı'},{key:'crimson',label:'CRIMSON',color:'var(--crimson)'},{key:'red',label:'RED/HATER',color:'var(--red)'},{key:'orange',label:'ORANGE',color:'var(--orange)'},{key:'bots',label:'BOT',color:'var(--blue)'},{key:'stalkers',label:'STALKER',color:'var(--purple)'},{key:'videos',label:'Video'}],allUsers:[],userFilter:'',threatFilter:'',userPage:1,pageSize:25,messages:[],msgPage:1,totalMsgs:0,msgQuery:'',msgAuthor:'',msgSource:'',deleteCandidates:[],graphData:null,liveVideoId:'',liveMessages:[],liveRunning:false,advQuery:'',advMode:'text',searchResults:{},identityLinks:[],equilibria:[],datasetItems:[],systemStatus:{},ytEmail:'',ytPass:'',loginStatus:'',loginOk:false,modalOpen:false,modalUser:null,socket:null,threatChart:null,timelineChart:null};},
  computed:{filteredUsers(){return this.allUsers.filter(u=>{const f=this.userFilter.toLowerCase();const ok=!f||String(u.author||'').toLowerCase().includes(f);const t=!this.threatFilter||u.threat_level===this.threatFilter;return ok&&t;});},totalUserPages(){return Math.max(1,Math.ceil(this.filteredUsers.length/this.pageSize));},pagedUsers(){const s=(this.userPage-1)*this.pageSize;return this.filteredUsers.slice(s,s+this.pageSize);},totalMsgPages(){return Math.max(1,Math.ceil(this.totalMsgs/this.pageSize));}},
  mounted(){this.socket=io('/ws');this.bindSocket();this.loadDashboard();},
  methods:{
    async api(path,params={},method='GET'){const opts={method,headers:{'Content-Type':'application/json'}};let url=path;if(method==='GET'){url += '?' + new URLSearchParams(params).toString();} else {opts.body=JSON.stringify(params);} const res=await fetch(url,opts); if(!res.ok) throw new Error((await res.json()).error || 'API hatası'); return res.json();},
    pct(v){return `${Math.round((v||0)*100)}%`;}, fmtTs(ts){if(!ts) return ''; return new Date(ts*1000).toLocaleString();},
    async changeTab(tab){this.activeTab=tab; if(tab==='dashboard') await this.loadDashboard(); if(tab==='users') await this.loadUsers(); if(tab==='messages') await this.loadMessages(1); if(tab==='graph') await this.loadGraph(); if(tab==='stats') await this.loadStats(); if(tab==='settings') await this.loadStatus(); if(tab==='dataset') await this.loadPendingDataset();},
    async loadDashboard(){const d=await this.api('/api/stats'); this.dashboard=d; this.renderThreatChart();},
    renderThreatChart(){const c=document.getElementById('threat-chart'); if(!c) return; if(this.threatChart) this.threatChart.destroy(); this.threatChart=new Chart(c,{type:'bar',data:{labels:['CRIMSON','RED','ORANGE','YELLOW','GREEN'],datasets:[{data:[this.dashboard.crimson||0,this.dashboard.red||0,this.dashboard.orange||0,this.dashboard.yellow||0,this.dashboard.green||0],backgroundColor:['#8B0000','#E74C3C','#E67E22','#F1C40F','#2ECC71']}]},options:{plugins:{legend:{display:false}}}});},
    async loadUsers(){const d=await this.api('/api/users',{page:1,size:10000}); this.allUsers=d.users||[];},
    async showUser(author){const d=await this.api('/api/user/'+encodeURIComponent(author)); this.modalUser=d; this.modalOpen=true;},
    async analyzeAllUsers(){await this.api('/api/analyze/all',{},'POST'); this.statusMsg='✅ Tüm analiz tetiklendi';},
    async runClustering(){await this.api('/api/cluster',{},'POST'); this.statusMsg='✅ Kümeleme tamamlandı'; if(this.activeTab==='graph') this.loadGraph();},
    async loadMessages(page){this.msgPage=page; const d=await this.api('/api/messages',{page:this.msgPage,size:this.pageSize,q:this.msgQuery,author:this.msgAuthor,source:this.msgSource}); this.messages=d.messages||[]; this.totalMsgs=d.total||0;},
    async loadDeleteCandidates(){const d=await this.api('/api/review/delete-candidates',{limit:200}); this.deleteCandidates=d.candidates||[];},
    async loadGraph(){const d=await this.api('/api/graph'); this.graphData=d.graph_data; this.renderGraph();},
    renderGraph(){const data=this.graphData; const container=document.getElementById('graph-container'); if(!container) return; container.innerHTML=''; if(!data||!data.nodes?.length){container.innerHTML='<p style="padding:15px;color:var(--text2)">Veri yok.</p>'; return;} const W=container.clientWidth||800,H=500; const svg=d3.select(container).append('svg').attr('width',W).attr('height',H); const g=svg.append('g'); const sim=d3.forceSimulation(data.nodes).force('link',d3.forceLink(data.links).id(d=>d.id).distance(85)).force('charge',d3.forceManyBody().strength(-180)).force('center',d3.forceCenter(W/2,H/2)); const link=g.selectAll('line').data(data.links).enter().append('line').attr('stroke','#444'); const node=g.selectAll('circle').data(data.nodes).enter().append('circle').attr('r',6).attr('fill','#58a6ff').on('click',(_,d)=>this.showUser(d.id)); const label=g.selectAll('text').data(data.nodes).enter().append('text').text(d=>d.id).attr('font-size',9).attr('fill','#8b949e').attr('dy',15); sim.on('tick',()=>{link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y); node.attr('cx',d=>d.x).attr('cy',d=>d.y); label.attr('x',d=>d.x).attr('y',d=>d.y);});},
    async startLive(){if((this.liveVideoId||'').length!==11){alert('Geçerli bir Video ID girin'); return;} await this.api('/api/live/start',{video_id:this.liveVideoId},'POST'); this.liveRunning=true; this.statusMsg='⚡ Canlı monitör başlatıldı';},
    async stopLive(){await this.api('/api/live/stop',{},'POST'); this.liveRunning=false; this.statusMsg='Canlı monitör durduruldu';},
    async runGlobalSearch(){if((this.globalSearchQ||'').length<2) return; this.searchResults=await this.api('/api/search',{q:this.globalSearchQ,mode:this.globalSearchMode});},
    async advancedSearch(){if(!this.advQuery) return; this.searchResults=await this.api('/api/search',{q:this.advQuery,mode:this.advMode});},
    async loadStats(){const links=await this.api('/api/identity-links'); this.identityLinks=(links.links||[]).slice(0,50); const nash=await this.api('/api/nash'); this.equilibria=nash.equilibria||[]; this.renderTimelineChart();},
    renderTimelineChart(){const c=document.getElementById('timeline-chart'); if(!c) return; if(this.timelineChart) this.timelineChart.destroy(); this.timelineChart=new Chart(c,{type:'line',data:{labels:this.identityLinks.map((_,i)=>String(i+1)),datasets:[{label:'Benzerlik',data:this.identityLinks.map(i=>i.sim_score||0),borderColor:'#58a6ff'}]},options:{plugins:{legend:{display:false}}}});},
    async loadPendingDataset(){const d=await this.api('/api/dataset/pending'); this.datasetItems=(d.items||[]).map(x=>({...x,_label:'NORMAL'}));},
    async approveDataset(id,label){await this.api('/api/dataset/approve',label?{id,label}:{id},'POST'); this.statusMsg='✅ Dataset onayı tamamlandı'; this.loadPendingDataset();},
    async retrainModel(){const d=await this.api('/api/retrain',{},'POST'); this.statusMsg=d.success?`✅ Eğitim tamamlandı — F1: ${d.f1}`:`❌ ${d.error}`;},
    async loadStatus(){this.systemStatus=await this.api('/api/status');},
    async ytLogin(){const d=await this.api('/api/yt/login',{email:this.ytEmail,password:this.ytPass},'POST'); this.loginOk=!!d.success; this.loginStatus=d.message || (d.success?'✅ Giriş başarılı':'❌ Giriş başarısız');},
    async startScrape(){await this.api('/api/scrape',{},'POST'); this.statusMsg='🧹 Tarama arka planda çalışıyor...';},
    bindSocket(){this.socket.on('connected',()=>this.statusMsg='WebSocket bağlı'); this.socket.on('live_alert',(data)=>{this.liveMessages.unshift(data); this.alerts.unshift({...data,id:crypto.randomUUID()}); this.alerts=this.alerts.slice(0,50);}); this.socket.on('scrape_progress',(d)=>this.statusMsg=`Tarama sürüyor — +${d.new_messages||0} mesaj`); this.socket.on('scrape_done',(d)=>{this.statusMsg=`✅ Tarama tamamlandı — ${d.total_messages||0} mesaj`; this.loadDashboard();}); this.socket.on('login_result',(d)=>{this.loginOk=!!d.success; this.loginStatus=this.loginOk?'✅ Giriş başarılı':'❌ Giriş başarısız';});}
  }
}).mount('#app');
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 23 — FLASK API ROTALARı
# ═══════════════════════════════════════════════════════════════════════════════
def create_app():
    if not FLASK_OK:
        raise RuntimeError("Flask yüklü değil: pip install flask flask-socketio flask-cors")
    app = Flask(__name__)
    app.config["SECRET_KEY"] = CONFIG.get("flask_secret","secret")
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet",
                         namespace="/ws")
    global _socketio_instance; _socketio_instance = socketio

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    # ── Dashboard istatistikler ──────────────────────────────────────────────
    @app.route("/api/stats")
    def api_stats():
        try:
            total_messages = (db_exec("SELECT COUNT(*) as c FROM messages WHERE deleted=0",fetch="one") or {}).get("c",0)
            total_users    = (db_exec("SELECT COUNT(*) as c FROM user_profiles",fetch="one") or {}).get("c",0)
            videos         = (db_exec("SELECT COUNT(*) as c FROM scraped_videos",fetch="one") or {}).get("c",0)
            levels = db_exec("SELECT threat_level, COUNT(*) as c FROM user_profiles GROUP BY threat_level",fetch="all") or []
            lmap = {r["threat_level"]: r["c"] for r in levels}
            bots    = (db_exec("SELECT COUNT(*) as c FROM user_profiles WHERE bot_prob>?", (CONFIG["bot_threshold"],),fetch="one") or {}).get("c",0)
            stalkers= (db_exec("SELECT COUNT(*) as c FROM user_profiles WHERE stalker_score>?", (CONFIG["stalker_threshold"],),fetch="one") or {}).get("c",0)
            return jsonify({"total_messages":total_messages,"total_users":total_users,
                            "videos":videos,"bots":bots,"stalkers":stalkers,
                            "crimson":lmap.get("CRIMSON",0),"red":lmap.get("RED",0),
                            "orange":lmap.get("ORANGE",0),"yellow":lmap.get("YELLOW",0),
                            "green":lmap.get("GREEN",0)})
        except Exception as e:
            return jsonify({"error":str(e)})

    # ── Kullanıcılar ─────────────────────────────────────────────────────────
    @app.route("/api/users")
    def api_users():
        page  = int(request.args.get("page",1))
        size  = int(request.args.get("size",50))
        flt   = request.args.get("filter","")
        threat= request.args.get("threat","")
        offset= (page-1)*size
        where = "WHERE 1=1"
        params = []
        if flt:    where += " AND author LIKE ?"; params.append(f"%{flt}%")
        if threat: where += " AND threat_level=?"; params.append(threat)
        total = (db_exec(f"SELECT COUNT(*) as c FROM user_profiles {where}",
                         tuple(params),fetch="one") or {}).get("c",0)
        rows  = db_exec(f"SELECT * FROM user_profiles {where} ORDER BY threat_score DESC LIMIT ? OFFSET ?",
                        tuple(params)+( size, offset),fetch="all") or []
        return jsonify({"users":[dict(r) for r in rows],"total":total})

    @app.route("/api/user/<author>")
    def api_user_detail(author):
        row = db_exec("SELECT * FROM user_profiles WHERE author=?", (author,), fetch="one")
        if not row: return jsonify({"error":"Kullanıcı bulunamadı"})
        d = dict(row)
        # JSON alanları parse et
        for field in ["identity_vector","tfidf_vector","ngram_fingerprint","typo_fingerprint","pos_profile"]:
            if d.get(field):
                try: d[field] = json.loads(d[field])
                except: pass
        links = db_exec("SELECT * FROM identity_links WHERE user_a=? OR user_b=? LIMIT 20",
                        (author,author),fetch="all") or []
        d["identity_links"] = [dict(r) for r in links]
        if d.get("identity_vector") and isinstance(d["identity_vector"],dict):
            d["hate_breakdown"] = d["identity_vector"]
        return jsonify(d)

    @app.route("/api/user/<author>/messages")
    def api_user_messages(author):
        rows = db_exec("SELECT * FROM messages WHERE author=? AND deleted=0 ORDER BY timestamp DESC LIMIT 200",
                       (author,),fetch="all") or []
        return jsonify({"messages":[dict(r) for r in rows]})

    @app.route("/api/user/<author>/ban", methods=["POST"])
    def api_ban_user(author):
        db_exec("UPDATE user_profiles SET game_strategy='BAN' WHERE author=?", (author,))
        return jsonify({"success":True,"message":f"@{author} ban işareti eklendi"})

    # ── Mesajlar ──────────────────────────────────────────────────────────────
    @app.route("/api/messages")
    def api_messages():
        page   = int(request.args.get("page",1))
        size   = int(request.args.get("size",50))
        q      = request.args.get("q","")
        author = request.args.get("author","")
        source = request.args.get("source","")
        offset = (page-1)*size
        joins  = "LEFT JOIN user_profiles up ON m.author=up.author"
        where  = "WHERE m.deleted=0"
        params = []
        if q:
            # FTS5 tam metin arama
            try:
                fts_rows = db_exec(
                    "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? LIMIT 1000",
                    (q,),fetch="all") or []
                rowids = tuple(r["rowid"] for r in fts_rows)
                if rowids:
                    where += f" AND m.rowid IN ({','.join(['?']*len(rowids))})"
                    params.extend(rowids)
                else:
                    where += " AND m.message LIKE ?"; params.append(f"%{q}%")
            except:
                where += " AND m.message LIKE ?"; params.append(f"%{q}%")
        if author: where += " AND m.author LIKE ?"; params.append(f"%{author}%")
        if source: where += " AND m.source_type=?"; params.append(source)
        total = (db_exec(f"SELECT COUNT(*) as c FROM messages m {joins} {where}",
                         tuple(params),fetch="one") or {}).get("c",0)
        rows  = db_exec(
            f"SELECT m.*, up.threat_level, up.threat_score FROM messages m {joins} {where} "
            f"ORDER BY m.timestamp DESC LIMIT ? OFFSET ?",
            tuple(params)+(size,offset),fetch="all") or []
        return jsonify({"messages":[dict(r) for r in rows],"total":total})

    # ── Analiz ────────────────────────────────────────────────────────────────
    @app.route("/api/analyze/user", methods=["POST"])
    def api_analyze_user():
        author = request.form.get("author","")
        if not author: return jsonify({"error":"author gerekli"})
        result = analyze_user_full(author, run_ollama=True)
        return jsonify(result)

    @app.route("/api/analyze/all", methods=["POST"])
    def api_analyze_all():
        rows = db_exec("SELECT DISTINCT author FROM messages WHERE deleted=0",fetch="all") or []
        authors = [r["author"] for r in rows]
        count = 0
        for a in authors:
            try:
                analyze_user_full(a, run_ollama=False)
                count += 1
            except: pass
        return jsonify({"analyzed":count})

    @app.route("/api/analyze/message", methods=["POST"])
    def api_analyze_message():
        text   = request.form.get("message","")
        author = request.form.get("author","unknown")
        if not text: return jsonify({"error":"message gerekli"})
        hate = compute_hate_scores(text)
        bot  = heuristic_bot_score([text],[0])
        threat = calculate_threat_level({"hate_score":hate["overall"],"bot_prob":bot,
                                          "stalker_score":0,"impersonator_prob":0,"human_score":1-bot})
        return jsonify({"hate":hate,"bot_prob":bot,"threat":threat})

    # ── Kümeleme ──────────────────────────────────────────────────────────────
    @app.route("/api/cluster", methods=["POST"])
    def api_cluster():
        try:
            result = run_full_clustering()
            return jsonify(result)
        except Exception as e:
            return jsonify({"error":str(e)})

    @app.route("/api/graph")
    def api_graph():
        result = run_full_clustering()
        return jsonify(result)

    @app.route("/api/clusters")
    def api_clusters():
        rows = db_exec("SELECT * FROM graph_clusters ORDER BY created_at DESC LIMIT 1",fetch="one")
        if not rows: return jsonify({"clusters":{}})
        members = json.loads(rows["members"] or "[]")
        return jsonify({"cluster_id":rows["cluster_id"],"members":members})

    # ── Arama ─────────────────────────────────────────────────────────────────
    @app.route("/api/search")
    def api_search():
        q    = request.args.get("q","")
        mode = request.args.get("mode","text")
        if not q: return jsonify({"messages":[],"users":[]})
        users_out = []; msgs_out = []
        if mode in ("text","user"):
            # Kullanıcı ara
            urows = db_exec("SELECT * FROM user_profiles WHERE author LIKE ? LIMIT 20",
                            (f"%{q}%",),fetch="all") or []
            users_out = [dict(r) for r in urows]
        if mode in ("text","semantic"):
            # FTS5
            try:
                fts = db_exec("SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? LIMIT 200",
                              (q,),fetch="all") or []
                if fts:
                    rowids = tuple(r["rowid"] for r in fts)
                    mrows = db_exec(f"SELECT * FROM messages WHERE rowid IN ({','.join(['?']*len(rowids))}) AND deleted=0 LIMIT 100",
                                    rowids,fetch="all") or []
                    msgs_out = [dict(r) for r in mrows]
            except:
                mrows = db_exec("SELECT * FROM messages WHERE message LIKE ? AND deleted=0 LIMIT 100",
                                (f"%{q}%",),fetch="all") or []
                msgs_out = [dict(r) for r in mrows]
        if mode == "semantic" and SBERT_OK:
            emb = embed_text(q)
            if emb:
                chroma_results = chroma_search_similar(emb, n=20)
                for cr in chroma_results:
                    msg_id = cr.get("id","")
                    if not msg_id.startswith("user_"):
                        row = db_exec("SELECT * FROM messages WHERE id=? AND deleted=0",(msg_id,),fetch="one")
                        if row: msgs_out.append(dict(row))
        return jsonify({"messages":msgs_out[:100],"users":users_out})

    # ── Kimlik bağlantıları ───────────────────────────────────────────────────
    @app.route("/api/identity-links")
    def api_identity_links():
        rows = db_exec("SELECT * FROM identity_links ORDER BY sim_score DESC LIMIT 200",fetch="all") or []
        return jsonify({"links":[dict(r) for r in rows]})

    # ── Silme aday kuyruğu (otomatik silme yok) ──────────────────────────────
    @app.route("/api/review/delete-candidates")
    def api_review_delete_candidates():
        limit = int(request.args.get("limit", "300") or 300)
        limit = max(1, min(limit, 2000))
        candidates = build_deletion_candidates(limit)
        return jsonify({
            "success": True,
            "count": len(candidates),
            "allow_destructive_actions": destructive_actions_enabled(),
            "candidates": candidates
        })

    # ── Yorum silme ───────────────────────────────────────────────────────────
    @app.route("/api/delete/comment", methods=["POST"])
    def api_delete_comment():
        if not destructive_actions_enabled():
            return jsonify({
                "success": False,
                "error": "Güvenlik nedeniyle silme kapalı. allow_destructive_actions=true yapın."
            }), 403
        video_id  = request.form.get("video_id","")
        author    = request.form.get("author","")
        message   = request.form.get("message","")
        if not all([video_id, author]):
            return jsonify({"success":False,"error":"video_id ve author gerekli"})
        driver = _selenium_driver
        if not driver:
            return jsonify({"success":False,"error":"Selenium bağlantısı yok. Önce giriş yapın."})
        def _delete_bg():
            ok = delete_comment_selenium(driver, video_id, message, author)
            if _socketio_instance:
                _socketio_instance.emit("delete_result",
                    {"success":ok,"author":author,"video_id":video_id}, namespace="/ws")
        threading.Thread(target=_delete_bg, daemon=True).start()
        return jsonify({"success":True,"message":"Silme işlemi arka planda başlatıldı"})

    @app.route("/api/delete/live", methods=["POST"])
    def api_delete_live():
        if not destructive_actions_enabled():
            return jsonify({
                "success": False,
                "error": "Güvenlik nedeniyle canlı chat silme kapalı. allow_destructive_actions=true yapın."
            }), 403
        video_id = request.form.get("video_id","")
        author   = request.form.get("author","")
        message  = request.form.get("message","")
        driver = _selenium_driver
        if not driver:
            return jsonify({"success":False,"error":"Selenium bağlantısı yok."})
        ok = delete_live_chat_message_selenium(driver, video_id, author, message)
        return jsonify({"success":ok})

    # ── YouTube giriş ──────────────────────────────────────────────────────────
    @app.route("/api/yt/login", methods=["POST"])
    def api_yt_login():
        global _selenium_driver
        email = request.form.get("email", CONFIG.get("yt_email",""))
        password = request.form.get("password", CONFIG.get("yt_password",""))
        if CONFIG.get("require_env_credentials", True):
            env_email = os.environ.get("YT_EMAIL", "")
            env_password = os.environ.get("YT_PASSWORD", "")
            if not env_email or not env_password:
                return jsonify({
                    "success": False,
                    "message": "YT_EMAIL ve YT_PASSWORD ortam değişkenleri gerekli (require_env_credentials=true)."
                }), 400
            email, password = env_email, env_password
        if not email or not password:
            return jsonify({"success":False,"message":"Email ve şifre gerekli"})
        def _login_bg():
            global _selenium_driver
            if _selenium_driver:
                try: _selenium_driver.quit()
                except: pass
            _selenium_driver = create_firefox_driver(headless=False)
            ok = youtube_login(_selenium_driver, email, password)
            if _socketio_instance:
                _socketio_instance.emit("login_result",
                    {"success":ok,"email":email}, namespace="/ws")
        threading.Thread(target=_login_bg, daemon=True).start()
        return jsonify({"success":True,"message":"Giriş arka planda başlatıldı"})

    # ── Canlı monitör ─────────────────────────────────────────────────────────
    @app.route("/api/live/start", methods=["POST"])
    def api_live_start():
        video_id = request.form.get("video_id","")
        if not video_id: return jsonify({"success":False,"error":"video_id gerekli"})
        driver = _selenium_driver
        if not driver:
            return jsonify({"success":False,"error":"Selenium bağlantısı yok"})
        start_live_monitor(video_id, driver, socketio)
        return jsonify({"success":True,"video_id":video_id})

    @app.route("/api/live/stop", methods=["POST"])
    def api_live_stop():
        stop_live_monitor()
        return jsonify({"success":True})

    # ── Scraping ──────────────────────────────────────────────────────────────
    @app.route("/api/scrape", methods=["POST"])
    def api_scrape():
        def _run():
            def _emit_prog(data):
                if _socketio_instance:
                    try: _socketio_instance.emit("scrape_progress", data, namespace="/ws")
                    except: pass
            total = full_scrape_channel(_emit_prog)
            if _socketio_instance:
                _socketio_instance.emit("scrape_done", {"total_messages":total}, namespace="/ws")
        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"success":True,"total_messages":0,"message":"Tarama arka planda başlatıldı"})

    # ── Dataset ───────────────────────────────────────────────────────────────
    @app.route("/api/dataset/pending")
    def api_dataset_pending():
        rows = db_exec("SELECT * FROM dataset WHERE confirmed=0 ORDER BY created_at DESC LIMIT 100",
                       fetch="all") or []
        return jsonify({"items":[dict(r) for r in rows]})

    @app.route("/api/dataset/approve", methods=["POST"])
    def api_dataset_approve():
        item_id = request.form.get("id")
        label   = request.form.get("label")
        if not item_id: return jsonify({"success":False})
        approve_dataset_item(int(item_id), label)
        return jsonify({"success":True})

    @app.route("/api/retrain", methods=["POST"])
    def api_retrain():
        result = retrain_naive_bayes()
        return jsonify(result)

    @app.route("/api/retrain/approve", methods=["POST"])
    def api_retrain_approve():
        if check_retrain_needed():
            result = retrain_naive_bayes()
            return jsonify(result)
        return jsonify({"message":"Henüz yeterli yeni veri yok"})

    # ── Oyun kuramı ──────────────────────────────────────────────────────────
    @app.route("/api/nash")
    def api_nash():
        nash = find_nash_equilibria()
        return jsonify({"equilibria": [[e[0],e[1],e[2],e[3]] for e in nash]})

    # ── Sistem durumu ─────────────────────────────────────────────────────────
    @app.route("/api/status")
    def api_status():
        return jsonify({
            "Selenium":        SELENIUM_OK,
            "Flask":           FLASK_OK,
            "Transformers":    TRANSFORMERS_OK,
            "Sentence-BERT":   SBERT_OK,
            "Torch":           TORCH_OK,
            "spaCy":           SPACY_OK,
            "LangDetect":      LANGDETECT_OK,
            "fasttext":        FASTTEXT_OK,
            "BERTopic":        BERTOPIC_OK,
            "ChromaDB":        CHROMA_OK,
            "HMMlearn":        HMMLEARN_OK,
            "Ollama":          OLLAMA_OK,
            "Device":          DEVICE,
            "Selenium Driver": _selenium_driver is not None,
            "Live Monitor":    _live_monitor_active,
            "Ollama Model":    CONFIG.get("ollama_model","phi4:14b"),
        })

    # ── Kullanıcı hesap bilgisi (Selenium) ───────────────────────────────────
    @app.route("/api/user/<author>/account")
    def api_user_account(author):
        row = db_exec("SELECT author_cid FROM user_profiles WHERE author=?", (author,), fetch="one")
        cid = row["author_cid"] if row else ""
        if not cid: return jsonify({"error":"Channel ID bulunamadı"})
        driver = _selenium_driver
        if not driver: return jsonify({"error":"Selenium bağlantısı yok"})
        info = inspect_user_account_selenium(driver, cid)
        if info:
            upsert_user_profile(author, {
                "account_created": info.get("account_created",""),
                "subscriber_count": info.get("subscriber_count",0),
                "is_new_account": int(info.get("is_new_account",False))
            })
        return jsonify(info)

    # ── WebSocket ─────────────────────────────────────────────────────────────
    @socketio.on("connect", namespace="/ws")
    def ws_connect(): emit("connected", {"status":"OK"})

    @socketio.on("ping", namespace="/ws")
    def ws_ping(): emit("pong", {"ts":int(time.time())})

    return app, socketio

# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 24 — ANA PROGRAM
# ═══════════════════════════════════════════════════════════════════════════════
def bootstrap():
    """Sistem başlatma — DB, ChromaDB, model önyükleme"""
    log.info("=" * 60)
    log.info("  YT GUARDIAN v2.0 başlatılıyor")
    log.info("  Kanal: %s", CONFIG.get("channel_url",""))
    log.info("  Cihaz: %s | Ollama: %s", DEVICE, CONFIG.get("ollama_model",""))
    log.info("=" * 60)

    Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
    init_db()
    init_chroma()

    # TF-IDF: mevcut mesajlarla yükle
    rows = db_exec("SELECT message FROM messages LIMIT 10000", fetch="all") or []
    if rows:
        texts = [r["message"] for r in rows]
        fit_tfidf(texts)
        log.info("✅ TF-IDF: %d mesajla güncellendi", len(texts))

    # Q-table yükle (varsa)
    _qtable.load("qtable.npy")

    log.info("✅ Bootstrap tamamlandı")

def run_cli_scrape():
    """Komut satırından tek seferlik tarama"""
    bootstrap()
    log.info("▶ Kanal taraması başlıyor: %s", CONFIG["channel_url"])
    total = full_scrape_channel()
    log.info("✅ Tarama tamamlandı: %d mesaj", total)
    # Tüm kullanıcıları analiz et
    rows = db_exec("SELECT DISTINCT author FROM messages WHERE deleted=0",fetch="all") or []
    for row in rows:
        try:
            analyze_user_full(row["author"], run_ollama=False)
        except Exception as e:
            log.warning("Analiz hatası @%s: %s", row["author"], e)
    _qtable.save("qtable.npy")
    log.info("✅ Analiz tamamlandı")

def main():
    global CONFIG
    parser = argparse.ArgumentParser(description="YT Guardian v2.0")
    parser.add_argument("--scrape", action="store_true", help="Sadece tarama yap (web panel başlatma)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--config", type=str, default="yt_guardian_config.json")
    parser.add_argument("--analyze-all", action="store_true", help="Tüm kullanıcıları analiz et")
    args = parser.parse_args()

    # Konfigürasyon dosyası belirtilmişse yeniden yükle
    CONFIG = load_config(args.config)

    bootstrap()

    if args.scrape:
        run_cli_scrape()
        return

    if args.analyze_all:
        rows = db_exec("SELECT DISTINCT author FROM messages WHERE deleted=0",fetch="all") or []
        for row in rows:
            try: analyze_user_full(row["author"], run_ollama=False)
            except: pass
        _qtable.save("qtable.npy")
        log.info("✅ Tüm kullanıcı analizi tamamlandı")
        return

    # Web paneli başlat
    if not FLASK_OK:
        log.error("Flask yüklü değil. Lütfen: pip install flask flask-socketio flask-cors eventlet")
        sys.exit(1)

    app, socketio = create_app()
    port = args.port
    log.info("🌐 Web paneli başlatılıyor: http://localhost:%d", port)
    log.info("   Dashboard → http://localhost:%d", port)
    log.info("   İptal için Ctrl+C")

    try:
        socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        log.info("Durduruldu")
        stop_live_monitor()
        _qtable.save("qtable.npy")
        if _selenium_driver:
            try: _selenium_driver.quit()
            except: pass

if __name__ == "__main__":
    main()
