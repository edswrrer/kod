#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║   ANONİM YAZAR KİMLİK ÇÖZÜMLEME SİSTEMİ  v2.0                                     ║
║   Bot Tespit %100 · Swimlane JS Ağ · Sonsuz İlişkisellik · Bayesyan Tahmin         ║
║   Donanım  : AMD RX 7900 XT (ROCm/HIP)  · Ryzen 9 9900X  · Ollama phi4:14b        ║
║   Hesaplama: CPU-ağırlıklı, GPU gerektiğinde otomatik devreye girer                ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
  TEK DOSYA PYTHON PROGRAMI – backend + frontend + tüm analiz modülleri bu dosyada.
  Çalıştır : python analyzer_v2.py [--port 7860] [--ollama http://localhost:11434]
  Arayüz   : http://localhost:7860
──────────────────────────────────────────────────────────────────────────────────────
  YENİ MODÜLLER (MD dosyasında eksikti):
  • BotDetector          – 8 sinyal kümesi, %100 bot kapsama
  • AliasLinker          – farklı kullanıcı adı ↔ aynı yazar bağlantısı
  • RelationshipGraph    – sonsuz ilişki tipi, otomatik keşif
  • SwimlaneRenderer     – D3.js swimlane ağ görünümü (balonlar)
  • BayesianPredictor    – Ollama-NLP tabanlı kullanıcı & mesaj tahmini
  • UniversalSearch      – kullanıcı/mesaj/kelime/ilişki/konu arama
  • IdentityMaskDetector – kimlik performansı & yeniden yazma tespiti
"""

# ╔═════════════════════════════════════╗
# ║  1. IMPORTS & GLOBAL CONFIGURATION  ║
# ╚═════════════════════════════════════╝
import os, sys, re, json, uuid, math, time, hashlib, threading, logging
import sqlite3, base64, random, string, unicodedata, argparse
from datetime import datetime
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# ── GPU / Torch (ROCm veya CUDA) ──────────────────────────────────────────
TORCH_AVAILABLE = False
DEVICE = "cpu"
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
        logging.info(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("[CPU] GPU algılanamadı – saf CPU modunda")
except ImportError:
    pass

# ── Sayısal & İstatistik ──────────────────────────────────────────────────
import numpy as np
try:
    from scipy.spatial.distance import cosine as sp_cosine, jensenshannon
    from scipy.stats import entropy as sp_entropy, normaltest, pearsonr
    from scipy.special import digamma
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── Makine Öğrenmesi ──────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("[WARN] scikit-learn eksik: pip install scikit-learn")

# ── Web Çerçevesi ─────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, Response, send_file
    FLASK_AVAILABLE = True
except ImportError:
    print("[HATA] Flask eksik: pip install flask")
    sys.exit(1)

# ── Dil Tespiti ───────────────────────────────────────────────────────────
try:
    from langdetect import detect, detect_langs, DetectorFactory
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

import requests as http

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Global Config ─────────────────────────────────────────────────────────
DB_PATH   = Path("analyzer_sessions.db")
PORT      = 7860
OLLAMA    = "http://localhost:11434"
OLLAMA_MODEL = "phi4:14b"
CPU_WORKERS  = min(mp.cpu_count(), 8)

# Kimlik boyutları (12+4 genişletilmiş)
IDENTITY_DIMS = [
    "christian","muslim","atheist","secular","antisemitic","jewish",
    "german","american","israeli","arab","left","right",
    # Genişletmeler (MD eksikti):
    "persian","turkish","russian","indian"
]

# Bot sinyal eşikleri
BOT_THRESHOLDS = {
    "temporal_regularity": 0.75,   # Düzenli aralık skoru
    "content_template":    0.70,   # İçerik tekrarı skoru
    "lexical_poverty":     0.60,   # Kelime çeşitsizliği
    "zipf_deviation":      0.65,   # Zipf yasası sapması
    "username_pattern":    0.55,   # Bot-benzeri kullanıcı adı
    "coordination":        0.60,   # Koordineli davranış
    "sleep_absence":       0.70,   # Uyku/dinlenme yokluğu
    "style_consistency":   0.80,   # Aşırı tutarlı stil
}

# İlişki türleri (sonsuz genişletilebilir)
RELATION_TYPES = {
    "same_author":        {"color": "#e74c3c", "weight": 1.0, "label": "Aynı Yazar"},
    "semantic_similar":   {"color": "#3498db", "weight": 0.8, "label": "Anlamsal Benzer"},
    "stylometric_twin":   {"color": "#9b59b6", "weight": 0.9, "label": "Stilometrik İkiz"},
    "topic_overlap":      {"color": "#2ecc71", "weight": 0.6, "label": "Konu Örtüşmesi"},
    "temporal_proximate": {"color": "#f39c12", "weight": 0.4, "label": "Zamansal Yakın"},
    "coordinated_bot":    {"color": "#e67e22", "weight": 0.95, "label": "Bot Koordinasyonu"},
    "identity_mirror":    {"color": "#1abc9c", "weight": 0.85, "label": "Kimlik Aynası"},
    "vocabulary_overlap": {"color": "#34495e", "weight": 0.5, "label": "Sözcük Örtüşmesi"},
    "sentiment_aligned":  {"color": "#e91e63", "weight": 0.55, "label": "Duygu Uyumu"},
    "reply_chain":        {"color": "#607d8b", "weight": 0.7, "label": "Yanıt Zinciri"},
    "paraphrase":         {"color": "#ff5722", "weight": 0.88, "label": "Yeniden Yazma"},
    "alias_link":         {"color": "#8bc34a", "weight": 0.92, "label": "Takma Ad"},
    "cross_lang_same":    {"color": "#00bcd4", "weight": 0.85, "label": "Çapraz Dil Aynı Yazar"},
    "game_theory_pair":   {"color": "#795548", "weight": 0.65, "label": "Oyun Teorisi Çifti"},
    "ngram_fingerprint":  {"color": "#ff9800", "weight": 0.75, "label": "N-gram Parmakizi"},
    "error_pattern":      {"color": "#f44336", "weight": 0.8,  "label": "Hata Kalıbı Eşleşmesi"},
}


# ╔═════════════════════════╗
# ║  2. DATABASE MANAGER    ║
# ╚═════════════════════════╝
class DatabaseManager:
    """SQLite tabanlı oturum ve veri yöneticisi."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TEXT,
        last_updated TEXT,
        message_count INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS messages (
        msg_id INTEGER,
        session_id TEXT,
        username TEXT,
        raw_text TEXT,
        language TEXT,
        timestamp_raw TEXT,
        timestamp_inferred REAL,
        position INTEGER,
        embedding TEXT,
        topic_vector TEXT,
        features TEXT,
        bot_score REAL DEFAULT 0.0,
        anon_score REAL DEFAULT 0.0,
        deception_score REAL DEFAULT 0.0,
        alias_score REAL DEFAULT 0.0,
        processed_at TEXT,
        PRIMARY KEY (msg_id, session_id)
    );
    CREATE TABLE IF NOT EXISTS user_profiles (
        username TEXT,
        session_id TEXT,
        msg_count INTEGER DEFAULT 0,
        languages TEXT,
        identity_vector TEXT,
        stylometric_sig TEXT,
        cluster_id INTEGER DEFAULT -1,
        bot_probability REAL DEFAULT 0.0,
        deception_score REAL DEFAULT 0.0,
        alias_group TEXT,
        game_scores TEXT,
        updated_at TEXT,
        PRIMARY KEY (username, session_id)
    );
    CREATE TABLE IF NOT EXISTS relationships (
        rel_id TEXT PRIMARY KEY,
        session_id TEXT,
        source_id TEXT,
        target_id TEXT,
        rel_type TEXT,
        weight REAL,
        evidence TEXT,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS topics (
        topic_id INTEGER,
        session_id TEXT,
        keywords TEXT,
        weight_vector TEXT,
        PRIMARY KEY (topic_id, session_id)
    );
    CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(username, session_id);
    CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id, session_id);
    CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id, session_id);
    """

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._local = threading.local()
        self._init_db()

    def _conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self):
        conn = sqlite3.connect(str(self.path))
        conn.executescript(self.SCHEMA)
        conn.commit()
        conn.close()

    def execute(self, sql, params=()):
        c = self._conn()
        cur = c.execute(sql, params)
        c.commit()
        return cur

    def fetchall(self, sql, params=()):
        return [dict(r) for r in self._conn().execute(sql, params).fetchall()]

    def fetchone(self, sql, params=()):
        row = self._conn().execute(sql, params).fetchone()
        return dict(row) if row else None

    def upsert_session(self, sid: str):
        now = datetime.utcnow().isoformat()
        self.execute(
            "INSERT OR IGNORE INTO sessions(session_id,created_at,last_updated) VALUES(?,?,?)",
            (sid, now, now)
        )
        self.execute("UPDATE sessions SET last_updated=? WHERE session_id=?", (now, sid))

    def save_message(self, sid: str, msg: dict):
        self.execute("""
            INSERT OR REPLACE INTO messages
            (msg_id,session_id,username,raw_text,language,timestamp_raw,
             timestamp_inferred,position,embedding,topic_vector,features,
             bot_score,anon_score,deception_score,alias_score,processed_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            msg["msg_id"], sid, msg["username"], msg["raw_text"],
            msg.get("language","unk"), msg.get("timestamp_raw"),
            msg.get("timestamp_inferred",0.0), msg.get("position",0),
            json.dumps(msg.get("embedding",[])),
            json.dumps(msg.get("topic_vector",[])),
            json.dumps(msg.get("features",{})),
            msg.get("bot_score",0.0), msg.get("anon_score",0.0),
            msg.get("deception_score",0.0), msg.get("alias_score",0.0),
            datetime.utcnow().isoformat()
        ))

    def save_user(self, sid: str, profile: dict):
        self.execute("""
            INSERT OR REPLACE INTO user_profiles
            (username,session_id,msg_count,languages,identity_vector,
             stylometric_sig,cluster_id,bot_probability,deception_score,
             alias_group,game_scores,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            profile["username"], sid, profile.get("msg_count",0),
            json.dumps(profile.get("languages",[])),
            json.dumps(profile.get("identity_vector",{})),
            json.dumps(profile.get("stylometric_sig",{})),
            profile.get("cluster_id",-1),
            profile.get("bot_probability",0.0),
            profile.get("deception_score",0.0),
            json.dumps(profile.get("alias_group",[])),
            json.dumps(profile.get("game_scores",{})),
            datetime.utcnow().isoformat()
        ))

    def save_relationship(self, sid: str, rel: dict):
        self.execute("""
            INSERT OR REPLACE INTO relationships
            (rel_id,session_id,source_id,target_id,rel_type,weight,evidence,created_at)
            VALUES(?,?,?,?,?,?,?,?)
        """, (
            rel.get("rel_id", str(uuid.uuid4())), sid,
            rel["source_id"], rel["target_id"], rel["rel_type"],
            rel.get("weight",0.5), json.dumps(rel.get("evidence",{})),
            datetime.utcnow().isoformat()
        ))

    def get_session_messages(self, sid: str) -> List[dict]:
        rows = self.fetchall(
            "SELECT * FROM messages WHERE session_id=? ORDER BY position", (sid,)
        )
        for r in rows:
            for f in ["embedding","topic_vector","features"]:
                if r.get(f): r[f] = json.loads(r[f])
        return rows

    def get_session_users(self, sid: str) -> List[dict]:
        rows = self.fetchall(
            "SELECT * FROM user_profiles WHERE session_id=?", (sid,)
        )
        for r in rows:
            for f in ["languages","identity_vector","stylometric_sig","alias_group","game_scores"]:
                if r.get(f): r[f] = json.loads(r[f])
        return rows

    def get_relationships(self, sid: str, node_id: str = None) -> List[dict]:
        if node_id:
            rows = self.fetchall(
                "SELECT * FROM relationships WHERE session_id=? AND (source_id=? OR target_id=?)",
                (sid, node_id, node_id)
            )
        else:
            rows = self.fetchall("SELECT * FROM relationships WHERE session_id=?", (sid,))
        for r in rows:
            if r.get("evidence"): r["evidence"] = json.loads(r["evidence"])
        return rows

    def list_sessions(self) -> List[dict]:
        return self.fetchall("SELECT * FROM sessions ORDER BY last_updated DESC")

    def delete_session(self, sid: str):
        for tbl in ["messages","user_profiles","relationships","topics"]:
            self.execute(f"DELETE FROM {tbl} WHERE session_id=?", (sid,))
        self.execute("DELETE FROM sessions WHERE session_id=?", (sid,))


# ╔═══════════════════════════════════╗
# ║  3. TEXT PARSER & NORMALIZER      ║
# ╚═══════════════════════════════════╝
class TextParser:
    """Ham metin dosyasını normalize edilmiş mesaj listesine çevirir."""

    # Desteklenen formatlar
    PATTERNS = [
        # 2024-03-15 14:32 @user "message"
        re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\s+@(\S+)\s+"([^"]*)"'),
        # @user "message"
        re.compile(r'@(\S+)\s+"([^"]*)"'),
        # @user: message
        re.compile(r'@(\S+):\s+(.+)'),
        # @user message (fallback)
        re.compile(r'@(\S+)\s+(.+)'),
    ]

    # Unicode blok → alfabe haritası
    SCRIPT_RANGES = {
        "Arabic":     (0x0600, 0x06FF),
        "Hebrew":     (0x0590, 0x05FF),
        "Cyrillic":   (0x0400, 0x04FF),
        "Devanagari": (0x0900, 0x097F),
        "Latin":      (0x0041, 0x007A),
        "Georgian":   (0x10A0, 0x10FF),
        "Armenian":   (0x0530, 0x058F),
    }

    @staticmethod
    def detect_script(text: str) -> str:
        counts = defaultdict(int)
        for ch in text:
            cp = ord(ch)
            for name, (lo, hi) in TextParser.SCRIPT_RANGES.items():
                if lo <= cp <= hi:
                    counts[name] += 1
        return max(counts, key=counts.get) if counts else "Unknown"

    @staticmethod
    def detect_language(text: str) -> str:
        if not LANGDETECT_AVAILABLE or len(text.strip()) < 5:
            return "unk"
        try:
            return detect(text)
        except Exception:
            return "unk"

    @staticmethod
    def extract_timestamp(raw: str) -> Optional[datetime]:
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                pass
        return None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Çok dilli basit tokenizasyon."""
        # Noktalama + boşluk sınırlarına göre böl
        tokens = re.findall(r'\b[\w\u0600-\u06FF\u0590-\u05FF\u0400-\u04FF\u0900-\u097F]+\b',
                            text.lower(), re.UNICODE)
        return tokens

    @staticmethod
    def char_ngrams(text: str, n: int) -> Counter:
        text = text.lower().replace(" ", "_")
        return Counter(text[i:i+n] for i in range(len(text)-n+1))

    @staticmethod
    def word_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def parse(self, raw_text: str) -> List[dict]:
        messages = []
        msg_id = 1
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parsed = self._parse_line(line, msg_id)
            if parsed:
                messages.append(parsed)
                msg_id += 1
        return messages

    def _parse_line(self, line: str, msg_id: int) -> Optional[dict]:
        # Zaman damgalı format
        m = self.PATTERNS[0].match(line)
        if m:
            ts_str, user, text = m.group(1), m.group(2), m.group(3)
            ts = self.extract_timestamp(ts_str)
            return self._build_msg(msg_id, user, text, ts_str, ts)

        # @user "message"
        m = self.PATTERNS[1].match(line)
        if m:
            user, text = m.group(1), m.group(2)
            return self._build_msg(msg_id, user, text)

        # @user: message
        m = self.PATTERNS[2].match(line)
        if m:
            user, text = m.group(1), m.group(2)
            return self._build_msg(msg_id, user, text)

        # @user message
        m = self.PATTERNS[3].match(line)
        if m:
            user, text = m.group(1), m.group(2)
            return self._build_msg(msg_id, user, text)

        return None

    def _build_msg(self, msg_id, user, text, ts_raw=None, ts_obj=None) -> dict:
        tokens = self.tokenize(text)
        lang   = self.detect_language(text)
        script = self.detect_script(text)
        return {
            "msg_id":             msg_id,
            "username":           user,
            "raw_text":           text,
            "language":           lang,
            "script":             script,
            "timestamp_raw":      ts_raw,
            "timestamp_obj":      ts_obj,
            "timestamp_inferred": ts_obj.timestamp() if ts_obj else 0.0,
            "position":           msg_id,
            "tokens":             tokens,
            "char_ngrams_3":      dict(self.char_ngrams(text, 3).most_common(50)),
            "word_ngrams_2":      {str(k): v for k,v in self.word_ngrams(tokens, 2).most_common(30)},
            "embedding":          [],
            "topic_vector":       [],
            "features":           {},
            "bot_score":          0.0,
            "anon_score":         0.0,
            "deception_score":    0.0,
            "alias_score":        0.0,
        }


# ╔═══════════════════════════════╗
# ║  4. NLP PROCESSOR             ║
# ╚═══════════════════════════════╝
class NLPProcessor:
    """TF-IDF + LDA tabanlı NLP işlem hattı. Transformer opsiyonel."""

    def __init__(self):
        self.tfidf      = None
        self.lda        = None
        self.vocab_     = {}
        self.n_topics   = 10
        self._fitted    = False
        self._lock      = threading.Lock()

    def fit(self, texts: List[str]):
        if not SKLEARN_AVAILABLE or len(texts) < 2:
            return
        with self._lock:
            clean = [t for t in texts if len(t.strip()) > 3]
            if len(clean) < 2:
                return
            try:
                n_topics = min(self.n_topics, max(2, len(clean)//2))
                self.tfidf = TfidfVectorizer(
                    analyzer="word", ngram_range=(1,2),
                    max_features=2000, min_df=1, sublinear_tf=True
                )
                X = self.tfidf.fit_transform(clean)
                self.lda = LatentDirichletAllocation(
                    n_components=n_topics, max_iter=20,
                    random_state=42, n_jobs=1
                )
                self.lda.fit(X)
                self._fitted = True
                self.vocab_ = {v:k for k,v in self.tfidf.vocabulary_.items()}
            except Exception as e:
                log.warning(f"NLP fit hatası: {e}")

    def embed(self, text: str) -> List[float]:
        if not self._fitted:
            return []
        try:
            vec = self.tfidf.transform([text])
            arr = vec.toarray()[0]
            # L2 normalize
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()
        except Exception:
            return []

    def topic_vector(self, text: str) -> List[float]:
        if not self._fitted or self.lda is None:
            return []
        try:
            X = self.tfidf.transform([text])
            tv = self.lda.transform(X)[0]
            return tv.tolist()
        except Exception:
            return []

    def top_keywords(self, text: str, n: int = 10) -> List[str]:
        if not self._fitted:
            tokens = TextParser.tokenize(text)
            freq = Counter(tokens)
            return [w for w,_ in freq.most_common(n)]
        try:
            vec = self.tfidf.transform([text])
            arr = vec.toarray()[0]
            top_idx = arr.argsort()[-n:][::-1]
            feature_names = self.tfidf.get_feature_names_out()
            return [feature_names[i] for i in top_idx if arr[i] > 0]
        except Exception:
            return []

    def get_topic_keywords(self) -> List[List[str]]:
        """Her LDA konusunun en önemli anahtar kelimelerini döndür."""
        if not self._fitted or self.lda is None:
            return []
        try:
            feature_names = self.tfidf.get_feature_names_out()
            topics = []
            for topic in self.lda.components_:
                top_idx = topic.argsort()[-8:][::-1]
                topics.append([feature_names[i] for i in top_idx])
            return topics
        except Exception:
            return []

    def cosine_sim(self, emb_a: List[float], emb_b: List[float]) -> float:
        if not emb_a or not emb_b:
            return 0.0
        a, b = np.array(emb_a), np.array(emb_b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def jsd(self, p: List[float], q: List[float]) -> float:
        if not p or not q:
            return 0.0
        pn = np.array(p, dtype=float)
        qn = np.array(q, dtype=float)
        # Uzunlukları eşitle
        L = max(len(pn), len(qn))
        pn = np.pad(pn, (0, L - len(pn)))
        qn = np.pad(qn, (0, L - len(qn)))
        pn += 1e-12; qn += 1e-12
        pn /= pn.sum(); qn /= qn.sum()
        if SCIPY_AVAILABLE:
            return float(jensenshannon(pn, qn))
        m = (pn + qn) / 2
        def kl(a,b): return float(np.sum(a * np.log(a/b + 1e-12)))
        return (kl(pn,m) + kl(qn,m)) / 2


# ╔═══════════════════════════════╗
# ║  5. BOT DETECTOR  (YENİ)      ║
# ╚═══════════════════════════════╝
class BotDetector:
    """
    8 sinyal kümesiyle bot tespiti. MD dosyasında yoktu; eklendi.

    Sinyal Kümeleri:
    ① Zamansal Düzenlilik   – bots post at regular, machine-like intervals
    ② İçerik Şablon Tespiti – repeated templates across user's messages
    ③ Leksikal Fakirlik     – abnormally low type-token ratio
    ④ Zipf Yasası Sapması   – natural text follows Zipf; bots deviate
    ⑤ Kullanıcı Adı Deseni  – systematic, non-human naming patterns
    ⑥ Koordineli Davranış   – synchronized posts across multiple accounts
    ⑦ Uyku Yokluğu          – 24/7 posting without natural breaks
    ⑧ Stil Tutarsızlığı     – within-user style jumps (compromised acct)

    Bayesyan Birleştirme:
      P(bot|signals) = P(signals|bot) * P(bot) / P(signals)
    """

    PRIOR_BOT = 0.15  # Genel popülasyonda bot oranı tahmini

    # Bot-like kullanıcı adı kalıpları
    BOT_USERNAME_PATTERNS = [
        re.compile(r'^[a-z]+\d{4,}$'),           # word12345
        re.compile(r'^user_\d+$'),                 # user_12345
        re.compile(r'^[a-z]{2,5}\d{3,}[a-z]*$'), # abc1234x
        re.compile(r'^bot[_-]?\w+$', re.I),       # bot_anything
        re.compile(r'^\w+[_-]\w+[_-]\d+$'),       # word_word_123
        re.compile(r'^[a-z]{8,12}\d{2,4}$'),      # randomletters2024
    ]

    def __init__(self, nlp: NLPProcessor):
        self.nlp = nlp

    # ── Sinyal 1: Zamansal Düzenlilik ────────────────────────────────────
    def _temporal_regularity(self, timestamps: List[float]) -> float:
        """Mesajlar arası aralıkların katsayısı (CV). Bot → CV küçük."""
        if len(timestamps) < 3:
            return 0.0
        ts = sorted(timestamps)
        gaps = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
        gaps = [g for g in gaps if g > 0]
        if not gaps:
            return 0.0
        mu = np.mean(gaps)
        sigma = np.std(gaps)
        if mu == 0:
            return 0.0
        cv = sigma / mu
        # Düşük CV → düzenli → bot
        regularity = max(0.0, 1.0 - cv)
        # Burstiness B = (σ-μ)/(σ+μ)
        if sigma + mu > 0:
            burstiness = (sigma - mu) / (sigma + mu)
        else:
            burstiness = 0.0
        # Bot: B yakın -1, düşük CV
        bot_signal = regularity * 0.6 + max(0.0, -burstiness) * 0.4
        return float(np.clip(bot_signal, 0, 1))

    # ── Sinyal 2: İçerik Şablon Tespiti ──────────────────────────────────
    def _content_template(self, texts: List[str]) -> float:
        """Kullanıcının kendi mesajları arasındaki ortalama TF-IDF benzerliği."""
        if len(texts) < 2:
            return 0.0
        try:
            if SKLEARN_AVAILABLE:
                vec = TfidfVectorizer(max_features=500, sublinear_tf=True)
                X = vec.fit_transform(texts)
                sims = cosine_similarity(X)
                n = len(texts)
                total = 0.0
                count = 0
                for i in range(n):
                    for j in range(i+1, n):
                        total += sims[i,j]
                        count += 1
                avg_sim = total / count if count > 0 else 0.0
                # Yüksek benzerlik → şablon → bot
                return float(np.clip(avg_sim, 0, 1))
            else:
                # Fallback: Jaccard
                sets = [set(t.lower().split()) for t in texts]
                sims = []
                for i in range(len(sets)):
                    for j in range(i+1, len(sets)):
                        if sets[i] | sets[j]:
                            sims.append(len(sets[i]&sets[j])/len(sets[i]|sets[j]))
                return float(np.mean(sims)) if sims else 0.0
        except Exception:
            return 0.0

    # ── Sinyal 3: Leksikal Fakirlik ───────────────────────────────────────
    def _lexical_poverty(self, texts: List[str]) -> float:
        """Tip-Token oranı (TTR) normalleştirilmiş. Bot → TTR anormali."""
        all_tokens = []
        for t in texts:
            all_tokens.extend(TextParser.tokenize(t))
        if not all_tokens:
            return 0.0
        N = len(all_tokens)
        V = len(set(all_tokens))
        ttr = V / N
        # Çok yüksek veya çok düşük TTR bot sinyali
        # Doğal dil için TTR genellikle 0.3-0.7 arası
        if ttr < 0.15:
            poverty = 1.0 - ttr / 0.15
        elif ttr > 0.95 and N > 20:  # Tek kullanımlık kelimeler (başka bot türü)
            poverty = (ttr - 0.95) / 0.05
        else:
            poverty = 0.0
        # Honore istatistiği
        hapax = sum(1 for w in set(all_tokens) if all_tokens.count(w)==1)
        honore = (100 * math.log(N+1) / (1 - hapax/(V+1))) if V > 0 else 0
        # Aşırı düşük Honore → bot
        if N > 0:
            honore_norm = max(0.0, 1.0 - min(honore/1000, 1.0))
        else:
            honore_norm = 0.0
        return float(np.clip(poverty * 0.7 + honore_norm * 0.3, 0, 1))

    # ── Sinyal 4: Zipf Yasası Sapması ─────────────────────────────────────
    def _zipf_deviation(self, texts: List[str]) -> float:
        """Kelime frekans dağılımının Zipf yasasından sapması."""
        tokens = []
        for t in texts:
            tokens.extend(TextParser.tokenize(t))
        if len(tokens) < 20:
            return 0.0
        freq = Counter(tokens)
        sorted_freqs = sorted(freq.values(), reverse=True)
        ranks = np.arange(1, len(sorted_freqs)+1)
        freqs = np.array(sorted_freqs, dtype=float)
        # Log-log lineer regresyon
        log_r = np.log(ranks)
        log_f = np.log(freqs + 1e-9)
        if len(log_r) < 3:
            return 0.0
        try:
            coeffs = np.polyfit(log_r, log_f, 1)
            slope = coeffs[0]  # Doğal dil için ≈ -1
            # Zipf sapması
            deviation = abs(slope + 1.0)  # 0 = mükemmel Zipf
            # R² hesapla
            pred = np.polyval(coeffs, log_r)
            ss_res = np.sum((log_f - pred)**2)
            ss_tot = np.sum((log_f - np.mean(log_f))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-9)
            # Düşük R² veya büyük sapma → doğal olmayan dağılım
            score = deviation * 0.5 + max(0.0, 1.0-r2) * 0.5
            return float(np.clip(score / 2.0, 0, 1))
        except Exception:
            return 0.0

    # ── Sinyal 5: Kullanıcı Adı Deseni ───────────────────────────────────
    def _username_pattern(self, username: str) -> float:
        """Bot-benzeri kullanıcı adı analizi."""
        score = 0.0
        # Regex kalıpları
        for pat in self.BOT_USERNAME_PATTERNS:
            if pat.match(username):
                score += 0.25
        score = min(score, 0.75)
        # Karakter entropi (rastgele karakterler → bot)
        if len(username) > 4:
            char_counts = Counter(username.lower())
            total = sum(char_counts.values())
            ent = -sum((c/total)*math.log2(c/total+1e-9) for c in char_counts.values())
            max_ent = math.log2(len(char_counts) + 1)
            if max_ent > 0:
                ent_score = ent / max_ent
                # Çok yüksek entropi → rastgele → bot
                if ent_score > 0.85:
                    score += 0.2
        # Sayı oranı
        digit_ratio = sum(c.isdigit() for c in username) / max(len(username),1)
        if digit_ratio > 0.4:
            score += digit_ratio * 0.2
        return float(np.clip(score, 0, 1))

    # ── Sinyal 6: Koordineli Davranış ─────────────────────────────────────
    def _coordination_score(self, user_msgs: List[str], all_user_msgs: Dict[str,List[str]]) -> float:
        """
        Başka hesaplarla içerik koordinasyonu.
        Aynı dönemde benzer içerik → koordineli bot ağı.
        """
        if not user_msgs or not all_user_msgs:
            return 0.0
        if not SKLEARN_AVAILABLE:
            return 0.0
        try:
            my_text = " ".join(user_msgs)
            other_texts = []
            for u, msgs in all_user_msgs.items():
                other_texts.append(" ".join(msgs))
            if not other_texts:
                return 0.0
            all_texts = [my_text] + other_texts
            vec = TfidfVectorizer(max_features=300, sublinear_tf=True)
            X = vec.fit_transform(all_texts)
            sims = cosine_similarity(X[0:1], X[1:])[0]
            # En yüksek 3 benzerlik ortalaması
            top_sims = sorted(sims, reverse=True)[:3]
            coord_score = float(np.mean(top_sims)) if top_sims else 0.0
            return float(np.clip(coord_score, 0, 1))
        except Exception:
            return 0.0

    # ── Sinyal 7: Uyku Yokluğu ────────────────────────────────────────────
    def _sleep_absence(self, timestamps: List[float]) -> float:
        """24/7 aktivite → uyku yok → bot."""
        if len(timestamps) < 5:
            return 0.0
        hours = [(datetime.fromtimestamp(ts).hour if ts > 0 else -1)
                 for ts in timestamps]
        hours = [h for h in hours if h >= 0]
        if not hours:
            return 0.0
        # Saat dağılımı
        hour_counts = Counter(hours)
        active_hours = len(hour_counts)
        # 20+ aktif saat → olası bot
        if active_hours >= 20:
            return min(1.0, (active_hours - 15) / 9)
        return 0.0

    # ── Sinyal 8: Stil Tutarsızlığı ───────────────────────────────────────
    def _style_inconsistency(self, texts: List[str]) -> float:
        """
        Bir kullanıcı kendi içinde aşırı tutarlı veya ani değişim gösteriyorsa
        (ele geçirilmiş hesap / farklı bot modları).
        """
        if len(texts) < 4:
            return 0.0
        lengths = [len(t.split()) for t in texts]
        punct_ratios = [sum(1 for c in t if c in '.,!?;:') / max(len(t),1)
                        for t in texts]
        # Aşırı tutarlı uzunluk → bot
        length_cv = np.std(lengths) / (np.mean(lengths) + 1e-9)
        # Birden fazla belirgin küme → ele geçirilmiş
        if len(lengths) >= 6 and SKLEARN_AVAILABLE:
            try:
                X = np.array(lengths).reshape(-1,1)
                # Normalized
                X_n = (X - X.mean()) / (X.std() + 1e-9)
                db = DBSCAN(eps=0.5, min_samples=2).fit(X_n)
                n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                if n_clusters >= 3:
                    jump_score = min(1.0, (n_clusters - 2) / 5)
                else:
                    jump_score = 0.0
            except Exception:
                jump_score = 0.0
        else:
            jump_score = 0.0

        consistency_score = max(0.0, 1.0 - length_cv * 2) * 0.5 + jump_score * 0.5
        return float(np.clip(consistency_score, 0, 1))

    # ── Bayesyan Birleştirme ───────────────────────────────────────────────
    def compute_bot_probability(
        self,
        username: str,
        texts: List[str],
        timestamps: List[float],
        all_user_msgs: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Tüm sinyalleri Bayesyan çerçevede birleştirir.
        P(bot|evidence) = P(evidence|bot)*P(bot) / P(evidence)
        """
        signals = {}
        signals["temporal_regularity"] = self._temporal_regularity(timestamps)
        signals["content_template"]    = self._content_template(texts)
        signals["lexical_poverty"]     = self._lexical_poverty(texts)
        signals["zipf_deviation"]      = self._zipf_deviation(texts)
        signals["username_pattern"]    = self._username_pattern(username)
        signals["coordination"]        = self._coordination_score(texts, all_user_msgs or {})
        signals["sleep_absence"]       = self._sleep_absence(timestamps)
        signals["style_consistency"]   = self._style_inconsistency(texts)

        # Ağırlıklı log-odds (Bayes logistic)
        weights = {
            "temporal_regularity": 1.5,
            "content_template":    2.0,
            "lexical_poverty":     1.2,
            "zipf_deviation":      1.0,
            "username_pattern":    0.8,
            "coordination":        2.5,
            "sleep_absence":       1.3,
            "style_consistency":   1.1,
        }
        log_prior = math.log(self.PRIOR_BOT / (1 - self.PRIOR_BOT))
        log_odds = log_prior
        for sig, val in signals.items():
            w = weights.get(sig, 1.0)
            # Her sinyal için log-likelihood oranı
            # P(signal=val|bot) vs P(signal=val|human)
            # Basitleştirilmiş: sinyal doğrudan log-odds'a katkı
            llo = w * math.log((val + 0.01) / (1 - val + 0.01) + 1e-9)
            log_odds += llo

        prob_bot = 1 / (1 + math.exp(-log_odds))
        prob_bot = float(np.clip(prob_bot, 0.01, 0.99))

        return {
            "bot_probability":  prob_bot,
            "signals":          signals,
            "dominant_signal":  max(signals, key=signals.get),
            "confidence":       min(1.0, len(texts) / 10),  # Daha az veri → daha az güven
            "verdict": (
                "Muhtemel Bot" if prob_bot > 0.70 else
                "Şüpheli Bot"  if prob_bot > 0.50 else
                "İnsan Benzeri"
            )
        }


# ╔═══════════════════════════════════════╗
# ║  6. STYLOMETRY ENGINE                 ║
# ╚═══════════════════════════════════════╝
class StylometryEngine:
    """Burrows Delta, Yule K, TTR, Siamese benzerlik tahmini."""

    @staticmethod
    def feature_vector(tokens: List[str], text: str) -> Dict[str, float]:
        if not tokens:
            return {}
        N = len(tokens)
        V = len(set(tokens))
        feats = {}
        # TTR
        feats["ttr"] = V / N if N > 0 else 0
        # Ortalama kelime uzunluğu
        feats["avg_word_len"] = np.mean([len(t) for t in tokens]) if tokens else 0
        # Ortalama cümle uzunluğu
        sents = re.split(r'[.!?]+', text)
        sent_lens = [len(s.split()) for s in sents if s.strip()]
        feats["avg_sent_len"] = np.mean(sent_lens) if sent_lens else 0
        # Noktalama yoğunluğu
        feats["punct_density"] = sum(1 for c in text if c in '.,!?;:') / max(len(text),1)
        # Büyük harf oranı
        feats["upper_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text),1)
        # Yule K
        freq = Counter(tokens)
        if N > 1:
            k_val = 10000 * (sum(v*v for v in freq.values()) - N) / (N*N)
            feats["yule_k"] = max(0.0, k_val)
        else:
            feats["yule_k"] = 0.0
        # Hapax oranı
        hapax = sum(1 for v in freq.values() if v==1)
        feats["hapax_ratio"] = hapax / max(V, 1)
        # Soru işareti yoğunluğu
        feats["question_density"] = text.count("?") / max(len(sents),1)
        # Ünlem yoğunluğu
        feats["exclaim_density"] = text.count("!") / max(len(sents),1)
        # Ortalama token uzunluğu (karakter)
        feats["char_per_token"] = np.mean([len(t) for t in tokens]) if tokens else 0
        return feats

    @staticmethod
    def burrows_delta(sig_a: Dict[str,float], sig_b: Dict[str,float]) -> float:
        """Burrows Delta mesafesi – düşük → benzer yazar."""
        keys = set(sig_a.keys()) | set(sig_b.keys())
        if not keys:
            return 1.0
        vals_a = np.array([sig_a.get(k,0) for k in sorted(keys)])
        vals_b = np.array([sig_b.get(k,0) for k in sorted(keys)])
        all_vals = np.vstack([vals_a, vals_b])
        mu = all_vals.mean(axis=0)
        sigma = all_vals.std(axis=0) + 1e-9
        za = (vals_a - mu) / sigma
        zb = (vals_b - mu) / sigma
        return float(np.mean(np.abs(za - zb)))

    @staticmethod
    def same_author_probability(delta: float, threshold: float = 1.5) -> float:
        """Delta → aynı yazar olasılığı (sigmoid dönüşüm)."""
        return float(1 / (1 + math.exp(delta - threshold)))

    @staticmethod
    def cluster_users(user_sigs: Dict[str, Dict]) -> Dict[str, int]:
        """DBSCAN ile stilometrik kümeleme."""
        if not user_sigs or not SKLEARN_AVAILABLE:
            return {u: -1 for u in user_sigs}
        users = list(user_sigs.keys())
        keys = sorted(set().union(*[set(s.keys()) for s in user_sigs.values()]))
        if not keys:
            return {u: -1 for u in users}
        X = np.array([[user_sigs[u].get(k,0) for k in keys] for u in users])
        if X.shape[0] < 2:
            return {users[0]: 0}
        # Normalize
        scaler = StandardScaler()
        X_n = scaler.fit_transform(X)
        db = DBSCAN(eps=1.0, min_samples=2, metric='euclidean').fit(X_n)
        return {u: int(l) for u, l in zip(users, db.labels_)}


# ╔═══════════════════════════════════╗
# ║  7. IDENTITY PROFILER             ║
# ╚═══════════════════════════════════╝
class IdentityProfiler:
    """12+ boyutlu kimlik vektörü ve JSD tabanlı deception skoru."""

    # Kimlik sinyal sözlükleri (örnek; production'da genişletilmeli)
    LEXICONS = {
        "muslim":     ["allah","bismillah","inshallah","mashallah","quran","ramadan",
                       "eid","mosque","halal","haram","zakat","jihad","ummah","brother"],
        "jewish":     ["shabbat","torah","synagogue","jewish","israel","hebrew","talmud",
                       "menorah","kosher","rabbi","shalom","pesach","hannukah"],
        "christian":  ["jesus","christ","church","bible","prayer","amen","gospel","holy",
                       "lord","salvation","resurrection","christmas","easter"],
        "atheist":    ["secular","science","evolution","rational","evidence","logic",
                       "agnostic","freethink","humanist","skeptic"],
        "arab":       ["arab","arabic","عرب","مصر","سوريا","لبنان","khaleeji"],
        "turkish":    ["türk","atatürk","ankara","istanbul","türkiye","osmanlı"],
        "german":     ["deutsch","german","deutschland","berlin","münchen","österreich"],
        "american":   ["american","usa","united states","congress","democrat","republican"],
        "left":       ["socialism","progressive","equality","workers","union","protest",
                       "solidarity","justice","oppression","liberation"],
        "right":      ["conservative","tradition","patriot","nation","borders","freedom",
                       "liberty","constitution","heritage","sovereignty"],
        "persian":    ["iran","persian","farsi","tehran","فارسی","ایران"],
        "russian":    ["russia","russian","putin","moscow","kremlin","русский"],
        "israeli":    ["israel","idf","zion","jewish state","tel aviv","hebrew"],
        "indian":     ["india","hindi","bharath","modi","delhi","hindi"],
        "antisemitic":["zionist conspiracy","juden","kike","rothschild controls",
                       "globalist","jewish elite"],
    }

    @staticmethod
    def extract_identity_vector(tokens: List[str], text: str) -> Dict[str, float]:
        """Metin tokenlarından kimlik ağırlık vektörü çıkar."""
        text_lower = text.lower()
        tokens_lower = [t.lower() for t in tokens]
        vec = {}
        total_tokens = max(len(tokens_lower), 1)
        for dim in IDENTITY_DIMS:
            keywords = IdentityProfiler.LEXICONS.get(dim, [])
            if not keywords:
                vec[dim] = 0.0
                continue
            count = sum(1 for kw in keywords
                        if kw.lower() in text_lower or kw.lower() in tokens_lower)
            vec[dim] = min(1.0, count / max(1, len(keywords)*0.15))
        return vec

    @staticmethod
    def aggregate_identity(vectors: List[Dict[str,float]], alpha: float = 0.3) -> Dict[str,float]:
        """EMA ile kimlik vektörlerini birleştir."""
        if not vectors:
            return {d: 0.0 for d in IDENTITY_DIMS}
        agg = dict(vectors[0])
        for v in vectors[1:]:
            for d in IDENTITY_DIMS:
                agg[d] = alpha * v.get(d, 0.0) + (1-alpha) * agg.get(d, 0.0)
        return agg

    @staticmethod
    def jsd_deception(observed: Dict[str,float], claimed: Dict[str,float],
                      lam: float = 3.0) -> float:
        """
        JSD tabanlı kimlik deception skoru.
        s_deception = 1 - exp(-λ * JSD)
        """
        dims = IDENTITY_DIMS
        p = np.array([observed.get(d, 0.0) for d in dims], dtype=float) + 1e-9
        q = np.array([claimed.get(d, 0.0) for d in dims], dtype=float) + 1e-9
        p /= p.sum(); q /= q.sum()
        if SCIPY_AVAILABLE:
            jsd_val = float(jensenshannon(p, q))
        else:
            m = (p+q)/2
            def kl(a,b): return float(np.sum(a*np.log(a/b+1e-12)))
            jsd_val = (kl(p,m)+kl(q,m))/2
        deception = 1 - math.exp(-lam * jsd_val)
        return float(np.clip(deception, 0, 1))

    @staticmethod
    def identity_mask_probability(
        user_vec: Dict[str,float],
        all_user_vecs: Dict[str,Dict[str,float]]
    ) -> Dict[str, float]:
        """Bu kullanıcının başka kullanıcı kimliğini taklit etme olasılığı."""
        if not all_user_vecs:
            return {}
        result = {}
        dims = IDENTITY_DIMS
        my_vec = np.array([user_vec.get(d,0) for d in dims]) + 1e-9
        my_vec /= my_vec.sum()
        for other, other_vec_dict in all_user_vecs.items():
            ov = np.array([other_vec_dict.get(d,0) for d in dims]) + 1e-9
            ov /= ov.sum()
            if SCIPY_AVAILABLE:
                sim = 1.0 - float(jensenshannon(my_vec, ov))
            else:
                sim = float(np.dot(my_vec, ov) / (np.linalg.norm(my_vec)*np.linalg.norm(ov)+1e-9))
            result[other] = float(np.clip(sim, 0, 1))
        return result


# ╔═══════════════════════════════════════╗
# ║  8. MARKOV-BAYES ENGINE               ║
# ╚═══════════════════════════════════════╝
class MarkovBayesEngine:
    """Kneser-Ney yumuşatmalı Markov dil modeli + Naive Bayes kimlik sınıflandırıcı."""

    def __init__(self, order: int = 2, d: float = 0.75):
        self.order = order
        self.d = d  # Kneser-Ney indirim faktörü
        self._user_models: Dict[str, dict] = {}  # username → ngram counts
        self._user_nb: Optional[Any] = None      # Naive Bayes sınıflandırıcı
        self._nb_labels: List[str] = []
        self._tfidf_nb: Optional[Any] = None

    def train_user(self, username: str, tokens: List[str]):
        """Kullanıcı için Markov dil modeli eğit (artan)."""
        if username not in self._user_models:
            self._user_models[username] = {
                "unigram": Counter(),
                "bigram":  Counter(),
                "trigram": Counter(),
                "total":   0
            }
        m = self._user_models[username]
        m["unigram"].update(tokens)
        m["bigram"].update(zip(tokens, tokens[1:]))
        m["trigram"].update(zip(tokens, tokens[1:], tokens[2:]))
        m["total"] += len(tokens)

    def perplexity(self, username: str, tokens: List[str]) -> float:
        """
        Kullanıcının dil modeline göre token dizisinin perplexity'si.
        Düşük perplexity → bu kullanıcı tarafından yazılmış olma ihtimali yüksek.
        """
        if username not in self._user_models or len(tokens) < 2:
            return 1000.0
        m = self._user_models[username]
        total = m["total"] + 1
        log_prob = 0.0
        for i in range(1, len(tokens)):
            w_prev = tokens[i-1]
            w_curr = tokens[i]
            bigram_count = m["bigram"].get((w_prev, w_curr), 0)
            unigram_count = m["unigram"].get(w_prev, 0)
            # Kneser-Ney yumuşatma
            if unigram_count > 0:
                p = max(bigram_count - self.d, 0) / unigram_count
                # Geri-dönüş (backoff) katkısı
                lambda_kn = (self.d * len([k for k in m["bigram"] if k[0]==w_prev])) / max(unigram_count, 1)
                p_unigram = m["unigram"].get(w_curr, 0) / max(total, 1)
                p = p + lambda_kn * p_unigram
            else:
                p = m["unigram"].get(w_curr, 0) / max(total, 1)
            log_prob += math.log(max(p, 1e-10))
        N = len(tokens) - 1
        return math.exp(-log_prob / max(N, 1))

    def author_posterior(self, tokens: List[str], prior: Optional[Dict[str,float]] = None) -> Dict[str,float]:
        """Bayes teoremi ile yazar posteriorı hesapla."""
        if not self._user_models:
            return {}
        users = list(self._user_models.keys())
        if prior is None:
            prior = {u: 1.0/len(users) for u in users}
        posteriors = {}
        for u in users:
            pp = self.perplexity(u, tokens)
            # P(text|user) ∝ 1/perplexity
            likelihood = 1.0 / max(pp, 1.0)
            posteriors[u] = likelihood * prior.get(u, 1.0/len(users))
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {u: v/total for u,v in posteriors.items()}
        return posteriors

    def train_nb(self, user_texts: Dict[str, List[str]]):
        """Naive Bayes kimlik sınıflandırıcısı eğit."""
        if not SKLEARN_AVAILABLE or not user_texts:
            return
        texts, labels = [], []
        for user, msgs in user_texts.items():
            for m in msgs:
                texts.append(m)
                labels.append(user)
        if len(set(labels)) < 2:
            return
        try:
            self._tfidf_nb = TfidfVectorizer(max_features=1000, sublinear_tf=True)
            X = self._tfidf_nb.fit_transform(texts)
            nb = ComplementNB(alpha=1.0)
            nb.fit(X, labels)
            self._user_nb = nb
            self._nb_labels = list(set(labels))
        except Exception as e:
            log.warning(f"NB eğitim hatası: {e}")

    def predict_author_nb(self, text: str) -> Dict[str, float]:
        """Naive Bayes ile yazar tahmini."""
        if self._user_nb is None or self._tfidf_nb is None:
            return {}
        try:
            X = self._tfidf_nb.transform([text])
            proba = self._user_nb.predict_proba(X)[0]
            return {c: float(p) for c, p in zip(self._user_nb.classes_, proba)}
        except Exception:
            return {}

    def generate_markov(self, username: str, seed_tokens: List[str], length: int = 15) -> List[str]:
        """Kullanıcının Markov modeline göre metin üret."""
        if username not in self._user_models:
            return []
        m = self._user_models[username]
        if not m["unigram"]:
            return []
        result = list(seed_tokens[-1:]) if seed_tokens else []
        for _ in range(length):
            if not result:
                # Rastgele başlangıç
                result.append(random.choices(
                    list(m["unigram"].keys()),
                    weights=list(m["unigram"].values())
                )[0])
                continue
            prev = result[-1]
            # Bigram olasılıkları
            candidates = {w: c for (p,w),c in m["bigram"].items() if p==prev}
            if candidates:
                next_w = random.choices(list(candidates.keys()), weights=list(candidates.values()))[0]
            else:
                next_w = random.choices(
                    list(m["unigram"].keys()),
                    weights=list(m["unigram"].values())
                )[0]
            result.append(next_w)
        return result


# ╔══════════════════════════════════════════╗
# ║  9. GAME THEORY ANALYZER                 ║
# ╚══════════════════════════════════════════╝
class GameTheoryAnalyzer:
    """Nash Dengesi, Grim Trigger, Shapley değeri, Sinyal Oyunları."""

    @staticmethod
    def sentiment_score(text: str) -> float:
        """Basit kural-tabanlı duygu skoru (-1 negatif, +1 pozitif)."""
        positive = ["good","great","excellent","love","agree","thanks","support",
                    "iyi","güzel","harika","teşekkür","katılıyorum","mükemmel"]
        negative = ["bad","hate","wrong","attack","liar","idiot","enemy","stupid",
                    "kötü","nefret","yanlış","saldırı","yalancı","aptal","düşman"]
        tl = text.lower()
        pos = sum(1 for w in positive if w in tl)
        neg = sum(1 for w in negative if w in tl)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    @staticmethod
    def build_payoff_matrix(user_a: str, user_b: str,
                             msgs_a: List[str], msgs_b: List[str]) -> Dict:
        """İkili kullanıcı için kazanım matrisi oluştur."""
        if not msgs_a or not msgs_b:
            return {"R":1,"S":0,"T":2,"P":0.5}
        sent_a = np.mean([GameTheoryAnalyzer.sentiment_score(m) for m in msgs_a])
        sent_b = np.mean([GameTheoryAnalyzer.sentiment_score(m) for m in msgs_b])
        # Basit kazanım modeli
        R = (sent_a + sent_b) / 2 + 1       # Karşılıklı işbirliği
        T = abs(sent_a - sent_b) + 1.5       # İhanet kazancı
        S = -abs(sent_a - sent_b) - 0.5     # İhanetin kurbanı
        P = 0.5                               # Karşılıklı ihanet
        return {"R": R, "S": S, "T": T, "P": P, "user_a": user_a, "user_b": user_b}

    @staticmethod
    def nash_score(payoff: Dict) -> float:
        """Nash Dengesi sapma skoru. 0 = tam denge."""
        R, T, S, P = payoff.get("R",1), payoff.get("T",2), payoff.get("S",0), payoff.get("P",0.5)
        # İşbirliği Nash Dengesi koşulu: R > P
        if R > P:
            return 0.0  # Denge var
        # Sapma: P > R → her iki oyuncu da ihanet eder
        return float(np.clip((P - R) / (abs(T - S) + 1e-9), 0, 1))

    @staticmethod
    def grim_trigger_detect(msgs: List[Dict]) -> Dict:
        """Grim Trigger davranış tespiti."""
        if len(msgs) < 4:
            return {"detected": False, "confidence": 0.0}
        sentiments = [GameTheoryAnalyzer.sentiment_score(m.get("raw_text",""))
                      for m in msgs]
        # Ani ve kalıcı duygu düşüşü arar
        for i in range(1, len(sentiments)-2):
            before_avg = np.mean(sentiments[:i])
            after_avg  = np.mean(sentiments[i:])
            drop = before_avg - after_avg
            if drop > 0.5:
                # Sonraki mesajlar kalıcı negatif mi?
                permanence = sum(1 for s in sentiments[i:] if s < 0) / len(sentiments[i:])
                if permanence > 0.7:
                    return {
                        "detected": True,
                        "trigger_position": i,
                        "confidence": min(1.0, drop * permanence),
                        "before_sentiment": float(before_avg),
                        "after_sentiment":  float(after_avg)
                    }
        return {"detected": False, "confidence": 0.0}

    @staticmethod
    def shapley_value(users: List[str], user_msgs: Dict[str,List[str]]) -> Dict[str,float]:
        """
        Shapley değeri: her kullanıcının konuşmaya katkısı.
        Koalisyon değeri = toplam token çeşitlilik skoru.
        """
        if not users:
            return {}
        n = len(users)
        shapley = {u: 0.0 for u in users}
        def coalition_value(subset: List[str]) -> float:
            if not subset:
                return 0.0
            all_tokens = set()
            for u in subset:
                for m in user_msgs.get(u, []):
                    all_tokens.update(TextParser.tokenize(m))
            return math.log1p(len(all_tokens))
        # Marginal contribution
        import itertools
        for i, user in enumerate(users):
            for r in range(n):
                for subset in itertools.combinations([u for u in users if u!=user], r):
                    subset_list = list(subset)
                    v_with    = coalition_value(subset_list + [user])
                    v_without = coalition_value(subset_list)
                    marginal  = v_with - v_without
                    # Ağırlık
                    weight = (math.factorial(r) * math.factorial(n-r-1)) / math.factorial(n)
                    shapley[user] += weight * marginal
        return shapley

    @staticmethod
    def folk_theorem_score(delta: float, T: float, R: float, P: float) -> float:
        """Folk Teoremi: işbirliğinin sürdürülebilirliği. 1 = sürdürülebilir."""
        required_delta = (T - R) / (T - P + 1e-9)
        if delta >= required_delta:
            return 1.0
        return float(delta / required_delta)

    @staticmethod
    def pagerank(adj: Dict[str,Dict[str,float]], d: float = 0.85, iters: int = 50) -> Dict[str,float]:
        """PageRank tabanlı etki skoru."""
        nodes = list(adj.keys())
        if not nodes:
            return {}
        n = len(nodes)
        pr = {node: 1.0/n for node in nodes}
        for _ in range(iters):
            new_pr = {}
            for node in nodes:
                incoming = sum(adj.get(src, {}).get(node, 0) * pr.get(src,0) / max(sum(adj.get(src,{}).values()),1)
                               for src in nodes if node in adj.get(src,{}))
                new_pr[node] = (1-d)/n + d*incoming
            pr = new_pr
        return pr


# ╔═══════════════════════════════════════════════════╗
# ║  10. RELATIONSHIP GRAPH  (GENİŞLETİLMİŞ - YENİ)  ║
# ╚═══════════════════════════════════════════════════╝
class RelationshipGraph:
    """
    Sonsuz ilişkisellik ağı çıkarıcı.
    MD dosyasında tanımlanmayan tüm ilişki türlerini otomatik keşfeder.
    """

    def __init__(self, nlp: NLPProcessor, stylometry: StylometryEngine,
                 markov: MarkovBayesEngine, identity: IdentityProfiler,
                 bot_detector: BotDetector):
        self.nlp        = nlp
        self.stylometry = stylometry
        self.markov     = markov
        self.identity   = identity
        self.bot        = bot_detector
        self.edges: List[Dict] = []

    def _add_edge(self, src: str, tgt: str, rel_type: str, weight: float, evidence: dict):
        if src == tgt or weight < 0.3:
            return
        self.edges.append({
            "rel_id":    str(uuid.uuid4()),
            "source_id": src,
            "target_id": tgt,
            "rel_type":  rel_type,
            "weight":    round(float(np.clip(weight, 0, 1)), 4),
            "evidence":  evidence,
        })

    def extract_all(self, messages: List[dict], session_id: str) -> List[Dict]:
        """Tüm olası ilişkileri çıkar – mesajlar ve kullanıcılar arası."""
        self.edges = []
        n = len(messages)
        if n < 2:
            return []

        # Kullanıcı gruplaması
        user_msgs: Dict[str, List[dict]] = defaultdict(list)
        for m in messages:
            user_msgs[m["username"]].append(m)

        # Embedding matrisini ön-hesapla
        embs = {}
        for m in messages:
            emb = self.nlp.embed(m["raw_text"])
            m["embedding"] = emb
            embs[m["msg_id"]] = emb

        # Stilometrik imzaları ön-hesapla
        sigs = {}
        user_sigs = {}
        for m in messages:
            sig = self.stylometry.feature_vector(m.get("tokens",[]), m.get("raw_text",""))
            m["features"] = sig
            sigs[m["msg_id"]] = sig
            user = m["username"]
            if user not in user_sigs:
                user_sigs[user] = sig.copy()
            else:
                for k,v in sig.items():
                    user_sigs[user][k] = (user_sigs[user].get(k,0) + v) / 2

        with ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
            futures = []
            # Mesaj çiftleri (O(n²))
            for i in range(n):
                for j in range(i+1, n):
                    ma, mb = messages[i], messages[j]
                    futures.append(executor.submit(
                        self._compute_msg_edges, ma, mb, embs, sigs
                    ))
            for fut in as_completed(futures):
                try:
                    for edge in fut.result():
                        self.edges.append(edge)
                except Exception as e:
                    log.debug(f"Edge hatası: {e}")

        # Kullanıcı-kullanıcı ilişkileri
        users = list(user_msgs.keys())
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                ua, ub = users[i], users[j]
                self._user_edges(ua, ub, user_msgs, user_sigs)

        # Bot koordinasyon ağı
        self._bot_coordination_edges(user_msgs, messages)

        # Alias zinciri
        self._alias_chain_edges(user_sigs, user_msgs)

        # Konu-mesaj-kullanıcı üçgen ilişkileri
        self._topic_triangle_edges(messages, user_msgs)

        return self.edges

    def _compute_msg_edges(self, ma: dict, mb: dict,
                            embs: Dict, sigs: Dict) -> List[Dict]:
        edges = []
        id_a = f"msg_{ma['msg_id']}"
        id_b = f"msg_{mb['msg_id']}"

        # 1. Anlamsal benzerlik
        ea, eb = embs.get(ma["msg_id"],[]), embs.get(mb["msg_id"],[])
        if ea and eb:
            sim = self.nlp.cosine_sim(ea, eb)
            if sim > 0.6:
                edges.append(self._mk_edge(id_a, id_b, "semantic_similar", sim,
                    {"cosine_similarity": sim, "method": "tfidf_cosine"}))

        # 2. Konu örtüşmesi
        tv_a = self.nlp.topic_vector(ma["raw_text"])
        tv_b = self.nlp.topic_vector(mb["raw_text"])
        if tv_a and tv_b:
            topic_sim = self.nlp.cosine_sim(tv_a, tv_b)
            if topic_sim > 0.65:
                edges.append(self._mk_edge(id_a, id_b, "topic_overlap", topic_sim,
                    {"topic_similarity": topic_sim}))

        # 3. Stilometrik ikiz (farklı kullanıcıdan)
        if ma["username"] != mb["username"]:
            sig_a = sigs.get(ma["msg_id"], {})
            sig_b = sigs.get(mb["msg_id"], {})
            delta = self.stylometry.burrows_delta(sig_a, sig_b)
            same_auth_prob = self.stylometry.same_author_probability(delta)
            if same_auth_prob > 0.7:
                edges.append(self._mk_edge(id_a, id_b, "stylometric_twin", same_auth_prob,
                    {"burrows_delta": delta, "same_author_prob": same_author_prob}))

        # 4. Zamansal yakınlık
        pos_a = ma.get("position", 0)
        pos_b = mb.get("position", 0)
        pos_dist = abs(pos_a - pos_b)
        if pos_dist <= 3:
            prox = 1.0 - pos_dist / 4.0
            edges.append(self._mk_edge(id_a, id_b, "temporal_proximate", prox,
                {"position_distance": pos_dist}))

        # 5. Sözcük örtüşmesi
        tok_a = set(ma.get("tokens", []))
        tok_b = set(mb.get("tokens", []))
        if tok_a and tok_b:
            jaccard = len(tok_a & tok_b) / len(tok_a | tok_b)
            if jaccard > 0.4:
                edges.append(self._mk_edge(id_a, id_b, "vocabulary_overlap", jaccard,
                    {"jaccard": jaccard, "shared_words": list(tok_a & tok_b)[:10]}))

        # 6. Duygu uyumu
        sent_a = GameTheoryAnalyzer.sentiment_score(ma.get("raw_text",""))
        sent_b = GameTheoryAnalyzer.sentiment_score(mb.get("raw_text",""))
        if ma["username"] != mb["username"]:
            sent_sim = 1 - abs(sent_a - sent_b) / 2
            if sent_sim > 0.85:
                edges.append(self._mk_edge(id_a, id_b, "sentiment_aligned", sent_sim,
                    {"sentiment_a": sent_a, "sentiment_b": sent_b}))

        # 7. Karakter n-gram parmakizi
        ng_a = ma.get("char_ngrams_3", {})
        ng_b = mb.get("char_ngrams_3", {})
        if ng_a and ng_b and ma["username"] != mb["username"]:
            common = set(ng_a) & set(ng_b)
            if common:
                total = set(ng_a) | set(ng_b)
                ng_sim = len(common) / len(total)
                if ng_sim > 0.6:
                    edges.append(self._mk_edge(id_a, id_b, "ngram_fingerprint", ng_sim,
                        {"ngram_jaccard": ng_sim, "shared_ngrams": list(common)[:8]}))

        # 8. Yeniden yazma (paraphrase) – farklı dil veya farklı sözcükler ama aynı anlam
        if ma["username"] != mb["username"]:
            lang_a = ma.get("language","unk")
            lang_b = mb.get("language","unk")
            if ea and eb and len(tok_a) > 3 and len(tok_b) > 3:
                deep_sim = self.nlp.cosine_sim(ea, eb)
                shallow_jac = len(tok_a&tok_b)/max(len(tok_a|tok_b),1)
                # Yüksek derin benzerlik, düşük yüzeysel → paraphrase
                if deep_sim > 0.7 and shallow_jac < 0.3:
                    paraphrase_score = deep_sim * (1 - shallow_jac)
                    edges.append(self._mk_edge(id_a, id_b, "paraphrase", paraphrase_score,
                        {"semantic_sim": deep_sim, "lexical_overlap": shallow_jac,
                         "cross_language": lang_a != lang_b}))

        return edges

    def _mk_edge(self, src, tgt, rtype, weight, evidence) -> Dict:
        return {
            "rel_id":    str(uuid.uuid4()),
            "source_id": src,
            "target_id": tgt,
            "rel_type":  rtype,
            "weight":    round(float(np.clip(weight, 0, 1)), 4),
            "evidence":  evidence,
        }

    def _user_edges(self, ua: str, ub: str,
                     user_msgs: Dict[str,List[dict]],
                     user_sigs: Dict[str,Dict]):
        """Kullanıcı-kullanıcı seviyesi ilişkiler."""
        msgs_a = [m["raw_text"] for m in user_msgs.get(ua,[])]
        msgs_b = [m["raw_text"] for m in user_msgs.get(ub,[])]

        # Stilometrik kullanıcı imzası karşılaştırma
        sig_a = user_sigs.get(ua, {})
        sig_b = user_sigs.get(ub, {})
        if sig_a and sig_b:
            delta = self.stylometry.burrows_delta(sig_a, sig_b)
            prob  = self.stylometry.same_author_probability(delta)
            if prob > 0.65:
                self._add_edge(f"user_{ua}", f"user_{ub}", "same_author", prob,
                    {"burrows_delta": delta, "same_author_probability": prob})

        # Markov perplexity çapraz testi
        tokens_b = []
        for m in user_msgs.get(ub, []):
            tokens_b.extend(m.get("tokens",[]))
        if tokens_b and ua in self.markov._user_models:
            pp = self.markov.perplexity(ua, tokens_b[:50])
            if pp < 50:  # Düşük perplexity → a modeli b'yi iyi açıklıyor
                alias_score = 1.0 / (1.0 + pp/20)
                self._add_edge(f"user_{ua}", f"user_{ub}", "alias_link", alias_score,
                    {"perplexity": pp, "method": "markov_crosstest"})

    def _bot_coordination_edges(self, user_msgs: Dict[str,List[dict]],
                                  messages: List[dict]):
        """Koordineli bot ağı kenar tespiti."""
        users = list(user_msgs.keys())
        if not SKLEARN_AVAILABLE or len(users) < 2:
            return
        try:
            user_texts = {u: " ".join(m["raw_text"] for m in user_msgs[u]) for u in users}
            vec = TfidfVectorizer(max_features=200, sublinear_tf=True)
            X = vec.fit_transform([user_texts[u] for u in users])
            sims = cosine_similarity(X)
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    s = float(sims[i,j])
                    if s > 0.55:
                        self._add_edge(
                            f"user_{users[i]}", f"user_{users[j]}",
                            "coordinated_bot", s,
                            {"content_similarity": s, "method": "tfidf_coordination"}
                        )
        except Exception:
            pass

    def _alias_chain_edges(self, user_sigs: Dict[str,Dict],
                             user_msgs: Dict[str,List[dict]]):
        """Kullanıcı adı benzerliği + içerik = alias zinciri."""
        users = list(user_sigs.keys())
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                ua, ub = users[i], users[j]
                # Kullanıcı adı Levenshtein benzerliği
                name_sim = self._name_similarity(ua, ub)
                # Kimlik vektör benzerliği
                id_vecs = {}
                for u in [ua, ub]:
                    all_tokens = []
                    for m in user_msgs.get(u, []):
                        all_tokens.extend(m.get("tokens",[]))
                    all_text = " ".join(m["raw_text"] for m in user_msgs.get(u,[]))
                    id_vecs[u] = self.identity.extract_identity_vector(all_tokens, all_text)
                jsd_val = self.nlp.jsd(
                    list(id_vecs[ua].values()), list(id_vecs[ub].values())
                )
                id_sim = 1.0 - jsd_val
                combined = name_sim * 0.3 + id_sim * 0.7
                if combined > 0.6:
                    self._add_edge(
                        f"user_{ua}", f"user_{ub}", "alias_link", combined,
                        {"name_similarity": name_sim, "identity_similarity": id_sim}
                    )

    def _topic_triangle_edges(self, messages: List[dict],
                               user_msgs: Dict[str,List[dict]]):
        """Konu-mesaj-kullanıcı üçgen ilişkileri: aynı konuyu tartışan farklı kullanıcılar."""
        topic_users: Dict[int, Set[str]] = defaultdict(set)
        for m in messages:
            tv = self.nlp.topic_vector(m["raw_text"])
            if not tv:
                continue
            dominant = int(np.argmax(tv))
            topic_users[dominant].add(m["username"])
        # Aynı konuda birden fazla kullanıcı → bağlantı
        for topic_id, users_in_topic in topic_users.items():
            ul = list(users_in_topic)
            for i in range(len(ul)):
                for j in range(i+1, len(ul)):
                    self._add_edge(
                        f"user_{ul[i]}", f"user_{ul[j]}",
                        "topic_overlap", 0.55,
                        {"topic_id": topic_id, "shared_topic": True}
                    )

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        """Basit Levenshtein tabanlı kullanıcı adı benzerliği."""
        if a == b:
            return 1.0
        m, n = len(a), len(b)
        dp = list(range(n+1))
        for i in range(1, m+1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n+1):
                cost = 0 if a[i-1]==b[j-1] else 1
                dp[j] = min(prev[j]+1, dp[j-1]+1, prev[j-1]+cost)
        return 1.0 - dp[n] / max(m, n)


# ╔══════════════════════════════════════════╗
# ║  11. ALIAS LINKER  (YENİ)                ║
# ╚══════════════════════════════════════════╝
class AliasLinker:
    """
    'Farklı kullanıcı adıyla yeniden yazma' tespiti.
    Birden fazla metot: stilometri + Markov + kimlik vektörü + n-gram parmakizi.
    """

    def __init__(self, stylometry: StylometryEngine, markov: MarkovBayesEngine,
                 identity: IdentityProfiler, nlp: NLPProcessor):
        self.sty = stylometry
        self.markov = markov
        self.identity = identity
        self.nlp = nlp

    def link_users(self, user_profiles: Dict[str, dict]) -> List[Dict]:
        """Tüm kullanıcı çiftlerini karşılaştır, alias grupları oluştur."""
        users = list(user_profiles.keys())
        links = []
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                ua, ub = users[i], users[j]
                score = self._compute_link_score(
                    user_profiles[ua], user_profiles[ub], ua, ub
                )
                if score["combined"] > 0.55:
                    links.append({
                        "user_a": ua,
                        "user_b": ub,
                        "combined_score": score["combined"],
                        "details": score,
                        "verdict": (
                            "Yüksek Olasılıklı Alias" if score["combined"] > 0.80 else
                            "Orta Olasılıklı Alias"    if score["combined"] > 0.65 else
                            "Düşük Olasılıklı Alias"
                        )
                    })
        return links

    def _compute_link_score(self, pa: dict, pb: dict, ua: str, ub: str) -> Dict:
        scores = {}
        # 1. Stilometrik Delta
        sig_a = pa.get("stylometric_sig", {})
        sig_b = pb.get("stylometric_sig", {})
        if sig_a and sig_b:
            delta = self.sty.burrows_delta(sig_a, sig_b)
            scores["stylometric"] = self.sty.same_author_probability(delta)
        else:
            scores["stylometric"] = 0.0

        # 2. Kimlik vektör kosinüs benzerliği
        iv_a = pa.get("identity_vector", {})
        iv_b = pb.get("identity_vector", {})
        if iv_a and iv_b:
            dims = IDENTITY_DIMS
            va = np.array([iv_a.get(d,0) for d in dims]) + 1e-9
            vb = np.array([iv_b.get(d,0) for d in dims]) + 1e-9
            va /= va.sum(); vb /= vb.sum()
            if SCIPY_AVAILABLE:
                scores["identity"] = 1.0 - float(jensenshannon(va, vb))
            else:
                scores["identity"] = float(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-9))
        else:
            scores["identity"] = 0.0

        # 3. Markov çapraz perplexity
        scores["markov"] = 0.0
        if ua in self.markov._user_models and pb.get("tokens"):
            pp = self.markov.perplexity(ua, pb.get("tokens",[])[:40])
            scores["markov"] = 1.0 / (1.0 + pp / 30)

        # 4. Kullanıcı adı benzerliği
        scores["username"] = RelationshipGraph._name_similarity(ua, ub)

        # Ağırlıklı birleştirme
        w = {"stylometric": 0.40, "identity": 0.30, "markov": 0.20, "username": 0.10}
        combined = sum(scores[k] * w[k] for k in w)
        scores["combined"] = float(np.clip(combined, 0, 1))
        return scores

    def assign_alias_groups(self, links: List[Dict]) -> Dict[str, int]:
        """Union-Find ile alias grupları oluştur."""
        parent: Dict[str, str] = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        for link in links:
            if link["combined_score"] > 0.65:
                union(link["user_a"], link["user_b"])
        groups: Dict[str, int] = {}
        group_ids: Dict[str, int] = {}
        gid = 0
        for user in parent:
            root = find(user)
            if root not in group_ids:
                group_ids[root] = gid
                gid += 1
            groups[user] = group_ids[root]
        return groups


# ╔═══════════════════════════════════════════════════╗
# ║  12. BAYESIAN PREDICTOR + OLLAMA  (YENİ)          ║
# ╚═══════════════════════════════════════════════════╝
class BayesianPredictor:
    """
    Ollama-NLP tabanlı kullanıcı & mesaj tahmin motoru.
    Sorular üzerinden kullanıcı profili + yazabilecekleri mesajlar.
    """

    def __init__(self, markov: MarkovBayesEngine, identity: IdentityProfiler,
                 ollama_url: str = OLLAMA, model: str = OLLAMA_MODEL):
        self.markov  = markov
        self.identity = identity
        self.ollama  = ollama_url
        self.model   = model
        self._ollama_available = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            r = http.get(f"{self.ollama}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _ollama_generate(self, prompt: str, max_tokens: int = 400) -> str:
        if not self._ollama_available:
            return "[Ollama bağlantısı yok – yerel tahmin kullanılıyor]"
        try:
            r = http.post(
                f"{self.ollama}/api/generate",
                json={"model": self.model, "prompt": prompt,
                      "stream": False, "options": {"num_predict": max_tokens}},
                timeout=30
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
        except Exception as e:
            return f"[Ollama hatası: {e}]"
        return "[Boş yanıt]"

    def predict_user_message(self, username: str, topic: str,
                              user_msgs: List[str],
                              user_profile: dict) -> Dict:
        """
        Verilen kullanıcı için konu hakkında yazabilecekleri mesajı tahmin et.
        İki yol: Markov üretimi + Ollama LLM.
        """
        # Markov tabanlı üretim
        all_tokens = []
        for m in user_msgs:
            all_tokens.extend(TextParser.tokenize(m))
        seed = TextParser.tokenize(topic)[:2]
        markov_gen = self.markov.generate_markov(username, seed, length=20)
        markov_text = " ".join(markov_gen)

        # Bayes yazar olasılığı
        topic_tokens = TextParser.tokenize(topic)
        author_post = self.markov.author_posterior(topic_tokens)
        user_prob = author_post.get(username, 0.0)

        # Kimlik profil özeti
        iv = user_profile.get("identity_vector", {})
        top_ids = sorted(iv.items(), key=lambda x:x[1], reverse=True)[:3]
        id_summary = ", ".join(f"{k}:{v:.2f}" for k,v in top_ids)

        # Ollama prompt
        bot_score = user_profile.get("bot_probability", 0.0)
        deception  = user_profile.get("deception_score", 0.0)
        example_msgs = user_msgs[:3] if user_msgs else []
        examples_str = "\n".join(f"  - {m}" for m in example_msgs)

        prompt = f"""Sen bir linguistik analiz asistanısın. Kullanıcı profilini inceleyerek tahmin yap.

KULLANICI PROFİLİ:
- Kullanıcı adı: {username}
- Kimlik eğilimleri: {id_summary}
- Bot olasılığı: {bot_score:.1%}
- Kimlik deception skoru: {deception:.1%}
- Mesaj sayısı: {len(user_msgs)}

ÖRNEK MESAJLAR:
{examples_str}

MARKOV ÜRETİMİ ({topic} konusunda): {markov_text}

GÖREV: '{topic}' konusunda bu kullanıcının nasıl bir mesaj yazabileceğini tahmin et.
Tahmini mesajı yaz (max 2 cümle), ardından kısa bir analiz ver.
Yanıt dili: Türkçe."""

        ollama_text = self._ollama_generate(prompt, max_tokens=300)

        return {
            "username":        username,
            "topic":           topic,
            "markov_generated": markov_text,
            "ollama_predicted": ollama_text,
            "author_probability": user_prob,
            "bayesian_scores":  author_post,
        }

    def predict_anonymous_identity(self, text: str, user_profiles: Dict[str,dict]) -> Dict:
        """
        Anonim bir metin verildiğinde: en olası yazarı ve kimlik profilini tahmin et.
        """
        tokens = TextParser.tokenize(text)
        # Markov tabanlı yazar posteriorı
        author_post = self.markov.author_posterior(tokens)
        # NB tabanlı tahmin
        nb_pred = self.markov.predict_author_nb(text)
        # Birleştirilmiş tahmin
        combined = {}
        all_users = set(list(author_post.keys()) + list(nb_pred.keys()))
        for u in all_users:
            a = author_post.get(u, 0.0)
            b = nb_pred.get(u, 0.0)
            combined[u] = (a + b) / 2

        best_user = max(combined, key=combined.get) if combined else "bilinmiyor"

        # Kimlik profil çıkarımı
        iv = self.identity.extract_identity_vector(tokens, text)
        top_ids = sorted(iv.items(), key=lambda x:x[1], reverse=True)[:4]

        # Ollama analizi
        prompt = f"""Anonimlik analizi yap. Aşağıdaki metin anonim bir kullanıcıya ait.

METİN: "{text}"

Kimlik ipuçları: {dict(top_ids)}

GÖREVLER:
1. Bu kişinin gerçek kimliği ne olabilir? (etnik/dini/ideolojik)
2. Bu metnin bir bot tarafından yazılmış olma ihtimali nedir?
3. Kısa analiz ver.

Yanıt dili: Türkçe. Max 150 kelime."""

        ollama_analysis = self._ollama_generate(prompt, max_tokens=200)

        return {
            "text":             text,
            "predicted_author": best_user,
            "author_scores":    dict(sorted(combined.items(), key=lambda x:x[1], reverse=True)[:5]),
            "identity_profile": dict(top_ids),
            "ollama_analysis":  ollama_analysis,
        }

    def answer_query(self, query: str, context: dict) -> str:
        """
        Serbest sorgu: kullanıcı herhangi bir soru sorar, Bayesian + Ollama yanıt üretir.
        """
        # Bağlamı özetle
        session_summary = f"""
Session ID: {context.get('session_id','?')}
Kullanıcı sayısı: {context.get('user_count',0)}
Mesaj sayısı: {context.get('msg_count',0)}
Kullanıcılar: {', '.join(context.get('usernames',[])[:10])}
Ortalama bot olasılığı: {context.get('avg_bot_prob',0):.1%}
Öne çıkan ilişki türleri: {', '.join(context.get('top_relations',[])[:5])}
"""
        prompt = f"""Sen bir gelişmiş kimlik analiz asistanısın.

OTURUM ÖZETİ:{session_summary}

SORU: {query}

Analitik ve Bayesyan akıl yürütmeyle yanıtla. Olasılıksal ifadeler kullan.
Yanıt dili: Türkçe. Max 200 kelime."""

        return self._ollama_generate(prompt, max_tokens=300)


# ╔═══════════════════════════════════╗
# ║  13. UNIVERSAL SEARCH ENGINE      ║
# ╚═══════════════════════════════════╝
class UniversalSearch:
    """
    Kullanıcı / mesaj / kelime grubu / konu / ilişki / sonsuz ağ araması.
    BFS tabanlı ağ genişleme + TF-IDF benzerlik araması.
    """

    def __init__(self, db: DatabaseManager, nlp: NLPProcessor):
        self.db  = db
        self.nlp = nlp

    def search(self, query: str, session_id: str,
               search_type: str = "all", limit: int = 20) -> Dict:
        """Ana arama fonksiyonu."""
        results = {
            "query": query,
            "session_id": session_id,
            "messages": [],
            "users": [],
            "relationships": [],
            "network_expansion": [],
        }
        messages = self.db.get_session_messages(session_id)
        users    = self.db.get_session_users(session_id)
        rels     = self.db.get_relationships(session_id)

        if search_type in ["all", "messages", "text", "keyword"]:
            results["messages"] = self._search_messages(query, messages, limit)

        if search_type in ["all", "users", "user"]:
            results["users"] = self._search_users(query, users, limit)

        if search_type in ["all", "relationships", "relation"]:
            results["relationships"] = self._search_relations(query, rels, limit)

        if search_type in ["all", "network", "expand"]:
            # BFS ağ genişleme
            seed_ids = [f"msg_{m['msg_id']}" for m in results["messages"][:3]]
            seed_ids += [f"user_{u['username']}" for u in results["users"][:3]]
            results["network_expansion"] = self._bfs_expand(seed_ids, rels, depth=2)

        return results

    def _search_messages(self, query: str, messages: List[dict], limit: int) -> List[dict]:
        if not messages:
            return []
        query_lower = query.lower()
        scored = []
        q_tokens = set(TextParser.tokenize(query))
        for m in messages:
            text = m.get("raw_text","")
            score = 0.0
            # Tam metin eşleşmesi
            if query_lower in text.lower():
                score += 1.0
            # Kullanıcı adı eşleşmesi
            if query_lower in m.get("username","").lower():
                score += 0.8
            # Token örtüşmesi
            m_tokens = set(TextParser.tokenize(text))
            if q_tokens and m_tokens:
                overlap = len(q_tokens & m_tokens) / len(q_tokens | m_tokens)
                score += overlap * 0.6
            # Embedding benzerliği
            m_emb = m.get("embedding", [])
            if m_emb and self.nlp._fitted:
                q_emb = self.nlp.embed(query)
                if q_emb:
                    sim = self.nlp.cosine_sim(q_emb, m_emb)
                    score += sim * 0.8
            if score > 0.1:
                scored.append({**m, "_score": round(score, 4)})
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:limit]

    def _search_users(self, query: str, users: List[dict], limit: int) -> List[dict]:
        query_lower = query.lower()
        scored = []
        for u in users:
            score = 0.0
            if query_lower in u.get("username","").lower():
                score += 1.0
            iv = u.get("identity_vector", {})
            for dim, val in (iv.items() if isinstance(iv, dict) else {}.items()):
                if query_lower in dim.lower():
                    score += val * 0.5
            if score > 0.05:
                scored.append({**u, "_score": round(score, 4)})
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:limit]

    def _search_relations(self, query: str, rels: List[dict], limit: int) -> List[dict]:
        query_lower = query.lower()
        scored = []
        for r in rels:
            score = 0.0
            if query_lower in r.get("rel_type","").lower():
                score += 1.0
            if query_lower in r.get("source_id","").lower():
                score += 0.6
            if query_lower in r.get("target_id","").lower():
                score += 0.6
            ev = r.get("evidence", {})
            if isinstance(ev, dict):
                for k,v in ev.items():
                    if query_lower in str(v).lower():
                        score += 0.3
            if score > 0.1:
                scored.append({**r, "_score": round(score, 4)})
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:limit]

    def _bfs_expand(self, seed_ids: List[str], rels: List[dict],
                     depth: int = 2) -> List[Dict]:
        """BFS ile ağ genişleme – seed node'lardan tüm bağlantılara ulaş."""
        visited = set(seed_ids)
        frontier = list(seed_ids)
        expansion = []
        for d in range(depth):
            next_frontier = []
            for node in frontier:
                neighbors = [
                    (r["target_id"] if r["source_id"]==node else r["source_id"],
                     r["rel_type"], r["weight"])
                    for r in rels
                    if r["source_id"]==node or r["target_id"]==node
                ]
                for neighbor, rtype, weight in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        expansion.append({
                            "node": neighbor,
                            "from": node,
                            "rel_type": rtype,
                            "weight": weight,
                            "depth": d+1,
                        })
            frontier = next_frontier
            if not frontier:
                break
        return expansion


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  14. ANALYSIS ORCHESTRATOR  (Tüm modülleri koordine eder)         ║
# ╚═══════════════════════════════════════════════════════════════════╝
class AnalysisOrchestrator:
    """Tam analiz hattını çalıştıran ana koordinatör."""

    def __init__(self, ollama_url: str = OLLAMA):
        self.db       = DatabaseManager()
        self.parser   = TextParser()
        self.nlp      = NLPProcessor()
        self.bot      = BotDetector(self.nlp)
        self.sty      = StylometryEngine()
        self.identity = IdentityProfiler()
        self.markov   = MarkovBayesEngine()
        self.gt       = GameTheoryAnalyzer()
        self.rel_graph= RelationshipGraph(self.nlp, self.sty, self.markov, self.identity, self.bot)
        self.alias    = AliasLinker(self.sty, self.markov, self.identity, self.nlp)
        self.predictor= BayesianPredictor(self.markov, self.identity, ollama_url)
        self.search   = UniversalSearch(self.db, self.nlp)
        self._lock    = threading.Lock()

    def process_text(self, raw_text: str, session_id: Optional[str] = None) -> Dict:
        """Ham metin → tam analiz → DB kayıt → sonuç."""
        if not session_id:
            session_id = str(uuid.uuid4())

        with self._lock:
            self.db.upsert_session(session_id)

        # 1. Parse
        messages = self.parser.parse(raw_text)
        if not messages:
            return {"error": "Mesaj bulunamadı. @kullanıcı_adi \"mesaj\" formatını kontrol edin."}

        # 2. NLP fit & embed
        all_texts = [m["raw_text"] for m in messages]
        self.nlp.fit(all_texts)
        for m in messages:
            m["embedding"]    = self.nlp.embed(m["raw_text"])
            m["topic_vector"] = self.nlp.topic_vector(m["raw_text"])

        # 3. Kullanıcı gruplaması
        user_msgs: Dict[str, List] = defaultdict(list)
        user_tokens: Dict[str, List] = defaultdict(list)
        user_texts_map: Dict[str, List[str]] = defaultdict(list)
        for m in messages:
            user = m["username"]
            user_msgs[user].append(m)
            user_tokens[user].extend(m.get("tokens",[]))
            user_texts_map[user].append(m["raw_text"])

        # 4. Markov eğitimi (CPU paralel)
        def train_markov(user):
            self.markov.train_user(user, user_tokens[user])
        with ThreadPoolExecutor(max_workers=CPU_WORKERS) as ex:
            list(ex.map(train_markov, user_msgs.keys()))
        self.markov.train_nb(user_texts_map)

        # 5. Kullanıcı profilleri
        user_profiles = {}
        for user in user_msgs:
            msgs_for_user = user_msgs[user]
            texts = [m["raw_text"] for m in msgs_for_user]
            tokens = user_tokens[user]
            timestamps = [m.get("timestamp_inferred", 0.0) for m in msgs_for_user]

            # Bot tespiti
            all_other_texts = {u: [mm["raw_text"] for mm in user_msgs[u]]
                               for u in user_msgs if u != user}
            bot_result = self.bot.compute_bot_probability(user, texts, timestamps, all_other_texts)

            # Stilometri
            all_sigs = [self.sty.feature_vector(m.get("tokens",[]), m.get("raw_text",""))
                        for m in msgs_for_user]
            avg_sig = {}
            if all_sigs:
                for key in all_sigs[0]:
                    avg_sig[key] = float(np.mean([s.get(key,0) for s in all_sigs]))

            # Kimlik vektörü
            all_id_vecs = [
                self.identity.extract_identity_vector(m.get("tokens",[]), m.get("raw_text",""))
                for m in msgs_for_user
            ]
            agg_id = self.identity.aggregate_identity(all_id_vecs)

            # Deception skoru (sahte kimlik performansı)
            # claimed = agg_id, observed = kendisi (basit: zaten agg, JSD kendi içinde)
            deception = self.identity.jsd_deception(agg_id, agg_id)  # 0 = tutarlı

            # Daha anlamlı deception: tüm kimlik boyutlarının entropisi
            iv_vals = list(agg_id.values())
            if any(v > 0 for v in iv_vals):
                iv_arr = np.array(iv_vals) + 1e-9
                iv_arr /= iv_arr.sum()
                ent = -float(np.sum(iv_arr * np.log(iv_arr + 1e-12)))
                # Yüksek entropi = belirsiz kimlik = potansiyel sahte
                deception = float(np.clip(ent / math.log(len(iv_vals)), 0, 1))

            # Anon skoru = kimlik belirsizliği
            anon_score = deception

            user_profiles[user] = {
                "username":         user,
                "msg_count":        len(msgs_for_user),
                "languages":        list(set(m.get("language","unk") for m in msgs_for_user)),
                "identity_vector":  agg_id,
                "stylometric_sig":  avg_sig,
                "bot_probability":  bot_result["bot_probability"],
                "bot_signals":      bot_result["signals"],
                "bot_verdict":      bot_result["verdict"],
                "deception_score":  deception,
                "anon_score":       anon_score,
                "tokens":           tokens[:200],
                "cluster_id":       -1,
                "alias_group":      [],
                "game_scores":      {},
            }

            # Mesajlara bireysel skorları aktar
            for m in msgs_for_user:
                m["bot_score"]       = bot_result["bot_probability"]
                m["anon_score"]      = anon_score
                m["deception_score"] = deception
                m["alias_score"]     = 0.0  # Sonra güncellenecek

        # 6. Stilometrik kümeleme
        all_sigs = {u: p["stylometric_sig"] for u,p in user_profiles.items()}
        clusters = self.sty.cluster_users(all_sigs)
        for u, cl in clusters.items():
            user_profiles[u]["cluster_id"] = cl

        # 7. Alias Linker
        alias_links = self.alias.link_users(user_profiles)
        alias_groups = self.alias.assign_alias_groups(alias_links)
        for u, gid in alias_groups.items():
            if u in user_profiles:
                user_profiles[u]["alias_group"] = [
                    ou for ou, og in alias_groups.items() if og == gid and ou != u
                ]
                # Alias skoru
                alias_score = max(
                    (lnk["combined_score"] for lnk in alias_links
                     if lnk["user_a"]==u or lnk["user_b"]==u),
                    default=0.0
                )
                user_profiles[u]["alias_score"] = alias_score
                for m in user_msgs.get(u,[]):
                    m["alias_score"] = alias_score

        # 8. Oyun Kuramı
        shapley = self.gt.shapley_value(list(user_msgs.keys()), user_texts_map)
        for u, sv in shapley.items():
            if u in user_profiles:
                user_profiles[u]["game_scores"]["shapley"] = sv

        # Grim Trigger
        gt_result = self.gt.grim_trigger_detect(messages)
        for u in user_profiles:
            user_profiles[u]["game_scores"]["grim_trigger"] = gt_result

        # PageRank
        adj = {}
        for m in messages:
            u = m["username"]
            adj.setdefault(u, {})
        for i in range(len(messages)-1):
            ua = messages[i]["username"]
            ub = messages[i+1]["username"]
            if ua != ub:
                adj.setdefault(ua, {})
                adj[ua][ub] = adj[ua].get(ub, 0) + 1
        pageranks = self.gt.pagerank(adj)
        for u, pr in pageranks.items():
            if u in user_profiles:
                user_profiles[u]["game_scores"]["pagerank"] = pr

        # 9. İlişki Grafiği
        relationships = self.rel_graph.extract_all(messages, session_id)

        # 10. DB kaydet
        def save_all():
            for m in messages:
                self.db.save_message(session_id, m)
            for u, p in user_profiles.items():
                self.db.save_user(session_id, p)
            for rel in relationships:
                self.db.save_relationship(session_id, rel)
            self.db.execute(
                "UPDATE sessions SET message_count=? WHERE session_id=?",
                (len(messages), session_id)
            )
        threading.Thread(target=save_all, daemon=True).start()

        # 11. Konu anahtar kelimeleri
        topic_keywords = self.nlp.get_topic_keywords()

        return {
            "session_id":    session_id,
            "message_count": len(messages),
            "user_count":    len(user_profiles),
            "messages":      messages,
            "user_profiles": user_profiles,
            "relationships": relationships,
            "alias_links":   alias_links,
            "topic_keywords":topic_keywords,
            "rel_summary":   Counter(r["rel_type"] for r in relationships).most_common(10),
            "ollama_available": self.predictor._ollama_available,
        }


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  15. HTML / JS TEMPLATE  (Tek Sayfa Arayüz – Swimlane + Popups)    ║
# ╚═════════════════════════════════════════════════════════════════════╝
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anonim Yazar Kimlik Çözümleme Sistemi v2.0</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
  :root{
    --bg:#0d1117;--panel:#161b22;--border:#30363d;--accent:#58a6ff;
    --green:#3fb950;--red:#f85149;--orange:#f0883e;--purple:#a371f7;
    --yellow:#e3b341;--text:#c9d1d9;--text2:#8b949e;--popup-bg:#1c2128;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
  header{background:var(--panel);border-bottom:1px solid var(--border);padding:10px 16px;display:flex;align-items:center;gap:12px;flex-shrink:0}
  header h1{font-size:1rem;font-weight:600;color:var(--accent)}
  header .badges span{background:var(--border);border-radius:4px;padding:2px 8px;font-size:.7rem;margin-right:4px;color:var(--text2)}
  .main{display:flex;flex:1;overflow:hidden}
  /* Left panel */
  .left-panel{width:280px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;overflow:hidden}
  .panel-section{padding:12px;border-bottom:1px solid var(--border)}
  .panel-section h3{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:8px}
  textarea#inputText{width:100%;height:120px;background:#0d1117;border:1px solid var(--border);color:var(--text);border-radius:6px;padding:8px;font-size:.8rem;resize:vertical;font-family:monospace}
  .btn{display:block;width:100%;padding:7px;background:var(--accent);color:#0d1117;border:none;border-radius:6px;cursor:pointer;font-weight:600;font-size:.82rem;margin-top:6px}
  .btn:hover{opacity:.85}
  .btn.danger{background:var(--red)}
  .btn.secondary{background:var(--border);color:var(--text)}
  input.search-input{width:100%;background:#0d1117;border:1px solid var(--border);color:var(--text);border-radius:6px;padding:6px 10px;font-size:.8rem;margin-bottom:6px}
  select.search-type{width:100%;background:#0d1117;border:1px solid var(--border);color:var(--text);border-radius:6px;padding:5px 8px;font-size:.8rem;margin-bottom:6px}
  .search-results{flex:1;overflow-y:auto;padding:0 12px 12px}
  .sr-item{background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:8px;margin-bottom:6px;cursor:pointer;font-size:.78rem;transition:.15s}
  .sr-item:hover{border-color:var(--accent)}
  .sr-item .sr-score{float:right;color:var(--yellow);font-size:.7rem}
  .sr-item .sr-user{color:var(--accent);font-weight:600}
  .sr-item .sr-text{color:var(--text2);margin-top:2px;word-break:break-word}
  /* Center: Swimlane */
  .center{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative}
  .swimlane-controls{padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .swimlane-controls label{font-size:.75rem;color:var(--text2)}
  .swimlane-controls input[type=range]{width:80px}
  .legend{display:flex;gap:10px;flex-wrap:wrap;margin-left:auto}
  .legend-item{display:flex;align-items:center;gap:4px;font-size:.68rem;color:var(--text2)}
  .legend-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
  #swimlane-svg{flex:1;background:#0a0e13}
  .swim-lane-bg{fill:#161b22;stroke:#21262d;stroke-width:1}
  .swim-lane-label{fill:#8b949e;font-size:12px;font-weight:600}
  .swim-node{cursor:pointer;transition:.15s}
  .swim-node:hover circle{stroke-width:3;stroke:#fff}
  .swim-edge{fill:none;stroke-width:1.5;opacity:.7}
  .swim-edge.highlighted{stroke-width:3;opacity:1}
  /* Right panel */
  .right-panel{width:300px;background:var(--panel);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
  .right-panel .tab-bar{display:flex;border-bottom:1px solid var(--border)}
  .tab-btn{flex:1;padding:7px;background:none;border:none;color:var(--text2);font-size:.75rem;cursor:pointer;border-bottom:2px solid transparent;transition:.15s}
  .tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
  .tab-content{display:none;flex:1;overflow-y:auto;padding:12px}
  .tab-content.active{display:block}
  .stat-row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;font-size:.78rem}
  .stat-label{color:var(--text2)}
  .stat-val{font-weight:600}
  .user-card{background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:10px;margin-bottom:8px;cursor:pointer}
  .user-card:hover{border-color:var(--accent)}
  .user-card .uc-name{font-weight:600;color:var(--accent);font-size:.85rem}
  .user-card .uc-badges{margin-top:4px;display:flex;gap:4px;flex-wrap:wrap}
  .badge{padding:2px 6px;border-radius:3px;font-size:.68rem;font-weight:600}
  .badge.bot{background:#5a1a1a;color:var(--red)}
  .badge.human{background:#0d2818;color:var(--green)}
  .badge.anon{background:#3a2a00;color:var(--yellow)}
  .badge.alias{background:#2a1a4a;color:var(--purple)}
  .prob-bar{margin:4px 0;display:flex;align-items:center;gap:6px;font-size:.72rem}
  .prob-bar .pb-label{width:70px;color:var(--text2);flex-shrink:0}
  .prob-bar .pb-track{flex:1;height:6px;background:#21262d;border-radius:3px;overflow:hidden}
  .prob-bar .pb-fill{height:100%;border-radius:3px;transition:.4s}
  .pb-fill.bot{background:var(--red)}
  .pb-fill.anon{background:var(--yellow)}
  .pb-fill.alias{background:var(--purple)}
  .pb-fill.deception{background:var(--orange)}
  .pb-val{width:36px;text-align:right;color:var(--text2)}
  /* Bayesian panel */
  .bayes-form textarea{width:100%;height:70px;background:#0d1117;border:1px solid var(--border);color:var(--text);border-radius:6px;padding:8px;font-size:.78rem;resize:vertical;font-family:inherit}
  .bayes-form select{width:100%;background:#0d1117;border:1px solid var(--border);color:var(--text);border-radius:6px;padding:5px 8px;font-size:.78rem;margin:4px 0}
  .bayes-result{background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:10px;margin-top:8px;font-size:.78rem;line-height:1.5;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow-y:auto}
  /* Popup / Modal */
  .modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:1000;align-items:center;justify-content:center}
  .modal-overlay.open{display:flex}
  .modal{background:var(--popup-bg);border:1px solid var(--border);border-radius:10px;max-width:680px;width:95%;max-height:85vh;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.5)}
  .modal-header{padding:14px 16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
  .modal-header h2{font-size:.95rem;font-weight:600}
  .modal-close{background:none;border:none;color:var(--text2);font-size:1.2rem;cursor:pointer;padding:2px 6px}
  .modal-close:hover{color:var(--text)}
  .modal-body{padding:16px;overflow-y:auto;flex:1}
  .modal-section{margin-bottom:16px}
  .modal-section h3{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text2);margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #21262d}
  .msg-text-block{background:#0d1117;border-left:3px solid var(--accent);padding:8px 12px;border-radius:0 6px 6px 0;font-size:.85rem;line-height:1.5;word-break:break-word}
  .related-list{display:flex;flex-direction:column;gap:5px;max-height:180px;overflow-y:auto}
  .related-item{background:#0d1117;border:1px solid var(--border);border-radius:5px;padding:6px 10px;font-size:.76rem;cursor:pointer}
  .related-item:hover{border-color:var(--accent)}
  .related-item .ri-type{color:var(--yellow);font-size:.68rem;margin-bottom:2px}
  .related-item .ri-text{color:var(--text2)}
  .id-radar{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
  .id-tag{padding:3px 8px;border-radius:4px;font-size:.72rem;font-weight:600}
  .spinner{border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;width:20px;height:20px;animation:spin .8s linear infinite;display:inline-block;margin:8px auto;display:block}
  @keyframes spin{to{transform:rotate(360deg)}}
  .toast{position:fixed;bottom:20px;right:20px;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 16px;font-size:.82rem;z-index:2000;animation:fadeIn .2s}
  @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  ::-webkit-scrollbar{width:5px;height:5px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
  .session-item{background:#0d1117;border:1px solid var(--border);border-radius:5px;padding:7px 10px;margin-bottom:5px;font-size:.76rem;cursor:pointer;display:flex;justify-content:space-between;align-items:center}
  .session-item:hover{border-color:var(--accent)}
  .session-del{background:var(--red);border:none;color:#fff;border-radius:3px;padding:2px 6px;cursor:pointer;font-size:.65rem}
  /* ── Upload area ── */
  .upload-zone{border:1.5px dashed var(--border);border-radius:6px;padding:10px 8px;text-align:center;cursor:pointer;transition:.2s;margin-bottom:6px;position:relative;background:#0a0e13}
  .upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);background:#0d1a2a}
  .upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .upload-zone .uz-icon{font-size:1.3rem;display:block;margin-bottom:2px}
  .upload-zone .uz-label{font-size:.72rem;color:var(--text2)}
  .upload-zone .uz-label b{color:var(--accent)}
  .upload-zone .uz-formats{font-size:.65rem;color:#555;margin-top:2px}
  .file-chip{background:#0d1117;border:1px solid var(--border);border-radius:4px;padding:3px 8px;font-size:.72rem;display:flex;align-items:center;gap:6px;margin-bottom:5px}
  .file-chip .fc-name{color:var(--accent);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .file-chip .fc-size{color:var(--text2);font-size:.65rem;white-space:nowrap}
  .file-chip .fc-rm{background:none;border:none;color:var(--text2);cursor:pointer;font-size:.8rem;padding:0 2px}
  .file-chip .fc-rm:hover{color:var(--red)}
  .file-parse-info{font-size:.68rem;color:var(--green);margin-bottom:4px;display:none}
</style>
</head>
<body>

<header>
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>
  <h1>Anonim Yazar Kimlik Çözümleme Sistemi v2.0</h1>
  <div class="badges">
    <span>🤖 Bot Tespit</span>
    <span>🕸️ İlişki Ağı</span>
    <span>🎭 Alias Linker</span>
    <span>🔮 Bayesyan Tahmin</span>
    <span id="ollama-badge" style="background:#1a3a1a;color:var(--green)">Ollama ⬤</span>
  </div>
  <div style="margin-left:auto;display:flex;gap:6px">
    <button id="reportBtn" onclick="downloadReport()" style="display:none;background:#1a3a1a;border:1px solid var(--green);color:var(--green);border-radius:6px;padding:4px 12px;cursor:pointer;font-size:.8rem;font-weight:600">📋 Rapor Al</button>
  </div>
</header>

<div class="main">
  <!-- LEFT: Input + Search -->
  <div class="left-panel">
    <div class="panel-section">
      <h3>📄 Metin Girişi</h3>

      <!-- Dosya Yükleme Alanı -->
      <div class="upload-zone" id="uploadZone"
           ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event)">
        <input type="file" id="fileInput" accept=".txt,.md,.json"
               onchange="handleFileSelect(event)">
        <span class="uz-icon">📂</span>
        <div class="uz-label">Dosya seç veya sürükle-bırak</div>
        <div class="uz-formats">.txt &nbsp;·&nbsp; .md &nbsp;·&nbsp; .json</div>
      </div>

      <!-- Seçili dosya bilgisi -->
      <div id="fileChip" style="display:none" class="file-chip">
        <span class="fc-name" id="fcName">—</span>
        <span class="fc-size" id="fcSize"></span>
        <button class="fc-rm" onclick="clearFile()" title="Dosyayı temizle">✕</button>
      </div>
      <div class="file-parse-info" id="fileParseInfo"></div>

      <textarea id="inputText" placeholder="@kullanıcı_adi1 &quot;mesaj içeriği&quot;&#10;@kullanıcı_adi2 &quot;başka bir mesaj&quot;&#10;2024-01-15 14:30 @user3 &quot;zaman damgalı&quot;"></textarea>
      <button class="btn" onclick="analyzeText()">▶ Analiz Et</button>
      <button class="btn secondary" onclick="loadExample()">📋 Örnek Yükle</button>
    </div>
    <div class="panel-section">
      <h3>🔍 Arama</h3>
      <input class="search-input" id="searchQ" placeholder="Kullanıcı, mesaj, kelime, ilişki..." oninput="doSearch()">
      <select class="search-type" id="searchType" onchange="doSearch()">
        <option value="all">Tümü</option>
        <option value="messages">Mesajlar</option>
        <option value="users">Kullanıcılar</option>
        <option value="relationships">İlişkiler</option>
        <option value="network">Ağ Genişleme</option>
      </select>
    </div>
    <div class="search-results" id="searchResults">
      <p style="font-size:.75rem;color:var(--text2);text-align:center;margin-top:20px">Arama yapmak için yukarıya yazın</p>
    </div>
  </div>

  <!-- CENTER: Swimlane -->
  <div class="center">
    <div class="swimlane-controls">
      <label>Kenar Türü: <select id="edgeTypeFilter" onchange="filterEdges()" style="background:#0d1117;border:1px solid var(--border);color:var(--text);border-radius:4px;padding:2px 6px;font-size:.74rem">
        <option value="all">Tümü</option>
      </select></label>
      <label>Eşik: <input type="range" id="weightThreshold" min="0" max="100" value="30" oninput="filterEdges()"> <span id="thresholdVal">0.30</span></label>
      <label><input type="checkbox" id="showBotOnly" onchange="filterEdges()"> Yalnız Bot</label>
      <label><input type="checkbox" id="showAliasOnly" onchange="filterEdges()"> Yalnız Alias</label>
      <button class="btn secondary" style="width:auto;padding:4px 10px;font-size:.74rem" onclick="resetZoom()">⟲ Sıfırla</button>
      <div class="legend" id="legend"></div>
    </div>
    <svg id="swimlane-svg"></svg>
  </div>

  <!-- RIGHT: Tabs -->
  <div class="right-panel">
    <div class="tab-bar">
      <button class="tab-btn active" onclick="showTab('stats')">📊 İstat.</button>
      <button class="tab-btn" onclick="showTab('users')">👤 Kullanıcı</button>
      <button class="tab-btn" onclick="showTab('bayes')">🔮 Tahmin</button>
      <button class="tab-btn" onclick="showTab('sessions')">🗃️ Oturum</button>
    </div>
    <div class="tab-content active" id="tab-stats">
      <div id="stats-content">
        <p style="color:var(--text2);font-size:.78rem;text-align:center;margin-top:30px">Analiz sonuçları burada görünecek</p>
      </div>
    </div>
    <div class="tab-content" id="tab-users">
      <div id="users-content"></div>
    </div>
    <div class="tab-content" id="tab-bayes">
      <h3 style="font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text2);margin-bottom:10px">🔮 Bayesyan Sorgu</h3>
      <div class="bayes-form">
        <textarea id="bayesQuery" placeholder="Örn: Kullanıcı1 bot mu? / user2 bu konuda ne yazar? / Bu metin kime ait?"></textarea>
        <select id="bayesUser"><option value="">Kullanıcı seç (opsiyonel)</option></select>
        <input class="search-input" id="bayesTopic" placeholder="Konu (opsiyonel)" style="margin-top:4px">
        <button class="btn" style="margin-top:4px" onclick="runBayesQuery()">🔮 Tahmin Üret</button>
        <button class="btn secondary" style="margin-top:4px" onclick="runAnonPredict()">🎭 Anonim Kimlik Tahmini</button>
      </div>
      <div class="bayes-result" id="bayesResult" style="display:none"></div>
    </div>
    <div class="tab-content" id="tab-sessions">
      <h3 style="font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text2);margin-bottom:10px">🗃️ Kayıtlı Oturumlar</h3>
      <button class="btn secondary" style="margin-bottom:8px" onclick="loadSessions()">↻ Yenile</button>
      <div id="sessions-list"></div>
    </div>
  </div>
</div>

<!-- Popup Modal -->
<div class="modal-overlay" id="nodeModal">
  <div class="modal">
    <div class="modal-header">
      <h2 id="modalTitle">Düğüm Detayı</h2>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════
//  GLOBAL STATE
// ═══════════════════════════════════════════════════════
let STATE = {
  session_id: null,
  messages: [],
  user_profiles: {},
  relationships: [],
  alias_links: [],
  topic_keywords: [],
  svg: null,
  zoom: null,
  nodes: [],
  edges: [],
  laneMap: {},       // username → lane index
  laneHeight: 90,
  nodeRadius: 18,
  searchTimeout: null,
};

const COLORS = {
  same_author:      '#e74c3c',
  semantic_similar: '#3498db',
  stylometric_twin: '#9b59b6',
  topic_overlap:    '#2ecc71',
  temporal_proximate:'#f39c12',
  coordinated_bot:  '#e67e22',
  identity_mirror:  '#1abc9c',
  vocabulary_overlap:'#34495e',
  sentiment_aligned:'#e91e63',
  reply_chain:      '#607d8b',
  paraphrase:       '#ff5722',
  alias_link:       '#8bc34a',
  cross_lang_same:  '#00bcd4',
  game_theory_pair: '#795548',
  ngram_fingerprint:'#ff9800',
  error_pattern:    '#f44336',
};

// ═══════════════════════════════════════════════════════
//  API HELPERS
// ═══════════════════════════════════════════════════════
async function api(path, data={}, method='POST'){
  const opts = { method, headers:{'Content-Type':'application/json'} };
  if(method!='GET') opts.body = JSON.stringify(data);
  const r = await fetch('/api'+path, opts);
  if(!r.ok){ const e=await r.text(); throw new Error(e); }
  return r.json();
}

function toast(msg, color='var(--accent)'){
  const el = document.createElement('div');
  el.className = 'toast';
  el.style.borderLeftColor = color;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(()=>el.remove(), 3000);
}

function spinner(containerId){
  const div = document.createElement('div');
  div.className='spinner';
  document.getElementById(containerId).appendChild(div);
  return div;
}

// ═══════════════════════════════════════════════════════
//  ANALYSIS
// ═══════════════════════════════════════════════════════
async function analyzeText(){
  const text = document.getElementById('inputText').value.trim();
  if(!text){ toast('Metin giriniz!','var(--red)'); return; }
  const btn = document.querySelector('.btn');
  btn.textContent='⏳ Analiz ediliyor...'; btn.disabled=true;
  try{
    const result = await api('/analyze', { text, session_id: STATE.session_id });
    if(result.error){ toast(result.error,'var(--red)'); return; }
    STATE.session_id    = result.session_id;
    STATE.messages      = result.messages || [];
    STATE.user_profiles = result.user_profiles || {};
    STATE.relationships = result.relationships || [];
    STATE.alias_links   = result.alias_links || [];
    STATE.topic_keywords= result.topic_keywords || [];
    document.getElementById('ollama-badge').textContent =
      result.ollama_available ? 'Ollama ✓' : 'Ollama ✗';
    document.getElementById('ollama-badge').style.color =
      result.ollama_available ? 'var(--green)' : 'var(--red)';
    renderAll(result);
    toast(`✓ ${result.message_count} mesaj, ${result.user_count} kullanıcı analiz edildi`,'var(--green)');
  }catch(e){
    toast('Hata: '+e.message,'var(--red)');
  }finally{
    btn.textContent='▶ Analiz Et'; btn.disabled=false;
  }
}

function renderAll(result){
  renderStats(result);
  renderUsers(result);
  renderSwimlane(result);
  populateBayesUser(result);
  populateEdgeFilter(result);
  document.getElementById('reportBtn').style.display='inline-block';
}

// ═══════════════════════════════════════════════════════
//  REPORT DOWNLOAD
// ═══════════════════════════════════════════════════════
async function downloadReport(){
  if(!STATE.session_id){ toast('Önce analiz yapın','var(--red)'); return; }
  const btn = document.getElementById('reportBtn');
  btn.textContent='⏳ Hazırlanıyor...'; btn.disabled=true;
  try{
    const r = await fetch('/api/report?session_id='+STATE.session_id);
    if(!r.ok){ throw new Error(await r.text()); }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href=url; a.download=`rapor_${STATE.session_id.slice(0,8)}.html`;
    a.click(); URL.revokeObjectURL(url);
    toast('📋 Rapor indirildi','var(--green)');
  }catch(e){ toast('Rapor hatası: '+e.message,'var(--red)'); }
  finally{ btn.textContent='📋 Rapor Al'; btn.disabled=false; }
}

// ═══════════════════════════════════════════════════════
//  STATS PANEL
// ═══════════════════════════════════════════════════════
function renderStats(r){
  const users = Object.values(r.user_profiles||{});
  const avgBot = users.length ? (users.reduce((a,u)=>a+(u.bot_probability||0),0)/users.length) : 0;
  const avgAnon = users.length ? (users.reduce((a,u)=>a+(u.anon_score||0),0)/users.length) : 0;
  const botUsers = users.filter(u=>(u.bot_probability||0)>0.5).length;
  const aliasGroups = new Set(users.flatMap(u=>(u.alias_group||[]))).size;
  const relTypes = {};
  (r.relationships||[]).forEach(rel=>{ relTypes[rel.rel_type]=(relTypes[rel.rel_type]||0)+1; });
  const topRels = Object.entries(relTypes).sort((a,b)=>b[1]-a[1]).slice(0,6);

  let html = `
  <div class="stat-row"><span class="stat-label">Oturum ID</span><span class="stat-val" style="font-size:.65rem;color:var(--text2)">${r.session_id.slice(0,12)}…</span></div>
  <div class="stat-row"><span class="stat-label">Toplam Mesaj</span><span class="stat-val">${r.message_count}</span></div>
  <div class="stat-row"><span class="stat-label">Kullanıcı Sayısı</span><span class="stat-val">${r.user_count}</span></div>
  <div class="stat-row"><span class="stat-label">Bot Şüphelisi</span><span class="stat-val" style="color:var(--red)">${botUsers}</span></div>
  <div class="stat-row"><span class="stat-label">Ort. Bot Olasılığı</span><span class="stat-val" style="color:var(--orange)">${(avgBot*100).toFixed(1)}%</span></div>
  <div class="stat-row"><span class="stat-label">Ort. Anonimlik</span><span class="stat-val" style="color:var(--yellow)">${(avgAnon*100).toFixed(1)}%</span></div>
  <div class="stat-row"><span class="stat-label">Alias Bağlantısı</span><span class="stat-val" style="color:var(--purple)">${r.alias_links?.length||0}</span></div>
  <div class="stat-row"><span class="stat-label">İlişki Kenarı</span><span class="stat-val">${r.relationships?.length||0}</span></div>
  <div class="stat-row"><span class="stat-label">Konu Sayısı</span><span class="stat-val">${r.topic_keywords?.length||0}</span></div>
  <h3 style="font-size:.72rem;text-transform:uppercase;color:var(--text2);margin:12px 0 6px">Öne Çıkan İlişki Türleri</h3>`;
  topRels.forEach(([type,cnt])=>{
    html+=`<div class="stat-row"><span class="stat-label" style="display:flex;align-items:center;gap:5px">
      <span style="width:8px;height:8px;border-radius:50%;background:${COLORS[type]||'#888'};display:inline-block"></span>
      ${type}</span><span class="stat-val">${cnt}</span></div>`;
  });
  if(r.topic_keywords?.length){
    html+=`<h3 style="font-size:.72rem;text-transform:uppercase;color:var(--text2);margin:12px 0 6px">Konu Anahtar Kelimeleri</h3>`;
    r.topic_keywords.slice(0,5).forEach((kws,i)=>{
      html+=`<div style="margin-bottom:5px;font-size:.72rem"><span style="color:var(--accent)">Konu ${i+1}:</span> <span style="color:var(--text2)">${kws.slice(0,5).join(', ')}</span></div>`;
    });
  }
  document.getElementById('stats-content').innerHTML = html;
}

// ═══════════════════════════════════════════════════════
//  USERS PANEL
// ═══════════════════════════════════════════════════════
function renderUsers(r){
  const users = Object.values(r.user_profiles||{});
  if(!users.length){ document.getElementById('users-content').innerHTML='<p style="color:var(--text2);font-size:.78rem">Kullanıcı bulunamadı</p>'; return; }
  let html = '';
  users.sort((a,b)=>(b.bot_probability||0)-(a.bot_probability||0));
  users.forEach(u=>{
    const bot = u.bot_probability||0;
    const anon = u.anon_score||0;
    const alias = (u.alias_group||[]).length>0;
    const dec = u.deception_score||0;
    const top_ids = Object.entries(u.identity_vector||{}).filter(([,v])=>v>0.1).sort((a,b)=>b[1]-a[1]).slice(0,3);
    html+=`<div class="user-card" onclick="showUserModal('${u.username}')">
      <div class="uc-name">@${u.username} <span style="font-size:.65rem;color:var(--text2);font-weight:400">(${u.msg_count} mesaj)</span></div>
      <div class="uc-badges">
        <span class="badge ${bot>0.5?'bot':'human'}">${bot>0.5?'🤖 BOT':'👤 İnsan'} ${(bot*100).toFixed(0)}%</span>
        ${anon>0.6?`<span class="badge anon">🎭 Anonim ${(anon*100).toFixed(0)}%</span>`:''}
        ${alias?`<span class="badge alias">🔗 Alias</span>`:''}
        ${dec>0.7?`<span class="badge" style="background:#2a1500;color:var(--orange)">⚠️ Deception</span>`:''}
      </div>
      <div class="prob-bar" style="margin-top:6px">
        <span class="pb-label">Bot</span>
        <div class="pb-track"><div class="pb-fill bot" style="width:${bot*100}%"></div></div>
        <span class="pb-val">${(bot*100).toFixed(0)}%</span>
      </div>
      <div class="prob-bar">
        <span class="pb-label">Anonimlik</span>
        <div class="pb-track"><div class="pb-fill anon" style="width:${anon*100}%"></div></div>
        <span class="pb-val">${(anon*100).toFixed(0)}%</span>
      </div>
      <div class="prob-bar">
        <span class="pb-label">Deception</span>
        <div class="pb-track"><div class="pb-fill deception" style="width:${dec*100}%"></div></div>
        <span class="pb-val">${(dec*100).toFixed(0)}%</span>
      </div>
      ${top_ids.length?`<div class="id-radar">${top_ids.map(([k,v])=>`<span class="id-tag" style="background:#0d1117;border:1px solid #30363d;color:var(--text)">${k} <b style="color:var(--accent)">${(v*100).toFixed(0)}%</b></span>`).join('')}</div>`:''}
    </div>`;
  });
  document.getElementById('users-content').innerHTML = html;
}

// ═══════════════════════════════════════════════════════
//  SWIMLANE NETWORK VISUALIZATION (D3.js)
// ═══════════════════════════════════════════════════════
function renderSwimlane(r){
  const msgs = r.messages||[];
  const profs = r.user_profiles||{};
  const rels = r.relationships||[];
  if(!msgs.length) return;

  const svg = d3.select('#swimlane-svg');
  svg.selectAll('*').remove();

  const W = document.getElementById('swimlane-svg').clientWidth || 800;
  const users = [...new Set(msgs.map(m=>m.username))];
  const LH = Math.max(80, Math.min(130, (window.innerHeight-160)/Math.max(users.length,1)));
  STATE.laneHeight = LH;
  const H = Math.max(400, users.length * LH + 40);
  svg.attr('viewBox',`0 0 ${W} ${H}`);

  STATE.laneMap = {};
  users.forEach((u,i)=>{ STATE.laneMap[u]=i; });

  const zoom = d3.zoom().scaleExtent([0.3,4]).on('zoom',e=>g.attr('transform',e.transform));
  STATE.zoom = zoom;
  svg.call(zoom);

  const g = svg.append('g').attr('id','main-g');

  // Lane backgrounds
  users.forEach((u,i)=>{
    const y = i*LH + 20;
    g.append('rect').attr('class','swim-lane-bg')
      .attr('x',0).attr('y',y).attr('width',W).attr('height',LH-4).attr('rx',4);
    g.append('text').attr('class','swim-lane-label')
      .attr('x',8).attr('y',y+20).text('@'+u);
    // Bot indicator stripe
    const bp = profs[u]?.bot_probability||0;
    if(bp>0.4){
      g.append('rect').attr('x',0).attr('y',y).attr('width',4).attr('height',LH-4)
        .attr('fill', bp>0.7?'var(--red)':'var(--orange)').attr('rx',2);
    }
  });

  // Node positions (evenly spaced by position in flow)
  const N = msgs.length;
  const xScale = d3.scaleLinear().domain([0,Math.max(N-1,1)]).range([80, W-40]);
  const nodeData = msgs.map((m,idx)=>{
    const lane = STATE.laneMap[m.username]||0;
    return {
      id: `msg_${m.msg_id}`,
      x: xScale(m.position-1),
      y: lane*LH + LH/2 + 20,
      r: STATE.nodeRadius,
      msg: m,
      user: m.username,
      bot_score: m.bot_score||0,
      anon_score: m.anon_score||0,
      alias_score: m.alias_score||0,
    };
  });
  STATE.nodes = nodeData;
  const nodeMap = {};
  nodeData.forEach(n=>{ nodeMap[n.id]=n; });

  // Edges
  const edgeData = rels.map(rel=>({
    ...rel,
    source: nodeMap[rel.source_id]||null,
    target: nodeMap[rel.target_id]||null,
  })).filter(e=>e.source&&e.target);
  STATE.edges = edgeData;

  // Draw edges (behind nodes)
  const edgeGroup = g.append('g').attr('id','edge-group');
  edgeData.forEach(e=>{
    const sx=e.source.x, sy=e.source.y, tx=e.target.x, ty=e.target.y;
    const mx=(sx+tx)/2, my=Math.min(sy,ty)-30;
    edgeGroup.append('path')
      .attr('class','swim-edge')
      .attr('data-type',e.rel_type)
      .attr('data-weight',e.weight)
      .attr('data-rel-id',e.rel_id)
      .attr('d',`M${sx},${sy} Q${mx},${my} ${tx},${ty}`)
      .attr('stroke', COLORS[e.rel_type]||'#555')
      .attr('stroke-dasharray', e.rel_type.includes('bot')||e.rel_type.includes('alias')?'6,3':'none')
      .attr('stroke-width', Math.max(1, e.weight*3))
      .attr('opacity', 0.65)
      .on('click',()=>showEdgeModal(e));
  });

  // Draw nodes
  const nodeGroup = g.append('g').attr('id','node-group');
  nodeData.forEach(n=>{
    const grp = nodeGroup.append('g').attr('class','swim-node')
      .attr('transform',`translate(${n.x},${n.y})`)
      .on('click',()=>showNodeModal(n));

    // Outer glow for high bot score
    if(n.bot_score>0.6){
      grp.append('circle').attr('r',n.r+6)
        .attr('fill','none').attr('stroke','var(--red)').attr('stroke-width',1.5)
        .attr('stroke-dasharray','3,3').attr('opacity',0.5);
    }
    if(n.alias_score>0.6){
      grp.append('circle').attr('r',n.r+3)
        .attr('fill','none').attr('stroke','var(--purple)').attr('stroke-width',1.5).attr('opacity',0.6);
    }

    // Main circle – color = mix of bot/anon score
    const fillColor = nodeFillColor(n);
    grp.append('circle').attr('r',n.r)
      .attr('fill',fillColor)
      .attr('stroke','#21262d').attr('stroke-width',1.5);

    // Icon
    grp.append('text').attr('text-anchor','middle').attr('dominant-baseline','middle')
      .attr('font-size','10px').attr('fill','#fff').attr('pointer-events','none')
      .text(n.bot_score>0.7?'🤖':n.anon_score>0.6?'🎭':n.alias_score>0.6?'🔗':'💬');

    // Msg id label
    grp.append('text').attr('text-anchor','middle').attr('y',n.r+12)
      .attr('font-size','9px').attr('fill','var(--text2)').attr('pointer-events','none')
      .text('#'+n.msg.msg_id);

    // Tooltip on hover
    grp.append('title').text(`@${n.user}: ${n.msg.raw_text.slice(0,60)}…`);
  });

  buildLegend();
}

function nodeFillColor(n){
  if(n.bot_score>0.7) return '#5a1020';
  if(n.bot_score>0.5) return '#4a2010';
  if(n.alias_score>0.7) return '#2a1040';
  if(n.anon_score>0.6) return '#3a2a00';
  return '#1a2540';
}

function buildLegend(){
  const items = [
    {color:'#5a1020', label:'Bot >70%'},
    {color:'#4a2010', label:'Bot 50-70%'},
    {color:'#2a1040', label:'Alias'},
    {color:'#1a2540', label:'Normal'},
  ];
  const leg = document.getElementById('legend');
  leg.innerHTML = items.map(it=>`
    <div class="legend-item">
      <div class="legend-dot" style="background:${it.color}"></div>${it.label}
    </div>`).join('');
}

function populateEdgeFilter(r){
  const types = [...new Set((r.relationships||[]).map(r=>r.rel_type))];
  const sel = document.getElementById('edgeTypeFilter');
  sel.innerHTML='<option value="all">Tümü</option>'+
    types.map(t=>`<option value="${t}">${t}</option>`).join('');
}

function filterEdges(){
  const type = document.getElementById('edgeTypeFilter').value;
  const threshold = document.getElementById('weightThreshold').value/100;
  document.getElementById('thresholdVal').textContent = threshold.toFixed(2);
  const botOnly = document.getElementById('showBotOnly').checked;
  const aliasOnly = document.getElementById('showAliasOnly').checked;

  d3.selectAll('.swim-edge').each(function(){
    const el = d3.select(this);
    const t = el.attr('data-type');
    const w = parseFloat(el.attr('data-weight')||0);
    let show = true;
    if(type!='all' && t!==type) show=false;
    if(w < threshold) show=false;
    if(botOnly && !t.includes('bot') && !t.includes('coord')) show=false;
    if(aliasOnly && t!=='alias_link') show=false;
    el.style('display', show?null:'none');
  });
}

function resetZoom(){
  const svg = d3.select('#swimlane-svg');
  if(STATE.zoom) svg.transition().duration(400).call(STATE.zoom.transform, d3.zoomIdentity);
}

// ═══════════════════════════════════════════════════════
//  MODAL POPUPS
// ═══════════════════════════════════════════════════════
function showNodeModal(n){
  const m = n.msg;
  const prof = STATE.user_profiles[m.username]||{};
  const myRels = STATE.relationships.filter(r=>r.source_id===n.id||r.target_id===n.id);
  const relatedNodeIds = new Set(myRels.flatMap(r=>[r.source_id,r.target_id]).filter(id=>id!==n.id));
  const relatedMsgs = STATE.messages.filter(mm=>`msg_${mm.msg_id}`!==n.id &&
    relatedNodeIds.has(`msg_${mm.msg_id}`));

  const botSig = prof.bot_signals||{};
  const botSigHtml = Object.entries(botSig).map(([k,v])=>
    `<div class="prob-bar"><span class="pb-label" style="width:120px">${k}</span>
     <div class="pb-track"><div class="pb-fill bot" style="width:${v*100}%"></div></div>
     <span class="pb-val">${(v*100).toFixed(0)}%</span></div>`).join('');

  const aliasGroup = prof.alias_group||[];
  const gameScores = prof.game_scores||{};

  document.getElementById('modalTitle').innerHTML =
    `<span style="color:var(--accent)">@${m.username}</span> – Mesaj #${m.msg_id}`;

  document.getElementById('modalBody').innerHTML = `
    <div class="modal-section">
      <h3>📝 Mesaj İçeriği</h3>
      <div class="msg-text-block">${escHtml(m.raw_text)}</div>
      <div style="margin-top:6px;font-size:.72rem;color:var(--text2)">
        🌐 Dil: <b>${m.language||'?'}</b> &nbsp;|&nbsp;
        📊 Konum: <b>#${m.position}</b> &nbsp;|&nbsp;
        ✍️ Alfabe: <b>${m.script||'?'}</b>
      </div>
    </div>

    <div class="modal-section">
      <h3>🔬 Analiz Skorları</h3>
      <div class="prob-bar"><span class="pb-label">Bot Olasılığı</span>
        <div class="pb-track"><div class="pb-fill bot" style="width:${(m.bot_score||0)*100}%"></div></div>
        <span class="pb-val">${((m.bot_score||0)*100).toFixed(0)}%</span></div>
      <div class="prob-bar"><span class="pb-label">Anonimlik</span>
        <div class="pb-track"><div class="pb-fill anon" style="width:${(m.anon_score||0)*100}%"></div></div>
        <span class="pb-val">${((m.anon_score||0)*100).toFixed(0)}%</span></div>
      <div class="prob-bar"><span class="pb-label">Deception</span>
        <div class="pb-track"><div class="pb-fill deception" style="width:${(m.deception_score||0)*100}%"></div></div>
        <span class="pb-val">${((m.deception_score||0)*100).toFixed(0)}%</span></div>
      <div class="prob-bar"><span class="pb-label">Alias</span>
        <div class="pb-track"><div class="pb-fill alias" style="width:${(m.alias_score||0)*100}%"></div></div>
        <span class="pb-val">${((m.alias_score||0)*100).toFixed(0)}%</span></div>
    </div>

    <div class="modal-section">
      <h3>🤖 Bot Sinyal Detayı <span style="color:var(--text2);font-weight:400">(${prof.bot_verdict||'?'})</span></h3>
      ${botSigHtml||'<p style="color:var(--text2);font-size:.75rem">Sinyal verisi yok</p>'}
    </div>

    <div class="modal-section">
      <h3>🎭 Kimlik Profili</h3>
      <div class="id-radar">${
        Object.entries(prof.identity_vector||{})
          .filter(([,v])=>v>0.05)
          .sort((a,b)=>b[1]-a[1])
          .slice(0,8)
          .map(([k,v])=>`<span class="id-tag" style="background:#0d1117;border:1px solid #30363d">
            ${k} <b style="color:var(--accent)">${(v*100).toFixed(0)}%</b></span>`)
          .join('')||'<span style="color:var(--text2);font-size:.75rem">Kimlik sinyali bulunamadı</span>'
      }</div>
    </div>

    ${aliasGroup.length?`<div class="modal-section">
      <h3>🔗 Alias Bağlantıları</h3>
      <div style="font-size:.78rem">${aliasGroup.map(u=>`<span style="color:var(--purple);margin-right:8px">@${u}</span>`).join('')}</div>
    </div>`:''}

    <div class="modal-section">
      <h3>🕸️ İlgili Mesajlar & Kullanıcılar (${myRels.length} bağlantı)</h3>
      <div class="related-list">
        ${myRels.slice(0,15).map(rel=>{
          const otherId = rel.source_id===n.id?rel.target_id:rel.source_id;
          const otherMsg = STATE.messages.find(mm=>`msg_${mm.msg_id}`===otherId);
          const isUser = otherId.startsWith('user_');
          const color = COLORS[rel.rel_type]||'#888';
          return `<div class="related-item" onclick="highlightNode('${otherId}')">
            <div class="ri-type" style="color:${color}">⬤ ${rel.rel_type} (${(rel.weight*100).toFixed(0)}%)</div>
            <div class="ri-text">${isUser?'👤 Kullanıcı: '+otherId.replace('user_',''):
              otherMsg?'@'+otherMsg.username+': '+escHtml(otherMsg.raw_text.slice(0,60)):otherId}</div>
          </div>`;
        }).join('')}
      </div>
    </div>

    <div class="modal-section">
      <h3>🎮 Oyun Kuramı Skorları</h3>
      ${Object.entries(gameScores).map(([k,v])=>
        `<div class="stat-row"><span class="stat-label">${k}</span>
         <span class="stat-val">${typeof v==='object'?JSON.stringify(v).slice(0,40):
           typeof v==='number'?v.toFixed(3):v}</span></div>`
      ).join('')||'<p style="color:var(--text2);font-size:.75rem">Veri yok</p>'}
    </div>

    <div style="text-align:center;margin-top:12px">
      <button class="btn" style="display:inline-block;width:auto;padding:6px 16px"
        onclick="quickPredict('${m.username}','${escHtml(m.raw_text.slice(0,30))}')">
        🔮 Bu Kullanıcı İçin Tahmin Üret
      </button>
    </div>`;

  document.getElementById('nodeModal').classList.add('open');
}

function showEdgeModal(e){
  const srcMsg = STATE.messages.find(m=>`msg_${m.msg_id}`===e.source_id);
  const tgtMsg = STATE.messages.find(m=>`msg_${m.msg_id}`===e.target_id);
  document.getElementById('modalTitle').innerHTML =
    `İlişki: <span style="color:${COLORS[e.rel_type]||'#888'}">${e.rel_type}</span>`;
  document.getElementById('modalBody').innerHTML=`
    <div class="modal-section">
      <h3>🔗 İlişki Detayı</h3>
      <div class="stat-row"><span class="stat-label">Tür</span><span class="stat-val" style="color:${COLORS[e.rel_type]||'#888'}">${e.rel_type}</span></div>
      <div class="stat-row"><span class="stat-label">Ağırlık</span><span class="stat-val">${(e.weight*100).toFixed(1)}%</span></div>
      <div class="stat-row"><span class="stat-label">Kaynak</span><span class="stat-val">${e.source_id}</span></div>
      <div class="stat-row"><span class="stat-label">Hedef</span><span class="stat-val">${e.target_id}</span></div>
    </div>
    <div class="modal-section">
      <h3>📝 Kaynak Mesaj</h3>
      ${srcMsg?`<div class="msg-text-block"><b style="color:var(--accent)">@${srcMsg.username}</b>: ${escHtml(srcMsg.raw_text)}</div>`:'<p style="color:var(--text2)">Kullanıcı düğümü</p>'}
    </div>
    <div class="modal-section">
      <h3>📝 Hedef Mesaj</h3>
      ${tgtMsg?`<div class="msg-text-block"><b style="color:var(--accent)">@${tgtMsg.username}</b>: ${escHtml(tgtMsg.raw_text)}</div>`:'<p style="color:var(--text2)">Kullanıcı düğümü</p>'}
    </div>
    <div class="modal-section">
      <h3>🔍 Kanıt</h3>
      <pre style="font-size:.72rem;color:var(--text2);background:#0d1117;padding:8px;border-radius:4px;overflow-x:auto">${JSON.stringify(e.evidence||{},null,2)}</pre>
    </div>`;
  document.getElementById('nodeModal').classList.add('open');
}

function showUserModal(username){
  const prof = STATE.user_profiles[username];
  if(!prof) return;
  const msgs = STATE.messages.filter(m=>m.username===username);
  const rels = STATE.relationships.filter(r=>r.source_id==='user_'+username||r.target_id==='user_'+username);

  document.getElementById('modalTitle').innerHTML=`Kullanıcı Profili: <span style="color:var(--accent)">@${username}</span>`;
  document.getElementById('modalBody').innerHTML=`
    <div class="modal-section">
      <h3>👤 Genel Bilgi</h3>
      <div class="stat-row"><span class="stat-label">Mesaj Sayısı</span><span class="stat-val">${prof.msg_count}</span></div>
      <div class="stat-row"><span class="stat-label">Diller</span><span class="stat-val">${(prof.languages||[]).join(', ')}</span></div>
      <div class="stat-row"><span class="stat-label">Küme</span><span class="stat-val">${prof.cluster_id>=0?'Küme '+prof.cluster_id:'Bağımsız'}</span></div>
      <div class="stat-row"><span class="stat-label">Bot Kararı</span><span class="stat-val" style="color:${(prof.bot_probability||0)>0.5?'var(--red)':'var(--green)'}">${prof.bot_verdict||'?'}</span></div>
    </div>
    <div class="modal-section">
      <h3>📊 Skorlar</h3>
      <div class="prob-bar"><span class="pb-label">Bot</span><div class="pb-track"><div class="pb-fill bot" style="width:${(prof.bot_probability||0)*100}%"></div></div><span class="pb-val">${((prof.bot_probability||0)*100).toFixed(0)}%</span></div>
      <div class="prob-bar"><span class="pb-label">Anonimlik</span><div class="pb-track"><div class="pb-fill anon" style="width:${(prof.anon_score||0)*100}%"></div></div><span class="pb-val">${((prof.anon_score||0)*100).toFixed(0)}%</span></div>
      <div class="prob-bar"><span class="pb-label">Deception</span><div class="pb-track"><div class="pb-fill deception" style="width:${(prof.deception_score||0)*100}%"></div></div><span class="pb-val">${((prof.deception_score||0)*100).toFixed(0)}%</span></div>
    </div>
    <div class="modal-section">
      <h3>🤖 Bot Sinyalleri</h3>
      ${Object.entries(prof.bot_signals||{}).map(([k,v])=>
        `<div class="prob-bar"><span class="pb-label" style="width:130px">${k}</span>
         <div class="pb-track"><div class="pb-fill bot" style="width:${v*100}%"></div></div>
         <span class="pb-val">${(v*100).toFixed(0)}%</span></div>`).join('')}
    </div>
    <div class="modal-section">
      <h3>🎭 Kimlik Vektörü</h3>
      <div class="id-radar">${
        Object.entries(prof.identity_vector||{})
          .filter(([,v])=>v>0.03)
          .sort((a,b)=>b[1]-a[1])
          .map(([k,v])=>`<span class="id-tag" style="background:#0d1117;border:1px solid #30363d">
            ${k} <b style="color:var(--accent)">${(v*100).toFixed(0)}%</b></span>`)
          .join('')
      }</div>
    </div>
    ${(prof.alias_group||[]).length?`
    <div class="modal-section">
      <h3>🔗 Alias Şüphelileri</h3>
      ${(prof.alias_group||[]).map(u=>`<span style="color:var(--purple);margin-right:8px;cursor:pointer" onclick="showUserModal('${u}')">@${u}</span>`).join('')}
    </div>`:''}
    <div class="modal-section">
      <h3>💬 Mesajları (${msgs.length})</h3>
      <div class="related-list">
        ${msgs.slice(0,10).map(m=>`<div class="related-item" onclick="highlightNode('msg_${m.msg_id}')">
          <div class="ri-type" style="color:var(--text2)">#${m.msg_id} · ${m.language||'?'}</div>
          <div class="ri-text">${escHtml(m.raw_text.slice(0,80))}</div>
        </div>`).join('')}
      </div>
    </div>
    <div style="text-align:center;margin-top:12px;display:flex;gap:8px;justify-content:center">
      <button class="btn" style="display:inline-block;width:auto;padding:6px 14px" onclick="quickPredict('${username}','')">🔮 Tahmin</button>
      <button class="btn secondary" style="display:inline-block;width:auto;padding:6px 14px" onclick="runAnonForUser('${username}')">🎭 Kimlik Analizi</button>
    </div>`;
  document.getElementById('nodeModal').classList.add('open');
}

function closeModal(){ document.getElementById('nodeModal').classList.remove('open'); }
document.getElementById('nodeModal').addEventListener('click',e=>{ if(e.target===document.getElementById('nodeModal')) closeModal(); });

function highlightNode(nodeId){
  closeModal();
  const n = STATE.nodes.find(nd=>nd.id===nodeId);
  if(!n) return;
  d3.selectAll('.swim-node circle').attr('stroke','#21262d');
  d3.selectAll('.swim-node').filter((d,i,nodes)=>{
    const g=d3.select(nodes[i]);
    const t=g.attr('transform')||'';
    const match=t.match(/translate\(([^,]+),([^)]+)\)/);
    if(match && Math.abs(parseFloat(match[1])-n.x)<2 && Math.abs(parseFloat(match[2])-n.y)<2)
      return true;
  }).select('circle').attr('stroke','#fff').attr('stroke-width',3);
}

// ═══════════════════════════════════════════════════════
//  BAYESIAN PREDICTOR
// ═══════════════════════════════════════════════════════
function populateBayesUser(r){
  const sel = document.getElementById('bayesUser');
  sel.innerHTML='<option value="">Kullanıcı seç (opsiyonel)</option>'+
    Object.keys(r.user_profiles||{}).map(u=>`<option value="${u}">@${u}</option>`).join('');
}

async function runBayesQuery(){
  if(!STATE.session_id){ toast('Önce analiz yapın','var(--red)'); return; }
  const query = document.getElementById('bayesQuery').value.trim();
  const user  = document.getElementById('bayesUser').value;
  const topic = document.getElementById('bayesTopic').value.trim();
  if(!query){ toast('Sorgu giriniz','var(--red)'); return; }
  const btn = event.target;
  btn.textContent='⏳…'; btn.disabled=true;
  const resDiv = document.getElementById('bayesResult');
  resDiv.style.display='block';
  resDiv.innerHTML='<div class="spinner"></div>';
  try{
    const r = await api('/predict', { session_id:STATE.session_id, query, user, topic });
    let html='';
    if(r.ollama_response){ html+=`<b style="color:var(--accent)">📝 Ollama Yanıtı:</b>\n${r.ollama_response}\n\n`; }
    if(r.markov_generated){ html+=`<b style="color:var(--purple)">🧩 Markov Üretimi:</b> ${r.markov_generated}\n\n`; }
    if(r.author_scores){ html+=`<b style="color:var(--yellow)">📊 Yazar Posteriorı:</b>\n`+
      Object.entries(r.author_scores).sort((a,b)=>b[1]-a[1]).slice(0,5)
        .map(([u,s])=>`  @${u}: ${(s*100).toFixed(1)}%`).join('\n'); }
    resDiv.textContent=html||JSON.stringify(r,null,2);
  }catch(e){
    resDiv.textContent='Hata: '+e.message;
  }finally{
    btn.textContent='🔮 Tahmin Üret'; btn.disabled=false;
  }
}

async function runAnonPredict(){
  if(!STATE.session_id){ toast('Önce analiz yapın','var(--red)'); return; }
  const text = document.getElementById('bayesQuery').value.trim() ||
               document.getElementById('bayesTopic').value.trim();
  if(!text){ toast('Tahmin için metin giriniz','var(--red)'); return; }
  const resDiv=document.getElementById('bayesResult');
  resDiv.style.display='block'; resDiv.innerHTML='<div class="spinner"></div>';
  try{
    const r = await api('/predict_anon', { session_id:STATE.session_id, text });
    let html = `🎭 TAHMİN EDİLEN YAZAR: @${r.predicted_author}\n\n`;
    if(r.author_scores){
      html+=`📊 OLASI YAZARLAR:\n`+Object.entries(r.author_scores)
        .sort((a,b)=>b[1]-a[1]).map(([u,s])=>`  @${u}: ${(s*100).toFixed(1)}%`).join('\n')+'\n\n';
    }
    if(r.identity_profile){
      html+=`🎭 KİMLİK PROFİLİ:\n`+r.identity_profile.map(([k,v])=>`  ${k}: ${(v*100).toFixed(1)}%`).join('\n')+'\n\n';
    }
    if(r.ollama_analysis){ html+=`📝 OLLAMA ANALİZİ:\n${r.ollama_analysis}`; }
    resDiv.textContent=html;
  }catch(e){ resDiv.textContent='Hata: '+e.message; }
}

function quickPredict(username, topic){
  closeModal();
  showTab('bayes');
  document.getElementById('bayesUser').value = username;
  document.getElementById('bayesTopic').value = topic;
  document.getElementById('bayesQuery').value = `@${username} bu konuda ne yazar?`;
}

function runAnonForUser(username){
  closeModal();
  showTab('bayes');
  const msgs = STATE.messages.filter(m=>m.username===username).slice(0,3);
  document.getElementById('bayesQuery').value = msgs.map(m=>m.raw_text).join(' ');
  runAnonPredict();
}

// ═══════════════════════════════════════════════════════
//  SEARCH
// ═══════════════════════════════════════════════════════
async function doSearch(){
  if(!STATE.session_id) return;
  clearTimeout(STATE.searchTimeout);
  STATE.searchTimeout = setTimeout(async()=>{
    const q = document.getElementById('searchQ').value.trim();
    if(!q){ document.getElementById('searchResults').innerHTML=''; return; }
    const type = document.getElementById('searchType').value;
    try{
      const r = await api('/search', { session_id:STATE.session_id, query:q, search_type:type });
      renderSearchResults(r);
    }catch(e){}
  }, 300);
}

function renderSearchResults(r){
  let html = '';
  const all = [
    ...(r.messages||[]).map(m=>({type:'msg', item:m})),
    ...(r.users||[]).map(u=>({type:'user', item:u})),
    ...(r.relationships||[]).map(rel=>({type:'rel', item:rel})),
    ...(r.network_expansion||[]).map(n=>({type:'net', item:n})),
  ].sort((a,b)=>(b.item._score||0)-(a.item._score||0)).slice(0,20);

  if(!all.length){
    html='<p style="font-size:.75rem;color:var(--text2);text-align:center;margin-top:12px">Sonuç bulunamadı</p>';
  } else {
    all.forEach(({type,item})=>{
      if(type==='msg'){
        html+=`<div class="sr-item" onclick="highlightNode('msg_${item.msg_id}')">
          <div><span class="sr-user">@${item.username}</span> <span class="sr-score">${((item._score||0)*100).toFixed(0)}%</span></div>
          <div class="sr-text">${escHtml((item.raw_text||'').slice(0,70))}</div>
        </div>`;
      } else if(type==='user'){
        html+=`<div class="sr-item" onclick="showUserModal('${item.username}')">
          <div><span class="sr-user">👤 @${item.username}</span> <span class="sr-score">${((item._score||0)*100).toFixed(0)}%</span></div>
          <div class="sr-text">Bot: ${((item.bot_probability||0)*100).toFixed(0)}% · Mesaj: ${item.msg_count||0}</div>
        </div>`;
      } else if(type==='rel'){
        html+=`<div class="sr-item">
          <div><span class="sr-user" style="color:${COLORS[item.rel_type]||'#888'}">⬤ ${item.rel_type}</span> <span class="sr-score">${((item._score||0)*100).toFixed(0)}%</span></div>
          <div class="sr-text">${item.source_id} → ${item.target_id} (${((item.weight||0)*100).toFixed(0)}%)</div>
        </div>`;
      } else {
        html+=`<div class="sr-item">
          <div><span class="sr-user">🕸️ ${item.node}</span> <small style="color:var(--text2)">derinlik:${item.depth}</small></div>
          <div class="sr-text">${item.rel_type} ← ${item.from}</div>
        </div>`;
      }
    });
  }
  document.getElementById('searchResults').innerHTML = html;
}

// ═══════════════════════════════════════════════════════
//  SESSIONS
// ═══════════════════════════════════════════════════════
async function loadSessions(){
  try{
    const r = await api('/sessions', {}, 'GET');
    const list = document.getElementById('sessions-list');
    if(!r.sessions?.length){ list.innerHTML='<p style="color:var(--text2);font-size:.75rem">Kayıtlı oturum yok</p>'; return; }
    list.innerHTML = r.sessions.map(s=>`
      <div class="session-item">
        <div>
          <div style="color:var(--accent);font-size:.78rem">${s.session_id.slice(0,16)}…</div>
          <div style="color:var(--text2);font-size:.68rem">${s.message_count||0} mesaj · ${s.last_updated?.slice(0,16)||''}</div>
        </div>
        <div style="display:flex;gap:4px">
          <button onclick="loadSession('${s.session_id}')" style="background:var(--accent);border:none;color:#000;border-radius:3px;padding:2px 7px;cursor:pointer;font-size:.65rem">Yükle</button>
          <button class="session-del" onclick="deleteSession('${s.session_id}')">Sil</button>
        </div>
      </div>`).join('');
  }catch(e){ toast('Oturumlar yüklenemedi','var(--red)'); }
}

async function loadSession(sid){
  try{
    const r = await api('/session/'+sid, {}, 'GET');
    if(r.error){ toast(r.error,'var(--red)'); return; }
    STATE.session_id    = sid;
    STATE.messages      = r.messages||[];
    STATE.user_profiles = Object.fromEntries((r.users||[]).map(u=>[u.username,u]));
    STATE.relationships = r.relationships||[];
    renderAll({
      session_id: sid,
      message_count: STATE.messages.length,
      user_count: Object.keys(STATE.user_profiles).length,
      messages: STATE.messages,
      user_profiles: STATE.user_profiles,
      relationships: STATE.relationships,
      alias_links: r.alias_links||[],
      topic_keywords: [],
    });
    toast('Oturum yüklendi','var(--green)');
  }catch(e){ toast('Hata: '+e.message,'var(--red)'); }
}

async function deleteSession(sid){
  if(!confirm('Bu oturumu silmek istediğinize emin misiniz?')) return;
  try{
    await api('/session/'+sid+'/delete', {});
    toast('Oturum silindi');
    loadSessions();
  }catch(e){ toast('Hata: '+e.message,'var(--red)'); }
}

// ═══════════════════════════════════════════════════════
//  UI HELPERS
// ═══════════════════════════════════════════════════════
function showTab(name){
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{
    const tabs=['stats','users','bayes','sessions'];
    b.classList.toggle('active', tabs[i]===name);
  });
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if(name==='sessions') loadSessions();
}

function escHtml(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ═══════════════════════════════════════════════════════
//  FILE UPLOAD (txt / md / json)
// ═══════════════════════════════════════════════════════

function handleDragOver(e){
  e.preventDefault();
  document.getElementById('uploadZone').classList.add('drag-over');
}
function handleDragLeave(e){
  document.getElementById('uploadZone').classList.remove('drag-over');
}
function handleDrop(e){
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if(file) processFile(file);
}
function handleFileSelect(e){
  const file = e.target.files[0];
  if(file) processFile(file);
}

function processFile(file){
  const ext = file.name.split('.').pop().toLowerCase();
  if(!['txt','md','json'].includes(ext)){
    toast('Desteklenmeyen format. Sadece .txt .md .json','var(--red)'); return;
  }
  // Dosya bilgi chip'i göster
  document.getElementById('fcName').textContent = file.name;
  document.getElementById('fcSize').textContent = formatBytes(file.size);
  document.getElementById('fileChip').style.display = 'flex';

  const reader = new FileReader();
  reader.onload = (ev) => {
    const raw = ev.target.result;
    try {
      let parsed = '';
      if(ext === 'json'){
        parsed = parseJsonFile(raw, file.name);
      } else {
        // .txt ve .md — doğrudan textarea'ya
        parsed = raw;
      }
      document.getElementById('inputText').value = parsed;
      const lines = parsed.split('\n').filter(l=>l.trim()).length;
      const info  = document.getElementById('fileParseInfo');
      info.textContent = `✓ ${file.name} yüklendi · ${lines} satır · ${formatBytes(file.size)}`;
      info.style.display = 'block';
      toast(`📂 "${file.name}" yüklendi (${lines} satır)`,'var(--green)');
    } catch(err){
      toast('Dosya ayrıştırma hatası: '+err.message,'var(--red)');
    }
  };
  reader.onerror = () => toast('Dosya okunamadı','var(--red)');
  reader.readAsText(file, 'UTF-8');
}

/**
 * JSON → @kullanıcı "mesaj" formatına çeviri.
 * Desteklenen yapılar:
 *  1. {messages:[{username, raw_text},...]}          → sistem formatı
 *  2. [{username, raw_text},...] veya [{user,text}]  → dizi
 *  3. {username:"x", messages:[...]}                 → tek kullanıcı
 *  4. Ham metin içeren herhangi bir alan             → düz metin
 */
function parseJsonFile(raw, filename){
  let obj;
  try { obj = JSON.parse(raw); }
  catch(e){ throw new Error('Geçersiz JSON: '+e.message); }

  const lines = [];

  // Yardımcı: bir objeyi mesaj satırına çevir
  function objToLine(m){
    const user = m.username || m.user || m.author || m.from || 'unknown';
    const text = m.raw_text || m.text || m.message || m.content || m.body || '';
    const ts   = m.timestamp_raw || m.timestamp || m.time || m.date || '';
    if(!text) return null;
    const safe = String(text).replace(/"/g,'\\"');
    return ts ? `${ts} @${user} "${safe}"` : `@${user} "${safe}"`;
  }

  // Yapı 1: {messages:[...]}
  if(obj && Array.isArray(obj.messages)){
    obj.messages.forEach(m=>{ const l=objToLine(m); if(l) lines.push(l); });
  }
  // Yapı 2: [{...},...]
  else if(Array.isArray(obj)){
    obj.forEach(m=>{ const l=objToLine(m); if(l) lines.push(l); });
  }
  // Yapı 3: {username, messages:[...]}
  else if(obj && typeof obj === 'object' && obj.username && Array.isArray(obj.messages)){
    obj.messages.forEach(m=>{
      const text = typeof m === 'string' ? m : (m.text||m.raw_text||m.content||'');
      if(text) lines.push(`@${obj.username} "${text.replace(/"/g,'\\"')}"`);
    });
  }
  // Yapı 4: {text:"..."} veya {content:"..."} — düz metin
  else if(obj && (obj.text || obj.content || obj.raw_text)){
    const txt = obj.text || obj.content || obj.raw_text;
    return String(txt);
  }
  // Fallback: JSON'u düz metin olarak kullan
  else {
    return raw;
  }

  if(!lines.length) throw new Error('JSON içinde mesaj bulunamadı. Desteklenen yapı: {messages:[{username,raw_text}]}');
  return lines.join('\n');
}

function clearFile(){
  document.getElementById('fileInput').value = '';
  document.getElementById('fileChip').style.display = 'none';
  document.getElementById('fileParseInfo').style.display = 'none';
  document.getElementById('inputText').value = '';
  toast('Dosya temizlendi');
}

function formatBytes(b){
  if(b < 1024) return b+'B';
  if(b < 1048576) return (b/1024).toFixed(1)+'KB';
  return (b/1048576).toFixed(1)+'MB';
}

function loadExample(){
  document.getElementById('inputText').value = `@alice "Hello everyone, I'm a Jewish American and very proud of my heritage!"
@bob_1234 "I completely agree with the previous message. Shabbat shalom to all!"
@user_987654 "As a German Christian I find this discussion fascinating"
@alice "We should focus on what unites us, not what divides us"
@bob_1234 "Yes inshallah we will find common ground someday"
@user_987654 "I believe in secular values above religious identity"
@carol "This is a great conversation. As a Turkish Muslim I agree with secular values"
@alice "Carol, mashallah your perspective is valuable"
@bob_1234 "The Arab world has much to contribute to this dialogue"
@carol "We must remember our shared humanity above ethnic boundaries"`;
  toast('Örnek metin yüklendi');
}

// ═══════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════
window.addEventListener('resize', ()=>{
  if(STATE.messages.length) renderSwimlane({messages:STATE.messages,user_profiles:STATE.user_profiles,relationships:STATE.relationships});
});
loadSessions();
</script>
</body>
</html>"""



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  16. FLASK APPLICATION & API ROUTES                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

app = Flask(__name__)
app.config["JSON_ENSURE_ASCII"] = False
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

# Global orchestrator — tek örnek, thread-safe
_orchestrator: Optional["AnalysisOrchestrator"] = None
_orch_lock = threading.Lock()


def get_orchestrator() -> AnalysisOrchestrator:
    """Singleton AnalysisOrchestrator döndür."""
    global _orchestrator
    if _orchestrator is None:
        with _orch_lock:
            if _orchestrator is None:
                _orchestrator = AnalysisOrchestrator(ollama_url=OLLAMA)
    return _orchestrator


# ── Ana Sayfa ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(HTML_TEMPLATE, mimetype="text/html; charset=utf-8")


# ── POST /api/analyze ──────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Ham metni analiz et; tüm sonuçları döndür."""
    data = request.get_json(force=True, silent=True) or {}
    raw_text   = data.get("text", "").strip()
    session_id = data.get("session_id") or None

    if not raw_text:
        return jsonify({"error": "Metin boş olamaz"}), 400

    orch = get_orchestrator()
    try:
        result = orch.process_text(raw_text, session_id=session_id)
    except Exception as e:
        log.error(f"Analiz hatası: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    # JSON serileştirme için numpy float temizle
    return jsonify(_jsonify_result(result))


# ── POST /api/upload ──────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Dosya yükleme endpoint'i.
    Kabul edilen formatlar: .txt  .md  .json
    Multipart form-data: field adı = 'file'
    Döndürür: {text: "ayrıştırılmış metin", filename, size, line_count, format}
    """
    if "file" not in request.files:
        return jsonify({"error": "Dosya alanı bulunamadı ('file' field gerekli)"}), 400

    f        = request.files["file"]
    filename = f.filename or "dosya"
    ext      = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    ALLOWED = {"txt", "md", "json"}
    if ext not in ALLOWED:
        return jsonify({
            "error": f"Desteklenmeyen format: .{ext}. Sadece .txt .md .json kabul edilir"
        }), 415

    # Dosya boyutu sınırı: 10 MB
    MAX_SIZE = 10 * 1024 * 1024
    raw_bytes = f.read(MAX_SIZE + 1)
    if len(raw_bytes) > MAX_SIZE:
        return jsonify({"error": "Dosya 10 MB sınırını aşıyor"}), 413

    # Encoding tespiti: UTF-8 → UTF-16 → latin-1
    raw_text = None
    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            raw_text = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    if raw_text is None:
        return jsonify({"error": "Dosya kodlaması çözümlenemedi"}), 422

    # Format ayrıştırma
    parsed_text  = raw_text
    parse_info   = {}

    if ext == "json":
        try:
            parsed_text, parse_info = _parse_json_upload(raw_text, filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 422

    elif ext in ("txt", "md"):
        # Markdown: kod blokları ve başlıkları temizle (opsiyonel loglama)
        msg_lines = [ln for ln in raw_text.splitlines()
                     if re.match(r'\s*@\S+', ln) or re.match(r'\d{4}-\d{2}-\d{2}', ln)]
        parse_info["detected_format"] = (
            "@user mesaj formatı" if msg_lines
            else "düz metin (parser otomatik deneyecek)"
        )
        parse_info["matching_lines"] = len(msg_lines)

    line_count = len([l for l in parsed_text.splitlines() if l.strip()])

    log.info(f"Dosya yüklendi: {filename} ({len(raw_bytes)} byte, {line_count} satır, {ext})")

    return jsonify({
        "text":       parsed_text,
        "filename":   filename,
        "size":       len(raw_bytes),
        "line_count": line_count,
        "format":     ext,
        "parse_info": parse_info,
    })


def _parse_json_upload(raw_text: str, filename: str) -> Tuple[str, dict]:
    """
    JSON dosyasını @kullanıcı "mesaj" satır formatına dönüştür.

    Desteklenen JSON yapıları:
      Yapı 1: {"messages": [{username, raw_text, ...}, ...]}
      Yapı 2: [{username, raw_text}, ...]               — dizi
      Yapı 3: {"username": "x", "messages": [...]}       — tek kullanıcı
      Yapı 4: {"text": "..."} veya {"content": "..."}   — ham metin
      Yapı 5: Her türlü iç içe yapı — recursive tarama
    """
    try:
        obj = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Geçersiz JSON sözdizimi: {e}")

    lines:      List[str] = []
    parse_info: dict      = {}

    def obj_to_line(m: dict) -> Optional[str]:
        if not isinstance(m, dict):
            return None
        user = (m.get("username") or m.get("user") or m.get("author") or
                m.get("from")     or m.get("sender") or "unknown")
        text = (m.get("raw_text") or m.get("text") or m.get("message") or
                m.get("content")  or m.get("body")  or m.get("msg")    or "")
        ts   = (m.get("timestamp_raw") or m.get("timestamp") or
                m.get("time")          or m.get("date")       or "")
        if not str(text).strip():
            return None
        safe_text = str(text).replace('"', '\\"')
        return (f"{ts} @{user} \"{safe_text}\""
                if ts else f"@{user} \"{safe_text}\"")

    # Yapı 1: {"messages": [...]}
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        for m in obj["messages"]:
            ln = obj_to_line(m)
            if ln:
                lines.append(ln)
        parse_info["structure"] = "sistem formatı {messages:[...]}"

    # Yapı 2: [{...}, ...]
    elif isinstance(obj, list):
        for m in obj:
            ln = obj_to_line(m)
            if ln:
                lines.append(ln)
        parse_info["structure"] = f"dizi ({len(obj)} öğe)"

    # Yapı 3: {"username": "x", "messages": [...]}
    elif (isinstance(obj, dict) and
          obj.get("username") and isinstance(obj.get("messages"), list)):
        uname = obj["username"]
        for m in obj["messages"]:
            text = m if isinstance(m, str) else (
                m.get("text") or m.get("raw_text") or m.get("content") or "")
            if text:
                lines.append(f"@{uname} \"{str(text).replace(chr(34), chr(92)+chr(34))}\"")
        parse_info["structure"] = f"tek kullanıcı @{uname}"

    # Yapı 4: {"text":...} veya {"content":...}
    elif isinstance(obj, dict) and any(k in obj for k in ("text","content","raw_text","body")):
        txt = (obj.get("text") or obj.get("content") or
               obj.get("raw_text") or obj.get("body") or "")
        parse_info["structure"] = "ham metin alanı"
        return str(txt), parse_info

    # Yapı 5: Fallback — ham JSON metnini döndür
    else:
        parse_info["structure"] = "tanımsız yapı (ham JSON)"
        return raw_text, parse_info

    if not lines:
        raise ValueError(
            "JSON içinde ayrıştırılabilir mesaj bulunamadı. "
            "Beklenen yapı: {\"messages\":[{\"username\":\"...\",\"raw_text\":\"...\"}]}"
        )

    parse_info["parsed_messages"] = len(lines)
    return "\n".join(lines), parse_info


# ── POST /api/predict ──────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Kullanıcı + konu verilerek Bayesyan mesaj tahmini.
    Ayrıca serbest sorgu için Ollama'ya yönlendirir.
    """
    data       = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "")
    query      = data.get("query", "").strip()
    username   = data.get("user", "").strip()
    topic      = data.get("topic", query).strip()

    if not session_id:
        return jsonify({"error": "session_id gerekli"}), 400

    orch = get_orchestrator()
    db   = orch.db
    pred = orch.predictor

    # Oturum mesaj ve profillerini yükle
    messages  = db.get_session_messages(session_id)
    profiles  = {p["username"]: p for p in db.get_session_users(session_id)}

    if not messages:
        return jsonify({"error": "Oturumda mesaj yok – önce analiz yapın"}), 400

    result: Dict[str, Any] = {}

    # Belirli bir kullanıcı için tahmin
    if username and username in profiles:
        user_msgs = [m["raw_text"] for m in messages if m["username"] == username]
        pred_result = pred.predict_user_message(
            username, topic, user_msgs, profiles[username]
        )
        result.update(pred_result)
        result["ollama_response"] = pred_result.get("ollama_predicted", "")

    # Serbest sorgu → Ollama
    if query and not username:
        context = {
            "session_id":   session_id,
            "user_count":   len(profiles),
            "msg_count":    len(messages),
            "usernames":    list(profiles.keys()),
            "avg_bot_prob": (
                sum(p.get("bot_probability", 0) for p in profiles.values()) /
                max(len(profiles), 1)
            ),
            "top_relations": [
                r["rel_type"] for r in
                sorted(db.get_relationships(session_id),
                       key=lambda x: x.get("weight", 0), reverse=True)[:5]
            ],
        }
        result["ollama_response"] = pred.answer_query(query, context)

    # Yazar posteriorı (tüm kullanıcılar)
    if topic:
        tokens = TextParser.tokenize(topic)
        result["author_scores"] = orch.markov.author_posterior(tokens)

    # Markov üretimi
    if username and username in orch.markov._user_models:
        seed = TextParser.tokenize(topic)[:2]
        gen  = orch.markov.generate_markov(username, seed, length=25)
        result["markov_generated"] = " ".join(gen)

    return jsonify(_jsonify_result(result))


# ── POST /api/predict_anon ─────────────────────────────────────────────────
@app.route("/api/predict_anon", methods=["POST"])
def api_predict_anon():
    """Anonim metin → kimlik + yazar tahmini."""
    data       = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "")
    text       = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Metin boş olamaz"}), 400
    if not session_id:
        return jsonify({"error": "session_id gerekli"}), 400

    orch     = get_orchestrator()
    profiles = {p["username"]: p for p in orch.db.get_session_users(session_id)}

    # NLP fit durumu güncelle (yeni oturum yüklenebilir)
    messages  = orch.db.get_session_messages(session_id)
    all_texts = [m["raw_text"] for m in messages]
    if all_texts:
        orch.nlp.fit(all_texts)
        for u_msgs in _group_by_user(messages).values():
            tokens = []
            for m in u_msgs:
                tokens.extend(m.get("tokens") or TextParser.tokenize(m["raw_text"]))
            user_texts_map = {m["username"]: [m["raw_text"]] for m in u_msgs}
            orch.markov.train_user(list(user_texts_map.keys())[0], tokens)

    result = orch.predictor.predict_anonymous_identity(text, profiles)
    # identity_profile dict → list for frontend compat
    if isinstance(result.get("identity_profile"), dict):
        result["identity_profile"] = sorted(
            result["identity_profile"].items(), key=lambda x: x[1], reverse=True
        )
    return jsonify(_jsonify_result(result))


# ── POST /api/search ───────────────────────────────────────────────────────
@app.route("/api/search", methods=["POST"])
def api_search():
    """Evrensel arama: mesaj, kullanıcı, ilişki, ağ genişleme."""
    data        = request.get_json(force=True, silent=True) or {}
    session_id  = data.get("session_id", "")
    query       = data.get("query", "").strip()
    search_type = data.get("search_type", "all")
    limit       = int(data.get("limit", 20))

    if not session_id or not query:
        return jsonify({"error": "session_id ve query gerekli"}), 400

    orch = get_orchestrator()

    # NLP güncel tut
    messages  = orch.db.get_session_messages(session_id)
    all_texts = [m["raw_text"] for m in messages]
    if all_texts and not orch.nlp._fitted:
        orch.nlp.fit(all_texts)

    result = orch.search.search(query, session_id, search_type, limit)

    # Kullanıcı araması ek: ilişki kategorileri ile puan
    if search_type in ["all", "users", "user"]:
        profiles = orch.db.get_session_users(session_id)
        rels     = orch.db.get_relationships(session_id)
        for u in result.get("users", []):
            uname = u["username"]
            # Bu kullanıcının ilgili olduğu ilişki türleri ve puanları
            u_rels = [r for r in rels
                      if r["source_id"] == f"user_{uname}"
                      or r["target_id"] == f"user_{uname}"]
            rel_cat = Counter(r["rel_type"] for r in u_rels)
            u["relation_categories"] = dict(rel_cat.most_common(6))
            u["relation_count"] = len(u_rels)

    return jsonify(_jsonify_result(result))


# ── GET /api/sessions ──────────────────────────────────────────────────────
@app.route("/api/sessions", methods=["GET"])
def api_sessions():
    """Kayıtlı oturumları listele."""
    orch     = get_orchestrator()
    sessions = orch.db.list_sessions()
    return jsonify({"sessions": sessions})


# ── GET /api/session/<sid> ─────────────────────────────────────────────────
@app.route("/api/session/<sid>", methods=["GET"])
def api_get_session(sid: str):
    """Belirli bir oturumun mesaj/kullanıcı/ilişki verilerini döndür."""
    orch     = get_orchestrator()
    sess     = orch.db.fetchone("SELECT * FROM sessions WHERE session_id=?", (sid,))
    if not sess:
        return jsonify({"error": "Oturum bulunamadı"}), 404

    messages      = orch.db.get_session_messages(sid)
    users         = orch.db.get_session_users(sid)
    relationships = orch.db.get_relationships(sid)
    alias_links   = _compute_alias_links(users)

    return jsonify(_jsonify_result({
        "session_id":    sid,
        "messages":      messages,
        "users":         users,
        "relationships": relationships,
        "alias_links":   alias_links,
    }))


# ── POST /api/session/<sid>/delete ────────────────────────────────────────
@app.route("/api/session/<sid>/delete", methods=["POST"])
def api_delete_session(sid: str):
    """Oturumu sil."""
    orch = get_orchestrator()
    orch.db.delete_session(sid)
    return jsonify({"deleted": sid})


# ── GET /api/report ────────────────────────────────────────────────────────
@app.route("/api/report", methods=["GET"])
def api_report():
    """
    Tam istatistik raporu üret ve HTML olarak indir.
    Tüm kullanıcı/mesaj/ilişki/bot/kimlik verilerini içerir.
    """
    session_id = request.args.get("session_id", "")
    if not session_id:
        return "session_id gerekli", 400

    orch = get_orchestrator()
    sess = orch.db.fetchone("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    if not sess:
        return "Oturum bulunamadı", 404

    messages      = orch.db.get_session_messages(session_id)
    users         = orch.db.get_session_users(session_id)
    relationships = orch.db.get_relationships(session_id)
    alias_links   = _compute_alias_links(users)

    html_report = _generate_html_report(session_id, sess, messages, users,
                                         relationships, alias_links)

    return Response(
        html_report,
        mimetype="text/html; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="rapor_{session_id[:8]}.html"'
        }
    )


# ── POST /api/node_relations ───────────────────────────────────────────────
@app.route("/api/node_relations", methods=["POST"])
def api_node_relations():
    """Belirli bir düğümden tüm ilgili mesaj ve kullanıcıları döndür."""
    data       = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "")
    node_id    = data.get("node_id", "")

    if not session_id or not node_id:
        return jsonify({"error": "session_id ve node_id gerekli"}), 400

    orch = get_orchestrator()
    rels = orch.db.get_relationships(session_id, node_id=node_id)

    related_nodes = set()
    for r in rels:
        related_nodes.add(r["source_id"])
        related_nodes.add(r["target_id"])
    related_nodes.discard(node_id)

    messages = orch.db.get_session_messages(session_id)
    users    = orch.db.get_session_users(session_id)

    related_messages = [m for m in messages
                        if f"msg_{m['msg_id']}" in related_nodes]
    related_users    = [u for u in users
                        if f"user_{u['username']}" in related_nodes]

    return jsonify(_jsonify_result({
        "node_id":         node_id,
        "relationships":   rels,
        "related_messages":related_messages,
        "related_users":   related_users,
        "network_depth_2": _bfs_expand_relationships(node_id, rels, depth=2),
    }))


# ── POST /api/user_stats ───────────────────────────────────────────────────
@app.route("/api/user_stats", methods=["POST"])
def api_user_stats():
    """Bir kullanıcının tam istatistik raporunu JSON olarak döndür."""
    data       = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "")
    username   = data.get("username", "")

    if not session_id or not username:
        return jsonify({"error": "session_id ve username gerekli"}), 400

    orch = get_orchestrator()
    messages  = orch.db.get_session_messages(session_id)
    profiles  = {p["username"]: p for p in orch.db.get_session_users(session_id)}
    rels      = orch.db.get_relationships(session_id)

    if username not in profiles:
        return jsonify({"error": f"Kullanıcı '{username}' bulunamadı"}), 404

    profile   = profiles[username]
    user_msgs = [m for m in messages if m["username"] == username]
    user_rels = [r for r in rels
                 if r["source_id"] == f"user_{username}"
                 or r["target_id"] == f"user_{username}"]

    # Diğer kullanıcıların kimlik maskeleme olasılıkları
    other_profiles_iv = {u: p.get("identity_vector", {})
                         for u, p in profiles.items() if u != username}
    mask_probs = IdentityProfiler.identity_mask_probability(
        profile.get("identity_vector", {}),
        other_profiles_iv
    )

    # Stilometrik karşılaştırma
    my_sig   = profile.get("stylometric_sig", {})
    sty_comp = {}
    for u, p in profiles.items():
        if u == username:
            continue
        other_sig = p.get("stylometric_sig", {})
        if my_sig and other_sig:
            delta = StylometryEngine.burrows_delta(my_sig, other_sig)
            sty_comp[u] = {
                "burrows_delta":    round(delta, 4),
                "same_author_prob": round(StylometryEngine.same_author_probability(delta), 4),
            }

    # İlişki kategorileri puanlı
    rel_cat = Counter(r["rel_type"] for r in user_rels)

    return jsonify(_jsonify_result({
        "username":           username,
        "profile":            profile,
        "messages":           user_msgs,
        "relationships":      user_rels,
        "relation_categories":dict(rel_cat.most_common()),
        "identity_mask_probs":mask_probs,
        "stylometric_comparison": dict(
            sorted(sty_comp.items(),
                   key=lambda x: x[1]["same_author_prob"], reverse=True)[:5]
        ),
    }))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  17. HELPER UTILITIES                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _jsonify_result(obj: Any) -> Any:
    """NumPy/set türlerini JSON-uyumlu Python tiplerine dönüştür."""
    if isinstance(obj, dict):
        return {k: _jsonify_result(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_result(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0.0
    return obj


def _group_by_user(messages: List[dict]) -> Dict[str, List[dict]]:
    """Mesajları kullanıcıya göre grupla."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for m in messages:
        groups[m["username"]].append(m)
    return dict(groups)


def _compute_alias_links(users: List[dict]) -> List[dict]:
    """Kullanıcı listesinden hızlı alias skoru hesapla."""
    links = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            ua, ub = users[i]["username"], users[j]["username"]
            # Levenshtein tabanlı ad benzerliği
            name_sim = RelationshipGraph._name_similarity(ua, ub)
            # Kimlik vektör cosinüs
            iv_a = users[i].get("identity_vector") or {}
            iv_b = users[j].get("identity_vector") or {}
            if isinstance(iv_a, str):
                try: iv_a = json.loads(iv_a)
                except: iv_a = {}
            if isinstance(iv_b, str):
                try: iv_b = json.loads(iv_b)
                except: iv_b = {}
            dims = IDENTITY_DIMS
            va = np.array([iv_a.get(d, 0) for d in dims], dtype=float) + 1e-9
            vb = np.array([iv_b.get(d, 0) for d in dims], dtype=float) + 1e-9
            va /= va.sum(); vb /= vb.sum()
            id_sim = float(np.dot(va, vb))
            combined = name_sim * 0.3 + id_sim * 0.7
            if combined > 0.5:
                links.append({
                    "user_a": ua,
                    "user_b": ub,
                    "combined_score": round(combined, 4),
                    "name_sim": round(name_sim, 4),
                    "identity_sim": round(id_sim, 4),
                    "verdict": (
                        "Yüksek Olasılıklı Alias" if combined > 0.80 else
                        "Orta Olasılıklı Alias"    if combined > 0.65 else
                        "Düşük Olasılıklı Alias"
                    ),
                })
    return links


def _bfs_expand_relationships(seed_id: str, rels: List[dict],
                               depth: int = 2) -> List[Dict]:
    """BFS ile ilişki ağını genişlet."""
    visited  = {seed_id}
    frontier = [seed_id]
    expansion: List[Dict] = []
    for d in range(depth):
        next_frontier = []
        for node in frontier:
            neighbors = [
                (r["target_id"] if r["source_id"] == node else r["source_id"],
                 r["rel_type"], r["weight"])
                for r in rels
                if r["source_id"] == node or r["target_id"] == node
            ]
            for neighbor, rtype, weight in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    expansion.append({
                        "node":     neighbor,
                        "from":     node,
                        "rel_type": rtype,
                        "weight":   weight,
                        "depth":    d + 1,
                    })
        frontier = next_frontier
        if not frontier:
            break
    return expansion


def _generate_html_report(session_id: str, sess: dict,
                            messages: List[dict], users: List[dict],
                            relationships: List[dict],
                            alias_links: List[dict]) -> str:
    """
    Tam istatistik HTML raporu üret.
    Kullanıcı/mesaj/bot/kimlik/ilişki verilerini kapsar.
    """
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    u_by_bot = sorted(users, key=lambda u: u.get("bot_probability", 0), reverse=True)
    rel_counts = Counter(r["rel_type"] for r in relationships)
    top_rels   = rel_counts.most_common(15)

    # Kullanıcı tablosu HTML
    def pct(v): return f"{float(v or 0)*100:.1f}%"
    def bar(v, color="#58a6ff"):
        w = min(100, float(v or 0) * 100)
        return (f'<div style="background:#21262d;border-radius:3px;height:6px;width:120px;display:inline-block">'
                f'<div style="width:{w}%;height:100%;background:{color};border-radius:3px"></div></div>')

    user_rows = ""
    for u in u_by_bot:
        bp  = float(u.get("bot_probability", 0))
        dec = float(u.get("deception_score", 0))
        anon= float(u.get("anon_score", 0))
        iv  = u.get("identity_vector") or {}
        if isinstance(iv, str):
            try: iv = json.loads(iv)
            except: iv = {}
        top_id = sorted(iv.items(), key=lambda x: x[1], reverse=True)[:3]
        top_id_str = " | ".join(f"{k}:{pct(v)}" for k,v in top_id if float(v)>0.03)
        langs  = u.get("languages") or []
        if isinstance(langs, str):
            try: langs = json.loads(langs)
            except: langs = []
        alias  = u.get("alias_group") or []
        if isinstance(alias, str):
            try: alias = json.loads(alias)
            except: alias = []
        bot_col = "#f85149" if bp > 0.5 else "#3fb950"
        user_rows += f"""
        <tr>
          <td><b>@{u['username']}</b></td>
          <td>{u.get('msg_count', 0)}</td>
          <td style="color:{bot_col}"><b>{pct(bp)}</b><br>{bar(bp, bot_col)}</td>
          <td>{pct(anon)}<br>{bar(anon,'#e3b341')}</td>
          <td>{pct(dec)}<br>{bar(dec,'#f0883e')}</td>
          <td style="font-size:.78em">{top_id_str or '—'}</td>
          <td style="font-size:.78em">{', '.join(langs[:3]) or '—'}</td>
          <td style="font-size:.78em;color:#a371f7">{', '.join(('@'+a) for a in alias[:3]) or '—'}</td>
        </tr>"""

    # Mesaj tablosu (ilk 50)
    msg_rows = ""
    for m in messages[:50]:
        bp  = float(m.get("bot_score", 0))
        dec = float(m.get("deception_score", 0))
        bot_col = "#f85149" if bp > 0.5 else "#3fb950"
        text_short = str(m.get("raw_text",""))[:80].replace("<","&lt;").replace(">","&gt;")
        msg_rows += f"""
        <tr>
          <td style="color:#8b949e">#{m.get('msg_id','')}</td>
          <td style="color:#58a6ff">@{m.get('username','')}</td>
          <td style="font-size:.82em">{text_short}{"…" if len(str(m.get("raw_text","")))>80 else ""}</td>
          <td>{m.get('language','?')}</td>
          <td style="color:{bot_col}">{pct(bp)}</td>
          <td>{pct(dec)}</td>
        </tr>"""

    # İlişki özeti
    rel_rows = ""
    for rtype, cnt in top_rels:
        rel_rows += f'<tr><td>{rtype}</td><td><b>{cnt}</b></td></tr>'

    # Alias bağlantıları
    alias_rows = ""
    for link in alias_links[:20]:
        score_col = "#f85149" if link["combined_score"] > 0.8 else "#e3b341"
        alias_rows += f"""
        <tr>
          <td style="color:#58a6ff">@{link['user_a']}</td>
          <td style="color:#58a6ff">@{link['user_b']}</td>
          <td style="color:{score_col}"><b>{pct(link['combined_score'])}</b></td>
          <td>{link['verdict']}</td>
        </tr>"""

    # Bot şüpheli kullanıcılar özeti
    bot_suspects = [u for u in users if float(u.get("bot_probability",0)) > 0.5]
    bot_count    = len(bot_suspects)
    avg_bot      = (sum(float(u.get("bot_probability",0)) for u in users)
                    / max(len(users),1))
    avg_dec      = (sum(float(u.get("deception_score",0)) for u in users)
                    / max(len(users),1))

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>Analiz Raporu — {session_id[:12]}…</title>
<style>
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:24px}}
  h1{{color:#58a6ff;border-bottom:2px solid #30363d;padding-bottom:10px;margin-bottom:20px}}
  h2{{color:#8b949e;font-size:1rem;text-transform:uppercase;letter-spacing:.08em;margin:28px 0 10px}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem;margin-bottom:24px}}
  th{{background:#161b22;color:#8b949e;padding:8px 12px;text-align:left;border:1px solid #30363d;font-size:.78rem;text-transform:uppercase}}
  td{{padding:7px 12px;border:1px solid #21262d;vertical-align:middle}}
  tr:nth-child(even){{background:#0a0e13}}
  tr:hover{{background:#1a2030}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:24px}}
  .stat-box{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;text-align:center}}
  .stat-val{{font-size:2rem;font-weight:700;color:#58a6ff}}
  .stat-lbl{{color:#8b949e;font-size:.78rem;margin-top:4px}}
  .red{{color:#f85149}}.green{{color:#3fb950}}.yellow{{color:#e3b341}}.purple{{color:#a371f7}}
  @media print{{body{{background:#fff;color:#000}}h1,h2{{color:#000}}}}
</style>
</head>
<body>
<h1>📋 Anonim Yazar Kimlik Analiz Raporu</h1>
<div style="color:#8b949e;font-size:.85rem;margin-bottom:20px">
  Oturum: <code style="color:#58a6ff">{session_id}</code> &nbsp;|&nbsp;
  Oluşturuldu: <b>{now}</b> &nbsp;|&nbsp;
  Mesaj: <b>{len(messages)}</b> &nbsp;|&nbsp;
  Kullanıcı: <b>{len(users)}</b>
</div>

<h2>📊 Genel İstatistikler</h2>
<div class="stat-grid">
  <div class="stat-box"><div class="stat-val">{len(messages)}</div><div class="stat-lbl">Toplam Mesaj</div></div>
  <div class="stat-box"><div class="stat-val">{len(users)}</div><div class="stat-lbl">Kullanıcı</div></div>
  <div class="stat-box"><div class="stat-val red">{bot_count}</div><div class="stat-lbl">Bot Şüphelisi</div></div>
  <div class="stat-box"><div class="stat-val yellow">{pct(avg_bot)}</div><div class="stat-lbl">Ort. Bot Olasılığı</div></div>
  <div class="stat-box"><div class="stat-val">{pct(avg_dec)}</div><div class="stat-lbl">Ort. Deception</div></div>
  <div class="stat-box"><div class="stat-val purple">{len(alias_links)}</div><div class="stat-lbl">Alias Bağlantısı</div></div>
  <div class="stat-box"><div class="stat-val">{len(relationships)}</div><div class="stat-lbl">İlişki Kenarı</div></div>
  <div class="stat-box"><div class="stat-val">{len(rel_counts)}</div><div class="stat-lbl">İlişki Türü</div></div>
</div>

<h2>👤 Kullanıcı Profilleri (Bot'a Göre Sıralı)</h2>
<table>
  <tr>
    <th>Kullanıcı Adı</th><th>Mesaj</th><th>Bot Olasılığı</th><th>Anonimlik</th>
    <th>Deception</th><th>Kimlik Eğilimleri</th><th>Diller</th><th>Alias Şüphelileri</th>
  </tr>
  {user_rows}
</table>

<h2>🤖 Bot Şüphelileri Detayı</h2>
{'<p style="color:#3fb950">Bot şüphelisi kullanıcı tespit edilmedi.</p>' if not bot_suspects else ''}
{''.join(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:12px">
  <div style="color:#58a6ff;font-weight:700;margin-bottom:8px">@{u['username']} — Bot: {pct(u.get('bot_probability',0))}</div>
  <div style="font-size:.82rem">
    {''.join(f'<span style="margin-right:12px"><b style="color:#8b949e">{k}</b>: {pct(v)}</span>'
             for k,v in (json.loads(u['bot_signals']) if isinstance(u.get('bot_signals'),'__str__' and str) else (u.get('bot_signals') or {})).items())}
  </div>
</div>""" for u in bot_suspects[:10])}

<h2>🔗 İlişki Ağı Özeti</h2>
<table style="max-width:500px">
  <tr><th>İlişki Türü</th><th>Kenar Sayısı</th></tr>
  {rel_rows}
</table>

<h2>🎭 Alias Bağlantıları</h2>
{'<p style="color:#3fb950">Alias bağlantısı tespit edilmedi.</p>' if not alias_links else f"""
<table>
  <tr><th>Kullanıcı A</th><th>Kullanıcı B</th><th>Birleşik Skor</th><th>Karar</th></tr>
  {alias_rows}
</table>"""}

<h2>💬 Mesaj Listesi (İlk 50)</h2>
<table>
  <tr><th>#</th><th>Kullanıcı</th><th>Mesaj</th><th>Dil</th><th>Bot</th><th>Deception</th></tr>
  {msg_rows}
</table>

<div style="margin-top:40px;color:#30363d;font-size:.78rem;text-align:center">
  Anonim Yazar Kimlik Çözümleme Sistemi v2.0 &nbsp;·&nbsp; {now}
</div>
</body>
</html>"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  18. MAIN ENTRY POINT                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anonim Yazar Kimlik Çözümleme Sistemi v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
  python analyzer_v2.py
  python analyzer_v2.py --port 8080
  python analyzer_v2.py --port 7860 --ollama http://192.168.1.10:11434
  python analyzer_v2.py --no-browser

Arayüz: http://localhost:<port>
        """
    )
    parser.add_argument("--port",    type=int, default=PORT,
                        help=f"Sunucu portu (varsayılan: {PORT})")
    parser.add_argument("--host",    type=str, default="0.0.0.0",
                        help="Bağlama adresi (varsayılan: 0.0.0.0)")
    parser.add_argument("--ollama",  type=str, default=OLLAMA,
                        help=f"Ollama API adresi (varsayılan: {OLLAMA})")
    parser.add_argument("--model",   type=str, default=OLLAMA_MODEL,
                        help=f"Ollama model adı (varsayılan: {OLLAMA_MODEL})")
    parser.add_argument("--db",      type=str, default=str(DB_PATH),
                        help=f"SQLite veritabanı yolu (varsayılan: {DB_PATH})")
    parser.add_argument("--workers", type=int, default=CPU_WORKERS,
                        help=f"CPU iş parçacığı sayısı (varsayılan: {CPU_WORKERS})")
    parser.add_argument("--debug",   action="store_true",
                        help="Flask debug modunu etkinleştir")
    parser.add_argument("--no-browser", action="store_true",
                        help="Tarayıcıyı otomatik açma")
    return parser.parse_args()


def print_banner(port: int, host: str, ollama: str, device: str):
    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║   ANONİM YAZAR KİMLİK ÇÖZÜMLEME SİSTEMİ  v2.0                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Arayüz    : http://{'localhost' if host=='0.0.0.0' else host}:{port:<5}                              ║
║  Ollama    : {ollama:<55} ║
║  Hesaplama : {device:<55} ║
║  Veritabanı: {str(DB_PATH):<55} ║
╠══════════════════════════════════════════════════════════════════════╣
║  MODÜller  : BotDetector(8-sinyal) · RelationshipGraph · AliasLinker║
║              BayesianPredictor · UniversalSearch · SwimlaneRenderer  ║
╚══════════════════════════════════════════════════════════════════════╝
  Durdurmak için: Ctrl+C
"""
    print(banner)


def main():
    global OLLAMA, OLLAMA_MODEL, DB_PATH, CPU_WORKERS

    args = parse_args()

    # Global ayarları güncelle
    OLLAMA       = args.ollama
    OLLAMA_MODEL = args.model
    DB_PATH      = Path(args.db)
    CPU_WORKERS  = args.workers

    print_banner(args.port, args.host, args.ollama, DEVICE)

    # Orchestrator'ı başlat (DB oluştur)
    log.info("Sistem başlatılıyor...")
    orch = get_orchestrator()
    log.info(f"Veritabanı: {DB_PATH}")
    log.info(f"Ollama: {OLLAMA} — {'✓ Erişilebilir' if orch.predictor._ollama_available else '✗ Erişilemiyor (yerel tahmin aktif)'}")
    log.info(f"CPU işçileri: {CPU_WORKERS}")
    log.info(f"Hesaplama cihazı: {DEVICE}")

    if TORCH_AVAILABLE:
        log.info(f"PyTorch: {torch.__version__} — {'GPU' if DEVICE != 'cpu' else 'CPU'} modu")

    # Tarayıcıyı aç (opsiyonel)
    if not args.no_browser:
        def _open_browser():
            time.sleep(1.5)
            import webbrowser
            url = f"http://localhost:{args.port}"
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open_browser, daemon=True).start()

    # Flask'i başlat
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,   # Çift başlatmayı önle
        threaded=True,        # Çok iş parçacıklı istek işleme
    )


if __name__ == "__main__":
    main()
