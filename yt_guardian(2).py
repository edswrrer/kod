#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  YT GUARDIAN v2.1 — Tek Dosya YouTube Moderasyon & Tehdit Tespit Sistemi   ║
║  Kanal: @ShmirchikArt  |  2023-2026  |  Flask + SQLite + NLP + yt-dlp      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KURULUM (Ubuntu):                                                           ║
║    pip install flask flask-socketio requests numpy scikit-learn networkx    ║
║              yt-dlp langdetect python-dotenv                                ║
║    # Opsiyonel (Selenium ile profil inceleme + yorum silme için):           ║
║    pip install selenium google-api-python-client google-auth-oauthlib       ║
║              google-auth-httplib2                                            ║
║    sudo apt install firefox-esr geckodriver                                  ║
║                                                                              ║
║  VERİ TOPLAMA — HİÇBİR API ANAHTARI GEREKMEDENçalışır:                    ║
║    yt-dlp @ShmirchikArt/videos  → tüm videoları listeler                   ║
║    yt-dlp @ShmirchikArt/streams → tüm canlı yayınları listeler             ║
║    yt-dlp yorumları             → --write-comments ile API'siz yorum çeker  ║
║                                                                              ║
║  MODERATÖR ÖNERİLERİ — HİÇBİR API ANAHTARI GEREKMEDENçalışır:            ║
║    Yerel BART zero-shot + kural tabanlı motor                                ║
║    GET /api/suggest/comment/<comment_id>                                     ║
║    GET /api/suggest/user/<channel_id>                                        ║
║    POST /api/suggest/batch  {"comment_ids": [...]}                           ║
║                                                                              ║
║  .env DOSYASI (tümü opsiyonel — hiçbiri zorunlu değil):                    ║
║    YT_API_KEY=AIza...          # YouTube Data API (varsa yt-dlp yerine)     ║
║    YT_CLIENT_SECRETS=...       # OAuth2 — sadece yorum silme için           ║
║    YT_EMAIL=...                # Selenium girişi (opsiyonel)                 ║
║    YT_PASS=...                 # Selenium girişi (opsiyonel)                 ║
║    CHANNEL_HANDLE=ShmirchikArt # Kanal handle (varsayılan: ShmirchikArt)    ║
║    CHANNEL_ID=UCxxxxxxxxxxxxx  # Varsa manuel gir; yoksa otomatik bulunur   ║
║    FLASK_SECRET=rastgele_gizli_anahtar                                       ║
║                                                                              ║
║  ÇALIŞTIRMA:                                                                ║
║    python yt_guardian.py                                                    ║
║    → http://localhost:5000                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 1: IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, re, json, time, hashlib, threading, logging, sqlite3
import unicodedata, math, random, collections, datetime, traceback, queue
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Flask
from flask import Flask, render_template_string, request, jsonify, Response, session
try:
    from flask_socketio import SocketIO, emit
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False
    print("[UYARI] flask-socketio kurulu değil. Gerçek zamanlı özellikler devre dışı.")

# .env desteği
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv yoksa ortam değişkenlerini doğrudan kullan

# Google API
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GoogleRequest
    import pickle
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False
    print("[UYARI] google-api-python-client kurulu değil. API özellikleri kısıtlı.")

# Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options as FFOptions
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    print("[UYARI] selenium kurulu değil. Firefox otomasyon özellikleri devre dışı.")

# NLP & ML
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from langdetect import detect as langdetect_detect, DetectorFactory
    DetectorFactory.seed = 42
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# HTTP
import urllib.request, urllib.parse, urllib.error
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 2: KONFİGÜRASYON
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # API Anahtarları — HİÇBİR ZAMAN KODA YAZMA, .env KULLAN
    YT_API_KEY          = os.environ.get("YT_API_KEY", "")
    YT_CLIENT_SECRETS   = os.environ.get("YT_CLIENT_SECRETS", "client_secrets.json")
    YT_EMAIL            = os.environ.get("YT_EMAIL", "")
    YT_PASS             = os.environ.get("YT_PASS", "")
    CHANNEL_HANDLE      = os.environ.get("CHANNEL_HANDLE", "@ShmirchikArt")
    CHANNEL_ID          = os.environ.get("CHANNEL_ID", "")
    GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
    FLASK_SECRET        = os.environ.get("FLASK_SECRET", os.urandom(24).hex())

    # Sistem
    DB_PATH             = os.environ.get("DB_PATH", "yt_guardian.db")
    DATA_DIR            = Path(os.environ.get("DATA_DIR", "data"))
    PORT                = int(os.environ.get("PORT", 5000))
    DEBUG               = os.environ.get("DEBUG", "0") == "1"
    SCOPES              = ["https://www.googleapis.com/auth/youtube.force-ssl"]

    # Analiz Eşikleri
    SIM_THRESHOLD       = 0.65   # Kimlik eşleştirme eşiği
    BOT_THRESHOLD       = 0.70   # Bot karar eşiği
    HATE_THRESHOLD      = 0.60   # Nefret söylemi eşiği
    STALK_THRESHOLD     = 0.55   # Stalker eşiği
    LIVE_POLL_SEC       = 5      # Canlı yayın poll süresi
    MAX_COMMENTS_PER_VIDEO = 5000

    # Ağırlıklar (Bileşik benzerlik skoru için)
    W_EMBED  = 0.35
    W_NGRAM  = 0.25
    W_TYPO   = 0.15
    W_TIME   = 0.15
    W_TOPIC  = 0.10

    # Tehdit renk kodları
    THREAT_COLORS = {
        "ANTISEMITE":   "#000000",
        "HATER":        "#ff2200",
        "STALKER":      "#ff6600",
        "IMPERSONATOR": "#cc00cc",
        "COORDINATED":  "#8800cc",
        "BOT":          "#0066ff",
        "SUSPICIOUS":   "#ffaa00",
        "NORMAL":       "#00cc44",
    }

    # Nefret söylemi anahtar kelimeleri (çok-dilli, bağlam bağımlı)
    HATE_KEYWORDS = {
        "antisemite_en": [
            "kike", "yid", "hebe", "jewboy", "goy", "globalist conspiracy",
            "zionist control", "jewish agenda", "rothschild", "soros scheme",
            "jewish money", "protocols", "chosen people conspiracy", "banksters"
        ],
        "antisemite_tr": [
            "yahudi köpeği", "siyonist", "yahudi lobisi", "yahudi komplo",
            "haham devleti", "musevi ihaneti", "derin devlet yahudi"
        ],
        "antisemite_he": ["יהודי כלב", "קונספירציה יהודית", "רוטשילד שולט"],
        "antisemite_ar": ["اليهودي", "المؤامرة الصهيونية", "اليهود يتحكمون"],
        "hate_general_en": [
            "die", "kill yourself", "kys", "go back to", "sub-human",
            "vermin", "parasite", "filth", "degenerate race", "white power",
            "14 words", "heil", "groyper", "replacement theory"
        ],
        "hate_general_tr": [
            "öldür", "git öl", "pişlik", "yok olsun", "soykırım yapın",
            "seni bulacağız", "evinizi biliriz"
        ],
        "groyper_signals": [
            "groyper", "nick fuentes", "america first", "frog twitter",
            "the great replacement", "pepe", "NJF", "cozy.tv"
        ],
        "stalker_signals": [
            "seni izliyorum", "nerede olduğunu biliyorum", "seninle konuşacağız",
            "i'm watching you", "i know where you", "found your address",
            "we know who you are", "your family", "i found you"
        ],
        "bot_spam": [
            "click here", "free money", "earn $", "subscribe back",
            "follow for follow", "f4f", "s4s", "buy followers"
        ]
    }

    @classmethod
    def normalized_handle(cls) -> str:
        """@ ile başlayan veya başlamayan handle girdilerini normalize et."""
        return (cls.CHANNEL_HANDLE or "ShmirchikArt").strip().lstrip("@")

    @classmethod
    def channel_streams_url(cls) -> str:
        return f"https://www.youtube.com/@{cls.normalized_handle()}/streams"

Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 3: LOGLAMA
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("yt_guardian.log", encoding="utf-8")
    ]
)
log = logging.getLogger("YTGuardian")

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 4: VERİTABANI KATMANI
# ─────────────────────────────────────────────────────────────────────────────
class Database:
    def __init__(self, path: str = Config.DB_PATH):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    def _conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        try:
            c = self._conn().cursor()
            c.execute(sql, params)
            self._conn().commit()
            return c
        except Exception as e:
            log.error(f"DB hatası: {e} | SQL: {sql[:80]}")
            raise

    def fetchall(self, sql: str, params=()) -> List[Dict]:
        c = self._conn().cursor()
        c.execute(sql, params)
        rows = c.fetchall()
        return [dict(r) for r in rows]

    def fetchone(self, sql: str, params=()) -> Optional[Dict]:
        c = self._conn().cursor()
        c.execute(sql, params)
        r = c.fetchone()
        return dict(r) if r else None

    def _init_schema(self):
        conn = sqlite3.connect(self.path)
        conn.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS videos (
            video_id     TEXT PRIMARY KEY,
            title        TEXT,
            published_at TEXT,
            video_type   TEXT DEFAULT 'video',  -- video / stream
            description  TEXT,
            channel_id   TEXT,
            fetched_at   TEXT,
            comment_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS comments (
            comment_id    TEXT PRIMARY KEY,
            video_id      TEXT,
            author        TEXT,
            author_channel_id TEXT,
            text          TEXT,
            timestamp_utc INTEGER,
            timestamp_iso TEXT,
            lang_detected TEXT,
            source_type   TEXT DEFAULT 'comment',  -- comment / live_chat
            like_count    INTEGER DEFAULT 0,
            reply_count   INTEGER DEFAULT 0,
            is_live       INTEGER DEFAULT 0,
            is_deleted    INTEGER DEFAULT 0,
            analyzed      INTEGER DEFAULT 0,
            FOREIGN KEY(video_id) REFERENCES videos(video_id)
        );

        CREATE TABLE IF NOT EXISTS user_profiles (
            channel_id        TEXT PRIMARY KEY,
            username          TEXT,
            display_name      TEXT,
            account_created   TEXT,
            subscriber_count  INTEGER DEFAULT -1,
            video_count       INTEGER DEFAULT -1,
            profile_url       TEXT,
            avatar_url        TEXT,
            threat_level      TEXT DEFAULT 'UNKNOWN',
            threat_score      REAL DEFAULT 0.0,
            bot_score         REAL DEFAULT 0.0,
            hate_score        REAL DEFAULT 0.0,
            stalker_score     REAL DEFAULT 0.0,
            antisemite_score  REAL DEFAULT 0.0,
            groyper_score     REAL DEFAULT 0.0,
            impersonator_score REAL DEFAULT 0.0,
            coord_score       REAL DEFAULT 0.0,
            message_count     INTEGER DEFAULT 0,
            first_seen        TEXT,
            last_seen         TEXT,
            identity_vector   TEXT DEFAULT '{}',  -- JSON
            cluster_id        INTEGER DEFAULT -1,
            linked_accounts   TEXT DEFAULT '[]',  -- JSON list of channel_ids
            tfidf_vector      TEXT DEFAULT NULL,  -- JSON
            temporal_hist     TEXT DEFAULT '{}',  -- JSON hourly histogram
            notes             TEXT DEFAULT '',
            flagged           INTEGER DEFAULT 0,
            flagged_at        TEXT,
            flagged_reason    TEXT,
            profile_inspected INTEGER DEFAULT 0,
            inspected_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS threat_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            comment_id    TEXT,
            channel_id    TEXT,
            video_id      TEXT,
            threat_type   TEXT,
            threat_score  REAL,
            details       TEXT,  -- JSON
            created_at    TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS identity_links (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_a  TEXT,
            channel_b  TEXT,
            sim_score  REAL,
            method     TEXT,  -- tfidf / embedding / ngram / temporal / combined
            confidence TEXT,  -- HIGH / MEDIUM / LOW
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(channel_a, channel_b)
        );

        CREATE TABLE IF NOT EXISTS clusters (
            cluster_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT,
            method       TEXT,  -- dbscan / louvain / spectral
            member_count INTEGER,
            threat_level TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS live_sessions (
            session_id    TEXT PRIMARY KEY,
            video_id      TEXT,
            started_at    TEXT,
            ended_at      TEXT,
            is_active     INTEGER DEFAULT 1,
            live_chat_id  TEXT,
            message_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS analysis_queue (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            comment_id TEXT UNIQUE,
            priority   INTEGER DEFAULT 0,
            added_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_comments_author ON comments(author_channel_id);
        CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id);
        CREATE INDEX IF NOT EXISTS idx_comments_ts ON comments(timestamp_utc);
        CREATE INDEX IF NOT EXISTS idx_comments_analyzed ON comments(analyzed);
        CREATE INDEX IF NOT EXISTS idx_threat_channel ON threat_events(channel_id);
        CREATE INDEX IF NOT EXISTS idx_comments_text ON comments(text);
        """)
        conn.commit()
        conn.close()
        log.info(f"Veritabanı hazır: {self.path}")

db = Database()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 5: YOUTUBE API İSTEMCİSİ
# ─────────────────────────────────────────────────────────────────────────────
class YouTubeAPIClient:
    def __init__(self):
        self.api_key   = Config.YT_API_KEY
        self.service   = None
        self.oauth_svc = None
        self._build_service()

    def _build_service(self):
        if not HAS_GOOGLE_API:
            log.warning("Google API kütüphanesi yok.")
            return
        if self.api_key:
            try:
                self.service = build("youtube", "v3", developerKey=self.api_key)
                log.info("YouTube API (API Key) bağlantısı kuruldu.")
            except Exception as e:
                log.error(f"YouTube API Key hatası: {e}")
        self._build_oauth()

    def _build_oauth(self):
        """OAuth2 — yorum silme için gerekli."""
        if not HAS_GOOGLE_API:
            return
        creds = None
        token_path = "token.pickle"
        if os.path.exists(token_path):
            try:
                with open(token_path, "rb") as f:
                    creds = pickle.load(f)
            except Exception:
                pass
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(GoogleRequest())
            except Exception:
                creds = None
        if not creds or not creds.valid:
            secrets = Config.YT_CLIENT_SECRETS
            if not os.path.exists(secrets):
                log.warning(f"OAuth2 client_secrets bulunamadı: {secrets} — yorum silme devre dışı.")
                return
            try:
                flow = InstalledAppFlow.from_client_secrets_file(secrets, Config.SCOPES)
                creds = flow.run_local_server(port=0)
                with open(token_path, "wb") as f:
                    pickle.dump(creds, f)
            except Exception as e:
                log.error(f"OAuth2 hatası: {e}")
                return
        try:
            self.oauth_svc = build("youtube", "v3", credentials=creds)
            log.info("YouTube OAuth2 bağlantısı kuruldu. Yorum silme aktif.")
        except Exception as e:
            log.error(f"OAuth servis hatası: {e}")

    def _api_request(self, func, **kwargs):
        """API isteği — hata yönetimi ile."""
        for attempt in range(3):
            try:
                return func(**kwargs).execute()
            except Exception as e:
                err = str(e)
                if "quotaExceeded" in err:
                    log.error("YouTube API kotası aşıldı!")
                    return None
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    log.error(f"API hatası: {e}")
                    return None

    def get_channel_id(self, handle: str) -> Optional[str]:
        """Handle → Channel ID."""
        if not self.service:
            return None
        r = self._api_request(
            self.service.channels().list,
            part="id",
            forHandle=handle
        )
        if r and r.get("items"):
            return r["items"][0]["id"]
        return None

    def get_channel_info(self, channel_id: str) -> Optional[Dict]:
        """Kanal bilgileri — oluşturma tarihi, abone sayısı."""
        if not self.service:
            return None
        r = self._api_request(
            self.service.channels().list,
            part="snippet,statistics",
            id=channel_id
        )
        if not r or not r.get("items"):
            return None
        item = r["items"][0]
        return {
            "channel_id":       channel_id,
            "display_name":     item["snippet"].get("title", ""),
            "account_created":  item["snippet"].get("publishedAt", ""),
            "subscriber_count": int(item["statistics"].get("subscriberCount", -1)),
            "video_count":      int(item["statistics"].get("videoCount", -1)),
            "profile_url":      f"https://youtube.com/channel/{channel_id}",
            "avatar_url":       item["snippet"].get("thumbnails", {}).get("default", {}).get("url", ""),
        }

    def get_channel_videos(self, channel_id: str, max_results: int = 200,
                           after_date: str = "2023-01-01T00:00:00Z") -> List[Dict]:
        """Kanal videolarını listele (2023-2026)."""
        if not self.service:
            return []
        videos = []
        page_token = None
        while len(videos) < max_results:
            params = dict(
                part="id,snippet",
                channelId=channel_id,
                maxResults=min(50, max_results - len(videos)),
                order="date",
                publishedAfter=after_date,
                type="video"
            )
            if page_token:
                params["pageToken"] = page_token
            r = self._api_request(self.service.search().list, **params)
            if not r:
                break
            for item in r.get("items", []):
                vid_id = item["id"].get("videoId")
                if not vid_id:
                    continue
                snip = item["snippet"]
                videos.append({
                    "video_id":    vid_id,
                    "title":       snip.get("title", ""),
                    "published_at": snip.get("publishedAt", ""),
                    "description": snip.get("description", ""),
                    "channel_id":  channel_id,
                    "video_type":  "stream" if "stream" in snip.get("title", "").lower()
                                   or "live" in snip.get("title", "").lower() else "video"
                })
            page_token = r.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.1)
        return videos

    def get_video_comments(self, video_id: str,
                           max_results: int = Config.MAX_COMMENTS_PER_VIDEO) -> List[Dict]:
        """Video yorumlarını çek."""
        if not self.service:
            return []
        comments = []
        page_token = None
        while len(comments) < max_results:
            params = dict(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                textFormat="plainText"
            )
            if page_token:
                params["pageToken"] = page_token
            r = self._api_request(self.service.commentThreads().list, **params)
            if not r:
                break
            for item in r.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "comment_id":         item["id"] + "_top",
                    "video_id":           video_id,
                    "author":             top.get("authorDisplayName", ""),
                    "author_channel_id":  top.get("authorChannelId", {}).get("value", ""),
                    "text":               top.get("textDisplay", ""),
                    "timestamp_utc":      int(datetime.datetime.fromisoformat(
                                            top.get("publishedAt","2023-01-01T00:00:00Z")
                                            .replace("Z","+00:00")).timestamp()),
                    "timestamp_iso":      top.get("publishedAt",""),
                    "like_count":         int(top.get("likeCount", 0)),
                    "reply_count":        int(item["snippet"].get("totalReplyCount", 0)),
                    "source_type":        "comment",
                    "is_live":            0
                })
                # Reply'lar
                if item["snippet"].get("totalReplyCount", 0) > 0:
                    replies_r = self._api_request(
                        self.service.comments().list,
                        part="snippet",
                        parentId=item["id"],
                        maxResults=20,
                        textFormat="plainText"
                    )
                    if replies_r:
                        for rep in replies_r.get("items", []):
                            rs = rep["snippet"]
                            comments.append({
                                "comment_id":        rep["id"],
                                "video_id":          video_id,
                                "author":            rs.get("authorDisplayName",""),
                                "author_channel_id": rs.get("authorChannelId",{}).get("value",""),
                                "text":              rs.get("textDisplay",""),
                                "timestamp_utc":     int(datetime.datetime.fromisoformat(
                                                      rs.get("publishedAt","2023-01-01T00:00:00Z")
                                                      .replace("Z","+00:00")).timestamp()),
                                "timestamp_iso":     rs.get("publishedAt",""),
                                "like_count":        int(rs.get("likeCount",0)),
                                "reply_count":       0,
                                "source_type":       "reply",
                                "is_live":           0
                            })
            page_token = r.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.05)
        return comments

    def get_live_chat_messages(self, live_chat_id: str,
                               page_token: str = None) -> Tuple[List[Dict], Optional[str], int]:
        """Canlı yayın sohbet mesajlarını çek."""
        if not self.service:
            return [], None, 5000
        params = dict(
            part="snippet,authorDetails",
            liveChatId=live_chat_id,
            maxResults=200
        )
        if page_token:
            params["pageToken"] = page_token
        r = self._api_request(self.service.liveChatMessages().list, **params)
        if not r:
            return [], None, 5000
        msgs = []
        for item in r.get("items", []):
            snip = item["snippet"]
            auth = item["authorDetails"]
            msgs.append({
                "comment_id":        item["id"],
                "author":            auth.get("displayName",""),
                "author_channel_id": auth.get("channelId",""),
                "text":              snip.get("displayMessage",""),
                "timestamp_utc":     int(datetime.datetime.fromisoformat(
                                      snip.get("publishedAt","2023-01-01T00:00:00Z")
                                      .replace("Z","+00:00")).timestamp()),
                "timestamp_iso":     snip.get("publishedAt",""),
                "source_type":       "live_chat",
                "is_live":           1,
                "like_count":        0,
                "reply_count":       0
            })
        next_token   = r.get("nextPageToken")
        poll_ms      = r.get("pollingIntervalMillis", 5000)
        return msgs, next_token, poll_ms

    def get_video_live_chat_id(self, video_id: str) -> Optional[str]:
        """Video → live chat ID."""
        if not self.service:
            return None
        r = self._api_request(
            self.service.videos().list,
            part="liveStreamingDetails",
            id=video_id
        )
        if r and r.get("items"):
            return r["items"][0].get("liveStreamingDetails", {}).get("activeLiveChatId")
        return None

    def delete_comment_api(self, comment_id: str) -> bool:
        """OAuth2 ile yorum sil."""
        if not self.oauth_svc:
            log.warning("OAuth2 bağlantısı yok, silme işlemi yapılamıyor.")
            return False
        try:
            self.oauth_svc.comments().delete(id=comment_id).execute()
            log.info(f"Yorum silindi (API): {comment_id}")
            return True
        except Exception as e:
            log.error(f"API ile silme hatası: {e}")
            return False

    def ban_user_from_live(self, live_chat_id: str, channel_id: str) -> bool:
        """Canlı yayından kullanıcı engelle."""
        if not self.oauth_svc:
            return False
        try:
            self.oauth_svc.liveChatBans().insert(
                part="snippet",
                body={"snippet": {
                    "liveChatId": live_chat_id,
                    "type": "permanent",
                    "bannedUserDetails": {"channelId": channel_id}
                }}
            ).execute()
            return True
        except Exception as e:
            log.error(f"Ban hatası: {e}")
            return False

yt_api = YouTubeAPIClient()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 5b: YT-DLP KAZIYICI — API Anahtarı Olmadan Video/Yorum Çekme
# ─────────────────────────────────────────────────────────────────────────────
class YTDLPScraper:
    """
    yt-dlp kullanarak YouTube Data API anahtarına ihtiyaç duymadan
    @ShmirchikArt/videos ve @ShmirchikArt/streams sayfalarından
    video listesi ve yorum çeker.
    Kurulum: pip install yt-dlp
    """

    def _run_ytdlp(self, args: List[str], timeout: int = 180) -> Tuple[str, str]:
        """yt-dlp komutunu çalıştır, (stdout, stderr) döndür."""
        import subprocess
        try:
            result = subprocess.run(
                ["yt-dlp"] + args,
                capture_output=True, text=True, timeout=timeout
            )
            return result.stdout, result.stderr
        except FileNotFoundError:
            log.error("yt-dlp bulunamadı. Kurulum: pip install yt-dlp")
            return "", "yt-dlp not found"
        except subprocess.TimeoutExpired:
            log.error(f"yt-dlp zaman aşımına uğradı ({timeout}s).")
            return "", "timeout"
        except Exception as e:
            log.error(f"yt-dlp hatası: {e}")
            return "", str(e)

    def get_channel_videos(self, video_type: str = "videos",
                           max_results: int = 200) -> List[Dict]:
        """
        https://www.youtube.com/@HANDLE/videos veya /streams adresinden
        video listesini çek. video_type: 'videos' | 'streams'
        """
        handle = Config.normalized_handle()
        url = f"https://www.youtube.com/@{handle}/{video_type}"
        log.info(f"yt-dlp ile video listesi çekiliyor: {url}")

        # Her satır için JSON formatında çıktı al
        print_tmpl = (
            '{"video_id":"%(id)s","title":"%(title)s",'
            '"published_at":"%(upload_date>%Y-%m-%dT00:00:00Z|unknown)s",'
            '"channel_id":"%(channel_id)s","description":"%(description)s"}'
        )
        stdout, stderr = self._run_ytdlp([
            "--flat-playlist",
            "--playlist-end", str(max_results),
            "--print", print_tmpl,
            "--no-warnings",
            "--quiet",
            url
        ])

        videos = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                v = json.loads(line)
                v["video_type"] = "stream" if video_type == "streams" else "video"
                if not v.get("channel_id"):
                    v["channel_id"] = Config.CHANNEL_ID or handle
                videos.append(v)
            except json.JSONDecodeError:
                log.debug(f"JSON parse hatası: {line[:80]}")

        log.info(f"yt-dlp → {len(videos)} {video_type} bulundu.")
        return videos

    def get_all_channel_content(self, max_results: int = 200) -> List[Dict]:
        """Hem /videos hem /streams listelerini birleştir."""
        videos  = self.get_channel_videos("videos",  max_results)
        streams = self.get_channel_videos("streams", max_results)
        seen = set()
        combined = []
        for v in videos + streams:
            vid = v.get("video_id","")
            if vid and vid not in seen:
                seen.add(vid)
                combined.append(v)
        log.info(f"Toplam benzersiz içerik: {len(combined)}")
        return combined

    def get_video_comments(self, video_id: str,
                           max_results: int = Config.MAX_COMMENTS_PER_VIDEO) -> List[Dict]:
        """
        yt-dlp ile tek bir videonun yorumlarını çek.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        log.info(f"yt-dlp ile yorumlar çekiliyor: {video_id}")

        stdout, stderr = self._run_ytdlp([
            "--write-comments",
            "--skip-download",
            "--print-json",
            "--no-warnings",
            "--quiet",
            f"--extractor-args", f"youtube:max_comments={max_results}",
            url
        ], timeout=300)

        comments = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                raw_comments = data.get("comments") or []
                for c in raw_comments:
                    ts = c.get("timestamp", 0) or 0
                    try:
                        ts_iso = (datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
                                  if ts else "")
                    except Exception:
                        ts_iso = ""
                    cid = c.get("id") or hashlib.md5(
                        f"{video_id}{c.get('text','')}{ts}".encode()).hexdigest()
                    parent = c.get("parent", "root")
                    comments.append({
                        "comment_id":        cid,
                        "video_id":          video_id,
                        "author":            c.get("author", ""),
                        "author_channel_id": c.get("author_id", ""),
                        "text":              c.get("text", ""),
                        "timestamp_utc":     int(ts),
                        "timestamp_iso":     ts_iso,
                        "like_count":        int(c.get("like_count", 0) or 0),
                        "reply_count":       0,
                        "source_type":       "reply" if parent != "root" else "comment",
                        "is_live":           0
                    })
                    if len(comments) >= max_results:
                        break
            except json.JSONDecodeError:
                log.debug(f"Yorum JSON parse hatası: {line[:80]}")

        log.info(f"yt-dlp → {len(comments)} yorum: {video_id}")
        return comments

    def _extract_live_chat_text(self, payload: Dict[str, Any]) -> str:
        """
        Canlı sohbet tekrarlarından metni NLP için normalize edilmiş şekilde çek.
        JSON3/replay renderer varyasyonlarını toleranslı işler.
        """
        runs = (((payload.get("replayChatItemAction") or {}).get("actions") or [{}])[0]
                .get("addChatItemAction", {})
                .get("item", {})
                .get("liveChatTextMessageRenderer", {})
                .get("message", {})
                .get("runs", []))
        if runs:
            text = "".join((r.get("text") or "") for r in runs).strip()
            return ta.clean_text(text)
        # VTT satırı / sade fallback
        fallback = payload.get("text") or payload.get("message") or ""
        return ta.clean_text(str(fallback))

    def get_stream_replay_chat_messages(self, video_id: str,
                                        max_results: int = Config.MAX_COMMENTS_PER_VIDEO) -> List[Dict]:
        """
        Canlı yayın tekrar sohbetini (live chat replay) yt-dlp ile al.
        NLP tabanlı otomasyon: metin temizleme + dil tespiti pipeline'ına uygun format.
        """
        import tempfile
        url = f"https://www.youtube.com/watch?v={video_id}"
        comments: List[Dict] = []
        with tempfile.TemporaryDirectory(prefix="ytg_chat_") as tmpdir:
            outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
            self._run_ytdlp([
                "--skip-download",
                "--write-subs",
                "--sub-langs", "live_chat",
                "--sub-format", "json3/vtt/best",
                "--output", outtmpl,
                "--no-warnings",
                "--quiet",
                url
            ], timeout=300)

            candidates = list(Path(tmpdir).glob(f"{video_id}*.live_chat*"))
            for fpath in candidates:
                try:
                    if fpath.suffix in (".json", ".json3"):
                        data = json.loads(fpath.read_text(encoding="utf-8", errors="ignore"))
                        events = data.get("events", []) if isinstance(data, dict) else []
                        for idx, event in enumerate(events):
                            text = self._extract_live_chat_text(event)
                            if not text:
                                continue
                            ts_ms = (event.get("replayChatItemAction", {})
                                         .get("videoOffsetTimeMsec")) or 0
                            ts = int(int(ts_ms) / 1000) if str(ts_ms).isdigit() else 0
                            ts_iso = (datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
                                      if ts else "")
                            msg_id = hashlib.md5(f"{video_id}:{idx}:{text[:32]}".encode()).hexdigest()
                            comments.append({
                                "comment_id":        msg_id,
                                "video_id":          video_id,
                                "author":            "",
                                "author_channel_id": "",
                                "text":              text,
                                "timestamp_utc":     ts,
                                "timestamp_iso":     ts_iso,
                                "like_count":        0,
                                "reply_count":       0,
                                "source_type":       "live_chat_replay",
                                "is_live":           0
                            })
                    elif fpath.suffix == ".vtt":
                        for idx, line in enumerate(fpath.read_text(encoding="utf-8", errors="ignore").splitlines()):
                            line = line.strip()
                            if (not line or "-->" in line or line.startswith("WEBVTT")
                                    or line.isdigit()):
                                continue
                            text = ta.clean_text(line)
                            if len(text) < 2:
                                continue
                            msg_id = hashlib.md5(f"{video_id}:vtt:{idx}:{text[:32]}".encode()).hexdigest()
                            comments.append({
                                "comment_id":        msg_id,
                                "video_id":          video_id,
                                "author":            "",
                                "author_channel_id": "",
                                "text":              text,
                                "timestamp_utc":     0,
                                "timestamp_iso":     "",
                                "like_count":        0,
                                "reply_count":       0,
                                "source_type":       "live_chat_replay",
                                "is_live":           0
                            })
                except Exception as e:
                    log.debug(f"Live chat replay parse hatası ({fpath.name}): {e}")

        unique, seen = [], set()
        for c in comments:
            key = (c["comment_id"], c["text"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
            if len(unique) >= max_results:
                break
        log.info(f"yt-dlp → {len(unique)} live chat replay mesajı: {video_id}")
        return unique

ytdlp_scraper = YTDLPScraper()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 6: SELENIUM MODÜLÜ (Profil İnceleme + Yedek Silme)
# ─────────────────────────────────────────────────────────────────────────────
class SeleniumModule:
    """Firefox ile YouTube profillerini inceler ve yorum siler (API yoksa)."""

    def __init__(self):
        self.driver = None
        self.logged_in = False

    def _start_driver(self, headless: bool = True):
        if not HAS_SELENIUM:
            raise RuntimeError("selenium kurulu değil.")
        opts = FFOptions()
        if headless:
            opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.set_preference("dom.webnotifications.enabled", False)
        self.driver = webdriver.Firefox(options=opts)
        self.driver.set_page_load_timeout(30)
        log.info("Firefox başlatıldı.")

    def login(self, headless: bool = False) -> bool:
        """YouTube'a giriş yap."""
        if not HAS_SELENIUM:
            return False
        email = Config.YT_EMAIL
        pwd   = Config.YT_PASS
        if not email or not pwd:
            log.error("YT_EMAIL veya YT_PASS .env dosyasında tanımlanmamış!")
            return False
        try:
            if not self.driver:
                self._start_driver(headless=headless)
            self.driver.get("https://accounts.google.com/signin/v2/identifier")
            wait = WebDriverWait(self.driver, 20)
            # E-posta
            email_field = wait.until(EC.presence_of_element_located((By.NAME, "identifier")))
            email_field.send_keys(email)
            self.driver.find_element(By.ID, "identifierNext").click()
            time.sleep(2)
            # Şifre
            pwd_field = wait.until(EC.element_to_be_clickable((By.NAME, "Passwd")))
            pwd_field.send_keys(pwd)
            self.driver.find_element(By.ID, "passwordNext").click()
            time.sleep(3)
            # YouTube'a git
            self.driver.get("https://www.youtube.com")
            time.sleep(2)
            self.logged_in = True
            log.info("YouTube girişi başarılı (Selenium).")
            return True
        except Exception as e:
            log.error(f"Selenium giriş hatası: {e}")
            return False

    def inspect_channel_profile(self, channel_url: str) -> Dict:
        """Kanal profilini ziyaret et — oluşturma tarihi, abone sayısı."""
        result = {
            "profile_url": channel_url,
            "account_created": "bilinmiyor",
            "subscriber_count": -1,
            "video_count": -1,
            "is_new_account": False,
            "account_age_days": -1,
        }
        if not HAS_SELENIUM or not self.driver:
            return result
        try:
            self.driver.get(channel_url + "/about")
            time.sleep(2)
            # Abone sayısı
            try:
                sub_el = self.driver.find_element(By.CSS_SELECTOR,
                    "#subscriber-count, [id='subscriber-count'], .yt-spec-button-shape-next__button-text-content")
                result["subscriber_count"] = sub_el.text.strip()
            except NoSuchElementException:
                pass
            # Tarih — "Katılma tarihi: 15 Mart 2024" gibi
            try:
                page_src = self.driver.page_source
                date_match = re.search(
                    r"(Joined|Katılma tarihi|Joined on)[:\s]+([A-Za-z0-9\s,]+\d{4})",
                    page_src, re.IGNORECASE
                )
                if date_match:
                    result["account_created"] = date_match.group(2).strip()
                    # Hesap yaşı hesapla
                    for fmt in ["%b %d, %Y", "%B %d, %Y", "%d %B %Y"]:
                        try:
                            created = datetime.datetime.strptime(result["account_created"], fmt)
                            age_days = (datetime.datetime.utcnow() - created).days
                            result["account_age_days"] = age_days
                            result["is_new_account"] = age_days < 90  # 3 aydan yeni
                            break
                        except ValueError:
                            continue
            except Exception:
                pass
        except Exception as e:
            log.error(f"Profil inceleme hatası: {e}")
        return result

    def delete_comment_selenium(self, video_url: str, comment_text_snippet: str) -> bool:
        """Selenium ile yorum sil (API yoksa yedek yöntem)."""
        if not HAS_SELENIUM or not self.logged_in:
            return False
        try:
            self.driver.get(video_url)
            time.sleep(3)
            # Yorumu bul ve 3 noktalı menüye tıkla
            comments = self.driver.find_elements(By.CSS_SELECTOR, "#content-text")
            for c in comments:
                if comment_text_snippet.lower()[:30] in c.text.lower():
                    # 3 nokta menüsü
                    parent = c.find_element(By.XPATH, "../../../..")
                    menu_btn = parent.find_element(By.CSS_SELECTOR,
                        "#action-menu button, ytd-menu-renderer button")
                    menu_btn.click()
                    time.sleep(1)
                    delete_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH,
                            "//yt-formatted-string[contains(text(),'Delete') or contains(text(),'Sil')]"))
                    )
                    delete_btn.click()
                    time.sleep(1)
                    confirm_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR,
                            "paper-button[dialog-confirm], yt-button-renderer[dialog-confirm]"))
                    )
                    confirm_btn.click()
                    time.sleep(1)
                    log.info(f"Yorum silindi (Selenium): {comment_text_snippet[:40]}")
                    return True
        except Exception as e:
            log.error(f"Selenium silme hatası: {e}")
        return False

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

selenium_module = SeleniumModule()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 7: METİN ANALİZ & NLP ARAÇLARI
# ─────────────────────────────────────────────────────────────────────────────
class TextAnalyzer:
    """Dil tespiti, normalleştirme, TF-IDF, N-gram, stilometri."""

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """NFKC normalleştirme — homograph saldırısı tespiti."""
        return unicodedata.normalize("NFKC", text).lower().strip()

    @staticmethod
    def detect_language(text: str) -> Tuple[str, float]:
        """Dil tespiti."""
        if HAS_LANGDETECT and len(text.strip()) > 10:
            try:
                lang = langdetect_detect(text)
                return lang, 0.85
            except Exception:
                pass
        # Script tespiti
        has_hebrew  = bool(re.search(r'[\u0590-\u05FF]', text))
        has_arabic  = bool(re.search(r'[\u0600-\u06FF]', text))
        has_cyrillic= bool(re.search(r'[\u0400-\u04FF]', text))
        has_latin   = bool(re.search(r'[a-zA-Z]', text))
        if has_hebrew:  return "he", 0.70
        if has_arabic:  return "ar", 0.70
        if has_cyrillic:return "ru", 0.70
        if has_latin:   return "en", 0.50
        return "unknown", 0.30

    @staticmethod
    def extract_emojis(text: str) -> List[str]:
        return [c for c in text if unicodedata.category(c) in ("So", "Sm") or
                ord(c) > 0x1F000]

    @staticmethod
    def shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        freq = collections.Counter(text)
        total = len(text)
        return -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)

    @staticmethod
    def lexical_diversity(text: str) -> float:
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def ngram_fingerprint(text: str, n: int = 2) -> collections.Counter:
        tokens = text.lower().split()
        grams  = zip(*[tokens[i:] for i in range(n)])
        return collections.Counter(" ".join(g) for g in grams)

    @staticmethod
    def jaccard_similarity(counter_a: collections.Counter,
                           counter_b: collections.Counter) -> float:
        keys_a = set(counter_a.keys())
        keys_b = set(counter_b.keys())
        intersection = keys_a & keys_b
        union        = keys_a | keys_b
        if not union:
            return 0.0
        return len(intersection) / len(union)

    @staticmethod
    def typo_fingerprint(messages: List[str]) -> Dict:
        full = " ".join(messages)
        words = full.split()
        return {
            "caps_ratio":      sum(1 for c in full if c.isupper()) / max(1, len(full)),
            "punct_density":   sum(1 for c in full if c in "!?.,;:") / max(1, len(full)),
            "ellipsis_ratio":  full.count("...") / max(1, len(messages)),
            "exclaim_ratio":   full.count("!") / max(1, len(messages)),
            "double_letters":  len(re.findall(r"(\w)\1{2,}", full)),
            "avg_word_len":    sum(len(w) for w in words) / max(1, len(words)),
            "emoji_density":   sum(1 for c in full if unicodedata.category(c) in ("So","Sm")
                                   or ord(c) > 0x1F000) / max(1, len(messages)),
        }

    @staticmethod
    def cosine_sim_vectors(v1: List[float], v2: List[float]) -> float:
        a, b = np.array(v1), np.array(v2)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def burrows_delta(texts_a: List[str], texts_b: List[str],
                      top_n: int = 100) -> float:
        """Basitleştirilmiş Burrows Delta."""
        def word_freq(texts):
            words = " ".join(texts).lower().split()
            total = len(words)
            if total == 0:
                return {}
            return {w: c/total for w, c in collections.Counter(words).most_common(top_n)}

        fa = word_freq(texts_a)
        fb = word_freq(texts_b)
        all_words = list(set(fa) | set(fb))
        if not all_words:
            return 1.0
        fa_v = np.array([fa.get(w, 0) for w in all_words])
        fb_v = np.array([fb.get(w, 0) for w in all_words])
        mu   = (fa_v + fb_v) / 2
        sigma= np.std(np.vstack([fa_v, fb_v]), axis=0) + 1e-8
        z_a  = (fa_v - mu) / sigma
        z_b  = (fb_v - mu) / sigma
        return float(np.mean(np.abs(z_a - z_b)))

ta = TextAnalyzer()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 8: TEHDİT ALGILAMA
# ─────────────────────────────────────────────────────────────────────────────
class ThreatDetector:
    """Nefret söylemi, bot, stalker, koordineli saldırı tespiti."""

    def __init__(self):
        self._bart = None
        self._sbert = None

    def _load_bart(self):
        if self._bart is None:
            try:
                from transformers import pipeline as hf_pipeline
                self._bart = hf_pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1  # CPU; GPU: 0
                )
                log.info("BART Zero-shot yüklendi.")
            except Exception as e:
                log.warning(f"BART yüklenemedi ({e}) — sadece anahtar kelime analizi kullanılacak.")
                self._bart = "unavailable"

    def _load_sbert(self):
        if self._sbert is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sbert = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
                log.info("Sentence-BERT yüklendi.")
            except Exception as e:
                log.warning(f"SBERT yüklenemedi ({e})")
                self._sbert = "unavailable"

    def keyword_score(self, text: str, lang: str = "en") -> Dict[str, float]:
        """Anahtar kelime tabanlı tehdit skoru."""
        text_l = text.lower()
        scores = {cat: 0.0 for cat in ["antisemite", "hater", "stalker", "bot", "groyper"]}

        # Anti-semitizm
        antisemite_kws = (Config.HATE_KEYWORDS["antisemite_en"] +
                          Config.HATE_KEYWORDS["antisemite_tr"] +
                          Config.HATE_KEYWORDS.get(f"antisemite_{lang}", []))
        hits = sum(1 for kw in antisemite_kws if kw.lower() in text_l)
        scores["antisemite"] = min(1.0, hits * 0.35)

        # Groyper
        groyper_hits = sum(1 for kw in Config.HATE_KEYWORDS["groyper_signals"] if kw.lower() in text_l)
        scores["groyper"] = min(1.0, groyper_hits * 0.50)
        scores["hater"] = max(scores["hater"], scores["groyper"] * 0.7)

        # Genel nefret
        hate_kws = (Config.HATE_KEYWORDS["hate_general_en"] +
                    Config.HATE_KEYWORDS["hate_general_tr"])
        hits_h = sum(1 for kw in hate_kws if kw.lower() in text_l)
        scores["hater"] = max(scores["hater"], min(1.0, hits_h * 0.30))

        # Stalker
        stalk_hits = sum(1 for kw in Config.HATE_KEYWORDS["stalker_signals"] if kw.lower() in text_l)
        scores["stalker"] = min(1.0, stalk_hits * 0.45)

        # Bot/spam
        bot_hits = sum(1 for kw in Config.HATE_KEYWORDS["bot_spam"] if kw.lower() in text_l)
        scores["bot"] = min(1.0, bot_hits * 0.40)

        return scores

    def bart_zero_shot(self, text: str) -> Dict[str, float]:
        """BART ile zero-shot sınıflandırma."""
        self._load_bart()
        if self._bart == "unavailable" or not text.strip():
            return {}
        labels = [
            "antisemitic content",
            "hate speech",
            "harassment and stalking",
            "bot-generated spam",
            "groyper or white nationalist content",
            "neutral friendly message",
            "coordinated attack"
        ]
        label_map = {
            "antisemitic content": "antisemite",
            "hate speech": "hater",
            "harassment and stalking": "stalker",
            "bot-generated spam": "bot",
            "groyper or white nationalist content": "groyper",
            "neutral friendly message": "normal",
            "coordinated attack": "coordinated"
        }
        try:
            r = self._bart(text[:512], candidate_labels=labels)
            return {label_map[l]: s for l, s in zip(r["labels"], r["scores"])}
        except Exception as e:
            log.error(f"BART hatası: {e}")
            return {}

    def bot_score(self, messages: List[Dict]) -> float:
        """Burstiness + heuristik bot skoru."""
        if not messages:
            return 0.0
        texts = [m["text"] for m in messages]
        full_text = " ".join(texts)

        # Lexical diversity
        D = ta.lexical_diversity(full_text)
        # Shannon entropy
        H = ta.shannon_entropy(full_text) / 4.5
        # Ortalama uzunluk
        avg_len = sum(len(t) for t in texts) / max(1, len(texts))
        L = min(1.0, avg_len / 80)
        # Soru oranı
        Q = sum(1 for t in texts if "?" in t) / max(1, len(texts))
        # Noktalama
        P = sum(1 for c in full_text if c in "!?,.:;") / max(1, len(full_text))
        # Emoji
        E = sum(1 for c in full_text if unicodedata.category(c) in ("So","Sm") or
                ord(c) > 0x1F000) / max(1, len(full_text)) * 10
        E = min(1.0, E)
        # Büyük harf
        U = sum(1 for c in full_text if c.isupper()) / max(1, len(full_text))
        # Tekrar skoru
        if len(texts) > 1:
            same = sum(1 for i in range(1, len(texts)) if
                       texts[i].lower()[:30] == texts[i-1].lower()[:30])
            R = 1.0 - same / (len(texts) - 1)
        else:
            R = 1.0

        # Burstiness
        if len(messages) >= 2:
            dts = [messages[i+1]["timestamp_utc"] - messages[i]["timestamp_utc"]
                   for i in range(len(messages)-1) if messages[i+1]["timestamp_utc"] > 0]
            if dts:
                mu_dt = np.mean(dts)
                si_dt = np.std(dts)
                B = (si_dt - mu_dt) / max(1, si_dt + mu_dt)  # -1 ile +1 arası
            else:
                B = 0.0
        else:
            B = 0.0

        heuristic = 1.0 - (
            0.28 * D + 0.18 * H + 0.12 * L +
            0.10 * Q + 0.10 * P + 0.07 * E +
            0.05 * (1 - U) + 0.10 * R
        )
        # Çok düzenli mesaj → bot; burstiness düzeltmesi
        if B < -0.5:   # düzenli bot
            heuristic = max(heuristic, 0.75)
        elif B > 0.8:  # koordineli saldırı
            heuristic = max(heuristic, 0.65)

        return max(0.0, min(1.0, heuristic))

    def stalker_score(self, messages: List[Dict], channel_owner_id: str = "") -> float:
        """Stalker davranışı skoru."""
        if len(messages) < 3:
            return 0.0
        texts = [m["text"] for m in messages]
        full = " ".join(texts)
        kw_score = ta.TextAnalyzer.keyword_score(self, full).get("stalker", 0.0) \
            if hasattr(self, '_keyword_cache') else self.keyword_score(full).get("stalker", 0.0)

        # Kanal sahibine doğrudan atıf
        if channel_owner_id:
            direct_refs = sum(1 for t in texts if channel_owner_id.lower() in t.lower()
                              or "you" in t.lower() or "sen" in t.lower())
            ref_ratio = direct_refs / len(texts)
        else:
            ref_ratio = 0.0

        # Sık tekrar
        repeat_ratio = 1.0 - ta.lexical_diversity(full)

        score = 0.40 * kw_score + 0.35 * repeat_ratio + 0.25 * ref_ratio
        return min(1.0, score)

    def analyze_comment(self, text: str, lang: str = "en",
                        use_bart: bool = False) -> Dict[str, Any]:
        """Tek yorum analizi."""
        kw = self.keyword_score(text, lang)
        result = {
            "antisemite_score":  kw.get("antisemite", 0.0),
            "hate_score":        kw.get("hater", 0.0),
            "stalker_score":     kw.get("stalker", 0.0),
            "bot_score":         kw.get("bot", 0.0),
            "groyper_score":     kw.get("groyper", 0.0),
            "bart_scores":       {},
            "analysis_method":   "keyword"
        }
        if use_bart:
            bart = self.bart_zero_shot(text)
            if bart:
                result["antisemite_score"] = max(result["antisemite_score"],
                    bart.get("antisemite", 0.0) * 0.55)
                result["hate_score"]  = max(result["hate_score"],
                    bart.get("hater", 0.0) * 0.55)
                result["stalker_score"] = max(result["stalker_score"],
                    bart.get("stalker", 0.0) * 0.55)
                result["bart_scores"] = bart
                result["analysis_method"] = "keyword+bart"

        # Genel tehdit skoru
        result["threat_score"] = max(
            result["antisemite_score"],
            result["hate_score"],
            result["stalker_score"],
            result["groyper_score"],
            result["bot_score"] * 0.7
        )

        # Tehdit seviyesi
        s = result["threat_score"]
        a = result["antisemite_score"]
        g = result["groyper_score"]
        b = result["bot_score"]
        st= result["stalker_score"]
        if a >= Config.HATE_THRESHOLD or g >= Config.HATE_THRESHOLD:
            result["threat_level"] = "ANTISEMITE"
        elif b >= Config.BOT_THRESHOLD:
            result["threat_level"] = "BOT"
        elif st >= Config.STALK_THRESHOLD:
            result["threat_level"] = "STALKER"
        elif result["hate_score"] >= Config.HATE_THRESHOLD:
            result["threat_level"] = "HATER"
        elif s >= 0.25:
            result["threat_level"] = "SUSPICIOUS"
        else:
            result["threat_level"] = "NORMAL"
        return result

detector = ThreatDetector()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 9: KİMLİK BAĞLANTISI & KÜMELEME
# ─────────────────────────────────────────────────────────────────────────────
class IdentityLinker:
    """TF-IDF + N-gram + zamansal + stilometri ile hesap eşleştirme."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=5000,
            sublinear_tf=True
        ) if HAS_SKLEARN else None
        self._fitted = False

    def build_user_corpus(self) -> Tuple[List[str], List[str]]:
        """Tüm kullanıcılar için metin korpusu oluştur."""
        rows = db.fetchall("""
            SELECT author_channel_id, GROUP_CONCAT(text, ' ') as corpus
            FROM comments
            WHERE author_channel_id != ''
            GROUP BY author_channel_id
            HAVING COUNT(*) >= 3
        """)
        channel_ids = [r["author_channel_id"] for r in rows]
        corpora     = [ta.normalize_unicode(r["corpus"] or "") for r in rows]
        return channel_ids, corpora

    def fit_tfidf(self, corpora: List[str]):
        if not HAS_SKLEARN or not corpora:
            return
        try:
            self.vectorizer.fit(corpora)
            self._fitted = True
        except Exception as e:
            log.error(f"TF-IDF fit hatası: {e}")

    def get_similarity_matrix(self, channel_ids: List[str],
                              corpora: List[str]) -> np.ndarray:
        """N×N benzerlik matrisi."""
        if not HAS_SKLEARN or not self._fitted or not corpora:
            return np.eye(len(channel_ids))
        try:
            X = self.vectorizer.transform(corpora)
            return cosine_similarity(X)
        except Exception as e:
            log.error(f"Benzerlik matrisi hatası: {e}")
            return np.eye(len(channel_ids))

    def temporal_similarity(self, ch_a: str, ch_b: str) -> float:
        """Saatlik aktivite histogramı benzerliği."""
        def get_hist(cid):
            rows = db.fetchall(
                "SELECT timestamp_utc FROM comments WHERE author_channel_id=?", (cid,))
            hist = np.zeros(24)
            for r in rows:
                ts = r["timestamp_utc"]
                if ts:
                    hour = datetime.datetime.utcfromtimestamp(ts).hour
                    hist[hour] += 1
            total = hist.sum()
            return hist / total if total > 0 else hist
        h_a = get_hist(ch_a)
        h_b = get_hist(ch_b)
        return float(1.0 - np.sum(np.abs(h_a - h_b)) / 2.0)

    def find_links(self, threshold: float = Config.SIM_THRESHOLD) -> List[Dict]:
        """Tüm hesap bağlantılarını bul."""
        channel_ids, corpora = self.build_user_corpus()
        if len(channel_ids) < 2:
            return []
        self.fit_tfidf(corpora)
        sim_matrix = self.get_similarity_matrix(channel_ids, corpora)
        links = []
        n = len(channel_ids)
        for i in range(n):
            for j in range(i+1, n):
                s = float(sim_matrix[i][j])
                if s >= threshold:
                    # N-gram benzerliği
                    ng_a = ta.ngram_fingerprint(corpora[i])
                    ng_b = ta.ngram_fingerprint(corpora[j])
                    ng_sim = ta.jaccard_similarity(ng_a, ng_b)
                    # Zamansal benzerlik
                    t_sim = self.temporal_similarity(channel_ids[i], channel_ids[j])
                    # Bileşik skor
                    combined = (Config.W_EMBED * s + Config.W_NGRAM * ng_sim +
                                Config.W_TIME * t_sim)
                    conf = "HIGH" if combined >= 0.80 else "MEDIUM" if combined >= 0.65 else "LOW"
                    links.append({
                        "channel_a": channel_ids[i],
                        "channel_b": channel_ids[j],
                        "tfidf_sim": round(s, 3),
                        "ngram_sim": round(ng_sim, 3),
                        "time_sim":  round(t_sim, 3),
                        "combined":  round(combined, 3),
                        "confidence": conf,
                        "method":    "tfidf+ngram+temporal"
                    })
                    # DB kaydet
                    try:
                        db.execute("""
                            INSERT OR REPLACE INTO identity_links
                            (channel_a, channel_b, sim_score, method, confidence)
                            VALUES (?,?,?,?,?)
                        """, (channel_ids[i], channel_ids[j], combined,
                              "tfidf+ngram+temporal", conf))
                    except Exception:
                        pass
        return links

    def cluster_users(self, threshold: float = Config.SIM_THRESHOLD) -> Dict[str, int]:
        """DBSCAN ile kullanıcı kümeleme."""
        channel_ids, corpora = self.build_user_corpus()
        if len(channel_ids) < 3 or not HAS_SKLEARN:
            return {}
        self.fit_tfidf(corpora)
        sim_matrix = self.get_similarity_matrix(channel_ids, corpora)
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        try:
            labels = DBSCAN(
                eps=1.0 - threshold,
                min_samples=2,
                metric="precomputed"
            ).fit_predict(dist_matrix.astype(np.float64))
        except Exception as e:
            log.error(f"DBSCAN hatası: {e}")
            return {}
        result = {}
        for cid, label in zip(channel_ids, labels):
            result[cid] = int(label)
            db.execute(
                "UPDATE user_profiles SET cluster_id=? WHERE channel_id=?",
                (int(label), cid)
            )
        return result

    def build_network_graph(self) -> Dict:
        """NetworkX ile ilişki grafiği oluştur."""
        if not HAS_NX:
            return {"nodes": [], "edges": []}
        links = db.fetchall(
            "SELECT channel_a, channel_b, sim_score, confidence FROM identity_links "
            "WHERE sim_score >= ?", (Config.SIM_THRESHOLD,)
        )
        users = db.fetchall(
            "SELECT channel_id, username, threat_level, threat_score, cluster_id "
            "FROM user_profiles"
        )
        G = nx.Graph()
        user_map = {u["channel_id"]: u for u in users}

        for u in users:
            G.add_node(u["channel_id"], **u)
        for l in links:
            G.add_edge(l["channel_a"], l["channel_b"],
                       weight=l["sim_score"], confidence=l["confidence"])

        # Louvain topluluk tespiti
        try:
            import community as louvain_comm
            partition = louvain_comm.best_partition(G)
        except Exception:
            partition = {n: i for i, n in enumerate(G.nodes())}

        nodes_data = []
        for node in G.nodes():
            u = user_map.get(node, {})
            nodes_data.append({
                "id":          node,
                "label":       u.get("username", node[:12]),
                "threat":      u.get("threat_level", "UNKNOWN"),
                "score":       u.get("threat_score", 0),
                "cluster":     partition.get(node, -1),
                "color":       Config.THREAT_COLORS.get(u.get("threat_level","NORMAL"),
                                                        "#888888"),
                "size":        max(8, int(u.get("message_count", 1) ** 0.5) * 3)
            })
        edges_data = [
            {"source": e[0], "target": e[1],
             "weight": G[e[0]][e[1]].get("weight", 0.5)}
            for e in G.edges()
        ]
        return {"nodes": nodes_data, "edges": edges_data}

identity_linker = IdentityLinker()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 10: VERİ TOPLAMA & DEPOLAMA
# ─────────────────────────────────────────────────────────────────────────────
class DataCollector:
    """Videolardan/stream'lerden yorum toplama + DB kaydetme."""

    def save_video(self, v: Dict):
        db.execute("""
            INSERT OR REPLACE INTO videos
            (video_id, title, published_at, video_type, description, channel_id, fetched_at)
            VALUES (?,?,?,?,?,?,?)
        """, (v["video_id"], v.get("title",""), v.get("published_at",""),
              v.get("video_type","video"), v.get("description",""),
              v.get("channel_id",""), datetime.datetime.utcnow().isoformat()))

    def save_comment(self, c: Dict):
        text = c.get("text","")
        lang, _ = ta.detect_language(text)
        msg_id = hashlib.sha256(
            (c.get("comment_id","") + text).encode()
        ).hexdigest()[:16]
        db.execute("""
            INSERT OR IGNORE INTO comments
            (comment_id, video_id, author, author_channel_id, text,
             timestamp_utc, timestamp_iso, lang_detected, source_type,
             like_count, reply_count, is_live)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (c.get("comment_id", msg_id),
              c.get("video_id",""),
              c.get("author",""),
              c.get("author_channel_id",""),
              text,
              c.get("timestamp_utc",0),
              c.get("timestamp_iso",""),
              lang,
              c.get("source_type","comment"),
              c.get("like_count",0),
              c.get("reply_count",0),
              c.get("is_live",0)))
        # Kullanıcı profili güncelle
        if c.get("author_channel_id"):
            self._upsert_user(c)

    def _upsert_user(self, c: Dict):
        cid = c["author_channel_id"]
        existing = db.fetchone(
            "SELECT channel_id FROM user_profiles WHERE channel_id=?", (cid,))
        now = datetime.datetime.utcnow().isoformat()
        if not existing:
            db.execute("""
                INSERT OR IGNORE INTO user_profiles
                (channel_id, username, display_name, first_seen, last_seen, message_count)
                VALUES (?,?,?,?,?,1)
            """, (cid, ta.normalize_username(c.get("author","")),
                  c.get("author",""), now, now))
        else:
            db.execute("""
                UPDATE user_profiles
                SET last_seen=?, message_count=message_count+1,
                    username=COALESCE(NULLIF(username,''), ?)
                WHERE channel_id=?
            """, (now, c.get("author",""), cid))

    def collect_channel_dataset(self, channel_id: str,
                                after_date: str = "2023-01-01T00:00:00Z",
                                progress_cb=None) -> Dict:
        """
        Kanal için 2023-2026 tüm video yorumlarını topla.
        Öncelik: YouTube Data API (varsa) → yt-dlp (API anahtarı gerekmez).
        """
        log.info(f"Veri toplama başlıyor: {channel_id}")
        stats = {"videos": 0, "comments": 0, "live_chat_replay": 0, "errors": 0, "source": ""}

        # ── Video listesi ──────────────────────────────────────────────────
        use_api = bool(yt_api.service)
        if use_api:
            log.info("Kaynak: YouTube Data API")
            stats["source"] = "youtube_api"
            videos = yt_api.get_channel_videos(channel_id, max_results=500,
                                               after_date=after_date)
        else:
            log.info("Kaynak: yt-dlp (API anahtarı yok — /videos + /streams çekiliyor)")
            stats["source"] = "ytdlp"
            videos = ytdlp_scraper.get_all_channel_content(max_results=500)
            # after_date filtresi uygula
            filtered = []
            cutoff = after_date[:10]  # "2023-01-01"
            for v in videos:
                pub = (v.get("published_at") or "")[:10]
                if pub >= cutoff or pub == "unknown":
                    filtered.append(v)
            videos = filtered

        stats["videos"] = len(videos)
        log.info(f"{len(videos)} içerik bulundu.")

        # ── Her video için yorum çek ────────────────────────────────────────
        for i, v in enumerate(videos):
            try:
                self.save_video(v)
                if use_api:
                    comments = yt_api.get_video_comments(v["video_id"])
                else:
                    comments = ytdlp_scraper.get_video_comments(v["video_id"])
                # 2023-2026 stream replay sohbetlerini NLP hattına dahil et
                if v.get("video_type") == "stream":
                    replay_msgs = ytdlp_scraper.get_stream_replay_chat_messages(v["video_id"])
                    comments.extend(replay_msgs)
                    stats["live_chat_replay"] += len(replay_msgs)
                for c in comments:
                    c["video_id"] = v["video_id"]
                    self.save_comment(c)
                db.execute(
                    "UPDATE videos SET comment_count=? WHERE video_id=?",
                    (len(comments), v["video_id"])
                )
                stats["comments"] += len(comments)
                if progress_cb:
                    progress_cb(i+1, len(videos), v.get("title",""), len(comments))
                log.info(f"  [{i+1}/{len(videos)}] {v.get('title','')[:50]} — {len(comments)} yorum")
            except Exception as e:
                stats["errors"] += 1
                log.error(f"Video hatası ({v.get('video_id','?')}): {e}")
            # yt-dlp için ekstra bekleme gerekmiyor; API rate-limit yoksa hızlandır
            time.sleep(0.5 if not use_api else 0.2)

        log.info(f"Toplama tamamlandı: {stats}")
        return stats

    def collect_stream_replays_from_channel(self,
                                            streams_url: str = "https://www.youtube.com/@ShmirchikArt/streams",
                                            year_start: int = 2023,
                                            year_end: int = 2026) -> Dict[str, int]:
        """
        Verilen /streams adresinden canlı yayın tekrar sohbet verilerini topla.
        Yıllar: [year_start, year_end] aralığı.
        """
        videos = ytdlp_scraper.get_channel_videos("streams", max_results=1000)
        stats = {"streams": 0, "messages": 0, "errors": 0}
        for v in videos:
            published = (v.get("published_at") or "")[:4]
            year = int(published) if published.isdigit() else None
            if year is not None and not (year_start <= year <= year_end):
                continue
            try:
                self.save_video(v)
                replay_msgs = ytdlp_scraper.get_stream_replay_chat_messages(v["video_id"])
                for c in replay_msgs:
                    c["video_id"] = v["video_id"]
                    self.save_comment(c)
                db.execute(
                    "UPDATE videos SET comment_count=comment_count+? WHERE video_id=?",
                    (len(replay_msgs), v["video_id"])
                )
                stats["streams"] += 1
                stats["messages"] += len(replay_msgs)
            except Exception:
                stats["errors"] += 1
        log.info(f"Stream replay toplama tamamlandı ({streams_url}): {stats}")
        return stats

    def inspect_user_profiles(self, limit: int = 50) -> int:
        """API ile kullanıcı profillerini güncelle (oluşturma tarihi, abone)."""
        users = db.fetchall("""
            SELECT channel_id FROM user_profiles
            WHERE profile_inspected=0 AND channel_id != ''
            LIMIT ?
        """, (limit,))
        count = 0
        for u in users:
            cid = u["channel_id"]
            info = yt_api.get_channel_info(cid)
            if info:
                age_days = -1
                is_new = False
                created = info.get("account_created","")
                if created:
                    try:
                        dt = datetime.datetime.fromisoformat(created.replace("Z","+00:00"))
                        age_days = (datetime.datetime.now(datetime.timezone.utc) - dt).days
                        is_new = age_days < 90
                    except Exception:
                        pass
                db.execute("""
                    UPDATE user_profiles
                    SET display_name=?, account_created=?, subscriber_count=?,
                        video_count=?, avatar_url=?, profile_url=?,
                        profile_inspected=1, inspected_at=?
                    WHERE channel_id=?
                """, (info["display_name"], info["account_created"],
                      info["subscriber_count"], info["video_count"],
                      info["avatar_url"], info["profile_url"],
                      datetime.datetime.utcnow().isoformat(), cid))
                count += 1
            time.sleep(0.1)
        return count

collector = DataCollector()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 11: ANALİZ MOTORU
# ─────────────────────────────────────────────────────────────────────────────
class AnalysisEngine:
    """Tüm yorumları analiz et, profilleri güncelle."""

    def analyze_pending(self, batch_size: int = 100) -> int:
        """Bekleyen yorumları analiz et."""
        comments = db.fetchall("""
            SELECT comment_id, text, lang_detected, author_channel_id
            FROM comments
            WHERE analyzed=0
            LIMIT ?
        """, (batch_size,))
        if not comments:
            return 0
        for c in comments:
            try:
                text = c["text"] or ""
                lang = c["lang_detected"] or "en"
                result = detector.analyze_comment(text, lang)
                db.execute("UPDATE comments SET analyzed=1 WHERE comment_id=?",
                           (c["comment_id"],))
                cid = c["author_channel_id"]
                if cid:
                    # Profil skorlarını güncelle
                    existing = db.fetchone(
                        "SELECT * FROM user_profiles WHERE channel_id=?", (cid,))
                    if existing:
                        new_hate = max(float(existing.get("hate_score",0)),
                                       result["hate_score"])
                        new_anti = max(float(existing.get("antisemite_score",0)),
                                       result["antisemite_score"])
                        new_stalk= max(float(existing.get("stalker_score",0)),
                                       result["stalker_score"])
                        new_groyp= max(float(existing.get("groyper_score",0)),
                                       result["groyper_score"])
                        new_threat= max(new_hate, new_anti, new_stalk, new_groyp)
                        threat_lv = result["threat_level"]
                        db.execute("""
                            UPDATE user_profiles
                            SET hate_score=?, antisemite_score=?, stalker_score=?,
                                groyper_score=?, threat_score=?, threat_level=?
                            WHERE channel_id=?
                        """, (new_hate, new_anti, new_stalk, new_groyp,
                              new_threat, threat_lv, cid))
                        # Tehdit olayı kaydet
                        if new_threat >= Config.HATE_THRESHOLD:
                            video_row = db.fetchone(
                                "SELECT video_id FROM comments WHERE comment_id=?",
                                (c["comment_id"],)
                            ) or {}
                            db.execute("""
                                INSERT INTO threat_events
                                (comment_id, channel_id, video_id, threat_type,
                                 threat_score, details)
                                VALUES (?,?,?,?,?,?)
                            """, (
                                c["comment_id"],
                                cid,
                                video_row.get("video_id", ""),
                                threat_lv,
                                new_threat,
                                json.dumps(result, ensure_ascii=False)
                            ))
            except Exception as e:
                log.error(f"Analiz hatası ({c['comment_id']}): {e}")
        return len(comments)

    def run_bot_detection(self, batch_size: int = 200) -> int:
        """Tüm kullanıcılar için bot skoru hesapla."""
        users = db.fetchall("""
            SELECT channel_id FROM user_profiles
            WHERE message_count >= 5
            ORDER BY message_count DESC
            LIMIT ?
        """, (batch_size,))
        count = 0
        for u in users:
            cid = u["channel_id"]
            messages = db.fetchall("""
                SELECT text, timestamp_utc FROM comments
                WHERE author_channel_id=?
                ORDER BY timestamp_utc ASC
            """, (cid,))
            if len(messages) < 3:
                continue
            b_score = detector.bot_score(messages)
            db.execute(
                "UPDATE user_profiles SET bot_score=? WHERE channel_id=?",
                (b_score, cid)
            )
            if b_score >= Config.BOT_THRESHOLD:
                existing = db.fetchone(
                    "SELECT threat_level FROM user_profiles WHERE channel_id=?", (cid,))
                if existing and existing["threat_level"] in ("NORMAL","UNKNOWN"):
                    db.execute(
                        "UPDATE user_profiles SET threat_level='BOT', threat_score=? "
                        "WHERE channel_id=?", (b_score, cid))
            count += 1
        return count

    def run_stalker_detection(self, batch_size: int = 200) -> int:
        """Stalker skoru hesapla."""
        users = db.fetchall("""
            SELECT channel_id FROM user_profiles
            WHERE message_count >= 5
            LIMIT ?
        """, (batch_size,))
        channel_id = Config.CHANNEL_ID
        count = 0
        for u in users:
            cid = u["channel_id"]
            messages = db.fetchall("""
                SELECT text, timestamp_utc FROM comments
                WHERE author_channel_id=?
                ORDER BY timestamp_utc ASC
            """, (cid,))
            if len(messages) < 3:
                continue
            s_score = detector.stalker_score(messages, channel_id)
            db.execute(
                "UPDATE user_profiles SET stalker_score=? WHERE channel_id=?",
                (s_score, cid)
            )
            if s_score >= Config.STALK_THRESHOLD:
                db.execute(
                    "UPDATE user_profiles SET threat_level='STALKER', threat_score=MAX(threat_score,?) "
                    "WHERE channel_id=? AND threat_score < ?",
                    (s_score, cid, s_score))
            count += 1
        return count

    def full_analysis_run(self) -> Dict:
        """Tam analiz döngüsü."""
        log.info("Tam analiz başlıyor...")
        stats = {}
        # 1. Yorumları analiz et
        total = 0
        while True:
            n = self.analyze_pending(200)
            total += n
            if n == 0:
                break
        stats["comments_analyzed"] = total
        # 2. Bot tespiti
        stats["bot_detection"] = self.run_bot_detection(500)
        # 3. Stalker tespiti
        stats["stalker_detection"] = self.run_stalker_detection(500)
        # 4. Kimlik eşleştirme
        links = identity_linker.find_links()
        stats["identity_links"] = len(links)
        # 5. Kümeleme
        clusters = identity_linker.cluster_users()
        stats["clusters"] = len(set(clusters.values()) - {-1})
        log.info(f"Analiz tamamlandı: {stats}")
        return stats

engine = AnalysisEngine()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 12: CANLI YAYIN MONİTÖRÜ
# ─────────────────────────────────────────────────────────────────────────────
class LiveMonitor:
    """Canlı yayın sohbet monitörü — WebSocket ile web paneline iletir."""

    def __init__(self):
        self._running  = False
        self._thread   = None
        self._socketio = None
        self.current_video_id  = None
        self.current_chat_id   = None
        self.alert_queue       = queue.Queue()

    def start(self, video_id: str, socketio=None):
        if self._running:
            self.stop()
        self.current_video_id = video_id
        self._socketio = socketio
        chat_id = yt_api.get_video_live_chat_id(video_id)
        if not chat_id:
            log.warning(f"Live chat ID bulunamadı: {video_id}")
            return False
        self.current_chat_id = chat_id
        # DB'ye kaydet
        db.execute("""
            INSERT OR REPLACE INTO live_sessions
            (session_id, video_id, started_at, is_active, live_chat_id)
            VALUES (?,?,?,1,?)
        """, (video_id, video_id, datetime.datetime.utcnow().isoformat(), chat_id))
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, args=(chat_id, video_id), daemon=True)
        self._thread.start()
        log.info(f"Canlı yayın monitörü başlatıldı: {video_id}")
        return True

    def stop(self):
        self._running = False
        if self.current_video_id:
            db.execute(
                "UPDATE live_sessions SET is_active=0, ended_at=? WHERE video_id=?",
                (datetime.datetime.utcnow().isoformat(), self.current_video_id))
        self.current_video_id = None
        self.current_chat_id  = None

    def _poll_loop(self, chat_id: str, video_id: str):
        page_token = None
        while self._running:
            try:
                msgs, page_token, poll_ms = yt_api.get_live_chat_messages(
                    chat_id, page_token)
                for m in msgs:
                    m["video_id"] = video_id
                    collector.save_comment(m)
                    # Anlık analiz
                    text = m.get("text","")
                    lang, _ = ta.detect_language(text)
                    result  = detector.analyze_comment(text, lang)
                    if result["threat_score"] >= 0.25:
                        alert = {
                            "type":       "live_threat",
                            "comment_id": m["comment_id"],
                            "author":     m["author"],
                            "channel_id": m["author_channel_id"],
                            "text":       text[:200],
                            "threat":     result["threat_level"],
                            "score":      round(result["threat_score"], 3),
                            "color":      Config.THREAT_COLORS.get(
                                          result["threat_level"], "#888"),
                            "timestamp":  m["timestamp_iso"]
                        }
                        self.alert_queue.put(alert)
                        if self._socketio and HAS_SOCKETIO:
                            try:
                                self._socketio.emit("live_alert", alert)
                            except Exception:
                                pass
                        log.warning(
                            f"[CANLI TEHDİT] {result['threat_level']} — "
                            f"{m['author']}: {text[:60]}"
                        )
                db.execute(
                    "UPDATE live_sessions SET message_count=message_count+? "
                    "WHERE video_id=?", (len(msgs), video_id))
            except Exception as e:
                log.error(f"Live poll hatası: {e}")
            time.sleep(poll_ms / 1000.0)

live_monitor = LiveMonitor()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 13: FLASK UYGULAMASI
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET

if HAS_SOCKETIO:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    live_monitor._socketio = socketio
else:
    socketio = None

# ──── Yardımcı ────────────────────────────────────────────────────────────────
def _pagination(total, page, per_page=50):
    return {
        "total":      total,
        "page":       page,
        "per_page":   per_page,
        "pages":      math.ceil(total / per_page),
        "has_next":   page * per_page < total,
        "has_prev":   page > 1
    }

# ──── Ana Sayfa ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

# ──── Dashboard İstatistikleri ─────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    total_comments = (db.fetchone("SELECT COUNT(*) as n FROM comments") or {}).get("n",0)
    total_users    = (db.fetchone("SELECT COUNT(*) as n FROM user_profiles") or {}).get("n",0)
    pending        = (db.fetchone("SELECT COUNT(*) as n FROM comments WHERE analyzed=0") or {}).get("n",0)
    threat_counts  = db.fetchall("""
        SELECT threat_level, COUNT(*) as cnt
        FROM user_profiles
        GROUP BY threat_level ORDER BY cnt DESC
    """)
    recent_threats = db.fetchall("""
        SELECT te.threat_type, te.threat_score, te.created_at,
               up.username, c.text
        FROM threat_events te
        LEFT JOIN user_profiles up ON te.channel_id = up.channel_id
        LEFT JOIN comments c ON te.comment_id = c.comment_id
        ORDER BY te.created_at DESC LIMIT 20
    """)
    live_active = (db.fetchone(
        "SELECT COUNT(*) as n FROM live_sessions WHERE is_active=1") or {}).get("n",0)
    videos_count = (db.fetchone("SELECT COUNT(*) as n FROM videos") or {}).get("n",0)
    return jsonify({
        "total_comments":  total_comments,
        "total_users":     total_users,
        "pending_analysis":pending,
        "videos_count":    videos_count,
        "live_active":     live_active,
        "threat_breakdown": threat_counts,
        "recent_threats":   recent_threats,
        "channel_handle":  f"@{Config.normalized_handle()}",
    })

@app.route("/api/stats/realtime")
def api_stats_realtime():
    alerts = []
    try:
        while not live_monitor.alert_queue.empty():
            alerts.append(live_monitor.alert_queue.get_nowait())
    except Exception:
        pass
    return jsonify({
        "live_running": live_monitor._running,
        "current_video": live_monitor.current_video_id,
        "alerts": alerts
    })

# ──── Yorum Arama & Listeleme ──────────────────────────────────────────────────
@app.route("/api/comments")
def api_comments():
    q        = request.args.get("q","").strip()
    user_q   = request.args.get("user","").strip()
    threat   = request.args.get("threat","").strip().upper()
    video_id = request.args.get("video_id","").strip()
    page     = max(1, int(request.args.get("page",1)))
    per_page = int(request.args.get("per_page",50))
    offset   = (page-1)*per_page

    conditions = ["1=1"]
    params     = []
    if q:
        conditions.append("c.text LIKE ?")
        params.append(f"%{q}%")
    if user_q:
        conditions.append("(c.author LIKE ? OR c.author_channel_id LIKE ?)")
        params += [f"%{user_q}%", f"%{user_q}%"]
    if threat:
        conditions.append("up.threat_level=?")
        params.append(threat)
    if video_id:
        conditions.append("c.video_id=?")
        params.append(video_id)

    where = " AND ".join(conditions)
    total = (db.fetchone(
        f"SELECT COUNT(*) as n FROM comments c "
        f"LEFT JOIN user_profiles up ON c.author_channel_id=up.channel_id "
        f"WHERE {where}", params) or {}).get("n",0)

    rows = db.fetchall(
        f"SELECT c.comment_id, c.video_id, c.author, c.author_channel_id, "
        f"c.text, c.timestamp_iso, c.lang_detected, c.source_type, "
        f"c.like_count, c.is_live, c.is_deleted, "
        f"up.threat_level, up.threat_score, up.bot_score, "
        f"up.antisemite_score, up.stalker_score, up.groyper_score, "
        f"up.cluster_id, up.account_created, up.subscriber_count "
        f"FROM comments c "
        f"LEFT JOIN user_profiles up ON c.author_channel_id=up.channel_id "
        f"WHERE {where} "
        f"ORDER BY c.timestamp_utc DESC "
        f"LIMIT ? OFFSET ?",
        params + [per_page, offset]
    )
    return jsonify({"comments": rows, "pagination": _pagination(total, page, per_page)})

# ──── Kullanıcı Profili ────────────────────────────────────────────────────────
@app.route("/api/user/<channel_id>")
def api_user_profile(channel_id):
    u = db.fetchone("SELECT * FROM user_profiles WHERE channel_id=?", (channel_id,))
    if not u:
        return jsonify({"error": "Kullanıcı bulunamadı"}), 404
    return jsonify(u)

@app.route("/api/user/<channel_id>/messages")
def api_user_messages(channel_id):
    page     = max(1, int(request.args.get("page",1)))
    per_page = 50
    offset   = (page-1)*per_page
    total    = (db.fetchone(
        "SELECT COUNT(*) as n FROM comments WHERE author_channel_id=?",
        (channel_id,)) or {}).get("n",0)
    rows = db.fetchall(
        "SELECT comment_id, video_id, text, timestamp_iso, source_type, "
        "lang_detected, is_deleted "
        "FROM comments WHERE author_channel_id=? "
        "ORDER BY timestamp_utc DESC LIMIT ? OFFSET ?",
        (channel_id, per_page, offset)
    )
    return jsonify({"messages": rows, "pagination": _pagination(total, page, per_page)})

@app.route("/api/user/<channel_id>/links")
def api_user_links(channel_id):
    links = db.fetchall("""
        SELECT il.*, up.username as user_b_name, up.threat_level as user_b_threat
        FROM identity_links il
        LEFT JOIN user_profiles up ON il.channel_b = up.channel_id
        WHERE il.channel_a=? OR il.channel_b=?
        ORDER BY il.sim_score DESC
    """, (channel_id, channel_id))
    return jsonify({"links": links})

# ──── Video Listesi ────────────────────────────────────────────────────────────
@app.route("/api/videos")
def api_videos():
    q     = request.args.get("q","").strip()
    vtype = request.args.get("type","").strip()  # video / stream
    page  = max(1, int(request.args.get("page",1)))
    per_page = 50
    offset = (page-1)*per_page

    conditions = ["1=1"]
    params = []
    if q:
        conditions.append("title LIKE ?")
        params.append(f"%{q}%")
    if vtype:
        conditions.append("video_type=?")
        params.append(vtype)
    where = " AND ".join(conditions)
    total = (db.fetchone(f"SELECT COUNT(*) as n FROM videos WHERE {where}", params) or {}).get("n",0)
    rows  = db.fetchall(
        f"SELECT video_id, title, published_at, video_type, comment_count "
        f"FROM videos WHERE {where} "
        f"ORDER BY published_at DESC LIMIT ? OFFSET ?",
        params + [per_page, offset]
    )
    return jsonify({"videos": rows, "pagination": _pagination(total, page, per_page)})

# ──── Küme Listesi ─────────────────────────────────────────────────────────────
@app.route("/api/clusters")
def api_clusters():
    rows = db.fetchall("""
        SELECT cluster_id, COUNT(*) as member_count,
               GROUP_CONCAT(username, ', ') as members,
               MAX(threat_score) as max_threat,
               GROUP_CONCAT(DISTINCT threat_level, '/') as threat_types
        FROM user_profiles
        WHERE cluster_id >= 0
        GROUP BY cluster_id
        ORDER BY max_threat DESC
    """)
    return jsonify({"clusters": rows})

@app.route("/api/cluster/<int:cluster_id>/members")
def api_cluster_members(cluster_id):
    rows = db.fetchall("""
        SELECT channel_id, username, threat_level, threat_score,
               bot_score, antisemite_score, stalker_score, message_count,
               account_created, subscriber_count
        FROM user_profiles WHERE cluster_id=?
        ORDER BY threat_score DESC
    """, (cluster_id,))
    return jsonify({"members": rows})

# ──── İlişki Grafiği ───────────────────────────────────────────────────────────
@app.route("/api/graph")
def api_graph():
    graph = identity_linker.build_network_graph()
    return jsonify(graph)

# ──── Yorum Silme ──────────────────────────────────────────────────────────────
@app.route("/api/comment/delete", methods=["POST"])
def api_delete_comment():
    data = request.get_json() or {}
    comment_id  = data.get("comment_id","")
    use_selenium= data.get("use_selenium", False)
    video_url   = data.get("video_url","")
    text_snippet= data.get("text_snippet","")
    if not comment_id:
        return jsonify({"success": False, "error": "comment_id gerekli"}), 400
    # API ile dene
    success = yt_api.delete_comment_api(comment_id)
    if not success and use_selenium and HAS_SELENIUM:
        if not selenium_module.logged_in:
            selenium_module.login()
        success = selenium_module.delete_comment_selenium(video_url, text_snippet)
    if success:
        db.execute("UPDATE comments SET is_deleted=1 WHERE comment_id=?", (comment_id,))
    return jsonify({"success": success})

@app.route("/api/comments/bulk-delete", methods=["POST"])
def api_bulk_delete():
    data     = request.get_json() or {}
    ids      = data.get("comment_ids", [])
    results  = {"success": 0, "failed": 0, "ids": []}
    for cid in ids[:50]:  # Max 50 aynı anda
        ok = yt_api.delete_comment_api(cid)
        if ok:
            db.execute("UPDATE comments SET is_deleted=1 WHERE comment_id=?", (cid,))
            results["success"] += 1
            results["ids"].append(cid)
        else:
            results["failed"] += 1
        time.sleep(0.1)
    return jsonify(results)

# ──── Canlı Yayın Kontrolü ────────────────────────────────────────────────────
@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    data     = request.get_json() or {}
    video_id = data.get("video_id","").strip()
    if not video_id:
        return jsonify({"success": False, "error": "video_id gerekli"}), 400
    ok = live_monitor.start(video_id, socketio)
    return jsonify({"success": ok, "video_id": video_id})

@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    live_monitor.stop()
    return jsonify({"success": True})

@app.route("/api/live/alerts")
def api_live_alerts():
    alerts = []
    try:
        while not live_monitor.alert_queue.empty():
            alerts.append(live_monitor.alert_queue.get_nowait())
    except Exception:
        pass
    return jsonify({"alerts": alerts,
                    "live_running": live_monitor._running,
                    "current_video": live_monitor.current_video_id})

# ──── Veri Toplama İşlemleri ───────────────────────────────────────────────────
_collection_status = {"running": False, "progress": 0, "total": 0,
                      "current": "", "stats": {}}

@app.route("/api/collect/start", methods=["POST"])
def api_collect_start():
    data = request.get_json() or {}
    channel_id = data.get("channel_id", Config.CHANNEL_ID)
    if not channel_id and Config.CHANNEL_HANDLE:
        channel_id = yt_api.get_channel_id(Config.normalized_handle())
        if channel_id:
            Config.CHANNEL_ID = channel_id
    if not channel_id:
        return jsonify({"success": False, "error": "CHANNEL_ID bulunamadı"}), 400
    if _collection_status["running"]:
        return jsonify({"success": False, "error": "Zaten çalışıyor"}), 409

    def _run():
        _collection_status.update({"running": True, "progress": 0, "stats": {}})

        def progress_cb(i, total, title, n_comments):
            _collection_status.update({
                "progress": i, "total": total,
                "current": f"{title[:50]} ({n_comments} yorum)"
            })

        try:
            stats = collector.collect_channel_dataset(
                channel_id, after_date="2023-01-01T00:00:00Z",
                progress_cb=progress_cb)
            _collection_status["stats"] = stats
        except Exception as e:
            _collection_status["stats"] = {"error": str(e)}
        finally:
            _collection_status["running"] = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"success": True, "channel_id": channel_id})

@app.route("/api/collect/status")
def api_collect_status():
    return jsonify(_collection_status)

@app.route("/api/collect/stream-replays", methods=["POST"])
def api_collect_stream_replays():
    data = request.get_json() or {}
    streams_url = data.get("streams_url", Config.channel_streams_url()).strip() or Config.channel_streams_url()
    year_start = int(data.get("year_start", 2023))
    year_end = int(data.get("year_end", 2026))
    # Şu an yalnızca varsayılan kanal destekleniyor; tek noktadan güvenli toplama.
    if "shmirchikart" not in streams_url.lower():
        return jsonify({"success": False, "error": "Sadece varsayılan kanal destekleniyor."}), 400
    stats = collector.collect_stream_replays_from_channel(
        streams_url=streams_url,
        year_start=year_start,
        year_end=year_end
    )
    return jsonify({"success": True, "streams_url": streams_url, "stats": stats})

# ──── Analiz Tetikleme ─────────────────────────────────────────────────────────
_analysis_status = {"running": False, "stats": {}}

@app.route("/api/analyze/run", methods=["POST"])
def api_analyze_run():
    if _analysis_status["running"]:
        return jsonify({"success": False, "error": "Analiz çalışıyor"}), 409

    def _run():
        _analysis_status["running"] = True
        try:
            stats = engine.full_analysis_run()
            _analysis_status["stats"] = stats
        except Exception as e:
            _analysis_status["stats"] = {"error": str(e)}
        finally:
            _analysis_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/analyze/status")
def api_analyze_status():
    return jsonify(_analysis_status)

@app.route("/api/analyze/message", methods=["POST"])
def api_analyze_single():
    data = request.get_json() or {}
    text = data.get("text","").strip()
    if not text:
        return jsonify({"error": "text gerekli"}), 400
    lang, conf = ta.detect_language(text)
    result = detector.analyze_comment(text, lang)
    return jsonify({"language": lang, "lang_confidence": conf, **result})

# ──── Profil İnceleme ──────────────────────────────────────────────────────────
@app.route("/api/inspect/profiles", methods=["POST"])
def api_inspect_profiles():
    data  = request.get_json() or {}
    limit = int(data.get("limit", 50))

    def _run():
        count = collector.inspect_user_profiles(limit)
        log.info(f"{count} profil incelendi.")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "requested": limit})

@app.route("/api/inspect/channel/<channel_id>")
def api_inspect_channel(channel_id):
    info = yt_api.get_channel_info(channel_id)
    if not info:
        return jsonify({"error": "Kanal bulunamadı"}), 404
    # Yeni hesap kontrolü
    age_days = -1
    is_new   = False
    created  = info.get("account_created","")
    if created:
        try:
            dt = datetime.datetime.fromisoformat(created.replace("Z","+00:00"))
            age_days = (datetime.datetime.now(datetime.timezone.utc) - dt).days
            is_new   = age_days < 90
        except Exception:
            pass
    info["account_age_days"] = age_days
    info["is_new_account"]   = is_new
    # DB güncelle
    db.execute("""
        UPDATE user_profiles
        SET display_name=?, account_created=?, subscriber_count=?,
            video_count=?, avatar_url=?, profile_url=?,
            profile_inspected=1, inspected_at=?
        WHERE channel_id=?
    """, (info["display_name"], created,
          info["subscriber_count"], info["video_count"],
          info["avatar_url"], info["profile_url"],
          datetime.datetime.utcnow().isoformat(), channel_id))
    return jsonify(info)

# ──── Kullanıcı Bayraklama (Flagging) ─────────────────────────────────────────
@app.route("/api/user/<channel_id>/flag", methods=["POST"])
def api_flag_user(channel_id):
    data   = request.get_json() or {}
    reason = data.get("reason","Manuel moderatör kararı")
    db.execute("""
        UPDATE user_profiles
        SET flagged=1, flagged_at=?, flagged_reason=?
        WHERE channel_id=?
    """, (datetime.datetime.utcnow().isoformat(), reason, channel_id))
    return jsonify({"success": True})

@app.route("/api/user/<channel_id>/unflag", methods=["POST"])
def api_unflag_user(channel_id):
    db.execute(
        "UPDATE user_profiles SET flagged=0, flagged_reason='' WHERE channel_id=?",
        (channel_id,))
    return jsonify({"success": True})

@app.route("/api/user/<channel_id>/notes", methods=["POST"])
def api_user_notes(channel_id):
    data  = request.get_json() or {}
    notes = data.get("notes","")
    db.execute("UPDATE user_profiles SET notes=? WHERE channel_id=?",
               (notes, channel_id))
    return jsonify({"success": True})

# ──── Tehdit Listesi ───────────────────────────────────────────────────────────
@app.route("/api/threats")
def api_threats():
    level  = request.args.get("level","").upper()
    page   = max(1, int(request.args.get("page",1)))
    per_page = 50
    offset = (page-1)*per_page

    conditions = ["threat_score > 0"]
    params = []
    if level:
        conditions.append("threat_level=?")
        params.append(level)
    where = " AND ".join(conditions)

    total = (db.fetchone(
        f"SELECT COUNT(*) as n FROM user_profiles WHERE {where}", params) or {}).get("n",0)
    rows  = db.fetchall(
        f"SELECT channel_id, username, display_name, threat_level, threat_score, "
        f"bot_score, antisemite_score, stalker_score, groyper_score, "
        f"message_count, first_seen, last_seen, account_created, "
        f"subscriber_count, cluster_id, flagged "
        f"FROM user_profiles WHERE {where} "
        f"ORDER BY threat_score DESC LIMIT ? OFFSET ?",
        params + [per_page, offset]
    )
    return jsonify({"users": rows, "pagination": _pagination(total, page, per_page)})

# ──── Yeni Hesap Tespiti ───────────────────────────────────────────────────────
@app.route("/api/new-accounts")
def api_new_accounts():
    """90 günden yeni hesaplar — potansiyel troll/bot."""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=90)).isoformat()
    rows = db.fetchall("""
        SELECT channel_id, username, account_created, subscriber_count,
               threat_level, threat_score, message_count,
               (julianday('now') - julianday(account_created)) as age_days
        FROM user_profiles
        WHERE account_created > ? AND account_created != ''
        ORDER BY account_created DESC
        LIMIT 200
    """, (cutoff,))
    return jsonify({"new_accounts": rows})

# ──── Veri Dışa Aktarma ────────────────────────────────────────────────────────
@app.route("/api/export/comments")
def api_export_comments():
    threat = request.args.get("threat","").upper()
    fmt    = request.args.get("format","json")
    if threat:
        rows = db.fetchall("""
            SELECT c.comment_id, c.video_id, c.author, c.author_channel_id,
                   c.text, c.timestamp_iso, c.lang_detected, c.source_type,
                   up.threat_level, up.threat_score
            FROM comments c
            LEFT JOIN user_profiles up ON c.author_channel_id=up.channel_id
            WHERE up.threat_level=?
            ORDER BY c.timestamp_utc DESC
        """, (threat,))
    else:
        rows = db.fetchall("""
            SELECT c.comment_id, c.video_id, c.author, c.author_channel_id,
                   c.text, c.timestamp_iso, c.lang_detected, c.source_type,
                   up.threat_level, up.threat_score
            FROM comments c
            LEFT JOIN user_profiles up ON c.author_channel_id=up.channel_id
            ORDER BY c.timestamp_utc DESC LIMIT 10000
        """)
    if fmt == "jsonl":
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        return Response(lines, mimetype="application/x-ndjson",
                        headers={"Content-Disposition":
                                 "attachment;filename=yt_guardian_export.jsonl"})
    return jsonify({"comments": rows, "count": len(rows)})

@app.route("/api/export/users")
def api_export_users():
    rows = db.fetchall("""
        SELECT * FROM user_profiles
        WHERE threat_score > 0
        ORDER BY threat_score DESC
    """)
    return jsonify({"users": rows, "count": len(rows)})

# ──── Moderatör Öneri Sistemi (API Anahtarı Gerektirmez — Yerel BART/Kural Tabanlı) ──
def _build_moderator_suggestion(text: str, threat_result: Dict,
                                 user: Optional[Dict] = None) -> Dict:
    """
    Tehdit analizi sonucuna ve kullanıcı geçmişine göre
    moderatöre yerel AI (BART + kural tabanlı) önerisi üretir.
    Hiçbir harici API anahtarı gerekmez.
    """
    threat_level  = threat_result.get("threat_level", "NORMAL")
    threat_score  = threat_result.get("threat_score", 0.0)
    anti_score    = threat_result.get("antisemite_score", 0.0)
    hate_score    = threat_result.get("hate_score", 0.0)
    stalker_score = threat_result.get("stalker_score", 0.0)
    bot_score     = threat_result.get("bot_score", 0.0)
    groyper_score = threat_result.get("groyper_score", 0.0)

    msg_count      = int((user or {}).get("message_count", 1))
    is_new_account = False
    flagged        = bool((user or {}).get("flagged", False))

    if user and user.get("account_created"):
        try:
            created = datetime.datetime.fromisoformat(
                user["account_created"].replace("Z", "+00:00"))
            age_days = (datetime.datetime.now(datetime.timezone.utc) - created).days
            is_new_account = age_days < 90
        except Exception:
            pass

    # ── Öneri mantığı ────────────────────────────────────────────────────────
    action       = "INCELE"   # DELETE / BAN / WARN / REVIEW / IGNORE
    confidence   = 0.0
    reasons      = []
    next_steps   = []

    # 1. Antisemitizm / Groyper — en yüksek öncelik
    if anti_score >= 0.60 or groyper_score >= 0.60:
        action     = "KALICI_BAN"
        confidence = max(anti_score, groyper_score)
        reasons.append(f"Antisemitik/Groyper içerik tespit edildi (skor: {confidence:.0%})")
        next_steps  = [
            "Yorumu hemen sil",
            "Kullanıcıyı kanaldan kalıcı olarak engelle",
            "Tekrar eden saldırı işaretini not olarak kaydet",
            "Koordineli saldırı ihtimaline karşı kimlik bağlantılarını kontrol et"
        ]

    # 2. Stalker davranışı
    elif stalker_score >= Config.STALK_THRESHOLD:
        action     = "BAN_VE_RAPOR"
        confidence = stalker_score
        reasons.append(f"Stalker davranışı (skor: {stalker_score:.0%})")
        if is_new_account:
            reasons.append("Yeni hesap (< 90 gün) — sahte kimlik riski")
            confidence = min(1.0, confidence + 0.15)
        next_steps = [
            "Yorumu sil",
            "Kullanıcıyı engelle",
            "Diğer platformlarda takip edip etmediğini kontrol et",
            "Belirtiler devam ederse yetkili mercilere bildirmeyi değerlendir"
        ]

    # 3. Nefret söylemi
    elif hate_score >= Config.HATE_THRESHOLD:
        action     = "SIL_VE_UYAR"
        confidence = hate_score
        reasons.append(f"Nefret söylemi (skor: {hate_score:.0%})")
        if msg_count >= 5:
            reasons.append(f"Kullanıcının {msg_count} mesajı var — tekrar eden davranış")
            action = "BAN"
        next_steps = [
            "Yorumu sil",
            "Kullanıcıya son uyarı mesajı yaz (opsiyonel)",
            "Benzer içerikli diğer yorumlarını kontrol et"
        ]

    # 4. Bot/spam
    elif bot_score >= Config.BOT_THRESHOLD:
        action     = "SIL"
        confidence = bot_score
        reasons.append(f"Bot/spam davranışı (skor: {bot_score:.0%})")
        if msg_count >= 10:
            action = "BAN"
            reasons.append(f"Çok sayıda spam mesajı ({msg_count} adet)")
        next_steps = [
            "Yorumu sil",
            "Kullanıcının tüm yorumlarını toplu sil",
            "YouTube spam filtresine geri bildirim gönder"
        ]

    # 5. Şüpheli ama eşik altı
    elif threat_score >= 0.25:
        action     = "INCELE"
        confidence = threat_score
        reasons.append(f"Düşük seviye tehdit belirtisi (genel skor: {threat_score:.0%})")
        if is_new_account:
            reasons.append("Yeni hesap — dikkatli izle")
        if flagged:
            reasons.append("Kullanıcı daha önce bayraklanmış")
            action = "SIL"
        next_steps = [
            "Manuel inceleme yap",
            "Şüpheli kalıplar için diğer yorumlarını gözden geçir",
            "Gerekirse bayrakla"
        ]

    # 6. Normal
    else:
        action     = "EYLEM_YOK"
        confidence = 1.0 - threat_score
        reasons.append("Tehdit tespit edilmedi")
        next_steps = ["Herhangi bir işlem gerekmez"]

    # BART destekli ek güven skoru (yüklüyse)
    bart_scores = threat_result.get("bart_scores", {})
    if bart_scores:
        reasons.append(
            f"Makine öğrenmesi (BART) desteği: "
            f"en yüksek sınıf = {max(bart_scores, key=bart_scores.get)} "
            f"({max(bart_scores.values()):.0%})"
        )

    return {
        "action":          action,
        "confidence":      round(confidence, 3),
        "threat_level":    threat_level,
        "threat_score":    round(threat_score, 3),
        "reasons":         reasons,
        "next_steps":      next_steps,
        "analysis_method": threat_result.get("analysis_method", "keyword"),
        "generated_by":    "local_bart_rule_engine",  # hiçbir harici API kullanılmaz
    }


@app.route("/api/suggest/comment/<comment_id>")
def api_suggest_comment(comment_id):
    """
    Belirli bir yorum için moderatör önerisi üret.
    Hiçbir harici API anahtarı gerekmez — yerel BART + kural motoru kullanır.
    """
    comment = db.fetchone(
        "SELECT c.*, up.* FROM comments c "
        "LEFT JOIN user_profiles up ON c.author_channel_id = up.channel_id "
        "WHERE c.comment_id = ?", (comment_id,))
    if not comment:
        return jsonify({"error": "Yorum bulunamadı"}), 404

    text = comment.get("text", "")
    lang, _ = ta.detect_language(text)
    # BART yüklenebiliyorsa kullan (use_bart=True)
    threat_result = detector.analyze_comment(text, lang, use_bart=True)

    user = db.fetchone(
        "SELECT * FROM user_profiles WHERE channel_id = ?",
        (comment.get("author_channel_id", ""),))

    suggestion = _build_moderator_suggestion(text, threat_result, user)
    suggestion["comment_id"] = comment_id
    suggestion["comment_text"] = text[:200]
    suggestion["author"] = comment.get("author", "")
    return jsonify(suggestion)


@app.route("/api/suggest/user/<channel_id>")
def api_suggest_user(channel_id):
    """
    Bir kullanıcı için moderatör önerisi üret (son 20 mesajına bakarak).
    Hiçbir harici API anahtarı gerekmez.
    """
    user = db.fetchone(
        "SELECT * FROM user_profiles WHERE channel_id = ?", (channel_id,))
    if not user:
        return jsonify({"error": "Kullanıcı bulunamadı"}), 404

    messages = db.fetchall(
        "SELECT text, timestamp_utc FROM comments "
        "WHERE author_channel_id = ? ORDER BY timestamp_utc DESC LIMIT 20",
        (channel_id,))

    combined_text = " ".join(m["text"] for m in messages if m.get("text"))
    lang, _ = ta.detect_language(combined_text)
    threat_result = detector.analyze_comment(combined_text[:1024], lang, use_bart=True)

    # Bot skorunu da dahil et
    if len(messages) >= 3:
        threat_result["bot_score"] = max(
            threat_result.get("bot_score", 0.0),
            detector.bot_score(messages)
        )

    suggestion = _build_moderator_suggestion(combined_text, threat_result, user)
    suggestion["channel_id"]    = channel_id
    suggestion["username"]      = user.get("username", "")
    suggestion["message_count"] = user.get("message_count", 0)
    return jsonify(suggestion)


@app.route("/api/suggest/batch", methods=["POST"])
def api_suggest_batch():
    """
    Birden fazla yorum için toplu moderatör önerisi üret.
    Gövde: {"comment_ids": ["id1", "id2", ...]}  (max 20)
    Hiçbir harici API anahtarı gerekmez.
    """
    data = request.get_json() or {}
    ids  = data.get("comment_ids", [])[:20]
    results = []
    for cid in ids:
        comment = db.fetchone(
            "SELECT c.*, up.message_count, up.account_created, up.flagged "
            "FROM comments c "
            "LEFT JOIN user_profiles up ON c.author_channel_id = up.channel_id "
            "WHERE c.comment_id = ?", (cid,))
        if not comment:
            results.append({"comment_id": cid, "error": "bulunamadı"})
            continue
        text = comment.get("text", "")
        lang, _ = ta.detect_language(text)
        threat_result = detector.analyze_comment(text, lang, use_bart=False)  # hız için BART devre dışı
        suggestion = _build_moderator_suggestion(text, threat_result, comment)
        suggestion["comment_id"] = cid
        suggestion["author"]     = comment.get("author", "")
        results.append(suggestion)
    return jsonify({"suggestions": results, "count": len(results)})


if HAS_SOCKETIO:
    @socketio.on("connect")
    def on_connect():
        emit("connected", {"status": "ok",
                           "channel": f"@{Config.normalized_handle()}"})

    @socketio.on("ping_live")
    def on_ping():
        emit("pong", {"live": live_monitor._running,
                      "video": live_monitor.current_video_id})

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 14: HTML ŞABLONu (Tek Sayfa Web Paneli)
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🛡️ YT Guardian — @ShmirchikArt Moderasyon Paneli</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<style>
:root{
  --bg:#0a0a0f;--panel:#111118;--border:#1e1e2a;--accent:#7c4dff;
  --text:#e8e8f0;--muted:#666688;--danger:#ff2244;--warn:#ff9900;
  --ok:#00cc66;--info:#00aaff;--bot:#0066ff;--purple:#cc00ff;
  --card:#16161f;--input:#0d0d14;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif;font-size:13px;display:flex;height:100vh;overflow:hidden}
/* Sidebar */
#sidebar{width:200px;min-width:200px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:12px 0}
#sidebar .logo{padding:12px 16px 20px;font-size:15px;font-weight:700;color:var(--accent);border-bottom:1px solid var(--border);margin-bottom:8px}
#sidebar .logo span{font-size:11px;color:var(--muted);display:block;margin-top:2px}
#sidebar nav a{display:flex;align-items:center;gap:8px;padding:9px 16px;color:var(--muted);text-decoration:none;transition:.15s;font-size:12.5px}
#sidebar nav a:hover,#sidebar nav a.active{background:rgba(124,77,255,.12);color:var(--text)}
#sidebar nav a .badge{background:var(--danger);color:#fff;border-radius:10px;padding:1px 6px;font-size:10px;margin-left:auto}
/* Main */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
/* Topbar */
#topbar{background:var(--panel);border-bottom:1px solid var(--border);padding:8px 16px;display:flex;align-items:center;gap:10px}
#topbar input{background:var(--input);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:6px;font-size:12px;width:300px;outline:none}
#topbar input:focus{border-color:var(--accent)}
#topbar .filters{display:flex;gap:6px}
#topbar select{background:var(--input);border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:6px;font-size:12px;outline:none}
#topbar .btn{padding:6px 12px;border-radius:6px;border:none;cursor:pointer;font-size:12px;font-weight:600;transition:.15s}
.btn-accent{background:var(--accent);color:#fff}.btn-accent:hover{background:#9c6fff}
.btn-danger{background:var(--danger);color:#fff}.btn-danger:hover{opacity:.85}
.btn-ok{background:var(--ok);color:#000}.btn-sm{padding:4px 8px;font-size:11px}
.btn-warn{background:var(--warn);color:#000}
.btn-outline{background:transparent;border:1px solid var(--border)!important;color:var(--text)}
.btn-outline:hover{border-color:var(--accent)!important;color:var(--accent)}
#topbar .spacer{flex:1}
#live-indicator{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
#live-indicator .dot{width:8px;height:8px;border-radius:50%;background:#444}
#live-indicator.active .dot{background:var(--danger);animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
/* Content */
#content{flex:1;overflow-y:auto;padding:16px;display:none}
#content.active{display:block}
/* Tabs */
.tab-btn{background:transparent;border:none;color:var(--muted);padding:8px 14px;cursor:pointer;font-size:12.5px;border-bottom:2px solid transparent;transition:.15s}
.tab-btn.active,.tab-btn:hover{color:var(--text);border-bottom-color:var(--accent)}
/* Cards */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center}
.stat-card .n{font-size:26px;font-weight:700;color:var(--accent)}
.stat-card .l{font-size:11px;color:var(--muted);margin-top:4px}
/* Table */
.tbl-wrap{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.tbl-wrap table{width:100%;border-collapse:collapse}
.tbl-wrap th{background:rgba(255,255,255,.03);padding:8px 10px;text-align:left;font-size:11px;color:var(--muted);border-bottom:1px solid var(--border);white-space:nowrap}
.tbl-wrap td{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.03);vertical-align:top;max-width:320px;word-break:break-word}
.tbl-wrap tr:hover td{background:rgba(255,255,255,.02)}
/* Threat Badges */
.threat-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700;white-space:nowrap}
.t-ANTISEMITE{background:#000;color:#fff;border:1px solid #666}
.t-HATER{background:#ff2200;color:#fff}
.t-STALKER{background:#ff6600;color:#fff}
.t-IMPERSONATOR{background:#cc00cc;color:#fff}
.t-COORDINATED{background:#8800cc;color:#fff}
.t-BOT{background:#0066ff;color:#fff}
.t-SUSPICIOUS{background:#ff9900;color:#000}
.t-NORMAL{background:#00cc44;color:#000}
.t-UNKNOWN{background:#333;color:#aaa}
/* Score bar */
.score-bar{height:4px;border-radius:2px;background:var(--border);margin-top:3px;overflow:hidden}
.score-fill{height:100%;border-radius:2px;transition:.3s}
/* Comment card */
.comment-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:10px;margin-bottom:8px;position:relative}
.comment-card.threat-high{border-left:3px solid var(--danger)}
.comment-card.threat-warn{border-left:3px solid var(--warn)}
.comment-card .author{font-weight:600;font-size:12px;color:var(--accent)}
.comment-card .text{color:var(--text);margin:5px 0;line-height:1.5}
.comment-card .meta{font-size:10px;color:var(--muted);display:flex;gap:10px;flex-wrap:wrap}
.comment-card .actions{position:absolute;right:8px;top:8px;display:flex;gap:4px;opacity:0;transition:.15s}
.comment-card:hover .actions{opacity:1}
/* User Profile Panel */
#user-panel{width:320px;min-width:320px;background:var(--panel);border-left:1px solid var(--border);overflow-y:auto;padding:16px;display:none}
#user-panel.open{display:block}
#user-panel .avatar{width:60px;height:60px;border-radius:50%;background:var(--border);display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:10px}
#user-panel .uname{font-size:15px;font-weight:700}
#user-panel .uprop{font-size:11px;color:var(--muted);margin:3px 0}
#user-panel .uprop span{color:var(--text)}
/* Graph */
#graph-container{width:100%;height:calc(100vh - 160px);background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden;position:relative}
#graph-svg{width:100%;height:100%}
.graph-tooltip{position:absolute;background:rgba(0,0,0,.9);border:1px solid var(--border);border-radius:6px;padding:8px 12px;font-size:11px;pointer-events:none;display:none}
/* Live alerts */
#live-panel{position:fixed;right:16px;bottom:16px;width:320px;z-index:1000;display:flex;flex-direction:column;gap:6px}
.live-alert{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px;animation:slideIn .3s ease;border-left:3px solid var(--danger);display:flex;flex-direction:column;gap:4px}
@keyframes slideIn{from{transform:translateX(340px);opacity:0}to{transform:translateX(0);opacity:1}}
.live-alert .la-head{display:flex;align-items:center;gap:6px;font-weight:700;font-size:11px}
.live-alert .la-text{font-size:11px;color:var(--muted)}
.live-alert .la-close{margin-left:auto;cursor:pointer;color:var(--muted)}
/* Loader */
.spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
/* Pagination */
.pagination{display:flex;gap:6px;align-items:center;margin-top:12px}
.pagination button{background:var(--card);border:1px solid var(--border);color:var(--text);padding:5px 10px;border-radius:4px;cursor:pointer;font-size:11px}
.pagination button:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
.pagination button:disabled{opacity:.4;cursor:default}
.pagination .info{color:var(--muted);font-size:11px}
/* Highlight search */
.hl{background:rgba(255,200,0,.25);border-radius:2px}
/* Checkbox select */
.cb-select{cursor:pointer;width:14px;height:14px;accent-color:var(--accent)}
.selected-bar{background:rgba(124,77,255,.1);border:1px solid var(--accent);border-radius:6px;padding:8px 12px;display:flex;align-items:center;gap:10px;margin-bottom:10px;display:none}
/* Progress bar */
.progress-wrap{background:var(--border);border-radius:4px;height:6px;margin:8px 0}
.progress-fill{height:100%;border-radius:4px;background:var(--accent);transition:.3s}
/* New account badge */
.new-acc{font-size:9px;background:#ff6600;color:#fff;padding:1px 5px;border-radius:3px;margin-left:4px}
</style>
</head>
<body>

<!-- SIDEBAR -->
<div id="sidebar">
  <div class="logo">🛡️ YT Guardian<span>@ShmirchikArt</span></div>
  <nav>
    <a href="#" class="nav-link active" data-tab="dashboard">📊 Dashboard</a>
    <a href="#" class="nav-link" data-tab="comments">💬 Yorumlar</a>
    <a href="#" class="nav-link" data-tab="threats">⚠️ Tehdit Listesi</a>
    <a href="#" class="nav-link" data-tab="graph">🕸️ İlişki Haritası</a>
    <a href="#" class="nav-link" data-tab="videos">🎬 Videolar</a>
    <a href="#" class="nav-link" data-tab="clusters">🔗 Kümeler</a>
    <a href="#" class="nav-link" data-tab="live">📡 Canlı Yayın</a>
    <a href="#" class="nav-link" data-tab="newaccounts">🆕 Yeni Hesaplar</a>
    <a href="#" class="nav-link" data-tab="collect">⬇️ Veri Toplama</a>
  </nav>
</div>

<!-- MAIN -->
<div id="main">
  <!-- TOPBAR -->
  <div id="topbar">
    <input type="text" id="global-search" placeholder="🔍 Yorum veya kullanıcı ara..." autocomplete="off">
    <div class="filters">
      <select id="filter-threat">
        <option value="">Tüm tehditler</option>
        <option value="ANTISEMITE">ANTİSEMİT</option>
        <option value="HATER">HATER</option>
        <option value="STALKER">STALKER</option>
        <option value="BOT">BOT</option>
        <option value="GROYPER">GROYPER</option>
        <option value="SUSPICIOUS">ŞÜPHELİ</option>
        <option value="COORDINATED">KOORDİNELİ</option>
        <option value="NORMAL">NORMAL</option>
      </select>
      <select id="filter-source">
        <option value="">Tüm kaynaklar</option>
        <option value="comment">Yorum</option>
        <option value="live_chat">Canlı Sohbet</option>
        <option value="reply">Yanıt</option>
      </select>
    </div>
    <button class="btn btn-accent btn-sm" id="btn-search">Ara</button>
    <div class="spacer"></div>
    <div id="live-indicator">
      <div class="dot"></div><span id="live-status-txt">Canlı değil</span>
    </div>
  </div>

  <!-- CONTENT TABS -->
  <!-- DASHBOARD -->
  <div id="tab-dashboard" class="content active">
    <div class="stat-grid" id="stat-grid"></div>
    <div style="display:flex;gap:16px">
      <div style="flex:1">
        <h3 style="margin-bottom:10px;font-size:13px;color:var(--muted)">Son Tehdit Olayları</h3>
        <div id="recent-threats"></div>
      </div>
      <div style="width:220px">
        <h3 style="margin-bottom:10px;font-size:13px;color:var(--muted)">Tehdit Dağılımı</h3>
        <div id="threat-dist"></div>
      </div>
    </div>
    <div style="margin-top:16px">
      <button class="btn btn-accent" id="btn-run-analysis">⚙️ Tam Analizi Çalıştır</button>
      <button class="btn btn-outline" id="btn-inspect-profiles" style="margin-left:8px">👁️ Profilleri İncele</button>
      <span id="analysis-status" style="margin-left:12px;font-size:11px;color:var(--muted)"></span>
    </div>
    <div id="analysis-progress" style="display:none;margin-top:8px">
      <div class="progress-wrap"><div class="progress-fill" id="ap-fill" style="width:0%"></div></div>
      <div id="ap-text" style="font-size:11px;color:var(--muted)"></div>
    </div>
  </div>

  <!-- COMMENTS -->
  <div id="tab-comments" class="content">
    <div class="selected-bar" id="selected-bar">
      <span id="selected-count">0 seçili</span>
      <button class="btn btn-danger btn-sm" id="btn-bulk-delete">Seçilenleri Sil</button>
      <button class="btn btn-outline btn-sm" id="btn-clear-sel">Seçimi Temizle</button>
    </div>
    <div id="comments-list"></div>
    <div class="pagination" id="comments-pagination"></div>
  </div>

  <!-- THREATS -->
  <div id="tab-threats" class="content">
    <div class="tbl-wrap">
      <table id="threats-table">
        <thead><tr>
          <th><input type="checkbox" id="sel-all-threats" class="cb-select"></th>
          <th>Kullanıcı</th>
          <th>Tehdit</th>
          <th>Skor</th>
          <th>Anti-Sem</th>
          <th>Bot</th>
          <th>Stalker</th>
          <th>Groyper</th>
          <th>Mesaj</th>
          <th>Hesap Tarih</th>
          <th>Abone</th>
          <th>Küme</th>
          <th>İşlem</th>
        </tr></thead>
        <tbody id="threats-body"></tbody>
      </table>
    </div>
    <div class="pagination" id="threats-pagination"></div>
  </div>

  <!-- GRAPH -->
  <div id="tab-graph" class="content">
    <div style="margin-bottom:10px;display:flex;gap:8px;align-items:center">
      <button class="btn btn-accent btn-sm" id="btn-reload-graph">🔄 Grafiği Yenile</button>
      <button class="btn btn-outline btn-sm" id="btn-find-links">🔗 Bağlantıları Hesapla</button>
      <select id="graph-filter" style="margin-left:8px;background:var(--input);border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:6px;font-size:12px">
        <option value="">Tüm kullanıcılar</option>
        <option value="ANTISEMITE">Sadece ANTİSEMİT</option>
        <option value="HATER">Sadece HATER</option>
        <option value="BOT">Sadece BOT</option>
        <option value="STALKER">Sadece STALKER</option>
      </select>
      <span id="graph-info" style="font-size:11px;color:var(--muted);margin-left:8px"></span>
    </div>
    <div id="graph-container">
      <svg id="graph-svg"></svg>
      <div class="graph-tooltip" id="graph-tooltip"></div>
    </div>
  </div>

  <!-- VIDEOS -->
  <div id="tab-videos" class="content">
    <div style="margin-bottom:10px;display:flex;gap:8px">
      <input type="text" id="video-search" placeholder="Video başlığı ara..." style="background:var(--input);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:6px;font-size:12px;width:250px">
      <select id="video-type-filter" style="background:var(--input);border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:6px;font-size:12px">
        <option value="">Tüm tipler</option>
        <option value="stream">Canlı Yayın</option>
        <option value="video">Video</option>
      </select>
      <button class="btn btn-accent btn-sm" id="btn-video-search">Ara</button>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Başlık</th><th>Tür</th><th>Tarih</th><th>Yorum</th><th>İşlem</th>
        </tr></thead>
        <tbody id="videos-body"></tbody>
      </table>
    </div>
    <div class="pagination" id="videos-pagination"></div>
  </div>

  <!-- CLUSTERS -->
  <div id="tab-clusters" class="content">
    <div class="tbl-wrap" style="margin-bottom:16px">
      <table>
        <thead><tr>
          <th>Küme ID</th><th>Üyeler</th><th>Max Tehdit</th><th>Tehdit Tipleri</th><th>İşlem</th>
        </tr></thead>
        <tbody id="clusters-body"></tbody>
      </table>
    </div>
    <div id="cluster-detail"></div>
  </div>

  <!-- LIVE -->
  <div id="tab-live" class="content">
    <div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;max-width:500px;margin-bottom:16px">
      <h3 style="margin-bottom:12px;font-size:13px">Canlı Yayın Monitörü</h3>
      <div style="display:flex;gap:8px;margin-bottom:10px">
        <input type="text" id="live-video-id" placeholder="Video ID (ör: dQw4w9WgXcQ)" style="flex:1;background:var(--input);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:6px;font-size:12px">
        <button class="btn btn-ok btn-sm" id="btn-live-start">Başlat</button>
        <button class="btn btn-danger btn-sm" id="btn-live-stop">Durdur</button>
      </div>
      <div id="live-session-info" style="font-size:11px;color:var(--muted)"></div>
    </div>
    <h3 style="margin-bottom:10px;font-size:13px;color:var(--muted)">Canlı Tehdit Akışı</h3>
    <div id="live-alerts-log" style="max-height:400px;overflow-y:auto;display:flex;flex-direction:column;gap:6px"></div>
  </div>

  <!-- NEW ACCOUNTS -->
  <div id="tab-newaccounts" class="content">
    <p style="color:var(--muted);font-size:11px;margin-bottom:12px">Son 90 gün içinde oluşturulan hesaplar — olası troll / bot / koordineli saldırı göstergesi</p>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Kullanıcı</th><th>Hesap Yaşı</th><th>Oluşturma</th><th>Abone</th><th>Tehdit</th><th>Mesaj</th>
        </tr></thead>
        <tbody id="new-accounts-body"></tbody>
      </table>
    </div>
  </div>

  <!-- COLLECT -->
  <div id="tab-collect" class="content">
    <div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px;max-width:540px">
      <h3 style="margin-bottom:14px">📥 Kanal Veri Toplama (2023-2026)</h3>
      <div style="margin-bottom:12px">
        <label style="font-size:11px;color:var(--muted);display:block;margin-bottom:4px">Kanal ID</label>
        <input type="text" id="collect-channel-id" value="" placeholder="UCxxxxxxxxx (boş bırakırsan config'deki kullanılır)" style="width:100%;background:var(--input);border:1px solid var(--border);color:var(--text);padding:7px 12px;border-radius:6px;font-size:12px">
      </div>
      <div style="display:flex;gap:8px;margin-bottom:14px">
        <button class="btn btn-accent" id="btn-collect-start">⬇️ Toplamayı Başlat</button>
        <button class="btn btn-outline" id="btn-collect-status">Durum Sorgula</button>
      </div>
      <div class="progress-wrap" id="collect-progress-wrap" style="display:none">
        <div class="progress-fill" id="collect-fill" style="width:0%"></div>
      </div>
      <div id="collect-status" style="font-size:11px;color:var(--muted);margin-top:6px"></div>
      <div id="collect-log" style="margin-top:10px;max-height:200px;overflow-y:auto;font-size:10.5px;color:var(--muted);font-family:monospace;background:var(--input);padding:8px;border-radius:4px;display:none"></div>
    </div>
  </div>

</div><!-- /main -->

<!-- USER PANEL -->
<div id="user-panel">
  <button id="close-user-panel" style="float:right;background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px">✕</button>
  <div style="clear:both"></div>
  <div id="up-avatar" class="avatar">👤</div>
  <div id="up-name" class="uname">-</div>
  <div id="up-channel-id" style="font-size:10px;color:var(--muted);margin-top:2px;margin-bottom:10px"></div>
  <div id="up-props"></div>
  <div style="margin-top:12px;display:flex;gap:6px;flex-wrap:wrap">
    <button class="btn btn-danger btn-sm" id="up-btn-flag">🚩 Bayrakla</button>
    <button class="btn btn-warn btn-sm" id="up-btn-suggest">🤖 Öneri Al</button>
    <button class="btn btn-outline btn-sm" id="up-btn-inspect">🔍 Profili İncele</button>
    <a id="up-btn-yt" href="#" target="_blank" class="btn btn-outline btn-sm" style="text-decoration:none">▶ YouTube</a>
  </div>
  <div id="up-suggest-result" style="display:none;margin-top:8px;padding:8px;background:var(--input);border-radius:6px;font-size:11px;line-height:1.6"></div>
  <div style="margin-top:12px">
    <label style="font-size:10px;color:var(--muted)">Not ekle:</label>
    <textarea id="up-notes" style="width:100%;background:var(--input);border:1px solid var(--border);color:var(--text);padding:6px;border-radius:4px;font-size:11px;margin-top:4px;resize:vertical;height:60px"></textarea>
    <button class="btn btn-outline btn-sm" id="up-btn-save-notes" style="margin-top:4px">Kaydet</button>
  </div>
  <div style="margin-top:12px">
    <h4 style="font-size:11px;color:var(--muted);margin-bottom:6px">Bağlantılı Hesaplar</h4>
    <div id="up-links"></div>
  </div>
  <div style="margin-top:12px">
    <h4 style="font-size:11px;color:var(--muted);margin-bottom:6px">Son Mesajlar</h4>
    <div id="up-messages"></div>
  </div>
</div>

<!-- LIVE ALERTS PANEL -->
<div id="live-panel"></div>

<script>
// ─── GLOBALS ──────────────────────────────────────────────────────────────────
var currentTab = 'dashboard';
var commentsPage = 1;
var threatsPage = 1;
var videosPage = 1;
var selectedComments = new Set();
var currentUserId = null;
var searchDebounce = null;
var graphData = null;
var graphSimulation = null;

// ─── SOCKET.IO ────────────────────────────────────────────────────────────────
var socket = null;
try {
  socket = io();
  socket.on('connected', function(d){ console.log('WS bağlı', d); });
  socket.on('live_alert', function(alert){ showLiveAlert(alert); addToLiveLog(alert); });
} catch(e){ console.warn('SocketIO yok:', e); }

// ─── NAVIGATION ───────────────────────────────────────────────────────────────
$('.nav-link').click(function(e){
  e.preventDefault();
  var tab = $(this).data('tab');
  switchTab(tab);
  $('.nav-link').removeClass('active');
  $(this).addClass('active');
});

function switchTab(tab){
  currentTab = tab;
  $('.content').removeClass('active').hide();
  $('#tab-' + tab).addClass('active').show();
  if(tab==='dashboard')   loadDashboard();
  if(tab==='comments')    loadComments(1);
  if(tab==='threats')     loadThreats(1);
  if(tab==='graph')       loadGraph();
  if(tab==='videos')      loadVideos(1);
  if(tab==='clusters')    loadClusters();
  if(tab==='live')        updateLiveStatus();
  if(tab==='newaccounts') loadNewAccounts();
  if(tab==='collect')     {};
}
switchTab('dashboard');

// ─── DASHBOARD ────────────────────────────────────────────────────────────────
function loadDashboard(){
  $.get('/api/stats', function(d){
    var cards = [
      {n: d.total_comments, l:'Toplam Yorum'},
      {n: d.total_users,    l:'Kullanıcı'},
      {n: d.videos_count,   l:'Video/Stream'},
      {n: d.pending_analysis, l:'Bekleyen Analiz'},
      {n: d.live_active ? 'CANLI' : '-', l:'Canlı Oturum'},
    ];
    (d.threat_breakdown||[]).forEach(function(t){
      if(t.threat_level && t.threat_level !== 'NORMAL' && t.threat_level !== 'UNKNOWN')
        cards.push({n: t.cnt, l: t.threat_level});
    });
    var html = '';
    cards.forEach(function(c){
      html += '<div class="stat-card"><div class="n">' + c.n + '</div><div class="l">' + c.l + '</div></div>';
    });
    $('#stat-grid').html(html);

    // Son tehditler
    var th = '';
    (d.recent_threats||[]).forEach(function(t){
      th += '<div class="comment-card threat-high" style="margin-bottom:6px">' +
        '<div class="author">' + esc(t.username||'?') + ' <span class="threat-badge t-'+t.threat_type+'">' + t.threat_type + '</span></div>' +
        '<div class="text" style="font-size:11px;color:var(--muted)">' + esc((t.text||'').substring(0,120)) + '</div>' +
        '<div class="meta"><span>' + (t.created_at||'').substring(0,16) + '</span><span>Skor: ' + (t.threat_score||0).toFixed(2) + '</span></div>' +
        '</div>';
    });
    $('#recent-threats').html(th || '<div style="color:var(--muted);font-size:12px">Tehdit bulunamadı.</div>');

    // Dağılım
    var dist = '';
    (d.threat_breakdown||[]).forEach(function(t){
      var pct = Math.min(100, Math.round((t.cnt / Math.max(1, d.total_users)) * 100));
      dist += '<div style="margin-bottom:6px">' +
        '<div style="display:flex;justify-content:space-between;font-size:11px">' +
        '<span class="threat-badge t-' + t.threat_level + '">' + (t.threat_level||'?') + '</span>' +
        '<span style="color:var(--muted)">' + t.cnt + '</span></div>' +
        '<div class="score-bar"><div class="score-fill" style="width:' + pct + '%;background:' + threatColor(t.threat_level) + '"></div></div>' +
        '</div>';
    });
    $('#threat-dist').html(dist);
  });
}

$('#btn-run-analysis').click(function(){
  $(this).prop('disabled',true).html('<span class="spinner"></span> Analiz çalışıyor...');
  $('#analysis-status').text('');
  $('#analysis-progress').show();
  $.post('/api/analyze/run', function(){
    pollAnalysis();
  });
});

function pollAnalysis(){
  var t = setInterval(function(){
    $.get('/api/analyze/status', function(d){
      if(!d.running){
        clearInterval(t);
        $('#btn-run-analysis').prop('disabled',false).html('⚙️ Tam Analizi Çalıştır');
        $('#analysis-progress').hide();
        var s = d.stats;
        $('#analysis-status').text(
          'Tamamlandı: ' + (s.comments_analyzed||0) + ' yorum, ' +
          (s.identity_links||0) + ' bağlantı, ' + (s.clusters||0) + ' küme'
        );
        loadDashboard();
      } else {
        $('#ap-text').text('Analiz devam ediyor...');
        $('#ap-fill').css('width','60%');
      }
    });
  }, 2000);
}

$('#btn-inspect-profiles').click(function(){
  $(this).prop('disabled',true);
  $.post('/api/inspect/profiles', JSON.stringify({limit:100}), function(d){
    alert('Profil inceleme başlatıldı (100 kullanıcı).');
    $('#btn-inspect-profiles').prop('disabled',false);
  }, 'json');
});

// ─── COMMENTS ─────────────────────────────────────────────────────────────────
function loadComments(page){
  commentsPage = page;
  var q    = $('#global-search').val().trim();
  var thr  = $('#filter-threat').val();
  var src  = $('#filter-source').val();
  var params = {q:q, threat:thr, source:src, page:page, per_page:50};
  $.get('/api/comments', params, function(d){
    var html = '';
    (d.comments||[]).forEach(function(c){
      var tl = c.threat_level || 'NORMAL';
      var cls = (c.threat_score >= 0.6) ? 'threat-high' : (c.threat_score >= 0.25 ? 'threat-warn' : '');
      html += '<div class="comment-card ' + cls + '">' +
        '<div style="display:flex;align-items:center;gap:8px">' +
        '<input type="checkbox" class="cb-select comment-cb" data-id="' + esc(c.comment_id) + '">' +
        '<a href="#" class="author user-link" data-cid="' + esc(c.author_channel_id) + '">' + esc(c.author||'?') + '</a>' +
        (c.account_created && isNewAccount(c.account_created) ? '<span class="new-acc">YENİ</span>' : '') +
        ' <span class="threat-badge t-' + tl + '">' + tl + '</span>' +
        '</div>' +
        '<div class="text">' + highlight(esc(c.text||''), q) + '</div>' +
        '<div class="meta">' +
        '<span>' + (c.timestamp_iso||'').substring(0,16) + '</span>' +
        '<span>' + (c.source_type||'') + '</span>' +
        '<span>' + (c.lang_detected||'') + '</span>' +
        (c.threat_score > 0 ? '<span style="color:var(--warn)">Tehdit: ' + (c.threat_score||0).toFixed(2) + '</span>' : '') +
        '<span style="color:var(--muted);font-size:10px">VideoID: ' + esc(c.video_id||'') + '</span>' +
        '</div>' +
        '<div class="actions">' +
        '<button class="btn btn-warn btn-sm btn-suggest-comment" data-id="' + esc(c.comment_id) + '" title="Moderatör önerisi al (API gerektirmez)">🤖 Öneri</button>' +
        '<button class="btn btn-danger btn-sm btn-delete-comment" data-id="' + esc(c.comment_id) + '" data-text="' + esc((c.text||'').substring(0,40)) + '">Sil</button>' +
        '</div>' +
        '<div class="suggest-result" id="sug-' + esc(c.comment_id) + '" style="display:none;margin-top:6px;padding:6px 8px;background:var(--input);border-radius:4px;font-size:11px"></div>' +
        '</div>';
    });
    $('#comments-list').html(html || '<div style="color:var(--muted);text-align:center;padding:30px">Yorum bulunamadı.</div>');
    renderPagination('#comments-pagination', d.pagination, loadComments);
    updateSelectedBar();
  });
}

$(document).on('change', '.comment-cb', function(){
  var id = $(this).data('id');
  if($(this).is(':checked')) selectedComments.add(id);
  else selectedComments.delete(id);
  updateSelectedBar();
});
function updateSelectedBar(){
  if(selectedComments.size > 0){
    $('#selected-bar').show();
    $('#selected-count').text(selectedComments.size + ' seçili');
  } else {
    $('#selected-bar').hide();
  }
}
$('#btn-clear-sel').click(function(){
  selectedComments.clear();
  $('.comment-cb').prop('checked',false);
  updateSelectedBar();
});
$('#btn-bulk-delete').click(function(){
  if(!confirm(selectedComments.size + ' yorumu silmek istediğinden emin misin?')) return;
  $.ajax({
    url:'/api/comments/bulk-delete', type:'POST',
    contentType:'application/json',
    data: JSON.stringify({comment_ids: Array.from(selectedComments)}),
    success: function(d){
      alert('Silindi: ' + d.success + ', Başarısız: ' + d.failed);
      selectedComments.clear();
      updateSelectedBar();
      loadComments(commentsPage);
    }
  });
});
$(document).on('click', '.btn-delete-comment', function(){
  var id = $(this).data('id');
  var txt = $(this).data('text');
  if(!confirm('Bu yorumu sil?\n"' + txt + '"')) return;
  var btn = $(this);
  $.ajax({
    url:'/api/comment/delete', type:'POST',
    contentType:'application/json',
    data: JSON.stringify({comment_id: id}),
    success: function(d){
      if(d.success){ btn.closest('.comment-card').fadeOut(300, function(){$(this).remove()}); }
      else alert('Silme başarısız. OAuth2 kurulu mu?');
    }
  });
});

// ─── MODERATÖR ÖNERİ (API ANAHTARI GEREKMEDENçalışır) ────────────────────────
$(document).on('click', '.btn-suggest-comment', function(){
  var id  = $(this).data('id');
  var box = $('#sug-' + id);
  var btn = $(this);
  if(box.is(':visible')){ box.slideUp(150); return; }
  btn.prop('disabled', true).html('<span class="spinner"></span>');
  $.get('/api/suggest/comment/' + id, function(s){
    var actionColor = {
      'KALICI_BAN':'var(--danger)','BAN_VE_RAPOR':'var(--danger)',
      'BAN':'var(--danger)','SIL_VE_UYAR':'var(--warn)',
      'SIL':'var(--warn)','INCELE':'var(--info)',
      'EYLEM_YOK':'var(--ok)'
    }[s.action] || 'var(--muted)';
    var html = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">' +
      '<b style="color:' + actionColor + ';font-size:12px">' + esc(s.action||'?') + '</b>' +
      '<span style="color:var(--muted)">güven: ' + Math.round((s.confidence||0)*100) + '%</span>' +
      '<span style="color:var(--muted);font-size:10px">(' + esc(s.generated_by||'') + ')</span>' +
      '</div>';
    (s.reasons||[]).forEach(function(r){ html += '<div style="color:var(--text);margin:1px 0">• ' + esc(r) + '</div>'; });
    if((s.next_steps||[]).length){
      html += '<div style="color:var(--muted);margin-top:4px;font-weight:600">Önerilen Adımlar:</div>';
      s.next_steps.forEach(function(ns,i){ html += '<div style="color:var(--text);">' + (i+1) + '. ' + esc(ns) + '</div>'; });
    }
    box.html(html).slideDown(200);
    btn.prop('disabled', false).html('🤖 Öneri');
  }).fail(function(){
    box.html('<span style="color:var(--danger)">Öneri alınamadı.</span>').slideDown(200);
    btn.prop('disabled', false).html('🤖 Öneri');
  });
});

// ─── THREATS ──────────────────────────────────────────────────────────────────
function loadThreats(page){
  threatsPage = page;
  var level = $('#filter-threat').val();
  $.get('/api/threats', {level:level, page:page}, function(d){
    var rows = '';
    (d.users||[]).forEach(function(u){
      var tl = u.threat_level || 'UNKNOWN';
      var isNew = u.account_created && isNewAccount(u.account_created);
      rows += '<tr>' +
        '<td><input type="checkbox" class="cb-select" data-id="' + esc(u.channel_id) + '"></td>' +
        '<td><a href="#" class="user-link" data-cid="' + esc(u.channel_id) + '">' +
          esc(u.username||u.display_name||u.channel_id.substring(0,12)) +
          (isNew ? '<span class="new-acc">YENİ</span>' : '') +
          (u.flagged ? ' 🚩' : '') +
        '</a></td>' +
        '<td><span class="threat-badge t-' + tl + '">' + tl + '</span></td>' +
        '<td>' + scoreBar(u.threat_score||0) + '</td>' +
        '<td>' + scoreBar(u.antisemite_score||0) + '</td>' +
        '<td>' + scoreBar(u.bot_score||0) + '</td>' +
        '<td>' + scoreBar(u.stalker_score||0) + '</td>' +
        '<td>' + scoreBar(u.groyper_score||0) + '</td>' +
        '<td>' + (u.message_count||0) + '</td>' +
        '<td style="font-size:10px">' + esc((u.account_created||'').substring(0,10)) + '</td>' +
        '<td style="font-size:10px">' + formatSubs(u.subscriber_count) + '</td>' +
        '<td>' + (u.cluster_id >= 0 ? '🔗 ' + u.cluster_id : '-') + '</td>' +
        '<td style="white-space:nowrap">' +
          '<button class="btn btn-danger btn-sm btn-flag-user" data-cid="' + esc(u.channel_id) + '">🚩</button> ' +
          '<a href="/api/inspect/channel/' + esc(u.channel_id) + '" target="_blank" class="btn btn-outline btn-sm">👁️</a>' +
        '</td>' +
        '</tr>';
    });
    $('#threats-body').html(rows || '<tr><td colspan="13" style="text-align:center;color:var(--muted);padding:20px">Tehdit bulunamadı.</td></tr>');
    renderPagination('#threats-pagination', d.pagination, loadThreats);
  });
}
$(document).on('click', '.btn-flag-user', function(){
  var cid = $(this).data('cid');
  $.ajax({url:'/api/user/'+cid+'/flag', type:'POST',
    contentType:'application/json',
    data: JSON.stringify({reason:'Moderatör kararı'}),
    success: function(){ loadThreats(threatsPage); }
  });
});

// ─── GRAPH ────────────────────────────────────────────────────────────────────
function loadGraph(){
  $('#graph-info').text('Yükleniyor...');
  $.get('/api/graph', function(d){
    graphData = d;
    $('#graph-info').text((d.nodes||[]).length + ' düğüm, ' + (d.edges||[]).length + ' bağlantı');
    renderGraph(d);
  });
}
$('#btn-reload-graph').click(function(){ loadGraph(); });
$('#btn-find-links').click(function(){
  var btn = $(this);
  btn.prop('disabled',true).html('<span class="spinner"></span>');
  $.post('/api/analyze/run', function(){
    setTimeout(function(){
      btn.prop('disabled',false).html('🔗 Bağlantıları Hesapla');
      loadGraph();
    }, 3000);
  });
});

function renderGraph(data){
  var nodes = data.nodes || [];
  var edges = data.edges || [];
  if(!nodes.length){ $('#graph-svg').html('<text x="50%" y="50%" text-anchor="middle" fill="#666" font-size="14">Veri yok — Önce analiz çalıştır.</text>'); return; }
  var svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();
  var w = document.getElementById('graph-container').clientWidth;
  var h = document.getElementById('graph-container').clientHeight;
  var g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.1,5]).on('zoom', function(ev){ g.attr('transform', ev.transform); }));

  var sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(function(d){return d.id}).distance(80).strength(0.5))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(w/2, h/2))
    .force('collision', d3.forceCollide(20));

  var link = g.append('g').selectAll('line')
    .data(edges).join('line')
    .attr('stroke', '#333').attr('stroke-width', function(d){return Math.max(1, (d.weight||0.5)*3)});

  var node = g.append('g').selectAll('circle')
    .data(nodes).join('circle')
    .attr('r', function(d){return d.size||8})
    .attr('fill', function(d){return d.color||'#888'})
    .attr('stroke', '#222').attr('stroke-width', 1.5)
    .style('cursor','pointer')
    .call(d3.drag()
      .on('start', function(ev,d){if(!ev.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y;})
      .on('drag',  function(ev,d){d.fx=ev.x; d.fy=ev.y;})
      .on('end',   function(ev,d){if(!ev.active) sim.alphaTarget(0); d.fx=null; d.fy=null;})
    )
    .on('click', function(ev,d){ openUserPanel(d.id); })
    .on('mouseover', function(ev,d){
      var tt = $('#graph-tooltip');
      tt.html('<b>' + esc(d.label||d.id) + '</b><br>Tehdit: ' + d.threat + '<br>Skor: ' + (d.score||0).toFixed(2) + '<br>Küme: ' + d.cluster);
      tt.css({display:'block', left:(ev.offsetX+10)+'px', top:(ev.offsetY-10)+'px'});
    })
    .on('mouseout', function(){ $('#graph-tooltip').hide(); });

  var label = g.append('g').selectAll('text')
    .data(nodes).join('text')
    .text(function(d){return d.label})
    .attr('font-size','9px').attr('fill','#aaa')
    .attr('text-anchor','middle').attr('dy','20px');

  sim.on('tick', function(){
    link.attr('x1',function(d){return d.source.x}).attr('y1',function(d){return d.source.y})
        .attr('x2',function(d){return d.target.x}).attr('y2',function(d){return d.target.y});
    node.attr('cx',function(d){return d.x}).attr('cy',function(d){return d.y});
    label.attr('x',function(d){return d.x}).attr('y',function(d){return d.y});
  });
  graphSimulation = sim;
}

// ─── VIDEOS ───────────────────────────────────────────────────────────────────
function loadVideos(page){
  videosPage = page;
  var q = $('#video-search').val().trim();
  var t = $('#video-type-filter').val();
  $.get('/api/videos', {q:q, type:t, page:page}, function(d){
    var rows = '';
    (d.videos||[]).forEach(function(v){
      rows += '<tr>' +
        '<td>' + esc(v.title||'') + '</td>' +
        '<td><span style="font-size:10px;padding:2px 6px;border-radius:3px;background:' +
          (v.video_type==='stream'?'#7c4dff22':'#00cc4422') + '">' + (v.video_type||'video') + '</span></td>' +
        '<td style="font-size:10px">' + (v.published_at||'').substring(0,10) + '</td>' +
        '<td>' + (v.comment_count||0) + '</td>' +
        '<td>' +
          '<a href="https://youtube.com/watch?v=' + esc(v.video_id) + '" target="_blank" class="btn btn-outline btn-sm">▶</a> ' +
          '<button class="btn btn-accent btn-sm btn-view-video-comments" data-vid="' + esc(v.video_id) + '">Yorumlar</button> ' +
          '<button class="btn btn-ok btn-sm btn-monitor-live" data-vid="' + esc(v.video_id) + '">Canlı İzle</button>' +
        '</td>' +
        '</tr>';
    });
    $('#videos-body').html(rows || '<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:20px">Video bulunamadı.</td></tr>');
    renderPagination('#videos-pagination', d.pagination, loadVideos);
  });
}
$('#btn-video-search').click(function(){ loadVideos(1); });
$(document).on('click', '.btn-view-video-comments', function(){
  var vid = $(this).data('vid');
  $('#global-search').val('');
  // video_id filtresi ekle ve comments sekmesine geç
  switchTab('comments');
  commentsPage = 1;
  $.get('/api/comments', {video_id: vid, page:1}, function(d){
    var html = '';
    (d.comments||[]).forEach(function(c){
      var tl = c.threat_level || 'NORMAL';
      html += '<div class="comment-card">' +
        '<div><a href="#" class="user-link author" data-cid="' + esc(c.author_channel_id) + '">' + esc(c.author||'?') + '</a> ' +
        '<span class="threat-badge t-' + tl + '">' + tl + '</span></div>' +
        '<div class="text">' + esc(c.text||'') + '</div>' +
        '<div class="meta"><span>' + (c.timestamp_iso||'').substring(0,16) + '</span></div>' +
        '<div class="actions"><button class="btn btn-danger btn-sm btn-delete-comment" data-id="' + esc(c.comment_id) + '" data-text="' + esc((c.text||'').substring(0,40)) + '">Sil</button></div>' +
        '</div>';
    });
    $('#comments-list').html(html);
    renderPagination('#comments-pagination', d.pagination, function(p){
      $.get('/api/comments', {video_id:vid,page:p}, function(dd){
        // basit yeniden render
        location.hash = '';
      });
    });
  });
  $('.nav-link').removeClass('active');
  $('[data-tab="comments"]').addClass('active');
});
$(document).on('click', '.btn-monitor-live', function(){
  var vid = $(this).data('vid');
  $('#live-video-id').val(vid);
  switchTab('live');
  $('.nav-link').removeClass('active');
  $('[data-tab="live"]').addClass('active');
});

// ─── CLUSTERS ─────────────────────────────────────────────────────────────────
function loadClusters(){
  $.get('/api/clusters', function(d){
    var rows = '';
    (d.clusters||[]).forEach(function(c){
      rows += '<tr>' +
        '<td>🔗 ' + c.cluster_id + '</td>' +
        '<td>' + (c.member_count||0) + '</td>' +
        '<td>' + scoreBar(c.max_threat||0) + '</td>' +
        '<td style="font-size:10px">' + esc(c.threat_types||'') + '</td>' +
        '<td><button class="btn btn-outline btn-sm btn-view-cluster" data-cid="' + c.cluster_id + '">Üyeleri Gör</button></td>' +
        '</tr>';
    });
    $('#clusters-body').html(rows || '<tr><td colspan="5" style="color:var(--muted);text-align:center;padding:20px">Küme yok — Analiz çalıştır.</td></tr>');
  });
}
$(document).on('click', '.btn-view-cluster', function(){
  var cid = $(this).data('cid');
  $.get('/api/cluster/' + cid + '/members', function(d){
    var html = '<h4 style="margin:12px 0 8px;font-size:13px">Küme ' + cid + ' Üyeleri</h4><div class="tbl-wrap"><table><thead><tr><th>Kullanıcı</th><th>Tehdit</th><th>Skor</th><th>Bot</th><th>Mesaj</th></tr></thead><tbody>';
    (d.members||[]).forEach(function(u){
      html += '<tr><td><a href="#" class="user-link" data-cid="' + esc(u.channel_id) + '">' + esc(u.username||u.channel_id.substring(0,12)) + '</a></td>' +
        '<td><span class="threat-badge t-' + (u.threat_level||'UNKNOWN') + '">' + (u.threat_level||'?') + '</span></td>' +
        '<td>' + scoreBar(u.threat_score||0) + '</td>' +
        '<td>' + scoreBar(u.bot_score||0) + '</td>' +
        '<td>' + (u.message_count||0) + '</td></tr>';
    });
    html += '</tbody></table></div>';
    $('#cluster-detail').html(html);
  });
});

// ─── LIVE ─────────────────────────────────────────────────────────────────────
$('#btn-live-start').click(function(){
  var vid = $('#live-video-id').val().trim();
  if(!vid){ alert('Video ID gerekli!'); return; }
  $.ajax({url:'/api/live/start', type:'POST', contentType:'application/json',
    data: JSON.stringify({video_id:vid}),
    success: function(d){
      if(d.success){ updateLiveStatus(); }
      else alert('Hata: ' + (d.error||'bilinmiyor'));
    }
  });
});
$('#btn-live-stop').click(function(){
  $.post('/api/live/stop', function(){ updateLiveStatus(); });
});

function updateLiveStatus(){
  $.get('/api/live/alerts', function(d){
    if(d.live_running){
      $('#live-indicator').addClass('active');
      $('#live-status-txt').text('CANLI: ' + (d.current_video||''));
      $('#live-session-info').html('✅ Aktif: <b>' + esc(d.current_video||'') + '</b>');
    } else {
      $('#live-indicator').removeClass('active');
      $('#live-status-txt').text('Canlı değil');
      $('#live-session-info').text('Canlı oturum yok.');
    }
    (d.alerts||[]).forEach(addToLiveLog);
  });
}
function addToLiveLog(a){
  var el = $('<div class="live-alert">' +
    '<div class="la-head" style="color:' + (a.color||'#f00') + '">' +
      '⚠️ ' + esc(a.threat||'THREAT') + ' — ' + esc(a.author||'?') +
      ' <span style="margin-left:6px;font-size:10px;color:var(--muted)">' + (a.timestamp||'').substring(0,16) + '</span>' +
    '</div>' +
    '<div class="la-text">' + esc((a.text||'').substring(0,120)) + '</div>' +
    '<div class="la-text" style="font-size:10px">Skor: ' + (a.score||0).toFixed(2) + '</div>' +
    '</div>');
  $('#live-alerts-log').prepend(el);
  if($('#live-alerts-log .live-alert').length > 100)
    $('#live-alerts-log .live-alert:last').remove();
}
function showLiveAlert(a){
  var el = $('<div class="live-alert">' +
    '<div class="la-head">' +
      '<span style="color:' + (a.color||'#f00') + '">⚠️ ' + esc(a.threat||'') + '</span>' +
      ' — ' + esc(a.author||'?') +
      '<span class="la-close">✕</span>' +
    '</div>' +
    '<div class="la-text">' + esc((a.text||'').substring(0,80)) + '</div>' +
    '</div>');
  $('#live-panel').prepend(el);
  setTimeout(function(){ el.fadeOut(600, function(){$(this).remove()}); }, 8000);
  el.find('.la-close').click(function(){ el.remove(); });
  if($('#live-panel .live-alert').length > 5)
    $('#live-panel .live-alert:last').remove();
}
setInterval(function(){ if(currentTab==='live') updateLiveStatus(); }, 6000);

// ─── NEW ACCOUNTS ─────────────────────────────────────────────────────────────
function loadNewAccounts(){
  $.get('/api/new-accounts', function(d){
    var rows = '';
    (d.new_accounts||[]).forEach(function(u){
      var age = u.age_days ? Math.round(u.age_days) + ' gün' : '?';
      rows += '<tr>' +
        '<td><a href="#" class="user-link" data-cid="' + esc(u.channel_id) + '">' + esc(u.username||u.channel_id.substring(0,12)) + '</a></td>' +
        '<td style="color:var(--warn)">' + age + '</td>' +
        '<td style="font-size:10px">' + esc((u.account_created||'').substring(0,10)) + '</td>' +
        '<td>' + formatSubs(u.subscriber_count) + '</td>' +
        '<td><span class="threat-badge t-' + (u.threat_level||'UNKNOWN') + '">' + (u.threat_level||'?') + '</span></td>' +
        '<td>' + (u.message_count||0) + '</td>' +
        '</tr>';
    });
    $('#new-accounts-body').html(rows || '<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:20px">Profil bilgisi yok — Önce profil incele.</td></tr>');
  });
}

// ─── COLLECT ──────────────────────────────────────────────────────────────────
$('#btn-collect-start').click(function(){
  var cid = $('#collect-channel-id').val().trim();
  $(this).prop('disabled',true).html('<span class="spinner"></span> Başlatılıyor...');
  $.ajax({url:'/api/collect/start', type:'POST', contentType:'application/json',
    data: JSON.stringify({channel_id:cid||undefined}),
    success: function(d){
      if(d.success){
        $('#collect-progress-wrap').show();
        $('#collect-log').show();
        pollCollectStatus();
      } else {
        alert('Hata: ' + (d.error||'bilinmiyor'));
        $('#btn-collect-start').prop('disabled',false).html('⬇️ Toplamayı Başlat');
      }
    }
  });
});
$('#btn-collect-status').click(function(){ pollCollectStatus(true); });
function pollCollectStatus(once){
  function check(){
    $.get('/api/collect/status', function(d){
      var pct = d.total > 0 ? Math.round(d.progress/d.total*100) : 0;
      $('#collect-fill').css('width', pct + '%');
      $('#collect-status').text(
        d.running ? ('İşleniyor: ' + (d.current||'') + ' (' + d.progress + '/' + d.total + ')') :
        (d.stats.error ? 'Hata: '+d.stats.error : 'Tamamlandı: ' + JSON.stringify(d.stats))
      );
      if(d.running && !once) setTimeout(check, 2000);
      else {
        $('#btn-collect-start').prop('disabled',false).html('⬇️ Toplamayı Başlat');
        loadDashboard();
      }
    });
  }
  check();
}

// ─── USER PANEL ───────────────────────────────────────────────────────────────
$(document).on('click', '.user-link', function(e){
  e.preventDefault();
  var cid = $(this).data('cid');
  if(cid) openUserPanel(cid);
});
$('#close-user-panel').click(function(){ $('#user-panel').removeClass('open'); currentUserId=null; });

function openUserPanel(cid){
  if(!cid) return;
  currentUserId = cid;
  $('#user-panel').addClass('open');
  $('#up-name').text('Yükleniyor...');
  $('#up-props').html('');
  $('#up-links').html('');
  $('#up-messages').html('');
  $.get('/api/user/' + cid, function(u){
    $('#up-name').text(u.display_name || u.username || cid.substring(0,16));
    $('#up-channel-id').text(cid);
    var tl = u.threat_level || 'UNKNOWN';
    var isNew = u.account_created && isNewAccount(u.account_created);
    var props = [
      ['Tehdit', '<span class="threat-badge t-' + tl + '">' + tl + '</span>'],
      ['Skor', (u.threat_score||0).toFixed(2) + ' ' + scoreBar(u.threat_score||0)],
      ['Anti-Sem.', scoreBar(u.antisemite_score||0)],
      ['Bot', scoreBar(u.bot_score||0)],
      ['Stalker', scoreBar(u.stalker_score||0)],
      ['Groyper', scoreBar(u.groyper_score||0)],
      ['Mesaj', u.message_count||0],
      ['İlk Görülme', (u.first_seen||'').substring(0,10)],
      ['Son Görülme', (u.last_seen||'').substring(0,10)],
      ['Hesap Tarihi', esc(u.account_created||'-') + (isNew ? ' <span class="new-acc">YENİ!</span>' : '')],
      ['Abone', formatSubs(u.subscriber_count)],
      ['Küme', u.cluster_id >= 0 ? '🔗 ' + u.cluster_id : '-'],
      ['Bayrak', u.flagged ? '🚩 ' + esc(u.flagged_reason||'') : '-'],
    ];
    var phtml = '';
    props.forEach(function(p){
      phtml += '<div class="uprop"><b>' + p[0] + ':</b> <span>' + p[1] + '</span></div>';
    });
    if(u.notes) phtml += '<div class="uprop"><b>Not:</b> <span>' + esc(u.notes) + '</span></div>';
    $('#up-props').html(phtml);
    $('#up-notes').val(u.notes||'');
    if(u.profile_url) $('#up-btn-yt').attr('href', u.profile_url);
    else $('#up-btn-yt').attr('href', 'https://youtube.com/channel/' + cid);
    if(u.avatar_url) $('#up-avatar').html('<img src="' + esc(u.avatar_url) + '" style="width:60px;height:60px;border-radius:50%">');
  });
  // Bağlantılar
  $.get('/api/user/' + cid + '/links', function(d){
    var html = '';
    (d.links||[]).forEach(function(l){
      html += '<div style="font-size:11px;margin-bottom:4px;padding:4px 6px;background:var(--input);border-radius:4px">' +
        '<a href="#" class="user-link" data-cid="' + esc(l.channel_a === cid ? l.channel_b : l.channel_a) + '">' +
        esc(l.user_b_name || l.channel_b.substring(0,12)) + '</a>' +
        ' <span style="color:var(--muted)">' + (l.confidence||'') + ' (' + (l.sim_score||0).toFixed(2) + ')</span>' +
        '</div>';
    });
    $('#up-links').html(html || '<div style="color:var(--muted);font-size:11px">Bağlantı yok.</div>');
  });
  // Mesajlar
  $.get('/api/user/' + cid + '/messages', function(d){
    var html = '';
    (d.messages||[]).slice(0,10).forEach(function(m){
      html += '<div style="font-size:11px;padding:5px;border-bottom:1px solid var(--border)">' +
        '<span style="color:var(--muted)">' + (m.timestamp_iso||'').substring(0,10) + '</span> ' +
        esc((m.text||'').substring(0,80)) +
        '</div>';
    });
    $('#up-messages').html(html || '<div style="color:var(--muted);font-size:11px">Mesaj yok.</div>');
  });
}

$('#up-btn-flag').click(function(){
  if(!currentUserId) return;
  var r = prompt('Bayraklama sebebi:', 'Moderatör kararı');
  if(r === null) return;
  $.ajax({url:'/api/user/'+currentUserId+'/flag', type:'POST',
    contentType:'application/json',
    data: JSON.stringify({reason:r}),
    success: function(){ openUserPanel(currentUserId); }
  });
});
$('#up-btn-inspect').click(function(){
  if(!currentUserId) return;
  var btn = $(this);
  btn.prop('disabled',true).html('<span class="spinner"></span>');
  $.get('/api/inspect/channel/' + currentUserId, function(d){
    btn.prop('disabled',false).html('🔍 Profili İncele');
    openUserPanel(currentUserId);
    if(d.is_new_account) alert('⚠️ YENİ HESAP! ' + d.account_age_days + ' günlük (oluşturma: ' + d.account_created + ')');
  }).fail(function(){
    btn.prop('disabled',false).html('🔍 Profili İncele');
  });
});
$('#up-btn-suggest').click(function(){
  if(!currentUserId) return;
  var btn = $(this);
  var box = $('#up-suggest-result');
  btn.prop('disabled',true).html('<span class="spinner"></span>');
  box.hide();
  $.get('/api/suggest/user/' + currentUserId, function(s){
    var actionColor = {
      'KALICI_BAN':'var(--danger)','BAN_VE_RAPOR':'var(--danger)',
      'BAN':'var(--danger)','SIL_VE_UYAR':'var(--warn)',
      'SIL':'var(--warn)','INCELE':'var(--info)',
      'EYLEM_YOK':'var(--ok)'
    }[s.action] || 'var(--muted)';
    var html = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">' +
      '<b style="color:' + actionColor + ';font-size:13px">' + esc(s.action||'?') + '</b>' +
      '<span style="color:var(--muted)">güven: ' + Math.round((s.confidence||0)*100) + '%</span>' +
      '</div>';
    (s.reasons||[]).forEach(function(r){ html += '<div>• ' + esc(r) + '</div>'; });
    if((s.next_steps||[]).length){
      html += '<div style="color:var(--accent);margin-top:6px;font-weight:600">📋 Önerilen Adımlar:</div>';
      s.next_steps.forEach(function(ns,i){ html += '<div>' + (i+1) + '. ' + esc(ns) + '</div>'; });
    }
    html += '<div style="color:var(--muted);font-size:10px;margin-top:6px">Kaynak: ' + esc(s.generated_by||'') + ' | ' + esc(s.analysis_method||'') + '</div>';
    box.html(html).slideDown(200);
    btn.prop('disabled',false).html('🤖 Öneri Al');
  }).fail(function(){
    box.html('<span style="color:var(--danger)">Öneri alınamadı.</span>').slideDown(200);
    btn.prop('disabled',false).html('🤖 Öneri Al');
  });
});
$('#up-btn-save-notes').click(function(){
  if(!currentUserId) return;
  var notes = $('#up-notes').val();
  $.ajax({url:'/api/user/'+currentUserId+'/notes', type:'POST',
    contentType:'application/json',
    data: JSON.stringify({notes:notes}),
    success: function(){ $(this).text('✓ Kaydedildi'); setTimeout(function(){$('#up-btn-save-notes').text('Kaydet');},2000); }.bind(this)
  });
});

// ─── SEARCH ───────────────────────────────────────────────────────────────────
$('#global-search').on('input', function(){
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(function(){
    if(currentTab === 'comments') loadComments(1);
  }, 400);
});
$('#btn-search').click(function(){
  if(currentTab === 'threats') loadThreats(1);
  else loadComments(1);
});
$('#filter-threat').change(function(){
  if(currentTab === 'threats') loadThreats(1);
  else loadComments(1);
});

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function highlight(text, q){
  if(!q) return text;
  var rx = new RegExp('('+q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')', 'gi');
  return text.replace(rx, '<mark class="hl">$1</mark>');
}
function scoreBar(v){
  var pct = Math.round(Math.min(1, v||0)*100);
  var c = v >= 0.7 ? 'var(--danger)' : v >= 0.4 ? 'var(--warn)' : 'var(--ok)';
  return '<div style="font-size:10px">' + pct + '%</div>' +
         '<div class="score-bar"><div class="score-fill" style="width:' + pct + '%;background:' + c + '"></div></div>';
}
function threatColor(lvl){
  var m = {'ANTISEMITE':'#000','HATER':'#ff2200','STALKER':'#ff6600','BOT':'#0066ff','SUSPICIOUS':'#ff9900','NORMAL':'#00cc44','COORDINATED':'#8800cc','IMPERSONATOR':'#cc00cc'};
  return m[lvl] || '#888';
}
function formatSubs(n){
  if(n < 0) return '?';
  if(n >= 1000000) return (n/1000000).toFixed(1) + 'M';
  if(n >= 1000) return (n/1000).toFixed(1) + 'K';
  return String(n);
}
function isNewAccount(created){
  if(!created) return false;
  try {
    var d = new Date(created);
    return (Date.now() - d.getTime()) < 90*24*3600*1000;
  } catch(e){ return false; }
}
function renderPagination(selector, pag, cb){
  if(!pag) return;
  var html = '';
  html += '<button ' + (pag.has_prev?'':'disabled') + ' class="pg-btn" data-page="' + (pag.page-1) + '">◀</button>';
  html += '<span class="info">Sayfa ' + pag.page + ' / ' + pag.pages + ' (toplam ' + pag.total + ')</span>';
  html += '<button ' + (pag.has_next?'':'disabled') + ' class="pg-btn" data-page="' + (pag.page+1) + '">▶</button>';
  $(selector).html(html);
  $(selector).find('.pg-btn').click(function(){
    var p = parseInt($(this).data('page'));
    cb(p);
  });
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
setInterval(function(){
  $.get('/api/stats/realtime', function(d){
    if(d.live_running){
      $('#live-indicator').addClass('active');
      $('#live-status-txt').text('CANLI: ' + (d.current_video||''));
    }
    (d.alerts||[]).forEach(showLiveAlert);
  });
}, 5000);
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 15: GİRİŞ NOKTASI
# ─────────────────────────────────────────────────────────────────────────────
def _check_ytdlp() -> bool:
    """yt-dlp'nin kurulu ve çalışır durumda olup olmadığını test et."""
    import subprocess
    try:
        r = subprocess.run(["yt-dlp", "--version"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def check_environment():
    """Başlangıç kontrolü."""
    print("\n" + "="*60)
    print("  🛡️  YT GUARDIAN v2.0 — Başlatılıyor")
    print("="*60)
    warnings = []
    infos    = []

    # ── Veri Kaynağı ──────────────────────────────────────────────────────
    has_ytdlp = _check_ytdlp()
    if Config.YT_API_KEY:
        infos.append("✅ YouTube Data API anahtarı mevcut — birincil veri kaynağı: API")
    elif has_ytdlp:
        infos.append("✅ yt-dlp kurulu — birincil veri kaynağı: yt-dlp (API gerektirmez)")
        infos.append(f"   Kanallar: https://youtube.com/@{Config.normalized_handle()}/videos")
        infos.append(f"             {Config.channel_streams_url()}")
    else:
        warnings.append(
            "⚠️  Ne YT_API_KEY ne de yt-dlp mevcut!\n"
            "     Veri toplama için en az biri gereklidir.\n"
            "     yt-dlp kurulumu: pip install yt-dlp"
        )

    # ── Moderatör Öneri Motoru ─────────────────────────────────────────────
    infos.append(
        "✅ Moderatör öneri motoru: yerel BART + kural tabanlı "
        "(hiçbir API anahtarı gerekmez)"
    )

    # ── Diğer bağımlılıklar ───────────────────────────────────────────────
    if not os.path.exists(Config.YT_CLIENT_SECRETS):
        warnings.append("⚠️  client_secrets.json bulunamadı → OAuth2 / yorum silme devre dışı")
    if not HAS_SKLEARN:
        warnings.append("⚠️  scikit-learn yok → pip install scikit-learn")
    if not HAS_NX:
        warnings.append("⚠️  networkx yok → pip install networkx")
    if not HAS_SELENIUM:
        infos.append("ℹ️  selenium kurulu değil → Selenium özellikleri devre dışı (opsiyonel)")
    if not Config.YT_EMAIL:
        infos.append("ℹ️  YT_EMAIL tanımlı değil → Selenium girişi devre dışı")

    for info in infos:
        print("  " + info)
    for w in warnings:
        print("  " + w)
    if not warnings:
        print("\n  ✅ Zorunlu bağımlılıklar hazır")
    print(f"\n  📦 Veritabanı : {Config.DB_PATH}")
    print(f"  🌐 Panel      : http://localhost:{Config.PORT}")
    print(f"  📡 Kanal      : @{Config.normalized_handle()}")
    print(f"  🤖 Öneri API  : /api/suggest/comment/<id>  |  /api/suggest/user/<id>")
    print("="*60 + "\n")


if __name__ == "__main__":
    check_environment()

    # Kanal ID yükle:
    #   - API varsa API üzerinden al
    #   - API yoksa yt-dlp ile ilk videodan çıkar (veya handle'ı ID gibi kullan)
    if not Config.CHANNEL_ID and Config.CHANNEL_HANDLE:
        if yt_api.service and Config.YT_API_KEY:
            log.info(f"Kanal ID alınıyor (API): @{Config.normalized_handle()}")
            cid = yt_api.get_channel_id(Config.normalized_handle())
            if cid:
                Config.CHANNEL_ID = cid
                log.info(f"Kanal ID (API): {cid}")
            else:
                log.warning("Kanal ID alınamadı. CHANNEL_ID'yi .env'ye manuel ekle.")
        else:
            # yt-dlp ile tek video üzerinden kanal ID'yi çek
            import subprocess, json as _json
            try:
                url = f"https://www.youtube.com/@{Config.normalized_handle()}/videos"
                r = subprocess.run(
                    ["yt-dlp", "--flat-playlist", "--playlist-end", "1",
                     "--print", "%(channel_id)s", "--quiet", "--no-warnings", url],
                    capture_output=True, text=True, timeout=30
                )
                cid = r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""
                if cid and cid.startswith("UC"):
                    Config.CHANNEL_ID = cid
                    log.info(f"Kanal ID (yt-dlp): {cid}")
                else:
                    log.info("yt-dlp ile kanal ID alınamadı; handle kullanılıyor.")
                    Config.CHANNEL_ID = Config.normalized_handle()
            except Exception as e:
                log.warning(f"yt-dlp kanal ID sorgusu başarısız: {e}")
                Config.CHANNEL_ID = Config.normalized_handle()

    if HAS_SOCKETIO and socketio:
        socketio.run(app, host="0.0.0.0", port=Config.PORT,
                     debug=Config.DEBUG, use_reloader=False)
    else:
        app.run(host="0.0.0.0", port=Config.PORT,
                debug=Config.DEBUG, use_reloader=False)
