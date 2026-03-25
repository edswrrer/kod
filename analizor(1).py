#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════╗
║  YT GUARDIAN v5 — Canlı Yayın Bağlantı Modülü          ║
║  gir.py tabanlı · Selenium yerine Playwright            ║
║  Anti-bot stealth · Google Login · Live Chat Stream     ║
╚══════════════════════════════════════════════════════════╝

KURULUM:
  pip install playwright
  playwright install chromium

KULLANIM:
  python yt_guardian_v5.py                   # config.json'dan oku
  python yt_guardian_v5.py --email X --password Y
  python yt_guardian_v5.py --headless        # görünmez mod
  python yt_guardian_v5.py --video VIDEO_ID  # doğrudan video ID
"""

import asyncio
import json
import os
import random
import re
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Callable

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("yt_guardian_v5.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("YTGv5")

# ─── Varsayılan Yapılandırma (gir.py ile uyumlu) ──────────────────────────────
_DEFAULT_CFG = {
    "yt_email":    "physicus93@hotmail.com",
    "yt_password": "%C7JdE4,)$MS;4'",
    "target_channel": "https://www.youtube.com/@ShmirchikArt",
    "target_streams": "https://www.youtube.com/@ShmirchikArt/streams",
    "manual_login_timeout_sec": 180,
    "headless": False,
    "data_dir": "./yt_data",
    "cookies_file": "yt_cookies.json",
    "user_data_dir": "./playwright_profile",   # kalıcı profil (oturum hatırlama)
}

def load_config(cfg_file: str = "yt_guardian_config.json") -> dict:
    cfg = _DEFAULT_CFG.copy()
    if Path(cfg_file).exists():
        try:
            cfg.update(json.load(open(cfg_file, encoding="utf-8")))
            log.info("Config yüklendi: %s", cfg_file)
        except Exception as e:
            log.warning("Config okunamadı: %s", e)
    # Çevre değişkeni önceliği
    if os.environ.get("YT_EMAIL"):    cfg["yt_email"]    = os.environ["YT_EMAIL"]
    if os.environ.get("YT_PASSWORD"): cfg["yt_password"] = os.environ["YT_PASSWORD"]
    return cfg

CFG = load_config()

# ─── Playwright Import Kontrolü ───────────────────────────────────────────────
try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        TimeoutError as PWTimeout,
    )
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False
    log.error("Playwright yüklü değil! → pip install playwright && playwright install chromium")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# § 1 — TARAYICI BAŞLATMA (stealth, gir.py mantığı + Playwright gücü)
# ══════════════════════════════════════════════════════════════════════════════

async def make_browser(headless: bool = False) -> tuple:
    """
    Playwright Chromium başlat.
    - navigator.webdriver gizleme (CDP injection)
    - Kalıcı profil desteği (oturum hatırlama)
    - gir.py'deki tüm anti-bot argümanlarına karşılık gelen ayarlar
    """
    pw = await async_playwright().start()

    user_data_dir = CFG.get("user_data_dir", "./playwright_profile")
    Path(user_data_dir).mkdir(parents=True, exist_ok=True)

    launch_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-infobars",
        "--disable-notifications",
        "--disable-popup-blocking",
        "--disable-extensions",
        "--mute-audio",
        "--lang=tr-TR,tr",
        "--window-size=1920,1080",
    ]

    # Kalıcı profil: oturum açık kalır
    context: BrowserContext = await pw.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=headless,
        args=launch_args,
        locale="tr-TR",
        timezone_id="Europe/Istanbul",
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1920, "height": 1080},
        # Otomasyon izlerini sıfırla
        bypass_csp=True,
    )

    # CDP ile webdriver tespitini kapat — gir.py'deki ile aynı mantık
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver',   {get: () => undefined});
        Object.defineProperty(navigator, 'plugins',     {get: () => [1,2,3,4,5]});
        Object.defineProperty(navigator, 'languages',   {get: () => ['tr-TR','tr','en-US','en']});
        window.chrome = {runtime: {}};
        Object.defineProperty(navigator, 'permissions', {
            get: () => ({
                query: p => Promise.resolve({
                    state: p.name === 'notifications' ? Notification.permission : 'granted'
                })
            })
        });
    """)

    log.info("✅ Playwright Chromium başlatıldı (headless=%s)", headless)
    return pw, context


async def _human_type(page: Page, selector: str, text: str):
    """İnsan benzeri klavye yazımı (gir.py'deki time.sleep + send_keys mantığı)."""
    elem = page.locator(selector)
    await elem.click()
    await asyncio.sleep(random.uniform(0.2, 0.5))
    # JS ile temizle
    await elem.evaluate("el => el.value = ''")
    for ch in text:
        await page.keyboard.type(ch)
        await asyncio.sleep(random.uniform(0.04, 0.13))


# ══════════════════════════════════════════════════════════════════════════════
# § 2 — GOOGLE / YOUTUBE GİRİŞİ (gir.py mantığını birebir taşıdı)
# ══════════════════════════════════════════════════════════════════════════════

# E-posta alan stratejileri — gir.py ile birebir aynı öncelik sırası
_EMAIL_SELECTORS = [
    "#identifierId",                          # ← gir.py birincil
    'input[jsname="YPqjbf"]',
    'input[name="identifier"]',
    'input[type="email"]',
]

# Şifre alan stratejileri — gir.py By.NAME "Passwd" öncelikli
_PASS_SELECTORS = [
    'input[name="Passwd"]',                   # ← gir.py birincil
    'input[type="password"][name="Passwd"]',
    'input[type="password"][jsname="YPqjbf"]',
    'input[type="password"]',
]

# "Sonraki" düğmesi stratejileri
_NEXT_SELECTORS = [
    'button[jsname="LgbsSe"]',
    '#identifierNext button',
    '#passwordNext button',
    'button[type="submit"]',
]


async def _find_first(page: Page, selectors: list, timeout: int = 8000) -> Optional[object]:
    """Selector listesini sırayla dene, ilk bulunanı döndür."""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            await loc.wait_for(state="visible", timeout=timeout)
            return loc
        except Exception:
            continue
    return None


async def yt_login(context: BrowserContext, email: str, password: str) -> bool:
    """
    Google/YouTube otomatik giriş — gir.py adım adım mantığı:
      1. Oturum zaten açık mı? → atla
      2. ServiceLogin URL'ye git
      3. E-posta gir + Enter (gir.py: send_keys → Keys.ENTER)
      4. Şifre gir + Enter
      5. 2 dakikalık polling döngüsü (gir.py: timeout = time.time() + 120)
      6. Başarılı → cookies kaydet
    """
    page = await context.new_page()

    try:
        # ── Oturum zaten açık mı? (kalıcı profil) ──────────────────────────
        log.info("🔍 Mevcut oturum kontrol ediliyor...")
        await page.goto("https://www.youtube.com", wait_until="domcontentloaded")
        await asyncio.sleep(2)

        try:
            await page.locator("button#avatar-btn, yt-img-shadow#avatar").first.wait_for(
                state="visible", timeout=5000
            )
            log.info("✅ YouTube oturumu zaten aktif — giriş atlandı")
            await page.close()
            return True
        except Exception:
            pass  # Avatar yoksa giriş akışına devam

        # ── Otomatik giriş ─────────────────────────────────────────────────
        if not email or not password:
            log.warning("E-posta/şifre sağlanmadı — manuel giriş bekleniyor...")
            await page.goto("https://accounts.google.com/signin")
            return await _wait_for_login(page)

        # gir.py: driver.get("https://accounts.google.com/signin")
        login_url = (
            "https://accounts.google.com/ServiceLogin"
            "?service=youtube&uilel=3&passive=true"
            "&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin"
            "%3Faction_handle_signin%3Dtrue%26app%3Ddesktop%26hl%3Dtr%26next%3D%252F"
            "&hl=tr"
        )
        log.info("🚀 Google giriş sayfası açılıyor...")
        await page.goto(login_url, wait_until="domcontentloaded")
        await asyncio.sleep(2)

        # ── ADIM 1: E-posta ────────────────────────────────────────────────
        log.info("[1/4] E-posta giriliyor...")
        ef = await _find_first(page, _EMAIL_SELECTORS)
        if ef is None:
            # Fallback: doğrudan identifier sayfası
            await page.goto(
                "https://accounts.google.com/signin/v2/identifier?hl=tr&flowName=GlifWebSignIn"
            )
            await asyncio.sleep(2)
            ef = await _find_first(page, _EMAIL_SELECTORS)
        if ef is None:
            await page.screenshot(path="login_no_email.png")
            log.error("❌ E-posta alanı bulunamadı")
            await page.close()
            return False

        await ef.click()
        await asyncio.sleep(random.uniform(0.2, 0.5))
        await ef.evaluate("el => el.value = ''")
        for ch in email:
            await page.keyboard.type(ch)
            await asyncio.sleep(random.uniform(0.04, 0.13))

        # ── ADIM 2: "Sonraki" ──────────────────────────────────────────────
        log.info("[2/4] 'Sonraki' butonuna tıklanıyor...")
        next_btn = await _find_first(page, _NEXT_SELECTORS)
        if next_btn:
            await asyncio.sleep(0.3)
            await next_btn.click()
        else:
            # gir.py fallback: Keys.ENTER
            log.warning("Buton bulunamadı → Enter fallback")
            await page.keyboard.press("Enter")
        await asyncio.sleep(2.5)

        # ── ADIM 3: Şifre ──────────────────────────────────────────────────
        log.info("[3/4] Şifre giriliyor...")
        pf = await _find_first(page, _PASS_SELECTORS, timeout=12000)
        if pf is None:
            await page.screenshot(path="login_no_pass.png")
            log.warning("Şifre alanı bulunamadı → manuel mod")
            return await _wait_for_login(page)

        await pf.click()
        await asyncio.sleep(random.uniform(0.2, 0.5))
        await pf.evaluate("el => el.value = ''")
        for ch in password:
            await page.keyboard.type(ch)
            await asyncio.sleep(random.uniform(0.04, 0.13))

        # ── ADIM 4: Şifre "Sonraki" ────────────────────────────────────────
        log.info("[4/4] Şifre onaylanıyor...")
        pw_btn = await _find_first(page, _NEXT_SELECTORS)
        if pw_btn:
            await asyncio.sleep(0.3)
            await pw_btn.click()
        else:
            await page.keyboard.press("Enter")

        await asyncio.sleep(4)

        # ── CAPTCHA / 2FA kontrolü ─────────────────────────────────────────
        cur = page.url
        if "accounts.google.com" in cur:
            await page.screenshot(path="login_challenge.png")
            log.warning("⚠️ Doğrulama ekranı (CAPTCHA/2FA?) — manuel tamamlayın: login_challenge.png")
            ok = await _wait_for_login(page)
            if not ok:
                await page.close()
                return False

        # ── YouTube'a yönlendir + onayla ───────────────────────────────────
        await page.goto("https://www.youtube.com", wait_until="domcontentloaded")
        await asyncio.sleep(2)

        success = "youtube.com" in page.url and "accounts.google.com" not in page.url
        if success:
            await _save_cookies(context)
            log.info("✅ YouTube girişi başarılı: %s", email)
        else:
            await page.screenshot(path="login_failed.png")
            log.error("❌ Giriş başarısız | URL: %s", page.url)

        await page.close()
        return success

    except Exception as e:
        log.error("yt_login hatası: %s", e)
        try:
            await page.screenshot(path="login_error.png")
        except Exception:
            pass
        await page.close()
        return False


async def _wait_for_login(page: Page) -> bool:
    """
    gir.py'deki 120 saniyelik polling döngüsü:
      timeout = time.time() + 120
      while time.time() < timeout:
          if "accounts.google.com" not in driver.current_url:
              logged_in = True; break
    """
    timeout_sec = int(CFG.get("manual_login_timeout_sec", 180))
    log.info("⏳ Manuel giriş bekleniyor (%ds)...", timeout_sec)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        cur = page.url
        if "youtube.com" in cur and "accounts.google.com" not in cur:
            log.info("✅ Giriş algılandı")
            await _save_cookies(page.context)
            return True
        await asyncio.sleep(2)
    log.error("⛔ Giriş zaman aşımına uğradı")
    return False


async def _save_cookies(context: BrowserContext):
    """Playwright oturum çerezlerini kaydet (yt-dlp ile de kullanılabilir)."""
    try:
        data_dir = Path(CFG.get("data_dir", "./yt_data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        cookie_path = data_dir / CFG.get("cookies_file", "yt_cookies.json")
        cookies = await context.cookies()
        with open(cookie_path, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        log.info("✅ Cookies kaydedildi: %s", cookie_path)
    except Exception as e:
        log.warning("Cookie kaydetme hatası: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# § 3 — CANLI YAYIN BAĞLANTISI (gir.py: TARGET_CHANNEL + TARGET_STREAMS)
# ══════════════════════════════════════════════════════════════════════════════

async def navigate_to_streams(context: BrowserContext) -> Page:
    """
    gir.py adım adım:
      4. driver.get(TARGET_CHANNEL)
      5. driver.get(TARGET_STREAMS)
    """
    page = await context.new_page()

    target_channel = CFG.get("target_channel", "https://www.youtube.com/@ShmirchikArt")
    target_streams = CFG.get("target_streams", "https://www.youtube.com/@ShmirchikArt/streams")

    log.info("📺 Kanala gidiliyor: %s", target_channel)
    await page.goto(target_channel, wait_until="domcontentloaded")
    await asyncio.sleep(3)

    log.info("🎬 Yayınlar (Streams) sekmesine geçiliyor: %s", target_streams)
    await page.goto(target_streams, wait_until="domcontentloaded")
    await asyncio.sleep(3)

    log.info("🎯 Hedef sayfaya ulaşıldı: %s", page.url)
    return page


async def find_live_stream(page: Page) -> Optional[str]:
    """
    Streams sayfasında aktif canlı yayını ara.
    Dönüş: video_id (11 karakter) veya None
    """
    try:
        # "CANLI" veya "LIVE NOW" badge'i olan ilk videoyu bul
        live_selectors = [
            'ytd-badge-supported-renderer span:text("CANLI")',
            'ytd-badge-supported-renderer span:text("LIVE")',
            'ytd-badge-supported-renderer span:text("LIVE NOW")',
            '[aria-label*="LIVE"]',
        ]
        for sel in live_selectors:
            try:
                badge = page.locator(sel).first
                await badge.wait_for(state="visible", timeout=4000)
                # Üst video bağlantısına çık
                video_anchor = await badge.evaluate(
                    "el => el.closest('ytd-rich-item-renderer, ytd-video-renderer')"
                    "?.querySelector('a#video-title-link, a.ytd-thumbnail')?.href"
                )
                if video_anchor:
                    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", video_anchor)
                    if m:
                        vid = m.group(1)
                        log.info("🔴 Canlı yayın bulundu: %s", vid)
                        return vid
            except Exception:
                continue

        log.info("ℹ️ Şu an aktif canlı yayın yok; yakın zamanlı yayınlar listeleniyor")
        return None

    except Exception as e:
        log.warning("Canlı yayın arama hatası: %s", e)
        return None


async def connect_to_live(context: BrowserContext, video_id: str,
                           on_message: Optional[Callable] = None) -> Page:
    """
    Canlı yayına bağlan ve chat mesajlarını dinle.

    on_message(author, text, timestamp) callback'i her yeni mesajda tetiklenir.
    Callback verilmezse mesajlar log'a yazılır.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    log.info("⚡ Canlı yayına bağlanılıyor: %s", url)

    page = await context.new_page()
    await page.goto(url, wait_until="domcontentloaded")
    await asyncio.sleep(5)

    # Chat iframe varsa geç
    chat_frame = None
    for frame in page.frames:
        if "live_chat" in frame.url:
            chat_frame = frame
            break

    target = chat_frame or page
    log.info("✅ Chat kaynağı: %s", "iframe" if chat_frame else "ana sayfa")
    return page


async def poll_live_chat(page: Page, video_id: str,
                          on_message: Optional[Callable] = None,
                          interval: float = 3.0,
                          max_runtime_sec: int = 0):
    """
    Canlı chat mesajlarını periyodik olarak çek.

    Args:
        page:           connect_to_live'dan dönen Page
        video_id:       İzlenen video ID
        on_message:     Callback(author, text, timestamp_unix)
        interval:       Sorgu aralığı (saniye)
        max_runtime_sec: 0 = sonsuz döngü
    """
    seen: set = set()
    start = time.time()
    log.info("📡 Chat dinleniyor (interval=%.1fs)...", interval)

    while True:
        if max_runtime_sec > 0 and (time.time() - start) > max_runtime_sec:
            log.info("⏰ Maksimum süreye ulaşıldı (%ds)", max_runtime_sec)
            break

        # Chat iframe tekrar kontrol
        chat_frame = None
        for frame in page.frames:
            if "live_chat" in frame.url:
                chat_frame = frame
                break
        target = chat_frame or page

        try:
            items = await target.locator(
                "yt-live-chat-text-message-renderer, yt-live-chat-paid-message-renderer"
            ).all()

            for item in items:
                try:
                    author = await item.locator("#author-name").text_content()
                    text   = await item.locator("#message").text_content()
                    if not author or not text:
                        continue
                    author = author.strip()
                    text   = text.strip()
                    key    = f"{author}|{text}"
                    if key in seen:
                        continue
                    seen.add(key)

                    ts = int(time.time())
                    if on_message:
                        on_message(author, text, ts)
                    else:
                        log.info("[CHAT] @%-20s | %s", author, text[:80])

                except Exception:
                    continue

        except Exception as e:
            log.debug("Chat çekme hatası: %s", e)

        await asyncio.sleep(interval)


# ══════════════════════════════════════════════════════════════════════════════
# § 4 — COOKIE YÜKLEMESİ (önceki oturum)
# ══════════════════════════════════════════════════════════════════════════════

async def load_cookies(context: BrowserContext) -> bool:
    """Kaydedilmiş Playwright JSON çerezlerini yükle."""
    cookie_path = Path(CFG.get("data_dir", "./yt_data")) / CFG.get("cookies_file", "yt_cookies.json")
    if not cookie_path.exists():
        return False
    try:
        cookies = json.load(open(cookie_path, encoding="utf-8"))
        await context.add_cookies(cookies)
        log.info("✅ %d çerez yüklendi: %s", len(cookies), cookie_path)
        return True
    except Exception as e:
        log.warning("Çerez yükleme hatası: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# § 5 — ANA AKIŞ (gir.py: setup_driver + yt_login_and_navigate)
# ══════════════════════════════════════════════════════════════════════════════

async def run(email: str = "", password: str = "",
              headless: bool = False,
              video_id: str = "",
              on_message: Optional[Callable] = None,
              max_runtime_sec: int = 0):
    """
    gir.py'deki yt_login_and_navigate() işlevinin Playwright karşılığı:
      1. Tarayıcı başlat (setup_driver)
      2. Giriş yap (yt_login_and_navigate → steps 1-3)
      3. Kanala git (step 4)
      4. Yayınlar sekmesine geç (step 5)
      5. Canlı yayın varsa bağlan ve chat'i dinle
    """
    email    = email    or CFG.get("yt_email", "")
    password = password or CFG.get("yt_password", "")

    pw, context = await make_browser(headless=headless)

    try:
        # Kaydedilmiş çerezleri yükle (opsiyonel)
        await load_cookies(context)

        # ── Giriş ─────────────────────────────────────────────────────────
        logged_in = await yt_login(context, email, password)
        if not logged_in:
            log.error("❌ Giriş yapılamadı. İşlem durduruluyor.")
            return

        # ── Kanala Git + Yayınlar Sekmesi ─────────────────────────────────
        streams_page = await navigate_to_streams(context)

        # ── Canlı Yayın Arama ─────────────────────────────────────────────
        if not video_id:
            video_id = await find_live_stream(streams_page)

        await streams_page.close()

        if not video_id:
            log.info("ℹ️ Şu an aktif canlı yayın yok. Kanal sayfası açık bırakılıyor.")
            log.info("   Tarayıcıyı manuel olarak kapatabilirsiniz.")
            # Tarayıcıyı açık tut (gir.py: "İşlem tamamlandı. Tarayıcıyı manuel olarak kapatabilirsiniz.")
            await asyncio.sleep(999999)
            return

        # ── Canlı Yayına Bağlan ────────────────────────────────────────────
        live_page = await connect_to_live(context, video_id, on_message)

        log.info("🎯 Canlı yayın izleniyor: https://www.youtube.com/watch?v=%s", video_id)
        log.info("   Ctrl+C ile durdurabilirsiniz.")

        await poll_live_chat(
            live_page,
            video_id=video_id,
            on_message=on_message,
            interval=3.0,
            max_runtime_sec=max_runtime_sec,
        )

    except KeyboardInterrupt:
        log.info("\n🛑 Kullanıcı tarafından durduruldu.")
    except Exception as e:
        log.error("Beklenmeyen hata: %s", e)
    finally:
        log.info("🔒 Tarayıcı kapatılıyor...")
        await context.close()
        await pw.stop()
        log.info("✅ Çıkış tamamlandı.")


# ══════════════════════════════════════════════════════════════════════════════
# § 6 — ÖRNEK CALLBACK + ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def default_on_message(author: str, text: str, timestamp: int):
    """Varsayılan mesaj işleyici — terminale yaz."""
    from datetime import datetime
    ts_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    print(f"[{ts_str}] @{author:<25} | {text}")


def main():
    parser = argparse.ArgumentParser(
        description="YT Guardian v5 — Playwright tabanlı canlı yayın bağlantısı"
    )
    parser.add_argument("--email",    type=str, default="", help="Google e-posta")
    parser.add_argument("--password", type=str, default="", help="Google şifre")
    parser.add_argument("--headless", action="store_true", help="Görünmez mod")
    parser.add_argument("--video",    type=str, default="", help="Doğrudan Video ID (11 karakter)")
    parser.add_argument("--duration", type=int, default=0,  help="Maksimum çalışma süresi (saniye, 0=sonsuz)")
    parser.add_argument("--config",   type=str, default="yt_guardian_config.json")
    args = parser.parse_args()

    # Config yeniden yükle
    global CFG
    CFG = load_config(args.config)

    asyncio.run(
        run(
            email=args.email,
            password=args.password,
            headless=args.headless,
            video_id=args.video,
            on_message=default_on_message,
            max_runtime_sec=args.duration,
        )
    )


if __name__ == "__main__":
    main()
