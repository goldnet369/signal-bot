"""
Interactive Telegram signal bot: Crypto (top coins by volume) + Gold (XAUUSD).
User taps a button -> bot analyses the market right now -> sends a signal
(entry, stop loss, TP1/TP2/TP3, chart) or says honestly that there is no good setup.

Strategies (each fires on the last CLOSED 15m candle):
  1. Turtle breakout (Richard Dennis)         - Donchian 20 + ADX
  2. Holy Grail pullback (Linda Raschke)      - ADX>=22, pullback to EMA20
  3. RSI(2) pullback (Larry Connors)          - extreme RSI(2) with SMA200 filter
  4. Bollinger squeeze breakout (J. Bollinger)- low volatility then band break
  5. MACD momentum + EMA cross                - MACD flip / EMA9-21 cross
A signal is sent only if the 1h trend agrees and the combined score is high enough.
No signal can be guaranteed to win.
"""
import io
import json
import os
import threading
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
ALLOWED = {x.strip() for x in os.environ.get("ALLOWED_CHAT_IDS", "").split(",") if x.strip()}
TOP_N = int(os.environ.get("TOP_N", "40"))
MIN_SCORE = int(os.environ.get("MIN_SCORE", "3"))

API = f"https://api.telegram.org/bot{TOKEN}"
BIN = "https://data-api.binance.vision/api/v3"
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
TV_LINK_GOLD = "https://www.tradingview.com/chart/?symbol=OANDA:XAUUSD"
TEHRAN = dt.timezone(dt.timedelta(hours=3, minutes=30))

BTN_CRYPTO = "🪙 سیگنال کریپتو"
BTN_GOLD = "🥇 سیگنال طلا"
BTN_NEWS = "📅 خبرهای امروز"
BTN_WG = "🐋 نهنگ طلا"
BTN_WC = "🐋 نهنگ کریپتو"
KEYBOARD = {"keyboard": [[{"text": BTN_CRYPTO}, {"text": BTN_GOLD}],
                         [{"text": BTN_WC}, {"text": BTN_WG}], [{"text": BTN_NEWS}]],
            "resize_keyboard": True}

STABLES = {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "USDE", "EUR", "EURI",
           "AEUR", "XUSD", "USD1", "PYUSD", "UST"}
GOLD_MIN_SL = 2.5
CALM_SECONDS = 15
_last_use = {}
_cache = {}


# ------------------------------------------------------------------ telegram
def tg(method, **kw):
    files = kw.pop("files", None)
    try:
        r = requests.post(f"{API}/{method}", data=kw, files=files, timeout=60)
        return r.json()
    except Exception as e:
        print("telegram error", method, e)
        return {}


def send_text(chat, text, keyboard=True):
    kw = {"chat_id": chat, "text": text, "disable_web_page_preview": True}
    if keyboard:
        kw["reply_markup"] = json.dumps(KEYBOARD)
    tg("sendMessage", **kw)


def send_photo(chat, buf, caption):
    buf.seek(0)
    res = tg("sendPhoto", chat_id=chat, caption=caption[:1000],
             files={"photo": ("chart.png", buf, "image/png")})
    if not res.get("ok"):
        send_text(chat, caption)


# ------------------------------------------------------------------ helpers
def fmt(p):
    p = float(p)
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 100:
        return f"{p:.2f}"
    if p >= 1:
        return f"{p:.4f}"
    return f"{p:.6f}"


def tehran_time(ts):
    return pd.Timestamp(ts).tz_convert(TEHRAN).strftime("%H:%M")


def cached(key, ttl, fn):
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    val = fn()
    _cache[key] = (now, val)
    return val


# ------------------------------------------------------------------ indicators
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    h, l = df["High"], df["Low"]
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def macd(s):
    m = ema(s, 12) - ema(s, 26)
    return m, ema(m, 9)


def h1_trend(df1h):
    c = df1h["Close"]
    e50, e200, last = ema(c, 50).iloc[-1], ema(c, 200).iloc[-1], c.iloc[-1]
    if e50 > e200 and last > e50:
        return "UP"
    if e50 < e200 and last < e50:
        return "DOWN"
    return "FLAT"


# ------------------------------------------------------------------ analysis
def build_levels(entry, side, swing, a, floor=0.0, cap_atr=3.5):
    if side == "BUY":
        sl = min(swing - 0.2 * a, entry - 1.2 * a)
        risk = max(entry - sl, floor)
        sl = entry - risk
        tps = [entry + risk * k for k in (1, 2, 3)]
    else:
        sl = max(swing + 0.2 * a, entry + 1.2 * a)
        risk = max(sl - entry, floor)
        sl = entry + risk
        tps = [entry - risk * k for k in (1, 2, 3)]
    if risk > cap_atr * a:
        return None
    return sl, tps, risk


def analyze(df15, df1h, kind, tv=None, floor=0.0):
    """Return dict with 'signal' (or None) and 'state' (always)."""
    if len(df15) < 210 or len(df1h) < 210:
        return {"signal": None, "state": {"note": "داده کافی نیست"}}
    c = df15["Close"]
    last = float(c.iloc[-1])
    a = float(atr(df15).iloc[-1])
    adx_s, pdi, mdi = adx(df15)
    adx15 = float(adx_s.iloc[-1])
    e9, e21, e20, e50 = ema(c, 9), ema(c, 21), ema(c, 20), ema(c, 50)
    r14 = float(rsi(c, 14).iloc[-1])
    r2 = float(rsi(c, 2).iloc[-1])
    sma200 = float(c.rolling(200).mean().iloc[-1])
    hh = float(df15["High"].iloc[-21:-1].max())
    ll = float(df15["Low"].iloc[-21:-1].min())
    hi, lo = float(df15["High"].iloc[-1]), float(df15["Low"].iloc[-1])

    hits = {"BUY": [], "SELL": []}
    # 1) Turtle breakout
    if adx15 >= 20:
        if last > hh:
            hits["BUY"].append("Turtle: شکست سقف ۲۰ کندل")
        if last < ll:
            hits["SELL"].append("Turtle: شکست کف ۲۰ کندل")
    # 2) Holy Grail
    if adx15 >= 22:
        if pdi.iloc[-1] > mdi.iloc[-1] and lo <= e20.iloc[-1] < last and last > e50.iloc[-1]:
            hits["BUY"].append("Holy Grail: پولبک به EMA20 در روند صعودی")
        if mdi.iloc[-1] > pdi.iloc[-1] and hi >= e20.iloc[-1] > last and last < e50.iloc[-1]:
            hits["SELL"].append("Holy Grail: پولبک به EMA20 در روند نزولی")
    # 3) Connors RSI2
    if not np.isnan(sma200):
        if r2 < 10 and last > sma200:
            hits["BUY"].append(f"Connors RSI2: افراط فروش ({r2:.0f}) در روند صعودی")
        if r2 > 90 and last < sma200:
            hits["SELL"].append(f"Connors RSI2: افراط خرید ({r2:.0f}) در روند نزولی")
    # 4) Bollinger squeeze breakout
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    upper, lower = ma20 + 2 * sd20, ma20 - 2 * sd20
    width = (upper - lower) / ma20
    squeezed = width.iloc[-6:-1].min() <= width.iloc[-120:].quantile(0.2)
    if squeezed:
        if last > upper.iloc[-1]:
            hits["BUY"].append("Bollinger: شکست باند بالا بعد از فشردگی")
        if last < lower.iloc[-1]:
            hits["SELL"].append("Bollinger: شکست باند پایین بعد از فشردگی")
    # 5) MACD flip / EMA cross
    m, sig = macd(c)
    if m.iloc[-2] <= sig.iloc[-2] and m.iloc[-1] > sig.iloc[-1]:
        hits["BUY"].append("MACD: کراس صعودی")
    if m.iloc[-2] >= sig.iloc[-2] and m.iloc[-1] < sig.iloc[-1]:
        hits["SELL"].append("MACD: کراس نزولی")
    if e9.iloc[-2] <= e21.iloc[-2] and e9.iloc[-1] > e21.iloc[-1]:
        hits["BUY"].append("EMA9/21: کراس صعودی")
    if e9.iloc[-2] >= e21.iloc[-2] and e9.iloc[-1] < e21.iloc[-1]:
        hits["SELL"].append("EMA9/21: کراس نزولی")

    trend = h1_trend(df1h)
    adx1h = float(adx(df1h)[0].iloc[-1])
    vol_ratio = None
    if "Volume" in df15.columns and kind == "crypto":
        v = df15["Volume"]
        base = float(v.iloc[-21:-1].mean())
        vol_ratio = float(v.iloc[-1]) / base if base > 0 else None

    state = {"trend": trend, "adx15": adx15, "adx1h": adx1h, "rsi": r14, "price": last,
             "hits": {k: len(v) for k, v in hits.items()}}

    best = None
    for side in ("BUY", "SELL"):
        mine = hits[side]
        if not mine:
            continue
        if (side == "BUY" and trend != "UP") or (side == "SELL" and trend != "DOWN"):
            continue
        if (side == "BUY" and r14 > 78) or (side == "SELL" and r14 < 22):
            continue
        extras = []
        ok_tv = None
        if tv:
            ok_tv = _tv_agree(tv.get("1h"), side)
            if ok_tv is False:
                continue
            if ok_tv:
                extras.append("TradingView ۱ساعته موافقه")
            if _tv_agree(tv.get("4h"), side):
                extras.append("TradingView ۴ساعته موافقه")
        if adx1h >= 20:
            extras.append("روند ۱ساعته قدرت داره (ADX)")
        if vol_ratio is not None and vol_ratio >= 1.5:
            extras.append(f"حجم {vol_ratio:.1f} برابر میانگین")
        score = len(mine) + 1 + len(extras)
        if score < MIN_SCORE:
            continue
        swing = float(df15["Low"].iloc[-9:].min() if side == "BUY" else df15["High"].iloc[-9:].max())
        lv = build_levels(last, side, swing, a, floor=floor)
        if lv is None:
            continue
        sl, tps, risk = lv
        cand = {"side": side, "entry": last, "sl": sl, "tps": tps, "risk": risk, "score": score,
                "reasons": mine, "extras": extras, "trend": trend, "adx": adx15, "rsi": r14}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return {"signal": best, "state": state}


def _tv_agree(rec, side):
    if not rec:
        return None
    good = ("BUY", "STRONG_BUY") if side == "BUY" else ("SELL", "STRONG_SELL")
    bad = ("SELL", "STRONG_SELL") if side == "BUY" else ("BUY", "STRONG_BUY")
    return True if rec in good else (False if rec in bad else None)


def no_signal_text(name, st):
    t = {"UP": "صعودی", "DOWN": "نزولی", "FLAT": "بدون روند مشخص"}.get(st.get("trend"), "—")
    if "note" in st:
        return f"{name}: {st['note']}"
    why = []
    if st["trend"] == "FLAT":
        why.append("روند ۱ساعته مشخص نیست")
    if sum(st["hits"].values()) == 0:
        why.append("هیچ استراتژی‌ای فعال نشده")
    else:
        why.append("استراتژی فعال شده ولی با روند ۱ساعته یا امتیاز کافی جور نبود")
    return (f"⏸ {name}: الان سیگنال خوبی نیست.\n"
            f"قیمت: {fmt(st['price'])} | روند ۱ساعته: {t} | RSI: {st['rsi']:.0f} | ADX: {st['adx15']:.0f}\n"
            f"دلیل: {'، '.join(why)}.\nالکی سیگنال نمی‌دم؛ چند دقیقه بعد دوباره امتحان کن.")


# ------------------------------------------------------------------ chart + message
def chart_png(df, side, entry, sl, tps, title):
    d = df.iloc[-70:]
    e20 = ema(df["Close"], 20).iloc[-70:]
    e50 = ema(df["Close"], 50).iloc[-70:]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for i, (o, h, l, c) in enumerate(zip(d["Open"], d["High"], d["Low"], d["Close"])):
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=col, lw=0.8)
        body = max(abs(c - o), (d["High"].max() - d["Low"].min()) * 0.002)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o, c)), 0.6, body, color=col))
    xs = range(len(d))
    ax.plot(xs, e20.values, color="#ffb74d", lw=1, label="EMA20")
    ax.plot(xs, e50.values, color="#64b5f6", lw=1, label="EMA50")
    ax.axhline(entry, color="#ffffff", lw=1, ls="--")
    ax.axhline(sl, color="#ef5350", lw=1.2)
    for tp in tps:
        ax.axhline(tp, color="#66bb6a", lw=1)
    x_txt = len(d) + 0.5
    ax.text(x_txt, entry, f"ENTRY {fmt(entry)}", color="white", fontsize=7, va="center")
    ax.text(x_txt, sl, f"SL {fmt(sl)}", color="#ef5350", fontsize=7, va="center")
    for i, tp in enumerate(tps, 1):
        ax.text(x_txt, tp, f"TP{i} {fmt(tp)}", color="#66bb6a", fontsize=7, va="center")
    pad = (max(tps + [d["High"].max()]) - min(tps + [sl, d["Low"].min()])) * 0.03
    ax.set_ylim(min(d["Low"].min(), sl, *tps) - pad, max(d["High"].max(), sl, *tps) + pad)
    ax.set_xlim(-1, len(d) + 11)
    ax.tick_params(colors="#9aa0a6", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.set_title(title, color="white", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, facecolor="#0e1117", labelcolor="white", edgecolor="#30363d")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def signal_caption(label, sig, extra_line=""):
    icon = "🟢 خرید" if sig["side"] == "BUY" else "🔴 فروش"
    stars = "⭐" * min(sig["score"], 5)
    reasons = "\n".join(f"• {r}" for r in sig["reasons"])
    extras = "\n".join(f"• {e}" for e in sig["extras"])
    trend = "صعودی" if sig["trend"] == "UP" else "نزولی"
    txt = (f"{icon} {label}\n{stars} امتیاز {min(sig['score'], 5)}/5\n\n"
           f"📍 ورود: {fmt(sig['entry'])}\n"
           f"🛑 حد ضرر: {fmt(sig['sl'])}  (ریسک {fmt(sig['risk'])})\n"
           f"🎯 TP1: {fmt(sig['tps'][0])}\n🎯 TP2: {fmt(sig['tps'][1])}\n🎯 TP3: {fmt(sig['tps'][2])}\n\n"
           f"📊 تحلیل (کندل ۱۵ دقیقه):\n{reasons}\n"
           f"• روند ۱ساعته: {trend}\n{extras}\n"
           f"• ADX={sig['adx']:.0f} ، RSI={sig['rsi']:.0f}\n\n"
           f"🧭 بعد از TP1 حد ضرر رو بیار روی نقطه ورود. ریسک هر معامله حداکثر ۱٪ حساب.\n")
    if extra_line:
        txt += extra_line + "\n"
    return txt + "⚠️ سود تضمینی نیست؛ این پیشنهاد آموزشیه، نه مشاوره مالی."


# ------------------------------------------------------------------ crypto
def binance_klines(symbol, interval, limit=300):
    r = requests.get(f"{BIN}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
    r.raise_for_status()
    rows = r.json()
    now_ms = time.time() * 1000
    rows = [x for x in rows if x[6] < now_ms]
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume", "ct", "q", "n", "tb", "tq", "i"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = df[col].astype(float)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def top_symbols(n):
    def fetch():
        r = requests.get(f"{BIN}/ticker/24hr", timeout=40)
        r.raise_for_status()
        out = []
        for x in r.json():
            s = x["symbol"]
            if not s.endswith("USDT"):
                continue
            base = s[:-4]
            if base in STABLES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
                continue
            out.append((float(x["quoteVolume"]), s))
        out.sort(reverse=True)
        return [s for _, s in out[:n]]
    return cached(f"top{n}", 1800, fetch)


def crypto_one(symbol):
    try:
        d15 = binance_klines(symbol, "15m")
        d1h = binance_klines(symbol, "1h")
        res = analyze(d15, d1h, "crypto")
        res["symbol"], res["df15"] = symbol, d15
        return res
    except Exception as e:
        print("crypto error", symbol, e)
        return None


def crypto_flow(chat):
    send_text(chat, f"⏳ دارم {TOP_N} ارز پرمعامله رو اسکن می‌کنم (حدود ۱۰ تا ۲۰ ثانیه)...")
    syms = top_symbols(TOP_N)
    with ThreadPoolExecutor(8) as ex:
        results = [r for r in ex.map(crypto_one, syms) if r]
    if not results:
        send_text(chat, "نتونستم به اطلاعات بازار وصل بشم. چند دقیقه دیگه دوباره امتحان کن.")
        return
    sigs = sorted([r for r in results if r["signal"]], key=lambda r: -r["signal"]["score"])
    if not sigs:
        btc = next((r for r in results if r["symbol"] == "BTCUSDT"), None)
        extra = ""
        if btc:
            extra = "\n\n" + no_signal_text("بیت‌کوین (معیار بازار)", btc["state"])
        send_text(chat, f"⏸ از {len(results)} ارز بررسی‌شده، الان هیچ‌کدوم سیگنال باکیفیت نداره.\n"
                        f"الکی سیگنال نمی‌دم؛ ۱۵ دقیقه دیگه دوباره امتحان کن." + extra)
        return
    send_text(chat, f"✅ از {len(results)} ارز، {len(sigs)} سیگنال پیدا شد. بهترین‌ها:")
    for r in sigs[:3]:
        s, sym = r["signal"], r["symbol"]
        label = f"{sym[:-4]}/USDT"
        buf = chart_png(r["df15"], s["side"], s["entry"], s["sl"], s["tps"], f"{label} 15m")
        send_photo(chat, buf, signal_caption(label, s))


def crypto_symbol_flow(chat, base):
    sym = base.upper() + "USDT"
    r = crypto_one(sym)
    if not r:
        send_text(chat, f"ارز {base.upper()} رو توی بایننس پیدا نکردم. مثال درست: BTC یا SOL")
        return
    if r["signal"]:
        s = r["signal"]
        label = f"{base.upper()}/USDT"
        buf = chart_png(r["df15"], s["side"], s["entry"], s["sl"], s["tps"], f"{label} 15m")
        send_photo(chat, buf, signal_caption(label, s))
    else:
        send_text(chat, no_signal_text(f"{base.upper()}/USDT", r["state"]))


# ------------------------------------------------------------------ gold
def gold_df(interval, period):
    import yfinance as yf
    try:
        yf.set_tz_cache_location("/tmp/yfcache")
    except Exception:
        pass
    for sym in ("XAUUSD=X", "GC=F"):
        try:
            df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False)
        except Exception as e:
            print("yf error", sym, e)
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        if len(df) < 50:
            continue
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        return df, sym
    return None, None


def tv_summary(interval):
    try:
        from tradingview_ta import TA_Handler
        a = TA_Handler(symbol="XAUUSD", exchange="OANDA", screener="forex", interval=interval).get_analysis()
        return a.summary.get("RECOMMENDATION"), a.indicators.get("close")
    except Exception as e:
        print("tv error", interval, e)
        return None, None


def get_news():
    def fetch():
        data = requests.get(CAL_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0"}).json()
        if not isinstance(data, list):
            raise ValueError("bad calendar")
        return [(pd.to_datetime(e["date"], utc=True), e.get("title", "News"))
                for e in data if e.get("country") == "USD" and e.get("impact") == "High"]
    try:
        return cached("cal", 1800, fetch), True
    except Exception as e:
        print("calendar error", e)
        old = _cache.get("cal")
        return (old[1], True) if old else ([], False)


def news_flow(chat):
    events, ok = get_news()
    if not ok:
        send_text(chat, "تقویم خبری الان در دسترس نیست. تقویم اقتصادی رو خودت چک کن.")
        return
    now = pd.Timestamp.now(tz=TEHRAN)
    today = sorted([(t, n) for t, n in events if t.tz_convert(TEHRAN).date() == now.date()])
    if not today:
        send_text(chat, "✅ امروز خبر مهم دلار (تایم خبری) نداریم.")
        return
    lines = "\n".join(f"• {tehran_time(t)} — {n}" for t, n in today)
    send_text(chat, f"📅 امروز خبر مهم دلار داریم (ساعت به وقت ایران):\n{lines}\n\n"
                    "نزدیک این ساعت‌ها اسپرد و نوسان طلا بالاست؛ ربات ۱۵ دقیقه قبل و بعد خبر سیگنال عادی نمی‌ده.")


def gold_news_signal(events, df5):
    now = pd.Timestamp.now(tz="UTC")
    a5 = float(atr(df5).iloc[-1])
    for t, title in events:
        mins = (now - t).total_seconds() / 60.0
        if not (0 <= mins <= 25):
            continue
        before = df5[df5.index + pd.Timedelta(minutes=5) <= t]
        post = df5[df5.index >= t]
        if before.empty or len(post) < 2:
            continue
        ref, last = float(before["Close"].iloc[-1]), float(df5["Close"].iloc[-1])
        move = last - ref
        if abs(move) < 4.0:
            continue
        side = "BUY" if move > 0 else "SELL"
        l2 = df5.iloc[-2:]
        if side == "BUY" and not (l2["Close"] > l2["Open"]).all():
            continue
        if side == "SELL" and not (l2["Close"] < l2["Open"]).all():
            continue
        ext = float(post["High"].max() if side == "BUY" else post["Low"].min())
        if abs(ext - ref) == 0 or abs(ext - last) / abs(ext - ref) >= 0.5:
            continue
        rec = df5.iloc[-3:]
        swing = float(rec["Low"].min() if side == "BUY" else rec["High"].max())
        lv = build_levels(last, side, swing, a5, floor=GOLD_MIN_SL, cap_atr=12 / max(a5, 0.01))
        if lv is None or lv[2] > 12:
            continue
        sl, tps, risk = lv
        return {"side": side, "entry": last, "sl": sl, "tps": tps, "risk": risk, "score": 3,
                "reasons": [f"واکنش به خبر «{title}»: حرکت {move:+.1f}$ و ادامه‌دار"],
                "extras": [], "trend": "UP" if side == "BUY" else "DOWN", "adx": 0, "rsi": 50}, title
    return None, None


def gold_flow(chat):
    send_text(chat, "⏳ دارم بازار طلا رو تحلیل می‌کنم...")
    df15, src = gold_df("15m", "30d")
    df1h, _ = gold_df("1h", "60d")
    if df15 is None or df1h is None:
        send_text(chat, "نتونستم قیمت طلا رو بگیرم. چند دقیقه دیگه دوباره امتحان کن.")
        return
    now = pd.Timestamp.now(tz="UTC")
    if (now - df15.index[-1]).total_seconds() / 60 > 40:
        send_text(chat, "🔒 بازار طلا الان بسته‌ست (آخر هفته یا تعطیلی). دوشنبه صبح دوباره امتحان کن.")
        return
    rec15, tv_close = tv_summary("15m")
    rec1h, _ = tv_summary("1h")
    rec4h, _ = tv_summary("4h")
    off = 0.0
    if tv_close:
        o = float(tv_close) - float(df15["Close"].iloc[-1])
        if abs(o) < 80 and (src == "GC=F" or abs(o) > 1.5):
            off = o
    if off:
        for d in (df15, df1h):
            for c in ("Open", "High", "Low", "Close"):
                d[c] = d[c] + off

    events, _ = get_news()
    in_quiet = any(-15 <= (t - now).total_seconds() / 60 <= 15 for t, _ in events)
    near = [(t, n) for t, n in events if 0 < (t - now).total_seconds() / 60 <= 60]
    warn = ""
    if near:
        t, n = near[0]
        warn = f"⚠️ خبر مهم نزدیکه: {n} ساعت {tehran_time(t)} (وقت ایران)"

    df5, _ = gold_df("5m", "5d")
    if df5 is not None and any(0 <= (now - t).total_seconds() / 60 <= 25 for t, _ in events):
        if off:
            for c in ("Open", "High", "Low", "Close"):
                df5[c] = df5[c] + off
        sig, title = gold_news_signal(events, df5)
        if sig:
            buf = chart_png(df5, sig["side"], sig["entry"], sig["sl"], sig["tps"], "XAUUSD 5m NEWS")
            send_photo(chat, buf, "📰 سیگنال تایم خبری\n" + signal_caption("طلا (XAUUSD)", sig,
                       "⚠️ ریسک خبری بالاست؛ حجم رو نصف کن."))
            return
    if in_quiet:
        send_text(chat, "📰 الان تایم خبریه (۱۵ دقیقه قبل تا ۱۵ دقیقه بعد خبر مهم دلار). "
                        "سیگنال عادی نمی‌دم چون اسپرد و نوسان بالاست. بعد از خبر دوباره امتحان کن.\n" + warn)
        return
    res = analyze(df15, df1h, "gold", tv={"1h": rec1h, "4h": rec4h}, floor=GOLD_MIN_SL)
    if res["signal"]:
        s = res["signal"]
        tvtxt = f"TradingView: {rec1h or '—'} (۱ساعته) | {rec4h or '—'} (۴ساعته) | {rec15 or '—'} (۱۵دقیقه)"
        buf = chart_png(df15, s["side"], s["entry"], s["sl"], s["tps"], "XAUUSD 15m")
        send_photo(chat, buf, signal_caption("طلا (XAUUSD)", s, tvtxt + ("\n" + warn if warn else "")))
    else:
        send_text(chat, no_signal_text("طلا (XAUUSD)", res["state"]) + ("\n" + warn if warn else ""))


# ------------------------------------------------------------------ router
# ------------------------------------------------------------------ whales
def money(x):
    x = float(x)
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    if x >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def market_clock():
    """Sessions, liquidity level, bank holidays. Returns dict."""
    now = pd.Timestamp.now(tz="UTC")
    ny = now.tz_convert("America/New_York")
    ldn = now.tz_convert("Europe/London")
    tok = now.tz_convert("Asia/Tokyo")
    syd = now.tz_convert("Australia/Sydney")
    wd = ny.weekday()  # Mon=0
    closed = (wd == 5) or (wd == 4 and ny.hour >= 17) or (wd == 6 and ny.hour < 17)
    open_ = {
        "سیدنی": wd < 5 and 7 <= syd.hour < 16,
        "توکیو": tok.weekday() < 5 and 9 <= tok.hour < 18,
        "لندن": ldn.weekday() < 5 and 8 <= ldn.hour < 17,
        "نیویورک": wd < 5 and 8 <= ny.hour < 17,
    }
    if closed:
        open_ = {k: False for k in open_}
    # approximate bank holidays (US / UK), 2026-2027
    us_h = {"2026-11-11": "Veterans Day", "2026-11-26": "Thanksgiving", "2026-12-25": "Christmas",
            "2027-01-01": "New Year", "2027-01-18": "MLK Day", "2027-02-15": "Presidents Day"}
    uk_h = {"2026-12-25": "Christmas", "2026-12-28": "Boxing Day", "2027-01-01": "New Year"}
    hol = []
    d = ny.strftime("%Y-%m-%d")
    if d in us_h:
        hol.append(f"تعطیلی بانک‌های آمریکا ({us_h[d]})")
    d2 = ldn.strftime("%Y-%m-%d")
    if d2 in uk_h:
        hol.append(f"تعطیلی بانک‌های بریتانیا ({uk_h[d2]})")
    if closed:
        lvl, score = "بازار بسته است", 0
    elif open_["لندن"] and open_["نیویورک"]:
        lvl, score = "بسیار بالا (همپوشانی لندن و نیویورک)", 4
    elif open_["لندن"] or open_["نیویورک"]:
        lvl, score = "بالا", 3
    elif open_["توکیو"]:
        lvl, score = "متوسط (فقط آسیا)", 2
    elif open_["سیدنی"]:
        lvl, score = "کم", 1
    else:
        lvl, score = "خیلی کم", 1
    if hol and score > 1:
        lvl += " — ولی به‌خاطر تعطیلی بانکی کمتر از معمول"
        score -= 1
    return {"closed": closed, "open": [k for k, v in open_.items() if v], "level": lvl,
            "score": score, "holidays": hol}


def gc_volume_flow():
    """Gold futures (GC=F) volume pressure on 15m candles. None if unavailable."""
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="5d", interval="15m", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "Close", "Volume"]].dropna()
        df = df[df["Volume"] > 0]
        if len(df) < 60:
            return None
        recent = df.iloc[-8:]
        ratio = float(recent["Volume"].mean() / df["Volume"].iloc[:-8].mean())
        move = float(recent["Close"].iloc[-1] - recent["Open"].iloc[0])
        up_vol = float(recent.loc[recent["Close"] >= recent["Open"], "Volume"].sum())
        buy_share = up_vol / float(recent["Volume"].sum())
        return {"ratio": ratio, "move": move, "buy_share": buy_share}
    except Exception as e:
        print("gc error", e)
        return None


def cot_gold():
    """Latest CFTC managed-money positioning for gold futures (weekly)."""
    try:
        r = requests.get("https://publicreporting.cftc.gov/resource/72hh-3qpy.json",
                         params={"cftc_contract_market_code": "088691",
                                 "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 2},
                         timeout=20)
        rows = r.json()
        a = rows[0]
        net = int(float(a["m_money_positions_long_all"])) - int(float(a["m_money_positions_short_all"]))
        chg = None
        if len(rows) > 1:
            b = rows[1]
            chg = net - (int(float(b["m_money_positions_long_all"])) - int(float(b["m_money_positions_short_all"])))
        return {"net": net, "chg": chg, "date": str(a["report_date_as_yyyy_mm_dd"])[:10]}
    except Exception as e:
        print("cot error", e)
        return None


def gold_whale_flow(chat):
    send_text(chat, "🐋 دارم نقدینگی و رفتار نهنگ‌های طلا رو بررسی می‌کنم...")
    clk = market_clock()
    now = pd.Timestamp.now(tz=TEHRAN)
    lines = [f"🐋 نهنگ طلا — ساعت {now.strftime('%H:%M')} (وقت ایران)", ""]
    # liquidity
    lines.append(f"💧 نقدینگی: {clk['level']}")
    if clk["open"]:
        lines.append("بازارهای باز: " + "، ".join(clk["open"]))
    for h in clk["holidays"]:
        lines.append(f"🏦 {h}")
    if clk["closed"]:
        lines.append("\n🔒 بازار طلا بسته‌ست؛ دوشنبه صبح دوباره امتحان کن.")
        send_text(chat, "\n".join(lines))
        return
    # news
    events, _ = get_news()
    utc_now = pd.Timestamp.now(tz="UTC")
    quiet = any(-15 <= (t - utc_now).total_seconds() / 60 <= 15 for t, _ in events)
    near = [(t, n) for t, n in events if 15 < (t - utc_now).total_seconds() / 60 <= 60]
    if quiet:
        lines.append("📰 الان تایم خبر مهم دلاره (نوسان و اسپرد بالا).")
    elif near:
        t, n = near[0]
        lines.append(f"📰 خبر مهم تا کمتر از یک ساعت دیگه: {n} ساعت {tehran_time(t)}")
    else:
        lines.append("📰 خبر مهمی نزدیک نیست.")
    # whale proxy: gold futures volume
    gc = gc_volume_flow()
    pressure = None
    if gc:
        r, mv, bs = gc["ratio"], gc["move"], gc["buy_share"]
        if r >= 1.5 and mv > 0:
            pressure = "UP"
            txt = "خرید سنگین (حجم بالا + قیمت بالا رفته)"
        elif r >= 1.5 and mv < 0:
            pressure = "DOWN"
            txt = "فروش سنگین (حجم بالا + قیمت پایین اومده)"
        elif r >= 1.5:
            txt = "حجم بالا ولی قیمت جابه‌جا نشده (جذب یا توزیع؛ جهت نامشخص)"
        elif r < 0.8:
            txt = "حجم کمه؛ نهنگ‌ها الان فعال نیستن"
        else:
            txt = "حجم معمولیه؛ فعالیت غیرعادی نمی‌بینم"
        lines.append(f"\n🐳 حجم آتی طلا (۲ ساعت اخیر): {r:.1f}× میانگین، حرکت {mv:+.1f}$ → {txt}")
    else:
        lines.append("\n🐳 حجم آتی طلا الان در دسترس نیست.")
    cot = cot_gold()
    if cot:
        side = "خرید" if cot["net"] > 0 else "فروش"
        ch = ""
        if cot["chg"] is not None:
            ch = f"، تغییر هفته: {cot['chg']:+,}"
        lines.append(f"🏛 گزارش COT ({cot['date']}): صندوق‌های بزرگ خالص {side} ({cot['net']:+,} قرارداد{ch})")
    # trend bias
    df1h, _ = gold_df("1h", "60d")
    rec1h, _ = tv_summary("1h")
    rec4h, _ = tv_summary("4h")
    bias = h1_trend(df1h) if df1h is not None and len(df1h) >= 210 else "FLAT"
    up_set, dn_set = ("BUY", "STRONG_BUY"), ("SELL", "STRONG_SELL")
    ups = sum(1 for x in (rec1h, rec4h) if x in up_set) + (bias == "UP")
    dns = sum(1 for x in (rec1h, rec4h) if x in dn_set) + (bias == "DOWN")
    if ups >= 2 and ups > dns:
        direction = "UP"
    elif dns >= 2 and dns > ups:
        direction = "DOWN"
    else:
        direction = "FLAT"
    dname = {"UP": "صعودی 📈", "DOWN": "نزولی 📉", "FLAT": "خنثی/نامشخص ↔️"}[direction]
    lines.append(f"\n📊 جهت بازار: {dname}  (TradingView: ۱ساعته {rec1h or '—'} | ۴ساعته {rec4h or '—'})")
    # verdict
    verdict, go = "", False
    if quiet or near:
        verdict = "⏳ صبر کن؛ بعد از خبر وارد شو."
    elif clk["score"] <= 1:
        verdict = "⚠️ نقدینگی کمه؛ ورود پرریسکه. بهتره صبر کنی تا لندن یا نیویورک باز بشه."
    elif direction == "FLAT":
        verdict = "⏸ جهت مشخص نیست؛ الان وارد نشو."
    elif pressure and pressure != direction:
        verdict = "⚠️ فشار حجم خلاف جهت روند ۱ساعته‌ست؛ صبر کن."
    else:
        verdict = f"✅ شرایط برای ورود {'خرید' if direction == 'UP' else 'فروش'} مناسبه، البته فقط با سیگنال بعدی."
        go = True
    lines.append("\n" + verdict)
    lines.append("\nاین تحلیل تضمین سود نیست؛ نهنگ‌ها رو مستقیم نمی‌بینیم، فقط ردپاشون (حجم و گزارش COT) رو.")
    send_text(chat, "\n".join(lines))
    if go:
        gold_flow(chat)


def whale_trades(symbol):
    """Large taker trades from the latest 1000 aggTrades. Returns dict or None."""
    try:
        r = requests.get(f"{BIN}/aggTrades", params={"symbol": symbol, "limit": 1000}, timeout=20)
        r.raise_for_status()
        rows = r.json()
        if len(rows) < 50:
            return None
        notional = [float(x["p"]) * float(x["q"]) for x in rows]
        med = float(np.median(notional))
        thr = max(20000.0, med * 15)
        buy = sell = 0.0
        n = 0
        for x, v in zip(rows, notional):
            if v >= thr:
                n += 1
                if x["m"]:
                    sell += v      # buyer is maker -> taker sold
                else:
                    buy += v
        span = (rows[-1]["T"] - rows[0]["T"]) / 60000.0
        return {"symbol": symbol, "n": n, "buy": buy, "sell": sell, "total": buy + sell,
                "span": span, "thr": thr}
    except Exception as e:
        print("whale trades error", symbol, e)
        return None


def hour_flow(symbol):
    """1h volume spike and taker-buy share from klines."""
    try:
        r = requests.get(f"{BIN}/klines", params={"symbol": symbol, "interval": "1h", "limit": 26}, timeout=20)
        r.raise_for_status()
        rows = r.json()[:-1]  # drop open candle
        q = [float(x[7]) for x in rows]
        last = rows[-1]
        ratio = q[-1] / max(float(np.mean(q[:-1])), 1.0)
        tb_share = float(last[10]) / max(float(last[7]), 1.0)
        chg = (float(last[4]) / float(last[1]) - 1) * 100
        return {"ratio": ratio, "tb": tb_share, "chg": chg}
    except Exception as e:
        print("hour flow error", symbol, e)
        return None


def whale_crypto_flow(chat):
    send_text(chat, "🐋 دارم معاملات بزرگ ارزهای پرمعامله رو بررسی می‌کنم (حدود ۲۰ ثانیه)...")
    syms = top_symbols(30)
    with ThreadPoolExecutor(8) as ex:
        trades = [t for t in ex.map(whale_trades, syms) if t]
    trades = [t for t in trades if t["total"] >= 100000]
    if not trades:
        send_text(chat, "الان معامله‌ی بزرگ و غیرعادی ندیدم؛ نهنگ‌ها ساکتن. چند دقیقه بعد دوباره امتحان کن.")
        return
    trades.sort(key=lambda t: -t["total"])
    top = trades[:5]
    with ThreadPoolExecutor(8) as ex:
        hf = list(ex.map(lambda t: hour_flow(t["symbol"]), top))
        an = list(ex.map(lambda t: crypto_one(t["symbol"]), top))
    out = ["🐋 نهنگ کریپتو — بیشترین معاملات بزرگ همین الان:"]
    sig_list = []
    for t, h, a in zip(top, hf, an):
        base = t["symbol"][:-4]
        tot = t["total"]
        bp = t["buy"] / tot * 100
        lean = "UP" if bp >= 60 else ("DOWN" if bp <= 40 else "FLAT")
        line = [f"🔹 {base}/USDT — {t['n']} معامله بزرگ در {t['span']:.0f} دقیقه اخیر، جمعاً {money(tot)}",
                f"   خرید {bp:.0f}٪ | فروش {100-bp:.0f}٪ → " +
                {"UP": "نهنگ‌ها خریدارن 🟢", "DOWN": "نهنگ‌ها فروشنده‌ان 🔴", "FLAT": "متعادل ⚪"}[lean]]
        if h:
            line.append(f"   حجم ساعت اخیر {h['ratio']:.1f}× میانگین، قیمت ۱ساعته {h['chg']:+.1f}٪، خرید فعال {h['tb']*100:.0f}٪")
        if a:
            st = a["state"]
            trend = st.get("trend", "FLAT")
            tname = {"UP": "صعودی", "DOWN": "نزولی", "FLAT": "بدون روند"}[trend]
            if lean == trend and lean != "FLAT":
                view = "چشم‌انداز کوتاه‌مدت " + ("مثبت ✅" if lean == "UP" else "منفی ✅") + " (نهنگ و روند هم‌جهت)"
            elif lean == "FLAT" or trend == "FLAT":
                view = "چشم‌انداز نامشخص؛ نشونه‌ها هم‌جهت نیستن"
            else:
                view = "هشدار ⚠️ نهنگ‌ها خلاف روند ۱ساعته‌ان؛ احتمال برگشت یا تله"
            line.append(f"   روند ۱ساعته: {tname} → {view}")
            if a["signal"]:
                s = a["signal"]
                sig_list.append((t["symbol"], a))
                tps = " | ".join(fmt(x) for x in s["tps"])
                line.append(f"   📍 سیگنال {'خرید' if s['side']=='BUY' else 'فروش'}: ورود {fmt(s['entry'])} | "
                            f"حد ضرر {fmt(s['sl'])} | TP: {tps}")
            else:
                line.append("   فعلاً سیگنال ورود باکیفیت نداره.")
        out.append("\n".join(line))
    out.append("\nℹ️ حد ضرر و حد سود خود نهنگ‌ها قابل دیدن نیست. سطوح بالا رو ربات از ساختار قیمت حساب کرده، "
               "نه از روی نهنگ‌ها. تضمین سود نیست.")
    send_text(chat, "\n\n".join(out))
    for sym, a in sig_list[:2]:
        s = a["signal"]
        label = f"{sym[:-4]}/USDT"
        buf = chart_png(a["df15"], s["side"], s["entry"], s["sl"], s["tps"], f"{label} 15m")
        send_photo(chat, buf, signal_caption(label, s))



WELCOME = ("سلام 👋 من ربات سیگنال کریپتو و طلام.\n\n"
           "🪙 سیگنال کریپتو: ارزهای پرمعامله رو اسکن می‌کنم و بهترین فرصت‌ها رو می‌فرستم\n"
           "🥇 سیگنال طلا: طلای فارکس رو همین لحظه تحلیل می‌کنم\n"
           "🐋 نهنگ کریپتو: ارزهایی که الان معاملات بزرگ دارن + چشم‌انداز\n"
           "🐋 نهنگ طلا: نقدینگی، ساعت بازارها، خبر و جهت؛ الان وارد شم یا نه؟\n"
           "📅 خبرهای امروز: خبرهای مهم دلار (تایم خبری)\n\n"
           "یه ارز دلخواه هم می‌تونی تایپ کنی، مثلاً: SOL\n\n"
           "هر سیگنال: نقطه ورود، حد ضرر و ۳ حد سود. اگه فرصت خوبی نباشه، صادقانه می‌گم.\n"
           "⚠️ هیچ سیگنالی سود تضمینی نداره و این‌ها مشاوره مالی نیست.")


def handle(update):
    msg = update.get("message")
    if not msg or "text" not in msg:
        return
    chat = msg["chat"]["id"]
    text = msg["text"].strip()
    if ALLOWED and str(chat) not in ALLOWED:
        send_text(chat, "این ربات خصوصیه.", keyboard=False)
        return
    try:
        if text in ("/start", "/help"):
            send_text(chat, WELCOME)
            return
        action = None
        if text == BTN_CRYPTO or text.lower() == "/crypto":
            action = lambda: crypto_flow(chat)
        elif text == BTN_GOLD or text.lower() == "/gold":
            action = lambda: gold_flow(chat)
        elif text == BTN_WG or text.lower() == "/whalegold":
            action = lambda: gold_whale_flow(chat)
        elif text == BTN_WC or text.lower() == "/whalecrypto":
            action = lambda: whale_crypto_flow(chat)
        elif text == BTN_NEWS or text.lower() == "/news":
            action = lambda: news_flow(chat)
        elif text.isalnum() and text.isascii() and 2 <= len(text) <= 10:
            action = lambda: crypto_symbol_flow(chat, text)
        else:
            send_text(chat, "از دکمه‌های پایین استفاده کن یا اسم یه ارز بنویس (مثلاً BTC).")
            return
        now = time.time()
        if now - _last_use.get(chat, 0) < CALM_SECONDS:
            send_text(chat, "یه لحظه صبر کن و دوباره بزن 🙏")
            return
        _last_use[chat] = now
        action()
    except Exception as e:
        print("handler error", repr(e))
        send_text(chat, "یه خطای موقتی پیش اومد. چند لحظه بعد دوباره امتحان کن.")


# ------------------------------------------------------------------ server + loop
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def run_health():
    port = int(os.environ.get("PORT", "10000"))
    HTTPServer(("0.0.0.0", port), Health).serve_forever()


def main():
    if not TOKEN:
        raise SystemExit("TELEGRAM_TOKEN is not set")
    threading.Thread(target=run_health, daemon=True).start()
    tg("deleteWebhook")
    pool = ThreadPoolExecutor(6)
    offset = 0
    print("bot started")
    while True:
        try:
            r = requests.get(f"{API}/getUpdates", params={"timeout": 30, "offset": offset}, timeout=45).json()
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                pool.submit(handle, u)
        except Exception as e:
            print("poll error", repr(e))
            time.sleep(3)


if __name__ == "__main__":
    main()
