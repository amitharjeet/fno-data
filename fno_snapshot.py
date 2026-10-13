"""
Daily F&O snapshot for the watchlist, built only from NSE's public end-of-day files.
Writes data/latest.json and data/latest.md (plus a cached price history in data/history/).
No logins, no API keys. Runs on GitHub Actions (see .github/workflows/fno-daily.yml).
"""
import io, json, os, sys, time, zipfile, datetime as dt
import numpy as np
import pandas as pd
import requests

SYMBOLS = ["TVSMOTOR", "SHRIRAMFIN", "ADANIENT", "ADANIGREEN", "ADANIPOWER", "JSWENERGY",
           "RELIANCE", "BDL", "SOLARINDS", "BEL", "OBEROIRLTY", "KAYNES", "HINDZINC"]
BASE = "https://nsearchives.nseindia.com/content"
HIST_DIR = "data/history"
HIST_DAYS = 330          # calendar days of cash history to keep (enough for a 200 DMA)
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/",
})

def log(*a):
    print(*a, file=sys.stderr, flush=True)

def get(url, tries=3):
    """Return bytes, or None if the file does not exist (holiday / not yet published)."""
    for i in range(tries):
        try:
            r = S.get(url, timeout=45)
            if r.status_code == 200 and len(r.content) > 200:
                return r.content
            if r.status_code in (403, 404):
                if r.status_code == 403:
                    log("403 from", url)
                return None
        except requests.RequestException as e:
            log("retry", i, url, e)
        time.sleep(1.5 * (i + 1))
    return None

def read_zip_csv(b):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip() for c in df.columns]
    return df

def fo_url(d):  return f"{BASE}/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
def cm_url(d):  return f"{BASE}/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"

# ---------------------------------------------------------------- history
def load_hist(sym):
    p = f"{HIST_DIR}/{sym}.csv"
    if os.path.exists(p):
        return pd.read_csv(p, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

def update_history(last_session):
    os.makedirs(HIST_DIR, exist_ok=True)
    hist = {s: load_hist(s) for s in SYMBOLS}
    have = max([h["date"].max() for h in hist.values() if len(h)] or [pd.NaT])
    start = last_session - dt.timedelta(days=HIST_DAYS)
    if pd.notna(have):
        start = max(start, have.date() + dt.timedelta(days=1))
    d, fetched, missing = start, 0, 0
    while d <= last_session:
        if d.weekday() < 5:
            b = get(cm_url(d))
            if b:
                cm = read_zip_csv(b)
                cm = cm[(cm["SctySrs"].astype(str).str.strip() == "EQ") & (cm["TckrSymb"].isin(SYMBOLS))]
                for _, r in cm.iterrows():
                    row = pd.DataFrame([{"date": pd.Timestamp(d), "open": r["OpnPric"], "high": r["HghPric"],
                                         "low": r["LwPric"], "close": r["ClsPric"], "volume": r["TtlTradgVol"]}])
                    hist[r["TckrSymb"]] = pd.concat([hist[r["TckrSymb"]], row], ignore_index=True)
                fetched += 1
                time.sleep(0.4)
            else:
                missing += 1       # holiday, or file not published
        d += dt.timedelta(days=1)
    log(f"history: fetched {fetched} sessions, {missing} non-trading/missing days")
    cutoff = pd.Timestamp(last_session - dt.timedelta(days=HIST_DAYS))
    for s, h in hist.items():
        h = h.drop_duplicates("date").sort_values("date")
        h = h[h["date"] >= cutoff]
        h.to_csv(f"{HIST_DIR}/{s}.csv", index=False)
        hist[s] = h.reset_index(drop=True)
    return hist

# ---------------------------------------------------------------- indicators
def rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au, ad = up.ewm(alpha=1 / n, adjust=False).mean(), dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def supertrend(df, n=10, mult=3.0):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    hl2 = (h + l) / 2
    ub, lb = (hl2 + mult * atr).values.copy(), (hl2 - mult * atr).values.copy()
    fu, fl, st, dirn = ub.copy(), lb.copy(), np.full(len(df), np.nan), np.ones(len(df))
    cv = c.values
    for i in range(1, len(df)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cv[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or cv[i - 1] < fl[i - 1]) else fl[i - 1]
        if dirn[i - 1] == 1:
            dirn[i] = -1 if cv[i] < fl[i] else 1
        else:
            dirn[i] = 1 if cv[i] > fu[i] else -1
        st[i] = fl[i] if dirn[i] == 1 else fu[i]
    return st, dirn, atr

def candle_flags(df):
    out = []
    for i in range(max(1, len(df) - 3), len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        body, rng = abs(r.close - r.open), max(r.high - r.low, 1e-9)
        lo_sh, up_sh = min(r.open, r.close) - r.low, r.high - max(r.open, r.close)
        f = []
        if body <= 0.1 * rng: f.append("doji")
        if lo_sh >= 2 * body and up_sh <= 0.3 * rng and body > 0: f.append("hammer-like")
        if up_sh >= 2 * body and lo_sh <= 0.3 * rng and body > 0: f.append("shooting-star-like")
        if r.close > r.open and p.close < p.open and r.close >= p.open and r.open <= p.close: f.append("bullish engulfing")
        if r.close < r.open and p.close > p.open and r.open >= p.close and r.close <= p.open: f.append("bearish engulfing")
        if r.high < p.high and r.low > p.low: f.append("inside bar")
        if r.open > p.high: f.append("gap up")
        if r.open < p.low: f.append("gap down")
        out.append({"date": str(r.date.date()), "flags": f})
    return out

def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)

def technicals(h):
    n = len(h)
    if n < 15:
        return {"error": f"only {n} sessions of history"}
    c = h["close"]
    t = {"sessions_of_history": n}
    for k in (20, 50, 200):
        t[f"sma{k}"] = r2(c.rolling(k).mean().iloc[-1]) if n >= k else None
    t["dma200_note"] = None if n >= 200 else f"only {n} sessions, 200 DMA not available"
    t["rsi14"] = r2(rsi(c).iloc[-1]); t["rsi14_prev"] = r2(rsi(c).iloc[-2])
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; sig = macd.ewm(span=9, adjust=False).mean()
    t["macd"], t["macd_signal"], t["macd_hist"] = r2(macd.iloc[-1]), r2(sig.iloc[-1]), r2((macd - sig).iloc[-1])
    t["macd_hist_prev"] = r2((macd - sig).iloc[-2])
    st, dirn, atr = supertrend(h)
    t["supertrend"], t["supertrend_dir"], t["atr14"] = r2(st[-1]), ("UP" if dirn[-1] == 1 else "DOWN"), r2(atr.iloc[-1])
    v = h["volume"]
    t["vol_last"], t["vol_avg20"] = int(v.iloc[-1]), int(v.iloc[-21:-1].mean()) if n > 21 else None
    t["vol_ratio"] = r2(v.iloc[-1] / v.iloc[-21:-1].mean()) if n > 21 else None
    last = h.iloc[-1]; H, L, C = last.high, last.low, last.close
    P = (H + L + C) / 3
    t["pivots"] = {"P": r2(P), "R1": r2(2 * P - L), "S1": r2(2 * P - H), "R2": r2(P + H - L), "S2": r2(P - (H - L)),
                   "basis": f"classic pivots on {last.date.date()} H/L/C"}
    t["last3"] = [{"date": str(r.date.date()), "o": r2(r.open), "h": r2(r.high), "l": r2(r.low), "c": r2(r.close), "v": int(r.volume)}
                  for r in h.tail(3).itertuples()]
    t["high3"], t["low3"] = r2(h.tail(3).high.max()), r2(h.tail(3).low.min())
    t["candles"] = candle_flags(h)
    t["crossover_20_50"] = None
    if n >= 52:
        s20, s50 = c.rolling(20).mean(), c.rolling(50).mean()
        diff = (s20 - s50).dropna()
        if len(diff) > 5 and np.sign(diff.iloc[-1]) != np.sign(diff.iloc[-6:-1]).mean().round():
            t["crossover_20_50"] = "20/50 DMA crossover within last 5 sessions"
    return t

# ---------------------------------------------------------------- F&O
def buildup(p, oi):
    if p > 0 and oi > 0: return "LONG BUILDUP"
    if p < 0 and oi > 0: return "SHORT BUILDUP"
    if p > 0 and oi < 0: return "SHORT COVERING"
    if p < 0 and oi < 0: return "LONG UNWINDING"
    return "NEUTRAL"

def max_pain(strikes):
    ks = sorted(set(strikes["strike"]))
    best, bk = None, None
    for k in ks:
        ce = strikes[strikes.typ == "CE"]; pe = strikes[strikes.typ == "PE"]
        pain = ((k - ce.strike).clip(lower=0) * ce.oi).sum() + ((pe.strike - k).clip(lower=0) * pe.oi).sum()
        if best is None or pain < best: best, bk = pain, k
    return bk

def fo_for(sym, fo, session):
    f = fo[(fo["TckrSymb"] == sym) & (fo["FinInstrmTp"] == "STF")].copy()
    if f.empty:
        return {"error": "no futures rows in bhavcopy"}
    f["exp"] = pd.to_datetime(f["XpryDt"]); f = f.sort_values("exp")
    near = f.iloc[0]
    out = {"near_expiry": str(near.exp.date()), "calendar_days_to_expiry": (near.exp.date() - session).days,
           "fut_close": r2(near.ClsPric), "fut_prev_close": r2(near.PrvsClsgPric),
           "spot_underlying": r2(near.UndrlygPric), "lot_size": int(near.NewBrdLotQty) if "NewBrdLotQty" in near else None,
           "oi": int(near.OpnIntrst), "oi_change": int(near.ChngInOpnIntrst),
           "oi_change_pct": r2(100 * near.ChngInOpnIntrst / max(near.OpnIntrst - near.ChngInOpnIntrst, 1)),
           "fut_volume": int(near.TtlTradgVol)}
    pchg = near.ClsPric - near.PrvsClsgPric
    out["fut_price_change_pct"] = r2(100 * pchg / near.PrvsClsgPric)
    out["buildup"] = buildup(pchg, near.ChngInOpnIntrst)
    out["premium_vs_spot"] = r2(near.ClsPric - near.UndrlygPric)
    out["premium_pct"] = r2(100 * (near.ClsPric - near.UndrlygPric) / near.UndrlygPric)
    if len(f) > 1:
        nx = f.iloc[1]
        out["next_expiry"] = str(nx.exp.date()); out["next_oi"] = int(nx.OpnIntrst); out["next_oi_change"] = int(nx.ChngInOpnIntrst)
        out["rollover_pct"] = r2(100 * nx.OpnIntrst / (nx.OpnIntrst + near.OpnIntrst))
    o = fo[(fo["TckrSymb"] == sym) & (fo["FinInstrmTp"] == "STO")].copy()
    o["exp"] = pd.to_datetime(o["XpryDt"]); o = o[o.exp == near.exp]
    if o.empty:
        out["options"] = {"error": "no option rows for near expiry"}
        return out
    s = pd.DataFrame({"strike": o["StrkPric"].astype(float), "typ": o["OptnTp"].str.strip(),
                      "oi": o["OpnIntrst"].astype(float), "doi": o["ChngInOpnIntrst"].astype(float),
                      "ltp": o["ClsPric"].astype(float)})
    ce, pe = s[s.typ == "CE"], s[s.typ == "PE"]
    top = lambda d, col, k=5: [{"strike": r2(r.strike), "oi": int(r.oi), "oi_change": int(r.doi), "ltp": r2(r.ltp)}
                                for r in d.sort_values(col, ascending=False).head(k).itertuples()]
    out["options"] = {"expiry": str(near.exp.date()), "call_oi_total": int(ce.oi.sum()), "put_oi_total": int(pe.oi.sum()),
                      "pcr_oi": r2(pe.oi.sum() / ce.oi.sum()) if ce.oi.sum() else None,
                      "max_pain": r2(max_pain(s)),
                      "top_call_oi": top(ce, "oi"), "top_put_oi": top(pe, "oi"),
                      "top_call_oi_added": top(ce, "doi", 3), "top_put_oi_added": top(pe, "doi", 3)}
    return out

# ---------------------------------------------------------------- main
def main():
    today = dt.datetime.now(IST).date()
    session, fo_bytes = None, None
    for i in range(0, 10):
        d = today - dt.timedelta(days=i)
        if d.weekday() >= 5: continue
        b = get(fo_url(d))
        if b: session, fo_bytes = d, b; break
    if not session:
        sys.exit("ERROR: could not download any F&O bhavcopy in the last 10 days (NSE may be blocking this runner)")
    log("last completed F&O session:", session)
    fo = read_zip_csv(fo_bytes)
    for c in ("TckrSymb", "FinInstrmTp", "OptnTp"):
        if c in fo: fo[c] = fo[c].astype(str).str.strip()
    hist = update_history(session)

    ban = []
    b = get(f"{BASE}/fo/fo_secban.csv")
    if b:
        ban = [x.strip() for x in b.decode("utf-8", "ignore").splitlines() if x.strip() and not x.lower().startswith("securities")]
        ban = [x.split(",")[-1].strip() if "," in x else x for x in ban]
    result = {"generated_at_ist": dt.datetime.now(IST).isoformat(timespec="minutes"),
              "data_session": str(session), "source": "NSE end-of-day bhavcopy files (public)",
              "fo_ban_list_today_raw": ban, "stocks": {}}
    for s in SYMBOLS:
        try:
            tech, fut = technicals(hist[s]), fo_for(s, fo, session)
        except Exception as e:                                     # never let one stock kill the run
            tech, fut = {"error": str(e)}, {"error": str(e)}
        result["stocks"][s] = {"in_fo_ban": any(s == x for x in ban), "technicals": tech, "fno": fut}
    os.makedirs("data", exist_ok=True)
    json.dump(result, open("data/latest.json", "w"), indent=1)
    open("data/latest.md", "w").write(to_md(result))
    log("done")

def to_md(R):
    L = [f"# F&O snapshot — data for session {R['data_session']} (generated {R['generated_at_ist']} IST)",
         f"Source: {R['source']}. F&O ban list raw: {R['fo_ban_list_today_raw']}", ""]
    for s, v in R["stocks"].items():
        t, f = v["technicals"], v["fno"]
        L.append(f"## {s}" + ("  **[IN F&O BAN]**" if v["in_fo_ban"] else ""))
        if "error" in t or "error" in f:
            L.append(f"- ERROR tech={t.get('error')} fno={f.get('error')}\n"); continue
        L.append(f"- Last3: " + " | ".join(f"{x['date']} O{x['o']} H{x['h']} L{x['l']} C{x['c']} V{x['v']}" for x in t["last3"]))
        L.append(f"- Vol vs 20d avg: {t['vol_ratio']}x | SMA20/50/200: {t['sma20']}/{t['sma50']}/{t['sma200']} {t['dma200_note'] or ''}")
        L.append(f"- RSI14 {t['rsi14']} (prev {t['rsi14_prev']}) | MACD {t['macd']} sig {t['macd_signal']} hist {t['macd_hist']} (prev {t['macd_hist_prev']}) | Supertrend {t['supertrend']} {t['supertrend_dir']} | ATR14 {t['atr14']}")
        L.append(f"- Pivots {t['pivots']} | 3-day high/low {t['high3']}/{t['low3']} {t['crossover_20_50'] or ''}")
        L.append("- Candles: " + "; ".join(f"{c['date']}: {', '.join(c['flags']) or '-'}" for c in t["candles"]))
        L.append(f"- Futures {f['near_expiry']} ({f['calendar_days_to_expiry']}d): close {f['fut_close']} ({f['fut_price_change_pct']}%), OI {f['oi']:,} ({f['oi_change']:+,}, {f['oi_change_pct']}%) => {f['buildup']}; premium {f['premium_vs_spot']} ({f['premium_pct']}%); lot {f['lot_size']}; rollover {f.get('rollover_pct')}%")
        o = f["options"]
        if "error" in o: L.append(f"- Options: {o['error']}\n"); continue
        L.append(f"- Options {o['expiry']}: PCR(OI) {o['pcr_oi']} | max pain {o['max_pain']} | call OI {o['call_oi_total']:,} put OI {o['put_oi_total']:,}")
        L.append("  - Top call OI (resistance): " + ", ".join(f"{x['strike']} ({x['oi']:,}, {x['oi_change']:+,})" for x in o["top_call_oi"]))
        L.append("  - Top put OI (support): " + ", ".join(f"{x['strike']} ({x['oi']:,}, {x['oi_change']:+,})" for x in o["top_put_oi"]))
        L.append("  - Fresh call writing: " + ", ".join(f"{x['strike']} ({x['oi_change']:+,})" for x in o["top_call_oi_added"]) +
                 " | Fresh put writing: " + ", ".join(f"{x['strike']} ({x['oi_change']:+,})" for x in o["top_put_oi_added"]))
        L.append("")
    return "\n".join(L)

if __name__ == "__main__":
    main()
