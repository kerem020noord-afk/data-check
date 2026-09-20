"""
Gold (XAUUSD) analyse-script.

Haalt zowel de actuele prijs als 90 dagen historische dagdata op van dezelfde
bron: Kraken's PAXG/USD-markt. PAXG is een token dat 1-op-1 inwisselbaar is
voor 1 troy ounce fysiek goud en volgt daardoor de spotprijs van goud veel
directer dan COMEX-futures (die een prijsverschil van tientallen dollars met
spot kunnen hebben door contango). Geen API-key nodig, geen relevante
rate-limit voor een check elke paar minuten. Kraken is bewust gekozen boven
Binance: Binance's publieke API blokkeert verzoeken vanuit de VS (HTTP 451),
en GitHub Actions-runners draaien doorgaans in de VS — Kraken heeft die
blokkade niet. Berekent SMA20, SMA50, RSI14 en steun/weerstand-niveaus,
detecteert een regelgebaseerd buy/sell-signaal en stuurt daarbij optioneel
een pushmelding via ntfy.sh.

Twee manieren om te draaien:
  - python gold_analysis.py         -> blijft continu draaien (elke
    CHECK_INTERVAL_MINUTES minuten), tot je Ctrl+C indrukt. Voor lokaal gebruik.
  - python gold_analysis.py --once  -> voert precies één check uit en stopt.
    Voor gebruik via een cron-achtige scheduler zoals GitHub Actions, waar elke
    run een nieuw, stateless proces is.

De ntfy-topic-naam kan via de omgevingsvariabele NTFY_TOPIC worden aangeleverd
(bijv. als GitHub Actions secret) en valt anders terug op de waarde hieronder.
"""

import os
import sys
import time
import base64
import argparse
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

# pandas en matplotlib zijn hier gepind op versies die niet worden geblokkeerd
# door Windows Smart App Control / applicatiebeheerbeleid (recente pip-wheels
# van pandas 3.x en de nieuwste matplotlib worden op sommige Windows 11-systemen
# geweigerd omdat ze nog geen "reputatie" hebben opgebouwd).
REQUIRED_PACKAGES = {
    "requests": "requests",
    "pandas": "pandas==2.2.3",
    "numpy": "numpy",
    "matplotlib": "matplotlib==3.8.4",
}


def ensure_packages():
    missing = []
    for module, pip_spec in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_spec)
    if missing:
        print(f"Ontbrekende packages installeren: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_packages()

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================= CONFIG =============================
# Alle tijdstempels in meldingen/logs gebruiken expliciet deze tijdzone i.p.v.
# de systeemklok. Nodig omdat GitHub Actions-runners op UTC draaien: zonder
# dit zou een melding "20:44" tonen terwijl het lokaal (Nederland) 22:44 is,
# wat de vermelde prijs ten onrechte "fout" doet lijken bij het terugkijken.
LOCAL_TZ = ZoneInfo("Europe/Amsterdam")

CHECK_INTERVAL_MINUTES = 5  # hoe vaak de analyse opnieuw draait
HEARTBEAT_INTERVAL_MINUTES = 20  # hoe vaak er een heartbeat-melding gaat, ook zonder setup (4 checks)
HEARTBEAT_EVERY_N_CHECKS = max(1, round(HEARTBEAT_INTERVAL_MINUTES / CHECK_INTERVAL_MINUTES))

# NTFY_TOPIC komt bij voorkeur uit de omgevingsvariabele NTFY_TOPIC (zo kan hij
# als GitHub Actions secret worden aangeleverd, los van de code). De waarde
# hieronder is alleen de fallback voor lokaal draaien zonder die variabele.
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "kerim-goud-signalen-1x61")

# Eén bron voor zowel de live prijs als de historische candles, zodat er
# nooit een spot/futures-mismatch binnen één check kan ontstaan. Kraken i.p.v.
# Binance omdat Binance vanuit de VS (o.a. GitHub Actions-runners) met
# HTTP 451 wordt geblokkeerd.
SPOT_PAIR = "PAXGUSD"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
HISTORY_DAYS = 100  # zonder 'since' geeft Kraken tot 720 candles (~2 jaar) terug
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

SMA_SHORT = 20
SMA_LONG = 50
RSI_PERIOD = 14

EXTREMA_WINDOW = 5        # dagen links/rechts om een lokale high/low te bepalen
LEVEL_MERGE_PCT = 0.005   # niveaus binnen 0.5% van elkaar worden samengevoegd
PROXIMITY_PCT = 0.003     # 0.3% afstand tot een niveau voor een signaal
SL_BUFFER_PCT = 0.002     # 0.2% buffer voorbij het niveau voor de stop loss
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Prijsactie-detectie (naast RSI), zodat sterke bewegingen ook zonder RSI-
# extreem een setup kunnen triggeren:
MOMENTUM_LOOKBACK = 20        # aantal candles voor de gemiddelde range-baseline
MOMENTUM_RANGE_MULTIPLIER = 1.8  # candle-range moet dit x groter zijn dan het gemiddelde
REJECTION_PROXIMITY_PCT = 0.005  # hoe dicht high/low een niveau moet raken (0,5%)
REJECTION_WICK_RATIO = 0.55      # schaduw moet minstens dit deel van de candle-range zijn
TREND_CONFIRM_CANDLES = 3        # opeenvolgende hogere/lagere toppen+bodems voor trendbevestiging

# Vroege-waarschuwing-detectie: apart van en ruimer dan de echte GOUD SIGNAAL-
# drempels hierboven. Bedoeld om eerder te alarmeren op basis van dezelfde,
# nu al beschikbare data — geen voorspelling van toekomstige prijs.
EARLY_WARNING_RSI_BUFFER = 5          # RSI binnen deze marge van 35/65 telt als 'nadert'
EARLY_WARNING_LEVEL_PROXIMITY_PCT = 0.01  # 1% i.p.v. de 0,3% van een echt signaal
SQUEEZE_LOOKBACK_SHORT = 5            # candles voor de korte-termijn-range
SQUEEZE_LOOKBACK_LONG = 20            # candles voor de normale-range-baseline
SQUEEZE_RATIO_THRESHOLD = 0.6         # korte range moet dit x kleiner zijn dan normaal
WARNING_COOLDOWN_MINUTES = 60         # min. tijd tussen twee waarschuwingen met dezelfde inhoud

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "gold_analysis.csv")
CHART_PATH = os.path.join(SCRIPT_DIR, "gold_chart.png")
# Onthoudt het laatst afgehandelde heartbeat-blok. Dit bestand wordt in de
# GitHub Actions-workflow teruggecommit naar de repo, zodat de --once modus
# ook zonder gedeeld procesgeheugen precies 1x per HEARTBEAT_INTERVAL_MINUTES
# een heartbeat stuurt, hoe lang een cron-tik ook vertraagd is.
HEARTBEAT_STATE_PATH = os.path.join(SCRIPT_DIR, "heartbeat_state.txt")
# Onthoudt de laatst verstuurde vroege-waarschuwing (inhoud + tijdstip), zodat
# een aanhoudende conditie niet elke paar minuten opnieuw meldt. Wordt net als
# heartbeat_state.txt door de GitHub Actions-workflow teruggecommit.
WARNING_STATE_PATH = os.path.join(SCRIPT_DIR, "warning_state.txt")
DISCLAIMER = (
    "Dit is een op regels gebaseerde indicatie, geen voorspelling. "
    "Wacht altijd op bevestiging op de grafiek zelf voordat je handelt."
)
EARLY_WARNING_DISCLAIMER = (
    "Vroege waarschuwing, geen signaal: condities naderen een drempel of "
    "volatiliteit is samengeperst. Dit voorspelt geen richting of timing — "
    "puur een 'let op' op basis van dezelfde regels."
)
# ====================================================================

SESSION = requests.Session()
SESSION.headers.update(REQUEST_HEADERS)


def now_local():
    return datetime.now(LOCAL_TZ)


def _get_with_retry(url, timeout=10, retries=1, backoff=3, **kwargs):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = SESSION.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff)
    raise last_exc


def _fetch_ohlc():
    since = int(time.time()) - HISTORY_DAYS * 86400
    params = {"pair": SPOT_PAIR, "interval": 1440, "since": since}
    resp = _get_with_retry(KRAKEN_OHLC_URL, params=params, timeout=15)
    payload = resp.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken OHLC-endpoint gaf een fout terug: {payload['error']}")
    return payload["result"][SPOT_PAIR]


def fetch_current_price():
    # Zelfde bron (Kraken) als fetch_history(), zodat prijs en candles nooit
    # uit twee verschillende markten komen binnen één check. Als de losse
    # ticker-aanroep faalt, valt dit terug op de laatste OHLC-close van
    # dezelfde bron — geen kruisbestuiving met een andere markt/aanbieder.
    try:
        resp = _get_with_retry(KRAKEN_TICKER_URL, params={"pair": SPOT_PAIR}, timeout=10)
        payload = resp.json()
        if payload.get("error"):
            raise RuntimeError(f"Kraken Ticker-endpoint gaf een fout terug: {payload['error']}")
        return float(payload["result"][SPOT_PAIR]["c"][0])  # c[0] = laatste handelsprijs
    except Exception as primary_error:
        try:
            ohlc = _fetch_ohlc()
            return float(ohlc[-1][4])  # index 4 = close
        except Exception as fallback_error:
            raise RuntimeError(
                "Kan de actuele XAUUSD-prijs (via PAXG/USD op Kraken) niet ophalen: "
                f"ticker-endpoint faalde ({primary_error}) en de OHLC-backup faalde ook ({fallback_error})"
            )


def fetch_history():
    try:
        ohlc = _fetch_ohlc()
        df = pd.DataFrame(
            ohlc,
            columns=["Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Trades"],
        )
        df["Date"] = pd.to_datetime(df["Time"], unit="s").dt.normalize()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = df[col].astype(float)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Close"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("lege historische dataset ontvangen")
        return df
    except Exception as e:
        raise RuntimeError(f"Kan historische XAUUSD-data (via PAXG/USD op Kraken) niet ophalen: {e}")


def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def compute_indicators(df):
    df["SMA20"] = df["Close"].rolling(window=SMA_SHORT, min_periods=1).mean()
    df["SMA50"] = df["Close"].rolling(window=SMA_LONG, min_periods=1).mean()
    df["RSI14"] = compute_rsi(df["Close"])
    return df


def find_local_extrema(df, window=EXTREMA_WINDOW):
    highs, lows = [], []
    n = len(df)
    for i in range(window, n - window):
        seg_high = df["High"].iloc[i - window : i + window + 1]
        seg_low = df["Low"].iloc[i - window : i + window + 1]
        if df["High"].iloc[i] == seg_high.max():
            highs.append((df["Date"].iloc[i], float(df["High"].iloc[i])))
        if df["Low"].iloc[i] == seg_low.min():
            lows.append((df["Date"].iloc[i], float(df["Low"].iloc[i])))
    df["Local_High"] = df["Date"].isin([d for d, _ in highs])
    df["Local_Low"] = df["Date"].isin([d for d, _ in lows])
    return [v for _, v in highs], [v for _, v in lows]


def merge_levels(levels, pct=LEVEL_MERGE_PCT):
    if not levels:
        return []
    levels = sorted(set(levels))
    merged = [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl - merged[-1]) / merged[-1] <= pct:
            merged[-1] = (merged[-1] + lvl) / 2
        else:
            merged.append(lvl)
    return merged


def nearest_level(levels, price, direction):
    if not levels:
        return None
    if direction == "below":
        candidates = [l for l in levels if l <= price]
        return max(candidates) if candidates else None
    candidates = [l for l in levels if l >= price]
    return min(candidates) if candidates else None


def build_buy_signal(entry, sl_ref, resistance_levels, reasons):
    sl = sl_ref * (1 - SL_BUFFER_PCT)
    targets = sorted(r for r in resistance_levels if r > entry)
    tp1 = targets[0] if targets else None
    tp2 = targets[1] if len(targets) > 1 else None
    risk = entry - sl
    rr1 = (tp1 - entry) / risk if tp1 is not None and risk > 0 else None
    rr2 = (tp2 - entry) / risk if tp2 is not None and risk > 0 else None
    return {
        "direction": "BUY",
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr1": rr1,
        "rr2": rr2,
        "level": sl_ref,
        "reasons": reasons,
    }


def build_sell_signal(entry, sl_ref, support_levels, reasons):
    sl = sl_ref * (1 + SL_BUFFER_PCT)
    targets = sorted((s for s in support_levels if s < entry), reverse=True)
    tp1 = targets[0] if targets else None
    tp2 = targets[1] if len(targets) > 1 else None
    risk = sl - entry
    rr1 = (entry - tp1) / risk if tp1 is not None and risk > 0 else None
    rr2 = (entry - tp2) / risk if tp2 is not None and risk > 0 else None
    return {
        "direction": "SELL",
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr1": rr1,
        "rr2": rr2,
        "level": sl_ref,
        "reasons": reasons,
    }


def detect_momentum_candle(df, lookback=MOMENTUM_LOOKBACK, multiplier=MOMENTUM_RANGE_MULTIPLIER):
    # Een candle-range die significant groter is dan het gemiddelde van de
    # laatste `lookback` candles duidt op een sterke stoot in een richting,
    # ook als de RSI (die alleen naar close-to-close kijkt) neutraal blijft.
    if len(df) < lookback + 1:
        return None
    ranges = df["High"] - df["Low"]
    avg_range = ranges.iloc[-(lookback + 1):-1].mean()
    last_range = ranges.iloc[-1]
    if avg_range <= 0:
        return None
    ratio = last_range / avg_range
    if ratio < multiplier:
        return None
    last = df.iloc[-1]
    direction = "BUY" if last["Close"] >= last["Open"] else "SELL"
    return {
        "direction": direction,
        "reason": f"sterke momentum-candle: range {ratio:.1f}x groter dan gemiddelde van laatste {lookback} candles",
    }


def detect_rejection_candle(df, nearest_support, nearest_resistance,
                             proximity_pct=REJECTION_PROXIMITY_PCT, wick_ratio=REJECTION_WICK_RATIO):
    # Een candle die een niveau test (high/low raakt het) maar terugvalt met
    # een lange schaduw in de richting van de afwijzing, is een klassiek
    # prijsactie-signaal dat RSI-drempels volledig kunnen missen.
    last = df.iloc[-1]
    candle_range = last["High"] - last["Low"]
    if candle_range <= 0:
        return None
    body_top = max(last["Open"], last["Close"])
    body_bottom = min(last["Open"], last["Close"])
    upper_wick = last["High"] - body_top
    lower_wick = body_bottom - last["Low"]

    if nearest_resistance is not None:
        dist = abs(last["High"] - nearest_resistance) / nearest_resistance
        if dist <= proximity_pct and (upper_wick / candle_range) >= wick_ratio:
            return {
                "direction": "SELL",
                "reason": f"afwijzing op weerstand {nearest_resistance:.2f}: bovenschaduw is "
                          f"{upper_wick / candle_range:.0%} van de candle-range",
            }
    if nearest_support is not None:
        dist = abs(last["Low"] - nearest_support) / nearest_support
        if dist <= proximity_pct and (lower_wick / candle_range) >= wick_ratio:
            return {
                "direction": "BUY",
                "reason": f"afwijzing op steun {nearest_support:.2f}: onderschaduw is "
                          f"{lower_wick / candle_range:.0%} van de candle-range",
            }
    return None


def detect_trend_confirmation(df, n=TREND_CONFIRM_CANDLES):
    # Opeenvolgende hogere toppen+bodems (of lagere toppen+bodems) bevestigen
    # een trend over meerdere candles, in plaats van op één losse candle te
    # steunen.
    if len(df) < n + 1:
        return None
    recent = df.iloc[-(n + 1):]
    highs = recent["High"].values
    lows = recent["Low"].values
    higher_highs = all(highs[i] > highs[i - 1] for i in range(1, len(highs)))
    higher_lows = all(lows[i] > lows[i - 1] for i in range(1, len(lows)))
    lower_highs = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))
    lower_lows = all(lows[i] < lows[i - 1] for i in range(1, len(lows)))
    if higher_highs and higher_lows:
        return {"direction": "BUY", "reason": f"{n} opeenvolgende hogere toppen én bodems (opwaartse trend)"}
    if lower_highs and lower_lows:
        return {"direction": "SELL", "reason": f"{n} opeenvolgende lagere toppen én bodems (neerwaartse trend)"}
    return None


def detect_rsi_approaching(rsi, df, buffer=EARLY_WARNING_RSI_BUFFER):
    # RSI binnen `buffer` punten van 35/65 én nog bewegend in die richting
    # (niet net omgekeerd) — een vroeg 'let op', geen bevestigd signaal.
    if rsi is None or len(df) < 4:
        return None
    prev_rsi = df["RSI14"].iloc[-4]
    if pd.isna(prev_rsi):
        return None
    if RSI_OVERSOLD <= rsi < RSI_OVERSOLD + buffer and rsi < prev_rsi:
        return {
            "key": "rsi_approach_buy",
            "direction": "BUY",
            "reason": f"RSI nadert oversold: {rsi:.1f} en dalend (was {prev_rsi:.1f})",
        }
    if RSI_OVERBOUGHT - buffer < rsi <= RSI_OVERBOUGHT and rsi > prev_rsi:
        return {
            "key": "rsi_approach_sell",
            "direction": "SELL",
            "reason": f"RSI nadert overbought: {rsi:.1f} en stijgend (was {prev_rsi:.1f})",
        }
    return None


def detect_level_approaching(current_price, nearest_support, nearest_resistance,
                              proximity_pct=EARLY_WARNING_LEVEL_PROXIMITY_PCT):
    # Prijs binnen een ruimere marge (1%) van een niveau, maar nog niet binnen
    # de strakke 0,3% die een echt signaal vereist.
    if nearest_support is not None:
        dist = abs(current_price - nearest_support) / nearest_support
        if PROXIMITY_PCT < dist <= proximity_pct:
            return {
                "key": "level_approach_buy",
                "direction": "BUY",
                "reason": f"prijs nadert steun {nearest_support:.2f} (nu {dist:.2%} weg)",
            }
    if nearest_resistance is not None:
        dist = abs(current_price - nearest_resistance) / nearest_resistance
        if PROXIMITY_PCT < dist <= proximity_pct:
            return {
                "key": "level_approach_sell",
                "direction": "SELL",
                "reason": f"prijs nadert weerstand {nearest_resistance:.2f} (nu {dist:.2%} weg)",
            }
    return None


def detect_volatility_squeeze(df, short=SQUEEZE_LOOKBACK_SHORT, long=SQUEEZE_LOOKBACK_LONG,
                               threshold=SQUEEZE_RATIO_THRESHOLD):
    # Ongebruikelijk kleine ranges t.o.v. de normale baseline duiden vaak op
    # een 'samenpersing' vlak vóór een uitbraak — het 'voorspellende' deel:
    # verhoogde kans op een grote beweging, zonder richting.
    if len(df) < long + 1:
        return None
    ranges = df["High"] - df["Low"]
    avg_short = ranges.iloc[-short:].mean()
    avg_long = ranges.iloc[-(long + 1):-1].mean()
    if avg_long <= 0:
        return None
    ratio = avg_short / avg_long
    if ratio <= threshold:
        return {
            "key": "squeeze",
            "direction": None,
            "reason": f"volatiliteit samengeperst: laatste {short} candles gemiddeld {ratio:.1f}x "
                      f"de normale range van {long} candles — verhoogde kans op een uitbraak "
                      f"(richting onbekend)",
        }
    return None


def detect_early_warnings(current_price, rsi, df, nearest_support, nearest_resistance):
    warnings = []
    for w in (
        detect_rsi_approaching(rsi, df),
        detect_level_approaching(current_price, nearest_support, nearest_resistance),
        detect_volatility_squeeze(df),
    ):
        if w is not None:
            warnings.append(w)
    return warnings


def warning_direction(warnings):
    # Alleen een richting teruggeven als de actieve waarschuwingen het EENS
    # zijn (squeeze heeft geen richting en telt niet mee). Bij tegenstrijdige
    # signalen (bv. RSI nadert oversold, prijs nadert weerstand) geven we
    # bewust geen richting/zone — dat zou een schijnzekerheid suggereren.
    directions = {w["direction"] for w in warnings if w["direction"] is not None}
    return directions.pop() if len(directions) == 1 else None


def build_warning_zone(direction, current_price, nearest_support, nearest_resistance,
                        support_levels, resistance_levels):
    if direction == "BUY":
        sl_ref = nearest_support if nearest_support is not None else current_price
        zone_low, zone_high = sl_ref, current_price
        sl = sl_ref * (1 - SL_BUFFER_PCT)
        targets = sorted(r for r in resistance_levels if r > current_price)
    else:  # SELL
        sl_ref = nearest_resistance if nearest_resistance is not None else current_price
        zone_low, zone_high = current_price, sl_ref
        sl = sl_ref * (1 + SL_BUFFER_PCT)
        targets = sorted((s for s in support_levels if s < current_price), reverse=True)
    return {
        "zone_low": zone_low,
        "zone_high": zone_high,
        "sl": sl,
        "tp1": targets[0] if targets else None,
        "tp2": targets[1] if len(targets) > 1 else None,
    }


def detect_signal(current_price, rsi, df, support_levels, resistance_levels,
                   nearest_support, nearest_resistance):
    reasons_buy, reasons_sell = [], []

    # 1) Bestaande logica: prijs dicht bij niveau + RSI-extreem
    if nearest_support is not None and rsi is not None:
        dist = abs(current_price - nearest_support) / nearest_support
        if dist <= PROXIMITY_PCT and rsi < RSI_OVERSOLD:
            reasons_buy.append(f"prijs binnen {PROXIMITY_PCT:.1%} van steun {nearest_support:.2f} en RSI oversold ({rsi:.1f})")
    if nearest_resistance is not None and rsi is not None:
        dist = abs(current_price - nearest_resistance) / nearest_resistance
        if dist <= PROXIMITY_PCT and rsi > RSI_OVERBOUGHT:
            reasons_sell.append(f"prijs binnen {PROXIMITY_PCT:.1%} van weerstand {nearest_resistance:.2f} en RSI overbought ({rsi:.1f})")

    # 2) Prijsactie: momentum-candle, afwijzingscandle, trendbevestiging
    for detector_result in (
        detect_momentum_candle(df),
        detect_rejection_candle(df, nearest_support, nearest_resistance),
        detect_trend_confirmation(df),
    ):
        if detector_result is None:
            continue
        target = reasons_buy if detector_result["direction"] == "BUY" else reasons_sell
        target.append(detector_result["reason"])

    # Tegenstrijdige redenen (zowel buy- als sell-argumenten) -> te onduidelijk, geen signaal
    if reasons_buy and not reasons_sell:
        sl_ref = nearest_support if nearest_support is not None else float(df["Low"].iloc[-1])
        return build_buy_signal(current_price, sl_ref, resistance_levels, reasons_buy)
    if reasons_sell and not reasons_buy:
        sl_ref = nearest_resistance if nearest_resistance is not None else float(df["High"].iloc[-1])
        return build_sell_signal(current_price, sl_ref, support_levels, reasons_sell)
    return None


def format_nl(value):
    return f"{value:.2f}".replace(".", ",")


def encode_header_value(value):
    # HTTP-headers moeten latin-1-veilig zijn; niet-ASCII tekens (zoals —) worden
    # daarom als RFC 2047-encoded word verstuurd, wat ntfy automatisch decodeert.
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        b64 = base64.b64encode(value.encode("utf-8")).decode("ascii")
        return f"=?UTF-8?B?{b64}?="


def send_ntfy(title, message):
    # Elke ntfy-poging krijgt een expliciete, niet te missen logregel — succes
    # of mislukt — zodat een falende melding nooit stilletjes verdwijnt in de
    # GitHub Actions-logs.
    if not NTFY_TOPIC or "wijzig-dit" in NTFY_TOPIC:
        print(f"[NTFY] MISLUKT ({title}): NTFY_TOPIC is nog niet aangepast naar een eigen "
              f"unieke naam — melding overgeslagen.")
        return False
    try:
        resp = SESSION.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": encode_header_value(title)},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[NTFY] OK ({title}): melding verstuurd naar ntfy.sh/{NTFY_TOPIC}")
        return True
    except Exception as e:
        print(f"[NTFY] MISLUKT ({title}): kon niet versturen naar ntfy.sh/{NTFY_TOPIC}: {e}")
        return False


def send_signal_notification(signal):
    lines = [
        f"{signal['direction']} signaal XAUUSD",
        "Waarom: " + "; ".join(signal["reasons"]),
        f"Entry: {signal['entry']:.2f}",
        f"Stop Loss: {signal['sl']:.2f}",
    ]
    if signal["tp1"] is not None:
        rr_txt = f" (R:R {signal['rr1']:.2f})" if signal["rr1"] is not None else ""
        lines.append(f"TP1: {signal['tp1']:.2f}{rr_txt}")
    if signal["tp2"] is not None:
        rr_txt = f" (R:R {signal['rr2']:.2f})" if signal["rr2"] is not None else ""
        lines.append(f"TP2: {signal['tp2']:.2f}{rr_txt}")
    message = "\n".join(lines)
    send_ntfy("GOUD SIGNAAL", message)


def send_heartbeat_notification(now, current_price, rsi, signal):
    time_txt = now.strftime("%H:%M")
    rsi_txt = format_nl(rsi) if rsi is not None else "n.v.t."
    status_txt = f"{signal['direction']}-setup actief" if signal is not None else "nog geen setup"
    message = f"{time_txt} — Prijs: {format_nl(current_price)}, RSI: {rsi_txt}, {status_txt}"
    send_ntfy("Goud check — actief", message)


def _read_warning_state():
    try:
        if os.path.exists(WARNING_STATE_PATH):
            with open(WARNING_STATE_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if "|" in content:
                ts_str, key = content.split("|", 1)
                return datetime.fromisoformat(ts_str), key
    except (OSError, ValueError) as e:
        print(f"Let op: kon waarschuwing-statusbestand niet lezen ({e}); "
              f"ga uit van 'nog geen eerdere waarschuwing'.")
    return None, None


def _mark_warning_sent(now, key):
    try:
        with open(WARNING_STATE_PATH, "w", encoding="utf-8") as f:
            f.write(f"{now.isoformat()}|{key}")
    except OSError as e:
        print(f"Let op: kon waarschuwing-statusbestand niet wegschrijven ({e}).")


def should_send_warning(now, warnings):
    # Stuur alleen als de combinatie van actieve waarschuwingen NIEUW is, of
    # als dezelfde combinatie al langer dan WARNING_COOLDOWN_MINUTES aanhoudt
    # — zo geen herhaalspam zolang een conditie blijft gelden, maar wel een
    # nieuwe melding zodra er iets verandert (bv. squeeze komt erbij).
    if not warnings:
        return False, None
    key = ",".join(sorted(w["key"] for w in warnings))
    last_time, last_key = _read_warning_state()
    if last_key == key and last_time is not None:
        try:
            elapsed_minutes = (now - last_time).total_seconds() / 60
            if elapsed_minutes < WARNING_COOLDOWN_MINUTES:
                return False, key
        except TypeError:
            pass  # oude/naive tijdstempel uit een eerdere versie; behandel als 'geen eerdere state'
    return True, key


def send_warning_notification(now, current_price, warnings, direction, zone):
    lines = [f"Prijs: {format_nl(current_price)}"]
    if direction is not None:
        lines.append(f"Mogelijke richting: {direction}")
    for w in warnings:
        lines.append(f"- {w['reason']}")
    if zone is not None:
        lines.append(f"Zone: {format_nl(zone['zone_low'])} - {format_nl(zone['zone_high'])}")
        lines.append(f"Stop Loss (indicatief): {format_nl(zone['sl'])}")
        if zone["tp1"] is not None:
            lines.append(f"TP1 (indicatief): {format_nl(zone['tp1'])}")
        if zone["tp2"] is not None:
            lines.append(f"TP2 (indicatief): {format_nl(zone['tp2'])}")
    lines.append(EARLY_WARNING_DISCLAIMER)
    message = "\n".join(lines)
    title = "GOUD VROEGE WAARSCHUWING" + (f" ({direction})" if direction else "")
    send_ntfy(title, message)


def make_chart(df, support_levels, resistance_levels, current_price, signal):
    fig, ax = plt.subplots(figsize=(13, 7))
    try:
        ax.plot(df["Date"], df["Close"], label="Close", color="#c9a227", linewidth=1.5)
        ax.plot(df["Date"], df["SMA20"], label=f"SMA{SMA_SHORT}", color="#1f77b4", linewidth=1)
        ax.plot(df["Date"], df["SMA50"], label=f"SMA{SMA_LONG}", color="#d62728", linewidth=1)

        for lvl in support_levels:
            ax.axhline(lvl, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
        for lvl in resistance_levels:
            ax.axhline(lvl, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

        last_date = df["Date"].iloc[-1]
        ax.scatter([last_date], [current_price], color="black", zorder=5,
                   label="Actuele prijs (live)")

        if signal is not None:
            color = "green" if signal["direction"] == "BUY" else "red"
            marker = "^" if signal["direction"] == "BUY" else "v"
            ax.scatter([last_date], [signal["entry"]], color=color, marker=marker,
                       s=200, zorder=6, label=f"{signal['direction']}-signaal")
            ax.annotate(
                signal["direction"],
                xy=(last_date, signal["entry"]),
                xytext=(10, 20 if signal["direction"] == "BUY" else -25),
                textcoords="offset points",
                fontsize=11, fontweight="bold", color=color,
            )

        ax.set_title("XAUUSD (goud) — prijs, SMA20/SMA50 en steun/weerstand")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Prijs (USD)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(CHART_PATH, dpi=150)
    finally:
        plt.close(fig)


def run_once(send_heartbeat=False):
    now = now_local()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    current_price = fetch_current_price()
    df = fetch_history()

    df = compute_indicators(df)
    raw_highs, raw_lows = find_local_extrema(df)
    resistance_levels = merge_levels(raw_highs)
    support_levels = merge_levels(raw_lows)

    df.to_csv(CSV_PATH, index=False)

    nearest_support = nearest_level(support_levels, current_price, "below")
    nearest_resistance = nearest_level(resistance_levels, current_price, "above")
    current_rsi = df["RSI14"].iloc[-1]
    current_rsi = float(current_rsi) if pd.notna(current_rsi) else None

    signal = detect_signal(
        current_price, current_rsi, df, support_levels, resistance_levels,
        nearest_support, nearest_resistance,
    )
    warnings = detect_early_warnings(
        current_price, current_rsi, df, nearest_support, nearest_resistance,
    )

    make_chart(df, support_levels, resistance_levels, current_price, signal)

    print("=" * 60)
    print(f"GOUD (XAUUSD) ANALYSE — laatste check: {timestamp}")
    print("=" * 60)
    print(f"Actuele prijs        : {current_price:.2f} USD")
    print(f"RSI (14)             : {current_rsi:.2f}" if current_rsi is not None else "RSI (14)             : n.v.t.")
    print(f"Dichtstbijzijnde steun    : {nearest_support:.2f}" if nearest_support else "Dichtstbijzijnde steun    : geen gevonden")
    print(f"Dichtstbijzijnde weerstand: {nearest_resistance:.2f}" if nearest_resistance else "Dichtstbijzijnde weerstand: geen gevonden")
    print(f"Data opgeslagen als   : {CSV_PATH}")
    print(f"Grafiek opgeslagen als: {CHART_PATH}")
    print("-" * 60)

    if signal is not None:
        print(f">>> Mogelijk {signal['direction']}-SIGNAAL gedetecteerd <<<")
        print("Waarom:")
        for reason in signal["reasons"]:
            print(f"  - {reason}")
        print(f"Entry     : {signal['entry']:.2f}")
        print(f"Stop Loss : {signal['sl']:.2f}")
        print(f"TP1       : {signal['tp1']:.2f}" if signal["tp1"] is not None else "TP1       : geen volgend niveau gevonden")
        if signal["rr1"] is not None:
            print(f"  R:R naar TP1 : {signal['rr1']:.2f}")
        print(f"TP2       : {signal['tp2']:.2f}" if signal["tp2"] is not None else "TP2       : geen volgend niveau gevonden")
        if signal["rr2"] is not None:
            print(f"  R:R naar TP2 : {signal['rr2']:.2f}")
        send_signal_notification(signal)
    else:
        print("Geen duidelijke setup op dit moment.")

    print("-" * 60)
    should_warn, warning_key = should_send_warning(now, warnings)
    if warnings:
        direction = warning_direction(warnings)
        zone = None
        if direction is not None:
            zone = build_warning_zone(
                direction, current_price, nearest_support, nearest_resistance,
                support_levels, resistance_levels,
            )
        print(f"Vroege waarschuwing(en) actief ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w['reason']}")
        if direction is not None:
            print(f"Mogelijke richting: {direction}")
            print(f"  Zone      : {zone['zone_low']:.2f} - {zone['zone_high']:.2f}")
            print(f"  Stop Loss : {zone['sl']:.2f} (indicatief)")
            print(f"  TP1       : {zone['tp1']:.2f} (indicatief)" if zone["tp1"] is not None else "  TP1       : geen volgend niveau gevonden")
            print(f"  TP2       : {zone['tp2']:.2f} (indicatief)" if zone["tp2"] is not None else "  TP2       : geen volgend niveau gevonden")
        else:
            print("Mogelijke richting: geen eenduidige richting (tegenstrijdig of alleen squeeze) — geen zone getoond.")
        if should_warn:
            send_warning_notification(now, current_price, warnings, direction, zone)
            _mark_warning_sent(now, warning_key)
        else:
            print("Geen nieuwe waarschuwingsmelding (zelfde conditie recent al gestuurd, cooldown actief).")
    else:
        print("Geen vroege waarschuwingen op dit moment.")

    print("-" * 60)
    if send_heartbeat:
        print(f"Heartbeat-ronde (elke {HEARTBEAT_INTERVAL_MINUTES} min): heartbeat-melding wordt verstuurd.")
        send_heartbeat_notification(now, current_price, current_rsi, signal)
    else:
        print(f"Geen heartbeat-ronde deze keer (heartbeat gaat elke {HEARTBEAT_INTERVAL_MINUTES} min).")

    print("-" * 60)
    print(DISCLAIMER)
    print("=" * 60)


def run_forever():
    print(f"Doorlopende XAUUSD-analyse gestart — check elke {CHECK_INTERVAL_MINUTES} minuten, "
          f"heartbeat elke {HEARTBEAT_EVERY_N_CHECKS} checks (~{HEARTBEAT_INTERVAL_MINUTES} min).")
    print("Druk op Ctrl+C om te stoppen.")
    tick_count = 0
    try:
        while True:
            tick_count += 1  # telt elke ronde, ook bij een mislukte check, zodat het uur-ritme niet verspringt
            is_heartbeat_tick = (tick_count == 1) or (tick_count % HEARTBEAT_EVERY_N_CHECKS == 0)
            timestamp = now_local().strftime("%Y-%m-%d %H:%M:%S %Z")
            try:
                run_once(send_heartbeat=is_heartbeat_tick)
            except Exception as e:
                print_check_failed(timestamp, e)
                if is_heartbeat_tick:
                    print(f"[{timestamp}] Heartbeat overgeslagen deze ronde wegens fout "
                          f"(volgende heartbeat over {HEARTBEAT_EVERY_N_CHECKS} checks).")
                print(f"[{timestamp}] Sla deze ronde over, volgende poging over {CHECK_INTERVAL_MINUTES} minuten.")
            time.sleep(CHECK_INTERVAL_MINUTES * 60)
    except KeyboardInterrupt:
        print("\nScript gestopt door gebruiker (Ctrl+C). Tot de volgende keer!")
        sys.exit(0)


def _heartbeat_block_start(now):
    block_minute = (now.minute // HEARTBEAT_INTERVAL_MINUTES) * HEARTBEAT_INTERVAL_MINUTES
    block_start = now.replace(minute=block_minute, second=0, microsecond=0)
    return block_start.strftime("%Y-%m-%dT%H:%M")


def _read_last_heartbeat_block():
    try:
        if os.path.exists(HEARTBEAT_STATE_PATH):
            with open(HEARTBEAT_STATE_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
    except OSError as e:
        print(f"Let op: kon heartbeat-statusbestand niet lezen ({e}); "
              f"ga uit van 'nog geen eerdere heartbeat'.")
    return None


def _mark_heartbeat_block_done(block):
    try:
        with open(HEARTBEAT_STATE_PATH, "w", encoding="utf-8") as f:
            f.write(block)
    except OSError as e:
        print(f"Let op: kon heartbeat-statusbestand niet wegschrijven ({e}).")


def print_check_failed(timestamp, error):
    # Een luide, niet te missen waarschuwing i.p.v. één stille printregel,
    # zodat een volledig mislukte check (bv. de bron onbereikbaar) meteen
    # opvalt in de GitHub Actions-logs, in plaats van te verdwijnen naast een
    # groen "Success"-vinkje.
    print("!" * 60)
    print(f"[{timestamp}] KRITIEKE FOUT: de volledige check is mislukt — geen prijs/candles "
          f"opgehaald, geen signaalcheck, geen ntfy-poging deze ronde.")
    print(f"[{timestamp}] Reden: {error}")
    print("!" * 60)


def run_once_stateless():
    # Voor gebruik in een scheduler (GitHub Actions, cron): elke aanroep is
    # een nieuw proces zonder in-memory geheugen van vorige runs, dus de
    # heartbeat-telling van run_forever() werkt hier niet. In plaats daarvan
    # wordt het statusbestand gebruikt — en pas bijgewerkt NA een geslaagde
    # check, zodat een falende run het heartbeat-ritme niet stilletjes laat
    # "doortikken" zonder dat er ooit echt een melding is verstuurd.
    now = now_local()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    current_block = _heartbeat_block_start(now)
    is_heartbeat = _read_last_heartbeat_block() != current_block
    try:
        run_once(send_heartbeat=is_heartbeat)
    except Exception as e:
        print_check_failed(timestamp, e)
        if is_heartbeat:
            print(f"[{timestamp}] Heartbeat NIET verstuurd en NIET als afgehandeld gemarkeerd "
                  f"— wordt bij de volgende geslaagde check opnieuw geprobeerd.")
        return

    if is_heartbeat:
        _mark_heartbeat_block_done(current_block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD (goud) analyse")
    parser.add_argument(
        "--once", action="store_true",
        help="Voer één check uit en stop, i.p.v. continu te blijven draaien "
             "(voor gebruik via een scheduler zoals GitHub Actions of cron).",
    )
    args = parser.parse_args()

    if args.once:
        run_once_stateless()
    else:
        run_forever()
