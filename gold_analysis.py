"""
Gold (XAUUSD) analyse-script.

Haalt zowel de actuele prijs als 90 dagen historische dagdata op van dezelfde
bron: Binance's PAXG/USDT-markt. PAXG is een token dat 1-op-1 inwisselbaar is
voor 1 troy ounce fysiek goud en volgt daardoor de spotprijs van goud veel
directer dan COMEX-futures (die een prijsverschil van tientallen dollars met
spot kunnen hebben door contango). Geen API-key nodig, geen relevante
rate-limit voor een check elke paar minuten. Berekent SMA20, SMA50, RSI14 en
steun/weerstand-niveaus, detecteert een regelgebaseerd buy/sell-signaal en
stuurt daarbij optioneel een pushmelding via ntfy.sh.

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
CHECK_INTERVAL_MINUTES = 5  # hoe vaak de analyse opnieuw draait
HEARTBEAT_INTERVAL_MINUTES = 20  # hoe vaak er een heartbeat-melding gaat, ook zonder setup (4 checks)
HEARTBEAT_EVERY_N_CHECKS = max(1, round(HEARTBEAT_INTERVAL_MINUTES / CHECK_INTERVAL_MINUTES))

# NTFY_TOPIC komt bij voorkeur uit de omgevingsvariabele NTFY_TOPIC (zo kan hij
# als GitHub Actions secret worden aangeleverd, los van de code). De waarde
# hieronder is alleen de fallback voor lokaal draaien zonder die variabele.
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "kerim-goud-signalen-1x61")

# Eén bron voor zowel de live prijs als de historische candles, zodat er
# nooit een spot/futures-mismatch binnen één check kan ontstaan.
SPOT_SYMBOL = "PAXGUSDT"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
HISTORY_PARAMS = {"symbol": SPOT_SYMBOL, "interval": "1d", "limit": 100}
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "gold_analysis.csv")
CHART_PATH = os.path.join(SCRIPT_DIR, "gold_chart.png")
# Onthoudt het laatst afgehandelde heartbeat-blok. Dit bestand wordt in de
# GitHub Actions-workflow teruggecommit naar de repo, zodat de --once modus
# ook zonder gedeeld procesgeheugen precies 1x per HEARTBEAT_INTERVAL_MINUTES
# een heartbeat stuurt, hoe lang een cron-tik ook vertraagd is.
HEARTBEAT_STATE_PATH = os.path.join(SCRIPT_DIR, "heartbeat_state.txt")
DISCLAIMER = (
    "Dit is een op regels gebaseerde indicatie, geen voorspelling. "
    "Wacht altijd op bevestiging op de grafiek zelf voordat je handelt."
)
# ====================================================================

SESSION = requests.Session()
SESSION.headers.update(REQUEST_HEADERS)


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


def _fetch_klines():
    resp = _get_with_retry(BINANCE_KLINES_URL, params=HISTORY_PARAMS, timeout=15)
    return resp.json()


def fetch_current_price():
    # Zelfde bron (Binance) als fetch_history(), zodat prijs en candles nooit
    # uit twee verschillende markten komen binnen één check. Als de losse
    # ticker-aanroep faalt, valt dit terug op de laatste klines-close van
    # dezelfde bron — geen kruisbestuiving met een andere markt/aanbieder.
    try:
        resp = _get_with_retry(BINANCE_TICKER_URL, params={"symbol": SPOT_SYMBOL}, timeout=10)
        return float(resp.json()["price"])
    except Exception as primary_error:
        try:
            klines = _fetch_klines()
            return float(klines[-1][4])  # index 4 = close
        except Exception as fallback_error:
            raise RuntimeError(
                "Kan de actuele XAUUSD-prijs (via PAXG/USDT op Binance) niet ophalen: "
                f"ticker-endpoint faalde ({primary_error}) en de klines-backup faalde ook ({fallback_error})"
            )


def fetch_history():
    try:
        klines = _fetch_klines()
        df = pd.DataFrame(
            klines,
            columns=[
                "OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime",
                "QuoteVolume", "Trades", "TakerBaseVolume", "TakerQuoteVolume", "Ignore",
            ],
        )
        df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms").dt.normalize()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = df[col].astype(float)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Close"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("lege historische dataset ontvangen")
        return df
    except Exception as e:
        raise RuntimeError(f"Kan historische XAUUSD-data (via PAXG/USDT op Binance) niet ophalen: {e}")


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
    if not NTFY_TOPIC or "wijzig-dit" in NTFY_TOPIC:
        print("Let op: NTFY_TOPIC is nog niet aangepast naar een eigen unieke naam "
              "— pushmelding wordt overgeslagen.")
        return False
    try:
        SESSION.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": encode_header_value(title)},
            timeout=10,
        )
        return True
    except Exception as e:
        print(f"Kon pushmelding niet versturen naar ntfy.sh: {e}")
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
    if send_ntfy("GOUD SIGNAAL", message):
        print(f"Pushmelding (signaal) verstuurd naar ntfy.sh/{NTFY_TOPIC}")


def send_heartbeat_notification(now, current_price, rsi, signal):
    time_txt = now.strftime("%H:%M")
    rsi_txt = format_nl(rsi) if rsi is not None else "n.v.t."
    status_txt = f"{signal['direction']}-setup actief" if signal is not None else "nog geen setup"
    message = f"{time_txt} — Prijs: {format_nl(current_price)}, RSI: {rsi_txt}, {status_txt}"
    if send_ntfy("Goud check — actief", message):
        print(f"Heartbeat verstuurd naar ntfy.sh/{NTFY_TOPIC}: \"{message}\"")


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
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

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
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                run_once(send_heartbeat=is_heartbeat_tick)
            except Exception as e:
                print(f"[{timestamp}] FOUT tijdens deze check: {e}")
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


def should_send_heartbeat(now):
    # Stateless-veilige heartbeat via een klein statusbestand i.p.v. een
    # tijdvenster: "is dit een nieuw HEARTBEAT_INTERVAL_MINUTES-blok t.o.v. de
    # laatst opgeslagen waarde?". Dat werkt correct bij elke vertraging (ook
    # 10+ minuten of een overgeslagen tik) en stuurt nooit dubbel, ook niet als
    # er toevallig meerdere checks binnen hetzelfde blok op tijd draaien.
    current_block = _heartbeat_block_start(now)
    last_block = None
    try:
        if os.path.exists(HEARTBEAT_STATE_PATH):
            with open(HEARTBEAT_STATE_PATH, "r", encoding="utf-8") as f:
                last_block = f.read().strip()
    except OSError as e:
        print(f"Let op: kon heartbeat-statusbestand niet lezen ({e}); "
              f"ga uit van 'nog geen eerdere heartbeat'.")

    if last_block == current_block:
        return False

    try:
        with open(HEARTBEAT_STATE_PATH, "w", encoding="utf-8") as f:
            f.write(current_block)
    except OSError as e:
        print(f"Let op: kon heartbeat-statusbestand niet wegschrijven ({e}).")

    return True


def run_once_stateless():
    # Voor gebruik in een scheduler (GitHub Actions, cron): elke aanroep is
    # een nieuw proces zonder in-memory geheugen van vorige runs, dus de
    # heartbeat-telling van run_forever() werkt hier niet. In plaats daarvan
    # bepaalt should_send_heartbeat() dit via het statusbestand.
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    is_heartbeat = should_send_heartbeat(now)
    try:
        run_once(send_heartbeat=is_heartbeat)
    except Exception as e:
        print(f"[{timestamp}] FOUT tijdens deze check: {e}")
        if is_heartbeat:
            print(f"[{timestamp}] Heartbeat overgeslagen deze ronde wegens fout.")


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
