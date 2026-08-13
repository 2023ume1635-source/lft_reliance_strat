#!/usr/bin/env python3
"""
reliance_market_depth_updated.py

Reliance Industries / RELIND - Real-Time Market Depth via ICICI Breeze API
==========================================================================

This version fixes the main problems in the earlier collector:

1. Event-driven gaps
   - Breeze depth ticks may not arrive every second.
   - The websocket callback now only caches the latest quote/depth.
   - A separate writer loop writes exactly one snapshot per second.

2. Duplicate timestamps / no milliseconds
   - Rows include both second-level timestamp and millisecond receive timestamps.

3. Stale-book awareness
   - Rows include DepthAgeSec and IsStale so you know whether a row is fresh or
     repeated from the last received depth tick.

4. Quote/depth separation
   - Quote ticks update LTP cache.
   - Depth ticks update book cache.
   - The per-second row combines latest known depth + latest known LTP.

5. Reduced callback workload
   - No file writing or heavy logging inside websocket callback.
   - This reduces the chance of callback backlog.

6. Robust depth parsing
   - Handles depth as list-of-dicts or dict.
   - Searches for the required key across depth objects.

Output:
    Original 32-column TSV format + extra diagnostics:

    TimeStamp (per second),
    5 bid levels: Orders, Qty, Price,
    5 ask levels: Price, Qty, Orders,
    LTP,
    ReceiveTimeMs,
    LastDepthUpdateTimeMs,
    LastQuoteUpdateTimeMs,
    DepthAgeSec,
    QuoteAgeSec,
    IsStale,
    DepthTickCount,
    QuoteTickCount,
    OtherTickCount,
    ExchangeTime

Usage:
    python reliance_market_depth_updated.py
    python reliance_market_depth_updated.py --stale-threshold 2.5
    python reliance_market_depth_updated.py --duration 600
    python reliance_market_depth_updated.py --log-raw-keys

Prerequisites:
    pip install breeze-connect python-dotenv
"""

import os
import sys
import csv
import time
import json
import signal
import logging
import argparse
import threading
from copy import deepcopy
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from breeze_connect import BreezeConnect


# ---------------------------------------------------------------------------
# Constants / Defaults
# ---------------------------------------------------------------------------
STOCK_CODE = "RELIND"          # ICICI Breeze internal code for Reliance Industries
EXCHANGE_CODE = "NSE"
PRODUCT_TYPE = "cash"

BASE_HEADERS = [
    "TimeStamp (per second)",

    "Orders", "Qty", "Price",   # Bid 1
    "Orders", "Qty", "Price",   # Bid 2
    "Orders", "Qty", "Price",   # Bid 3
    "Orders", "Qty", "Price",   # Bid 4
    "Orders", "Qty", "Price",   # Bid 5

    "Price", "Qty", "Orders",   # Ask 1
    "Price", "Qty", "Orders",   # Ask 2
    "Price", "Qty", "Orders",   # Ask 3
    "Price", "Qty", "Orders",   # Ask 4
    "Price", "Qty", "Orders",   # Ask 5

    "LTP",
]

EXTRA_HEADERS = [
    "ReceiveTimeMs",
    "LastDepthUpdateTimeMs",
    "LastQuoteUpdateTimeMs",
    "DepthAgeSec",
    "QuoteAgeSec",
    "IsStale",
    "DepthTickCount",
    "QuoteTickCount",
    "OtherTickCount",
    "ExchangeTime",
]

HEADERS = BASE_HEADERS + EXTRA_HEADERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_dt():
    return datetime.now()


def fmt_sec(dt):
    if dt is None:
        return ""
    return dt.strftime("%H:%M:%S")


def fmt_ms(dt):
    if dt is None:
        return ""
    return dt.strftime("%H:%M:%S.%f")[:-3]


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def parse_args():
    parser = argparse.ArgumentParser(description="RELIND per-second market-depth collector via ICICI Breeze")
    parser.add_argument("--stock-code", default=STOCK_CODE, help="Breeze stock code, default RELIND")
    parser.add_argument("--exchange-code", default=EXCHANGE_CODE, help="Exchange code, default NSE")
    parser.add_argument("--product-type", default=PRODUCT_TYPE, help="Product type, default cash")
    parser.add_argument("--output", default=None, help="Output TSV filename")
    parser.add_argument("--stale-threshold", type=float, default=2.5,
                        help="DepthAgeSec above this marks IsStale=1; default 2.5 sec")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional run duration in seconds; default run until Ctrl+C")
    parser.add_argument("--write-empty-before-first-depth", action="store_true",
                        help="Write rows even before first depth tick; fields blank")
    parser.add_argument("--log-raw-keys", action="store_true",
                        help="Log tick keys and occasional raw tick samples for debugging")
    parser.add_argument("--raw-log-every", type=int, default=100,
                        help="If --log-raw-keys, log every Nth raw tick; default 100")
    parser.add_argument("--console-every", type=int, default=1,
                        help="Console log every N written rows; default 1")
    parser.add_argument("--no-console-book", action="store_true",
                        help="Disable top-of-book console logs")
    parser.add_argument("--flush-every", type=int, default=1,
                        help="Flush file every N written rows; default 1")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------
class MarketState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_depth = None
        self.latest_ltp = ""
        self.latest_exchange_time = ""
        self.last_depth_update_dt = None
        self.last_quote_update_dt = None
        self.last_any_tick_dt = None

        self.depth_tick_count = 0
        self.quote_tick_count = 0
        self.other_tick_count = 0
        self.raw_tick_count = 0

    def update_quote(self, ltp=None, exchange_time=""):
        with self.lock:
            self.last_any_tick_dt = now_dt()
            self.last_quote_update_dt = self.last_any_tick_dt
            if ltp not in (None, ""):
                self.latest_ltp = ltp
            if exchange_time not in (None, ""):
                self.latest_exchange_time = exchange_time
            self.quote_tick_count += 1

    def update_depth(self, depth, ltp=None, exchange_time=""):
        with self.lock:
            self.last_any_tick_dt = now_dt()
            self.last_depth_update_dt = self.last_any_tick_dt
            self.latest_depth = deepcopy(depth)
            if ltp not in (None, ""):
                self.latest_ltp = ltp
            if exchange_time not in (None, ""):
                self.latest_exchange_time = exchange_time
            self.depth_tick_count += 1

    def update_other(self):
        with self.lock:
            self.last_any_tick_dt = now_dt()
            self.other_tick_count += 1

    def snapshot(self):
        with self.lock:
            return {
                "latest_depth": deepcopy(self.latest_depth),
                "latest_ltp": self.latest_ltp,
                "latest_exchange_time": self.latest_exchange_time,
                "last_depth_update_dt": self.last_depth_update_dt,
                "last_quote_update_dt": self.last_quote_update_dt,
                "last_any_tick_dt": self.last_any_tick_dt,
                "depth_tick_count": self.depth_tick_count,
                "quote_tick_count": self.quote_tick_count,
                "other_tick_count": self.other_tick_count,
                "raw_tick_count": self.raw_tick_count,
            }

    def inc_raw(self):
        with self.lock:
            self.raw_tick_count += 1
            return self.raw_tick_count


# ---------------------------------------------------------------------------
# Breeze depth parser
# ---------------------------------------------------------------------------
def extract_exchange_time(ticks):
    """Try common timestamp fields in Breeze tick payload."""
    for key in [
        "exchange_time", "exchangeTime", "time", "datetime", "ltt", "last_trade_time",
        "lastTradeTime", "ltt_time", "timestamp",
    ]:
        val = ticks.get(key)
        if val not in (None, ""):
            return str(val)
    return ""


def extract_ltp(ticks, current_ltp=""):
    """Try common LTP fields."""
    for key in ["last", "ltp", "LTP", "last_price", "lastPrice", "close"]:
        val = ticks.get(key)
        if val not in (None, ""):
            return val
    return current_ltp


def depth_objects(depth):
    """
    Normalize Breeze depth payload to a list of dictionaries.

    Breeze has appeared in different shapes across examples:
      - depth is a list of dicts
      - depth is a dict
      - depth list contains one dict with all BestBuyRate-1..5 keys
      - depth list contains level-wise dicts
    """
    if depth is None:
        return []
    if isinstance(depth, dict):
        return [depth]
    if isinstance(depth, list):
        return [x for x in depth if isinstance(x, dict)]
    return []


def get_depth_value(depth, key, level=None):
    """
    Robustly fetch a key from depth.

    First search all depth objects for the exact key. If not found and level is
    supplied, try the level-indexed dict as a fallback.
    """
    objs = depth_objects(depth)
    for d in objs:
        if key in d and d.get(key) not in (None, ""):
            return d.get(key)

    if level is not None and 0 <= level - 1 < len(objs):
        d = objs[level - 1]
        if key in d and d.get(key) not in (None, ""):
            return d.get(key)

    return ""


def top_of_book(depth):
    return {
        "bid_price": get_depth_value(depth, "BestBuyRate-1", 1),
        "bid_qty": get_depth_value(depth, "BestBuyQty-1", 1),
        "bid_orders": get_depth_value(depth, "BuyNoOfOrders-1", 1),
        "ask_price": get_depth_value(depth, "BestSellRate-1", 1),
        "ask_qty": get_depth_value(depth, "BestSellQty-1", 1),
        "ask_orders": get_depth_value(depth, "SellNoOfOrders-1", 1),
    }


def build_depth_row(depth):
    """Build the original 30 depth columns: bids then asks."""
    row = []

    # Best 5 Bids -> Orders, Qty, Price
    for i in range(1, 6):
        row.append(get_depth_value(depth, f"BuyNoOfOrders-{i}", i))
        row.append(get_depth_value(depth, f"BestBuyQty-{i}", i))
        row.append(get_depth_value(depth, f"BestBuyRate-{i}", i))

    # Best 5 Asks -> Price, Qty, Orders
    for i in range(1, 6):
        row.append(get_depth_value(depth, f"BestSellRate-{i}", i))
        row.append(get_depth_value(depth, f"BestSellQty-{i}", i))
        row.append(get_depth_value(depth, f"SellNoOfOrders-{i}", i))

    return row


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
def make_on_ticks(state: MarketState, args):
    def on_ticks(ticks):
        if not isinstance(ticks, dict):
            state.update_other()
            return

        raw_n = state.inc_raw()
        if args.log_raw_keys and (raw_n <= 5 or raw_n % args.raw_log_every == 0):
            logging.info("RAW tick #%d keys=%s sample=%s", raw_n, list(ticks.keys()), json.dumps(ticks, default=str)[:1500])

        exchange_time = extract_exchange_time(ticks)

        # Quote ticks: cache LTP.
        if ticks.get("quotes") == "Quotes Data":
            ltp = extract_ltp(ticks)
            state.update_quote(ltp=ltp, exchange_time=exchange_time)
            return

        # Depth ticks: cache latest depth and possibly LTP.
        if "depth" in ticks:
            ltp = extract_ltp(ticks)
            state.update_depth(ticks.get("depth"), ltp=ltp, exchange_time=exchange_time)
            return

        # Some Breeze payloads may carry LTP without quotes marker.
        possible_ltp = extract_ltp(ticks, current_ltp="")
        if possible_ltp not in (None, ""):
            state.update_quote(ltp=possible_ltp, exchange_time=exchange_time)
        else:
            state.update_other()

    return on_ticks


def writer_loop(state: MarketState, writer, tsv_file, stop_event, args):
    """Write exactly one latest-book snapshot per second."""
    row_count = 0
    start = time.time()

    # Align writes to next integer second boundary.
    next_write = int(time.time()) + 1

    while not stop_event.is_set():
        now_ts = time.time()
        sleep_for = max(0.0, next_write - now_ts)
        if stop_event.wait(timeout=sleep_for):
            break

        receive_dt = now_dt()
        snap = state.snapshot()
        depth = snap["latest_depth"]

        if depth is None and not args.write_empty_before_first_depth:
            next_write += 1
            if args.duration is not None and time.time() - start >= args.duration:
                stop_event.set()
            continue

        depth_age = ""
        quote_age = ""
        is_stale = 1

        if snap["last_depth_update_dt"] is not None:
            depth_age_val = (receive_dt - snap["last_depth_update_dt"]).total_seconds()
            depth_age = f"{depth_age_val:.3f}"
            is_stale = int(depth_age_val > args.stale_threshold)

        if snap["last_quote_update_dt"] is not None:
            quote_age_val = (receive_dt - snap["last_quote_update_dt"]).total_seconds()
            quote_age = f"{quote_age_val:.3f}"

        row = [fmt_sec(receive_dt)]
        row.extend(build_depth_row(depth))
        row.append(snap["latest_ltp"])
        row.extend([
            fmt_ms(receive_dt),
            fmt_ms(snap["last_depth_update_dt"]),
            fmt_ms(snap["last_quote_update_dt"]),
            depth_age,
            quote_age,
            is_stale,
            snap["depth_tick_count"],
            snap["quote_tick_count"],
            snap["other_tick_count"],
            snap["latest_exchange_time"],
        ])

        writer.writerow(row)
        row_count += 1

        if args.flush_every > 0 and row_count % args.flush_every == 0:
            tsv_file.flush()

        if not args.no_console_book and args.console_every > 0 and row_count % args.console_every == 0:
            tob = top_of_book(depth)
            logging.info(
                "%s | Bid: %s x %s (%s) | Ask: %s x %s (%s) | LTP: %s | DepthAge=%s | stale=%s",
                args.stock_code,
                tob["bid_price"] or "-",
                tob["bid_qty"] or "-",
                tob["bid_orders"] or "-",
                tob["ask_price"] or "-",
                tob["ask_qty"] or "-",
                tob["ask_orders"] or "-",
                snap["latest_ltp"] or "-",
                depth_age or "-",
                is_stale,
            )

        next_write += 1

        if args.duration is not None and time.time() - start >= args.duration:
            stop_event.set()

    tsv_file.flush()
    logging.info("Writer stopped. Rows written: %d", row_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    api_key = os.getenv("ICICI_APP_KEY")
    api_secret = os.getenv("ICICI_SECRET_KEY")
    session_token = os.getenv("ICICI_SESSION_TOKEN")

    if not all([api_key, api_secret, session_token]):
        logging.error("Missing credentials in .env file: ICICI_APP_KEY, ICICI_SECRET_KEY, ICICI_SESSION_TOKEN")
        sys.exit(1)

    if args.output is None:
        args.output = f"reliance_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"

    out_path = Path(args.output)
    logging.info("Output file: %s", out_path)

    tsv_file = open(out_path, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(tsv_file, delimiter="\t")
    writer.writerow(HEADERS)
    tsv_file.flush()

    state = MarketState()
    stop_event = threading.Event()

    breeze = BreezeConnect(api_key=api_key)
    breeze.generate_session(api_secret=api_secret, session_token=session_token)
    breeze.on_ticks = make_on_ticks(state, args)

    writer_thread = threading.Thread(
        target=writer_loop,
        args=(state, writer, tsv_file, stop_event, args),
        daemon=True,
        name="per_second_writer",
    )

    def shutdown(signum=None, frame=None):
        if stop_event.is_set():
            return
        logging.info("Shutting down...")
        stop_event.set()

        try:
            breeze.unsubscribe_feeds(
                exchange_code=args.exchange_code,
                stock_code=args.stock_code,
                product_type=args.product_type,
                get_market_depth=True,
                get_exchange_quotes=True,
            )
        except Exception as e:
            logging.debug("unsubscribe ignored: %s", e)

        try:
            breeze.ws_disconnect()
        except Exception as e:
            logging.debug("ws_disconnect ignored: %s", e)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        logging.info("Connecting to ICICI Breeze websocket...")
        breeze.ws_connect()
        time.sleep(3)  # socket.io handshake buffer, useful on Windows

        logging.info("Subscribing to %s (%s) depth + quotes...", args.stock_code, args.exchange_code)
        resp = breeze.subscribe_feeds(
            exchange_code=args.exchange_code,
            stock_code=args.stock_code,
            product_type=args.product_type,
            get_market_depth=True,
            get_exchange_quotes=True,
        )
        logging.info("Subscribe response: %s", resp)

        writer_thread.start()
        logging.info("Streaming per-second snapshots. Press Ctrl+C to stop.")

        while not stop_event.is_set():
            time.sleep(0.5)

    except Exception as e:
        logging.exception("Fatal error: %s", e)
        stop_event.set()

    finally:
        shutdown()
        writer_thread.join(timeout=5)
        try:
            tsv_file.close()
        except Exception:
            pass
        snap = state.snapshot()
        logging.info(
            "Saved to %s | depth_ticks=%d quote_ticks=%d other_ticks=%d raw_ticks=%d",
            out_path,
            snap["depth_tick_count"],
            snap["quote_tick_count"],
            snap["other_tick_count"],
            snap["raw_tick_count"],
        )


if __name__ == "__main__":
    main()
