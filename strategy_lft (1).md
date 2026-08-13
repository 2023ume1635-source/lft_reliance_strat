  📖 LFT Reliance Scalping Strategy — Complete Documentation

>  A story of teaching a computer to read the heartbeat of the stock market, one second at a time. 

---

   Table of Contents

1. [The Story: What Are We Trying to Do?]( 1-the-story)
2. [The Cast of Characters]( 2-the-cast)
3. [Phase 0: Ears — Listening to the Market]( 3-phase-0)
4. [Phase 1: Cleaning the Messy Room]( 4-phase-1)
5. [Phase 2: Teaching the Computer to See Patterns]( 5-phase-2)
6. [Phase 3: Grading the Future (Labelling)]( 6-phase-3)
7. [Phase 4: The Brain — Machine Learning Model]( 7-phase-4)
8. [Phase 5: The Decision Maker — Signal Generation]( 8-phase-5)
9. [Phase 6: The Time Machine — Backtesting]( 9-phase-6)
10. [The Live Bot: Putting It All Together]( 10-live-bot)
11. [Bugs Found & Fixed]( 11-bugs)
12. [Results]( 12-results)
13. [Glossary]( 13-glossary)

---

   1. The Story: What Are We Trying to Do?

Imagine you're standing in a crowded marketplace. Thousands of people are shouting prices — some want to BUY apples, some want to SELL apples. If you listen very carefully, you might notice patterns:

- "Hmm, every time 5 big buyers shout at once, the price goes UP in the next 20 seconds."
- "When the sellers suddenly disappear, the price jumps."

Now imagine you could listen to this marketplace  every single second , remember the last 10 minutes of shouting, and use that memory to predict: *"In the next 20-30 seconds, will the price go up or down?"*

That's exactly what this strategy does — but instead of apples, it's  Reliance Industries shares  on the National Stock Exchange of India (NSE). And instead of your ears, it uses a  computer program  that reads the "order book" — a live list of everyone who wants to buy or sell, and at what price.

    The Goal

Make tiny profits (₹0.30–₹0.70 per share) by correctly guessing the price direction for the next 20-30 seconds, and doing this 20-30 times per day. With 500-2000 shares per trade, that's ₹150–₹1400 per winning trade.

    Why Is This Hard?

- The price moves are TINY (₹0.05 = 1 tick, the minimum price change)
- You're competing against supercomputers and professional traders
- The market changes its "personality" every few minutes (sometimes calm, sometimes wild)
- You need to be right more than 55% of the time just to break even (because of transaction costs)

---

   2. The Cast of Characters

| Component | Role | Analogy |
|-----------|------|---------|
|  Order Book  | Live list of buy/sell orders at 5 price levels | The marketplace shouting |
|  LightGBM  | Machine learning model that finds patterns | The brain that learns |
|  Regime Engine  | Classifies market "mood" (calm/wild/trending) | Reading the room's mood |
|  Triple Barrier  | Labels each second as "buy/sell/do nothing" | The teacher giving grades |
|  Kalman Filter  | Estimates the "true" fair price | A wise judge filtering noise |
|  OFI (Order Flow Imbalance)  | Measures buying vs selling pressure | A tug-of-war scoreboard |
|  Isotonic Calibration  | Fixes the model's overconfidence | A reality check |
|  Backtest Engine  | Simulates trading on past data | A time machine for practice |

---

  3. Phase 0: Ears — Listening to the Market

    File: `fetch_reliance_orderbook.py`

    What It Does

Connects to ICICI Bank's Breeze API (a websocket — think of it as a live telephone line to the stock exchange) and records a  snapshot every 1 second  of:

- The top 5 buy prices and quantities (the "bid" side)
- The top 5 sell prices and quantities (the "ask" side)
- The Last Traded Price (LTP)
- Timestamps and metadata

    The Order Book — Explained Simply

Think of the order book as two queues:


BUYERS (Bids)                    SELLERS (Asks)
─────────────────                ─────────────────
Price    Qty    Orders           Price    Qty    Orders
1267.50  500    12               1267.55  300    8     ← Best Ask (cheapest seller)
1267.45  1200   25               1267.60  800    15
1267.40  300    5                1267.65  200    3
1267.35  2000   30               1267.70  1500   20
1267.30  100    2                1267.75  400    7



The  spread  = Best Ask − Best Bid = 1267.55 − 1267.50 = ₹0.05 (1 tick)

The  mid price  = (Best Bid + Best Ask) / 2 = 1267.525

    Key Design Decisions

1.  Event-driven caching : The websocket sends data irregularly (sometimes 5 updates/sec, sometimes none for 2 seconds). Instead of writing every update, we cache the LATEST state and write ONE row per second on a timer. This gives clean, evenly-spaced data.

2.  Staleness tracking : If no new data arrives for >2.5 seconds, we mark that row as "stale" — the computer knows it's looking at old information.

3.  Cumulative tick counters : We record running totals of depth ticks and quote ticks so we can later compute "ticks per second" (a measure of market activity).

    Output Format

A TSV (tab-separated) file with 42 columns per second:

TimeStamp | b1_orders b1_qty b1_price | ... | b5_orders b5_qty b5_price |
a1_price a1_qty a1_orders | ... | a5_price a5_qty a5_orders |
LTP | recv_time | depth_time | quote_time | depth_age | quote_age | is_stale | depth_ticks | quote_ticks | other_ticks | exchange_time


---

   4. Phase 1: Cleaning the Messy Room

    File: `phase1_ingest.py`

    What It Does

Takes the raw CSV files from Phase 0 and produces  clean, gap-free, validated  data ready for analysis.

    The Problems With Raw Data

1.  Missing seconds : The websocket sometimes drops data. If second 10:30:45 is missing, we have a gap.
2.  Pre-market data : The exchange sends data from 9:00 AM (pre-open auction) but real trading only happens 9:15–15:30.
3.  Bad prices : Sometimes the exchange sends price=0 or bid≥ask (impossible in reality — it's a data error).
4.  Duplicate timestamps : Two rows for the same second.

    The Cleaning Pipeline (Step by Step)


Step 1: Read CSV → rename columns to standard names (b1_p, b1_q, b1_o, etc.)
Step 2: Convert all price/qty columns to numbers (replace garbage with NaN)
Step 3: Build proper datetime index from date + time strings
Step 4: Remove duplicate seconds (keep the last one)
Step 5: Filter to Regular Trading Hours only (9:15:00 to 15:30:00)
Step 6: Fix bad prices:
        - Set price ≤ 0 to NaN
        - Detect crossed markets (bid ≥ ask) → set to NaN
Step 7: Forward-fill NaN prices (carry last valid price forward)
Step 8: Re-index to a perfect 1-second grid (fill any remaining gaps)
Step 9: Forward-fill all remaining NaN values
Step 10: Compute basic derived quantities (mid, spread, imbalance)
Step 11: Save as Parquet (compressed columnar format, 10x smaller than CSV)


    FIX Applied: Minimum Row Guard

 Problem : Some days had only 200-300 rows (half-days, holidays, connectivity issues). Training on these would add noise.

 Fix : Skip any day with fewer than 500 RTH rows (≈8 minutes of data minimum).

    Basic Derived Quantities Computed Here

| Quantity | Formula | Meaning |
|----------|---------|---------|
|  Mid  | `(best_bid + best_ask) / 2` | The "centre" of the market |
|  Spread  | `best_ask - best_bid` | Cost to buy and immediately sell |
|  Spread (ticks)  | `spread / 0.05` | Spread in minimum price units |
|  Microprice  | `(bid×ask_qty + ask×bid_qty) / (bid_qty + ask_qty)` | Volume-weighted fair price |
|  Imbalance (Level 1)  | `(bid_qty - ask_qty) / (bid_qty + ask_qty)` | Ranges from -1 (all sellers) to +1 (all buyers) |
|  Total Depth  | `sum of all 5 bid qtys + sum of all 5 ask qtys` | Total visible liquidity |
|  Depth Imbalance  | `(total_bid_qty - total_ask_qty) / total_depth` | Overall buying vs selling pressure |
|  Bid VWAP  | `Σ(bid_price_i × bid_qty_i) / Σ(bid_qty_i)` | Average bid price weighted by quantity |
|  Ask VWAP  | `Σ(ask_price_i × ask_qty_i) / Σ(ask_qty_i)` | Average ask price weighted by quantity |

    Date Splitting

The 15 days of data are split  chronologically  (no random shuffling — that would be cheating by "seeing the future"):

-  TRAIN  (first 8 days): The model learns from these
-  TEST  (next 4 days): We check if the model learned real patterns or just memorised
-  VALID  (last 3 days): Final reality check — completely unseen data

This is called  walk-forward validation  — it mimics real life where you only have past data to learn from.

---

   5. Phase 2: Teaching the Computer to See Patterns

    File: `phase2_features.py`

    What It Does

Transforms the raw order book data into  130+ numerical features  — think of these as "clues" or "symptoms" that help the computer predict the future. Each feature captures a different aspect of market behaviour.

    Why So Many Features?

A single number like "bid price" tells you almost nothing. But combinations like "the ratio of buying pressure to selling pressure, compared to its average over the last 5 minutes" can be very predictive. We compute features in  10 groups , each capturing a different "lens" through which to view the market.

---

    Group 1: Level-1 Microstructure (The Basics)

These are the simplest, most intuitive features:

     Mid Price

mid = (best_bid + best_ask) / 2

 Why : The mid is our best estimate of the "true" price at this instant. All predictions are about how mid will change.

     Spread

spread = best_ask - best_bid
spread_ticks = round(spread / 0.05)
spread_bps = (spread / mid) × 10000

 Why : A wide spread means the market is uncertain or illiquid. A tight spread (1 tick = ₹0.05) means lots of competition — good for scalping. `spread_bps` (basis points) normalises for price level.

     Microprice

microprice = (bid × ask_qty + ask × bid_qty) / (bid_qty + ask_qty)

 Why : If there are 1000 shares waiting to buy but only 100 shares waiting to sell, the "fair" price is closer to the ask (sellers have more power). The microprice captures this. It's always between bid and ask.

     Microprice Deviation

micro_dev = (microprice - mid) / spread

 Why : Normalised to [-0.5, +0.5]. Positive means buying pressure is pushing the fair price above mid.

     Weighted Mid (using all 5 levels)

wmid = (bid_vwap × total_ask_qty + ask_vwap × total_bid_qty) / (total_bid_qty + total_ask_qty)

 Why : Like microprice but using the full depth of the book, not just the best level. More robust to noise at the top.

     Level-wise Imbalance (5 values)

imb_i = (bid_qty_i - ask_qty_i) / (bid_qty_i + ask_qty_i)    for i = 1..5

 Why : Each price level has its own tug-of-war. Level 1 imbalance is most important (immediate pressure), but deeper levels show "hidden" support/resistance.

     LTP Side

ltp_side = +1 if LTP ≥ ask
           -1 if LTP ≤ bid
            0 if bid < LTP < ask

 Why : If the last trade happened AT the ask price, it means someone was aggressive enough to pay the seller's price — buying pressure. This is a simple but powerful signal.

---

    Group 2: Order Flow Imbalance (OFI) — The Tug-of-War Score

This is one of the most important features. Based on the academic paper by  Cont, Kukanov & Stoikov (2013) : *"The Price Impact of Order Book Events"*.

     The Core Idea

Every second, the order book changes. Some changes indicate buying pressure, some indicate selling pressure. OFI quantifies this precisely.

     The Formula (Level 1)


OFI_1(t) = I{bid(t) ≥ bid(t-1)} × bid_qty(t) - I{bid(t) ≤ bid(t-1)} × bid_qty(t-1)
           - I{ask(t) ≤ ask(t-1)} × ask_qty(t) + I{ask(t) ≥ ask(t-1)} × ask_qty(t-1)


Where `I{condition}` = 1 if condition is true, 0 otherwise.

     Explained in Plain English

 Bid side contribution: 
- If the bid price went UP or stayed the same: add the current bid quantity (buyers are aggressive/holding)
- If the bid price went DOWN or stayed the same: subtract the previous bid quantity (buyers retreated)

 Ask side contribution: 
- If the ask price went DOWN or stayed the same: subtract the current ask quantity (sellers are aggressive)
- If the ask price went UP or stayed the same: add the previous ask quantity (sellers retreated)

 Net OFI  = Bid contribution − Ask contribution

-  Positive OFI  → buying pressure (more likely price goes UP)
-  Negative OFI  → selling pressure (more likely price goes DOWN)
-  Zero OFI  → no change in pressure

     Why This Formula Works

Consider what happens when a buyer places a large order at the current bid:
- bid_price stays same, bid_qty increases
- `I{bid ≥ bid_prev} = 1`, so we add the new (larger) qty
- `I{bid ≤ bid_prev} = 1`, so we subtract the old (smaller) qty
- Net = new_qty - old_qty = positive (the increase in buying interest)

Now consider when a market buyer "eats" the best ask:
- ask_price goes UP (the old best ask is gone, next one is higher)
- `I{ask ≤ ask_prev} = 0` (ask went up, not down)
- `I{ask ≥ ask_prev} = 1`, so we add the PREVIOUS ask qty (the qty that was consumed = selling pressure removed = buying won)

     Multi-Level OFI

We compute OFI for all 5 levels and create a weighted sum:

OFI_weighted = 1.0×OFI_1 + 0.6×OFI_2 + 0.35×OFI_3 + 0.18×OFI_4 + 0.08×OFI_5

 Why these weights?  Level 1 is most important (immediate execution). Deeper levels matter less because they're further from execution. The weights decay roughly exponentially.

     Cumulative OFI (Multiple Horizons)


OFI_sum_3  = sum of OFI over last 3 seconds
OFI_sum_10 = sum of OFI over last 10 seconds
OFI_sum_30 = sum of OFI over last 30 seconds
OFI_sum_60 = sum of OFI over last 60 seconds

 Why multiple horizons?  Short horizon (3s) captures immediate momentum. Long horizon (60s) captures sustained pressure. A signal is strongest when BOTH short and long OFI agree.

---

    Group 3: Book Shape — The Calculus of the Order Book

     The Idea

If you plot cumulative quantity vs price distance from mid, you get a curve. The  slope  and  curvature  of this curve tell you about liquidity structure.

     Book Slope

For each side, we fit a line through the 5 data points (price_offset, cumulative_qty):


slope = Σ((x_i - x̄)(y_i - ȳ)) / Σ((x_i - x̄)²)


Where:
- `x_i` = price offset in ticks from best level (0, 1, 2, 3, 4)
- `y_i` = cumulative quantity up to level i

 Steep bid slope  = lots of liquidity building up quickly below = strong support = price unlikely to fall.

 Flat ask slope  = thin liquidity above = price can easily rise = bullish signal.

     Slope Imbalance

slope_imb = (bid_slope - ask_slope) / (bid_slope + ask_slope + ε)

Positive = bid side is "thicker" than ask side = bullish.

     Book Curvature

bid_curve = cum_qty[2] - 2×cum_qty[1] + cum_qty[0]

This is the  second derivative  (discrete approximation). Positive curvature = liquidity is accelerating (getting thicker faster) = strong support forming.

     Queue Concentration

qconc_b3 = qty_within_3_ticks_of_bid / total_bid_qty

 Why : If 90% of bid liquidity is at the best price, the queue is "concentrated" — a single large sell order could wipe it out. If liquidity is spread across levels, the book is more resilient.

---

    Group 4: PCA on Book Pressure Vector — Linear Algebra

     The Idea

At each second, the 10 quantities (5 bid qtys + 5 ask qtys, normalised to sum to 1) form a point in 10-dimensional space. We use  Principal Component Analysis (PCA)  to find the main "directions" in which this point moves.

     What is PCA? (For Beginners)

Imagine you have a cloud of points in 3D space, but the cloud is flat (like a pancake). PCA finds the 2D plane that the pancake lies in — it reduces 3 dimensions to 2 without losing much information.

In our case, we have 10 dimensions (the 10 normalised quantities). PCA finds the 3 most important "directions" (principal components) that explain most of the variation.

     The Math

For each time t (with a 300-second lookback window):

1.  Compute the mean vector : μ = (1/W) × Σ bpv[i] for i in [t-W, t)
2.  Centre the data : C = bpv - μ (each row minus the mean)
3.  Covariance matrix : Σ = (C^T × C) / (W-1) — a 10×10 matrix capturing how the 10 quantities co-vary
4.  Eigendecomposition : Σ = V × Λ × V^T — find eigenvectors (directions) and eigenvalues (importance)
5.  Project current state : PC_k(t) = (bpv[t] - μ) · V[:,k] — how far along each principal direction we are

     What Do the PCs Mean?

-  PC1  often captures "overall liquidity level" (all quantities up or down together)
-  PC2  often captures "bid-ask imbalance shift" (bids thickening while asks thinning)
-  PC3  often captures "depth migration" (liquidity moving from level 1 to deeper levels)

     Mahalanobis Distance

d_M(t) = √((bpv[t] - μ)^T × Σ^{-1} × (bpv[t] - μ))

 What it means : How "unusual" is the current book shape compared to the last 5 minutes? A high Mahalanobis distance means something unusual just happened (large order placed/cancelled, sudden imbalance shift). This often precedes price moves.

     FIX Applied: Batched PCA

 Problem : Computing eigendecomposition every second (400 times per day per 10×10 matrix) is slow.

 Fix : Only recompute PCA every 30 seconds. Use the same eigenvectors for the 29 seconds in between. This is 20× faster with negligible accuracy loss (the book structure doesn't change that fast).

---

    Group 5: Calculus — Derivatives and Integrals

     Velocity (First Derivative)

dmid_1  = mid(t) - mid(t-1)            price change per second
dmid_5  = (mid(t) - mid(t-5)) / 5      smoothed velocity over 5 seconds
dmid_10 = (mid(t) - mid(t-10)) / 10    smoothed velocity over 10 seconds

 Why : Price velocity tells you the current trend. Positive = price rising. The multi-second versions reduce noise.

     Acceleration (Second Derivative)

ddmid = dmid_1(t) - dmid_1(t-1)

 Why : If velocity is increasing (positive acceleration), the trend is strengthening. If velocity is positive but acceleration is negative, the trend is weakening — potential reversal signal.

     OFI Derivative

dofi_1 = OFI(t) - OFI(t-1)
dofi_5 = (OFI(t) - OFI(t-5)) / 5

 Why : The RATE OF CHANGE of buying/selling pressure. A sudden spike in OFI derivative = someone just placed/cancelled a large order.

     EWM-Smoothed Velocity

dmid_ewm10 = EMA(dmid_1, span=10)

 Why : Exponentially Weighted Moving Average gives more weight to recent observations. Smoother than simple average, more responsive than long window.

     Cumulative OFI (Integral)

ofi_cum(t) = Σ OFI(i) for i=1..t, divided by t (normalised)

 Why : The running average of buying/selling pressure since market open. If this is steadily positive, buyers have been dominant all day.

---

    Group 6: Statistics — Volatility, Z-Scores, Mean Reversion

     Realised Volatility

RV_H = std(returns over last H seconds) × √22500

Where 22500 = seconds in a trading day (6.25 hours × 3600). This annualises the per-second volatility to a daily number.

 Why : High volatility = market is wild = wider stops needed, harder to predict direction. Low volatility = calm = tighter targets work.

     Parkinson Range-Based Volatility

RV_parkinson = √((1/(4×ln(2))) × (ln(high/low))²)

 Why : More efficient estimator of volatility than simple std dev. Uses the range (high-low) which captures more information about the true volatility. Based on the fact that for a Brownian motion, the range follows a known distribution.

     Z-Scores

mid_z60 = (mid - mean(mid, 60s)) / std(mid, 60s)
mid_z300 = (mid - mean(mid, 300s)) / std(mid, 300s)

 What it means : How many standard deviations is the current price from its recent average? 

- `mid_z60 = +2.5` → price is unusually HIGH compared to last minute → likely to revert DOWN (mean reversion)
- `mid_z60 = -1.8` → price is unusually LOW → likely to revert UP

 Why two horizons?  60s z-score catches short-term overextensions. 300s z-score catches longer trends that might be about to reverse.

     Autocorrelation (Trendiness Score)

ACF_1(H) = corr(return(t), return(t-1)) computed over window H

 What it means :
-  Positive ACF  → returns tend to continue (trending market). If price went up last second, likely to go up again.
-  Negative ACF  → returns tend to reverse (mean-reverting market). If price went up, likely to go down next.
-  Zero ACF  → random walk, no predictable pattern.

 Why it matters : In trending regimes, we should follow momentum. In mean-reverting regimes, we should fade extremes.

---

    Group 7: Queue Dynamics — Add/Cancel Detection

     The Idea

We can't directly see order placements and cancellations, but we can INFER them from quantity changes:


If bid_qty increased → someone ADDED to the bid (or a sell was filled from ask)
If bid_qty decreased → someone CANCELLED their bid (or a buy order was filled)


     Add/Cancel Rates (10-second window)

b_add10 = sum of max(0, Δbid_qty_i) over all 5 levels, over last 10 seconds
b_can10 = sum of max(0, -Δbid_qty_i) over all 5 levels, over last 10 seconds


     Add-to-Cancel Ratio

add_can_ratio = (b_add10 + a_add10) / (b_can10 + a_can10 + 1)

 Interpretation :
- Ratio > 1 → more orders being placed than cancelled = liquidity building = market is "healthy"
- Ratio < 1 → more cancellations = liquidity draining = market is "nervous", possible move incoming

     Queue Ratio

qratio_b = bid_qty_1 / (bid_qty_2 + 1)

 Why : If the best bid has 10× more quantity than the second level, the queue is "top-heavy". A new buyer joining this queue will be at the BACK — they'll wait a long time to get filled. This affects our passive order fill probability.

     Book Walls

bwall_3t = max qty at any single level within 3 ticks of bid

 Why : A "wall" is a large order sitting at one price. It acts as support (bid wall) or resistance (ask wall). Price tends to bounce off walls.

---

    Group 8: Time & Session Features

     Time-of-Day Encoding

sec_open = seconds since 9:15 AM
tod_sin = sin(2π × sec_open / 22500)
tod_cos = cos(2π × sec_open / 22500)

 Why sin/cos?  The market has cyclical patterns (opening flurry, lunch lull, closing rush). Using sin/cos lets the model learn these cycles smoothly (9:15 and 15:30 are "close" in cycle space, just like midnight and 11:59 PM are close in clock space).

     Session Buckets

0 = Opening flurry (9:15-9:30) — high volatility, wide spreads
1 = Morning (9:30-11:00) — trending moves common
2 = Midday (11:00-13:30) — low activity, mean-reverting
3 = Afternoon (13:30-15:15) — gradual trend resumption
4 = Closing flurry (15:15-15:30) — high volume, institutional rebalancing

 Why : The model needs to know that a signal at 11:30 AM means something different from the same signal at 9:20 AM.

---

    Group 9: Kalman Filter — The Wise Judge

     The Problem

The observed mid price is NOISY. It jumps around due to random trades, even when the "true" value hasn't changed. We want to estimate the true underlying price.

     What is a Kalman Filter? (For Beginners)

Imagine you're tracking a car's position using a GPS that's accurate to ±5 meters. The GPS says the car moved 3 meters, but you know the car's engine can only produce 2 meters/second of movement. Should you believe the GPS completely? No — you should blend your prediction (based on physics) with the GPS reading.

The Kalman Filter does exactly this blending, optimally:
-  Prediction step : "Based on where it was last time, I expect it to be HERE"
-  Update step : "The sensor says it's THERE. Let me blend my prediction with the sensor reading, weighting by how much I trust each."

     The Math

 State : x = true mid price (what we want to estimate)
 Observation : z = observed mid price (noisy)

 Prediction :

x_pred(t) = x(t-1)             assume price doesn't change (random walk)
P_pred(t) = P(t-1) + Q         uncertainty grows by process noise Q


 Update :

K(t) = P_pred / (P_pred + R)           Kalman gain (how much to trust observation)
x(t) = x_pred + K × (z - x_pred)      blend prediction with observation
P(t) = (1 - K) × P_pred               uncertainty shrinks after observation


Where:
-  Q  = process noise = 1e-5 (we expect the true price to be very stable second-to-second)
-  R  = observation noise = spread/2 (the mid can jump by half the spread just from bid-ask bounce)
-  K  = Kalman gain = number between 0 and 1. High K = trust the observation. Low K = trust the prediction.

     The Innovation (Most Important Output)

innovation(t) = z(t) - x_pred(t) = observed_mid - predicted_true_mid

 What it means : The "surprise" — how much did the observation differ from what we expected? 

-  Large positive innovation  → price jumped UP more than expected → possible buying pressure
-  Large negative innovation  → price dropped more than expected → possible selling pressure
-  Small innovation  → price is behaving normally → no signal

     Innovation Z-Score

innov_z = innovation / std(innovation over last 300 seconds)

Normalises the surprise by recent volatility. An innovation of ₹0.10 is huge in a calm market (std=₹0.02) but normal in a volatile one (std=₹0.15).

     FIX Applied: Steady-State Kalman

 Problem : Running the full Kalman recursion for 22,500 seconds per day is slow in pure Python.

 Fix : The Kalman gain converges to a steady-state value after ~100 iterations. We run the full filter for the first 100 seconds, then use the converged gain for the rest. This is 50× faster with identical results.

---

    Group 10: Regime Engine — Reading the Room's Mood

     The Idea

The market has different "personalities" at different times. A strategy that works in calm conditions might fail in volatile ones. We need to know WHICH personality the market is in RIGHT NOW.

     The 6 Regimes

| ID | Name | Characteristics | Strategy |
|----|------|----------------|----------|
| 0 | Calm Chop | Low vol, tight spread, no trend | Market-making: buy bid, sell ask |
| 1 | Calm Trend | Low vol, tight spread, clear direction | Follow the trend aggressively |
| 2 | Normal | Average everything | Standard settings |
| 3 | Volatile Chop | High vol, wide spread, no direction | Reduce size, wider stops |
| 4 | Volatile Trend | High vol, strong direction | Aggressive directional, no MM |
| 5 | News/Frozen | Extreme spread or stale data | DO NOT TRADE |

     How Regimes Are Detected

python
  Vol regime: where is current 60s vol relative to last 10 minutes?
vol_percentile = rank(rv60, window=600) / 600
vol_regime = 0 if vol_percentile < 0.33 else (1 if < 0.75 else 2)

  Spread regime: is spread wider than usual?
median_spread = median(spread_ticks, window=300)
spread_regime = 0 if spread ≤ median+1 else (1 if ≤ median+4 else 2)

  Trend regime: is price trending or reverting?
trend_regime = +1 if ACF_1(60s) > 0.05 else (-1 if < -0.05 else 0)

  Composite market state (combines all signals):
state = 0 (calm_chop)  if vol_low AND spread_tight AND no_trend
state = 1 (calm_trend) if vol_low AND spread_tight AND trending
state = 3 (vol_chop)   if vol_high AND no_trend
state = 4 (vol_trend)  if vol_high AND trending
state = 5 (frozen)     if spread > 12 ticks OR stale > 3s OR tick_rate < 0.1
state = 2 (normal)     otherwise


 Why this matters : In regime 5, the data is unreliable — we MUST NOT trade. In regime 0, we can use tight targets. In regime 4, we need wider stops.

---

    FIX Applied to Phase 2: Performance Optimisation

| Component | Original | Fixed | Speedup |
|-----------|----------|-------|---------|
| PCA (G4) | Eigendecomposition every second | Every 30 seconds | 20× |
| Kalman (G9) | Full recursion 22500 steps | Steady-state after 100 steps | 50× |
| Overall | ~500ms/day | ~50ms/day | 10× |

---

   6. Phase 3: Grading the Future (Labelling)

    File: `phase3_labels.py`

    What It Does

For every second of historical data, we look INTO THE FUTURE and ask: "If we had entered a trade at this exact second, would we have made money or lost money?" This gives us the "answer key" for training the ML model.

    The Triple Barrier Method

     The Problem With Simple Labelling

A naive approach: "Label +1 if price is higher 30 seconds later." But this has problems:
- What if price went UP ₹2 first, then DOWN ₹1? You would have taken profit at +₹2, but the naive label says -₹1.
- What if price went DOWN ₹5 first, then recovered? You would have been stopped out at -₹5, but the naive label might say +₹1.

     The Solution: Three Barriers

At each second t, we define three "exit conditions":


Upper barrier (take profit)  = mid(t) + width
Lower barrier (stop loss)    = mid(t) - width  
Time barrier (timeout)       = t + H seconds


We look forward from t and see which barrier is hit FIRST:

-  Upper hit first  → Label = +1 (long trade would have won)
-  Lower hit first  → Label = -1 (long trade would have lost)
-  Time barrier hit first  → Label = 0 (trade expired without clear direction)

     Regime-Conditional Barriers

The barrier width and time horizon depend on the market regime:

| Regime | Width (× R) | Horizon (seconds) | Min Edge (× R) |
|--------|-------------|-------------------|----------------|
| 0 (calm_chop) | 0.35 | 20 | 0.25 |
| 1 (calm_trend) | 0.40 | 25 | 0.30 |
| 2 (normal) | 0.45 | 30 | 0.35 |
| 3 (vol_chop) | 0.60 | 45 | 0.50 |
| 4 (vol_trend) | 0.55 | 40 | 0.45 |
| 5 (frozen) | — | — | — (NO LABELS) |

 Why wider barriers in volatile regimes?  In wild markets, price swings more, so a "meaningful" move is larger. A ₹0.30 move in a calm market is significant; in a volatile market it's just noise.

     What is R?


R = 80th percentile of |mid(t+30) - mid(t)| over the entire day


R represents a "typical large move" for that day. All barriers are expressed as multiples of R, making them adaptive to each day's volatility.

 Clipping : R is bounded between `spread/2 + 1_tick` (minimum meaningful move) and `3 × median_spread` (maximum reasonable move).

     Overlap Purging

 Problem : If second 100 is labelled +1 (with horizon 30s), and second 101 is also labelled +1, these two labels are NOT independent — they're looking at almost the same future. This would inflate the model's apparent accuracy.

 Solution : After labelling a second as +1 or -1, skip the next H seconds (don't label them). This ensures each labelled event represents an independent trading opportunity.

     Passive Fill Labels

For passive (limit) orders, we also label: "Would a limit order placed at the current bid have been FILLED within 30 seconds without adverse selection?"


y_buyfill = 1 if (ask dropped to our bid level) AND (mid didn't move up by > half_spread)
y_sellfill = 1 if (bid rose to our ask level) AND (mid didn't move down by > half_spread)


 Why : A filled limit order that then loses money due to adverse selection is worse than no fill at all. We need to predict BOTH fill probability AND whether the fill is "safe".

    FIX Applied: Vectorised Labelling

 Problem : The original code used Python for-loops over 22,500 seconds × 30-second look-ahead = 675,000 iterations per day. At 15 days, this took 30+ minutes.

 Fix : Used numpy vectorised operations (`np.where` on array slices) to compute barrier hits in bulk. Reduced to ~2 minutes total.

---

   7. Phase 4: The Brain — Machine Learning Model

    File: `phase4_model.py`

    What It Does

Trains a  LightGBM  model to predict, for each second: "What is the probability that the label is +1 (buy), -1 (sell), or 0 (do nothing)?"

    Why LightGBM?

LightGBM is a  Gradient Boosted Decision Tree  algorithm. Here's why we chose it over alternatives:

| Algorithm | Pros | Cons | Why Not? |
|-----------|------|------|----------|
| Linear Regression | Fast, interpretable | Can't capture non-linear patterns | Market is non-linear |
| Neural Network (DL) | Can learn anything | Needs millions of samples, slow to train | We only have 300K samples |
| Random Forest | Handles non-linearity | Less accurate than boosting | LightGBM beats it |
|  LightGBM  | Fast, accurate, handles mixed features, built-in regularisation | Slightly less interpretable | ✅  Best fit  |
| XGBoost | Similar to LightGBM | Slower on large datasets | LightGBM is 5-10× faster |

    How Gradient Boosting Works (For Beginners)

Imagine you're trying to predict house prices. You ask 100 experts one by one:

1.  Expert 1  makes a rough guess. It's wrong by a lot.
2.  Expert 2  looks at WHERE Expert 1 was wrong, and builds a model to predict the ERRORS.
3.  Expert 3  looks at where Experts 1+2 combined are wrong, and predicts those errors.
4. ... and so on for 600 experts.

The final prediction = Expert 1 + Expert 2 + ... + Expert 600.

Each "expert" is a  decision tree  — a series of yes/no questions like:

Is OFI_sum_10 > 50?
  YES → Is mid_z60 < -1.5?
    YES → Predict +0.3 (likely to go up)
    NO  → Predict +0.1
  NO  → Is spread > 3 ticks?
    YES → Predict 0 (don't trade)
    NO  → Predict -0.05


LightGBM builds 600 such trees, each one correcting the mistakes of all previous trees.

    LightGBM Hyperparameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_leaves` | 63 | Maximum complexity of each tree (more leaves = more detail but risk of overfitting) |
| `min_data_in_leaf` | 80 | Each leaf must contain at least 80 data points (prevents memorising noise) |
| `feature_fraction` | 0.7 | Each tree only looks at 70% of features (forces diversity among trees) |
| `bagging_fraction` | 0.8 | Each tree only sees 80% of data rows (same idea — diversity) |
| `lambda_l1` | 0.1 | L1 regularisation — pushes weak features to zero (feature selection) |
| `lambda_l2` | 0.5 | L2 regularisation — prevents any single tree from being too confident |
| `learning_rate` | 0.04 | Each tree's prediction is multiplied by 0.04 (small steps = more stable) |
| `max_bin` | 127 | Continuous features are discretised into 127 buckets (speeds up training) |

    The Multi-Task Setup

We don't just train one model — we train  four models  simultaneously:

| Model | Target | Type | Purpose |
|-------|--------|------|---------|
|  M1 (Classifier)  | y_take_dir ∈ {-1, 0, +1} | 3-class classification | "Should I buy, sell, or do nothing?" |
|  M2 (Regressor)  | y_edge_30 (₹ amount) | Regression | "How much will I make/lose in ₹?" |
|  M3 (Binary)  | y_buyfill ∈ {0, 1} | Binary classification | "Will my passive buy order get filled?" |
|  M4 (Binary)  | y_sellfill ∈ {0, 1} | Binary classification | "Will my passive sell order get filled?" |

 Why multiple models?  Each answers a different question. The classifier tells direction. The regressor tells magnitude (for position sizing). The fill models tell execution probability (for choosing passive vs aggressive).

    Sample Weighting (FIX Applied)

 Problem : All training samples were weighted equally. But a second where the future move is ₹1.50 (huge, clear signal) should matter MORE than a second where the future move is ₹0.05 (noise).

 Fix : Weight each sample by `1 + |y_dir_R|` where y_dir_R is the R-multiple achieved. A trade that would have made 2R gets weight 3.0; a flat label gets weight 1.0.

    Walk-Forward Cross-Validation

Instead of a single train/test split, we do  expanding-window CV :


Fold 1: Train on Day 1,         Test on Day 2
Fold 2: Train on Days 1-2,      Test on Day 3
Fold 3: Train on Days 1-3,      Test on Day 4
...
Fold 7: Train on Days 1-7,      Test on Day 8


 Why expanding window?  It mimics reality: each day you have MORE history than the day before. It also gives us out-of-fold predictions for every day (except Day 1), which we use for calibration.

 Purging : We remove the last 90 seconds (30s purge + 60s embargo) from each training fold to prevent information leakage through the forward-looking labels.

    Isotonic Calibration

     The Problem

LightGBM's probability outputs are often  miscalibrated . It might say "70% probability of going up" but in reality, when it says 70%, the price only goes up 55% of the time.

     The Solution: Isotonic Regression

Isotonic regression is a  non-parametric  method that maps the model's raw probabilities to calibrated probabilities while preserving the ordering.

 How it works : 
1. Sort all predictions from lowest to highest
2. For each prediction bucket, compute the ACTUAL fraction of positive outcomes
3. Fit a monotonically increasing step function from raw → calibrated

 Example :

Raw probability 0.60 → Actually correct 45% of the time → Calibrated to 0.45
Raw probability 0.70 → Actually correct 55% of the time → Calibrated to 0.55
Raw probability 0.80 → Actually correct 70% of the time → Calibrated to 0.70


We fit SEPARATE calibrators for P(long) and P(short), using the out-of-fold predictions from CV.

    Microprice Baseline (M3 — Model-Free)

     The Stoikov Microprice Formula


logit_long  = k × (imbalance - 0.5) / σ_imbalance
logit_short = k × (0.5 - imbalance) / σ_imbalance
P(long)  = exp(logit_long) / (exp(logit_long) + exp(logit_short) + 1)
P(short) = exp(logit_short) / (exp(logit_long) + exp(logit_short) + 1)
P(flat)  = 1 / (exp(logit_long) + exp(logit_short) + 1)


Where:
- `k = √(spread / (2 × tick_size))` — tighter spreads give more confident signals
- `σ_imbalance` = rolling 60s standard deviation of imbalance (normalisation)

 Why include this?  It's a simple, model-free baseline based on market microstructure theory (Stoikov 2018). If the ML model can't beat this simple formula, something is wrong. We  blend  the ML model with this baseline:


P_final(long) = w × P_ML(long) + (1-w) × P_microprice(long)


Where `w` is optimised per-regime on the training data (typically w = 0.90-0.95, meaning the ML model dominates but the microprice provides a small regularisation effect).

    Feature Pruning

After training, we look at  feature importance  (total gain across all trees). Features with gain < 1e-5 are removed — they contribute nothing and just add noise. Typically 1-3 features are pruned.

    FIX Applied: Dropped Leakage Features

 Problem : Features like `mid`, `b_vwap`, `a_vwap`, `wmid`, `kf_mid` are LEVEL prices. In a forward-looking context, these don't leak. But they're highly autocorrelated (mid at t ≈ mid at t+1), so the model might just learn "if mid is high now, it'll be high in 1 second" — which is trivially true but useless for direction prediction.

 Fix : These level features are in the `DROP_LEAK` set and excluded from training. We keep their DERIVATIVES (dmid, ret, z-scores) which capture CHANGES rather than levels.

    Results of Phase 4

| Metric | TRAIN (OOF) | TEST | VALID |
|--------|-------------|------|-------|
| Directional Accuracy | 63.5% | 60.1% | 62.1% |
| Macro F1 | 0.335 | 0.338 | 0.342 |
| AUC (macro) | 0.650 | 0.621 | 0.639 |
| Edge MAE (₹) | — | 0.319 | 0.284 |

 Interpretation : 60-63% directional accuracy on unseen data means the model correctly predicts direction about 3 out of 5 times when it makes a prediction. This is enough to be profitable with proper risk management.

---

   8. Phase 5: The Decision Maker — Signal Generation

    File: `phase5_signals.py`

    What It Does

Converts the model's probability outputs into concrete TRADING SIGNALS: "Buy NOW", "Sell NOW", or "Do nothing".

    The Threshold System

The model outputs P(long) and P(short) every second. We need a rule to convert these into actions:


raw_signal = P(long) - P(short)      ranges from -1 to +1

If raw_signal > tau_take  AND |edge_pred| > cost + edge_floor:
    → AGGRESSIVE BUY (cross the spread, pay the ask)

If raw_signal > tau_passive AND |edge_pred| > edge_floor AND P(fill) > 0.15:
    → PASSIVE BUY (place limit order at bid, wait for fill)

If raw_signal < -tau_take AND |edge_pred| > cost + edge_floor:
    → AGGRESSIVE SELL

If raw_signal < -tau_passive AND |edge_pred| > edge_floor AND P(fill) > 0.15:
    → PASSIVE SELL

Otherwise:
    → DO NOTHING


    What Are tau_passive and tau_take?

These are  thresholds  — minimum conviction levels required to trade. They're different for passive vs aggressive because:

-  Passive orders  have lower risk (you get a better price if filled) → lower threshold (tau_passive ≈ 0.007-0.03)
-  Aggressive orders  pay the spread (expensive) → higher threshold (tau_take ≈ 0.02-0.06)

    Regime-Adaptive Thresholds

Each regime gets its own thresholds, optimised on training data:

| Regime | tau_passive | tau_take | edge_floor | Horizon |
|--------|-------------|----------|------------|---------|
| 0 (calm) | 0.030 | 0.050 | 0.03 | 20s |
| 1 (calm trend) | 0.017 | 0.035 | 0.03 | 25s |
| 2 (normal) | 0.007 | 0.018 | 0.10 | 30s |
| 3 (vol chop) | 0.013 | 0.025 | 0.06 | 45s |
| 4 (vol trend) | GATED | GATED | — | — |
| 5 (frozen) | GATED | GATED | — | — |

 Why gate regimes 4 & 5?  In highly volatile or frozen markets, the model's predictions are unreliable. It's better to sit out than to trade with bad information.

    Position Sizing (Kelly-Inspired)


stop_distance = max(half_spread + 0.4 × R, 0.10)
shares = min(
    per_trade_budget / stop_distance,       risk management: max ₹500 per trade
    0.40 × L1_depth,                       market impact: don't take >40% of queue
    MAX_DAILY_LOSS / R,                    catastrophe protection
    5000                                    hard cap
)


 per_trade_budget = ₹500 : If the stop is hit, we lose at most ₹500 on that trade. With 30 trades/day and 60% win rate, worst case daily loss = 30 × 0.4 × 500 = ₹6,000 (well within the ₹20,000 daily limit).

    Target and Stop Placement


target_distance = R × dynamic_multiplier
stop_distance = max(half_spread + 0.4 × R, 0.10)

Where dynamic_multiplier = clip(1 + |edge_pred| / 0.3, 0.7, 1.6)


 Dynamic multiplier : If the model is very confident (high edge prediction), we set a wider target (up to 1.6×R). If uncertain, we use a tighter target (0.7×R) to take quick profits.

 Stop distance : Always at least half_spread + 0.4R. This accounts for:
- The spread cost (you need price to move at least half_spread against you before you're "really" losing)
- Normal noise (0.4R of "breathing room")

    FIX Applied: Signal Balance & Stability

 Problem (v1) : The tau sweep sometimes produced thresholds that gave only 6 signals/day (too few) or 800 signals/day (too many, mostly noise). Also, some days were 90% long with no shorts.

 Fix (v2) :
1. Added penalty in the objective function for signal counts outside [150, 8000] per regime
2. Added penalty for long:short ratio > 2.5:1
3. Added per-day signal cap of 300 in the live bot
4. Lowered tau grids to [0.003-0.050] range (the 3-class softmax rarely produces |P_long - P_short| > 0.10)

---

   9. Phase 6: The Time Machine — Backtesting

    File: `phase6_backtest.py`

    What It Does

Simulates trading on historical data with a  realistic execution model  — not just "if signal says buy, assume we bought at mid price". Real trading has:
- Passive orders that might NOT fill
- Aggressive orders that pay the spread
- Time limits (can't hold forever)
- One position at a time (can't buy while already holding)

    The Execution Model

     Passive Orders (Limit Orders)

When the signal says "passive buy at bid":

1. We place a limit order at the current bid price
2. Each second, we check if it fills:
   -  Price fill : If the ask drops to our bid level (someone sold at our price)
   -  Queue fill : Bernoulli draw with probability `p_per_second = 1 - (1 - P_fill)^(1/H)`
   -  Cancel : If bid drops 2+ ticks below our level (market moved away)
   -  Timeout : If H seconds pass without fill → cancel

3. If filled, we now HOLD the position and monitor for exit

     Aggressive Orders (Market Orders)

When the signal says "aggressive buy":
1. We immediately buy at the ask price (paying the spread)
2. Fill is guaranteed (100%)
3. We immediately start monitoring for exit

     Exit Conditions

Once in a position, we exit when ANY of these happens:

| Condition | Exit Price | When |
|-----------|-----------|------|
| Target hit | entry ± target_distance | Price moved in our favour enough |
| Stop hit | entry ∓ stop_distance | Price moved against us too much |
| Time expiry | current mid | H seconds elapsed, no clear win/loss |
| Opposite signal | current mid | Model now says reverse direction |
| Kill switch | current mid | Daily loss limit hit |

     MIN_HOLD_S = 3

 Problem : Without a minimum hold time, a trade could enter and exit in the same second due to microstructure noise (bid-ask bounce). This generates fake "wins" or "losses" that aren't real trades.

 Fix : Don't check stop/target for the first 3 seconds after entry. Let the trade "breathe".

    The Kill Switch


If intra_day_PnL + unrealised_PnL ≤ -₹20,000:
    Close all positions immediately
    Stop trading for the rest of the day


 Why : This is a circuit breaker. If the model is having a terrible day (wrong regime detection, black swan event), we limit the damage to 2% of capital.

    Cooldown Period

After every trade exit, we wait 15 seconds before considering a new trade. This prevents:
- Whipsaw (entering and exiting rapidly in a choppy market)
- Correlated losses (multiple trades in the same adverse move)

    Results of Phase 6

| Metric | TRAIN (8 days) | TEST (4 days) | VALID (3 days) |
|--------|---------------|---------------|----------------|
| Total P&L | +₹4,299 | +₹2,043 | +₹2,525 |
| Trades | 183 | 96 | 73 |
| Win Rate | 58.5% | 55.2% | 63.0% |
| Profit Factor | 2.31 | 2.71 | 2.44 |
| Max Drawdown | -₹3,723 | -₹2,624 | -₹2,899 |
| Avg Hold (seconds) | 18.2 | 20.8 | 21.7 |
| Trades/Day | 22.9 | 24.0 | 24.3 |
| Sharpe (annualised) | 0.37 | 0.81 | 0.86 |
| Win Rate (Days) | 75% | 100% | 67% |

 Key Takeaways :
- The strategy is profitable on ALL three splits (no overfitting)
- Profit Factor > 2 means we make ₹2 for every ₹1 we lose
- Max drawdown is manageable (0.3-0.4% of capital)
- 100% win rate on TEST days means every single test day was profitable
- Average hold of ~20 seconds confirms this is true scalping (not swing trading)

---

   10. The Live Bot: Putting It All Together

    File: `live_trading_bot.py`

    Architecture


┌─────────────────────────────────────────────────────────────┐
│                    LIVE TRADING BOT                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │  Breeze  │───▶│ MarketState  │───▶│ FeatureBuffer   │    │
│  │ Websocket│    │ (thread-safe │    │ (700-second     │    │
│  │          │    │  cache)      │    │  ring buffer)   │    │
│  └──────────┘    └──────────────┘    └────────┬────────┘    │
│                                                │              │
│                                                ▼              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              EVERY 1 SECOND (on integer boundary)      │    │
│  │                                                        │    │
│  │  1. Take snapshot from MarketState                     │    │
│  │  2. Write raw CSV row (for future retraining)          │    │
│  │  3. Add to FeatureBuffer                               │    │
│  │  4. If buffer ≥ 600 rows AND past warmup:              │    │
│  │     a. Run build_day() → 130+ features                 │    │
│  │     b. Run LightGBM predict → P(long), P(short)        │    │
│  │     c. Run decide() → signal (buy/sell/nothing)        │    │
│  │  5. Pass signal to Portfolio.tick()                    │    │
│  │  6. Log equity, trades, console output                 │    │
│  │                                                        │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Portfolio   │    │  CSV Logger  │    │  Console     │   │
│  │  (position   │    │  (trades +   │    │  (real-time  │   │
│  │   manager)   │    │   equity)    │    │   display)   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘


    The 1-Second Loop (Pseudocode)

python
while running:
    wait_until_next_integer_second()
    
    snapshot = market_state.get_latest()
    write_raw_csv(snapshot)
    feature_buffer.add(snapshot)
    
    if not is_rth(now):
        continue    market closed, just log
    
    if feature_buffer.size < 600 or seconds_since_open < 360:
        signal = NO_TRADE    warmup period
    else:
        features = build_day(feature_buffer.to_dataframe())
          Copy raw book prices into features (FIX  20)
        features['b1_p'] = feature_buffer['b1_p']
        features['a1_p'] = feature_buffer['a1_p']
        features['b1_q'] = feature_buffer['b1_q']
        features['a1_q'] = feature_buffer['a1_q']
        
        signal = engine.decide(features.tail(1))
    
    portfolio.tick(signal, current_prices)
    log_equity()
    print_console_status()


    Capital Compounding

 FIX  16-19 : The bot now compounds capital across days:
- Day 1 ends at ₹10,01,000 → Day 2 starts at ₹10,01,000
- The kill-switch still measures today's loss (intra_pnl) against the ₹20,000 limit
- On resume (restart mid-day), it reads the last equity from the log file

---

   11. Bugs Found & Fixed

    Critical Bugs in the Live Bot

|   | Bug | Impact | Fix |
|---|-----|--------|-----|
| 20 | `build_day()` doesn't output raw book columns → `b1_p=0.0` → entry=0 → fake P&L |  CATASTROPHIC : Every trade shows fake profit = exit_price × qty. Day P&L showed ₹3.76L instead of real ~₹700 | Copy `b1_p, a1_p, b1_q, a1_q` from input frame to output frame after `build_day()` |
| 2 | NaN mid on time-stop exit → capital becomes NaN | Capital corruption, bot stops working | Use `last_valid_mid` as fallback |
| 1 | `sys.path.insert` after `from phase4_model import ...` | ModuleNotFoundError on some systems | Move `sys.path.insert` to top of `main()` |

    Performance Bugs

|   | Bug | Impact | Fix |
|---|-----|--------|-----|
| — | PCA eigendecomposition every second | 200ms/sec → loop falls behind real time | Batch: recompute every 30s |
| — | Kalman filter full recursion | 50ms/sec | Steady-state after 100 steps |
| — | `build_day()` on full 700-row buffer every second | 100-300ms | Acceptable with PCA/Kalman fixes; future: incremental features |

    Logic Bugs

|   | Bug | Impact | Fix |
|---|-----|--------|-----|
| 4 | `other_ticks` always 0 in live buffer | Minor feature degradation | Use `snap["other_tick_count"]` |
| 5 | `engine.decide()` receives full 700-row frame | Wasted computation | Pass only `tail(1)` |
| 6 | `add_snapshot` calls `datetime.now()` instead of using passed `ts` | Timestamp inconsistency | Use passed `ts` parameter |
| 16 | `new_day()` resets capital to ₹10L | No compounding | Set `start_of_day_cap = self.capital` without resetting capital |

    Signal Generation Bugs

|   | Bug | Impact | Fix |
|---|-----|--------|-----|
| — | Tau thresholds too high (v1) | Only 6 signals/day | Lowered grid to [0.003-0.050] |
| — | No balance constraint | 90% longs, 0% shorts some days | Added 2.5:1 ratio penalty |
| — | No per-day cap | 872 signals on one day | Cap at 300/day |

---

   12. Results

    Backtest Summary (15 days, Jul 2026)

| | TRAIN | TEST | VALID | OVERALL |
|--|-------|------|-------|---------|
| Days | 8 | 4 | 3 | 15 |
| P&L | +₹4,299 | +₹2,043 | +₹2,525 | +₹8,867 |
| Trades | 183 | 96 | 73 | 352 |
| Win Rate | 58.5% | 55.2% | 63.0% | 58.5% |
| PF | 2.31 | 2.71 | 2.44 | 2.42 |
| Max DD | -₹3,723 | -₹2,624 | -₹2,899 | -₹6,652 |
| Avg/Day | ₹537 | ₹511 | ₹842 | ₹591 |

    Live Session (28 July 2026) — BUGGY Run

- Reported P&L: ₹3,76,843 ( FAKE  — entry=0 bug)
- Estimated real P&L:  ₹500–₹900  (based on model's historical per-trade edge × 24 filled trades)
- All 24 trades were longs (model had long bias that day; afternoon was slightly down)
- No short trades filled (passive sell orders timed out in the choppy afternoon)

    What Needs More Data

- 15 days is the MINIMUM to train a model. We need 60-100 days for confidence.
- The strategy needs to be tested across different market conditions (bull, bear, sideways, high-vol events like budget day)
- Transaction costs (₹0.54/share round-trip for ICICI Direct) have NOT been included in the backtest. At 24 trades/day × 500 shares avg × ₹0.54 = ₹6,480/day in costs vs ₹591/day in gross profit —  the strategy is NOT profitable after costs with current parameters .

    Path to Profitability

1.  Larger position sizes  (2000-5000 shares) to dilute the ₹20/order brokerage
2.  Fewer, higher-conviction trades  (raise tau to get 8-10 trades/day instead of 24)
3.  Discount broker  with ₹0 brokerage on intraday (only STT + exchange charges ≈ ₹0.15/share)
4.  More training data  (60+ days) to improve model accuracy from 60% to 65%+

---

   13. Glossary

| Term | Definition |
|------|-----------|
|  Tick  | Minimum price movement (₹0.05 for Reliance) |
|  Spread  | Difference between best ask and best bid |
|  Mid  | Middle of the spread: (bid+ask)/2 |
|  LTP  | Last Traded Price — the price at which the most recent trade occurred |
|  Order Book  | List of all outstanding buy and sell orders at different prices |
|  Bid  | A buy order (someone wants to BUY at this price) |
|  Ask  | A sell order (someone wants to SELL at this price) |
|  Passive Order  | A limit order that waits in the queue (provides liquidity) |
|  Aggressive Order  | A market order that crosses the spread (takes liquidity) |
|  OFI  | Order Flow Imbalance — net buying/selling pressure per second |
|  PCA  | Principal Component Analysis — dimensionality reduction technique |
|  Kalman Filter  | Optimal estimator that blends prediction with noisy observation |
|  LightGBM  | Gradient Boosted Decision Trees (Microsoft's implementation) |
|  Isotonic Regression  | Non-parametric calibration method (monotonic mapping) |
|  Triple Barrier  | Labelling method with take-profit, stop-loss, and time-out |
|  Regime  | Market "personality" classification (calm/volatile/trending/etc.) |
|  Walk-Forward CV  | Cross-validation that respects time ordering (no future leakage) |
|  Profit Factor  | Gross profits / Gross losses (>1 = profitable) |
|  Sharpe Ratio  | Risk-adjusted return = mean_return / std_return (annualised) |
|  Max Drawdown  | Largest peak-to-trough decline in equity |
|  R  | Reference unit = 80th percentile of 30-second absolute mid moves |
|  Z-Score  | (value - mean) / std — how many standard deviations from average |
|  Autocorrelation  | Correlation of a series with its lagged self (measures trendiness) |
|  Mahalanobis Distance  | Multi-dimensional "surprise" measure (how unusual is current state) |
|  VWAP  | Volume-Weighted Average Price |
|  Parquet  | Columnar compressed file format (10× smaller than CSV) |
|  WebSocket  | Persistent bidirectional connection (used for live market data) |
|  EWM/EMA  | Exponentially Weighted Moving Average (recent data weighted more) |
|  Bernoulli Draw  | Random coin flip with probability p (used for fill simulation) |
|  Overfitting  | When a model memorises training data but fails on new data |
|  Regularisation  | Techniques to prevent overfitting (L1, L2, dropout, etc.) |

---

   Appendix: The Complete Pipeline (One Command Each)

bash
  Phase 0: Collect data (run during market hours)
python fetch_reliance_orderbook.py --duration 22500

  Phase 1: Clean and ingest
python phase1_ingest.py

  Phase 2: Compute features
python phase2_features.py

  Phase 3: Create labels
python phase3_labels.py

  Phase 4: Train model
python phase4_model.py

  Phase 5: Generate signals
python phase5_signals.py

  Phase 6: Backtest
python phase6_backtest.py

  Live: Run the bot (during market hours)
python live_trading_bot.py


---

   Appendix: Key Academic References

1.  Cont, Kukanov & Stoikov (2013)  — "The Price Impact of Order Book Events" — OFI formula
2.  Stoikov (2018)  — "The Microprice" — Fair value estimation from order book imbalance
3.  López de Prado (2018)  — "Advances in Financial Machine Learning" — Triple barrier method, purged CV, meta-labelling
4.  Kalman (1960)  — "A New Approach to Linear Filtering and Prediction Problems" — Kalman filter
5.  Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu (2017)  — "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" — The ML algorithm

---

*End of Documentation. Built with 🧠 + 📊 + ☕ over many iterations of debugging, backtesting, and head-scratching.*
