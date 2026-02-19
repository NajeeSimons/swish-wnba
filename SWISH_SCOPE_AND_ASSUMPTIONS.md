# SWISH: Scope and Assumptions (for reviewers)

## What SWISH measures

- **Shotmaking from the field**: expected points per **field-goal attempt** (FGA) given shot context (distance, 2 vs 3, clock, clutch, corner, pull-up, etc.), then actual vs expected → SWISH.
- **Not true shooting**: Free throws are **not** included. So we are effectively working with an **eFG%-style** expected value (points per FGA), not TS%. SWISH here is explicitly FGA-only. A future version could add FTA/FTM and a TS-style metric.

## Fouled-on-shot handling (Heat feedback)

- I keep only **(not fouled)** OR **(fouled and made)**. I drop **fouled and missed** when the data has a foul-on-shot flag so I don’t penalize players for getting fouled.

## Data

- **Source**: shufinskiy/nba_data WNBA shotdetail (one row per FGA).
- **Model**: Logistic regression on make probability; expected pts/shot = P(make) × shot value (2 or 3).

## Quick answers for reviewers

- **“Is this eFG or TS?”**  
  eFG-style: FGA only, no free throws. If I show TS%, I will add it as a separate filter/metric.

- **“Do you count fouled-and-missed as misses?”**  
  No. When I have a foul flag I filter to (not fouled) OR (fouled and made).

- **“Can you add FTs later?”**  
  Yes! Will do.
