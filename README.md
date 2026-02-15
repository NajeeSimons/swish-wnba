# SWISH — WNBA Contextual Shotmaking

**SWISH** = *Shotmaking With Intelligent Shot Handling* — a metric for who scores more (or less) than expected given shot difficulty (distance, 2 vs 3, clock, context).

- **SWISH score** = Actual points per shot − Expected points per shot (per player).  
- **+0.15** ≈ 15 extra points per 100 shots above expectation (not a shooting %).

---

## How to run

1. **Clone the repo** and open the folder in R/RStudio.

2. **Install R packages** (if needed):
   ```r
   install.packages(c("dplyr", "ggplot2", "readr", "caret", "pROC"))
   ```

3. **Get shot data** (e.g. [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data)) and put the WNBA shot files in a `wnba_data/` folder in this directory.

4. **Run the pipeline:**
   ```r
   setwd("path/to/swish-wnba")   # or Session → Set Working Directory
   source("run_swish.R")
   ```
   Outputs (CSVs + plots) go to `output/`.

---

## What the model uses (shot attributes)

- **Distance** (and distance²) — rim vs mid-range vs 3  
- **2 vs 3** — shot type  
- **Time** — shot clock remaining, game seconds remaining  
- **Context** — clutch (e.g. last 5 min close), corner 3, pull-up  

We don’t use angle or defender distance in this version; the public WNBA shot data we use doesn’t include them in the pipeline yet.

---

## Outputs
<img width="1200" height="900" alt="swish_3pt_top15_2024_2025" src="https://github.com/user-attachments/assets/452c4d1e-bb2b-4828-bdd3-85c2f77aacf5" />
<img width="1200" height="900" alt="swish_top15_overall_2024_2025" src="https://github.com/user-attachments/assets/d0dd9482-31f6-4b4f-99c7-c5f1ed4e3b3c" />



---

## License

MIT (or add your preferred license).
