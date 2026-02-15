# =============================================================================
# SWISH: Shotmaking With Intelligent Shot Handling — Single Run Script
# =============================================================================
# Run this script to load data, fit the model, compute SWISH, save CSVs & plots.
# Requires: wnba_shot_making_model.R in the same folder (or set path_to_model below).
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIG — change these and re-run
# -----------------------------------------------------------------------------
SEASONS       <- 2024:2025          # e.g. 2025 only, or 2024:2025 for 2-year
SEASONTYPE    <- "both"             # "rg" = regular, "po" = playoffs, "both"
DATA_DIR      <- "wnba_data"
MIN_SHOTS     <- 400                # min total shots for overall SWISH (200 for 1-yr)
MIN_3PT       <- 175                # min 3PA for 3pt SWISH (100 for 1-yr)
OUTPUT_DIR    <- "output"           # folder for CSVs and plots
path_to_model <- "wnba_shot_making_model.R"

# Season label for filenames (e.g. "2025" or "2024_2025")
season_label <- if (length(SEASONS) == 1) as.character(SEASONS) else paste0(min(SEASONS), "_", max(SEASONS))

# -----------------------------------------------------------------------------
# PACKAGES
# -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(readr)
library(caret)
library(pROC)

# -----------------------------------------------------------------------------
# LOAD MODEL HELPERS (load_wnba_shots, create_shot_features)
# -----------------------------------------------------------------------------
if (!file.exists(path_to_model)) stop("Place wnba_shot_making_model.R in this folder or set path_to_model.")
source(path_to_model)

# -----------------------------------------------------------------------------
# DATA: load + features + points per shot
# -----------------------------------------------------------------------------
shots_raw <- load_wnba_shots(data_dir = DATA_DIR, seasons = SEASONS, seasontype = SEASONTYPE)
shots     <- create_shot_features(shots_raw)
shots     <- shots %>%
  mutate(
    shot_value = ifelse(is_three, 3, 2),
    actual_points_per_shot = ifelse(shot_made == 1, shot_value, 0)
  )

# -----------------------------------------------------------------------------
# TRAIN / TEST + MODEL
# -----------------------------------------------------------------------------
set.seed(42)
train_i   <- createDataPartition(shots$shot_made, p = 0.8, list = FALSE)
train_data <- shots[train_i, ]
test_data  <- shots[-train_i, ]

model <- glm(
  shot_made ~ shot_distance + I(shot_distance^2)
    + is_three + shot_clock_remaining + game_seconds_remaining
    + is_clutch + is_corner_3 + is_pull_up,
  data = train_data,
  family = binomial(link = "logit")
)

# AUC
test_data$expected_make_prob <- predict(model, newdata = test_data, type = "response")
auc_score <- as.numeric(pROC::auc(pROC::roc(test_data$shot_made, test_data$expected_make_prob)))
cat("Model AUC:", round(auc_score, 3), "\n")

# -----------------------------------------------------------------------------
# EXPECTED VALUES ON ALL SHOTS + SWISH TABLES
# -----------------------------------------------------------------------------
shots$expected_make_prob     <- predict(model, newdata = shots, type = "response")
shots$expected_points_per_shot <- shots$expected_make_prob * shots$shot_value

player_swish <- shots %>%
  group_by(player_name) %>%
  summarize(
    total_shots = n(),
    actual_rts = mean(actual_points_per_shot),
    expected_rts = mean(expected_points_per_shot),
    swish_score = actual_rts - expected_rts,
    .groups = "drop"
  ) %>%
  filter(total_shots >= MIN_SHOTS) %>%
  arrange(desc(swish_score))

shots_3pt <- shots %>% filter(is_three == TRUE)
shots_3pt$expected_make_prob_3pt     <- predict(model, newdata = shots_3pt, type = "response")
shots_3pt$expected_points_per_shot_3pt <- shots_3pt$expected_make_prob_3pt * 3
shots_3pt$actual_points_per_shot_3pt <- ifelse(shots_3pt$shot_made == 1, 3, 0)

player_swish_3pt <- shots_3pt %>%
  group_by(player_name) %>%
  summarize(
    total_3pt = n(),
    actual_rts_3pt = mean(actual_points_per_shot_3pt),
    expected_rts_3pt = mean(expected_points_per_shot_3pt),
    swish_score_3pt = actual_rts_3pt - expected_rts_3pt,
    .groups = "drop"
  ) %>%
  filter(total_3pt >= MIN_3PT) %>%
  arrange(desc(swish_score_3pt))

# -----------------------------------------------------------------------------
# OUTPUT FOLDER + SAVE CSVs
# -----------------------------------------------------------------------------
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

write_csv(player_swish,    file.path(OUTPUT_DIR, paste0("player_swish_overall_", season_label, ".csv")))
write_csv(player_swish_3pt, file.path(OUTPUT_DIR, paste0("player_swish_3pt_", season_label, ".csv")))
cat("Saved CSVs to", OUTPUT_DIR, "\n")

# -----------------------------------------------------------------------------
# TIERED COLORS (easier for viewers: Elite / Above avg / Average / Below / Struggling)
# -----------------------------------------------------------------------------
tier_cuts <- c(-Inf, -0.15, -0.05, 0.05, 0.15, Inf)
tier_labs <- c("Struggling", "Below avg", "Average", "Above avg", "Elite")

player_swish     <- player_swish     %>% mutate(swish_tier = cut(swish_score,     breaks = tier_cuts, labels = tier_labs))
player_swish_3pt <- player_swish_3pt %>% mutate(swish_tier = cut(swish_score_3pt, breaks = tier_cuts, labels = tier_labs))

# Distinct colors so Elite vs Above avg are easy to tell apart (not two similar greens)
tier_colors <- c("Struggling" = "#b71c1c", "Below avg" = "#e57373", "Average" = "#9e9e9e",
                 "Above avg" = "#00897b", "Elite" = "#1b5e20")

# -----------------------------------------------------------------------------
# PLOTS: overall (actual vs expected + top 15) + 3pt (actual vs expected + top 15)
# -----------------------------------------------------------------------------
# Overall — actual vs expected
p1 <- ggplot(player_swish, aes(x = expected_rts, y = actual_rts, label = player_name, color = swish_tier)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 3) +
  geom_text(hjust = -0.15, size = 2.5, check_overlap = TRUE, show.legend = FALSE) +
  scale_color_manual(values = tier_colors, drop = FALSE) +
  labs(
    title = "SWISH: Actual vs Expected Points Per Shot",
    subtitle = "Above the line = scores more than expected. +0.10 = 10 extra pts per 100 shots (not a %).",
    x = "Expected points per shot", y = "Actual points per shot", color = "Tier",
    caption = paste0(season_label, " · Min ", MIN_SHOTS, " shots")
  ) +
  theme_minimal() +
  theme(plot.subtitle = element_text(size = 10, color = "gray30"))

# One-line "how to read" — so you don't have to over-explain
swish_how_to_read <- "Higher bar = more points per shot above expectation. +0.10 = 10 extra points per 100 shots (not a %)."

# Overall — top 15 bar
p2 <- player_swish %>% head(15) %>%
  ggplot(aes(x = reorder(player_name, swish_score), y = swish_score, fill = swish_tier)) +
  geom_col() +
  scale_fill_manual(values = tier_colors, drop = FALSE) +
  coord_flip() +
  labs(
    title = "Top 15 SWISH (Overall Shotmaking)",
    subtitle = swish_how_to_read,
    x = NULL, y = "SWISH (pts/shot above expected)", fill = "Tier",
    caption = paste0(season_label, " · Min ", MIN_SHOTS, " shots")
  ) +
  theme_minimal() +
  theme(
    plot.subtitle = element_text(size = 10, color = "gray30", margin = margin(b = 10)),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  )

# 3pt — actual vs expected
p3 <- ggplot(player_swish_3pt, aes(x = expected_rts_3pt, y = actual_rts_3pt, label = player_name, color = swish_tier)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 3) +
  geom_text(hjust = -0.15, size = 2.5, check_overlap = TRUE, show.legend = FALSE) +
  scale_color_manual(values = tier_colors, drop = FALSE) +
  labs(
    title = "SWISH: 3-Point Shotmaking (Actual vs Expected)",
    subtitle = "Above the line = scores more than expected on 3s. +0.10 = 10 extra pts per 100 3s (not a %).",
    x = "Expected pts/shot (3pt)", y = "Actual pts/shot (3pt)", color = "Tier",
    caption = paste0(season_label, " · Min ", MIN_3PT, " 3PA")
  ) +
  theme_minimal() +
  theme(plot.subtitle = element_text(size = 10, color = "gray30"))

# 3pt — top 15 bar
p4 <- player_swish_3pt %>% head(15) %>%
  ggplot(aes(x = reorder(player_name, swish_score_3pt), y = swish_score_3pt, fill = swish_tier)) +
  geom_col() +
  scale_fill_manual(values = tier_colors, drop = FALSE) +
  coord_flip() +
  labs(
    title = "Top 15 SWISH — 3-Point Shotmaking",
    subtitle = swish_how_to_read,
    x = NULL, y = "SWISH 3pt (pts/shot above expected)", fill = "Tier",
    caption = paste0(season_label, " · Min ", MIN_3PT, " 3PA")
  ) +
  theme_minimal() +
  theme(
    plot.subtitle = element_text(size = 10, color = "gray30", margin = margin(b = 10)),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  )

ggsave(file.path(OUTPUT_DIR, paste0("swish_actual_vs_expected_", season_label, ".png")),       p1, width = 10, height = 8, dpi = 150)
ggsave(file.path(OUTPUT_DIR, paste0("swish_top15_overall_", season_label, ".png")),             p2, width = 8,  height = 6, dpi = 150)
ggsave(file.path(OUTPUT_DIR, paste0("swish_3pt_actual_vs_expected_", season_label, ".png")),    p3, width = 10, height = 8, dpi = 150)
ggsave(file.path(OUTPUT_DIR, paste0("swish_3pt_top15_", season_label, ".png")),                 p4, width = 8,  height = 6, dpi = 150)
cat("Saved 4 plots to", OUTPUT_DIR, "\n")

# -----------------------------------------------------------------------------
# PRINT SUMMARY
# -----------------------------------------------------------------------------
print(player_swish)
print(player_swish_3pt)
cat("\nDone. Check", OUTPUT_DIR, "for CSVs and PNGs.\n")



# -----------------------------------------------------------------------------
# BOTTOM TIER: query + picture (underperformers vs expectation)
# -----------------------------------------------------------------------------
# Query: players in bottom tiers (Struggling or Below avg) — same min-shot qualifiers
bottom_tier_overall <- player_swish %>%
  filter(swish_tier %in% c("Struggling", "Below avg")) %>%
  arrange(swish_score)  # worst first

bottom_tier_3pt <- player_swish_3pt %>%
  filter(swish_tier %in% c("Struggling", "Below avg")) %>%
  arrange(swish_score_3pt)

# Optional: save bottom-tier tables for reference
write_csv(bottom_tier_overall, file.path(OUTPUT_DIR, paste0("swish_bottom_tier_overall_", season_label, ".csv")))
write_csv(bottom_tier_3pt,     file.path(OUTPUT_DIR, paste0("swish_bottom_tier_3pt_", season_label, ".csv")))

# Bottom 15 overall bar chart (worst at top; negative = below expectation)
p5 <- player_swish %>% tail(15) %>%
  ggplot(aes(x = reorder(player_name, swish_score), y = swish_score, fill = swish_tier)) +
  geom_col() +
  geom_hline(yintercept = 0, linewidth = 0.3, color = "gray40") +
  scale_fill_manual(values = tier_colors, drop = FALSE) +
  coord_flip() +
  labs(
    title = "Bottom 15 SWISH (Overall Shotmaking)",
    subtitle = "Lower bar = fewer points per shot than expected for their shot mix. Negative = underperforming.",
    x = NULL, y = "SWISH (pts/shot above expected)", fill = "Tier",
    caption = paste0(season_label, " · Min ", MIN_SHOTS, " shots")
  ) +
  theme_minimal() +
  theme(
    plot.subtitle = element_text(size = 10, color = "gray30", margin = margin(b = 10)),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  )

ggsave(file.path(OUTPUT_DIR, paste0("swish_actual_vs_expected_", season_label, ".png")),       p1, width = 10, height = 8, dpi = 150)
ggsave(file.path(OUTPUT_DIR, paste0("swish_top15_overall_", season_label, ".png")),             p2, width = 8,  height = 6, dpi = 150)
ggsave(file.path(OUTPUT_DIR, paste0("swish_3pt_actual_vs_expected_", season_label, ".png")),    p3, width = 10, height = 8, dpi = 150)
ggsave(file.path(OUTPUT_DIR, paste0("swish_3pt_top15_", season_label, ".png")),                 p4, width = 8,  height = 6, dpi = 150)
ggsave(file.path(OUTPUT_DIR, paste0("swish_bottom15_overall_", season_label, ".png")),          p5, width = 8,  height = 6, dpi = 150)
cat("Saved 5 plots to", OUTPUT_DIR, "\n")

# -----------------------------------------------------------------------------
# PRINT SUMMARY
# -----------------------------------------------------------------------------
print(player_swish)
print(player_swish_3pt)
cat("\nDone. Check", OUTPUT_DIR, "for CSVs and PNGs.\n")
