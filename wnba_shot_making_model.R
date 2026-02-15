# ============================================================================
# SWISH MODEL: WNBA Shotmaking Analysis
# ============================================================================
# 
# SWISH: Shotmaking With Intelligent Shot Handling
#
# PURPOSE: Build a shot quality model that predicts expected points per shot
#          (RTS% - Real True Shooting) based on contextual features, then 
#          identify players who exceed expectations (true shotmakers).
#
# CONCEPT: Instead of just looking at raw FG%, we use points per shot (RTS%)
#          because it accounts for 3-pointers being worth more than 2-pointers.
#          We want to understand:
#          - What's the expected points per shot for this shot given its context?
#          - Which players consistently score more points per shot than expected?
#
# WHY RTS% INSTEAD OF FG%?
# - A 3-pointer is worth 3 points, a 2-pointer is worth 2 points
# - Player A: 40% on 3s → 1.2 points per shot
# - Player B: 60% on 2s → 1.2 points per shot
# - Both have same value, even though Player B has higher FG%
#
# ANALOGY: Think of it like a credit score. A 3-pointer in transition 
#          with no defender nearby is like a "safe loan" - high expected 
#          value. A contested mid-range with 2 seconds on the clock
#          is like a "risky loan" - low expected value.
#
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: LOAD REQUIRED PACKAGES
# ----------------------------------------------------------------------------
# Why these packages?
# - dplyr: Data manipulation (filter, mutate, group_by, summarize)
# - tidyr: Data reshaping (pivot_wider, pivot_longer)
# - ggplot2: Visualization
# - readr: Fast CSV reading
# - stringr: String manipulation
# - lubridate: Date/time handling
# - caret: Machine learning tools (train/test split, cross-validation)
# - glmnet: Regularized regression (helps prevent overfitting)
# - pROC: ROC curves for model evaluation

suppressPackageStartupMessages({
  library(dplyr)      # Data manipulation
  library(tidyr)      # Data reshaping
  library(ggplot2)    # Visualization
  library(readr)      # Fast CSV reading
  library(stringr)    # String operations
  library(lubridate)  # Date/time handling
  library(caret)      # ML tools (train/test split, CV)
  library(glmnet)     # Regularized regression
  library(pROC)       # ROC curves
  library(purrr)      # Functional programming
})

# ----------------------------------------------------------------------------
# STEP 2: DATA LOADING FUNCTION
# ----------------------------------------------------------------------------
# This function downloads WNBA shot data from the shufinskiy/nba_data repo.
# We'll use shotdetail data which contains individual shot attempts with
# location, time, game context, etc.

load_nba_data <- function(path = getwd(),
                          seasons = seq(1996, 2024),
                          data = c("datanba", "nbastats", "pbpstats", "shotdetail",
                                   "cdnnba", "nbastatsv3", "matchups"),
                          seasontype = 'rg',
                          league = 'nba',
                          in_memory = FALSE,
                          untar = FALSE){
  
  path <- normalizePath(path, mustWork = FALSE)
  
  if (length(data) > 1 & in_memory){
    stop("Parameter in_memory=True available only when loading a single data type")
  }
  
  # Build list of files we need based on seasons and data types
  if(seasontype == 'rg'){
    df <- expand.grid(data, seasons)
    need_data <- paste(df$Var1, df$Var2, sep = "_")
  } else if(seasontype == 'po'){
    df <- expand.grid(data, 'po', seasons)
    need_data <- paste(df$Var1, df$Var2, df$Var3, sep = "_")
  } else {
    df_rg <- expand.grid(data, seasons)
    df_po <- expand.grid(data, 'po', seasons)
    need_data <- c(paste(df_rg$Var1, df_rg$Var2, sep = "_"), 
                   paste(df_po$Var1, df_po$Var2, df_po$Var3, sep = "_"))
  }
  
  # Add 'wnba_' prefix if we're loading WNBA data
  if(tolower(league) == 'wnba'){
    need_data <- sapply(need_data, function(x){paste0('wnba_', x)}, USE.NAMES = FALSE)
  }
  
  # Download the index file that tells us where each dataset is stored
  temp <- tempfile()
  download.file("https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt", temp)
  f <- readLines(temp)
  unlink(temp)
  
  # Parse the index: format is "filename=url"
  v <- unlist(strsplit(f, "="))
  name_v <- v[seq(1, length(v), 2)]      # File names
  element_v <- v[seq(2, length(v), 2)]     # URLs
  
  # Find which files we need
  need_name <- name_v[which(name_v %in% need_data)]
  need_element <- element_v[which(name_v %in% need_data)]
  
  # Create directory if it doesn't exist
  if(!dir.exists(path)){
    dir.create(path)
  }
  
  # Download files
  if (in_memory){
    df <- data.frame()
  }
  for(i in seq_along(need_element)){
    if (in_memory){
      # Load directly into memory (good for small datasets)
      temp_file <- tempfile(fileext = ".tar.xz")
      download.file(need_element[i], destfile = temp_file, mode = "wb")
      temp_dir <- tempdir()
      untar(temp_file, exdir = temp_dir, files = paste0(
        gsub(".tar.xz", "", basename(need_element[i])),".csv"
      ))
      csv_file <- list.files(temp_dir, pattern = "\\.csv$", full.names = TRUE)
      if (length(csv_file) > 0) {
        tmp_df <- read.csv(csv_file)
        df <- rbind(df, tmp_df)
      }
      unlink(temp_file)
      unlink(csv_file)
      unlink(temp_dir)
    } else {
      # Save to disk (good for large datasets)
      destfile <- paste0(path, '/', need_name[i], ".tar.xz")
      download.file(need_element[i], destfile = destfile)
      if(untar){
        untar(destfile, paste0(need_name[i], ".csv"), exdir = path)
        unlink(destfile)
      }
    }
  }
  if (in_memory){
    return(df)
  }
}

# Helper function to read multiple CSV files
read_many_csv <- function(path, pattern){
  files <- list.files(path, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)
  purrr::map_dfr(files, function(fp){
    suppressWarnings(suppressMessages(readr::read_csv(fp, show_col_types = FALSE))) %>%
      mutate(source_file = basename(fp))
  })
}

# Normalize column names to lowercase (different sources use different conventions)
normalize_names <- function(df){
  names(df) <- tolower(names(df))
  df
}

# ----------------------------------------------------------------------------
# STEP 3: LOAD AND PREPARE SHOT DATA
# ----------------------------------------------------------------------------
# We'll load WNBA shot detail data. This contains every shot attempt with
# information like:
# - Shot location (x, y coordinates)
# - Shot distance
# - Shot type (2PT, 3PT)
# - Game time
# - Score
# - Player name
# - Made/missed

load_wnba_shots <- function(data_dir = "wnba_data", 
                            seasons = 2024:2025,
                            seasontype = 'rg'){
  
  cat("Loading WNBA shot data...\n")
  
  # Download data if not already present
  load_nba_data(path = data_dir,
                seasons = seasons,
                data = 'shotdetail',
                seasontype = seasontype,
                league = 'wnba',
                in_memory = FALSE,
                untar = TRUE)
  
  # Read all shot detail CSV files
  shots <- read_many_csv(data_dir, pattern = '^wnba_shotdetail_.*\\.csv$')
  
  if (is.null(shots) || nrow(shots) == 0) {
    stop("No shot data found. Check data_dir and seasons.")
  }
  
  # Normalize column names
  shots <- normalize_names(shots)
  
  cat(sprintf("Loaded %d shots\n", nrow(shots)))
  
  return(shots)
}

# ----------------------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# ----------------------------------------------------------------------------
# This is the MOST IMPORTANT step. We're creating features (variables)
# that help predict shot success. Think of features as "clues" that help
# our model understand the context of each shot.
#
# GOOD FEATURES TO CREATE:
# 1. Shot distance (closer = easier)
# 2. Shot angle (corner 3s vs above-break)
# 3. Time on shot clock (less time = harder)
# 4. Game time remaining (clutch situations)
# 5. Score differential (pressure situations)
# 6. Shot type (2PT vs 3PT)
# 7. Shot zone (restricted area, paint, mid-range, 3PT)
# 8. Whether shot was off a turnover (transition)
# 9. Whether shot was a putback (offensive rebound)
# 10. Quarter (late game pressure)

create_shot_features <- function(shots){
  
  cat("Creating shot features...\n")
  
  # First, let's identify key columns (different sources use different names)
  # We'll use a helper function to find columns by common name patterns
  
  find_col <- function(df, candidates){
    cand <- intersect(candidates, names(df))
    if (length(cand) == 0) return(NA_character_)
    cand[1]
  }
  
  # Identify important columns
  shot_made_col <- find_col(shots, c('shot_made_flag', 'shot_result', 'event_type', 'made'))
  shot_dist_col <- find_col(shots, c('shot_distance', 'distance', 'shot_dist'))
  shot_x_col <- find_col(shots, c('loc_x', 'x', 'shot_x', 'coordinate_x'))
  shot_y_col <- find_col(shots, c('loc_y', 'y', 'shot_y', 'coordinate_y'))
  shot_type_col <- find_col(shots, c('shot_type', 'shot_type_desc', 'shot_zone_basic'))
  shot_value_col <- find_col(shots, c('shot_value', 'points', 'pts'))
  period_col <- find_col(shots, c('period', 'quarter', 'qtr'))
  minutes_col <- find_col(shots, c('minutes_remaining', 'min_remaining', 'time_remaining'))
  seconds_col <- find_col(shots, c('seconds_remaining', 'sec_remaining'))
  game_time_col <- find_col(shots, c('game_time', 'time', 'game_clock'))
  score_col <- find_col(shots, c('score', 'score_margin', 'point_diff'))
  player_col <- find_col(shots, c('player_name', 'shooter', 'player', 'name'))
  game_id_col <- find_col(shots, c('game_id', 'gameid', 'game'))
  date_col <- find_col(shots, c('game_date', 'date', 'gamedate'))
  # LASER-inspired features (for better shot difficulty assessment)
  defender_dist_col <- find_col(shots, c('defender_distance', 'closest_defender_dist', 'def_dist', 
                                         'defender_dist', 'closest_defender_distance', 'contest_distance'))
  dribble_type_col <- find_col(shots, c('dribble_type', 'shot_type_detail', 'action_type', 
                                        'dribbles', 'shot_action'))
  touch_time_col <- find_col(shots, c('touch_time', 'time_of_possession', 'touch_duration', 
                                      'time_with_ball', 'possession_time'))
  
  # Create the target variable: did the shot go in? (1 = made, 0 = missed)
  # This is what we're trying to predict
  if (!is.na(shot_made_col)) {
    shots <- shots %>%
      mutate(
        # TARGET VARIABLE: shot_made (1 if made, 0 if missed)
        shot_made = case_when(
          !is.na(shots[[shot_made_col]]) ~ 
            as.numeric(shots[[shot_made_col]] %in% c(1, '1', TRUE, 'TRUE', 'MADE', 'MAKE')),
          TRUE ~ NA_real_
        )
      )
  } else {
    shots$shot_made <- NA_real_
  }
  
  # If we couldn't create shot_made from shot_made_flag, try other columns
  if (all(is.na(shots$shot_made))) {
    if (!is.na(shot_type_col)) {
      # Try to infer from event_type or shot_result
      shots <- shots %>%
        mutate(
          shot_made = case_when(
            str_detect(toupper(as.character(shots[[shot_type_col]])), "MADE|MAKE") ~ 1,
            str_detect(toupper(as.character(shots[[shot_type_col]])), "MISS") ~ 0,
            TRUE ~ NA_real_
          )
        )
    }
  }
  
  # Remove rows where we couldn't determine if shot was made
  shots <- shots %>% filter(!is.na(shot_made))
  
  cat(sprintf("After filtering: %d shots with known outcomes\n", nrow(shots)))
  
  # FEATURE 1: Shot Distance
  # Closer shots are easier. We'll use distance directly and also create
  # distance categories (0-3ft = layup, 3-10ft = close, 10-16ft = mid, etc.)
  if (!is.na(shot_dist_col)) {
    shots <- shots %>%
      mutate(
        shot_distance = as.numeric(shots[[shot_dist_col]]),
        # Distance categories
        dist_category = case_when(
          shot_distance <= 3 ~ "layup",
          shot_distance <= 10 ~ "close",
          shot_distance <= 16 ~ "mid_range",
          shot_distance <= 23.75 ~ "long_2",
          TRUE ~ "three_pt"
        )
      )
  } else {
    shots$shot_distance <- NA_real_
    shots$dist_category <- NA_character_
  }
  
  # FEATURE 2: Shot Angle (for 3-pointers)
  # Corner 3s are easier than above-the-break 3s. We can calculate angle
  # from basket center (0,0) to shot location.
  if (!is.na(shot_x_col) && !is.na(shot_y_col)) {
    shots <- shots %>%
      mutate(
        loc_x = as.numeric(shots[[shot_x_col]]),
        loc_y = as.numeric(shots[[shot_y_col]]),
        # Calculate angle from basket (0,0) to shot location
        # atan2 gives angle in radians, convert to degrees
        shot_angle_deg = atan2(loc_y, loc_x) * 180 / pi,
        # Normalize to 0-360
        shot_angle_deg = ifelse(shot_angle_deg < 0, shot_angle_deg + 360, shot_angle_deg),
        # Identify corner 3s (roughly 0-30 and 150-180 degrees from baseline)
        is_corner_3 = shot_distance > 22 & 
          (abs(shot_angle_deg) < 30 | abs(shot_angle_deg - 180) < 30 | 
           abs(shot_angle_deg - 360) < 30)
      )
  } else {
    shots$loc_x <- NA_real_
    shots$loc_y <- NA_real_
    shots$shot_angle_deg <- NA_real_
    shots$is_corner_3 <- FALSE
  }
  
  # FEATURE 3: Shot Type (2PT vs 3PT)
  # 3-pointers are harder but worth more. We need to identify them.
  if (!is.na(shot_value_col)) {
    shots <- shots %>%
      mutate(
        shot_value = as.numeric(shots[[shot_value_col]]),
        is_three = shot_value == 3
      )
  } else if (!is.na(shot_type_col)) {
    shots <- shots %>%
      mutate(
        is_three = str_detect(toupper(as.character(shots[[shot_type_col]])), "3|THREE")
      )
  } else {
    # Infer from distance (NBA 3PT line is 23.75 ft, WNBA is 22.15 ft)
    shots <- shots %>%
      mutate(
        is_three = ifelse(!is.na(shot_distance), shot_distance > 22, FALSE)
      )
  }
  
  # FEATURE 4: Shot Zone
  # Different zones have different expected FG%. Restricted area (rim) is
  # easiest, mid-range is hardest.
  if (!is.na(shot_type_col)) {
    zone_col <- find_col(shots, c('shot_zone_basic', 'zone', 'shot_zone'))
    if (!is.na(zone_col)) {
      shots <- shots %>%
        mutate(
          shot_zone = toupper(as.character(shots[[zone_col]])),
          is_restricted = str_detect(shot_zone, "RESTRICTED|RIM|PAINT"),
          is_paint = str_detect(shot_zone, "PAINT|IN THE PAINT"),
          is_mid_range = str_detect(shot_zone, "MID") & !is_three,
          is_above_break_3 = is_three & !is_corner_3
        )
    }
  }
  
  # FEATURE 5: Time on Shot Clock
  # Less time = more rushed = harder shot. We'll create a feature for
  # seconds remaining on shot clock.
  if (!is.na(minutes_col) && !is.na(seconds_col)) {
    shots <- shots %>%
      mutate(
        shot_clock_remaining = as.numeric(shots[[minutes_col]]) * 60 + 
          as.numeric(shots[[seconds_col]]),
        # Categorize: early (15+ sec), mid (7-15 sec), late (<7 sec)
        shot_clock_category = case_when(
          shot_clock_remaining >= 15 ~ "early",
          shot_clock_remaining >= 7 ~ "mid",
          shot_clock_remaining < 7 ~ "late",
          TRUE ~ "unknown"
        )
      )
  } else {
    shots$shot_clock_remaining <- NA_real_
    shots$shot_clock_category <- "unknown"
  }
  
  # FEATURE 6: Game Time Remaining
  # Late game situations are higher pressure. We'll calculate total seconds
  # remaining in game.
  if (!is.na(period_col) && !is.na(minutes_col) && !is.na(seconds_col)) {
    shots <- shots %>%
      mutate(
        period = as.numeric(shots[[period_col]]),
        minutes_remaining = as.numeric(shots[[minutes_col]]),
        seconds_remaining = as.numeric(shots[[seconds_col]]),
        # WNBA has 4 quarters of 10 minutes each
        game_seconds_remaining = (4 - period) * 600 + 
          (minutes_remaining * 60) + seconds_remaining,
        # Identify clutch time (last 5 minutes of 4th quarter or OT)
        is_clutch = (period >= 4 & game_seconds_remaining <= 300) | period > 4
      )
  } else {
    shots$game_seconds_remaining <- NA_real_
    shots$is_clutch <- FALSE
  }
  
  # FEATURE 7: Score Differential
  # Close games = more pressure. Blowouts = less pressure.
  # We'll need to parse score if it's in "TEAM_SCORE-OPP_SCORE" format
  if (!is.na(score_col)) {
    # Try to extract score differential
    score_str <- as.character(shots[[score_col]])
    # Helper function to parse score
    parse_score_margin <- function(s) {
      if (is.na(s)) return(NA_real_)
      if (str_detect(s, "-")) {
        parts <- strsplit(s, "-")[[1]]
        if (length(parts) == 2) {
          team_score <- as.numeric(parts[1])
          opp_score <- as.numeric(parts[2])
          if (!is.na(team_score) && !is.na(opp_score)) {
            return(team_score - opp_score)
          }
        }
      }
      # Try to convert directly to number
      num_val <- suppressWarnings(as.numeric(s))
      if (!is.na(num_val)) return(num_val)
      return(NA_real_)
    }
    
    shots <- shots %>%
      mutate(
        score_margin = map_dbl(score_str, parse_score_margin),
        # Categorize: close game (within 5), medium (5-15), blowout (>15)
        game_close = abs(score_margin) <= 5,
        game_blowout = abs(score_margin) > 15
      )
  } else {
    shots$score_margin <- NA_real_
    shots$game_close <- FALSE
    shots$game_blowout <- FALSE
  }
  
  # FEATURE 8: Defender Distance (LASER-inspired - MOST IMPORTANT for shot difficulty!)
  # Closer defender = harder shot. This is one of the strongest predictors of shot success.
  # LASER uses this as a key feature because contested shots are much harder than open ones.
  if (!is.na(defender_dist_col)) {
    shots <- shots %>%
      mutate(
        defender_distance = as.numeric(shots[[defender_dist_col]]),
        # Categorize defender distance (similar to NBA tracking data)
        is_wide_open = defender_distance >= 6,      # 6+ feet = wide open (easiest)
        is_open = defender_distance >= 4 & defender_distance < 6,  # 4-6 feet = open
        is_tight = defender_distance >= 2 & defender_distance < 4,  # 2-4 feet = tight (hard)
        is_very_tight = defender_distance < 2,      # <2 feet = very tight (hardest)
        # Overall contest level
        contest_level = case_when(
          is_wide_open ~ "wide_open",
          is_open ~ "open", 
          is_tight ~ "tight",
          is_very_tight ~ "very_tight",
          TRUE ~ "unknown"
        ),
        # Numeric contest score (higher = more contested = harder)
        contest_score = case_when(
          is_wide_open ~ 0,
          is_open ~ 1,
          is_tight ~ 2,
          is_very_tight ~ 3,
          TRUE ~ NA_real_
        )
      )
  } else {
    shots$defender_distance <- NA_real_
    shots$is_wide_open <- FALSE
    shots$is_open <- FALSE
    shots$is_tight <- FALSE
    shots$is_very_tight <- FALSE
    shots$contest_level <- "unknown"
    shots$contest_score <- NA_real_
  }
  
  # FEATURE 9: Dribble Type (LASER-inspired - Catch-and-shoot vs Pull-up)
  # Catch-and-shoot shots are significantly easier than pull-up shots.
  # This is especially important for 3-pointers - a catch-and-shoot 3 is much easier
  # than a pull-up 3, even at the same distance and defender distance.
  if (!is.na(dribble_type_col)) {
    shots <- shots %>%
      mutate(
        dribble_type_raw = toupper(as.character(shots[[dribble_type_col]])),
        is_catch_shoot = str_detect(dribble_type_raw, "CATCH|SPOT|ASSIST|NO_DRIBBLE|0_DRIBBLE"),
        is_pull_up = str_detect(dribble_type_raw, "PULL|DRIBBLE|JUMP") & !is_catch_shoot
      )
  } else {
    # Try to infer from event description
    event_desc_col <- find_col(shots, c('event_description', 'description', 'play', 'action'))
    if (!is.na(event_desc_col)) {
      shots <- shots %>%
        mutate(
          event_desc = toupper(as.character(shots[[event_desc_col]])),
          # Catch-and-shoot indicators
          is_catch_shoot = str_detect(event_desc, "CATCH|ASSIST|PASS|SPOT"),
          # Pull-up indicators (dribble before shot, not catch-and-shoot)
          is_pull_up = (str_detect(event_desc, "DRIBBLE|PULL|JUMP") | 
                       shot_distance > 10) & !is_catch_shoot
        )
    } else {
      # Default: assume catch-and-shoot for close shots, pull-up for far shots
      shots <- shots %>%
        mutate(
          is_catch_shoot = ifelse(!is.na(shot_distance), shot_distance <= 10, FALSE),
          is_pull_up = ifelse(!is.na(shot_distance), shot_distance > 10, FALSE)
        )
    }
  }
  
  # FEATURE 10: Touch Time (LASER-inspired - How long player has ball before shooting)
  # Quick shots (catch-and-shoot) are often easier than shots where player holds the ball.
  # This helps distinguish between different types of catch-and-shoot situations.
  if (!is.na(touch_time_col)) {
    shots <- shots %>%
      mutate(
        touch_time = as.numeric(shots[[touch_time_col]]),
        is_quick_shot = touch_time < 2,      # Less than 2 seconds = quick shot
        is_hold_shot = touch_time >= 2        # 2+ seconds = held the ball
      )
  } else {
    shots$touch_time <- NA_real_
    shots$is_quick_shot <- FALSE
    shots$is_hold_shot <- FALSE
  }
  
  # FEATURE 11: Context Features (Turnover, Putback, etc.)
  # These might be in event descriptions or separate columns
  # Check if event_desc already exists (from dribble type section)
  if (!"event_desc" %in% names(shots)) {
    event_desc_col <- find_col(shots, c('event_description', 'description', 'play', 'action'))
    if (!is.na(event_desc_col)) {
      shots <- shots %>%
        mutate(event_desc = toupper(as.character(shots[[event_desc_col]])))
    } else {
      shots$event_desc <- NA_character_
    }
  }
  
  # Now use event_desc for context features
  if ("event_desc" %in% names(shots) && !all(is.na(shots$event_desc))) {
    shots <- shots %>%
      mutate(
        is_off_turnover = str_detect(event_desc, "TURNOVER|STEAL"),
        is_putback = str_detect(event_desc, "PUTBACK|TIP|OFFENSIVE REBOUND"),
        is_fast_break = str_detect(event_desc, "FAST BREAK|BREAK")
      )
  } else {
    shots$is_off_turnover <- FALSE
    shots$is_putback <- FALSE
    shots$is_fast_break <- FALSE
  }
  
  # FEATURE 12: Interaction Terms (Prevent double-counting difficulty)
  # LASER methodology: Some features overlap (e.g., pull-ups are often more contested).
  # We create interaction terms to avoid double-counting the same difficulty.
  # Example: A contested pull-up shouldn't get penalized twice (once for pull-up, once for contest).
  shots <- shots %>%
    mutate(
      # Pull-up + tight contest (very difficult shot)
      pull_up_tight = is_pull_up & (is_tight | is_very_tight),
      # Catch-and-shoot + wide open (easiest shot)
      catch_wide_open = is_catch_shoot & is_wide_open,
      # Pull-up 3PT (harder than catch-and-shoot 3PT)
      pull_up_three = is_pull_up & is_three,
      # Contested 3PT (harder than open 3PT)
      contested_three = (is_tight | is_very_tight) & is_three
    )
  
  # FEATURE 9: Player and Game Identifiers
  # We'll keep these for later analysis (identifying best shotmakers)
  if (!is.na(player_col)) {
    shots <- shots %>%
      mutate(player_name = as.character(shots[[player_col]]))
  } else {
    shots$player_name <- "Unknown"
  }
  
  if (!is.na(game_id_col)) {
    shots <- shots %>%
      mutate(game_id = as.character(shots[[game_id_col]]))
  } else {
    shots$game_id <- paste0("game_", row_number())
  }
  
  # FEATURE 10: Season
  # Different seasons might have different league-wide shooting
  if (!is.na(date_col)) {
    shots <- shots %>%
      mutate(
        game_date = as.Date(shots[[date_col]]),
        season = year(game_date)
      )
  } else {
    # Try to extract from filename
    if ("source_file" %in% names(shots)) {
      shots <- shots %>%
        mutate(
          season = as.numeric(str_extract(source_file, "\\d{4}"))
        )
    } else {
      shots$season <- NA_real_
    }
  }
  
  cat("Feature engineering complete!\n")
  cat(sprintf("Created features for %d shots\n", nrow(shots)))
  
  return(shots)
}

# ----------------------------------------------------------------------------
# STEP 5: BUILD THE PREDICTIVE MODEL
# ----------------------------------------------------------------------------
# Now we'll build a model that predicts shot_made (0 or 1) based on our
# features. We'll use logistic regression, which is perfect for binary
# outcomes (made/missed).
#
# LOGISTIC REGRESSION EXPLANATION:
# - Regular regression predicts a continuous number (like points scored)
# - Logistic regression predicts a probability (0 to 1)
# - We use it to predict P(shot_made = 1 | features)
# - The model learns: "Given these features, what's the probability this shot goes in?"

build_shot_model <- function(shots, test_size = 0.2){
  
  cat("Building shot prediction model...\n")
  
  # Prepare features for modeling
  # We need to select numeric/categorical features and handle missing values
  # LASER-inspired: Include defender distance, dribble type, and interaction terms
  
  model_data <- shots %>%
    select(
      # Target variable
      shot_made,
      # Core features
      shot_distance,
      is_three,
      shot_clock_remaining,
      game_seconds_remaining,
      is_clutch,
      score_margin,
      game_close,
      is_off_turnover,
      is_putback,
      is_fast_break,
      is_corner_3,
      season,
      # LASER-inspired features (if available)
      defender_distance,
      contest_score,
      is_wide_open,
      is_open,
      is_tight,
      is_very_tight,
      is_catch_shoot,
      is_pull_up,
      touch_time,
      is_quick_shot,
      # Interaction terms (prevent double-counting)
      pull_up_tight,
      catch_wide_open,
      pull_up_three,
      contested_three
    ) %>%
    # Remove rows with too many missing values
    filter(!is.na(shot_made)) %>%
    # For missing numeric values, we'll use median imputation
    mutate(
      shot_distance = ifelse(is.na(shot_distance), median(shot_distance, na.rm = TRUE), shot_distance),
      shot_clock_remaining = ifelse(is.na(shot_clock_remaining), 12, shot_clock_remaining),  # Default to mid-clock
      game_seconds_remaining = ifelse(is.na(game_seconds_remaining), 1200, game_seconds_remaining),  # Default to mid-game
      score_margin = ifelse(is.na(score_margin), 0, score_margin),
      season = ifelse(is.na(season), 2024, season),
      # LASER-inspired features: Handle missing values
      defender_distance = ifelse(is.na(defender_distance), 
                                median(defender_distance, na.rm = TRUE), 
                                defender_distance),
      contest_score = ifelse(is.na(contest_score), 1, contest_score),  # Default to "open"
      touch_time = ifelse(is.na(touch_time), 2, touch_time),  # Default to 2 seconds
      # Set defaults for boolean features if missing
      is_wide_open = ifelse(is.na(is_wide_open), FALSE, is_wide_open),
      is_open = ifelse(is.na(is_open), TRUE, is_open),  # Default assumption: open shot
      is_tight = ifelse(is.na(is_tight), FALSE, is_tight),
      is_very_tight = ifelse(is.na(is_very_tight), FALSE, is_very_tight),
      is_catch_shoot = ifelse(is.na(is_catch_shoot), TRUE, is_catch_shoot),  # Default: catch-and-shoot
      is_pull_up = ifelse(is.na(is_pull_up), FALSE, is_pull_up),
      is_quick_shot = ifelse(is.na(is_quick_shot), TRUE, is_quick_shot)  # Default: quick shot
    )
  
  cat(sprintf("Model data: %d shots\n", nrow(model_data)))
  
  # Split into training and testing sets
  # We train on 80% of data, test on 20%
  set.seed(42)  # For reproducibility
  train_indices <- createDataPartition(model_data$shot_made, p = 1 - test_size, list = FALSE)
  train_data <- model_data[train_indices, ]
  test_data <- model_data[-train_indices, ]
  
  cat(sprintf("Training set: %d shots\n", nrow(train_data)))
  cat(sprintf("Test set: %d shots\n", nrow(test_data)))
  
  # Build logistic regression model
  # glm = Generalized Linear Model
  # family = binomial means we're doing logistic regression (binary outcome)
  # formula: shot_made ~ features means "predict shot_made using these features"
  
  # Build model with LASER-inspired features
  # Note: We use interaction terms to prevent double-counting difficulty
  # (e.g., pull_up_tight captures both pull-up AND tight contest in one term)
  
  model <- glm(
    shot_made ~ 
      # Core features
      shot_distance + 
      I(shot_distance^2) +  # Quadratic term (distance squared) - captures non-linear relationship
      is_three +
      shot_clock_remaining +
      I(shot_clock_remaining^2) +  # Quadratic for shot clock
      game_seconds_remaining +
      is_clutch +
      score_margin +
      game_close +
      is_off_turnover +
      is_putback +
      is_fast_break +
      is_corner_3 +
      # LASER-inspired features (if available in data)
      defender_distance +  # Closer defender = harder shot (MOST IMPORTANT addition!)
      contest_score +      # Overall contest level
      is_catch_shoot +     # Catch-and-shoot vs pull-up
      is_pull_up +
      touch_time +         # How long player has ball
      # Interaction terms (prevent double-counting difficulty)
      pull_up_tight +      # Pull-up + tight contest (very difficult)
      catch_wide_open +    # Catch-and-shoot + wide open (easiest)
      pull_up_three +     # Pull-up 3PT (harder than catch-and-shoot 3PT)
      contested_three +   # Contested 3PT (harder than open 3PT)
      factor(season),      # Season as categorical (different seasons = different baseline)
    data = train_data,
    family = binomial(link = "logit")  # Logistic regression
  )
  
  cat("Model training complete!\n")
  
  # Evaluate model on test set
  test_predictions <- predict(model, newdata = test_data, type = "response")
  test_data$expected_fg <- test_predictions
  
  # Calculate metrics
  # AUC (Area Under Curve) measures how well model separates made vs missed
  # 0.5 = random guessing, 1.0 = perfect prediction
  roc_obj <- roc(test_data$shot_made, test_predictions)
  auc_score <- auc(roc_obj)
  
  cat(sprintf("Model AUC: %.3f\n", auc_score))
  cat("(AUC > 0.7 is good, > 0.8 is excellent)\n")
  
  return(list(
    model = model,
    train_data = train_data,
    test_data = test_data,
    auc = auc_score
  ))
}

# ----------------------------------------------------------------------------
# STEP 6: CALCULATE SWISH SCORES (Expected vs Actual Points Per Shot)
# ----------------------------------------------------------------------------
# Now we'll use our model to predict expected points per shot for every shot,
# then calculate which players exceed expectations the most.
#
# SWISH SCORE = Actual Points Per Shot - Expected Points Per Shot
# Positive SWISH = Player exceeds expectations (true shotmaker)
# Negative SWISH = Player underperforms relative to shot difficulty
#
# WHY POINTS PER SHOT INSTEAD OF FG%?
# - A 3-pointer is worth 3 points, a 2-pointer is worth 2 points
# - We should measure value added, not just makes
# - Points per shot (RTS% = Real True Shooting) is a better metric for shotmaking
#
# EXAMPLE:
# - Player A: Makes 40% of 3s → 1.2 points per shot
# - Player B: Makes 60% of 2s → 1.2 points per shot
# - Both have same points per shot, even though Player B has higher FG%

calculate_shotmaking <- function(shots, model){
  
  cat("Calculating SWISH scores (expected vs actual points per shot)...\n")
  
  # Prepare shots for prediction (same features as training)
  # Include LASER-inspired features
  shots_for_pred <- shots %>%
    select(
      shot_distance,
      is_three,
      shot_clock_remaining,
      game_seconds_remaining,
      is_clutch,
      score_margin,
      game_close,
      is_off_turnover,
      is_putback,
      is_fast_break,
      is_corner_3,
      season,
      shot_made,
      player_name,
      # LASER-inspired features
      defender_distance,
      contest_score,
      is_wide_open,
      is_open,
      is_tight,
      is_very_tight,
      is_catch_shoot,
      is_pull_up,
      touch_time,
      is_quick_shot,
      # Interaction terms
      pull_up_tight,
      catch_wide_open,
      pull_up_three,
      contested_three
    ) %>%
    mutate(
      shot_distance = ifelse(is.na(shot_distance), median(shot_distance, na.rm = TRUE), shot_distance),
      shot_clock_remaining = ifelse(is.na(shot_clock_remaining), 12, shot_clock_remaining),
      game_seconds_remaining = ifelse(is.na(game_seconds_remaining), 1200, game_seconds_remaining),
      score_margin = ifelse(is.na(score_margin), 0, score_margin),
      season = ifelse(is.na(season), 2024, season),
      # Shot value: 3 points for 3PT, 2 points for 2PT
      shot_value = ifelse(is_three, 3, 2),
      # Handle missing LASER-inspired features
      defender_distance = ifelse(is.na(defender_distance), 
                                median(defender_distance, na.rm = TRUE), 
                                defender_distance),
      contest_score = ifelse(is.na(contest_score), 1, contest_score),
      touch_time = ifelse(is.na(touch_time), 2, touch_time),
      is_wide_open = ifelse(is.na(is_wide_open), FALSE, is_wide_open),
      is_open = ifelse(is.na(is_open), TRUE, is_open),
      is_tight = ifelse(is.na(is_tight), FALSE, is_tight),
      is_very_tight = ifelse(is.na(is_very_tight), FALSE, is_very_tight),
      is_catch_shoot = ifelse(is.na(is_catch_shoot), TRUE, is_catch_shoot),
      is_pull_up = ifelse(is.na(is_pull_up), FALSE, is_pull_up),
      is_quick_shot = ifelse(is.na(is_quick_shot), TRUE, is_quick_shot)
    )
  
  # Predict expected probability of make for each shot
  shots_for_pred$expected_make_prob <- predict(model, newdata = shots_for_pred, type = "response")
  
  # Calculate expected points per shot = expected_make_prob * shot_value
  # This is the "expected value" of the shot
  shots_for_pred$expected_points_per_shot <- shots_for_pred$expected_make_prob * shots_for_pred$shot_value
  
  # Calculate actual points per shot = (shot_made * shot_value) / 1
  # If shot is made, we get shot_value points. If missed, we get 0 points.
  shots_for_pred$actual_points_per_shot <- shots_for_pred$shot_made * shots_for_pred$shot_value
  
  # Calculate actual vs expected for each player
  player_shotmaking <- shots_for_pred %>%
    filter(!is.na(player_name) & player_name != "Unknown") %>%
    group_by(player_name) %>%
    summarize(
      total_shots = n(),
      total_makes = sum(shot_made, na.rm = TRUE),
      total_points = sum(actual_points_per_shot, na.rm = TRUE),
      # Actual points per shot (RTS%)
      actual_rts = mean(actual_points_per_shot, na.rm = TRUE),
      # Expected points per shot (based on shot context)
      expected_rts = mean(expected_points_per_shot, na.rm = TRUE),
      # SWISH Score = actual - expected (positive = exceeds expectations)
      # This is measured in points per shot above/below expected
      swish_score = actual_rts - expected_rts,
      # Also keep shotmaking_ability as alias for backward compatibility
      shotmaking_ability = actual_rts - expected_rts,
      # Total SWISH value (total extra points from exceeding expectations)
      total_swish_value = sum(actual_points_per_shot - expected_points_per_shot, na.rm = TRUE),
      # Also keep FG% for reference
      actual_fg = mean(shot_made, na.rm = TRUE),
      expected_fg = mean(expected_make_prob, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    # Filter to players with meaningful sample size (at least 100 shots)
    filter(total_shots >= 100) %>%
    arrange(desc(swish_score))
  
  cat(sprintf("Calculated SWISH scores for %d players\n", nrow(player_shotmaking)))
  
  return(list(
    shots_with_expected = shots_for_pred,
    player_shotmaking = player_shotmaking
  ))
}

# ----------------------------------------------------------------------------
# STEP 7: VISUALIZATION
# ----------------------------------------------------------------------------
# Create visualizations to understand the model and results

visualize_shotmaking <- function(player_shotmaking, output_dir = "output"){
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Plot 1: Actual Points Per Shot (RTS%) vs Expected Points Per Shot
  # Players above the diagonal line exceed expectations
  p1 <- ggplot(player_shotmaking, aes(x = expected_rts, y = actual_rts)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
    geom_point(aes(size = total_shots, color = swish_score), alpha = 0.7) +
    scale_color_gradient2(low = "red", mid = "white", high = "green", 
                          midpoint = 0, name = "SWISH\nScore\n(pts/shot)") +
    scale_size_continuous(name = "Total Shots", range = c(2, 8)) +
    labs(
      title = "SWISH Model: WNBA Shotmaking Analysis",
      subtitle = "Actual vs Expected Points Per Shot | Players above the line exceed expectations",
      x = "Expected Points Per Shot (based on shot context)",
      y = "Actual Points Per Shot (RTS%)",
      caption = "SWISH = Shotmaking With Intelligent Shot Handling | Minimum 100 shots"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 14, face = "bold"))
  
  ggsave(file.path(output_dir, "wnba_shotmaking_actual_vs_expected.png"), 
         p1, width = 10, height = 8, dpi = 300)
  
  # Plot 2: Top Shotmakers by SWISH Score
  top_shotmakers <- player_shotmaking %>%
    slice_max(order_by = swish_score, n = 20)
  
  p2 <- ggplot(top_shotmakers, aes(x = reorder(player_name, swish_score), 
                                    y = swish_score)) +
    geom_col(aes(fill = swish_score)) +
    scale_fill_gradient2(low = "red", mid = "white", high = "green", 
                         midpoint = 0) +
    coord_flip() +
    labs(
      title = "Top 20 WNBA Shotmakers by SWISH Score (2024-2025)",
      subtitle = "SWISH Score = Actual Points Per Shot - Expected Points Per Shot",
      x = "Player",
      y = "SWISH Score (Points Per Shot Above Expected)"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
  
  ggsave(file.path(output_dir, "wnba_top_shotmakers.png"), 
         p2, width = 10, height = 8, dpi = 300)
  
  cat("Visualizations saved to", output_dir, "\n")
  
  return(list(plot1 = p1, plot2 = p2))
}

# ----------------------------------------------------------------------------
# STEP 8: MAIN EXECUTION FUNCTION
# ----------------------------------------------------------------------------
# This ties everything together

run_shotmaking_analysis <- function(data_dir = "wnba_data",
                                     seasons = 2024:2025,
                                     seasontype = 'rg',
                                     output_dir = "output",
                                     min_shots = 100){
  
  cat("========================================\n")
  cat("SWISH MODEL: WNBA Shotmaking Analysis\n")
  cat("Shotmaking With Intelligent Shot Handling\n")
  cat("========================================\n\n")
  
  # Step 1: Load data
  shots <- load_wnba_shots(data_dir = data_dir, 
                           seasons = seasons, 
                           seasontype = seasontype)
  
  # Step 2: Create features
  shots <- create_shot_features(shots)
  
  # Step 3: Build model
  model_results <- build_shot_model(shots)
  
  # Step 4: Calculate shotmaking
  shotmaking_results <- calculate_shotmaking(shots, model_results$model)
  
  # Step 5: Visualize
  plots <- visualize_shotmaking(shotmaking_results$player_shotmaking, output_dir)
  
  # Step 6: Save results
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  write_csv(shotmaking_results$player_shotmaking, 
            file.path(output_dir, "wnba_player_shotmaking.csv"))
  
  cat("\n========================================\n")
  cat("SWISH MODEL ANALYSIS COMPLETE!\n")
  cat("========================================\n")
  cat(sprintf("Model AUC: %.3f\n", model_results$auc))
  cat(sprintf("Top Shotmaker: %s (SWISH Score: +%.3f)\n", 
              shotmaking_results$player_shotmaking$player_name[1],
              shotmaking_results$player_shotmaking$swish_score[1]))
  cat(sprintf("  Actual RTS%%: %.3f | Expected RTS%%: %.3f\n",
              shotmaking_results$player_shotmaking$actual_rts[1],
              shotmaking_results$player_shotmaking$expected_rts[1]))
  cat("\nResults saved to:", output_dir, "\n")
  
  return(list(
    model = model_results$model,
    player_shotmaking = shotmaking_results$player_shotmaking,
    shots_with_expected = shotmaking_results$shots_with_expected,
    plots = plots
  ))
}

# ============================================================================
# RUN THE ANALYSIS
# ============================================================================
# Uncomment the line below to run the full analysis

# results <- run_shotmaking_analysis(
#   data_dir = "wnba_data",
#   seasons = 2024:2025,
#   seasontype = 'rg',
#   output_dir = "output"
# )
