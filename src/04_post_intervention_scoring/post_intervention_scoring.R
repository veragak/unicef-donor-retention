############################################################
# POST-INTERVENTION NPS SCORING - WITH AUTO HEADER FIX
############################################################

library(tidyverse)
library(xgboost)

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("POST-INTERVENTION NPS SCORING WITH XGBOOST\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# ==========================================================
# HELPER FUNCTION: Fix CSV with headers in row 2
# ==========================================================

fix_csv_headers <- function(df) {
  # Check if first row looks like headers (all character/NA when converted to numeric)
  first_row_numeric <- suppressWarnings(as.numeric(df[1, 1:min(5, ncol(df))]))
  
  if (all(is.na(first_row_numeric))) {
    cat("  → Detected headers in row 2, fixing...\n")
    
    # Extract actual column names from row 1
    actual_names <- as.character(df[1, ])
    
    # Remove the header row
    df_fixed <- df[-1, ]
    
    # Set correct names
    names(df_fixed) <- actual_names
    
    return(df_fixed)
  } else {
    cat("  → Headers look OK\n")
    return(df)
  }
}

# ==========================================================
# 1. CHECK MODEL
# ==========================================================

cat("STEP 1: Checking prerequisites...\n")

if (!exists("model_xgb")) {
  stop("ERROR: model_xgb not found! Please run your XGBoost training script first.")
}
cat("  ✓ XGBoost model found\n")

# ==========================================================
# 2. LOAD AND FIX SIMULATION RESULTS
# ==========================================================

cat("\nSTEP 2: Loading simulation results...\n")

if (file.exists("simulated_intervention_results_proof_of_impact.csv")) {
  sim_path <- "simulated_intervention_results_proof_of_impact.csv"
} else {
  stop("ERROR: Cannot find simulated_intervention_results.csv")
}

sim_raw <- read_csv(sim_path, show_col_types = FALSE)
sim_results <- fix_csv_headers(sim_raw)

# Convert numeric columns
numeric_cols <- c(
  "imp_vital", "imp_approach", "imp_difference",
  "idx_capacity", "idx_trust", "idx_impact",
  "new_trust", "new_impact", "new_alignment_pct",
  "new_perc_breadth", "new_info_spending_more",
  "tot25_medium", "n_yrs_active", "nps",
  "perc_breadth", "info_spending_more", "alignment_pct",
  "donor_id", "attempts"
)

for (col in numeric_cols) {
  if (col %in% names(sim_results)) {
    sim_results[[col]] <- as.numeric(sim_results[[col]])
  }
}

cat(sprintf("  ✓ Loaded %d rows\n", nrow(sim_results)))

# ==========================================================
# 3. LOAD AND FIX FRAGILE DONORS
# ==========================================================

cat("\nSTEP 3: Loading fragile donors...\n")

if (file.exists("fragile_group_donors.csv")) {
  fragile_path <- "fragile_group_donors.csv"
} else {
  stop("ERROR: Cannot find fragile_group_donors.csv")
}

fragile_raw <- read_csv(fragile_path, show_col_types = FALSE)
fragile <- fix_csv_headers(fragile_raw)

# Convert numeric columns for fragile
for (col in numeric_cols) {
  if (col %in% names(fragile)) {
    fragile[[col]] <- as.numeric(fragile[[col]])
  }
}

cat(sprintf("  ✓ Loaded %d rows\n", nrow(fragile)))

# ==========================================================
# 4. FILTER SUCCESSFUL SIMULATIONS
# ==========================================================

cat("\nSTEP 4: Filtering valid simulations...\n")

# Handle the 'success' column (might be "True"/"False" strings)
if ("success" %in% names(sim_results)) {
  sim_results <- sim_results %>%
    mutate(success = (success == "True" | success == TRUE | success == 1))
}

# Calculate changes and success flags
sim_results <- sim_results %>%
  mutate(
    delta_trust = new_trust - idx_trust,
    delta_impact = new_impact - idx_impact,
    delta_alignment = new_alignment_pct - alignment_pct,
    delta_breadth = new_perc_breadth - perc_breadth,
    
    # Valid data check
    has_valid_data = !is.na(new_trust) & !is.na(new_impact) & 
                     !is.na(new_alignment_pct) & !is.na(new_perc_breadth) &
                     !is.na(donor_id)
  )

# Filter for valid data
sim_success <- sim_results %>%
  filter(has_valid_data == TRUE)

cat(sprintf("  ✓ Valid simulations: %d / %d (%.1f%%)\n", 
            nrow(sim_success), nrow(sim_results),
            100 * nrow(sim_success) / nrow(sim_results)))

if (nrow(sim_success) == 0) {
  stop("ERROR: No valid simulation results!")
}

# ==========================================================
# 5. MATCH DONORS AND BUILD FEATURES
# ==========================================================

cat("\nSTEP 5: Matching donors and building features...\n")

# Convert 0-based Python index to 1-based R index
sim_success <- sim_success %>%
  mutate(row_index = donor_id + 1)

# Validate indices
max_index <- max(sim_success$row_index, na.rm = TRUE)
if (max_index > nrow(fragile)) {
  cat(sprintf("  WARNING: Max index %d > available rows %d, filtering...\n", 
              max_index, nrow(fragile)))
  sim_success <- sim_success %>%
    filter(row_index <= nrow(fragile))
}

cat(sprintf("  ✓ Matched %d donors\n", nrow(sim_success)))

# Extract matching fragile donors
features_pre_raw <- fragile[sim_success$row_index, ]

# ==========================================================
# 6. BUILD PRE-INTERVENTION FEATURES
# ==========================================================

cat("\nSTEP 6: Building pre-intervention features...\n")

# Build feature matrix (match XGBoost training features)
features_pre <- features_pre_raw %>%
  select(
    n_yrs_active,
    idx_capacity,
    idx_trust,
    idx_impact,
    tot25_medium,
    perc_breadth,
    info_spending_more,
    alignment_pct
  )

# Convert to numeric
for (col in names(features_pre)) {
  features_pre[[col]] <- as.numeric(features_pre[[col]])
}

cat(sprintf("  ✓ Built matrix: %d × %d\n", nrow(features_pre), ncol(features_pre)))

# ==========================================================
# 7. BUILD POST-INTERVENTION FEATURES
# ==========================================================

cat("\nSTEP 7: Building post-intervention features...\n")

features_post <- features_pre

# Replace intervention-affected variables
features_post$idx_trust <- sim_success$new_trust
features_post$idx_impact <- sim_success$new_impact
features_post$perc_breadth <- sim_success$new_perc_breadth
features_post$info_spending_more <- sim_success$new_info_spending_more

# CRITICAL: Scale alignment from 0-100 to 0-1
features_post$alignment_pct <- sim_success$new_alignment_pct / 100

cat("  ✓ Updated 5 intervention variables\n")
cat("  ✓ Converted alignment_pct: 0-100 → 0-1\n")

# ==========================================================
# 8. FILTER COMPLETE CASES
# ==========================================================

cat("\nSTEP 8: Filtering complete cases...\n")

valid_rows <- complete.cases(features_pre) & complete.cases(features_post)

features_pre_clean <- features_pre[valid_rows, ]
features_post_clean <- features_post[valid_rows, ]
sim_success_clean <- sim_success[valid_rows, ]

cat(sprintf("  ✓ Complete cases: %d\n", sum(valid_rows)))

if (sum(valid_rows) == 0) {
  stop("ERROR: No complete cases!")
}

# ==========================================================
# 9. PREDICT NPS
# ==========================================================

cat("\nSTEP 9: Predicting NPS with XGBoost...\n")

dmat_pre <- xgb.DMatrix(data = as.matrix(features_pre_clean))
dmat_post <- xgb.DMatrix(data = as.matrix(features_post_clean))

nps_pred_pre <- predict(model_xgb, dmat_pre)
nps_pred_post <- predict(model_xgb, dmat_post)

cat("  ✓ Pre-intervention NPS predicted\n")
cat("  ✓ Post-intervention NPS predicted\n")

# ==========================================================
# 10. ASSEMBLE RESULTS
# ==========================================================

cat("\nSTEP 10: Assembling results...\n")

results <- tibble(
  donor_id = sim_success_clean$donor_id,
  code_id = features_pre_raw$code_id[valid_rows],
  
  n_yrs_active = features_pre_clean$n_yrs_active,
  idx_capacity = features_pre_clean$idx_capacity,
  tot25_medium = features_pre_clean$tot25_medium,
  
  pre_trust = features_pre_clean$idx_trust,
  pre_impact = features_pre_clean$idx_impact,
  pre_alignment = features_pre_clean$alignment_pct,
  pre_breadth = features_pre_clean$perc_breadth,
  pre_spending_more = features_pre_clean$info_spending_more,
  
  post_trust = features_post_clean$idx_trust,
  post_impact = features_post_clean$idx_impact,
  post_alignment = features_post_clean$alignment_pct,
  post_breadth = features_post_clean$perc_breadth,
  post_spending_more = features_post_clean$info_spending_more,
  
  delta_trust = post_trust - pre_trust,
  delta_impact = post_impact - pre_impact,
  delta_alignment = post_alignment - pre_alignment,
  delta_breadth = post_breadth - pre_breadth,
  delta_spending = post_spending_more - pre_spending_more,
  
  nps_pred_pre = nps_pred_pre,
  nps_pred_post = nps_pred_post,
  nps_delta = nps_pred_post - nps_pred_pre,
  
  nps_improved = nps_delta > 0,
  
  nps_category_pre = case_when(
    nps_pred_pre <= 6 ~ "Detractor",
    nps_pred_pre <= 8 ~ "Passive",
    TRUE ~ "Promoter"
  ),
  nps_category_post = case_when(
    nps_pred_post <= 6 ~ "Detractor",
    nps_pred_post <= 8 ~ "Passive",
    TRUE ~ "Promoter"
  ),
  
  rationale = sim_success_clean$qualitative_rationale
)

cat(sprintf("  ✓ Results for %d donors\n", nrow(results)))

# ==========================================================
# 11. SUMMARY STATISTICS
# ==========================================================

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("INTERVENTION IMPACT SUMMARY\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

cat("NPS PREDICTIONS:\n")
cat(sprintf("  Donors scored:          %d\n", nrow(results)))
cat(sprintf("  Mean NPS (pre):         %.2f\n", mean(results$nps_pred_pre)))
cat(sprintf("  Mean NPS (post):        %.2f\n", mean(results$nps_pred_post)))
cat(sprintf("  Mean NPS delta:         %+.2f\n", mean(results$nps_delta)))
cat(sprintf("  Median NPS delta:       %+.2f\n", median(results$nps_delta)))
cat(sprintf("  SD NPS delta:           %.2f\n\n", sd(results$nps_delta)))

cat("MOVEMENT:\n")
cat(sprintf("  Improved (Δ>0):  %d (%.1f%%)\n", 
            sum(results$nps_improved), 100 * mean(results$nps_improved)))
cat(sprintf("  Worsened (Δ<0):  %d (%.1f%%)\n", 
            sum(results$nps_delta < 0), 100 * mean(results$nps_delta < 0)))
cat(sprintf("  Unchanged (Δ=0): %d (%.1f%%)\n\n", 
            sum(results$nps_delta == 0), 100 * mean(results$nps_delta == 0)))

# ==========================================================
# 12. SAVE RESULTS
# ==========================================================

cat("STEP 12: Saving outputs...\n")

dir.create("outputs", showWarnings = FALSE)

write_csv(results, "outputs/fragile_donors_nps_comparison.csv")
cat("  ✓ Saved: outputs/fragile_donors_nps_comparison.csv\n")

# Save cleaned CSVs for future use
write_csv(sim_results, "outputs/simulated_intervention_results_CLEAN.csv")
write_csv(fragile, "outputs/fragile_group_donors_CLEAN.csv")
cat("  ✓ Saved cleaned CSVs\n")

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("✅ SCORING COMPLETE!\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

cat(sprintf("📊 Key Result: %.1f%% of donors showed NPS improvement\n",
            100 * mean(results$nps_improved)))
cat(sprintf("📈 Mean NPS change: %+.2f points\n\n", mean(results$nps_delta)))
