"""
UNICEF Donor Prompt Generation - UNBLINDED VERSION
===================================================
Implements Gui & Toubia (2023) unblinding principles.

KEY CHANGES FROM BLINDED VERSION:
1. Third-person donor profiles (not "You are a donor...")
2. Intervention-agnostic (no mention of "respond to interventions")
3. Explicit baseline attitudes extracted for experimental context
4. Factual, neutral tone (demographic report, not role-playing)

This enables testing multiple interventions without regenerating Phase 1.
"""

import pandas as pd
import json
import numpy as np

# =============================================================================
# INTERPRETATION FUNCTIONS (Unchanged)
# =============================================================================

def safe_value(value, default="not specified"):
    """Handle NaN values gracefully."""
    if pd.isna(value):
        return default
    return value

def format_numeric(value, decimal_places=2, default="not available"):
    """Format numeric values with proper handling of NaN."""
    if pd.isna(value):
        return default
    return f"{float(value):.{decimal_places}f}"

def interpret_capacity(value):
    """Interpret capacity index with context."""
    if pd.isna(value):
        return "not measured", "unknown"
    val = float(value)
    if val < -0.5:
        return f"{val:.2f}", "below-average financial capacity"
    elif val <= 0.5:
        return f"{val:.2f}", "average financial capacity"
    else:
        return f"{val:.2f}", "high financial capacity (top 25%)"

def interpret_breadth(value):
    """Interpret perception breadth with context."""
    if pd.isna(value):
        return "not measured", "unknown"
    val = float(value)
    if val <= 2:
        return f"{int(val)} out of 8 program areas", "low awareness"
    elif val <= 5:
        return f"{int(val)} out of 8 program areas", "moderate awareness"
    else:
        return f"{int(val)} out of 8 program areas", "high awareness"

def interpret_alignment(value):
    """Interpret alignment percentage with context."""
    if pd.isna(value):
        return "not measured", "unknown"
    val = float(value)
    if val < 50:
        return f"{val:.1f}%", "poorly aligned"
    elif val <= 75:
        return f"{val:.1f}%", "moderately aligned"
    else:
        return f"{val:.1f}%", "highly aligned"

def interpret_charities(value):
    """Interpret number of charities supported."""
    if pd.isna(value):
        return "unknown", "portfolio diversity unknown"
    
    if isinstance(value, str):
        try:
            val = int(value.split(':')[0].strip())
        except:
            return "unknown", "portfolio diversity unknown"
    else:
        try:
            val = int(float(value))
        except:
            return "unknown", "portfolio diversity unknown"
    
    if val == 0:
        return "0", "UNICEF is sole charitable focus"
    elif val == 1:
        return "1", "focused charitable loyalty"
    elif val <= 3:
        return f"{val}", "small portfolio of causes"
    else:
        return f"{val}", "resources spread across multiple causes"

# =============================================================================
# UNBLINDED DONOR PROFILE CONSTRUCTOR
# =============================================================================

def construct_unblinded_donor_profile(row):
    """
    Convert donor row into THIRD-PERSON, INTERVENTION-AGNOSTIC profile.
    
    KEY DIFFERENCES FROM BLINDED VERSION:
    - Uses third-person ("This donor is..." not "You are...")
    - Removes role-playing framing
    - Strips out baseline trust/impact (those go in experimental context)
    - Factual demographic/behavioral report only
    - No mention of "interventions" or "respond as this donor"
    
    This output is REUSABLE across multiple intervention tests.
    """
    
    # Demographics
    age_group = safe_value(row.get('age_group'), 'unspecified age')
    lifephase = safe_value(row.get('lifephase'), 'unspecified life stage')
    years_active = format_numeric(row.get('n_yrs_active'), 1, 'unknown')
    
    # Financial capacity
    capacity_score, capacity_desc = interpret_capacity(row.get('idx_capacity'))
    
    # Behavioral indicators (NO TRUST/IMPACT - those are outcomes)
    breadth_score, breadth_desc = interpret_breadth(row.get('perc_breadth'))
    alignment_score, alignment_desc = interpret_alignment(row.get('alignment_pct'))
    info_spending = safe_value(row.get('info_spending_more'), 'unknown')
    
    # Charitable portfolio
    num_charities, charity_desc = interpret_charities(row.get('nr_or_charities_supported'))
    
    # Qualitative voice
    comment_likely = row.get('you_indicated_that_it_is_very_likely_that_you_will_continue_supporting_unicef_in_the_coming_years_could_you_please_elaborate_on_this', '')
    comment_uncertain = row.get('you_indicated_that_its_not_completely_certain_yet_that_you_will_continue_to_support_unicef_in_the_coming_years_could_you_elaborate_on_this', '')
    
    comments = []
    if pd.notna(comment_likely) and str(comment_likely).strip():
        comments.append(f"When asked about likelihood of continuing support: '{comment_likely}'")
    if pd.notna(comment_uncertain) and str(comment_uncertain).strip():
        comments.append(f"When expressing uncertainty: '{comment_uncertain}'")
    
    if comments:
        voice_section = "Past survey feedback: " + " | ".join(comments)
    else:
        voice_section = "No detailed written feedback provided in recent surveys."
    
    # UNBLINDED PROFILE (Third-person, factual)
    profile = f"""DONOR PROFILE:

DEMOGRAPHICS:
- Age group: {age_group}
- Life stage: {lifephase}
- Tenure as UNICEF supporter: {years_active} years
- Financial capacity index: {capacity_score} ({capacity_desc})

BEHAVIORAL INDICATORS:
- Cause alignment: {alignment_score} ({alignment_desc})
  This reflects the match between donor's priority issues and perceived UNICEF focus areas.
  
- UNICEF program awareness: {breadth_score} ({breadth_desc})
  Donor identified UNICEF involvement in these program areas out of 8 total.
  
- Future giving intentions: {info_spending}
  Donor's stated plans regarding increasing/maintaining/decreasing support.

CHARITABLE PORTFOLIO:
- Other charities supported: {num_charities} ({charity_desc})

DONOR VOICE:
{voice_section}"""

    return profile

# =============================================================================
# BASELINE ATTITUDE EXTRACTOR
# =============================================================================

def extract_baseline_attitudes(row):
    """
    Extract baseline trust/impact scores for experimental context.
    
    These are SEPARATE from the donor profile because they are:
    1. Pre-treatment outcomes (not demographics)
    2. Needed for do(intervention) framing in Phase 2
    3. Should NOT be in the conditioning backstory (prevents focalism)
    """
    
    trust = row.get('idx_trust', np.nan)
    impact = row.get('idx_impact', np.nan)
    
    # Validate values
    trust_valid = trust in [0.0, 0.5, 1.0] if pd.notna(trust) else False
    impact_valid = impact in [0.0, 0.5, 1.0] if pd.notna(impact) else False
    
    return {
        'baseline_trust': float(trust) if trust_valid else None,
        'baseline_impact': float(impact) if impact_valid else None,
        'trust_valid': trust_valid,
        'impact_valid': impact_valid
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""
    
    filepath = 'fragile_group_donors.csv'
    
    print("\n" + "="*70)
    print("PHASE 1: UNBLINDED DONOR PROFILE GENERATION")
    print("="*70)
    print("\nImplementing Gui & Toubia (2023) unblinding principles:")
    print("  - Third-person profiles (not role-playing)")
    print("  - Intervention-agnostic (reusable across experiments)")
    print("  - Baseline attitudes extracted separately")
    print("="*70)
    
    # Load data
    print("\n📂 Loading donor data...")
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} donors")
    except FileNotFoundError:
        print(f"❌ ERROR: '{filepath}' not found!")
        return
    
    # Generate unblinded profiles
    print("\n🔄 Generating unblinded donor profiles...")
    df['donor_profile'] = df.apply(construct_unblinded_donor_profile, axis=1)
    print("✓ Generated intervention-agnostic profiles")
    
    # Extract baseline attitudes
    print("\n🔄 Extracting baseline attitudes...")
    baseline_data = df.apply(extract_baseline_attitudes, axis=1, result_type='expand')
    df = pd.concat([df, baseline_data], axis=1)
    
    valid_baselines = df['trust_valid'].sum()
    print(f"✓ Extracted baselines for {valid_baselines}/{len(df)} donors")
    
    if valid_baselines < len(df):
        print(f"⚠️  WARNING: {len(df) - valid_baselines} donors have invalid baseline scores")
    
    # Export files
    print("\n💾 Exporting files...")
    
    # JSONL format (donor profiles only)
    with open('fragile_donors_unblinded_profiles.jsonl', 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            f.write(json.dumps({
                'donor_id': idx,
                'profile': row['donor_profile'],
                'baseline_trust': row['baseline_trust'],
                'baseline_impact': row['baseline_impact']
            }, ensure_ascii=False) + '\n')
    
    # CSV format (full dataset)
    output_cols = list(df.columns)
    # Ensure donor_profile and baseline columns are included
    if 'donor_profile' not in output_cols:
        output_cols.append('donor_profile')
    
    df.to_csv('fragile_donors_unblinded.csv', index=False, encoding='utf-8')
    
    print("✓ Exported fragile_donors_unblinded_profiles.jsonl")
    print("✓ Exported fragile_donors_unblinded.csv")
    
    # Show sample
    print("\n" + "="*70)
    print("SAMPLE UNBLINDED DONOR PROFILE:")
    print("="*70)
    print(df['donor_profile'].iloc[0])
    print("")
    print(f"BASELINE ATTITUDES (stored separately):")
    print(f"  Trust:  {df['baseline_trust'].iloc[0]}")
    print(f"  Impact: {df['baseline_impact'].iloc[0]}")
    print("="*70)
    
    # Validation summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY:")
    print("="*70)
    print(f"Total donors processed:           {len(df)}")
    print(f"Valid baseline trust scores:      {df['trust_valid'].sum()}")
    print(f"Valid baseline impact scores:     {df['impact_valid'].sum()}")
    print(f"Fully valid donors:               {(df['trust_valid'] & df['impact_valid']).sum()}")
    
    # Distribution check
    if valid_baselines > 0:
        print(f"\nBaseline Trust distribution:")
        print(df[df['trust_valid']]['baseline_trust'].value_counts().sort_index())
        print(f"\nBaseline Impact distribution:")
        print(df[df['impact_valid']]['baseline_impact'].value_counts().sort_index())
    
    print("\n✅ Phase 1 complete!")
    print("\nKey differences from blinded version:")
    print("  ✓ Third-person profiles (no role-playing)")
    print("  ✓ Baseline attitudes extracted separately")
    print("  ✓ Intervention-agnostic (test multiple interventions)")
    print("  ✓ Factual demographic report (no 'respond as this donor')")
    print("\nNote: phase2_ensemble_simulation.py expects the conditioning_prompt column from phase1_prompt_generation.py.\n")

if __name__ == "__main__":
    main()
