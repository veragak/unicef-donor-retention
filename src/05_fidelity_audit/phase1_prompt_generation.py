"""
UNICEF Donor Prompt Generation with Pre-Intervention Calibration
==================================================================
Phase 1 of Silicon Sample methodology.

Generates narrative backstories for each donor and validates that the LLM
can accurately predict baseline attitudes from demographics alone (RMSE check).
"""

import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import asyncio
import aiohttp

# Import interpretation functions from original script
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
        return f"{val:.2f}", "below-average financial capacity (lower income, education, and property value compared to other donors)"
    elif val <= 0.5:
        return f"{val:.2f}", "average financial capacity (typical income, education, and property value)"
    else:
        return f"{val:.2f}", "high financial capacity (top 25% in income, education, and property value)"

def interpret_trust(value):
    """Interpret trust index with context."""
    if pd.isna(value):
        return "not measured", "no data on trust attitudes"
    val = float(value)
    if val == 0.0:
        return f"{val:.2f}", "sincere skepticism (you answered 'No' when asked if you appreciate UNICEF's approach AND if you think their work is vital)"
    elif val == 0.5:
        return f"{val:.2f}", "neutral trust (you answered 'Yes' to only one of two questions about UNICEF's approach and importance)"
    elif val == 1.0:
        return f"{val:.2f}", "strong belief (you answered 'Yes' when asked if you appreciate UNICEF's approach AND if you think their work is vital)"
    else:
        return f"{val:.2f}", "moderate trust"

def interpret_impact(value):
    """Interpret impact index with context."""
    if pd.isna(value):
        return "not measured", "no data on perceived impact"
    val = float(value)
    if val == 0.0:
        return f"{val:.2f}", "low perceived efficacy (you don't believe UNICEF makes a real difference or that supporting causes is a moral duty)"
    elif val == 0.5:
        return f"{val:.2f}", "moderate belief in impact (you answered 'Yes' to only one of two questions about UNICEF's effectiveness and moral duty to support causes)"
    elif val == 1.0:
        return f"{val:.2f}", "high belief in mission and duty (you strongly believe UNICEF makes a difference AND that supporting good causes is morally right)"
    else:
        return f"{val:.2f}", "moderate perceived impact"

def interpret_breadth(value):
    """Interpret perception breadth with context."""
    if pd.isna(value):
        return "not measured", "unknown"
    val = float(value)
    if val <= 2:
        return f"{int(val)} out of 8 program areas", "low awareness of UNICEF's work (you're only familiar with a few of their programs)"
    elif val <= 5:
        return f"{int(val)} out of 8 program areas", "moderate awareness (you recognize UNICEF's involvement in several program areas)"
    else:
        return f"{int(val)} out of 8 program areas", "high awareness (you have broad knowledge of UNICEF's diverse programs across education, health, water, etc.)"

def interpret_alignment(value):
    """Interpret alignment percentage with context."""
    if pd.isna(value):
        return "not measured", "unknown"
    val = float(value)
    if val < 50:
        return f"{val:.1f}%", "poorly aligned (UNICEF isn't focusing on most of your priority issues)"
    elif val <= 75:
        return f"{val:.1f}%", "moderately aligned (UNICEF addresses some of your priority concerns)"
    else:
        return f"{val:.1f}%", "highly aligned (UNICEF's work matches most of your charitable priorities)"

def interpret_charities(value):
    """Interpret number of charities supported."""
    if pd.isna(value):
        return "an unknown number of", "which means the distribution of your charitable giving is unclear"
    
    if isinstance(value, str):
        try:
            val = int(value.split(':')[0].strip())
        except:
            return "an unknown number of", "which means the distribution of your charitable giving is unclear"
    else:
        try:
            val = int(float(value))
        except:
            return "an unknown number of", "which means the distribution of your charitable giving is unclear"
    
    if val == 0:
        return "0", "meaning either you haven't disclosed other charitable commitments, or UNICEF is your sole focus"
    elif val == 1:
        return "1 other", "meaning your charitable loyalty is relatively focused"
    elif val <= 3:
        return f"{val} other", "meaning your charitable resources are split among a small portfolio of causes"
    else:
        return f"{val} other", "meaning your loyalty and resources are spread across multiple competing causes"

def construct_donor_backstory(row):
    """Convert a single donor row into a narrative conditioning prompt."""
    
    age_group = safe_value(row.get('age_group'), 'an unspecified age group')
    lifephase = safe_value(row.get('lifephase'), 'an unspecified life phase')
    years_active = format_numeric(row.get('n_yrs_active'), 1, 'an unknown number of')
    
    capacity_score, capacity_desc = interpret_capacity(row.get('idx_capacity'))
    trust_score, trust_desc = interpret_trust(row.get('idx_trust'))
    impact_score, impact_desc = interpret_impact(row.get('idx_impact'))
    
    breadth_score, breadth_desc = interpret_breadth(row.get('perc_breadth'))
    alignment_score, alignment_desc = interpret_alignment(row.get('alignment_pct'))
    
    info_spending = safe_value(row.get('info_spending_more'), 'unknown')
    
    num_charities, charity_desc = interpret_charities(row.get('nr_or_charities_supported'))
    
    # Qualitative voice capture
    comment_likely = row.get('you_indicated_that_it_is_very_likely_that_you_will_continue_supporting_unicef_in_the_coming_years_could_you_please_elaborate_on_this', '')
    comment_uncertain = row.get('you_indicated_that_its_not_completely_certain_yet_that_you_will_continue_to_support_unicef_in_the_coming_years_could_you_elaborate_on_this', '')
    
    comments = []
    if pd.notna(comment_likely) and str(comment_likely).strip():
        comments.append(f"When asked about likelihood of continuing support: '{comment_likely}'")
    if pd.notna(comment_uncertain) and str(comment_uncertain).strip():
        comments.append(f"When expressing uncertainty: '{comment_uncertain}'")
    
    if comments:
        voice_section = " Your own words from past interactions: " + " | ".join(comments)
    else:
        voice_section = " You have not provided detailed written feedback in recent surveys."
    
    prompt = f"""You are a UNICEF donor who has been classified as 'fragile' based on predictive modeling of donor retention risk.

DEMOGRAPHIC PROFILE:
You are in the {age_group} age group and your life stage is categorized as '{lifephase}'.

RELATIONSHIP HISTORY:
You have been supporting UNICEF for {years_active} years. Your financial capacity index is {capacity_score} — this represents {capacity_desc}. This index was calculated using principal component analysis of your income, education level, and property value.

CURRENT SENTIMENT & PERCEPTION:
Your trust index score is {trust_score} — this indicates {trust_desc}. Trust is measured by asking if you appreciate UNICEF's approach and believe their work is vital (scale: 0.0 = answered 'No' to both, 0.5 = mixed, 1.0 = 'Yes' to both).

Your perceived impact score is {impact_score} — this indicates {impact_desc}. Impact perception is measured by asking if you believe UNICEF makes a real difference and whether you feel supporting good causes is a moral duty (scale: 0.0 = 'No' to both, 0.5 = mixed, 1.0 = 'Yes' to both).

BEHAVIORAL INDICATORS:
- Cause Alignment: Your priorities align with UNICEF's work at {alignment_score}, meaning you are {alignment_desc}. This is calculated by comparing the issues you care about most with where you perceive UNICEF to be active.
- Awareness of UNICEF's Work: You identified UNICEF as being involved in {breadth_score}, indicating {breadth_desc}.
- Future Giving Intentions: When asked about your plans to increase, maintain, or decrease giving, you indicated: {info_spending}.

CHARITABLE PORTFOLIO:
You currently support {num_charities} charitable organizations beyond UNICEF, {charity_desc}.

YOUR VOICE:
{voice_section}

You are about to be exposed to different donor retention interventions. Respond as this specific donor would, drawing on these characteristics, your tenure, your current sentiment scores, and your expressed concerns or enthusiasm."""

    return prompt

def construct_calibration_prompt(row):
    """
    Create a prompt with ONLY demographics (no attitudes revealed).
    Used for pre-intervention calibration check.
    """
    age_group = safe_value(row.get('age_group'), 'an unspecified age group')
    lifephase = safe_value(row.get('lifephase'), 'an unspecified life phase')
    years_active = format_numeric(row.get('n_yrs_active'), 1, 'an unknown number of')
    capacity_score, capacity_desc = interpret_capacity(row.get('idx_capacity'))
    
    num_charities, charity_desc = interpret_charities(row.get('nr_or_charities_supported'))
    
    prompt = f"""You are simulating a UNICEF donor for research purposes.

DEMOGRAPHIC PROFILE:
- Age group: {age_group}
- Life stage: {lifephase}
- Years supporting UNICEF: {years_active}
- Financial capacity: {capacity_desc}
- Other charities supported: {num_charities}

Based ONLY on these demographics, predict this donor's likely attitudes:

Output a JSON object with:
{{
  "predicted_trust": 0.0 or 0.5 or 1.0,
  "predicted_impact": 0.0 or 0.5 or 1.0,
  "reasoning": "brief explanation"
}}

Trust scale: 0.0 = skeptical, 0.5 = neutral, 1.0 = strong believer
Impact scale: 0.0 = low perceived efficacy, 0.5 = moderate, 1.0 = high belief

Output ONLY the JSON object."""

    return prompt

# =============================================================================
# PRE-INTERVENTION CALIBRATION CHECK
# =============================================================================

async def run_calibration_check_sample(df, groq_api_key, sample_size=30):
    """
    Run pre-intervention calibration check on a sample of donors.
    
    Tests whether the LLM can accurately predict attitudes from demographics alone.
    """
    
    print("\n" + "="*70)
    print("PRE-INTERVENTION CALIBRATION CHECK")
    print("="*70)
    print(f"\nTesting LLM baseline prediction accuracy on {sample_size} donors...")
    
    # Sample donors
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # NOTE: This is a simplified synchronous version for demonstration
    # In production, you'd use async calls like in Phase 2
    
    print("\n⚠️  Note: Full calibration check requires API implementation")
    print("This is a template showing the methodology.\n")
    
    # Placeholder for actual implementation
    # Would call Groq API with calibration prompts and compare to actuals
    
    print("To implement:")
    print("1. Call Groq API with demographic-only prompts")
    print("2. Compare predicted vs actual trust/impact")
    print("3. Calculate RMSE and proportion agreement")
    print("4. Flag if RMSE > threshold (e.g., 0.3)")
    
    return {
        "status": "NOT_IMPLEMENTED",
        "message": "Calibration check template created. Implement async API calls for production use."
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""
    
    filepath = 'fragile_group_donors.csv'
    
    print("\n" + "="*70)
    print("PHASE 1: PROMPT GENERATION WITH CALIBRATION")
    print("="*70)
    
    # Load data
    print("\n📂 Loading donor data...")
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} donors")
    except FileNotFoundError:
        print(f"❌ ERROR: '{filepath}' not found!")
        return
    
    # Generate conditioning prompts
    print("\n🔄 Generating conditioning prompts...")
    df['conditioning_prompt'] = df.apply(construct_donor_backstory, axis=1)
    print("✓ Generated conditioning prompts")
    
    # Export files
    print("\n💾 Exporting files...")
    
    # JSONL format
    with open('fragile_donors_prompts.jsonl', 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            f.write(json.dumps({
                'donor_id': idx,
                'prompt': row['conditioning_prompt']
            }, ensure_ascii=False) + '\n')
    
    # CSV format
    df.to_csv('fragile_donors_with_prompts.csv', index=False, encoding='utf-8')
    
    print("✓ Exported fragile_donors_prompts.jsonl")
    print("✓ Exported fragile_donors_with_prompts.csv")
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE PROMPT:")
    print("="*60)
    print(df['conditioning_prompt'].iloc[0])
    print("="*60)
    
    # Optional: Run calibration check
    print("\n" + "="*70)
    print("CALIBRATION CHECK (Optional)")
    print("="*70)
    print("\nTo run pre-intervention calibration check:")
    print("1. Uncomment the calibration check code below")
    print("2. Add your Groq API key")
    print("3. Run async calibration validation")
    print("\nThis validates that the LLM can predict attitudes from demographics.")
    
    # Uncomment to run calibration:
    # calibration_result = asyncio.run(
    #     run_calibration_check_sample(df, groq_api_key="YOUR_KEY", sample_size=30)
    # )
    
    print("\n✅ Phase 1 complete!")
    print("\nNext step: Run phase2_ensemble_simulation.py\n")

if __name__ == "__main__":
    main()
