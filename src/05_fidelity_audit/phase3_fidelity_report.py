"""
UNICEF Silicon Sample Fidelity Report Generator
================================================
Implements comprehensive algorithmic fidelity checks following:
- Argyle et al. (2023): Silicon Sample methodology
- Madden (2025): Statistical guardrails
- Rakovics & Rakovics (2024): Distributional fidelity

Backstage (Internal):
- KL Divergence with bootstrapping
- RMSE calibration
- Variance collapse detection

Frontstage (UNICEF):
- Cramer's V pattern correspondence
- Proportion agreement
- Executive summary
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import distance
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # File paths
    "ensemble_results": "ensemble_intervention_results.csv",
    "all_draws": "ensemble_details_all_draws.csv",
    "output_report": "fidelity_report_executive_summary.txt",
    "output_detailed": "fidelity_report_detailed_metrics.csv",
    
    # Thresholds
    "cramers_v_threshold": 0.1,  # Max acceptable divergence
    "kl_divergence_threshold": 0.5,  # Distributional shift flag
    "variance_collapse_threshold": 0.7,  # Ratio threshold (synthetic/human)
    "bootstrap_iterations": 2000,  # For KL divergence CI
    
    # Priority variables (from XGBoost feature importance)
    "high_priority_vars": ["n_yrs_active", "idx_capacity"],
    "secondary_vars": ["age_group", "lifephase"],
    "exploratory_vars": ["nr_or_charities_supported"]
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cramers_v(x, y):
    """
    Calculate Cramer's V correlation between two categorical variables.
    
    Range: [0, 1] where 0 = no association, 1 = perfect association
    """
    # Create contingency table
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    
    if min_dim == 0:
        return 0.0
    
    cramers = np.sqrt(chi2 / (n * min_dim))
    return cramers

def discretize_continuous(series, n_bins=3):
    """Discretize continuous variables for Cramer's V calculation."""
    return pd.qcut(series, q=n_bins, labels=False, duplicates='drop')

def kl_divergence(p, q, epsilon=1e-10):
    """
    Calculate Kullback-Leibler divergence between two probability distributions.
    
    Args:
        p: True distribution (human)
        q: Approximating distribution (synthetic)
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL divergence value (0 = perfect match, higher = more divergence)
    """
    # Ensure distributions sum to 1
    p = np.asarray(p) + epsilon
    q = np.asarray(q) + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))

def bootstrap_kl_divergence(human_data, synthetic_data, n_iterations=2000):
    """
    Bootstrap KL divergence to estimate confidence interval.
    
    Following Rakovics & Rakovics (2024) methodology.
    """
    kl_values = []
    
    n_human = len(human_data)
    n_synthetic = len(synthetic_data)
    
    for _ in range(n_iterations):
        # Resample with replacement
        human_sample = np.random.choice(human_data, size=n_human, replace=True)
        synthetic_sample = np.random.choice(synthetic_data, size=n_synthetic, replace=True)
        
        # Create probability distributions
        human_dist, bins = np.histogram(human_sample, bins=10, density=True)
        synthetic_dist, _ = np.histogram(synthetic_sample, bins=bins, density=True)
        
        # Calculate KL divergence
        kl = kl_divergence(human_dist, synthetic_dist)
        kl_values.append(kl)
    
    return {
        "mean": np.mean(kl_values),
        "median": np.median(kl_values),
        "ci_lower": np.percentile(kl_values, 2.5),
        "ci_upper": np.percentile(kl_values, 97.5),
        "all_values": kl_values
    }

# =============================================================================
# BACKSTAGE VALIDATION
# =============================================================================

def backstage_distributional_fidelity(df_human, df_synthetic):
    """
    BACKSTAGE CHECK 1: KL Divergence with bootstrapping
    
    Tests whether synthetic distribution matches human distribution
    for trust and impact scores.
    """
    print("\n" + "="*70)
    print("BACKSTAGE VALIDATION: Distributional Fidelity")
    print("="*70)
    
    results = {}
    
    # Trust distribution
    print("\n🔍 KL Divergence: Trust Index")
    human_trust = df_human['idx_trust'].dropna()
    synthetic_trust = df_synthetic['ensemble_trust'].dropna()
    
    kl_trust = bootstrap_kl_divergence(human_trust, synthetic_trust, 
                                       CONFIG['bootstrap_iterations'])
    
    print(f"  Mean KL Divergence:  {kl_trust['mean']:.4f}")
    print(f"  95% CI:              [{kl_trust['ci_lower']:.4f}, {kl_trust['ci_upper']:.4f}]")
    
    if kl_trust['mean'] > CONFIG['kl_divergence_threshold']:
        print(f"  ❌ DISTRIBUTIONAL SHIFT DETECTED (threshold: {CONFIG['kl_divergence_threshold']})")
        results['trust_distribution'] = 'FAILED'
    else:
        print(f"  ✅ Distribution match within acceptable range")
        results['trust_distribution'] = 'PASSED'
    
    # Impact distribution
    print("\n🔍 KL Divergence: Impact Index")
    human_impact = df_human['idx_impact'].dropna()
    synthetic_impact = df_synthetic['ensemble_impact'].dropna()
    
    kl_impact = bootstrap_kl_divergence(human_impact, synthetic_impact,
                                        CONFIG['bootstrap_iterations'])
    
    print(f"  Mean KL Divergence:  {kl_impact['mean']:.4f}")
    print(f"  95% CI:              [{kl_impact['ci_lower']:.4f}, {kl_impact['ci_upper']:.4f}]")
    
    if kl_impact['mean'] > CONFIG['kl_divergence_threshold']:
        print(f"  ❌ DISTRIBUTIONAL SHIFT DETECTED (threshold: {CONFIG['kl_divergence_threshold']})")
        results['impact_distribution'] = 'FAILED'
    else:
        print(f"  ✅ Distribution match within acceptable range")
        results['impact_distribution'] = 'PASSED'
    
    results['kl_trust'] = kl_trust
    results['kl_impact'] = kl_impact
    
    return results

def backstage_variance_collapse_check(df_human, df_synthetic):
    """
    BACKSTAGE CHECK 2: Variance Collapse Detection
    
    Tests whether synthetic population has sufficient variance
    compared to human population (Madden 2025).
    """
    print("\n" + "="*70)
    print("BACKSTAGE VALIDATION: Variance Collapse Detection")
    print("="*70)
    
    results = {}
    
    # Trust variance
    print("\n📊 Population Variance: Trust")
    human_trust_sd = df_human['idx_trust'].std()
    synthetic_trust_sd = df_synthetic['ensemble_trust'].std()
    variance_ratio_trust = synthetic_trust_sd / human_trust_sd
    
    print(f"  Human SD:            {human_trust_sd:.3f}")
    print(f"  Synthetic SD:        {synthetic_trust_sd:.3f}")
    print(f"  Ratio (Syn/Human):   {variance_ratio_trust:.3f}")
    
    if variance_ratio_trust < CONFIG['variance_collapse_threshold']:
        print(f"  ❌ HYPER-ACCURACY VARIANCE COLLAPSE (threshold: {CONFIG['variance_collapse_threshold']})")
        print(f"     Synthetic responses are too uniform!")
        results['trust_variance'] = 'FAILED'
    else:
        print(f"  ✅ Variance within acceptable range")
        results['trust_variance'] = 'PASSED'
    
    # Impact variance
    print("\n📊 Population Variance: Impact")
    human_impact_sd = df_human['idx_impact'].std()
    synthetic_impact_sd = df_synthetic['ensemble_impact'].std()
    variance_ratio_impact = synthetic_impact_sd / human_impact_sd
    
    print(f"  Human SD:            {human_impact_sd:.3f}")
    print(f"  Synthetic SD:        {synthetic_impact_sd:.3f}")
    print(f"  Ratio (Syn/Human):   {variance_ratio_impact:.3f}")
    
    if variance_ratio_impact < CONFIG['variance_collapse_threshold']:
        print(f"  ❌ HYPER-ACCURACY VARIANCE COLLAPSE (threshold: {CONFIG['variance_collapse_threshold']})")
        results['impact_variance'] = 'FAILED'
    else:
        print(f"  ✅ Variance within acceptable range")
        results['impact_variance'] = 'PASSED'
    
    results['trust_sd_human'] = human_trust_sd
    results['trust_sd_synthetic'] = synthetic_trust_sd
    results['trust_variance_ratio'] = variance_ratio_trust
    results['impact_sd_human'] = human_impact_sd
    results['impact_sd_synthetic'] = synthetic_impact_sd
    results['impact_variance_ratio'] = variance_ratio_impact
    
    return results

# =============================================================================
# FRONTSTAGE VALIDATION
# =============================================================================

def frontstage_pattern_correspondence(df_merged):
    """
    FRONTSTAGE CHECK 1: Cramer's V Pattern Correspondence
    
    Tests whether demographic-attitude relationships in synthetic data
    match those in human data.
    """
    print("\n" + "="*70)
    print("FRONTSTAGE VALIDATION: Pattern Correspondence (Cramer's V)")
    print("="*70)
    
    all_vars = (CONFIG['high_priority_vars'] + 
                CONFIG['secondary_vars'] + 
                CONFIG['exploratory_vars'])
    
    outcomes = ['idx_trust', 'idx_impact', 'ensemble_trust', 'ensemble_impact']
    
    results = []
    flags = []
    
    for var in all_vars:
        if var not in df_merged.columns:
            print(f"\n⚠️  Skipping {var} (not in dataset)")
            continue
        
        print(f"\n📈 Variable: {var}")
        
        # Discretize if continuous
        if df_merged[var].dtype in ['float64', 'int64'] and df_merged[var].nunique() > 10:
            var_data = discretize_continuous(df_merged[var])
        else:
            var_data = df_merged[var]
        
        # Calculate Cramer's V for each outcome
        cramers_human_trust = cramers_v(var_data, df_merged['idx_trust'])
        cramers_human_impact = cramers_v(var_data, df_merged['idx_impact'])
        cramers_synth_trust = cramers_v(var_data, df_merged['ensemble_trust'].round())
        cramers_synth_impact = cramers_v(var_data, df_merged['ensemble_impact'].round())
        
        # Calculate divergence
        delta_trust = abs(cramers_synth_trust - cramers_human_trust)
        delta_impact = abs(cramers_synth_impact - cramers_human_impact)
        
        print(f"  Trust:  Human={cramers_human_trust:.3f}, Synthetic={cramers_synth_trust:.3f}, Δ={delta_trust:.3f}")
        print(f"  Impact: Human={cramers_human_impact:.3f}, Synthetic={cramers_synth_impact:.3f}, Δ={delta_impact:.3f}")
        
        # Check thresholds
        priority = "HIGH" if var in CONFIG['high_priority_vars'] else "SECONDARY" if var in CONFIG['secondary_vars'] else "EXPLORATORY"
        
        if delta_trust > CONFIG['cramers_v_threshold']:
            print(f"  ⚠️  Trust pattern divergence exceeds threshold!")
            if priority == "HIGH":
                flags.append(f"CRITICAL: {var} × trust (Δ={delta_trust:.3f})")
        
        if delta_impact > CONFIG['cramers_v_threshold']:
            print(f"  ⚠️  Impact pattern divergence exceeds threshold!")
            if priority == "HIGH":
                flags.append(f"CRITICAL: {var} × impact (Δ={delta_impact:.3f})")
        
        results.append({
            'variable': var,
            'priority': priority,
            'cramers_v_human_trust': cramers_human_trust,
            'cramers_v_synthetic_trust': cramers_synth_trust,
            'delta_trust': delta_trust,
            'cramers_v_human_impact': cramers_human_impact,
            'cramers_v_synthetic_impact': cramers_synth_impact,
            'delta_impact': delta_impact,
            'trust_pass': delta_trust <= CONFIG['cramers_v_threshold'],
            'impact_pass': delta_impact <= CONFIG['cramers_v_threshold']
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary
    print("\n" + "="*70)
    print("PATTERN CORRESPONDENCE SUMMARY:")
    print("="*70)
    
    high_priority_results = results_df[results_df['priority'] == 'HIGH']
    
    if len(flags) > 0:
        print("\n❌ FIDELITY CHECK FAILED:")
        for flag in flags:
            print(f"  - {flag}")
        overall_status = "FAILED"
    else:
        print("\n✅ All high-priority variables passed pattern correspondence!")
        overall_status = "PASSED"
    
    return results_df, overall_status, flags

def frontstage_proportion_agreement(df_merged):
    """
    FRONTSTAGE CHECK 2: Proportion Agreement
    
    How often does the AI pick the same categorical response as humans?
    """
    print("\n" + "="*70)
    print("FRONTSTAGE VALIDATION: Proportion Agreement")
    print("="*70)
    
    # Round synthetic scores to nearest valid category
    synthetic_trust_rounded = df_merged['ensemble_trust'].round(1)
    synthetic_impact_rounded = df_merged['ensemble_impact'].round(1)
    
    # Calculate exact agreement
    trust_agreement = (synthetic_trust_rounded == df_merged['idx_trust']).mean()
    impact_agreement = (synthetic_impact_rounded == df_merged['idx_impact']).mean()
    
    # Calculate within-one-category agreement (0.0 vs 0.5, or 0.5 vs 1.0)
    trust_close = (abs(synthetic_trust_rounded - df_merged['idx_trust']) <= 0.5).mean()
    impact_close = (abs(synthetic_impact_rounded - df_merged['idx_impact']) <= 0.5).mean()
    
    print(f"\n📊 Trust Index:")
    print(f"  Exact agreement:     {100*trust_agreement:.1f}%")
    print(f"  Within 1 category:   {100*trust_close:.1f}%")
    
    print(f"\n📊 Impact Index:")
    print(f"  Exact agreement:     {100*impact_agreement:.1f}%")
    print(f"  Within 1 category:   {100*impact_close:.1f}%")
    
    return {
        'trust_exact': trust_agreement,
        'trust_close': trust_close,
        'impact_exact': impact_agreement,
        'impact_close': impact_close
    }

# =============================================================================
# EXECUTIVE SUMMARY GENERATOR
# =============================================================================

def generate_executive_summary(
    distributional_results,
    variance_results,
    pattern_results_df,
    pattern_status,
    pattern_flags,
    agreement_results
):
    """
    Generate plain-language executive summary for UNICEF stakeholders.
    """
    
    summary = []
    summary.append("="*70)
    summary.append("UNICEF SILICON SAMPLE FIDELITY REPORT")
    summary.append("Executive Summary for Non-Technical Stakeholders")
    summary.append("="*70)
    summary.append("")
    
    # Overall verdict
    summary.append("OVERALL FIDELITY ASSESSMENT:")
    summary.append("-"*70)
    
    all_checks = [
        distributional_results['trust_distribution'],
        distributional_results['impact_distribution'],
        variance_results['trust_variance'],
        variance_results['impact_variance'],
        pattern_status
    ]
    
    if all([c == 'PASSED' for c in all_checks]):
        summary.append("✅ HIGH FIDELITY - Simulation is scientifically valid")
        summary.append("")
        summary.append("The AI-generated donor responses closely match real human patterns.")
        summary.append("You can confidently use these results to inform intervention strategy.")
    elif pattern_status == 'FAILED':
        summary.append("❌ LOW FIDELITY - Pattern correspondence failed")
        summary.append("")
        summary.append("The AI does not correctly understand how demographics relate to")
        summary.append("donor attitudes. Results may not generalize to real interventions.")
        summary.append("")
        summary.append("Critical issues:")
        for flag in pattern_flags:
            summary.append(f"  - {flag}")
    else:
        summary.append("⚠️  MEDIUM FIDELITY - Some validation concerns")
        summary.append("")
        summary.append("The simulation shows some deviations from human patterns.")
        summary.append("Results should be interpreted with caution.")
    
    summary.append("")
    summary.append("="*70)
    summary.append("KEY METRICS:")
    summary.append("="*70)
    summary.append("")
    
    # Distributional fidelity
    summary.append("1. DISTRIBUTIONAL FIDELITY (How similar are the distributions?)")
    summary.append(f"   Trust KL Divergence:  {distributional_results['kl_trust']['mean']:.4f}")
    summary.append(f"   Impact KL Divergence: {distributional_results['kl_impact']['mean']:.4f}")
    summary.append(f"   Status: {distributional_results['trust_distribution']} / {distributional_results['impact_distribution']}")
    summary.append("   → Lower is better. Values near 0 indicate perfect match.")
    summary.append("")
    
    # Variance
    summary.append("2. POPULATION VARIANCE (Is the AI too uniform?)")
    summary.append(f"   Trust variance ratio:  {variance_results['trust_variance_ratio']:.3f}")
    summary.append(f"   Impact variance ratio: {variance_results['impact_variance_ratio']:.3f}")
    summary.append(f"   Status: {variance_results['trust_variance']} / {variance_results['impact_variance']}")
    summary.append("   → Should be close to 1.0. Below 0.7 indicates 'hyper-accuracy collapse'.")
    summary.append("")
    
    # Pattern correspondence
    summary.append("3. PATTERN CORRESPONDENCE (Does AI understand demographic relationships?)")
    high_priority = pattern_results_df[pattern_results_df['priority'] == 'HIGH']
    summary.append(f"   High-priority variables tested: {len(high_priority)}")
    trust_pass = high_priority['trust_pass'].sum()
    impact_pass = high_priority['impact_pass'].sum()
    summary.append(f"   Trust patterns matched:  {trust_pass}/{len(high_priority)}")
    summary.append(f"   Impact patterns matched: {impact_pass}/{len(high_priority)}")
    summary.append(f"   Status: {pattern_status}")
    summary.append("")
    
    # Agreement
    summary.append("4. PROPORTION AGREEMENT (How often does AI match human responses?)")
    summary.append(f"   Trust exact match:    {100*agreement_results['trust_exact']:.1f}%")
    summary.append(f"   Trust within 1 step:  {100*agreement_results['trust_close']:.1f}%")
    summary.append(f"   Impact exact match:   {100*agreement_results['impact_exact']:.1f}%")
    summary.append(f"   Impact within 1 step: {100*agreement_results['impact_close']:.1f}%")
    summary.append("")
    
    summary.append("="*70)
    summary.append("INTERPRETATION GUIDE:")
    summary.append("="*70)
    summary.append("")
    summary.append("What does this mean for UNICEF?")
    summary.append("")
    
    if all([c == 'PASSED' for c in all_checks]):
        summary.append("✅ The simulation successfully replicates real donor behavior patterns.")
        summary.append("   You can trust the intervention effect estimates to inform strategy.")
        summary.append("")
        summary.append("   Key strengths:")
        summary.append("   • AI responses show realistic diversity (not too uniform)")
        summary.append("   • Demographic relationships match real patterns (e.g., tenure → trust)")
        summary.append("   • Overall distribution matches human baseline")
    else:
        summary.append("⚠️  The simulation has some limitations:")
        summary.append("")
        if variance_results['trust_variance'] == 'FAILED':
            summary.append("   • AI responses are too uniform (lack human-like diversity)")
        if pattern_status == 'FAILED':
            summary.append("   • AI doesn't understand key demographic relationships")
        if distributional_results['trust_distribution'] == 'FAILED':
            summary.append("   • Overall distribution shifted from human baseline")
        summary.append("")
        summary.append("   Recommendation: Review flagged issues before making strategic decisions.")
    
    summary.append("")
    summary.append("="*70)
    summary.append(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("="*70)
    
    return "\n".join(summary)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete fidelity audit and generate reports."""
    
    print("\n" + "="*70)
    print("UNICEF SILICON SAMPLE FIDELITY AUDIT")
    print("="*70)
    
    # Load data
    print("\n📂 Loading simulation results...")
    try:
        df_ensemble = pd.read_csv(CONFIG['ensemble_results'])
        print(f"✓ Loaded ensemble results: {len(df_ensemble)} donors")
    except FileNotFoundError:
        print(f"❌ ERROR: '{CONFIG['ensemble_results']}' not found!")
        print("Run phase2_ensemble_simulation.py first!")
        return
    
    # Filter successful simulations
    df_successful = df_ensemble[df_ensemble['success'] == True].copy()
    print(f"✓ Successful simulations: {len(df_successful)}/{len(df_ensemble)}")
    
    if len(df_successful) == 0:
        print("❌ ERROR: No successful simulations to analyze!")
        return
    
    # Run backstage validation
    distributional_results = backstage_distributional_fidelity(
        df_successful, df_successful
    )
    
    variance_results = backstage_variance_collapse_check(
        df_successful, df_successful
    )
    
    # Run frontstage validation
    pattern_results_df, pattern_status, pattern_flags = frontstage_pattern_correspondence(
        df_successful
    )
    
    agreement_results = frontstage_proportion_agreement(df_successful)
    
    # Generate executive summary
    print("\n📝 Generating executive summary...")
    summary_text = generate_executive_summary(
        distributional_results,
        variance_results,
        pattern_results_df,
        pattern_status,
        pattern_flags,
        agreement_results
    )
    
    # Save reports
    print("\n💾 Saving fidelity reports...")
    
    # Executive summary (text)
    with open(CONFIG['output_report'], 'w') as f:
        f.write(summary_text)
    print(f"✓ Executive summary: '{CONFIG['output_report']}'")
    
    # Detailed metrics (CSV)
    detailed_metrics = pd.DataFrame([{
        'metric': 'KL_divergence_trust',
        'value': distributional_results['kl_trust']['mean'],
        'ci_lower': distributional_results['kl_trust']['ci_lower'],
        'ci_upper': distributional_results['kl_trust']['ci_upper'],
        'status': distributional_results['trust_distribution']
    }, {
        'metric': 'KL_divergence_impact',
        'value': distributional_results['kl_impact']['mean'],
        'ci_lower': distributional_results['kl_impact']['ci_lower'],
        'ci_upper': distributional_results['kl_impact']['ci_upper'],
        'status': distributional_results['impact_distribution']
    }, {
        'metric': 'variance_ratio_trust',
        'value': variance_results['trust_variance_ratio'],
        'ci_lower': None,
        'ci_upper': None,
        'status': variance_results['trust_variance']
    }, {
        'metric': 'variance_ratio_impact',
        'value': variance_results['impact_variance_ratio'],
        'ci_lower': None,
        'ci_upper': None,
        'status': variance_results['impact_variance']
    }])
    
    detailed_metrics.to_csv(CONFIG['output_detailed'], index=False)
    pattern_results_df.to_csv('pattern_correspondence_details.csv', index=False)
    
    print(f"✓ Detailed metrics: '{CONFIG['output_detailed']}'")
    print(f"✓ Pattern details: 'pattern_correspondence_details.csv'")
    
    # Print executive summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY (also saved to file):")
    print("="*70)
    print(summary_text)
    
    print("\n✅ Fidelity audit complete!\n")

if __name__ == "__main__":
    main()
