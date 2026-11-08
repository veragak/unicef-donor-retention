"""
UNICEF Silicon Sample Intervention Simulator - ENSEMBLE VERSION
================================================================
Implements Argyle et al. (2023) algorithmic fidelity standards
and Madden (2025) statistical guardrails.

Key Features:
- 5 independent draws per donor (880 total API calls)
- Independent draw enforcement (Martingale property)
- Ensemble averaging with variance tracking
- Real-time cost estimation
- Qwen 3 32B via Groq API
"""

import pandas as pd
import asyncio
import aiohttp
import json
import os
import time
import numpy as np
from typing import Dict, List
from datetime import datetime

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio", "--break-system-packages"])
    import nest_asyncio
    nest_asyncio.apply()

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # API Settings
    "groq_api_key": os.environ.get("GROQ_API_KEY", ""),  # set GROQ_API_KEY in your environment (see .env.example)
    "groq_model": "qwen/qwen3-32b",
    
    # Ensemble Settings
    "n_draws_per_donor": 5,  # Independent draws for each donor
    "temperature": 0.7,  # Add stochasticity for ensemble diversity
    "max_tokens": 800,
    "timeout_seconds": 45,
    "max_retries": 3,
    "concurrent_limit": 20,
    
    # File Paths
    "input_file": "fragile_donors_with_prompts.csv",
    "output_file": "ensemble_intervention_results.csv",
    "ensemble_details_file": "ensemble_details_all_draws.csv",
    
    # Cost Tracking
    "input_cost_per_million": 0.59,  # Qwen 3 32B
    "output_cost_per_million": 0.79,
    "avg_input_tokens": 3000,
    "avg_output_tokens": 200
}

# =============================================================================
# INTERVENTION DEFINITION
# =============================================================================

INTERVENTION = """A UNICEF representative is sent to your house to talk you through how and who your donation helps. They show you personal photos and impact reports tailored to the causes you care about."""

SYSTEM_PROMPT = """You are a sophisticated simulation engine for computational social science research. Your task is to embody a specific donor profile and respond authentically to an intervention.

CRITICAL INSTRUCTIONS:
1. Stay completely in character based on the donor backstory provided
2. Consider how their current trust, impact perception, alignment, and awareness would shape their reaction
3. Think about realistic human psychology - not all donors respond positively; some may feel intruded upon
4. Output ONLY valid JSON - no preamble, no markdown formatting, no explanation outside the JSON
5. Ensure numeric values match the specified constraints exactly

RESPONSE FORMAT (valid JSON only):
{
  "new_trust": 0.0 or 0.5 or 1.0,
  "new_impact": 0.0 or 0.5 or 1.0,
  "new_alignment_pct": float between 0.0 and 100.0,
  "new_perc_breadth": integer between 0 and 8,
  "new_info_spending_more": 0 or 1,
  "qualitative_rationale": "brief explanation in donor's voice (2-3 sentences max)"
}"""

def create_user_prompt(conditioning_prompt: str, intervention: str) -> str:
    """
    Create user prompt for a SINGLE independent draw.
    
    CRITICAL: This function is stateless. Each call is a fresh draw with NO memory
    of previous responses. This ensures the Martingale property (order-invariance).
    """
    return f"""{conditioning_prompt}

---

INTERVENTION SCENARIO:
{intervention}

---

As this specific donor, respond to the intervention by updating your attitudes and explaining your reaction.

Output a JSON object with these exact fields:
- new_trust: 0.0, 0.5, or 1.0
- new_impact: 0.0, 0.5, or 1.0  
- new_alignment_pct: number between 0-100
- new_perc_breadth: integer between 0-8
- new_info_spending_more: 0 or 1
- qualitative_rationale: brief explanation

Output ONLY the JSON object."""

# =============================================================================
# GROQ API CLIENT
# =============================================================================

class GroqClient:
    """Async Groq API client with independent draw guarantee."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    async def generate(self, session: aiohttp.ClientSession, 
                      system_prompt: str, user_prompt: str,
                      temperature: float, max_tokens: int) -> str:
        """
        Make async API call to Groq.
        
        INDEPENDENT DRAW GUARANTEE:
        Each call is completely stateless. No conversation history is maintained.
        The messages array is constructed fresh for each request.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # CRITICAL: Fresh message array for each call (no history)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with session.post(self.base_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Track token usage
            self.call_count += 1
            if 'usage' in data:
                self.total_input_tokens += data['usage'].get('prompt_tokens', 0)
                self.total_output_tokens += data['usage'].get('completion_tokens', 0)
            
            return data["choices"][0]["message"]["content"]
    
    def get_cost_estimate(self) -> Dict[str, float]:
        """Calculate current cost based on token usage."""
        input_cost = (self.total_input_tokens / 1_000_000) * CONFIG["input_cost_per_million"]
        output_cost = (self.total_output_tokens / 1_000_000) * CONFIG["output_cost_per_million"]
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens
        }

# =============================================================================
# SINGLE DRAW PROCESSOR
# =============================================================================

async def process_single_draw(
    donor_id: int,
    draw_number: int,
    conditioning_prompt: str,
    baseline_trust: float,
    baseline_impact: float,
    intervention: str,
    client: GroqClient,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore
) -> Dict:
    """
    Process ONE independent draw for ONE donor.
    
    INDEPENDENT DRAW ENFORCEMENT:
    - No conversation history
    - No access to other donors' responses
    - No access to this donor's previous draws
    - Fresh API call with only the conditioning prompt + intervention
    
    This ensures the Martingale property: order-invariance.
    """
    
    async with semaphore:
        # Create fresh prompt (stateless)
        user_prompt = create_user_prompt(conditioning_prompt, intervention)
        
        for attempt in range(CONFIG["max_retries"]):
            try:
                # INDEPENDENT DRAW: Fresh API call
                raw_response = await asyncio.wait_for(
                    client.generate(
                        session,
                        SYSTEM_PROMPT,
                        user_prompt,
                        CONFIG["temperature"],
                        CONFIG["max_tokens"]
                    ),
                    timeout=CONFIG["timeout_seconds"]
                )
                
                # Clean response (handle <think> tags from Qwen)
                clean_response = raw_response.strip()
                
                if "<think>" in clean_response:
                    clean_response = clean_response.split("</think>")[1].strip()
                
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
                if not clean_response.startswith("{"):
                    start = clean_response.find("{")
                    end = clean_response.rfind("}") + 1
                    if start != -1 and end != 0:
                        clean_response = clean_response[start:end]
                    else:
                        raise ValueError(f"No JSON object found")
                
                response_data = json.loads(clean_response)
                
                # Validate required fields
                required_fields = [
                    "new_trust", "new_impact", "new_alignment_pct",
                    "new_perc_breadth", "new_info_spending_more",
                    "qualitative_rationale"
                ]
                
                if not all(f in response_data for f in required_fields):
                    raise ValueError("Missing required fields")
                
                # Add metadata
                response_data["donor_id"] = donor_id
                response_data["draw_number"] = draw_number
                response_data["baseline_trust"] = baseline_trust
                response_data["baseline_impact"] = baseline_impact
                response_data["delta_trust"] = response_data["new_trust"] - baseline_trust
                response_data["delta_impact"] = response_data["new_impact"] - baseline_impact
                response_data["success"] = True
                response_data["attempts"] = attempt + 1
                
                return response_data
                
            except Exception as e:
                if attempt < CONFIG["max_retries"] - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {
                        "donor_id": donor_id,
                        "draw_number": draw_number,
                        "new_trust": None,
                        "new_impact": None,
                        "new_alignment_pct": None,
                        "new_perc_breadth": None,
                        "new_info_spending_more": None,
                        "qualitative_rationale": f"FAILED: {str(e)[:100]}",
                        "baseline_trust": baseline_trust,
                        "baseline_impact": baseline_impact,
                        "delta_trust": None,
                        "delta_impact": None,
                        "success": False,
                        "attempts": CONFIG["max_retries"]
                    }

# =============================================================================
# ENSEMBLE PROCESSOR
# =============================================================================

async def process_donor_ensemble(
    donor_id: int,
    row: pd.Series,
    client: GroqClient,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore
) -> List[Dict]:
    """
    Process N independent draws for a single donor and return all draws.
    
    Each draw is completely independent - satisfies Argyle et al. (2023)
    requirement for "silicon subjects" with no shared memory.
    """
    
    conditioning_prompt = row['conditioning_prompt']
    baseline_trust = row.get('idx_trust', 0.0)
    baseline_impact = row.get('idx_impact', 0.0)
    
    # Create N independent draw tasks
    tasks = []
    for draw_num in range(CONFIG["n_draws_per_donor"]):
        task = process_single_draw(
            donor_id=donor_id,
            draw_number=draw_num,
            conditioning_prompt=conditioning_prompt,
            baseline_trust=baseline_trust,
            baseline_impact=baseline_impact,
            intervention=INTERVENTION,
            client=client,
            session=session,
            semaphore=semaphore
        )
        tasks.append(task)
    
    # Execute all draws in parallel (order-invariant)
    results = await asyncio.gather(*tasks)
    
    return results

def aggregate_ensemble(all_draws: List[Dict]) -> Dict:
    """
    Aggregate multiple draws into ensemble statistics.
    
    Returns:
    - Mean values (point estimates)
    - Standard deviations (within-donor variance)
    - All individual draw values
    """
    
    donor_id = all_draws[0]['donor_id']
    baseline_trust = all_draws[0]['baseline_trust']
    baseline_impact = all_draws[0]['baseline_impact']
    
    # Extract successful draws
    successful_draws = [d for d in all_draws if d['success']]
    
    if len(successful_draws) == 0:
        return {
            "donor_id": donor_id,
            "n_successful_draws": 0,
            "ensemble_trust": None,
            "ensemble_impact": None,
            "ensemble_alignment": None,
            "ensemble_breadth": None,
            "ensemble_info_spending": None,
            "sd_trust": None,
            "sd_impact": None,
            "baseline_trust": baseline_trust,
            "baseline_impact": baseline_impact,
            "delta_trust": None,
            "delta_impact": None,
            "success": False
        }
    
    # Calculate ensemble means
    trust_values = [d['new_trust'] for d in successful_draws]
    impact_values = [d['new_impact'] for d in successful_draws]
    alignment_values = [d['new_alignment_pct'] for d in successful_draws]
    breadth_values = [d['new_perc_breadth'] for d in successful_draws]
    info_spending_values = [d['new_info_spending_more'] for d in successful_draws]
    
    ensemble_trust = np.mean(trust_values)
    ensemble_impact = np.mean(impact_values)
    
    return {
        "donor_id": donor_id,
        "n_successful_draws": len(successful_draws),
        "ensemble_trust": ensemble_trust,
        "ensemble_impact": ensemble_impact,
        "ensemble_alignment": np.mean(alignment_values),
        "ensemble_breadth": np.mean(breadth_values),
        "ensemble_info_spending": np.mean(info_spending_values),
        "sd_trust": np.std(trust_values, ddof=1) if len(trust_values) > 1 else 0.0,
        "sd_impact": np.std(impact_values, ddof=1) if len(impact_values) > 1 else 0.0,
        "baseline_trust": baseline_trust,
        "baseline_impact": baseline_impact,
        "delta_trust": ensemble_trust - baseline_trust,
        "delta_impact": ensemble_impact - baseline_impact,
        "all_trust_draws": trust_values,
        "all_impact_draws": impact_values,
        "success": True
    }

# =============================================================================
# MAIN SIMULATION ORCHESTRATOR
# =============================================================================

async def run_ensemble_simulation(df: pd.DataFrame) -> tuple:
    """
    Run ensemble simulation with independent draw enforcement.
    
    Returns:
    - ensemble_results: Aggregated results (176 rows)
    - all_draws: Individual draw results (880 rows)
    """
    
    client = GroqClient(CONFIG["groq_api_key"], CONFIG["groq_model"])
    semaphore = asyncio.Semaphore(CONFIG["concurrent_limit"])
    
    total_donors = len(df)
    total_draws = total_donors * CONFIG["n_draws_per_donor"]
    
    print(f"\n{'='*70}")
    print(f"🚀 UNICEF ENSEMBLE INTERVENTION SIMULATOR")
    print(f"{'='*70}")
    print(f"Model:           {CONFIG['groq_model']}")
    print(f"Donors:          {total_donors}")
    print(f"Draws per donor: {CONFIG['n_draws_per_donor']}")
    print(f"Total API calls: {total_draws}")
    print(f"Temperature:     {CONFIG['temperature']} (stochastic for ensemble)")
    print(f"Parallel:        {CONFIG['concurrent_limit']} concurrent requests")
    print(f"{'='*70}")
    print(f"\n💰 COST ESTIMATE:")
    est_input_cost = (total_draws * CONFIG["avg_input_tokens"] / 1_000_000) * CONFIG["input_cost_per_million"]
    est_output_cost = (total_draws * CONFIG["avg_output_tokens"] / 1_000_000) * CONFIG["output_cost_per_million"]
    print(f"Estimated: ${est_input_cost + est_output_cost:.2f}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Process all donors
    timeout = aiohttp.ClientTimeout(total=CONFIG["timeout_seconds"])
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for idx, row in df.iterrows():
            task = process_donor_ensemble(
                donor_id=idx,
                row=row,
                client=client,
                session=session,
                semaphore=semaphore
            )
            tasks.append(task)
        
        print(f"🎲 Launching {total_draws} independent draws...\n")
        all_donor_results = await asyncio.gather(*tasks)
    
    elapsed_time = time.time() - start_time
    
    # Flatten all draws into single list
    all_draws = []
    for donor_draws in all_donor_results:
        all_draws.extend(donor_draws)
    
    # Aggregate ensembles
    ensemble_results = []
    for donor_draws in all_donor_results:
        agg = aggregate_ensemble(donor_draws)
        ensemble_results.append(agg)
    
    # Get actual costs
    costs = client.get_cost_estimate()
    
    # Print summary
    all_draws_df = pd.DataFrame(all_draws)
    ensemble_df = pd.DataFrame(ensemble_results)
    
    print(f"\n{'='*70}")
    print(f"✅ SIMULATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total draws:     {len(all_draws_df)}")
    print(f"Successful:      {all_draws_df['success'].sum()} ({100*all_draws_df['success'].mean():.1f}%)")
    print(f"Failed:          {(~all_draws_df['success']).sum()}")
    print(f"Time:            {elapsed_time/60:.1f} minutes")
    print(f"\n💰 ACTUAL COST:")
    print(f"Input tokens:    {costs['input_tokens']:,}")
    print(f"Output tokens:   {costs['output_tokens']:,}")
    print(f"Total cost:      ${costs['total_cost']:.2f}")
    print(f"{'='*70}\n")
    
    return ensemble_df, all_draws_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""
    
    if not CONFIG["groq_api_key"]:
        print("❌ ERROR: Please set the GROQ_API_KEY environment variable!")
        return
    
    print("📂 Loading donor data...")
    try:
        df = pd.read_csv(CONFIG['input_file'])
        print(f"✓ Loaded {len(df)} donors\n")
    except FileNotFoundError:
        print(f"❌ ERROR: '{CONFIG['input_file']}' not found!")
        print("Run phase1_prompt_generation.py first!")
        return
    
    if 'conditioning_prompt' not in df.columns:
        print("❌ ERROR: 'conditioning_prompt' column not found!")
        return
    
    # Verify baseline columns
    required_cols = ['idx_trust', 'idx_impact']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠️  WARNING: Missing baseline columns: {missing}")
        print("Will use 0.0 as default baseline values.\n")
    
    print("🔄 Starting ensemble simulation...\n")
    
    # Run simulation
    ensemble_df, all_draws_df = asyncio.run(run_ensemble_simulation(df))
    
    # Merge with original data
    print("🔗 Merging with original data...")
    final_ensemble = df.merge(
        ensemble_df,
        left_index=True,
        right_on='donor_id',
        how='left'
    )
    
    # Save results
    print("💾 Saving results...")
    final_ensemble.to_csv(CONFIG['output_file'], index=False)
    all_draws_df.to_csv(CONFIG['ensemble_details_file'], index=False)
    
    print(f"✓ Ensemble results: '{CONFIG['output_file']}'")
    print(f"✓ All draws (880):  '{CONFIG['ensemble_details_file']}'\n")
    
    # Summary statistics
    if ensemble_df['success'].sum() > 0:
        successful = ensemble_df[ensemble_df['success'] == True]
        
        print("📊 ENSEMBLE SUMMARY:")
        print(f"\n  PRE-INTERVENTION BASELINE:")
        if 'idx_trust' in df.columns:
            print(f"    Avg Trust:     {df['idx_trust'].mean():.2f}")
            print(f"    SD Trust:      {df['idx_trust'].std():.2f}")
        if 'idx_impact' in df.columns:
            print(f"    Avg Impact:    {df['idx_impact'].mean():.2f}")
            print(f"    SD Impact:     {df['idx_impact'].std():.2f}")
        
        print(f"\n  POST-INTERVENTION (ENSEMBLED):")
        print(f"    Avg Trust:     {successful['ensemble_trust'].mean():.2f}")
        print(f"    SD Trust:      {successful['ensemble_trust'].std():.2f}")
        print(f"    Avg Impact:    {successful['ensemble_impact'].mean():.2f}")
        print(f"    SD Impact:     {successful['ensemble_impact'].std():.2f}")
        
        print(f"\n  WITHIN-DONOR VARIANCE (avg across donors):")
        print(f"    Avg SD Trust:  {successful['sd_trust'].mean():.3f}")
        print(f"    Avg SD Impact: {successful['sd_impact'].mean():.3f}")
        
        print(f"\n  CHANGES:")
        print(f"    Δ Trust:       {successful['delta_trust'].mean():+.2f}")
        print(f"    Δ Impact:      {successful['delta_impact'].mean():+.2f}")
    
    print(f"\n{'='*70}")
    print("✅ READY FOR FIDELITY AUDIT (Phase 3)!")
    print(f"{'='*70}")
    print("\nNext step: Run phase3_fidelity_report.py\n")

if __name__ == "__main__":
    main()
