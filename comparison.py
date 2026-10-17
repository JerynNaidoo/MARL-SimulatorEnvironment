import torch
import numpy as np
import pandas as pd
from final import DirectBlueSkyEnv, BlueSkyPreprocessor, DecentralisedActor, GeometricRuleBaseline

def run_final_paper_benchmark(model_path="actor_policy_final.pt", num_eval_episodes=50):

    env = DirectBlueSkyEnv(dt=5.0, reward_scale=0.05)
    preprocessor = BlueSkyPreprocessor(k_nearest=5, max_aircraft=10)

    actor = DecentralisedActor(obs_dim=29, action_dim=2)
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    marl_results = {"CRR": [], "Breaches": [], "Total_Conflicts": []}
    tcas_results = {"CRR": [], "Breaches": [], "Total_Conflicts": []}

    with torch.no_grad():
        for ep in range(1, num_eval_episodes + 1):
            # 1. Evaluate Rule-Based Baseline (TCAS-like)
            raw_obs, _ = env.reset(num_aircraft=10)
            for _ in range(60):
                actions = GeometricRuleBaseline.select_actions(raw_obs)
                raw_obs, _, term, trunc, info = env.step(actions)
                if term or trunc:
                    break
            tcas_results["CRR"].append(info["conflict_resolution_rate"])
            tcas_results["Breaches"].append(info["separation_breaches"])
            tcas_results["Total_Conflicts"].append(info["total_conflicts"])

            # 2. Evaluate MARL Policy (Deterministic Greedy Mode: a = mu)
            raw_obs, _ = env.reset(num_aircraft=10)
            obs, _ = preprocessor.process(raw_obs)
            for _ in range(60):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                # Deterministic forward pass (zero exploration noise)
                actions = actor(obs_t).cpu().numpy() if obs_t.shape[0] > 0 else None
                raw_obs, _, term, trunc, info = env.step(actions)
                obs, _ = preprocessor.process(raw_obs)
                if term or trunc:
                    break
            marl_results["CRR"].append(info["conflict_resolution_rate"])
            marl_results["Breaches"].append(info["separation_breaches"])
            marl_results["Total_Conflicts"].append(info["total_conflicts"])

            if ep % 10 == 0:
                print(f"Completed Episode {ep}/{num_eval_episodes}...")

    env.close()

    summary_df = pd.DataFrame({
        "Metric": [
            "Mean Conflict Resolution Rate (%)",
            "Mean Separation Breaches per Sector",
            "Zero-Breach Mission Success Rate (%)"
        ],
        "Geometric Baseline (TCAS)": [
            f"{np.mean(tcas_results['CRR']):.2f}% ± {np.std(tcas_results['CRR']):.2f}",
            f"{np.mean(tcas_results['Breaches']):.2f}",
            f"{(np.sum(np.array(tcas_results['Breaches']) == 0) / num_eval_episodes) * 100:.1f}%"
        ],
        "Proposed MAPPO Model (Ours)": [
            f"{np.mean(marl_results['CRR']):.2f}% ± {np.std(marl_results['CRR']):.2f}",
            f"{np.mean(marl_results['Breaches']):.2f}",
            f"{(np.sum(np.array(marl_results['Breaches']) == 0) / num_eval_episodes) * 100:.1f}%"
        ]
    })

    print(summary_df.to_string(index=False))
    summary_df.to_csv("final_thesis_benchmark_table.csv", index=False)
    print("\nSaved formal comparison table to 'final_thesis_benchmark_table.csv'.")

if __name__ == "__main__":
    run_final_paper_benchmark()