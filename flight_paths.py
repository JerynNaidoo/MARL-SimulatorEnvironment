import numpy as np
import matplotlib.pyplot as plt
import torch
from final import DirectBlueSkyEnv, BlueSkyPreprocessor, DecentralisedActor, GeometricRuleBaseline

def record_flight_paths(env, policy_fn, is_marl=False, preprocessor=None):
    raw_obs, _ = env.reset(num_aircraft=10)
    trajectories = {ac_id: {'x': [], 'y': []} for ac_id in raw_obs['id']}
    
    for _ in range(60):
        for ac_id, x, y in zip(raw_obs['id'], raw_obs['x_nm'], raw_obs['y_nm']):
            trajectories[ac_id]['x'].append(x)
            trajectories[ac_id]['y'].append(y)
            
        if is_marl:
            obs, _ = preprocessor.process(raw_obs)
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            actions = policy_fn(obs_t).cpu().numpy() if obs_t.shape[0] > 0 else None
        else:
            actions = policy_fn(raw_obs)
            
        raw_obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            break
            
    return trajectories

def generate_spatial_comparison():
    env = DirectBlueSkyEnv(dt=5.0, reward_scale=0.05)
    preprocessor = BlueSkyPreprocessor(k_nearest=5, max_aircraft=10)
    actor = DecentralisedActor(obs_dim=29, action_dim=2)
    actor.load_state_dict(torch.load("actor_policy_final.pt"))
    actor.eval()

    with torch.no_grad():
        print("Simulating TCAS flight paths")
        tcas_paths = record_flight_paths(env, GeometricRuleBaseline.select_actions, is_marl=False)
        print("Simulating MAPPO flight paths")
        mappo_paths = record_flight_paths(env, actor, is_marl=True, preprocessor=preprocessor)

    env.close()

    # Generate Side-by-Side Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ac_id, path in tcas_paths.items():
        ax1.plot(path['x'], path['y'], label=ac_id, lw=1.5)
        ax1.scatter(path['x'][0], path['y'][0], marker='o') # Start
        ax1.scatter(path['x'][-1], path['y'][-1], marker='x') # End
    ax1.set_title("Rule-Based TCAS Baseline\n(Sharp turns, cascading secondary conflicts)", fontsize=11)
    ax1.set_xlabel("X Distance (NM)")
    ax1.set_ylabel("Y Distance (NM)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for ac_id, path in mappo_paths.items():
        ax2.plot(path['x'], path['y'], label=ac_id, lw=1.5)
        ax2.scatter(path['x'][0], path['y'][0], marker='o') # Start
        ax2.scatter(path['x'][-1], path['y'][-1], marker='x') # End
    ax2.set_title("Proposed MAPPO Policy\n(Smooth, cooperative multi-aircraft sequencing)", fontsize=11)
    ax2.set_xlabel("X Distance (NM)")
    ax2.set_ylabel("Y Distance (NM)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("trajectory_comparison.png", dpi=300)
    print("Saved trajectory comparison to 'trajectory_comparison.png'.")

if __name__ == "__main__":
    generate_spatial_comparison()