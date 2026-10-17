import os
import sys
import math
import random
import contextlib
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import bluesky as bs


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ==============================================================================
# 1. NEURAL NETWORK ARCHITECTURES
# ==============================================================================

class DecentralisedActor(nn.Module):
    def __init__(self, obs_dim: int = 29, action_dim: int = 2, init_log_std: float = -0.7):
        super().__init__()
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.LayerNorm(128),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.Tanh()
        )
        self.mean_head = layer_init(nn.Linear(128, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * init_log_std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return torch.tanh(self.mean_head(features))

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        action_mean = self.forward(obs)
        clamped_log_std = torch.clamp(self.log_std, -2.0, 0.5)
        action_std = torch.exp(clamped_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        action = action_mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, action_mean

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        action_mean = self.forward(obs)
        clamped_log_std = torch.clamp(self.log_std, -2.0, 0.5)
        action_std = torch.exp(clamped_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CentralisedCritic(nn.Module):
    def __init__(self, num_aircraft: int = 10):
        super().__init__()
        self.global_state_dim = (num_aircraft * 29) + 4

        self.value_network = nn.Sequential(
            layer_init(nn.Linear(self.global_state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.value_network(global_state)



class BlueSkyPreprocessor:
    def __init__(
        self,
        k_nearest: int = 5,
        max_aircraft: int = 10,
        sector_radius_nm: float = 35.0,
        v_min_kts: float = 150.0,
        v_max_kts: float = 350.0,
        separation_min_nm: float = 5.0
    ):
        self.k = k_nearest
        self.max_aircraft = max_aircraft
        self.r_max = sector_radius_nm
        self.v_min = v_min_kts
        self.v_max = v_max_kts
        self.d_sep = separation_min_nm

        self.ownship_dim = 4
        self.intruder_dim = 5
        self.obs_dim = self.ownship_dim + (self.k * self.intruder_dim)  # 29
        self.metadata_dim = 4
        self.global_state_dim = (self.max_aircraft * self.obs_dim) + self.metadata_dim  # 294

    @staticmethod
    def rotate_to_body_frame(dx: float, dy: float, heading_deg: float) -> Tuple[float, float]:
        hdg_rad = np.radians(heading_deg)
        x_body = dx * np.cos(hdg_rad) - dy * np.sin(hdg_rad)
        y_body = dx * np.sin(hdg_rad) + dy * np.cos(hdg_rad)
        return x_body, y_body

    def process(self, raw_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        n_active = len(raw_data['id'])
        if n_active == 0:
            return np.zeros((0, self.obs_dim), dtype=np.float32), np.zeros(self.global_state_dim, dtype=np.float32)

        actor_obs_list = []

        for i in range(n_active):
            v_i = raw_data['tas_kts'][i]
            hdg_i = raw_data['hdg_deg'][i]
            x_i, y_i = raw_data['x_nm'][i], raw_data['y_nm'][i]
            wpx_i, wpy_i = raw_data['wp_x_nm'][i], raw_data['wp_y_nm'][i]

            v_norm = np.clip((v_i - self.v_min) / (self.v_max - self.v_min), 0.0, 1.0)
            hdg_norm = ((((hdg_i + 180.0) % 360.0) - 180.0)) / 180.0

            dx_wp = wpx_i - x_i
            dy_wp = wpy_i - y_i
            dist_wp = np.hypot(dx_wp, dy_wp)
            dist_wp_norm = np.clip(dist_wp / self.r_max, 0.0, 1.0)

            bearing_wp = np.degrees(np.arctan2(dx_wp, dy_wp))
            rel_bearing_wp = (((bearing_wp - hdg_i + 180.0) % 360.0) - 180.0) / 180.0

            ownship_feat = [v_norm, hdg_norm, dist_wp_norm, rel_bearing_wp]

            vx_i = v_i * np.sin(np.radians(hdg_i))
            vy_i = v_i * np.cos(np.radians(hdg_i))

            threat_scored_intruders = []
            for j in range(n_active):
                if i == j:
                    continue
                dx_ij = raw_data['x_nm'][j] - x_i
                dy_ij = raw_data['y_nm'][j] - y_i
                dist_ij = np.hypot(dx_ij, dy_ij)

                vx_j = raw_data['tas_kts'][j] * np.sin(np.radians(raw_data['hdg_deg'][j]))
                vy_j = raw_data['tas_kts'][j] * np.cos(np.radians(raw_data['hdg_deg'][j]))

                dvx = vx_j - vx_i
                dvy = vy_j - vy_i
                dv2 = dvx**2 + dvy**2

                threat_score = 1.0 / max(dist_ij, 0.1)
                if dv2 > 1e-4:
                    t_cpa_hr = -(dx_ij * dvx + dy_ij * dvy) / dv2
                    t_cpa_sec = t_cpa_hr * 3600.0
                    if 0.0 < t_cpa_sec < 180.0:
                        d_cpa = np.hypot(dx_ij + dvx * t_cpa_hr, dy_ij + dvy * t_cpa_hr)
                        if d_cpa < 5.0:
                            threat_score += 3.0 * (1.0 - (d_cpa / 5.0)) * math.exp(-t_cpa_sec / 60.0)

                threat_scored_intruders.append((threat_score, dist_ij, j, dx_ij, dy_ij, vx_j, vy_j))

            threat_scored_intruders.sort(key=lambda item: item[0], reverse=True)
            top_intruders = threat_scored_intruders[:self.k]

            temp_intruders = []
            for _, dist_ij, j, dx_ij, dy_ij, vx_j, vy_j in top_intruders:
                rx_body, ry_body = self.rotate_to_body_frame(dx_ij, dy_ij, hdg_i)
                rvx_body, rvy_body = self.rotate_to_body_frame(vx_j - vx_i, vy_j - vy_i, hdg_i)

                temp_intruders.extend([
                    np.clip(rx_body / self.r_max, -1.0, 1.0),
                    np.clip(ry_body / self.r_max, -1.0, 1.0),
                    np.clip(rvx_body / self.v_max, -1.0, 1.0),
                    np.clip(rvy_body / self.v_max, -1.0, 1.0),
                    np.clip(dist_ij / self.r_max, 0.0, 1.0)
                ])

            missing = self.k - len(top_intruders)
            if missing > 0:
                temp_intruders.extend([0.0] * (missing * self.intruder_dim))

            actor_obs_list.append(np.array(ownship_feat + temp_intruders, dtype=np.float32))

        actor_obs_matrix = np.vstack(actor_obs_list)
        g_sector = self._compute_sector_metadata(raw_data, n_active)
        global_state = self._construct_global_state(actor_obs_matrix, g_sector)

        return actor_obs_matrix, global_state

    def _compute_sector_metadata(self, raw_data: Dict[str, np.ndarray], n_active: int) -> np.ndarray:
        g1_density = n_active / self.max_aircraft
        conflicts = 0
        min_dist_sector = self.r_max

        for i in range(n_active):
            for j in range(i + 1, n_active):
                dist = np.hypot(raw_data['x_nm'][i] - raw_data['x_nm'][j], raw_data['y_nm'][i] - raw_data['y_nm'][j])
                if dist < min_dist_sector:
                    min_dist_sector = dist
                if dist < self.d_sep:
                    conflicts += 1

        g2_conflict_ratio = (conflicts / n_active) if n_active > 0 else 0.0

        progresses = []
        for i in range(n_active):
            dist_curr = np.hypot(raw_data['wp_x_nm'][i] - raw_data['x_nm'][i], raw_data['wp_y_nm'][i] - raw_data['y_nm'][i])
            dist_init = max(raw_data['wp_dist_init_nm'][i], 1.0)
            progresses.append(np.clip(1.0 - (dist_curr / dist_init), 0.0, 1.0))

        g3_avg_progress = float(np.mean(progresses)) if len(progresses) > 0 else 0.0
        g4_stress = np.clip(1.0 - (min_dist_sector / self.d_sep), 0.0, 1.0)

        return np.array([g1_density, g2_conflict_ratio, g3_avg_progress, g4_stress], dtype=np.float32)

    def _construct_global_state(self, actor_obs_matrix: np.ndarray, g_sector: np.ndarray) -> np.ndarray:
        max_capacity = self.max_aircraft * self.obs_dim
        padded_obs = np.zeros(max_capacity, dtype=np.float32)

        flat_active = actor_obs_matrix.flatten()
        actual_size = min(flat_active.shape[0], max_capacity)
        padded_obs[:actual_size] = flat_active[:actual_size]

        return np.concatenate([padded_obs, g_sector.flatten()])


class DirectBlueSkyEnv(gym.Env):
    def __init__(self, dt: float = 5.0, ref_lat: float = 52.0, ref_lon: float = 4.0, reward_scale: float = 0.05):
        super().__init__()
        self.dt = dt
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.reward_scale = reward_scale 

        self.waypoints = {}
        self.wp_dist_init_dict = {}
        self.prev_wp_dist_dict = {}

        self.warned_pairs = set()
        self.breached_pairs = set()

        with suppress_stdout():
            bs.init(mode="sim")
            bs.stack.stack(f"DT {self.dt}")

    def reset_scenario_procedural(self, num_aircraft: int = 4) -> Dict[str, np.ndarray]:
        with suppress_stdout():
            bs.stack.stack("RESET")
            bs.stack.stack(f"DT {self.dt}")
            bs.stack.stack("PAN 52.0 4.0")

            self.waypoints.clear()
            self.wp_dist_init_dict.clear()
            self.prev_wp_dist_dict.clear()
            self.warned_pairs.clear()
            self.breached_pairs.clear()

            for i in range(num_aircraft):
                ac_id = f"AC{i+1:02d}"
                angle = (2.0 * math.pi * i) / num_aircraft

                r_spawn = random.uniform(22.0, 28.0)
                x_spawn = r_spawn * math.cos(angle)
                y_spawn = r_spawn * math.sin(angle)

                wp_x = -x_spawn + random.uniform(-2.0, 2.0)
                wp_y = -y_spawn + random.uniform(-2.0, 2.0)
                self.waypoints[ac_id] = (wp_x, wp_y)

                hdg = (math.degrees(math.atan2(wp_x - x_spawn, wp_y - y_spawn)) + 360.0) % 360.0
                spd = 250.0 + random.uniform(-10.0, 10.0)
                hdg = (hdg + random.uniform(-2.0, 2.0)) % 360.0

                lat = self.ref_lat + (y_spawn / 60.0)
                lon = self.ref_lon + (x_spawn / (60.0 * math.cos(math.radians(self.ref_lat))))

                bs.stack.stack(f"CRE {ac_id} B737 {lat:.5f} {lon:.5f} {hdg:.1f} 30000 {spd:.1f}")

            bs.stack.stack("OP")
            for _ in range(3):
                bs.sim.step()

        return self._extract_raw_bluesky_state()

    def reset(self, num_aircraft: int = 4, seed: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        raw_obs = self.reset_scenario_procedural(num_aircraft=num_aircraft)
        return raw_obs, {}

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, bool, bool, Dict[str, Any]]:
        raw_obs = self._extract_raw_bluesky_state()
        active_ids = list(raw_obs['id'])
        n_active = len(active_ids)

        if n_active > 0 and actions is not None and len(actions) == n_active:
            tr = self._get_traffic_instance()
            for i in range(n_active):
                ac_id = active_ids[i]
                if hasattr(tr, 'id') and ac_id in tr.id:
                    act_hdg = float(np.clip(actions[i, 0], -1.0, 1.0))
                    act_spd = float(np.clip(actions[i, 1], -1.0, 1.0))

                    hdg_delta = act_hdg * 15.0
                    spd_delta = act_spd * 10.0

                    new_hdg = (float(raw_obs['hdg_deg'][i]) + hdg_delta) % 360.0
                    new_spd = float(np.clip(float(raw_obs['tas_kts'][i]) + spd_delta, 150.0, 350.0))

                    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        try:
                            bs.stack.stack(f"HDG {ac_id} {new_hdg:.1f}")
                            bs.stack.stack(f"SPD {ac_id} {new_spd:.1f}")
                        except Exception:
                            pass

        try:
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                bs.sim.step()
        except Exception:
            bs.stack.clear()

        next_raw_obs = self._extract_raw_bluesky_state()
        next_ids = list(next_raw_obs['id'])

        raw_rewards = np.zeros(n_active, dtype=np.float32)

        if n_active > 0:
            for i in range(n_active):
                ac_id = active_ids[i]

                if actions is not None and i < len(actions):
                    raw_rewards[i] -= 0.005 * float(np.sum(np.clip(actions[i], -1.0, 1.0)**2))

                if ac_id in next_ids:
                    next_idx = next_ids.index(ac_id)
                    curr_x = float(next_raw_obs['x_nm'][next_idx])
                    curr_y = float(next_raw_obs['y_nm'][next_idx])
                    wpx = float(next_raw_obs['wp_x_nm'][next_idx])
                    wpy = float(next_raw_obs['wp_y_nm'][next_idx])

                    curr_dist = float(np.hypot(wpx - curr_x, wpy - curr_y))
                    prev_dist = self.prev_wp_dist_dict.get(ac_id, curr_dist)
                    delta_dist_nm = prev_dist - curr_dist

                    if delta_dist_nm > 0:
                        raw_rewards[i] += float(np.clip(delta_dist_nm * 0.3, 0.0, 0.5))
                    self.prev_wp_dist_dict[ac_id] = curr_dist

                    for j in range(i + 1, n_active):
                        other_id = active_ids[j]
                        if other_id in next_ids:
                            other_idx = next_ids.index(other_id)
                            dx = float(next_raw_obs['x_nm'][other_idx]) - curr_x
                            dy = float(next_raw_obs['y_nm'][other_idx]) - curr_y
                            dist_nm = float(np.hypot(dx, dy))
                            pair_key = tuple(sorted([ac_id, other_id]))

                            if dist_nm < 5.0:
                                self.breached_pairs.add(pair_key)
                                self.warned_pairs.add(pair_key)
                                sep_penalty = -25.0 * (1.0 + (5.0 - dist_nm) / 5.0)
                                raw_rewards[i] += sep_penalty
                                raw_rewards[j] += sep_penalty

                            elif dist_nm < 8.0:
                                self.warned_pairs.add(pair_key)
                                warn_penalty = -1.5 * ((8.0 - dist_nm) / 3.0)
                                raw_rewards[i] += warn_penalty
                                raw_rewards[j] += warn_penalty

                            vx_i = float(next_raw_obs['tas_kts'][next_idx]) * math.sin(math.radians(float(next_raw_obs['hdg_deg'][next_idx])))
                            vy_i = float(next_raw_obs['tas_kts'][next_idx]) * math.cos(math.radians(float(next_raw_obs['hdg_deg'][next_idx])))
                            vx_j = float(next_raw_obs['tas_kts'][other_idx]) * math.sin(math.radians(float(next_raw_obs['hdg_deg'][other_idx])))
                            vy_j = float(next_raw_obs['tas_kts'][other_idx]) * math.cos(math.radians(float(next_raw_obs['hdg_deg'][other_idx])))

                            dvx = vx_j - vx_i
                            dvy = vy_j - vy_i
                            dv2 = dvx**2 + dvy**2

                            if dv2 > 1e-4:
                                t_cpa_hr = -(dx * dvx + dy * dvy) / dv2
                                t_cpa_sec = t_cpa_hr * 3600.0
                                if 0.0 < t_cpa_sec < 180.0:
                                    d_cpa = math.hypot(dx + dvx * t_cpa_hr, dy + dvy * t_cpa_hr)
                                    if d_cpa < 5.0:
                                        cpa_penalty = -2.5 * (1.0 - (d_cpa / 5.0)) * math.exp(-t_cpa_sec / 60.0)
                                        raw_rewards[i] += cpa_penalty
                                        raw_rewards[j] += cpa_penalty
                else:
                    last_known = self.prev_wp_dist_dict.get(ac_id, 999.0)
                    raw_rewards[i] += 5.0 if last_known < 5.0 else 0.5

        scaled_rewards = raw_rewards * self.reward_scale

        terminated = (len(next_ids) == 0)
        truncated = False

        total_conflicts = len(self.warned_pairs.union(self.breached_pairs))
        unique_breaches = len(self.breached_pairs)
        resolved_conflicts = max(0, total_conflicts - unique_breaches)
        resolution_rate = (resolved_conflicts / total_conflicts * 100.0) if total_conflicts > 0 else 100.0

        info = {
            "separation_breaches": unique_breaches,
            "total_conflicts": total_conflicts,
            "conflict_resolution_rate": resolution_rate,
            "unscaled_reward": float(np.sum(raw_rewards))
        }

        return next_raw_obs, scaled_rewards, terminated, truncated, info

    def _get_traffic_instance(self):
        candidates = [
            getattr(bs, "traf", None),
            getattr(bs, "traffic", None),
            getattr(getattr(bs, "sim", None), "traf", None),
            getattr(getattr(bs, "sim", None), "traffic", None),
        ]
        for obj in candidates:
            if obj is not None and hasattr(obj, "id"):
                return obj
        raise AttributeError("Could not locate active BlueSky traffic instance.")

    def _extract_raw_bluesky_state(self) -> Dict[str, np.ndarray]:
        tr = self._get_traffic_instance()

        ids = np.array(tr.id)
        lats = np.array(tr.lat)
        lons = np.array(tr.lon)
        hdgs = np.array(tr.hdg)
        tass = np.array(tr.tas)

        x_nm = (lons - self.ref_lon) * 60.0 * np.cos(np.radians(self.ref_lat))
        y_nm = (lats - self.ref_lat) * 60.0

        wp_x_list = []
        wp_y_list = []
        wp_dist_init_list = []

        for ac_id, x, y in zip(ids, x_nm, y_nm):
            wpx, wpy = self.waypoints.get(ac_id, (0.0, 0.0))
            wp_x_list.append(wpx)
            wp_y_list.append(wpy)

            if ac_id not in self.wp_dist_init_dict:
                initial_dist = float(np.hypot(wpx - x, wpy - y))
                self.wp_dist_init_dict[ac_id] = max(initial_dist, 1e-3)
                self.prev_wp_dist_dict[ac_id] = self.wp_dist_init_dict[ac_id]

            wp_dist_init_list.append(self.wp_dist_init_dict[ac_id])

        return {
            'id': ids,
            'x_nm': x_nm.astype(np.float32),
            'y_nm': y_nm.astype(np.float32),
            'tas_kts': tass.astype(np.float32),
            'hdg_deg': hdgs.astype(np.float32),
            'wp_x_nm': np.array(wp_x_list, dtype=np.float32),
            'wp_y_nm': np.array(wp_y_list, dtype=np.float32),
            'wp_dist_init_nm': np.array(wp_dist_init_list, dtype=np.float32)
        }

    def close(self):
        with suppress_stdout():
            bs.stack.stack("QUIT")


class GeometricRuleBaseline:
    @staticmethod
    def select_actions(raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        n_active = len(raw_obs['id'])
        actions = np.zeros((n_active, 2), dtype=np.float32)

        for i in range(n_active):
            x_i, y_i = raw_obs['x_nm'][i], raw_obs['y_nm'][i]
            hdg_i = raw_obs['hdg_deg'][i]

            min_dist = 999.0
            most_critical_bearing = 0.0

            for j in range(n_active):
                if i == j:
                    continue
                dist = np.hypot(raw_obs['x_nm'][j] - x_i, raw_obs['y_nm'][j] - y_i)
                if dist < min_dist:
                    min_dist = dist
                    dx = raw_obs['x_nm'][j] - x_i
                    dy = raw_obs['y_nm'][j] - y_i
                    bearing = math.degrees(math.atan2(dx, dy))
                    most_critical_bearing = (((bearing - hdg_i + 180.0) % 360.0) - 180.0)

            if min_dist < 8.0:
                actions[i, 0] = 0.8 if most_critical_bearing >= 0 else -0.8
                actions[i, 1] = -0.5
            else:
                dx_wp = raw_obs['wp_x_nm'][i] - x_i
                dy_wp = raw_obs['wp_y_nm'][i] - y_i
                bearing_wp = math.degrees(math.atan2(dx_wp, dy_wp))
                rel_wp = (((bearing_wp - hdg_i + 180.0) % 360.0) - 180.0) / 45.0
                actions[i, 0] = float(np.clip(rel_wp, -1.0, 1.0))
                actions[i, 1] = 0.0

        return actions



class MAPPOTrainer:
    def __init__(
        self,
        env: DirectBlueSkyEnv,
        preprocessor: BlueSkyPreprocessor,
        actor: DecentralisedActor,
        critic: CentralisedCritic,
        lr_actor: float = 3e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.015,
        ppo_epochs: int = 5,
        mini_batch_size: int = 64
    ):
        self.env = env
        self.preprocessor = preprocessor
        self.actor = actor
        self.critic = critic

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

    def collect_rollout(self, steps_per_rollout: int = 64, num_aircraft: int = 4) -> Dict[str, Any]:
        obs_buf, global_buf = [], []
        action_buf, log_prob_buf = [], []
        reward_buf, value_buf = [], []
        done_buf = []

        raw_obs, _ = self.env.reset(num_aircraft=num_aircraft)
        actor_obs, global_state = self.preprocessor.process(raw_obs)
        latest_info = {"separation_breaches": 0, "conflict_resolution_rate": 100.0, "unscaled_reward": 0.0}

        for _ in range(steps_per_rollout):
            obs_tensor = torch.as_tensor(actor_obs, dtype=torch.float32)
            global_tensor = torch.as_tensor(global_state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                if obs_tensor.shape[0] > 0:
                    actions, log_probs, _, _ = self.actor.get_action(obs_tensor, deterministic=False)
                    actions_np = torch.clamp(actions, -1.0, 1.0).cpu().numpy()
                else:
                    actions = torch.zeros((0, 2), dtype=torch.float32)
                    log_probs = torch.zeros((0,), dtype=torch.float32)
                    actions_np = None

                value = self.critic(global_tensor)

            next_raw_obs, rewards, terminated, truncated, info = self.env.step(actions_np)
            next_actor_obs, next_global_state = self.preprocessor.process(next_raw_obs)
            done = terminated or truncated
            latest_info = info

            obs_buf.append(obs_tensor)
            global_buf.append(global_tensor.squeeze(0))
            action_buf.append(actions)
            log_prob_buf.append(log_probs)
            reward_buf.append(torch.as_tensor(rewards, dtype=torch.float32))
            value_buf.append(value.squeeze(-1))
            done_buf.append(done)

            actor_obs, global_state = next_actor_obs, next_global_state

            if done:
                raw_obs, _ = self.env.reset(num_aircraft=num_aircraft)
                actor_obs, global_state = self.preprocessor.process(raw_obs)

        with torch.no_grad():
            final_global_tensor = torch.as_tensor(global_state, dtype=torch.float32).unsqueeze(0)
            next_value = self.critic(final_global_tensor).item()

        return {
            "obs": obs_buf,
            "global_states": global_buf,
            "actions": action_buf,
            "log_probs": log_prob_buf,
            "rewards": reward_buf,
            "values": value_buf,
            "dones": done_buf,
            "next_value": next_value,
            "info": latest_info
        }

    def train_step(self, rollout: Dict[str, Any]) -> Tuple[float, float, float]:
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        steps = len(rewards)

        advantages = []
        gae = 0.0
        for t in reversed(range(steps)):
            v_curr = values[t].item()
            v_next = rollout["next_value"] if t == steps - 1 else values[t + 1].item()
            r_t = rewards[t].mean().item() if rewards[t].numel() > 0 else 0.0
            mask = 0.0 if dones[t] else 1.0

            delta = r_t + (self.gamma * v_next * mask) - v_curr
            gae = delta + (self.gamma * self.gae_lambda * mask * gae)
            advantages.insert(0, gae)

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = advantages_tensor + torch.tensor([v.item() for v in values], dtype=torch.float32)

        valid_indices = [i for i, obs in enumerate(rollout["obs"]) if obs.shape[0] > 0]
        if len(valid_indices) == 0:
            return 0.0, 0.0, 1.0

        b_obs = torch.cat([rollout["obs"][i] for i in valid_indices], dim=0)
        b_actions = torch.cat([rollout["actions"][i] for i in valid_indices], dim=0)
        b_old_log_probs = torch.cat([rollout["log_probs"][i] for i in valid_indices], dim=0).detach()

        adv_list = []
        for i in valid_indices:
            team_adv = advantages_tensor[i]
            r_team_mean = rollout["rewards"][i].mean() if rollout["rewards"][i].numel() > 0 else 0.0
            r_individual = rollout["rewards"][i]
            indiv_adv = team_adv + (r_individual - r_team_mean)
            adv_list.append(indiv_adv)

        b_advantages = torch.cat(adv_list)
        adv_mean = b_advantages.mean()
        adv_std = b_advantages.std(unbiased=False) + 1e-8
        b_advantages = (b_advantages - adv_mean) / adv_std

        b_global_states = torch.stack([rollout["global_states"][i] for i in range(steps)])
        b_returns = returns_tensor

        total_actor_loss, total_ratio = 0.0, 0.0
        actor_updates = 0
        n_samples = b_obs.shape[0]
        mb_size = min(self.mini_batch_size, n_samples)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, mb_size):
                mb_idx = perm[start:start + mb_size]
                new_log_probs, entropy = self.actor.evaluate_actions(b_obs[mb_idx], b_actions[mb_idx])
                ratios = torch.exp(new_log_probs - b_old_log_probs[mb_idx])

                surr1 = ratios * b_advantages[mb_idx]
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean() - (self.entropy_coef * entropy.mean())

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_opt.step()

                total_actor_loss += actor_loss.item()
                total_ratio += ratios.mean().item()
                actor_updates += 1

        total_critic_loss = 0.0
        critic_updates = 0
        n_critic_samples = b_global_states.shape[0]
        c_mb_size = min(self.mini_batch_size, n_critic_samples)

        for _ in range(self.ppo_epochs):
            c_perm = torch.randperm(n_critic_samples)
            for start in range(0, n_critic_samples, c_mb_size):
                c_idx = c_perm[start:start + c_mb_size]
                predicted_values = self.critic(b_global_states[c_idx]).squeeze(-1)

                critic_loss = F.smooth_l1_loss(predicted_values, b_returns[c_idx])

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_opt.step()

                total_critic_loss += critic_loss.item()
                critic_updates += 1

        avg_actor_loss = total_actor_loss / max(actor_updates, 1)
        avg_critic_loss = total_critic_loss / max(critic_updates, 1)
        avg_mean_ratio = total_ratio / max(actor_updates, 1)

        return avg_actor_loss, avg_critic_loss, avg_mean_ratio



def plot_training_results(df: pd.DataFrame):
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))

    axs[0, 0].plot(df["Iteration"], df["Total Reward"], color="navy", lw=1.6)
    axs[0, 0].set_title("Scaled Rollout Reward")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)

    axs[0, 1].plot(df["Iteration"], df["Conflict Resolution Rate (%)"], color="forestgreen", lw=1.6)
    axs[0, 1].axhline(95.0, color="red", linestyle=":", label="Proposal Target (95%)")
    axs[0, 1].set_title("Conflict Resolution Rate (%)")
    axs[0, 1].set_ylim(-5, 105)
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    axs[1, 0].plot(df["Iteration"], df["Separation Breaches"], color="crimson", lw=1.6)
    axs[1, 0].set_title("Unique Separation Breaches (< 5 NM)")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    axs[1, 1].plot(df["Iteration"], df["Critic Loss"], color="purple", lw=1.6)
    axs[1, 1].set_title("Critic Huber Loss")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    axs[2, 0].plot(df["Iteration"], df["Actor Loss"], color="darkorange", lw=1.6)
    axs[2, 0].set_title("Actor Loss")
    axs[2, 0].grid(True, linestyle="--", alpha=0.6)

    axs[2, 1].plot(df["Iteration"], df["Mean Ratio"], color="dimgray", lw=1.6)
    axs[2, 1].axhline(1.0, color="black", linestyle="--", alpha=0.5)
    axs[2, 1].set_title("PPO Mean Ratio (π / π_old)")
    axs[2, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("final.png", dpi=300)
    plt.show()



def main():
    TOTAL_ITERATIONS = 200
    STEPS_PER_ROLLOUT = 128


    env = DirectBlueSkyEnv(dt=5.0, reward_scale=0.05)
    preprocessor = BlueSkyPreprocessor(k_nearest=5, max_aircraft=10)

    actor = DecentralisedActor(obs_dim=29, action_dim=2, init_log_std=-0.7)
    critic = CentralisedCritic(num_aircraft=10)

    trainer = MAPPOTrainer(
        env=env,
        preprocessor=preprocessor,
        actor=actor,
        critic=critic,
        lr_actor=3e-4,
        lr_critic=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.015,
        ppo_epochs=5,
        mini_batch_size=64
    )

    history = {
        "Iteration": [],
        "Density": [],
        "Actor Loss": [],
        "Critic Loss": [],
        "Mean Ratio": [],
        "Total Reward": [],
        "Separation Breaches": [],
        "Conflict Resolution Rate (%)": []
    }

    curriculum_density = 2
    running_crr = []

    print(f"{'Iter':<5} | {'AC':<3} | {'Actor Loss':<10} | {'Critic Loss':<11} | {'Reward':<10} | {'Breaches':<8} | {'CRR (%)':<8}")

    try:
        for iteration in range(1, TOTAL_ITERATIONS + 1):
            if len(running_crr) >= 10 and np.mean(running_crr[-10:]) >= 85.0:
                if curriculum_density < 10:
                    curriculum_density = min(curriculum_density + 2, 10)
                    print(f"\n [CURRICULUM PROMOTION] Advancing Airspace Density to {curriculum_density} Aircraft!\n")
                    running_crr.clear()

            rollout = trainer.collect_rollout(steps_per_rollout=STEPS_PER_ROLLOUT, num_aircraft=curriculum_density)
            actor_loss, critic_loss, mean_ratio = trainer.train_step(rollout)
            total_reward = sum([r.sum().item() for r in rollout["rewards"]])

            breaches = rollout["info"]["separation_breaches"]
            crr = rollout["info"]["conflict_resolution_rate"]
            running_crr.append(crr)

            history["Iteration"].append(iteration)
            history["Density"].append(curriculum_density)
            history["Actor Loss"].append(actor_loss)
            history["Critic Loss"].append(critic_loss)
            history["Mean Ratio"].append(mean_ratio)
            history["Total Reward"].append(total_reward)
            history["Separation Breaches"].append(breaches)
            history["Conflict Resolution Rate (%)"].append(crr)

            if iteration % 10 == 0 or iteration == 1:
                print(f"{iteration:<5} | {curriculum_density:<3} | {actor_loss:<10.4f} | {critic_loss:<11.4f} | {total_reward:<10.2f} | {breaches:<8} | {crr:<8.1f}%")

        torch.save(actor.state_dict(), "final.pt")
        print("\n Model checkpoint saved as 'final.pt'.")

    except KeyboardInterrupt:
        print("\n Training interrupted by user.")
    finally:
        print("EXECUTING BENCHMARK: MARL POLICY vs RULE-BASED TCAS BASELINE (10 AIRCRAFT)")

        baseline_breaches, baseline_crr = [], []
        marl_breaches, marl_crr = [], []

        eval_episodes = 50
        with torch.no_grad():
            for ep in range(eval_episodes):
                raw_obs, _ = env.reset(num_aircraft=10)
                for _ in range(60):
                    b_actions = GeometricRuleBaseline.select_actions(raw_obs)
                    raw_obs, _, term, trunc, info = env.step(b_actions)
                    if term or trunc:
                        break
                baseline_breaches.append(info["separation_breaches"])
                baseline_crr.append(info["conflict_resolution_rate"])

                raw_obs, _ = env.reset(num_aircraft=10)
                obs, _ = preprocessor.process(raw_obs)
                for _ in range(60):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    actions = actor(obs_t).cpu().numpy() if obs_t.shape[0] > 0 else None
                    raw_obs, _, term, trunc, info = env.step(actions)
                    obs, _ = preprocessor.process(raw_obs)
                    if term or trunc:
                        break
                marl_breaches.append(info["separation_breaches"])
                marl_crr.append(info["conflict_resolution_rate"])

        print(f"BASELINE TCAS -> Avg Breaches: {np.mean(baseline_breaches):.2f} | Mean CRR: {np.mean(baseline_crr):.1f}%")
        print(f"OUR MARL MODEL-> Avg Breaches: {np.mean(marl_breaches):.2f} | Mean CRR: {np.mean(marl_crr):.1f}%")

        env.close()
        df = pd.DataFrame(history)
        if len(df) > 0:
            df.to_csv("final_test.csv", index=False)
            plot_training_results(df)


if __name__ == "__main__":
    main()