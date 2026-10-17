"""
BlueSkyMultiAgentEnv
====================
A PettingZoo ParallelEnv built directly on the BlueSky simulation core
(bs.traf / bs.stack / bs.sim), rather than on top of BlueSky-Gym's
single-ownship Gymnasium wrappers.

Every aircraft in the sector is an independent learning agent: each gets
its own observation (its own state + nearest-neighbour traffic) and its
own reward, and each is commanded independently every step. Agents that
reach their destination are removed from the episode (terminated), so the
environment supports a shrinking/variable agent population, which is the
norm in real ATC traffic rather than the exception.

Tested against pettingzoo.test.parallel_api_test and against Ray RLlib's
ParallelPettingZooEnv + multi-agent PPO.
"""
import functools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import bluesky as bs
from bluesky.simulation import ScreenIO

from pettingzoo import ParallelEnv
from gymnasium import spaces

# --- aircraft / world parameters ---------------------------------------
AC_TYPE = "A320"
AC_SPD = 150            # kt CAS, initial airspeed
ALT_FT = 20000
CENTER_LAT, CENTER_LON = 52.0, 4.0
SPAWN_RADIUS_NM = 40.0
DEST_RADIUS_NM = 40.0
DEST_REACHED_NM = 5.0   # destination counts as "reached" within this range

# --- RL / control parameters ---------------------------------------------
ACTION_FREQUENCY = 5    # bluesky sim ticks per RL step (with DT=1s -> 5s/step)
D_HEADING = 22.5        # deg, max heading change per action
D_VELOCITY = 20 / 3     # kt, max speed change per action
INTRUSION_NM = 5.0
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1.0
REACH_BONUS = 1.0
FAR_NM = 100.0          # filler distance for padding neighbor slots when few AC remain


class _ScreenDummy(ScreenIO):
    """Headless stand-in for BlueSky's screen I/O; swallows console echoes."""
    def echo(self, text="", flags=0):
        pass


class BlueSkyMultiAgentEnv(ParallelEnv):
    metadata = {"name": "bluesky_multiagent_v0", "render_modes": []}

    def __init__(self, n_agents=4, n_neighbors=3, max_steps=300, seed=None):
        self.n_neighbors = n_neighbors
        self.max_steps = max_steps
        self.possible_agents = [f"AC{i}" for i in range(n_agents)]
        self.agents = []
        self._dest = {}  # acid -> (lat, lon)

        # Detached = no networking between sim node and client; this is the
        # headless mode used for training (no BlueSky GUI process involved).
        bs.init(mode="sim", detached=True)
        bs.scr = _ScreenDummy()
        bs.stack.stack("DT 1;FF")  # 1s sim timestep, run as fast as possible

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Flat Box (not Dict) so this drops straight into RLlib's/SB3's default
        # MLP encoders with no extra catalog/model configuration.
        # [own_hdg_sin, own_hdg_cos, own_spd, own_drift_sin, own_drift_cos,
        #  (bearing_sin, bearing_cos, dist) x n_neighbors]
        k = self.n_neighbors
        low = np.array([-1, -1, -5, -1, -1] + [-1, -1, 0] * k, dtype=np.float32)
        high = np.array([1, 1, 5, 1, 1] + [1, 1, FAR_NM] * k, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(-1, 1, (2,), dtype=np.float32)  # [d_heading, d_speed]

    def reset(self, seed=None, options=None):
        bs.traf.reset()
        self.agents = self.possible_agents[:]
        self._dest = {}
        self.num_moves = 0
        rng = np.random.default_rng(seed)

        for acid in self.agents:
            spawn_brg, spawn_dist = rng.uniform(0, 360), rng.uniform(5, SPAWN_RADIUS_NM)
            dest_brg, dest_dist = rng.uniform(0, 360), rng.uniform(5, DEST_RADIUS_NM)
            slat, slon = bs.tools.geo.qdrpos(CENTER_LAT, CENTER_LON, spawn_brg, spawn_dist)
            dlat, dlon = bs.tools.geo.qdrpos(CENTER_LAT, CENTER_LON, dest_brg, dest_dist)
            init_hdg, _ = bs.tools.geo.kwikqdrdist(slat, slon, dlat, dlon)

            bs.traf.cre(acid, actype=AC_TYPE, aclat=slat, aclon=slon,
                        achdg=init_hdg, acspd=AC_SPD, acalt=ALT_FT)
            self._dest[acid] = (dlat, dlon)

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        for acid, action in actions.items():
            if acid not in bs.traf.id:
                continue
            idx = bs.traf.id2idx(acid)
            new_hdg = (bs.traf.hdg[idx] + float(action[0]) * D_HEADING) % 360
            new_spd = max(bs.traf.cas[idx] + float(action[1]) * D_VELOCITY, 1.0)
            bs.stack.stack(f"HDG {acid} {new_hdg}")
            bs.stack.stack(f"SPD {acid} {new_spd}")

        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()

        self.num_moves += 1
        truncate_all = self.num_moves >= self.max_steps

        obs, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for acid in list(self.agents):
            if acid not in bs.traf.id:
                continue

            reached = self._destination_reached(acid)
            rewards[acid] = self._get_reward(acid, reached)
            terminations[acid] = reached
            truncations[acid] = truncate_all
            infos[acid] = {}

            if reached:
                # Aircraft exits the sector: remove from the sim and from
                # future turns, same as a real flight leaving the airspace.
                obs[acid] = self._get_obs(acid)  # last observation before leaving
                bs.traf.delete(bs.traf.id2idx(acid))
                self.agents.remove(acid)
            else:
                obs[acid] = self._get_obs(acid)

        if truncate_all:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # --- helpers ---------------------------------------------------------

    def _destination_reached(self, acid):
        idx = bs.traf.id2idx(acid)
        dlat, dlon = self._dest[acid]
        _, dist = bs.tools.geo.kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], dlat, dlon)
        return dist < DEST_REACHED_NM

    def _neighbors(self, acid, k):
        idx = bs.traf.id2idx(acid)
        others = [a for a in self.agents if a != acid and a in bs.traf.id]
        dists, brgs = [], []
        for other in others:
            oidx = bs.traf.id2idx(other)
            qdr, dist = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.lat[oidx], bs.traf.lon[oidx]
            )
            dists.append(dist)
            brgs.append(qdr)
        order = np.argsort(dists) if dists else []
        dists = np.array(dists)[order][:k] if len(dists) else np.zeros(0)
        brgs = np.array(brgs)[order][:k] if len(brgs) else np.zeros(0)
        pad = k - len(dists)
        if pad > 0:
            dists = np.concatenate([dists, np.full(pad, FAR_NM)])
            brgs = np.concatenate([brgs, np.zeros(pad)])
        return dists, brgs

    def _get_obs(self, acid):
        idx = bs.traf.id2idx(acid)
        dlat, dlon = self._dest[acid]
        dest_brg, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], dlat, dlon)
        drift = np.deg2rad(bs.traf.hdg[idx] - dest_brg)

        dists, brgs = self._neighbors(acid, self.n_neighbors)
        own = np.array([
            np.sin(np.deg2rad(bs.traf.hdg[idx])),
            np.cos(np.deg2rad(bs.traf.hdg[idx])),
            (bs.traf.cas[idx] - AC_SPD) / 20.0,
            np.sin(drift),
            np.cos(drift),
        ])
        neighbor_feats = np.stack([
            np.sin(np.deg2rad(brgs)),
            np.cos(np.deg2rad(brgs)),
            np.clip(dists, 0, FAR_NM),
        ], axis=1).flatten()  # [sin, cos, dist] per neighbor, nearest first
        return np.concatenate([own, neighbor_feats]).astype(np.float32)

    def _get_reward(self, acid, reached):
        idx = bs.traf.id2idx(acid)
        dlat, dlon = self._dest[acid]
        dest_brg, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], dlat, dlon)
        drift = abs(np.deg2rad(bs.traf.hdg[idx] - dest_brg))
        reward = drift * DRIFT_PENALTY

        for other in self.agents:
            if other == acid or other not in bs.traf.id:
                continue
            oidx = bs.traf.id2idx(other)
            _, dist = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.lat[oidx], bs.traf.lon[oidx]
            )
            if dist < INTRUSION_NM:
                reward += INTRUSION_PENALTY

        if reached:
            reward += REACH_BONUS
        return float(reward)

    def render(self):
        pass  # headless by design; use BlueSky-Gym's envs for a visual demo

    def close(self):
        pass


if __name__ == "__main__":
    # Quick smoke test: random policy, prints per-step rewards.
    env = BlueSkyMultiAgentEnv(n_agents=4, n_neighbors=3, max_steps=20)
    obs, infos = env.reset(seed=0)
    print("Agents at reset:", env.agents)
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        print(f"step {env.num_moves:3d} | agents left: {len(env.agents):d} | rewards: "
              f"{ {k: round(v, 3) for k, v in rewards.items()} }")
