from pathlib import Path

import numpy as np
import pygame

import bluesky as bs
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1

DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1

NUM_INTRUDERS = 5
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5 # NM
DEFAULT_AGENT_ID = 'KL001'

WAYPOINT_DISTANCE_MIN = 100
WAYPOINT_DISTANCE_MAX = 150

D_HEADING = 45

AC_SPD = 150

NM2KM = 1.852

ACTION_FREQUENCY = 10

class HorizontalCREnv(gym.Env):
    """ 
    Horizontal Conflict Resolution Environment

    TODO:
    - look at adding waypoints instead of staying straight
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None, scenario_file=None, agent_ids=None, num_intruders=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.scenario_file = scenario_file
        self.num_intruders = NUM_INTRUDERS if num_intruders is None else num_intruders
        self.agent_ids = [] if agent_ids is None else list(agent_ids)
        self.intruder_ids = []
        self.waypoints = {}

        # Observation space should include ownship and intruder info
        # Maybe later also have an option for CNN based intruder info, could be interesting
        self.observation_space = spaces.Dict(
            {
                "intruder_distance": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "x_difference_speed": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "y_difference_speed": spaces.Box(-np.inf, np.inf, shape = (self.num_intruders,), dtype=np.float64),
                "waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "cos_drift": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "sin_drift": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # set correct sim speed
        bs.stack.stack('DT 5;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])

        if self.scenario_file is not None:
            self._load_scenario()
        else:
            bs.traf.reset()
            self._ensure_agents_exist()
            self._generate_conflicts()

        self._resolve_agent_ids()
        self._ensure_agents_exist()
        self._refresh_intruders()
        self._generate_waypoints()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._get_action(action)

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info

    def _load_scenario(self):
        scenario_path = str(Path(self.scenario_file).expanduser())
        if not Path(scenario_path).is_absolute():
            scenario_path = str((Path.cwd() / scenario_path).resolve())

        bs.traf.reset()
        bs.stack.stack(f'IC "{scenario_path}"')

    def _resolve_agent_ids(self):
        if self.agent_ids:
            return

        active_ids = [acid for acid in bs.traf.id if acid != '']
        if DEFAULT_AGENT_ID in bs.traf.id:
            self.agent_ids = [DEFAULT_AGENT_ID]
        elif active_ids:
            self.agent_ids = [active_ids[0]]
        else:
            self.agent_ids = [DEFAULT_AGENT_ID]

        self.action_space = spaces.Box(-1, 1, shape=(len(self.agent_ids),), dtype=np.float64)

    def _ensure_agents_exist(self):
        for agent_id in self.agent_ids:
            if agent_id not in bs.traf.id:
                bs.traf.cre(agent_id, actype="A320", acspd=AC_SPD)

        if not self.agent_ids:
            raise RuntimeError("HorizontalCREnv requires at least one controlled aircraft.")

    def _refresh_intruders(self):
        self.intruder_ids = [acid for acid in bs.traf.id if acid not in set(self.agent_ids)]
        if not self.intruder_ids:
            self._generate_conflicts()
            self.intruder_ids = [acid for acid in bs.traf.id if acid not in set(self.agent_ids)]

    def _generate_conflicts(self):
        if not self.agent_ids:
            self._resolve_agent_ids()
        for agent_id in self.agent_ids:
            if agent_id not in bs.traf.id:
                continue
            target_idx = bs.traf.id2idx(agent_id)
            for i in range(self.num_intruders):
                dpsi = np.random.randint(45, 315)
                cpa = np.random.randint(0, INTRUSION_DISTANCE)
                tlosh = np.random.randint(100, 1000)
                bs.traf.creconfs(acid=f'{agent_id}_{i}', actype="A320", targetidx=target_idx, dpsi=dpsi, dcpa=cpa, tlosh=tlosh)

    def _generate_waypoints(self):
        self.waypoints = {}
        for agent_id in self.agent_ids:
            self.waypoints[agent_id] = {"lat": [], "lon": [], "reach": []}
            for i in range(NUM_WAYPOINTS):
                wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
                wpt_hdg_init = 0
                ac_idx = bs.traf.id2idx(agent_id)
                wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)
                self.waypoints[agent_id]["lat"].append(wpt_lat)
                self.waypoints[agent_id]["lon"].append(wpt_lon)
                self.waypoints[agent_id]["reach"].append(0)

    def _get_agent_obs(self, agent_id):
        if agent_id not in bs.traf.id:
            bs.traf.cre(agent_id, actype="A320", acspd=AC_SPD)

        ac_idx = bs.traf.id2idx(agent_id)
        intruder_distance = []
        cos_bearing = []
        sin_bearing = []
        x_difference_speed = []
        y_difference_speed = []
        waypoint_distance = []
        waypoint_qdrs = []
        cos_drift = []
        sin_drift = []
        drift = []

        ac_hdg = bs.traf.hdg[ac_idx]
        for intruder_id in self.intruder_ids[:self.num_intruders]:
            int_idx = bs.traf.id2idx(intruder_id)
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            intruder_distance.append(int_dis * NM2KM)
            bearing = ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)
            cos_bearing.append(np.cos(np.deg2rad(bearing)))
            sin_bearing.append(np.sin(np.deg2rad(bearing)))
            heading_difference = ac_hdg - bs.traf.hdg[int_idx]
            x_dif = -np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            x_difference_speed.append(x_dif)
            y_difference_speed.append(y_dif)

        while len(intruder_distance) < self.num_intruders:
            intruder_distance.append(0.0)
            cos_bearing.append(0.0)
            sin_bearing.append(0.0)
            x_difference_speed.append(0.0)
            y_difference_speed.append(0.0)

        for lat, lon in zip(self.waypoints[agent_id]["lat"], self.waypoints[agent_id]["lon"]):
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
            waypoint_distance.append(wpt_dis * NM2KM)
            waypoint_qdrs.append(wpt_qdr)
            drift_value = ac_hdg - wpt_qdr
            drift_value = fn.bound_angle_positive_negative_180(drift_value)
            drift.append(drift_value)
            cos_drift.append(np.cos(np.deg2rad(drift_value)))
            sin_drift.append(np.sin(np.deg2rad(drift_value)))

        return {
            "intruder_distance": np.array(intruder_distance) / WAYPOINT_DISTANCE_MAX,
            "cos_difference_pos": np.array(cos_bearing),
            "sin_difference_pos": np.array(sin_bearing),
            "x_difference_speed": np.array(x_difference_speed) / AC_SPD,
            "y_difference_speed": np.array(y_difference_speed) / AC_SPD,
            "waypoint_distance": np.array(waypoint_distance) / WAYPOINT_DISTANCE_MAX,
            "cos_drift": np.array(cos_drift),
            "sin_drift": np.array(sin_drift),
        }

    def _get_obs(self):
        if not self.agent_ids:
            self._resolve_agent_ids()
            self._ensure_agents_exist()

        if len(self.agent_ids) == 1:
            return self._get_agent_obs(self.agent_ids[0])

        return {agent_id: self._get_agent_obs(agent_id) for agent_id in self.agent_ids}
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean()
        }

    def _get_reward(self):
        total_reward = 0.0
        for agent_id in self.agent_ids:
            agent_reward = 0.0
            reach_reward = self._check_waypoint(agent_id)
            drift_reward = self._check_drift(agent_id)
            intrusion_reward = self._check_intrusion(agent_id)
            agent_reward = reach_reward + drift_reward + intrusion_reward
            total_reward += agent_reward

        self.total_reward += total_reward

        if any(self.waypoints[agent_id]["reach"].count(0) > 0 for agent_id in self.agent_ids):
            return total_reward, 0
        return total_reward, 1
        
    def _check_waypoint(self, agent_id):
        reward = 0
        agent_waypoints = self.waypoints[agent_id]
        for index, distance in enumerate(self._get_agent_obs(agent_id)["waypoint_distance"]):
            if distance < DISTANCE_MARGIN / WAYPOINT_DISTANCE_MAX and agent_waypoints["reach"][index] != 1:
                agent_waypoints["reach"][index] = 1
                reward += REACH_REWARD
        return reward

    def _check_drift(self, agent_id):
        obs = self._get_agent_obs(agent_id)
        drift = abs(np.deg2rad(obs["sin_drift"][0])) if len(obs["sin_drift"]) else 0.0
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY

    def _check_intrusion(self, agent_id):
        ac_idx = bs.traf.id2idx(agent_id)
        reward = 0
        for intruder_id in self.intruder_ids[:self.num_intruders]:
            int_idx = bs.traf.id2idx(intruder_id)
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        return reward
    
    def _get_action(self, action):
        if not self.agent_ids:
            self._resolve_agent_ids()
        if np.isscalar(action):
            action = np.array([float(action)], dtype=np.float64)
        if len(action) != len(self.agent_ids):
            action = np.resize(action, len(self.agent_ids))
        for agent_id, action_value in zip(self.agent_ids, action):
            if agent_id not in bs.traf.id:
                continue
            ac_idx = bs.traf.id2idx(agent_id)
            current_heading = float(bs.traf.hdg[ac_idx])
            heading = float(np.mod(current_heading + float(action_value) * D_HEADING, 360.0))
            bs.stack.stack(f"HDG {agent_id} {heading}")

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        render_agent_id = self.agent_ids[0] if self.agent_ids else DEFAULT_AGENT_ID
        if render_agent_id not in bs.traf.id:
            return

        # draw ownship
        ac_idx = bs.traf.id2idx(render_agent_id)
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2-heading_end_x/2,self.window_height/2+heading_end_y/2),
            ((self.window_width/2)+heading_end_x/2,(self.window_height/2)-heading_end_y/2),
            width = 4
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 1
        )

        # draw intruders
        ac_length = 3

        for intruder_id in self.intruder_ids[:NUM_INTRUDERS]:
            int_idx = bs.traf.id2idx(intruder_id)
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine color
            if int_dis < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (self.window_width/2)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (self.window_height/2)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
                width = 2
            )

            # import code
            # code.interact(local=locals())

        # draw target waypoint
        for qdr, dis, reach in zip(
            self.waypoints.get(render_agent_id, {}).get("qdr", []),
            self.waypoints.get(render_agent_id, {}).get("distance", []),
            self.waypoints.get(render_agent_id, {}).get("reach", [])
        ):
            circle_x = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width
            circle_y = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width

            if reach:
                color = (155,155,155)
            else:
                color = (255,255,255)

            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        bs.stack.stack('quit')