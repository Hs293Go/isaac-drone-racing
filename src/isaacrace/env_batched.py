"""BatchedRaceEnv — a VecEnv: batched Quadrotors racing a gate course without Isaac Sim.

The env manages N batched drones, using vectorized methods in `isaacrace.dynamics` and
`isaacrace.course` to propagate dynamics, compute observations, rewards, and
gate-passing simultaneously. The PPO training loop can step all N envs in parallel with
negligible overhead, so we can train with hundreds of envs on a single CPU in minutes.

Since this env is Isaac-free, it predominantly complies with optimal_quad_control_rl's
native NED-FRD convention, switching to ENU to compute observations and gate-traversal
rewards only.

We ensure parity with RaceEnv by construction: both envs consume the same
``dyn.model_derivatives`` terms (verified against TU Delft's symbolic model in
tests/test_dynamics.py) and the same RaceCourse rules, at the same explicit-Euler step
(``dyn.DT``). A policy trained here therefore transfers to RaceEnv for demo/validation
on Isaac — the packaged demo model was trained exactly this way.
"""

from gymnasium import spaces
import numpy as np
from numpy.typing import ArrayLike
from stable_baselines3.common.vec_env import VecEnv

from isaacrace import perception
from isaacrace.config import FpvConfig
from isaacrace.conversions import vec_enu_ned
from isaacrace.course import RaceCourse
import isaacrace.dynamics as dyn


class BatchedRaceEnv(VecEnv):
    """Environment for a batch of drone racing for rapid training without Isaac Sim.

    This environment manages a (N, 16) state array for a batch of N parallel drones. Its
    public method `step_async` accepts a batch of (N, 4) motor commands, and `step_wait`
    integrates the dynamics forward one step, computes rewards, checks for gate passage
    and crashes, and returns a (N, 20) observation array, (N,) reward array, (N,) done
    flags, and list of N info dicts.
    """

    def __init__(
        self,
        course: RaceCourse,
        num_envs: int,
        seed: int = 0,
        max_steps: int = 1200,
        fpv: FpvConfig | None = None,
        randomize_params: bool = False,
        param_dr_pct: float = 0.1,
        perception_weight: float = 0.0,
        backpedal_weight: float = 0.0,
    ):
        """Initialize the batched racing environment.

        Args:
            course: the RaceCourse defining the gate layout and obs construction.
            num_envs: how many parallel envs to batch together.
            seed: random seed for reset randomization (obs and params).
            max_steps: max steps per episode (for truncation).
            fpv: if not None, config for the drone's FPV camera.
            randomize_params: if True, apply domain randomization to the drone's model
                parameters each episode (in reset); see param_dr_pct.
            param_dr_pct: if randomize_params, the ±% range to sample each parameter
                around its nominal value (e.g. 0.1 = ±10%).
            perception_weight: weight on the FOV-gated gate-visibility shaping reward
                (matches RaceEnv); 0 = the bare progress reward.
            backpedal_weight: weight on the tail-first penalty (reward nose-first);
                0 suppresses it.
        """
        obs_space = spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        super().__init__(num_envs, obs_space, act_space)
        self.render_mode = None
        self.course = course
        self.dt = dyn.DT
        self.max_steps = max_steps
        self.randomize_params = randomize_params
        self.param_dr_pct = param_dr_pct
        self.perception_weight = perception_weight
        self.backpedal_weight = backpedal_weight
        self._fpv = fpv if fpv is not None else FpvConfig()
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros((num_envs, 16))  # NED-FRD [pos, vel, euler, rates, motor]
        self.target_gate = np.zeros(num_envs, dtype=int)
        self.step_count = np.zeros(num_envs, dtype=int)
        self.params_arr = np.tile(
            np.asarray(dyn.PARAMS_5INCH, dtype=float), (num_envs, 1)
        )
        self._actions = None

    # ----------------------------------------------------------- internals
    def _params(self):
        """A QuadParams whose 23 fields are (N,) arrays (per-env params)."""
        return dyn.QuadParams._make(self.params_arr.T)

    def _obs(self):
        return self.course.build_obs(self.state, self.target_gate)

    def _reset_rows(self, idx):
        """Reset the given env rows in place (random gate, pose, motor; optional DR)."""
        m = idx.size
        if m == 0:
            return
        if self.randomize_params:
            for j in idx:
                self.params_arr[j] = np.asarray(
                    dyn.PARAMS_5INCH.randomized(self.rng, self.param_dr_pct)
                )
        g, state = self.course.sample_spawn(self.rng, m, randomize=True)
        self.target_gate[idx] = g
        self.state[idx] = state
        self.step_count[idx] = 0

    # ----------------------------------------------------------- VecEnv API
    def reset(self):
        """Reset all envs; return stacked obs (N, 20)."""
        self._reset_rows(np.arange(self.num_envs))
        return self._obs()

    def step_async(self, actions: ArrayLike):
        """Store the (N, 4) actions for the next step_wait.

        Args:
            actions: array of shape (N, 4) with values in [-1, 1] for the motor
              commands.
        """
        self._actions = np.asarray(actions, dtype=float)

    def step_wait(self):
        """Integrate one step for all envs.

        Returns:
            obs: (N, 20) array of observations after the step.
            reward: (N,) array of rewards for the step.
            dones: (N,) boolean array of done flags for the step.
            infos: list of N dicts with extra info for the step.
        """
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        n, num = self.num_envs, self.course.num_gates
        pos_old = vec_enu_ned(self.state[:, 0:3])
        self.state += self.dt * dyn.model_derivatives(
            self.state, self._actions, self._params()
        )
        self.state[:, 12:16] = np.clip(self.state[:, 12:16], -1.0, 1.0)
        self.step_count += 1

        pos_new = vec_enu_ned(self.state[:, 0:3])
        reward = self.course.progress(pos_old, pos_new, self.target_gate)
        reward -= self.course.RATE_PENALTY * np.linalg.norm(
            self.state[:, 9:12], axis=-1
        )

        # perception visibility (always measured for logging; rewarded only if on)
        gate_ned = self.course.gate_pos[self.target_gate % num]
        cos_a = perception.gate_bearing(
            self.state[:, 0:3], self.state[:, 6:9], gate_ned, self._fpv
        )
        vis = perception.visibility_reward(cos_a, self._fpv.fov_deg)
        reward += self.perception_weight * vis  # gate-visibility shaping (0 = off)
        # tail-first penalty, enshrined as part of PA (weight 0 suppresses it)
        reward -= self.backpedal_weight * perception.backpedal(
            self.state[:, 3:6], self.state[:, 6:9]
        )

        passed, collided = self.course.gate_passed(pos_old, pos_new, self.target_gate)
        self.target_gate = self.course.advance_gate(self.target_gate, passed)
        crashed = self.course.out_of_bounds(pos_new) | collided
        reward = np.where(crashed, self.course.CRASH_REWARD, reward)
        terminated = crashed
        truncated = self.step_count >= self.max_steps
        dones = terminated | truncated

        obs = self._obs()
        infos = [
            {
                "gate_passed": bool(passed[i]),
                "target_gate": int(self.target_gate[i]),
                "collision": bool(crashed[i]),
                "perception_visibility": float(vis[i]),
            }
            for i in range(n)
        ]
        done_idx = np.nonzero(dones)[0]
        for i in done_idx:
            infos[i]["terminal_observation"] = obs[i].copy()
            if truncated[i] and not terminated[i]:
                infos[i]["TimeLimit.truncated"] = True
        if done_idx.size:
            self._reset_rows(done_idx)
            obs[done_idx] = self.course.build_obs(
                self.state[done_idx], self.target_gate[done_idx]
            )
        return obs.astype(np.float32), reward.astype(np.float32), dones, infos

    def close(self):
        """No resources to release (pure NumPy)."""

    # --- remaining VecEnv abstract methods (minimal; PPO + VecMonitor need no more)
    def _sel(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices

    def get_attr(self, attr_name, indices=None):
        """Return a VecEnv attribute, replicated per selected env."""
        return [getattr(self, attr_name, None) for _ in self._sel(indices)]

    def set_attr(self, attr_name, value, indices=None):
        """Set an attribute on the batched env (shared across all envs)."""
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Unsupported: the batched env has no per-env method dispatch."""
        raise NotImplementedError("BatchedRaceEnv does not support env_method")

    def env_is_wrapped(self, wrapper_class, indices=None):
        """No per-env gym wrappers; always False."""
        return [False for _ in self._sel(indices)]
