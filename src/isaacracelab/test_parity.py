"""CPU parity test: torch ports vs the NumPy originals (no GPU / no Isaac needed).

This is the verifiable core of the Isaac Lab spike — it proves the torch dynamics /
conversions / course math match isaacrace's NumPy implementation to float precision, so
a policy sees identical observations and the classical plant applies identical forces.

    env -u PYTHONPATH uv run --group isaaclab python -m isaacracelab.test_parity
"""

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from isaacrace.config import RaceTrackConfig
import isaacrace.conversions as conv
from isaacrace.course import RaceCourse
import isaacrace.dynamics as dyn
from isaacracelab import dynamics_torch as dt
from isaacracelab.course_torch import RaceCourseTorch

torch.manual_seed(0)
rng = np.random.default_rng(0)
fails = []


def chk(name, a_np, b_torch, atol):
    """Assert a NumPy array and torch tensor agree within atol; record the max diff."""
    b = b_torch.detach().cpu().numpy()
    a = np.asarray(a_np)
    d = (
        float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        if a.size
        else 0.0
    )  # .astype(float) so bool arrays compare too
    ok = a.shape == b.shape and d <= atol
    print(f"  [{'OK' if ok else 'FAIL'}] {name:32s} max|Δ|={d:.2e}  shape={a.shape}")
    if not ok:
        fails.append(name)


def both(np_arr, dtype=torch.float64):
    """A random array as (numpy float64, torch tensor)."""
    return np_arr, torch.as_tensor(np_arr, dtype=dtype)


print("== conversions ==")
v_np, v_t = both(rng.standard_normal((9, 3)))
chk("vec_enu_ned", conv.vec_enu_ned(v_np), conv.vec_enu_ned(v_t), 1e-12)
chk("vec_flu_frd", conv.vec_flu_frd(v_np), conv.vec_flu_frd(v_t), 1e-12)
q_np = rng.standard_normal((9, 4))
q_np /= np.linalg.norm(q_np, axis=-1, keepdims=True)
q_t = torch.as_tensor(q_np, dtype=torch.float64)
chk("quat_aero_isaac", conv.quat_aero_isaac(q_np), conv.quat_aero_isaac(q_t), 1e-12)
chk(
    "quat_xyzw_to_euler_xyz",
    Rotation.from_quat(q_np).as_euler("xyz"),
    conv.quat_xyzw_to_euler_xyz(q_t),
    1e-9,
)
e_np = rng.uniform(-1.2, 1.2, (9, 3))  # bounded pitch (avoid gimbal sign flips)
chk(
    "euler_xyz_to_quat_xyzw",
    Rotation.from_euler("xyz", e_np).as_quat(),
    conv.euler_xyz_to_quat_xyzw(torch.as_tensor(e_np)),
    1e-9,
)
chk(
    "quat_rotate_xyzw",
    Rotation.from_quat(q_np).apply(v_np),
    conv.quat_rotate_xyzw(q_t, v_t),
    1e-9,
)

print("== dynamics ==")
p_np = np.asarray(dyn.PARAMS_5INCH)
p_t = torch.as_tensor(p_np, dtype=torch.float64)
w_np, w_t = both(rng.uniform(-1, 1, (11, 4)))
u_np, u_t = both(rng.uniform(-1, 1, (11, 4)))
vb_np, vb_t = both(rng.standard_normal((11, 3)))
chk("unproject_motor", dyn.unproject_motor(w_np), dt.unproject_motor(w_t), 1e-9)
dwn_np, dwr_np = dyn.motor_derivative(w_np, u_np, dyn.PARAMS_5INCH)
dwn_t, dwr_t = dt.motor_derivative(w_t, u_t, p_t)
chk("motor_derivative d_w", dwn_np, dwn_t, 1e-9)
chk("motor_derivative d_W", dwr_np, dwr_t, 1e-9)
chk(
    "rotor_thrusts",
    dyn.rotor_thrusts(w_np, dyn.PARAMS_5INCH),
    dt.rotor_thrusts(w_t, p_t),
    1e-12,
)
f_np, tq_np = dyn.body_force_torque_accel(w_np, u_np, vb_np, dyn.PARAMS_5INCH)
f_t, tq_t = dt.body_force_torque_accel(w_t, u_t, vb_t, p_t)
chk("bfta force", f_np, f_t, 1e-9)
chk("bfta torque", tq_np, tq_t, 1e-9)

print("== course rules ==")
cfg = RaceTrackConfig(
    gate_pos=[[2, 0, -1.5], [4, 2, -1.5], [2, 4, -1.5], [0, 2, -1.5], [2, 2, -2.0]],
    gate_yaw=[0, 1, 2, 3, 0],
    start_pos=[0, 0, -1.5],
)
course = RaceCourse(cfg)
tc = RaceCourseTorch(cfg, device="cpu")
m = 7
ws = rng.standard_normal((m, 16)).astype(np.float32)
tg = rng.integers(0, course.num_gates, m)
po = rng.standard_normal((m, 3)).astype(np.float32)
pn = rng.standard_normal((m, 3)).astype(np.float32)
ws_t = torch.as_tensor(ws, dtype=torch.float32)
tg_t = torch.as_tensor(tg, dtype=torch.long)
po_t, pn_t = torch.as_tensor(po), torch.as_tensor(pn)

chk("build_obs", course.build_obs(ws, tg), tc.build_obs(ws_t, tg_t), 1e-4)
p_pass, p_coll = course.gate_passed(po, pn, tg)
t_pass, t_coll = tc.gate_passed(po_t, pn_t, tg_t)
chk("gate_passed passed", p_pass, t_pass, 0)
chk("gate_passed collided", p_coll, t_coll, 0)
chk("progress", course.progress(po, pn, tg), tc.progress(po_t, pn_t, tg_t), 1e-4)
chk("out_of_bounds", course.out_of_bounds(pn), tc.out_of_bounds(pn_t), 0)
passed = rng.integers(0, 2, m).astype(bool)
chk(
    "advance_gate",
    course.advance_gate(tg, passed),
    tc.advance_gate(tg_t, torch.as_tensor(passed)),
    0,
)

print()
if fails:
    print(f"PARITY FAILED: {fails}")
    raise SystemExit(1)
print("PARITY OK — all torch ports match NumPy")
