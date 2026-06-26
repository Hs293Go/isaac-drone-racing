"""Rendering-feasibility smoke for vision-in-the-loop: TiledCamera throughput here.

Measures steps/s and GPU memory with an FPV TiledCamera (RGB) mounted on the racer, at a
given env count + resolution — vs physics-only (--no_camera) — to decide whether vision
racing is practical on this GPU and whether the RTX TiledCamera path is fast enough or a
depth ray-caster is needed. Cameras require the AppLauncher --enable_cameras flag.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/camera_smoke.py --num_envs 256 --headless --enable_cameras
"""

import argparse
from pathlib import Path
import subprocess  # noqa: S404
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TiledCamera rendering feasibility smoke")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--res", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--no_camera", action="store_true", help="physics-only baseline")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from isaaclab.sensors import TiledCameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402

from isaacracelab.racer_env import RacerEnv, RacerEnvCfg  # noqa: E402


def gpu_mem_mb():
    """Total GPU memory in use (MB); includes the RTX renderer, not just torch."""
    q = "--query-gpu=memory.used"
    fmt = "--format=csv,noheader,nounits"
    try:
        r = subprocess.run(  # noqa: S603
            ["/usr/bin/nvidia-smi", q, fmt], capture_output=True, text=True, check=False
        )
        return int(r.stdout.split("\n")[0])
    except (ValueError, OSError):
        return -1


cfg = RacerEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
if not args.no_camera:
    cfg.tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/fpv_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.12, 0.0, 0.03), rot=(0.5, -0.5, 0.5, -0.5), convention="world"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 30.0),
        ),
        width=args.res,
        height=args.res,
    )
env = RacerEnv(cfg)
dev = env.device


def rand_action():
    """Random motor commands in [-1, 1]."""
    return torch.rand((env.num_envs, 4), device=dev) * 2 - 1


env.reset()
for _ in range(30):  # warm up: RTX shader compile, kernel JIT
    env.step(rand_action())
    if not args.no_camera:
        _ = env._camera.data.output["rgb"]
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(args.steps):
    env.step(rand_action())
    if not args.no_camera:
        img = env._camera.data.output["rgb"]
torch.cuda.synchronize()
dt = time.perf_counter() - t0

sps = args.num_envs * args.steps / dt
cam = "off (physics-only)" if args.no_camera else f"rgb {args.res}x{args.res}"
lines = [
    f"# camera={cam}  num_envs={args.num_envs}",
    f"steps/s        = {sps:>12,.0f}   ({args.steps} steps in {dt:.2f}s)",
    f"GPU mem in use = {gpu_mem_mb():>9d} MB",
]
if not args.no_camera:
    lines.append(f"rgb shape      = {tuple(img.shape)}")
out = "\n".join(lines)
Path("/tmp/camera_smoke.txt").write_text(out + "\n", encoding="utf-8")  # noqa: S108
print("\n" + out)
env.close()
sim_app.close()
