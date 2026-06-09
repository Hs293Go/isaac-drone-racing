"""Demo: fly a trained policy through the racecourse and report gates cleared.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
        uv run python examples/demo.py session.headless=false      # windowed
    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
        uv run python examples/demo.py                             # headless
    ... uv run python examples/demo.py \
        session.demo.model=/path/to/model.zip track=figure8

Config lives in conf/ (track + session groups, see conf/config.yaml); override on
the CLI as above. Model defaults to the packaged models/race_ppo.zip; the
numpy._core shim lets you point session.demo.model at an OQCRL model (pickled
under numpy 2.x) in this numpy-1.26 venv.
"""

import contextlib
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Entry point: roll out a trained policy and report gates cleared."""
    import sys

    # numpy 2.x (foreign model pickle) -> numpy 1.26 (Isaac venv) shim, before
    # SB3/torch.
    import numpy as np

    sys.modules["numpy._core"] = np.core
    for _sub in ["multiarray", "umath", "numeric", "_multiarray_umath", "overrides"]:
        with contextlib.suppress(Exception):
            sys.modules["numpy._core." + _sub] = __import__(
                "numpy.core." + _sub, fromlist=[_sub]
            )

    from isaacrace.config import RaceTrackConfig, SessionConfig

    track_cfg = RaceTrackConfig.from_dict(
        OmegaConf.to_container(cfg.track, resolve=True)
    )
    session = SessionConfig.from_dict(OmegaConf.to_container(cfg.session, resolve=True))

    # The app must boot (with the configured headless flag) before importing the
    # env's isaacsim deps.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": session.headless})

    import carb
    import numpy as np
    from stable_baselines3 import PPO

    from isaacrace.course import RaceCourse
    from isaacrace.env import RaceEnv

    course = RaceCourse(track_cfg)
    env = RaceEnv(
        course,
        headless=session.headless,
        randomize_reset=False,
        seed=session.seed,
        max_steps=session.max_steps,
        fpv=session.fpv,
    )

    # Dock a second viewport showing the FPV camera, next to the third-person
    # view (GUI only).
    if session.fpv.enabled and not session.headless and env._fpv_cam_path:
        from omni.kit.viewport.utility import create_viewport_window

        w, h = session.fpv.resolution
        create_viewport_window(
            name="FPV", width=int(w), height=int(h), camera_path=env._fpv_cam_path
        )
        carb.log_warn(f"[demo] FPV viewport on {env._fpv_cam_path}")

    model_path = session.demo.model or str(
        Path(__file__).parent / "models" / "race_ppo.zip"
    )
    model = PPO.load(model_path, device="cpu")
    carb.log_warn(f"[demo] model={model_path}")

    obs, _ = env.reset()

    # Optional capture (windowed only): record the course view + FPV side-by-side to an
    # mp4. Replicator render products need the full RTX pipeline, which only comes up in
    # windowed mode; headless hard-crashes this pip-Isaac build, so we gate it out.
    cap = (
        session.demo.capture
        and not session.headless
        and session.fpv.enabled
        and env.quad.camera is not None
    )
    if session.demo.capture and not cap:
        why = (
            "session is headless (capture needs windowed RTX; session.headless=false)"
            if session.headless
            else "FPV camera disabled (set session.fpv.enabled=true)"
        )
        carb.log_warn(f"[demo] capture requested but skipped: {why}")
    writer = None
    if cap:
        import cv2
        from isaacsim.core.utils.extensions import enable_extension

        from isaacrace.camera import CourseCamera

        carb.log_warn("[demo] recording course view + FPV side-by-side -> mp4")

        # Replicator is a Kit extension that registers OmniGraph nodes lazily; enable it
        # and tick a couple updates so those nodes exist BEFORE we create a render
        # product (creating one against an unregistered graph hard-crashes the app).
        enable_extension("omni.replicator.core")
        for _ in range(2):
            app.update()

        # Static "main" camera framing the whole course, from an elevated 3/4 corner.
        centroid = course.gate_pos_enu.mean(axis=0)
        course_cam = CourseCamera(
            env.world.stage,
            "/World/course_cam",
            eye=centroid + np.array([2.5, -4.5, 2.5]),
            look_at=centroid,
            fov_deg=60.0,
        )
        course_cam.start_capture(session.fpv.resolution)
        env.quad.camera.start_capture(session.fpv.resolution)
        for _ in range(2):
            env.world.render()  # warm up: the render products' first frames are empty

        w, h = (int(v) for v in session.fpv.resolution)
        writer = cv2.VideoWriter(
            session.demo.capture_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(session.demo.capture_fps),
            (2 * w, h),  # side-by-side: course view | FPV
        )

    gates = 0
    for t in range(session.demo.steps):
        if not app.is_running():
            break
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        gates += int(info["gate_passed"])
        if cap:
            # windowed env.step already rendered with the new pose -> grab both cameras
            main = course_cam.grab()[..., ::-1]  # RGB -> BGR for cv2
            fpv = env.quad.camera.grab()[..., ::-1]
            if main.shape == fpv.shape == (h, w, 3):
                writer.write(np.ascontiguousarray(np.hstack([main, fpv])))
        if t % 100 == 0:
            s = env._world_state
            carb.log_warn(
                f"[demo] t={t} target_gate={info['target_gate']} gates={gates} "
                f"pos_enu={np.round(s[0:3], 2)} vel={np.round(s[3:6], 2)}"
            )
        if terminated or truncated:
            carb.log_warn(f"[demo] episode end t={t}: {info}")
            break

    if writer is not None:
        writer.release()
        carb.log_warn(f"[demo] capture (course | FPV) -> {session.demo.capture_path}")

    laps = gates / course.num_gates
    carb.log_warn(f"[demo] RESULT gates_passed={gates} laps={laps:.2f}")
    print(f"gates_passed={gates} laps={laps:.2f}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
