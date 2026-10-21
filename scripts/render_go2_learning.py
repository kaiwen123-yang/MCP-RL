"""Replay saved COMPLETE Go2 evaluations, synchronized by physical task time.

Usage (on the simulation server, with MUJOCO_GL=egl):
    python scripts/render_go2_learning.py --run-root work/complete-round \
        --assets /path/to/assets --output outputs/go2-replay

This restores recorded base poses and joint positions at 50 Hz and renders at
25 fps. It does not integrate physics, infer actions, optimize, or claim a new
experiment. A finished episode freezes visibly at its final recorded state.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from go2_robot_lab.simulator import Go2Sim
from go2_robot_lab.task import Task, make_cases


ARMS = (("initial", "共同起点", "initial/evaluation", "#91a4b6"),
        ("ordinary", "普通 PPO", "ordinary/round1/evaluation", "#59b5ee"),
        ("pathwise", "整段计划辅助学习", "pathwise/round1/evaluation", "#e7ae61"))
STAGES = ("直行", "转弯", "过门", "接近与停稳", "完整成功")
WIDTH, VIEW_HEIGHT, HEIGHT, FPS = 640, 450, 720, 25


def quaternion(rpy):
    roll, pitch, yaw = np.asarray(rpy) / 2
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch),
                              np.sin(pitch), np.cos(yaw), np.sin(yaw))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def load_episode(folder, case_index, case_name):
    saved = torch.load(folder / "transitions.pt", map_location="cpu", weights_only=True)
    batch = saved["batch"]
    selected = batch["case_index"] == case_index
    summary = json.loads((folder / "summary.json").read_text())
    episodes = [e for e in summary["episodes"] if e["case"] == case_name]
    if len(episodes) != 1 or not episodes[0]["complete"]:
        raise ValueError("Replay expects exactly one complete evaluation per case")
    states = torch.cat((batch["state"][selected][:1], batch["next_state"][selected])).numpy()
    memories = torch.cat((batch["memory"][selected][:1], batch["next_memory"][selected])).numpy()
    commands = batch["command"][selected].numpy()
    return states, memories, commands, episodes[0]


def add_route(renderer, task):
    points = task.reference(np.linspace(0.0, task.spec.total_length, 90))
    for point in points:
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([0.018, 0, 0]), np.array([*point[:2], 0.012]),
                           np.eye(3).flatten(), np.array([0.08, 0.45, 0.55, 0.8]))
        renderer.scene.ngeom += 1
    geom = renderer.scene.geoms[renderer.scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                       np.array([task.spec.goal_position_tolerance, 0.005, 0]),
                       np.array([*task.spec.goal, 0.008]), np.eye(3).flatten(),
                       np.array([0.15, 0.7, 0.45, 0.35]))
    renderer.scene.ngeom += 1


def render_case(args, case_index, spec, fonts):
    task = Task(spec)
    output = args.output
    sim = Go2Sim(args.assets / "rl_sar/policy/go2",
                 args.assets / "rl_sar_zoo/go2_description/mjcf",
                 output / "scenes" / spec.name, obstacles=task.obstacles())
    # Only visualization settings: the saved simulator states remain untouched.
    sim.model.geom_rgba[sim.model.geom_bodyid == 0, 3] = 0.55
    episodes = [load_episode(args.run_root / relative, case_index, spec.name)
                for _, _, relative, _ in ARMS]
    last_time = max(float(ep[3]["elapsed_seconds"]) for ep in episodes)
    frames = math.ceil((last_time + 1.2) * FPS)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1.65, 1.65, 0.1]
    camera.distance, camera.azimuth, camera.elevation = 8.0, -125, -57
    video_path = output / f"{spec.name}.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{WIDTH * 3}x{HEIGHT}", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
           "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    font, title_font, small, footer = fonts
    try:
        with mujoco.Renderer(sim.model, height=VIEW_HEIGHT, width=WIDTH) as renderer:
            for frame in range(frames):
                now = frame / FPS
                canvas = Image.new("RGB", (WIDTH * 3, HEIGHT), "#101a25")
                draw = ImageDraw.Draw(canvas)
                for panel, ((arm, title, _, color), (states, memories, commands, episode)) in enumerate(zip(ARMS, episodes)):
                    index = min(int(np.searchsorted(memories[:, 1], now, side="right") - 1), len(states) - 1)
                    index = max(0, index)
                    state, memory = states[index], memories[index]
                    sim.data.qpos[:3] = state[:3]
                    sim.data.qpos[3:7] = quaternion(state[3:6])
                    sim.data.qpos[sim.q_addresses] = state[12:24]
                    mujoco.mj_forward(sim.model, sim.data)
                    renderer.update_scene(sim.data, camera=camera)
                    add_route(renderer, task)
                    x = panel * WIDTH
                    canvas.paste(Image.fromarray(renderer.render()), (x, 70))
                    draw.rectangle((x, 0, x + WIDTH, 5), fill=color)
                    draw.text((x + 24, 17), title, font=title_font, fill=color)
                    draw.text((x + 24, 529), f"共同时间 {now:05.2f} s   ·   {spec.name}", font=font, fill="white")
                    finished = now >= episode["elapsed_seconds"]
                    stage = STAGES[int(memory[0])] if int(memory[0]) < len(STAGES) else str(memory[0])
                    if finished:
                        stage = "完整成功 · 保持最后一帧" if episode["success"] else episode["reason"]
                    draw.text((x + 24, 566), stage, font=font, fill=color)
                    speed = np.linalg.norm(state[6:8])
                    distance = np.linalg.norm(state[:2] - spec.goal)
                    draw.text((x + 24, 604), f"速度 {speed:.2f} m/s   距终点 {distance:.2f} m", font=small, fill="#c9d5df")
                    completion = episode["completion_seconds"]
                    result = f"完成用时 {completion:.2f} s" if completion is not None else f"结局 {episode['reason']}"
                    draw.text((x + 24, 637), result, font=small, fill="#c9d5df")
                    for event in episode["stage_exits"]:
                        position = x + 24 + 590 * float(event["time"]) / last_time
                        draw.line((position, 678, position, 690), fill=color, width=2)
                    draw.line((x + 24, 684, x + 614, 684), fill="#3b4a5b", width=3)
                    draw.line((x + 24, 684, x + 24 + 590 * min(now, episode["elapsed_seconds"]) / last_time, 684), fill=color, width=4)
                draw.text((24, 696), "实际保存轨迹回放 · 相同物理时间 · 原场景与成功判据 · 25 fps · 不包含站立预热", font=footer, fill="#a5b5c4")
                encoder.stdin.write(np.asarray(canvas).tobytes())
                if frame == round(9.5 * FPS):
                    canvas.save(output / f"{spec.name}-preview.png")
                if frame in (0, frames - 1):
                    canvas.save(output / f"{spec.name}-{'start' if frame == 0 else 'end'}.png")
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f"Video encoder exited with {code}")
    return {"case": spec.name, "video": video_path.name, "frames": frames, "fps": FPS,
            "episode_seconds": {arm[0]: data[3]["elapsed_seconds"] for arm, data in zip(ARMS, episodes)},
            "source": "saved 50 Hz physical states; no physics stepping or parameter updates"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=[c.name for c in make_cases()])
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = tuple(ImageFont.truetype(str(args.font), size) for size in (25, 30, 20, 16))
    records = []
    for index, case in enumerate(make_cases()):
        if args.case and case.name != args.case:
            continue
        records.append(render_case(args, index, case, fonts))
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    (args.output / "replay.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
