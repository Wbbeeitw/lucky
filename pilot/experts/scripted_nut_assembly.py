"""
Scripted expert for robosuite NutAssembly (Pilot, stock tolerance first).

Privileged-info scripted policy: read nut/peg poses from the sim, drive the
Panda end-effector through a grasp-transfer-insert sequence with the built-in
OSC controller, and record per-step data for the VLA pipeline.

Usage (server):
    MUJOCO_GL=egl python pilot/experts/scripted_nut_assembly.py \
        --n-episodes 5 --task stock --out /data/VTLA/data/stock
"""

import argparse
import os
import time

import numpy as np

import robosuite as suite
from robosuite.environments import MANIPULATION_ENVIRONMENTS

# ---------------------------------------------------------------------------
# Tunables (calibrate on first runs)
# ---------------------------------------------------------------------------
APPROACH_HEIGHT = 0.05      # m above nut center for approach
GRASP_Z_OFFSET = 0.005      # grasping height = nut center + this offset
LIFT_HEIGHT = 0.15          # safe transfer height above nut start
INSERT_STEP_Z = 0.001       # 1 mm descent per step (compliance = slow)
INSERT_DEPTH_TARGET = 0.02  # consider inserted after 2 cm of progress
GOTO_GAIN = 3.0             # proportional gain: delta -> action increment
GOTO_MAX_STEP = 0.1         # clamp on per-step increment (normalized)
GOTO_TOL = 0.005            # position tolerance for goto
STALL_STEPS_SPIRAL = 15     # consecutive no-progress steps before spiraling
SPIRAL_R_MAX = 0.004        # 4 mm max spiral radius
CONTROL_FREQ = 20


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------
def make_env():
    assert "NutAssembly" in MANIPULATION_ENVIRONMENTS
    env = suite.make(
        "NutAssembly",
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=224,
        camera_widths=224,
        reward_shaping=True,
        control_freq=1.0 / CONTROL_FREQ,
    )
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ee_pos(env):
    return np.array(env.sim.data.site_xpos[env.robots[0].robot_model["eef_site_id"][-1]])


def make_action(env, ee_target, gripper):
    """Build an OSC action increment toward ee_target. gripper: -1 open, +1 close."""
    cur = ee_pos(env)
    a = np.zeros(env.action_dim)
    a[:3] = np.clip((ee_target - cur) * GOTO_GAIN, -GOTO_MAX_STEP, GOTO_MAX_STEP)
    a[6] = gripper
    return a


def goto(env, target, gripper, tol=GOTO_TOL, max_steps=150):
    """Move EE toward target until close. Returns True if reached."""
    for _ in range(max_steps):
        if np.linalg.norm(ee_pos(env) - target) < tol:
            return True
        env.step(make_action(env, target, gripper))
    return np.linalg.norm(ee_pos(env) - target) < tol


def finger_pad_contact(env):
    """True if either finger pad geom currently touches the nut (V3-verified path)."""
    sim = env.sim
    id2name = {sim.model.geom_name2id(n): n for n in sim.model.geom_names if n}
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1, n2 = id2name.get(c.geom1, ""), id2name.get(c.geom2, "")
        if ("finger" in n1 and "Nut" in n2) or ("finger" in n2 and "Nut" in n1):
            return True
    return False


def read_privileged(env):
    """Privileged poses: which nut is the target, and the matching peg."""
    nut = env.nuts[env.nut_to_idx]
    peg = env.pegs[env.peg_to_idx]
    return (
        np.array(nut.get_position()),
        np.array(nut.get_orientation()),
        np.array(peg.get_position()),
    )


def peg_top(env, peg_pos):
    """Top of the peg: peg position + its half-height (read from the object body)."""
    peg = env.pegs[env.peg_to_idx]
    top_offset = peg.get_top() if hasattr(peg, "get_top") else peg_pos[2] + 0.1
    return np.array([peg_pos[0], peg_pos[1], top_offset])


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self):
        self.d = {}

    def add(self, key, value):
        self.d.setdefault(key, []).append(np.asarray(value))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **{k: np.stack(v) for k, v in self.d.items()})


def record_step(rec, env, obs, action, contacts):
    sim = env.sim
    rec.add("states", sim.get_state().flatten())
    rec.add("agentview_image", obs["agentview_image"])
    rec.add("wrist_image", obs["robot0_eye_in_hand_image"])
    rec.add("eef_pos", ee_pos(env))
    rec.add("eef_quat", np.array(env.sim.data.site_xpos[env.robots[0].robot_model["eef_site_id"][-1]]))  # placeholder quat; refine later
    rec.add("gripper_qpos", np.array(env.robots[0].gripper["joint_pos_applied"]))
    rec.add("contacts", contacts)
    rec.add("action", action)


def current_contacts(env):
    """List of (name1, name2, midpoint_xyz) for all current contacts."""
    sim = env.sim
    id2name = {sim.model.geom_name2id(n): n for n in sim.model.geom_names if n}
    out = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1, n2 = id2name.get(c.geom1, "id%d" % c.geom1), id2name.get(c.geom2, "id%d" % c.geom2)
        mid = (np.array(c.pos)).copy()
        out.append((n1, n2, mid))
    return out


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
def run_episode(env, rec):
    obs = env.reset()
    nut_pos, nut_quat, peg_pos = read_privileged(env)
    top = peg_top(env, peg_pos)

    grasp_quat = None  # keep current orientation throughout (top-down already at reset)

    # (1) approach above nut
    goto(env, nut_pos + np.array([0, 0, APPROACH_HEIGHT]), -1)
    # (2) descend to grasp height
    goto(env, nut_pos + np.array([0, 0, GRASP_Z_OFFSET]), -1)

    # (3) close until pad contact
    grasped = False
    for _ in range(40):
        a = make_action(env, ee_pos(env), 1)
        obs = env.step(a)
        rec.add("states", env.sim.get_state().flatten())
        rec.add("agentview_image", obs["agentview_image"])
        rec.add("wrist_image", obs["robot0_eye_in_hand_image"])
        rec.add("eef_pos", ee_pos(env))
        rec.add("gripper_qpos", np.array(env.robots[0].gripper["joint_pos_applied"]))
        rec.add("contacts", current_contacts(env))
        rec.add("action", a)
        if finger_pad_contact(env):
            grasped = True
            break
    if not grasped:
        return False, "no pad contact on close"

    # grasp sanity: lift 1 cm, nut must follow
    z0 = nut_pos[2]
    goto(env, ee_pos(env) + np.array([0, 0, 0.01]), 1, tol=0.002, max_steps=30)
    if abs(read_privileged(env)[0][2] - z0) < 0.005:
        return False, "nut did not follow lift"

    # (4) lift to safe height
    goto(env, nut_pos + np.array([0, 0, LIFT_HEIGHT]), 1)

    # (5) move above peg top
    goto(env, top + np.array([0, 0, 0.08]), 1)

    # (6) compliant insertion (stock: progress-based stall detection, no F/T yet)
    start_z = ee_pos(env)[2]
    depth, stall, spiral_r, spiral_ang = 0.0, 0, 0.0, 0.0
    for _ in range(500):
        new_depth = start_z - ee_pos(env)[2]
        stuck = (new_depth - depth) < 1e-4
        depth = max(depth, new_depth)

        if stuck:
            stall += 1
        else:
            stall = 0
        if stall > STALL_STEPS_SPIRAL:
            spiral_ang += np.pi / 6
            spiral_r = min(spiral_r + 0.0002, SPIRAL_R_MAX)
        else:
            spiral_r = max(spiral_r - 0.0005, 0.0)
        off = spiral_r * np.array([np.cos(spiral_ang), np.sin(spiral_ang)])

        target = ee_pos(env) + np.array([off[0], off[1], -INSERT_STEP_Z])
        a = make_action(env, target, 1)
        obs = env.step(a)
        rec.add("states", env.sim.get_state().flatten())
        rec.add("agentview_image", obs["agentview_image"])
        rec.add("wrist_image", obs["robot0_eye_in_hand_image"])
        rec.add("eef_pos", ee_pos(env))
        rec.add("gripper_qpos", np.array(env.robots[0].gripper["joint_pos_applied"]))
        rec.add("contacts", current_contacts(env))
        rec.add("action", a)

        if depth > INSERT_DEPTH_TARGET:
            break

    # (7) release
    for _ in range(20):
        env.step(make_action(env, ee_pos(env), -1))

    ok = env._check_success()
    return bool(ok), "ok" if ok else "insertion failed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--task", choices=["stock", "tight"], default="stock")
    ap.add_argument("--out", default="/data/VTLA/data/stock")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_env()
    env.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    succ = 0
    for ep in range(args.n_episodes):
        rec = Recorder()
        t0 = time.time()
        try:
            ok, why = run_episode(env, rec)
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"exception: {e}"
        if ok:
            rec.save(os.path.join(args.out, f"ep_{ep:04d}.npz"))
            succ += 1
        print(f"ep {ep:03d}: {'OK' if ok else 'FAIL'} ({why}) [{time.time()-t0:.1f}s]", flush=True)

    print(f"=== success {succ}/{args.n_episodes} ===")
    env.close()


if __name__ == "__main__":
    main()
