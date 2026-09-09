"""Scripted expert for robosuite NutAssembly (Pilot, stock tolerance).

Fixed against robosuite 1.5.2 actual API:
- nut_id only exists when nut_type is set -> pass nut_type="square" at make()
- nut/peg poses read via body ids (body_name2id + body_xpos)
"""

import argparse
import os
import time

import numpy as np

import robosuite as suite

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
APPROACH_HEIGHT = 0.05
GRASP_Z_OFFSET = 0.005
LIFT_HEIGHT = 0.15
INSERT_STEP_Z = 0.001
INSERT_DEPTH_TARGET = 0.02
PEG_TOP_OFFSET = 0.03
GOTO_GAIN = 3.0
GOTO_MAX_STEP = 0.1
GOTO_TOL = 0.005
STALL_STEPS_SPIRAL = 15
SPIRAL_R_MAX = 0.004
CONTROL_FREQ = 20


def make_env():
    return suite.make(
        "NutAssembly",
        robots=["Panda"],
        env_configuration="single-arm-opposed",
        nut_type="square",                    # <-- makes env.nut_id exist
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=224,
        camera_widths=224,
        reward_shaping=True,
        control_freq=1.0 / CONTROL_FREQ,
    )


def ee_pos(env):
    site_id = env.robots[0].eef_site_id
    if isinstance(site_id, dict):
        site_id = list(site_id.values())[0]
    return np.array(env.sim.data.site_xpos[site_id])


def make_action(env, ee_target, gripper):
    cur = ee_pos(env)
    a = np.zeros(env.action_dim)
    a[:3] = np.clip((ee_target - cur) * GOTO_GAIN, -GOTO_MAX_STEP, GOTO_MAX_STEP)
    a[6] = gripper
    return a


def goto(env, target, gripper, tol=GOTO_TOL, max_steps=150):
    for _ in range(max_steps):
        if np.linalg.norm(ee_pos(env) - target) < tol:
            return True
        env.step(make_action(env, target, gripper))
    return np.linalg.norm(ee_pos(env) - target) < tol


def finger_pad_contact(env):
    sim = env.sim
    id2name = {sim.model.geom_name2id(n): n for n in sim.model.geom_names if n}
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1, n2 = id2name.get(c.geom1, ""), id2name.get(c.geom2, "")
        if ("finger" in n1 and "Nut" in n2) or ("finger" in n2 and "Nut" in n1):
            return True
    return False


def read_privileged(env):
    """Target nut + matching peg world positions via body ids."""
    nut = env.nuts[env.nut_id]
    nut_body = nut.root_body
    nut_pos = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(nut_body)])
    peg_body = env.peg1_body_id if env.nut_id == 0 else env.peg2_body_id
    peg_pos = np.array(env.sim.data.body_xpos[peg_body])
    return nut_pos, peg_pos


def peg_top(peg_pos):
    return peg_pos + np.array([0, 0, PEG_TOP_OFFSET])


class Recorder:
    def __init__(self):
        self.d = {}

    def add(self, key, value):
        self.d.setdefault(key, []).append(np.asarray(value))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrs = {}
        for k, v in self.d.items():
            try:
                arrs[k] = np.stack(v)
            except ValueError:
                arrs[k] = np.empty(len(v), dtype=object)  # ragged (e.g. contacts)
                for i, x in enumerate(v):
                    arrs[k][i] = x
        np.savez_compressed(path, **arrs)


def current_contacts(env):
    sim = env.sim
    id2name = {sim.model.geom_name2id(n): n for n in sim.model.geom_names if n}
    out = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        out.append((id2name.get(c.geom1, "id%d" % c.geom1),
                    id2name.get(c.geom2, "id%d" % c.geom2),
                    np.array(c.pos).copy()))
    return out


def log_step(rec, env, obs, a):
    rec.add("states", env.sim.get_state().flatten())
    rec.add("agentview_image", obs["agentview_image"])
    rec.add("wrist_image", obs["robot0_eye_in_hand_image"])
    rec.add("eef_pos", ee_pos(env))
    rec.add("gripper_qpos", np.array(obs["robot0_gripper_qpos"]))
    rec.add("contacts", current_contacts(env))
    rec.add("action", a)


def run_episode(env, rec):
    obs = env.reset()
    nut_pos, peg_pos = read_privileged(env)
    top = peg_top(peg_pos)

    # (1)(2) approach + descend
    goto(env, nut_pos + np.array([0, 0, APPROACH_HEIGHT]), -1)
    goto(env, nut_pos + np.array([0, 0, GRASP_Z_OFFSET]), -1)

    # (3) close until pad contact
    grasped = False
    for _ in range(40):
        a = make_action(env, ee_pos(env), 1)
        obs, _, _, _ = env.step(a)
        log_step(rec, env, obs, a)
        if finger_pad_contact(env):
            grasped = True
            break
    if not grasped:
        return False, "no pad contact on close"

    # grasp sanity: nut follows a small lift
    z0 = read_privileged(env)[0][2]
    goto(env, ee_pos(env) + np.array([0, 0, 0.01]), 1, tol=0.002, max_steps=30)
    if abs(read_privileged(env)[0][2] - z0) < 0.005:
        return False, "nut did not follow lift"

    # (4)(5) lift + above peg
    goto(env, nut_pos + np.array([0, 0, LIFT_HEIGHT]), 1)
    goto(env, top + np.array([0, 0, 0.08]), 1)

    # (6) compliant insertion
    start_z = ee_pos(env)[2]
    depth, stall, spiral_r, spiral_ang = 0.0, 0, 0.0, 0.0
    for _ in range(500):
        new_depth = start_z - ee_pos(env)[2]
        stuck = (new_depth - depth) < 1e-4
        depth = max(depth, new_depth)
        stall = stall + 1 if stuck else 0
        if stall > STALL_STEPS_SPIRAL:
            spiral_ang += np.pi / 6
            spiral_r = min(spiral_r + 0.0002, SPIRAL_R_MAX)
        else:
            spiral_r = max(spiral_r - 0.0005, 0.0)
        off = spiral_r * np.array([np.cos(spiral_ang), np.sin(spiral_ang)])
        a = make_action(env, ee_pos(env) + np.array([off[0], off[1], -INSERT_STEP_Z]), 1)
        obs, _, _, _ = env.step(a)
        log_step(rec, env, obs, a)
        if depth > INSERT_DEPTH_TARGET:
            break

    # (7) release
    for _ in range(20):
        env.step(make_action(env, ee_pos(env), -1))
    ok = env._check_success()
    return bool(ok), "ok" if ok else "insertion failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--task", choices=["stock", "tight"], default="stock")
    ap.add_argument("--out", default="/data/VTLA/data/stock")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_env()
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    succ = 0
    for ep in range(args.n_episodes):
        rec = Recorder()
        t0 = time.time()
        try:
            ok, why = run_episode(env, rec)
        except Exception as e:  # noqa: BLE001
            ok, why = False, "exception: %s" % e
        if ok:
            rec.save(os.path.join(args.out, "ep_%04d.npz" % ep))
            succ += 1
        print("ep %03d: %s (%s) [%.1fs]" % (ep, "OK" if ok else "FAIL", why, time.time() - t0), flush=True)

    print("=== success %d/%d ===" % (succ, args.n_episodes))
    env.close()


if __name__ == "__main__":
    main()
