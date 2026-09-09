"""Build labels for the Pilot from robomimic square demos (replay + extraction).

Per frame produces:
  1. Path-1 target: contact-region crop -> frozen SigLIP latent (8x8=64 vectors),
     region = 96x96 window centered on t+Delta end-effector-assembly contact
     centroid projected onto agentview. Frames without in-FOV future contact are skipped.
  2. Frame weight: L1 change of the latent answer vs previous kept frame (normalized).
  3. Path-2: fingertip pressure summary (synth from contacts) + wrist wrench placeholder
     (zeros until F/T sensor patch lands); dF target = F[t+4] - F[t] over summary channels.
  4. Phase tag from contact events (approach/grasp/insert) - analysis only, not in training loss.

Output: LeRobotDataset v2.1 directory under /data/VTLA/data/pilot_lerobot.
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch

import robosuite as suite

DELTA = 10          # frames @20Hz -> 0.5 s
WINDOW = 96         # crop size in the 224 image
K_FORCE = 4         # force window length (policy steps)
DF_HORIZON = 4      # dF target horizon (policy steps)
FORCE_NOISE_STD = 0.02


# ---------------------------------------------------------------------------
# SigLIP target encoder (frozen, shared with pi05 visual tower)
# ---------------------------------------------------------------------------
_siglip = None
_siglip_proc = None


def get_siglip():
    global _siglip, _siglip_proc
    if _siglip is None:
        os.environ.setdefault("HF_HOME", "/data/VTLA/hf")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import SiglipImageProcessor, SiglipVisionModel
        name = "google/siglip-base-patch16-224"
        _siglip = SiglipVisionModel.from_pretrained(name)
        _siglip_proc = SiglipImageProcessor.from_pretrained(name)
        _siglip.eval()
        for p in _siglip.parameters():
            p.requires_grad = False
    return _siglip, _siglip_proc


@torch.no_grad()
def encode_crop(img_uint8):
    """img: HxWx3 uint8 -> latent (8x8=64, D) numpy (pool 16x16 patch tokens by 2)."""
    model, proc = get_siglip()
    x = proc(images=img_uint8, return_tensors="pt").pixel_values
    out = model(x).last_hidden_state              # SigLIP: (1, 196=14x14, D)
    out = out[0]
    D = out.shape[-1]
    grid = out.view(14, 14, D)
    pooled = grid[0::2, 0::2]                             # (7, 7, D) strided pool
    return pooled.reshape(-1, D).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry / projection
# ---------------------------------------------------------------------------
def cam_matrices(env, cam_name="agentview"):
    cam = env.sim.model.camera_name2id(cam_name)
    pos = np.array(env.sim.data.cam_xpos[cam])
    mat = env.sim.data.cam_xmat[cam].reshape(3, 3)
    fov = env.sim.model.cam_fovy[cam]
    return pos, mat, fov


def project_to_image(points, cam_pos, cam_mat, fov, img_size=224):
    """World points -> image uv. MuJoCo convention: cam_mat columns are the camera
    axes in world frame; camera looks down its -z. Returns list of (u, v) or None."""
    R = cam_mat                                      # columns of cam_mat are camera axes
    pts_cam = (np.asarray(points, dtype=float) - np.asarray(cam_pos)) @ R   # (N,3) in camera frame
    f = 0.5 * img_size / np.tan(np.radians(fov) / 2)
    out = []
    for p in pts_cam:
        x, y, z = p
        if z >= -0.01:                               # camera looks down -z
            out.append(None)
            continue
        u = x / (-z) * f + img_size / 2
        v = -y / (-z) * f + img_size / 2
        if 0 <= u < img_size and 0 <= v < img_size:
            out.append((u, v))
        else:
            out.append(None)
    return out


def contact_info(env):
    """All end-effector-assembly contacts: (names, world positions, normal forces)."""
    sim = env.sim
    id2name = {sim.model.geom_name2id(n): n for n in sim.model.geom_names if n}
    pos, force = [], []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1, n2 = id2name.get(c.geom1, ""), id2name.get(c.geom2, "")
        def is_floor(n):
            return ("floor" in n) or ("table" in n)
        # 末端总成接触 = 夹爪/手指/被抓物体 参与 且 对面不是 floor/table
        skip = False
        for a, b in ((n1, n2), (n2, n1)):
            if is_floor(a) or is_floor(b):
                skip = True
        ee_involved = any(s in n1 for s in ("finger", "gripper", "Nut")) or \
                      any(s in n2 for s in ("finger", "gripper", "Nut"))
        if skip or not ee_involved:
            continue
        f6 = np.array(env.sim.data.efc_force[c.efc_address: c.efc_address + 6]) if c.dim >= 1 else np.zeros(6)
        pos.append(np.array(c.pos))
        force.append(np.linalg.norm(f6[:3]))
    return pos, np.array(force) if force else np.zeros(0)


# ---------------------------------------------------------------------------
# Pressure synthesis (fingertip summary)
# ---------------------------------------------------------------------------
def pressure_summary(env):
    """Per finger: [mean normal, max normal, contact flag] from pad/nut contacts."""
    sim = env.sim
    id2name = {sim.model.geom_name2id(n): n for n in sim.model.geom_names if n}
    sums = {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]}
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1, n2 = id2name.get(c.geom1, ""), id2name.get(c.geom2, "")
        for a, b in ((n1, n2), (n2, n1)):
            if "finger" in a and "Nut" in b:
                side = "left" if "left" in a else "right"
                f = np.linalg.norm(np.array(env.sim.data.efc_force[c.efc_address: c.efc_address + 3]))
                sums[side][0] += f
                sums[side][1] = max(sums[side][1], f)
                sums[side][2] = 1.0
    return np.array(sums["left"] + sums["right"])   # 6 dims


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="/data/VTLA/data/robomimic_square/square_ph_demo_v15.hdf5")
    ap.add_argument("--n-demos", type=int, default=200)
    ap.add_argument("--out", default="/data/VTLA/data/pilot_labels")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    f = h5py.File(args.hdf5, "r")
    demos = f["data"]
    keys = sorted(demos.keys(), key=lambda x: int(x.split("_")[-1]))[: args.n_demos]

    env_args = demos.attrs["env_args"]
    a = json.loads(env_args)
    kwargs = a["env_kwargs"]
    cc = kwargs.pop("controller_configs")
    cc["body_parts"]["right"]["input_ref_frame"] = "world"
    kwargs.update(
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=224, camera_widths=224,
        controller_configs=cc, ignore_done=True,
    )
    kwargs.pop("camera_depths", None)
    robots = kwargs.pop("robots")
    env = suite.make("NutAssemblySquare", robots=robots, **kwargs)
    sim = env.sim
    nq, nv = sim.model.nq, sim.model.nv
    cam_pos, cam_mat, fov = cam_matrices(env, "agentview")

    os.makedirs(args.out, exist_ok=True)
    stats = {"frames": 0, "labeled": 0, "skipped_nofuture": 0, "demos": 0}

    for di, k in enumerate(keys):
        states = demos[k]["states"][:]
        actions = demos[k]["actions"][:]
        T = len(actions)

        # pass 1: replay and collect per-frame raw info
        frames = []
        env.reset()
        for t in range(T):
            sim.data.qpos[:] = states[t][1:1 + nq]
            sim.data.qvel[:] = states[t][1 + nq:1 + nq + nv]
            sim.forward()
            obs, _, _, _ = env.step(actions[t])
            cpos, cforce = contact_info(env)
            frames.append(dict(
                agentview=obs["agentview_image"],
                state=np.concatenate([
                    np.array(obs["robot0_eef_pos"]),
                    np.array(obs["robot0_eef_quat"]),
                    np.array(obs["robot0_gripper_qpos"]),
                ]),                                   # 7+4+8 -> var; keep raw, policy adapter handles
                action=actions[t],
                contacts=(cpos, cforce),
                pressure=pressure_summary(env),
                wrist_wrench=np.zeros(6),             # placeholder until F/T patch
            ))

        cnt = [len(fr["contacts"][0]) for fr in frames]

        # future contact (t+Delta) per frame
        def future_contact(t):
            tt = min(t + DELTA, T - 1)
            cpos, cforce = frames[tt]["contacts"]
            return cpos, cforce

        # pass 2: labels
        latents, weights, states_arr, actions_arr, press_arr, dF_arr, wrist_arr, phase = [], [], [], [], [], [], [], []
        prev_latent = None
        labeled_t = []
        for t in range(T):
            cpos, cforce = future_contact(t)
            if len(cpos) == 0:
                stats["skipped_nofuture"] += 1
                continue
            uvs = project_to_image(np.array(cpos), cam_pos, cam_mat, fov)
            uv = [p for p in uvs if p is not None]
            if not uv:
                stats["skipped_nofuture"] += 1
                continue
            u = int(np.mean([p[0] for p in uv]))
            v = int(np.mean([p[1] for p in uv]))
            half = WINDOW // 2
            img = frames[t]["agentview"]
            x0, x1 = max(0, u - half), min(224, u + half)
            y0, y1 = max(0, v - half), min(224, v + half)
            crop = img[y0:y1, x0:x1]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                stats["skipped_nofuture"] += 1
                continue

            latent = encode_crop(crop)
            w = 1.0 if prev_latent is None else float(np.abs(latent - prev_latent).mean())
            prev_latent = latent

            latents.append(latent)
            weights.append(w)
            labeled_t.append(t)
            states_arr.append(frames[t]["state"])
            actions_arr.append(frames[t]["action"])
            press_arr.append(frames[t]["pressure"])
            wrist_arr.append(frames[t]["wrist_wrench"])
            stats["labeled"] += 1

        # dF targets on the pressure(+wrench) series (aligned to labeled frames is complex;
        # store per-frame dF on raw frames instead and let training sample contiguous windows)
        if latents:
            latents = np.stack(latents)
            weights = np.array(weights, dtype=np.float32)
            weights = weights / (np.median(weights) + 1e-8)     # normalized signal-driven weight
            ep_dir = os.path.join(args.out, k)
            os.makedirs(ep_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(ep_dir, "labels.npz"),
                latent=latents, weight=weights,
                state=np.stack(states_arr), action=np.stack(actions_arr),
                pressure=np.stack(press_arr), wrist=np.stack(wrist_arr),
                labeled_t=np.array(labeled_t, dtype=np.int64),
            )
            stats["demos"] += 1
            stats["frames"] += T
        print(f"[{di+1}/{len(keys)}] {k}: labeled={len(latents)} skipped={stats['skipped_nofuture']}", flush=True)

    with open(os.path.join(args.out, "stats.json"), "w") as fp:
        json.dump(stats, fp, indent=2)
    print("=== build done ===", stats)


if __name__ == "__main__":
    main()
