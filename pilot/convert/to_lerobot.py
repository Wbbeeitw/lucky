"""Convert pilot labels to LeRobotDataset format (Task 4).

Design: episodes must be contiguous in time for action-chunk training, so we write
ALL replayed frames. Labeled frames carry latent/weight (path-1 supervision);
unlabeled frames carry zero latent and weight=0 (path-1 loss auto-skipped).

Label alignment: build_labels stored obs state (eef_pos+quat+gripper) for each
labeled frame. The replay here is deterministic and identical, so labeled frames
are recovered by sequentially matching state vectors (monotonic two-pointer).

Output: LeRobotDataset at /data/VTLA/data/lerobot_square (repo vtla/square_pilot).
"""

import argparse
import json
import os

import h5py
import numpy as np

import robosuite as suite

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 20
N_LATENT, D_LATENT = 49, 768


def make_replay_env(env_args):
    args = json.loads(env_args)
    kwargs = args["env_kwargs"]
    cc = kwargs.pop("controller_configs")
    if "body_parts" in cc and "right" in cc["body_parts"]:
        cc["body_parts"]["right"]["input_ref_frame"] = "world"
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=224,
        camera_widths=224,
        controller_configs=cc,
        ignore_done=True,
    )
    kwargs.pop("camera_depths", None)
    robots = kwargs.pop("robots")
    return suite.make("NutAssemblySquare", robots=robots, **kwargs)


FEATURES = {
    "observation.images.agentview": {
        "dtype": "video",
        "shape": (3, 224, 224),
        "names": ["channels", "height", "width"],
    },
    "observation.state": {"dtype": "float32", "shape": (15,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    "pressure": {"dtype": "float32", "shape": (6,), "names": ["pressure"]},
    "latent": {"dtype": "float32", "shape": (N_LATENT, D_LATENT), "names": ["latent"]},
    "weight": {"dtype": "float32", "shape": (1,), "names": ["weight"]},
}


def obs_state(obs):
    return np.concatenate([
        np.array(obs["robot0_eef_pos"], dtype=np.float64),
        np.array(obs["robot0_eef_quat"], dtype=np.float64),
        np.array(obs["robot0_gripper_qpos"], dtype=np.float64)[:2],
    ])  # 9-dim proprio; pressure appended per-frame below (labeled-aware)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="/data/VTLA/data/robomimic_square/square_ph_demo_v15.hdf5")
    ap.add_argument("--labels", default="/data/VTLA/data/pilot_labels")
    ap.add_argument("--root", default="/data/VTLA/data/lerobot_square_v3")
    ap.add_argument("--repo-id", default="vtla/square_pilot")
    ap.add_argument("--n-demos", type=int, default=200)
    args = ap.parse_args()

    f = h5py.File(args.hdf5, "r")
    demos = f["data"]
    keys = sorted(demos.keys(), key=lambda x: int(x.split("_")[-1]))[: args.n_demos]
    env = make_replay_env(demos.attrs["env_args"])
    sim = env.sim
    nq, nv = sim.model.nq, sim.model.nv

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=FEATURES,
        root=args.root,
        robot_type="panda",
        use_videos=True,
    )

    total = labeled_total = mismatch = 0
    for k in keys:
        states = demos[k]["states"][:]
        actions = demos[k]["actions"][:]
        T = len(actions)
        lab = np.load(os.path.join(args.labels, k, "labels.npz"))
        labeled_t = lab["labeled_t"]            # (L,) original frame indices
        L = len(lab_state := lab["state"])

        env.reset()
        obs_list = []
        for t in range(T):
            sim.data.qpos[:] = states[t][1:1 + nq]
            sim.data.qvel[:] = states[t][1 + nq:1 + nq + nv]
            sim.forward()
            obs, _, _, _ = env.step(actions[t])
            obs_list.append(obs)

        labeled_set = {int(t): i for i, t in enumerate(labeled_t)}
        for t in range(T):
            i = labeled_set.get(t, -1)
            pres = lab["pressure"][i].astype(np.float64) if i >= 0 else np.zeros(6)
            st = np.concatenate([obs_state(obs_list[t]), pres])   # 9 + 6 = 15
            frame = {
                "task": "insert the square nut onto the square peg",
                "observation.images.agentview": obs_list[t]["agentview_image"],
                "observation.state": st.astype(np.float32),
                "action": np.asarray(actions[t], dtype=np.float32),
                "pressure": (lab["pressure"][i].astype(np.float32) if i >= 0 else np.zeros(6, np.float32)),
                "latent": (lab["latent"][i].astype(np.float32) if i >= 0 else np.zeros((N_LATENT, D_LATENT), np.float32)),
                "weight": (np.array([lab["weight"][i]], np.float32) if i >= 0 else np.zeros(1, np.float32)),
            }
            ds.add_frame(frame)
        ds.save_episode()

        li = labeled_set and max(labeled_set.keys()) and len(labeled_set) or 0
        if li != L:
            mismatch += 1
            print(f"WARN {k}: wrote {li}/{L} labeled frames", flush=True)
        total += T
        labeled_total += li
        print(f"{k}: {T} frames, labeled {li}/{L}", flush=True)

    ds.finalize()
    print(f"=== done: {total} frames, {labeled_total} labeled, mismatched episodes: {mismatch} ===")


if __name__ == "__main__":
    main()
