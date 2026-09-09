"""Replay robomimic square PH demos in robosuite 1.5.2 - VERIFIED protocol.

Verified on server (3 demos): pad-x-nut contacts 501-596 per demo,
qpos replay error mean ~0.10 / max 0.25 (robosuite 1.5.0 -> 1.5.2 drift),
agentview 224x224 rendering OK.

Protocol (robomimic closed-loop convention):
  for t: set qpos/qvel from states[t] -> forward -> step(actions[t])
         compare against states[t+1]

Env config requirements discovered during debugging:
  - input_ref_frame="world" (robosuite 1.5.2 default is "base" - demo used world)
  - ignore_done=True (demo length exceeds default horizon)
  - controller_configs from demo env_args MUST be passed through
"""

import argparse
import json

import h5py
import numpy as np

import robosuite as suite


def make_replay_env(env_args, camera_height=224, camera_width=224):
    args = json.loads(env_args)
    kwargs = args["env_kwargs"]
    controller_configs = kwargs.pop("controller_configs")
    cc = controller_configs
    if "body_parts" in cc and "right" in cc["body_parts"]:
        cc["body_parts"]["right"]["input_ref_frame"] = "world"
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        controller_configs=controller_configs,
        ignore_done=True,
    )
    kwargs.pop("camera_depths", None)
    robots = kwargs.pop("robots")
    env = suite.make("NutAssemblySquare", robots=robots, **kwargs)
    return env


def geom_names(sim):
    return {sim.model.geom_name2id(n): n for n in sim.model.geom_names if n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="/data/VTLA/data/robomimic_square/square_ph_demo_v15.hdf5")
    ap.add_argument("--n-demos", type=int, default=3)
    args = ap.parse_args()

    f = h5py.File(args.hdf5, "r")
    demos = f["data"]
    env_args = demos.attrs["env_args"]
    keys = sorted(demos.keys(), key=lambda x: int(x.split("_")[-1]))[: args.n_demos]

    env = make_replay_env(env_args)
    sim = env.sim
    nq, nv = sim.model.nq, sim.model.nv
    id2name = geom_names(sim)

    for k in keys:
        states = demos[k]["states"][:]
        actions = demos[k]["actions"][:]
        obs = env.reset()

        pad_nut = 0
        contacts_total = 0
        qerr = []
        img_ok = True
        for t in range(len(actions) - 1):
            sim.data.qpos[:] = states[t][1:1 + nq]
            sim.data.qvel[:] = states[t][1 + nq:1 + nq + nv]
            sim.forward()
            obs, _, _, _ = env.step(actions[t])
            contacts_total += sim.data.ncon
            qerr.append(np.abs(sim.data.qpos[:nq] - states[t + 1][1:1 + nq]).max())
            for i in range(sim.data.ncon):
                c = sim.data.contact[i]
                n1, n2 = id2name.get(c.geom1, ""), id2name.get(c.geom2, "")
                if ("finger" in n1 and "Nut" in n2) or ("finger" in n2 and "Nut" in n1):
                    pad_nut += 1
            if "agentview_image" not in obs or obs["agentview_image"].shape != (224, 224, 3):
                img_ok = False

        print(
            "%s: len=%d | contacts=%d | pad-x-nut=%d | img224=%s | qpos_err max=%.4f mean=%.4f" %
            (k, len(actions), contacts_total, pad_nut, img_ok, max(qerr), np.mean(qerr)),
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    main()
