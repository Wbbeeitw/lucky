"""Pilot smoke training configs (Task 6.2).

Two configs for lerobot-train, differing ONLY in the dual-head switches:
  bare: add_region_head=False, add_force_head=False
  dual: add_region_head=True,  add_force_head=True, lambda 0.3/0.1

Dataset: vtla/square_pilot @ /data/VTLA/data/lerobot_square_v2 (state 15-dim)
Output:  /data/VTLA/ckpt/{bare,dual}_smoke
"""

BARE = """
# pi05 bare smoke (500 steps)
type = "pi05"
n_action_steps = 8
chunk_size = 8

# dual-head switches: OFF for bare
add_region_head = false
add_force_head = false

# processor / dtype
device = "cuda"
"""

DUAL = """
# pi05 dual smoke (500 steps)
type = "pi05"
n_action_steps = 8
chunk_size = 8

# dual-head switches: ON
add_region_head = true
add_force_head = true
lambda_region = 0.3
lambda_force = 0.1
n_pressure = 6

device = "cuda"
"""

TRAIN_CMD = """
# Smoke (500 steps each, ~20-30 min):
lerobot-train \\
  --dataset.repo_id=vtla/square_pilot \\
  --dataset.root=/data/VTLA/data/lerobot_square_v2 \\
  --policy.type=pi05 \\
  --policy.add_region_head={rh} \\
  --policy.add_force_head={fh} \\
  --policy.lambda_region=0.3 \\
  --policy.lambda_force=0.1 \\
  --policy.n_pressure=6 \\
  --policy.device=cuda \\
  --batch_size=32 \\
  --steps=500 \\
  --output_dir=/data/VTLA/ckpt/{name}_smoke \\
  --wandb.enable=false
"""

if __name__ == "__main__":
    print(BARE)
    print(DUAL)
    print(TRAIN_CMD.format(rh="false", fh="false", name="bare"))
    print(TRAIN_CMD.format(rh="true", fh="true", name="dual"))
