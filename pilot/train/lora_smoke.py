"""LoRA smoke test for pi05 dual-head (Task 6, LoRA route).

Verifies:
  1. LoRA mounts with our dual targets (expert + PaliGemma backbone q/v projections)
  2. trainable params drop from 4.15B to tens of millions
  3. aux heads (region/force) stay full-param trainable
  4. gradient flows into LoRA params of BOTH expert and backbone
  5. 50-step training loop runs
"""

import os
os.environ["HF_HOME"] = "/data/VTLA/hf"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05 import make_pi05_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.configs.default import PeftConfig

config = PI05Config(use_proprioceptive_memory=False, use_visual_memory=False,
                    n_action_steps=8, chunk_size=8, device="cuda")
config.input_features = {
    "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
}
config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
config.add_region_head = True
config.add_force_head = True

# LoRA: BOTH expert AND PaliGemma backbone attention q/v projections
peft = PeftConfig(
    method_type="LORA",
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=r"(.*gemma_expert.*self_attn\.(q|v)_proj)|(.*paligemma.*model\.self_attn\.(q|v)_proj)",
    full_training_modules=["state_proj", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"],
)
policy = PI05Policy(config, peft_config=peft) if False else None

# How does PI05Policy accept peft? Check signature
import inspect
sig = inspect.signature(PI05Policy.__init__)
print("PI05Policy.__init__ params:", list(sig.parameters.keys()))
src = inspect.getsource(PI05Policy.__init__)
print(src[:800])
