"""Patch script: apply dual-head modifications to lerobot pi05 sources.

Run ONCE from /home/enine/VTLA_scy:
    python pilot/patches/apply_pi05_dual_patch.py

Idempotent: skips hunks already present. Keeps a .orig backup of each touched file.
The user-approved design:
  - config: add_region_head / add_force_head / lambda_region / lambda_force / n_pressure
  - PI05Pytorch.forward: return (loss, cache) where cache carries prefix tokens +
    expert trunk features (only when heads enabled); Policy.forward computes the
    aux losses and adds lambda-weighted terms.
"""

import re
import shutil
from pathlib import Path

ROOT = Path("/home/enine/VTLA_scy/lerobot/src/lerobot/policies/pi05")
PATCH_DIR = Path("/home/enine/VTLA_scy/pilot/patches")

CONFIG_F = ROOT / "configuration_pi05.py"
MODEL_F = ROOT / "modeling_pi05.py"


def patch_config(src: str) -> str:
    if "add_region_head" in src:
        print("config: already patched")
        return src
    anchor = '    n_action_steps: int = 50  # Number of action steps to execute\n'
    assert anchor in src, "config anchor missing"
    add = anchor + """
    # --- Dual-pathway tactile heads (Pilot; see pilot/patches/aux_heads.py) ---
    add_region_head: bool = False  # Path-1: contact-region reconstruction (SigLIP latent)
    add_force_head: bool = False   # Path-2: dF pressure-change prediction
    lambda_region: float = 0.3
    lambda_force: float = 0.1
    n_pressure: int = 6            # fingertip pressure summary dims (input/state tail)
"""
    src = src.replace(anchor, add)
    return src


def patch_model(src: str) -> str:
    changed = False

    # 1) import aux heads
    if "aux_heads" not in src:
        anchor = "from torch import Tensor, nn\n"
        assert anchor in src, "import anchor missing"
        src = src.replace(
            anchor,
            anchor + "\nimport sys\nsys.path.insert(0, str(__import__('pathlib').Path('/home/enine/VTLA_scy/pilot/patches')))\nfrom aux_heads import RegionHead, ForceHead, region_loss, force_loss\n",
        )
        changed = True
        print("model: aux_heads import added")

    # 2) PI05Pytorch.__init__: create heads when config asks
    if "_region_head" not in src:
        anchor = "        self.model.to(config.device)"
        # PI05Pytorch init: find its own anchor (inside PI05Pytorch class scope)
        anchor2 = "        self.action_out_proj"
        assert anchor2 in src, "pytorch init anchor missing"
        # insert right after the first occurrence of action_out_proj assignment block
        m = re.search(r"(        self\.action_out_proj[^\n]*\n)", src)
        assert m, "action_out_proj assignment missing"
        add = m.group(1) + """
        # Dual-pathway heads (opt-in)
        self._aux_enabled = bool(getattr(config, "add_region_head", False) or getattr(config, "add_force_head", False))
        self._region_enabled = bool(getattr(config, "add_region_head", False))
        self._force_enabled = bool(getattr(config, "add_force_head", False))
        if self._region_enabled:
            self._region_head = RegionHead(
                cond_dim=self.config.vision_hidden_size if hasattr(self.config, "vision_hidden_size") else 2048,
                latent_dim=768, n_latent=49, n_queries=49, depth=3, hidden=512)
        if self._force_enabled:
            self._force_head = ForceHead(
                feat_dim=self.config.action_expert_variant if isinstance(self.config.action_expert_variant, int) else 1024,
                n_out=getattr(config, "n_pressure", 6))
"""
        src = src.replace(m.group(1), add, 1)
        changed = True
        print("model: PI05Pytorch heads init added")

    # 3) PI05Pytorch.forward: also return cache (prefix tokens + suffix trunk feats)
    old_ret = "        return F.mse_loss(u_t, v_t, reduction=\"none\")"
    if "aux_cache" not in src:
        assert old_ret in src, "fm return anchor missing"
        new_ret = """        aux_cache = None
        if getattr(self, "_aux_enabled", False):
            # prefix tokens: image+language fused (before expert); used by RegionHead.
            # suffix_out (expert trunk, state-injected) last-token features: used by ForceHead.
            aux_cache = {
                "prefix_embs": prefix_embs,
                "suffix_out": suffix_out,
            }
        return F.mse_loss(u_t, v_t, reduction="none"), aux_cache"""
        src = src.replace(old_ret, new_ret)
        changed = True
        print("model: PI05Pytorch.forward returns aux_cache")

    # 4) callers of self.model.forward inside PI05Policy.forward: adapt unpack
    old_call = """        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            prefix_mask=prefix_mask,
            states=states,
            state_masks=state_masks,
        )"""
    if "aux_cache" in src and old_call in src:
        new_call = """        losses, aux_cache = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            prefix_mask=prefix_mask,
            states=states,
            state_masks=state_masks,
        )"""
        src = src.replace(old_call, new_call)
        changed = True
        print("model: PI05Policy.forward unpacks aux_cache")

    # 5) PI05Policy.forward: add aux losses after loss_dict init
    old_ld = '        loss_dict = {"loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist()}'
    if "region_loss" in src and "lambda_region" in src:
        print("model: aux loss block already present")
    else:
        assert old_ld in src, "loss_dict anchor missing"
        add = old_ld + """

        # --- Dual-pathway auxiliary losses (Pilot) ---
        aux_total = None
        if getattr(self.model, "_aux_enabled", False):
            if aux_cache is not None:
                if getattr(self.model, "_region_enabled", False) and ("latent" in batch):
                    # RegionHead consumes the whole prefix (image+lang fused) — position-blind.
                    prefix_embs = aux_cache["prefix_embs"]
                    pred = self.model._region_head(aux_cache["prefix_embs"].float())
                    rl, rlog = region_loss(pred, batch["latent"].float(), batch["weight"].float())
                    loss_dict.update({k: v for k, v in rlog.items()})
                    aux_total = self.config.lambda_region * rl if aux_total is None else aux_total + self.config.lambda_region * rl
                if getattr(self.model, "_force_enabled", False) and ("pressure" in batch):
                    # dF target from pressure sequence in batch: (B, T, 6) -> change over horizon
                    p = batch["pressure"].float()
                    if p.ndim == 3 and p.shape[1] > 1:
                        df_gt = p[:, -1, :] - p[:, 0, :]
                    else:
                        df_gt = torch.zeros(p.shape[0], getattr(self.model._force_head, "net", None) is not None and p.shape[-1] or 6, device=p.device)
                    feat = aux_cache["suffix_out"].float().mean(dim=1)  # trunk summary (B, feat_dim)
                    fp = self.model._force_head(feat)
                    if fp.shape[-1] != df_gt.shape[-1]:
                        df_gt = F.interpolate(df_gt.unsqueeze(1), size=fp.shape[-1], mode="linear", align_corners=False).squeeze(1)
                    fl, flog = force_loss(fp, df_gt)
                    loss_dict.update(flog)
                    aux_total = self.config.lambda_force * fl if aux_total is None else aux_total + self.config.lambda_force * fl
            if aux_total is not None:
                loss_dict["aux_total"] = float(aux_total)"""
        src = src.replace(old_ld, add)
        changed = True
        print("model: PI05Policy aux loss block added")

    return src


def patch_model_returns(src: str) -> str:
    """PI05Pytorch.forward returns tuple now; fix the non-aux return path too."""
    if "return F.mse_loss(u_t, v_t, reduction=\"none\"), None" in src:
        return src
    src = src.replace(
        "return F.mse_loss(u_t, v_t, reduction=\"none\")",
        "return F.mse_loss(u_t, v_t, reduction=\"none\"), None",
    )
    return src


def main():
    for f in (CONFIG_F, MODEL_F):
        bak = f.with_suffix(".py.orig")
        if not bak.exists():
            shutil.copy(f, bak)
            print(f"backup: {bak}")

    cfg = CONFIG_F.read_text()
    CONFIG_F.write_text(patch_config(cfg))

    model = MODEL_F.read_text()
    model = patch_model_returns(patch_model(model))
    MODEL_F.write_text(model)

    # syntax check
    import ast
    for f in (CONFIG_F, MODEL_F):
        ast.parse(f.read_text())
        print(f"syntax OK: {f.name}")


if __name__ == "__main__":
    main()
