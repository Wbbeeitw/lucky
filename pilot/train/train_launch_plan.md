# 任务 6.2：训练启动配置（写完待审批，未执行）

## 训练矩阵

| | bare | dual |
|---|---|---|
| 基座 | π0.5（pi05_base） | 同左 |
| 微调方式 | **LoRA rank 32**（双挂：PaliGemma 主干 q/v + 动作专家 q/v） | 同左 |
| 投影层/两个辅助头 | — | **全参**（modules_to_save） |
| 输入 state | 15 维（本体 9 + 指尖压力 6） | 同左 |
| 辅助损失 | 无 | 0.3·L_region + 0.1·L_force |
| 数据 | lerobot_square_v2（30154 帧，action (50,7)） | 同一份 |
| 步数 | **先 500 冒烟 → 5,000 短程** | 同左、同 seed |

## 冒烟命令（500 步，各约 5–10 分钟）

**bare：**
```bash
cd /home/enine/VTLA_scy
export HF_HOME=/data/VTLA/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl

/data/VTLA/envs/lucky_vtla/bin/lerobot-train \
  --dataset.repo_id=vtla/square_pilot \
  --dataset.root=/data/VTLA/data/lerobot_square_v2 \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.add_region_head=false \
  --policy.add_force_head=false \
  --policy.device=cuda \
  --policy.repo_id=vtla/square_pilot_bare \
  --peft.method_type=LORA \
  --peft.r=32 \
  --peft.lora_alpha=64 \
  --peft.lora_dropout=0.05 \
  --peft.target_modules='(.*gemma_expert.*self_attn\.(q|v)_proj)|(.*gemma\..*self_attn\.(q|v)_proj)' \
  --peft.full_training_modules='["state_proj","action_in_proj","action_out_proj","time_mlp_in","time_mlp_out","_region_head","_force_head"]' \
  --batch_size=32 \
  --steps=500 \
  --output_dir=/data/VTLA/ckpt/bare_500 \
  --wandb.enable=false \
  --save_freq=250
```

**dual：** 同上，仅改三处
```bash
  --policy.add_region_head=true \
  --policy.add_force_head=true \
  --output_dir=/data/VTLA/ckpt/dual_500 \
```
（λ₁=0.3、λ₂=0.1 已是补丁默认值，显式写出亦可）

## 冒烟验收标准（500 步跑完即查）

1. bare：loss 正常下降，无 NaN；
2. dual：三个损失都在（action/region/force），region 和 force **量级合理**（region MSE 数十以内、force MSE < 10）且**呈下降趋势**；
3. dual 训练后：region_head/force_head 的 LoRA-外全参参数被保存（modules_to_save 生效）；
4. 显存峰值 < 80GB；速度 ≈ 1.7 步/秒（500 步 ≈ 5 分钟）。

## 全部通过后

- 各跑 **5,000 步短程**（约 50 分钟/行）→ 机制诊断三项（任务 6 收尾）：
  1. ΔF 头 MSE < 0.7 × 惯性基线
  2. 区域重建误差 < 平均 latent 基线
  3. dual 的 L_action 最终 ≥ bare 的 95%（辅助损失没拖垮主干）
- 诊断通过 → 各 25,000 步全量（过夜）→ 任务 7 评测。

## 风险与回滚

- peft CLI 参数格式若有出入 → 第一次冒烟立刻暴露，改参数重试即可（几分钟成本）；
- dual 训练不稳 → λ 减半重跑；
- LoRA 效果异常差 → 全参路线已验证可跑，可切换（checkpoint 空间充足）。
