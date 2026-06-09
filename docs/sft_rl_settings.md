# SFT Checkpoints and RL Settings

This document explains which supervised fine-tuned (SFT) checkpoint each base
model configuration expects before online RL starts. It is meant to complement
the Quick Start in the main README.

The information below is organized by base model and benchmark. In RL configs,
the SFT checkpoint is usually set by `model.model_path`; some base-model setups may also require auxiliary
paths such as `model.lora_path`.

## Summary Table

| Base model | Benchmark / config family | RL training task from config | Required SFT checkpoint | SFT setting |
|---|---|---|---|---|
| Pi0.5 | LIBERO Spatial / Object / Goal / Long (`libero_*_openpi_pi05`) | One LIBERO suite per RL run: `libero_spatial`, `libero_object`, `libero_goal`, or `libero_10` | [`RLinf/RLinf-Pi05-LIBERO-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT). Some configs use the alias `RLinf/RLinf-Pi05-SFT`, which redirects to the LIBERO SFT repo. | Shared few-shot LIBERO SFT checkpoint across the four LIBERO suites; RL is then run on the selected single suite/task family. |
| Pi0.5 | ManiSkill 25Main (`maniskill_*_openpi_pi05`) | `PutOnPlateInScene25Main-v3` | [`RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT) | ManiSkill 25Main SFT checkpoint used as the initial policy. |
| Pi0.5 | RoboTwin (`robotwin_adjust_bottle_ppo_openpi_pi05`) | `adjust_bottle` | [`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle`](https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle) | Single-task RoboTwin SFT checkpoint; RL continues on the same task. |
| OpenVLA-OFT | LIBERO Spatial / Object / Goal / Long (`libero_*_openvlaoft`) | One LIBERO suite per RL run: `libero_spatial`, `libero_object`, `libero_goal`, or `libero_10` | [`Haozhan72/Openvla-oft-SFT-libero-spatial-traj1`](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-spatial-traj1), [`Haozhan72/Openvla-oft-SFT-libero-object-traj1`](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-object-traj1), [`Haozhan72/Openvla-oft-SFT-libero-goal-traj1`](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-goal-traj1), [`Haozhan72/Openvla-oft-SFT-libero10-traj1`](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-traj1) | Suite-specific SFT checkpoint, then RL on the same selected LIBERO suite. This differs from Pi0.5 LIBERO, which reuses one shared few-shot SFT model across the four suites. |
| OpenVLA-OFT | LIBERO 90 / 130 (`libero_90_grpo_openvlaoft`, `libero_130_grpo_openvlaoft`) | `libero_90` or `libero_130` | [`RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora`](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora), [`RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora`](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora) | RLinf-trained SFT LoRA-base checkpoints for LIBERO-90 and LIBERO-130. |
| OpenVLA-OFT | ManiSkill (`maniskill_*_openvlaoft`) | Train: `PutOnPlateInScene25Main-v3`; eval: `maniskill_ood_template` | `model_path`: [`RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main). Config comments may still show the older alias `RLinf/Openvla-oft-SFT-libero10-trajall`, which redirects to Base-Main. `lora_path`: [`RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora`](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora). | Start from the Base-Main checkpoint plus the ManiSkill LoRA. Keep `actor.model.is_lora: True` and point `actor.model.lora_path` to the downloaded LoRA unless you merge the LoRA into the base checkpoint. |
| OpenVLA-OFT | RoboTwin (`robotwin_*_openvlaoft`) | One RoboTwin task per RL run: `beat_block_hammer`, `handover_block`, `lift_pot`, `move_can_pot`, `pick_dual_bottles`, `place_container_plate`, or `place_empty_cup` | Per-task SFT repos under `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-*`, for example [`place_empty_cup`](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup). | Single-task RoboTwin SFT checkpoint; RL continues on the same RoboTwin task. |
