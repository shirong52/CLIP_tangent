# 一阶切线对齐CLIP微调 — 实验代码实施计划

## TL;DR

> **Quick Summary**: 将 exp.md 中的一阶/切线对齐CLIP微调实验方案分解为可执行的代码任务。项目已有完整的数据集生成器和可视化管线，需要实现的核心是：模型封装、损失函数、训练管道、评估框架、基线对比和消融实验。
>
> **Deliverables**:
> - 项目基础设施（配置系统、依赖管理、目录结构）
> - CLIP 模型封装（支持投影头微调 / LoRA / 全量微调）
> - PyTorch 数据集封装（加载 orig/pert 配对）
> - 扰动算子库（文本端 + 图像端训练时增强）
> - InfoNCE 检索损失 + 切线对齐损失
> - 三阶段训练管道（Stage1: 投影头 / Stage2: LoRA / Stage3: 全量微调）
> - 评估框架（标准检索 Recall@K / 组合准确率 / 扰动一致性 / 消融）
> - 三个基线实现（冻结CLIP / 标准微调 / 增强一致性）
> - 实验跟踪（W&B）+ 结果聚合报告
>
> **Estimated Effort**: XL (Large Research Implementation)
> **Parallel Execution**: YES - 5 waves, max 8 concurrent tasks
> **Critical Path**: T1 → T4 → T8 → T15 → T22

---

## Context

### Original Request
用户要求根据 `exp.md` 的一阶/切线对齐CLIP微调实验方案，制定更细致的执行计划。当前 exp.md 涵盖了实验设计的完整框架但缺乏代码层面的实现细节。

### Interview Summary
**Key Discussions**:
- 项目已完成数据集生成管线（Blender合成渲染，6种扰动因素）
- 项目已完成CLIP嵌入提取和UMAP可视化分析
- 缺失的核心组件：训练管道、损失函数、评估框架、基线对比
- 使用 HuggingFace transformers 的 CLIPModel/CLIPProcessor
- Python 技术栈：torch, transformers, numpy, matplotlib, umap-learn

**Research Findings**:
- CLIP 加载方式已在 `extract_embeddings.py` 中验证（vision_model CLS → visual_projection, text_model EOS → text_projection）
- 数据集格式为 JSONL（每条记录包含 orig/pert 配对、因素标签、场景规格）
- 模型路径：`/root/autodl-tmp/model/clip_vit_b`
- 数据路径：`/root/autodl-tmp/dataset/composition/controlled_pairs/`
- **GPU 可用 (CUDA)**：启用 AMP、混合精度训练、Stage 3 全量微调
- **数据集已渲染**：Blender 合成图片已生成，可直接用于训练

### Metis Review
Metis consultation timed out. Self-conducted gap analysis:
- **Missing**: No requirements.txt, no config system, no training code
- **Assumption**: GPU available, Blender dataset already rendered
- **Guardrail**: Must not modify existing dataset/visualize code

---

## Work Objectives

### Core Objective
实现 exp.md 中描述的完整实验代码：CLIP切线对齐微调训练管道、损失函数、评估框架和基线对比。

### Concrete Deliverables
1. `requirements.txt` — Python 依赖清单
2. `config/` — YAML 配置系统 + CLI 参数覆盖
3. `src/models/clip_wrapper.py` — CLIP 模型封装（支持投影头/LoRA/全量微调）
4. `src/data/controlled_dataset.py` — PyTorch 数据集封装
5. `src/data/perturbation_ops.py` — 扰动算子库（文本端 + 图像端）
6. `src/losses/info_nce.py` — InfoNCE 双向检索损失
7. `src/losses/tangent.py` — 切线对齐损失 L_tan = 1 - cos(ΔI, ΔT)
8. `src/training/trainer.py` — 训练器（支持三阶段）
9. `src/training/train.py` — 训练入口脚本
10. `src/eval/retrieval.py` — 标准检索评估器（Recall@K）
11. `src/eval/compositional.py` — 组合准确率评估器
12. `src/eval/consistency.py` — 扰动一致性评估器
13. `src/baselines/frozen_clip.py` — 冻结CLIP基线
14. `src/baselines/standard_finetune.py` — 标准微调基线
15. `src/baselines/augmentation_consistency.py` — 增强一致性基线
16. `src/experiments/ablation.py` — 消融实验运行器
17. `src/experiments/tracking.py` — W&B 实验跟踪
18. `src/experiments/report.py` — 结果聚合报告
19. `scripts/run_stage1.sh` — Stage1 训练脚本
20. `scripts/run_stage2.sh` — Stage2 LoRA 训练脚本
21. `scripts/run_baselines.sh` — 基线运行脚本
22. `scripts/run_ablation.sh` — 消融实验脚本

### Definition of Done
- [ ] `pip install -r requirements.txt` 成功安装所有依赖
- [ ] `python src/training/train.py --config config/stage1.yaml` 可启动训练
- [ ] `python src/eval/retrieval.py --checkpoint <path>` 输出 Recall@1/5/10
- [ ] `python src/baselines/frozen_clip.py` 输出冻结CLIP基线结果
- [ ] `python src/experiments/ablation.py --config config/ablation.yaml` 运行消融实验
- [ ] 所有评估输出 JSON 格式结果到 `results/` 目录

### Must Have
- 切线对齐损失实现与 exp.md 公式一致：L_tan = 1 - cos(ΔI, ΔT)
- 三阶段训练（投影头 → LoRA → 全量微调）全部可运行
- 三个基线方法全部实现
- 标准检索 Recall@K 评估
- 消融实验覆盖 exp.md Section 9 的所有测试

### Must NOT Have (Guardrails)
- 不修改 `dataset/` 目录下的已有代码（数据集生成器）
- 不修改 `visualize/` 目录下的已有代码
- 不在损失函数中混用身份保留扰动和身份变更扰动（exp.md 关键规则）
- 不在没有显式标签的情况下跨因素混合训练
- 不将评估结果硬编码，必须从实际运行中产生

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: Tests after (research code, focus on functional correctness)
- **Framework**: pytest (will be included in requirements.txt)
- **Agent-Executed QA**: Every task includes CLI-based QA scenarios

### QA Policy
- **Training code**: Run with tiny config (2 epochs, 4 samples) → verify loss decreases
- **Loss functions**: Unit test with synthetic tensors → verify gradients flow
- **Evaluation**: Run on small subset → verify output JSON schema
- **Baselines**: Run with minimal config → verify results JSON produced
- Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation + independent modules, 8 parallel):
├── T1:  Project configuration + directory structure [quick]
├── T2:  Config system (YAML + CLI) [quick]
├── T3:  CLIP model wrapper [deep]
├── T4:  Dataset wrapper (PyTorch DataLoader) [unspecified-high]
├── T5:  Perturbation operator library [unspecified-high]
├── T6:  InfoNCE loss function [quick]
├── T7:  Tangent alignment loss function [quick]
└── T8:  Experiment logging utilities [quick]

Wave 2 (After Wave 1 — training + evaluation, 7 parallel):
├── T9:  Training pipeline (Stage 1: projection heads) [deep]
├── T10: Standard retrieval evaluator [unspecified-high]
├── T11: Compositional evaluator [unspecified-high]
├── T12: Perturbation consistency evaluator [unspecified-high]
├── T13: Baseline: frozen CLIP [unspecified-high]
├── T14: Baseline: standard fine-tune [unspecified-high]
└── T15: Baseline: augmentation consistency [unspecified-high]

Wave 3 (After Wave 2 — advanced features, 5 parallel):
├── T16: LoRA adapter (Stage 2 training) [deep]
├── T17: Ablation experiment runner [deep]
├── T18: Training run scripts + experiment configs [quick]
├── T19: Full evaluation pipeline integration [unspecified-high]
└── T20: Experiment tracking (W&B) [quick]

Wave 4 (After Wave 3 — integration + polish, 4 parallel):
├── T21: Stage 3 full fine-tune support [deep]
├── T22: Result aggregation and reporting [unspecified-high]
├── T23: Training curve visualization [visual-engineering]
└── T24: README + experiment documentation [writing]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── F1: Plan compliance audit [oracle]
├── F2: Code quality review [unspecified-high]
├── F3: Integration testing [unspecified-high]
└── F4: Scope fidelity check [deep]
-> Present results -> Get explicit user okay

Critical Path: T1 → T4 → T9 → T16 → T21 → F1-F4 → user okay
Parallel Speedup: ~75% faster than sequential
Max Concurrent: 8 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| T1   | —         | T2-T8  |
| T2   | T1        | T9,T18 |
| T3   | T1        | T9,T13-T15 |
| T4   | T1        | T9,T10-T12 |
| T5   | T1        | T9,T17 |
| T6   | T1        | T9,T14 |
| T7   | T1        | T9,T15 |
| T8   | T1        | T20,T22 |
| T9   | T2,T3,T4,T5,T6,T7 | T16,T19 |
| T10  | T3,T4     | T19,T22 |
| T11  | T3,T4     | T19,T22 |
| T12  | T3,T4,T5  | T19,T22 |
| T13  | T3,T4     | T17,T22 |
| T14  | T3,T4,T6  | T17,T22 |
| T15  | T3,T4,T7  | T17,T22 |
| T16  | T9,T3     | T21 |
| T17  | T9,T13,T14,T15 | T22 |
| T18  | T2        | — |
| T19  | T9,T10,T11,T12 | T21,T22 |
| T20  | T8        | — |
| T21  | T16,T19   | T22 |
| T22  | T10-T15,T17,T19,T21 | — |
| T23  | T20,T9    | — |
| T24  | T22       | — |

### Agent Dispatch Summary

- **Wave 1**: 8 tasks — T1-T2 → `quick`, T3 → `deep`, T4-T5 → `unspecified-high`, T6-T7 → `quick`, T8 → `quick`
- **Wave 2**: 7 tasks — T9 → `deep`, T10-T12 → `unspecified-high`, T13-T15 → `unspecified-high`
- **Wave 3**: 5 tasks — T16-T17 → `deep`, T18 → `quick`, T19 → `unspecified-high`, T20 → `quick`
- **Wave 4**: 4 tasks — T21 → `deep`, T22 → `unspecified-high`, T23 → `visual-engineering`, T24 → `writing`
- **FINAL**: 4 — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

---

### Wave 1: Foundation + Independent Modules

- [ ] 1. Project Configuration + Directory Structure

  **What to do**:
  - Create `requirements.txt` with all Python dependencies:
    ```
    torch>=2.0
    torchvision>=0.15
    transformers>=4.35
    pillow>=10.0
    numpy>=1.24
    matplotlib>=3.7
    umap-learn>=0.5
    pyyaml>=6.0
    wandb>=0.16
    tqdm>=4.65
    scipy>=1.11
    einops>=0.7
    peft>=0.7   # LoRA support
    ```
  - Create directory structure:
    ```
    src/
      __init__.py
      models/
        __init__.py
        clip_wrapper.py      (T3)
      data/
        __init__.py
        controlled_dataset.py (T4)
        perturbation_ops.py   (T5)
      losses/
        __init__.py
        info_nce.py           (T6)
        tangent.py            (T7)
      training/
        __init__.py
        trainer.py            (T9)
        train.py              (T9)
      eval/
        __init__.py
        retrieval.py          (T10)
        compositional.py      (T11)
        consistency.py        (T12)
      baselines/
        __init__.py
        frozen_clip.py        (T13)
        standard_finetune.py  (T14)
        augmentation_consistency.py (T15)
      experiments/
        __init__.py
        ablation.py           (T17)
        tracking.py           (T20)
        report.py             (T22)
      utils/
        __init__.py
        logging.py
        checkpoint.py
    config/
      stage1.yaml
      stage2.yaml
      stage3.yaml
      eval.yaml
      ablation.yaml
      baselines.yaml
    scripts/
      run_stage1.sh
      run_stage2.sh
      run_baselines.sh
      run_ablation.sh
    results/
    ```
  - Create all `__init__.py` files
  - Create placeholder config YAML files with documented defaults

  **Must NOT do**:
  - Do not modify existing `dataset/` or `visualize/` directories
  - Do not add `docs/` directory content

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File/directory creation and dependency listing is straightforward
  - **Skills**: []
    - No specialized skills needed
  - **Skills Evaluated but Omitted**:
    - None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T2-T8)
  - **Blocks**: T2, T3, T4, T5, T6, T7, T8
  - **Blocked By**: None (can start immediately)

  **References**:
  - `visualize/scripts/extract_embeddings.py` — Current import patterns (torch, transformers, PIL, numpy) to match dependency versions
  - `exp.md:36` — Training stages define config structure
  - `exp.md:72-76` — Training schedule informs stage1/2/3 config defaults

  **Acceptance Criteria**:
  - [ ] `requirements.txt` exists with all 12 dependencies
  - [ ] All directories and `__init__.py` files created
  - [ ] `config/stage1.yaml` has sections: model, data, training, loss
  - [ ] `python -c "import yaml; yaml.safe_load(open('config/stage1.yaml'))"` succeeds

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Install dependencies from requirements.txt
    Tool: Bash (pip)
    Preconditions: Fresh Python environment
    Steps:
      1. Run `pip install -r requirements.txt 2>&1 | tail -5`
      2. Verify all packages installed without errors
    Expected Result: All 12 packages installed successfully
    Failure Indicators: Any package fails to install
    Evidence: .sisyphus/evidence/task-1-pip-install.log

  Scenario: Directory structure is correct
    Tool: Bash (ls)
    Preconditions: None
    Steps:
      1. Run `find src/ -type d | sort`
      2. Run `find src/ -name "__init__.py" | wc -l`
      3. Verify all expected directories exist
    Expected Result: 9 directories under src/, 9 __init__.py files
    Failure Indicators: Missing directories or __init__.py files
    Evidence: .sisyphus/evidence/task-1-structure.txt

  Scenario: Config YAML is valid
    Tool: Bash (python)
    Preconditions: requirements.txt installed
    Steps:
      1. Run `python -c "import yaml; c=yaml.safe_load(open('config/stage1.yaml')); print(list(c.keys()))"`
      2. Verify output contains model, data, training, loss keys
    Expected Result: YAML parses successfully with expected top-level keys
    Failure Indicators: YAML parse error or missing keys
    Evidence: .sisyphus/evidence/task-1-config.txt
  ```

  **Commit**: YES
  - Message: `chore: project structure, dependencies, and config system`
  - Files: requirements.txt, src/, config/, scripts/
  - Pre-commit: `python -c "import yaml; yaml.safe_load(open('config/stage1.yaml'))"`

---

- [ ] 2. Config System (YAML + CLI Override)

  **What to do**:
  - Create `src/utils/config.py` with config loading:
    - `load_config(config_path: str) -> dict` — Load YAML, merge with CLI overrides
    - Support dot-notation CLI overrides: `--override training.lr=0.001`
    - Validate required fields exist for each config type
  - Create `src/utils/cli.py` with argument parser:
    - `--config` (required): path to YAML config
    - `--override` (repeatable): `key.subkey=value` pairs
    - `--output_dir` (optional): override results output path
  - Write config schemas:
    - `stage1.yaml`: model (clip_path, trainable_parts), data (jsonl_path, batch_size, num_workers), training (lr, epochs, lambda_tan, lambda_reg), loss (info_nce_temperature), output_dir
    - `eval.yaml`: model (checkpoint_path), data (jsonl_path, batch_size), eval (recall_k, factors)
    - `ablation.yaml`: base_config, ablations (list of overrides to apply)

  **Must NOT do**:
  - Do not use hydra or other heavy config frameworks (keep it simple with PyYAML)
  - Do not hardcode any paths in config loading code

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Config loading is a utility task with well-known patterns
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T3-T8)
  - **Blocks**: T9, T18
  - **Blocked By**: T1

  **References**:
  - `exp.md:29-31` — Loss hyperparameters (lambda_tan, lambda_reg) that must be configurable
  - `exp.md:72-76` — Training stages define which params are trainable per stage
  - `exp.md:93-99` — Ablation experiments define override patterns (lambda_tan=0, etc.)

  **Acceptance Criteria**:
  - [ ] `load_config("config/stage1.yaml")` returns nested dict
  - [ ] `--override training.lr=0.001` correctly overrides nested value
  - [ ] Missing required keys raise clear error messages
  - [ ] `config/ablation.yaml` defines all ablation variants from exp.md Section 9

  **QA Scenarios**:

  ```
  Scenario: Load and override config
    Tool: Bash (python)
    Preconditions: T1 complete, config files exist
    Steps:
      1. Run `python -c "from src.utils.config import load_config; c=load_config('config/stage1.yaml'); print(c['training']['lr'])"`
      2. Verify output is a float value
      3. Run override test: `python -c "from src.utils.config import load_config; c=load_config('config/stage1.yaml', overrides={'training.lr': 0.001}); print(c['training']['lr'])"`
    Expected Result: Config loads, override works, value changes to 0.001
    Failure Indicators: ImportError, KeyError, override doesn't apply
    Evidence: .sisyphus/evidence/task-2-config-load.txt

  Scenario: Invalid config raises error
    Tool: Bash (python)
    Preconditions: T1 complete
    Steps:
      1. Create temp file missing required key
      2. Attempt to load it
      3. Verify ValueError raised with helpful message
    Expected Result: Clear error about missing required key
    Failure Indicators: Silent failure or cryptic error
    Evidence: .sisyphus/evidence/task-2-config-error.txt
  ```

  **Commit**: YES
  - Message: `feat: YAML config system with CLI override support`
  - Files: src/utils/config.py, src/utils/cli.py
  - Pre-commit: `python -c "from src.utils.config import load_config; load_config('config/stage1.yaml')"`

---

- [ ] 3. CLIP Model Wrapper

  **What to do**:
  - Create `src/models/clip_wrapper.py` with `TangentCLIPWrapper(nn.Module)`:
    - Load pretrained CLIP via `transformers.CLIPModel.from_pretrained(model_path)`
    - Expose `encode_image(pixel_values)` → L2-normalized embeddings
    - Expose `encode_text(input_ids, attention_mask)` → L2-normalized embeddings
    - Support 3 training modes via `trainable_parts` parameter:
      - `"projection"`: Only projection heads + final layer norms trainable (Stage 1)
      - `"lora"`: Projection heads + LoRA adapters on attention layers (Stage 2)
      - `"full"`: All parameters trainable (Stage 3)
    - Projection head training: freeze all CLIP params, unfreeze `visual_projection`, `text_projection`, `visual_model.final_layer_norm`, `text_model.final_layer_norm`
    - LoRA integration: use `peft` library to inject LoRA into vision_model and text_model attention layers
    - Provide `save_checkpoint(path)` and `load_checkpoint(path)` methods
    - Provide `get_trainable_params()` → count for logging
  - Create `src/models/__init__.py` with `from .clip_wrapper import TangentCLIPWrapper`

  **Must NOT do**:
  - Do not modify the pretrained CLIP architecture
  - Do not add custom attention mechanisms (keep CLIP backbone unchanged)
  - Do not implement mixed precision in the wrapper (trainer handles that)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: CLIP model integration requires understanding of HuggingFace CLIP internals, projection head structure, and LoRA injection points
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T4-T8)
  - **Blocks**: T9, T13, T14, T15, T16
  - **Blocked By**: T1

  **References**:
  - `visualize/scripts/extract_embeddings.py:25-69` — CLIP loading and embedding extraction pattern to follow (vision_model → CLS → visual_projection, text_model → EOS → text_projection)
  - `exp.md:26-27` — Representation notation: zI = fI(x), zT = fT(y)
  - `exp.md:32-36` — Training stages define which parts are trainable
  - `exp.md:35` — Stage 2 mentions LoRA

  **Acceptance Criteria**:
  - [ ] `TangentCLIPWrapper` loads pretrained CLIP without errors
  - [ ] `encode_image()` returns L2-normalized tensor of shape (batch, embed_dim)
  - [ ] `encode_text()` returns L2-normalized tensor of shape (batch, embed_dim)
  - [ ] `trainable_parts="projection"` freezes 95%+ of parameters
  - [ ] `trainable_parts="lora"` adds LoRA adapters via peft
  - [ ] `save_checkpoint()` / `load_checkpoint()` round-trip works

  **QA Scenarios**:

  ```
  Scenario: Load CLIP and extract embeddings
    Tool: Bash (python)
    Preconditions: CLIP model at /root/autodl-tmp/model/clip_vit_b
    Steps:
      1. Run: `python -c "from src.models.clip_wrapper import TangentCLIPWrapper; m=TangentCLIPWrapper('/root/autodl-tmp/model/clip_vit_b'); print('Model loaded OK')"`
      2. Create dummy input: batch of 2 random images (224x224)
      3. Run encode_image, verify shape (2, 512) and L2 norm ≈ 1.0
      4. Run encode_text with sample captions, verify shape and norm
    Expected Result: Embeddings are (batch, 512), L2-normalized
    Failure Indicators: Shape mismatch, norm != 1.0, import error
    Evidence: .sisyphus/evidence/task-3-embeddings.txt

  Scenario: Projection-only training mode
    Tool: Bash (python)
    Preconditions: Model loaded
    Steps:
      1. Create wrapper with trainable_parts="projection"
      2. Count trainable vs total parameters
      3. Verify trainable < 5% of total
    Expected Result: Only projection heads and layer norms trainable
    Failure Indicators: Too many or too few trainable params
    Evidence: .sisyphus/evidence/task-3-projection-params.txt

  Scenario: LoRA training mode
    Tool: Bash (python)
    Preconditions: peft installed
    Steps:
      1. Create wrapper with trainable_parts="lora"
      2. Verify LoRA modules injected
      3. Count trainable params, verify between projection-only and full
    Expected Result: LoRA adapters present, param count in expected range
    Failure Indicators: LoRA injection fails, param count wrong
    Evidence: .sisyphus/evidence/task-3-lora-params.txt
  ```

  **Commit**: YES
  - Message: `feat: CLIP model wrapper with projection/LoRA/full training modes`
  - Files: src/models/clip_wrapper.py, src/models/__init__.py
  - Pre-commit: `python -c "from src.models.clip_wrapper import TangentCLIPWrapper"`

---

- [ ] 4. Dataset Wrapper (PyTorch DataLoader)

  **What to do**:
  - Create `src/data/controlled_dataset.py` with `ControlledPairDataset(torch.utils.data.Dataset)`:
    - Load JSONL file (train.jsonl / test.jsonl) produced by `build_controlled_dataset.py`
    - Each item returns dict with:
      - `image`: PIL Image loaded from `image` path
      - `image_plus`: PIL Image loaded from `image_plus` path
      - `caption`: str (original caption)
      - `caption_plus`: str (perturbed caption)
      - `factor`: str (e.g., "relation", "color", "shape")
      - `changed_field`: str
      - `pair_id`: str
    - Support filtering by factor: `dataset.filter_by_factor("relation")`
    - Support train/test split from same JSONL by `split` field
    - Create `create_dataloader(dataset, batch_size, num_workers, shuffle)` utility
    - Image preprocessing: use CLIP processor's image transforms (224x224, normalize with CLIP stats)
  - Create `src/data/text_processor.py`:
    - Wrap `CLIPProcessor` for text tokenization
    - `tokenize_captions(captions: list[str]) -> dict` → input_ids, attention_mask

  **Must NOT do**:
  - Do not hardcode dataset paths (pass via config)
  - Do not apply training-time augmentation in the dataset (that's T5's job)
  - Do not load all images into memory (lazy loading per __getitem__)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Dataset implementation requires understanding of JSONL format, image loading, and CLIP preprocessing pipeline
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T3, T5-T8)
  - **Blocks**: T9, T10, T11, T12
  - **Blocked By**: T1

  **References**:
  - `dataset/build_controlled_dataset.py:631-647` — JSONL record format (pair_id, image, image_plus, caption, caption_plus, factor, changed_field, meta)
  - `visualize/scripts/extract_embeddings.py:32-42` — CLIP image preprocessing (Resize 224, Normalize with CLIP mean/std)
  - `visualize/scripts/extract_embeddings.py:57-69` — CLIP text tokenization pattern
  - `exp.md:62-69` — Training flow steps 1-2: prepare paired mini-batch, sample matching perturbation

  **Acceptance Criteria**:
  - [ ] Dataset loads from JSONL without errors
  - [ ] `len(dataset)` matches JSONL line count
  - [ ] `dataset[0]` returns all required keys
  - [ ] `filter_by_factor("relation")` returns only relation samples
  - [ ] DataLoader produces batches with correct shapes
  - [ ] Text tokenizer produces input_ids of shape (batch, 77)

  **QA Scenarios**:

  ```
  Scenario: Load dataset and iterate
    Tool: Bash (python)
    Preconditions: JSONL file exists at test path
    Steps:
      1. Create small test JSONL with 10 records
      2. Load dataset: `ds = ControlledPairDataset("test.jsonl", image_root="/tmp")`
      3. Print len(ds), should be 10
      4. Iterate one batch through DataLoader
    Expected Result: Dataset loads, length correct, batch shapes correct
    Failure Indicators: FileNotFoundError, KeyError, shape mismatch
    Evidence: .sisyphus/evidence/task-4-dataset-load.txt

  Scenario: Filter by factor
    Tool: Bash (python)
    Preconditions: Dataset loaded with mixed factors
    Steps:
      1. Create test JSONL with 5 "relation" + 5 "color" records
      2. Filter: `rel_ds = ds.filter_by_factor("relation")`
      3. Verify len == 5 and all factors are "relation"
    Expected Result: Filtered dataset has correct subset
    Failure Indicators: Wrong count, wrong factor values
    Evidence: .sisyphus/evidence/task-4-filter.txt
  ```

  **Commit**: YES
  - Message: `feat: PyTorch dataset wrapper for controlled pair JSONL`
  - Files: src/data/controlled_dataset.py, src/data/text_processor.py, src/data/__init__.py
  - Pre-commit: `python -c "from src.data.controlled_dataset import ControlledPairDataset"`

---

- [ ] 5. Perturbation Operator Library

  **What to do**:
  - Create `src/data/perturbation_ops.py` with perturbation operators:
    - **Text-side operators** (exp.md Section 5):
      - `paraphrase(text) -> str`: Semantic-preserving rewording (use a paraphrase model or template-based)
      - `synonym_swap(text) -> str`: Replace adjectives/nouns with synonyms
      - `adjective_replace(text, old_adj, new_adj) -> str`: Controlled attribute change
      - `relation_replace(text, old_rel, new_rel) -> str`: Controlled relation change (e.g., "left of" → "right of")
      - `noun_replace(text, old_noun, new_noun) -> str`: Controlled object change
    - **Image-side operators** (identity-preserving, for training-time augmentation):
      - `color_jitter(image, strength=0.2) -> PIL.Image`: Mild color perturbation
      - `random_crop_resize(image, scale=(0.8, 1.0)) -> PIL.Image`: High-IoU crop
      - `gaussian_blur(image, sigma=1.0) -> PIL.Image`: Mild blur
      - `background_perturb(image, strength=0.3) -> PIL.Image`: Background-only perturbation
    - **Perturbation pair sampler**:
      - `sample_perturbation_pair(x, y, factor) -> (x_plus, y_plus)`: Sample matched perturbation pair preserving semantic correspondence
    - Register all operators in `PERTURBATION_REGISTRY` dict
    - Separate identity-preserving from identity-changing operators (exp.md critical rule)

  **Must NOT do**:
  - Do not mix identity-preserving and identity-changing perturbations in same loss (exp.md rule line 60)
  - Do not use heavy augmentations that change object identity
  - Do not implement 3D scene rendering (existing blender code handles that)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding of NLP perturbation techniques and image augmentation with semantic constraints
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T4, T6-T8)
  - **Blocks**: T9, T17
  - **Blocked By**: T1

  **References**:
  - `exp.md:45-60` — Complete perturbation operator table (factors, image/text operators, identity preservation rules)
  - `exp.md:54-58` — Recommended perturbation library by modality
  - `exp.md:60` — Critical rule: never mix identity-preserving and identity-changing in same loss
  - `dataset/build_controlled_dataset.py:109-119` — Caption template pattern to inform text perturbation

  **Acceptance Criteria**:
  - [ ] All 5 text operators implemented and produce valid strings
  - [ ] All 4 image operators implemented and produce valid PIL Images
  - [ ] `sample_perturbation_pair()` returns matched (x_plus, y_plus) tuple
  - [ ] Operators are registered in PERTURBATION_REGISTRY
  - [ ] Identity-preserving and identity-changing operators have separate flags

  **QA Scenarios**:

  ```
  Scenario: Text perturbation operators
    Tool: Bash (python)
    Preconditions: None
    Steps:
      1. Test each text operator with sample input
      2. Verify output differs from input but is valid English
      3. Verify identity-preserving ops (paraphrase) don't change semantics
    Expected Result: All operators produce valid perturbations
    Failure Indicators: Identical output, empty string, crash
    Evidence: .sisyphus/evidence/task-5-text-ops.txt

  Scenario: Image perturbation operators
    Tool: Bash (python)
    Preconditions: PIL installed
    Steps:
      1. Create dummy 224x224 RGB image
      2. Apply each image operator
      3. Verify output is valid PIL Image with same dimensions
    Expected Result: All operators produce valid images
    Failure Indicators: Wrong type, wrong dimensions, crash
    Evidence: .sisyphus/evidence/task-5-image-ops.txt

  Scenario: Identity preservation separation
    Tool: Bash (python)
    Preconditions: Operators implemented
    Steps:
      1. List all operators and their identity_preserving flag
      2. Verify no operator appears in both categories
      3. Verify PERTURBATION_REGISTRY has separate keys
    Expected Result: Clean separation of operator types
    Failure Indicators: Mixed categories
    Evidence: .sisyphus/evidence/task-5-registry.txt
  ```

  **Commit**: YES
  - Message: `feat: perturbation operator library (text + image, identity-preserving separation)`
  - Files: src/data/perturbation_ops.py
  - Pre-commit: `python -c "from src.data.perturbation_ops import PERTURBATION_REGISTRY"`

---

- [ ] 6. InfoNCE Loss Function

  **What to do**:
  - Create `src/losses/info_nce.py` with `InfoNCELoss(nn.Module)`:
    - Standard bidirectional InfoNCE: L_clip = (L_I→T + L_T→I) / 2
    - L_I→T = -log(exp(sim(zI_i, zT_i) / τ) / Σ_j exp(sim(zI_i, zT_j) / τ))
    - Temperature τ configurable (default 0.07)
    - Input: image_embeddings (B, D), text_embeddings (B, D)
    - Output: scalar loss
    - Use cosine similarity matrix computation
    - Support in-batch negatives (standard CLIP approach)
  - Provide `compute_similarity_matrix(zI, zT) -> (B, B)` utility

  **Must NOT do**:
  - Do not add label smoothing or other modifications to standard InfoNCE
  - Do not compute gradients unnecessarily in similarity matrix utility

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Well-known loss function with standard implementation
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T5, T7-T8)
  - **Blocks**: T9, T14
  - **Blocked By**: T1

  **References**:
  - `exp.md:27` — L_clip = (L_I→T + L_T→I) / 2
  - `exp.md:30` — Total loss: L = L_clip + λ_tan * L_tan + λ_reg * L_reg

  **Acceptance Criteria**:
  - [ ] Loss returns scalar tensor
  - [ ] Loss is differentiable (gradients flow)
  - [ ] Symmetric: swap I/T and loss is the same
  - [ ] Temperature parameter configurable

  **QA Scenarios**:

  ```
  Scenario: Basic InfoNCE computation
    Tool: Bash (python)
    Preconditions: torch installed
    Steps:
      1. Create random embeddings (4, 512), L2-normalize
      2. Compute loss: `loss = InfoNCELoss()(zI, zT)`
      3. Verify loss is scalar, positive, requires_grad
    Expected Result: Valid scalar loss with gradient
    Failure Indicators: Wrong shape, no gradient, negative loss
    Evidence: .sisyphus/evidence/task-6-infonce.txt

  Scenario: Symmetry check
    Tool: Bash (python)
    Preconditions: Loss implemented
    Steps:
      1. Compute loss(I, T) and loss(T, I)
      2. Verify they are approximately equal
    Expected Result: Bidirectional loss is symmetric
    Failure Indicators: Significant difference
    Evidence: .sisyphus/evidence/task-6-symmetry.txt
  ```

  **Commit**: YES (grouped with T7)
  - Message: `feat: InfoNCE bidirectional retrieval loss`
  - Files: src/losses/info_nce.py
  - Pre-commit: `python -c "from src.losses.info_nce import InfoNCELoss"`

---

- [ ] 7. Tangent Alignment Loss Function

  **What to do**:
  - Create `src/losses/tangent.py` with `TangentAlignmentLoss(nn.Module)`:
    - Compute difference vectors:
      - ΔI = fI(x⁺) - fI(x)  (image perturbation difference)
      - ΔT = fT(y⁺) - fT(y)  (text perturbation difference)
    - Tangent loss: L_tan = 1 - cos(ΔI, ΔT), averaged over batch
    - Optional regularization: L_reg to avoid degenerate large-difference solutions
    - Input: zI, zI_plus, zT, zT_plus (all B×D L2-normalized)
    - Output: scalar loss (L_tan), optional L_reg
    - Provide `compute_tangent_vectors(zI, zI_plus, zT, zT_plus) -> (delta_I, delta_T)` utility
  - Create `src/losses/combined.py` with `CombinedLoss(nn.Module)`:
    - L = L_clip + λ_tan * L_tan + λ_reg * L_reg
    - λ_tan and λ_reg configurable
    - Return dict with individual loss components for logging

  **Must NOT do**:
  - Do not normalize ΔI, ΔT before cosine (they carry magnitude information)
  - Do not clamp cosine similarity (let it be in [-1, 1])

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct implementation of the formula in exp.md
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T6, T8)
  - **Blocks**: T9, T15
  - **Blocked By**: T1

  **References**:
  - `exp.md:28-29` — ΔI = fI(x⁺) - fI(x), ΔT = fT(y⁺) - fT(y), L_tan = 1 - cos(ΔI, ΔT)
  - `exp.md:30` — Total loss: L = L_clip + λ_tan * L_tan + λ_reg * L_reg
  - `exp.md:37` — Optional weak regularizer on Δ norm

  **Acceptance Criteria**:
  - [ ] Tangent loss returns scalar in [0, 2] range (1 - cos)
  - [ ] Gradient flows through both zI_plus and zT_plus
  - [ ] When ΔI ≈ ΔT, loss approaches 0
  - [ ] When ΔI ⊥ ΔT, loss ≈ 1.0
  - [ ] CombinedLoss returns dict with all components

  **QA Scenarios**:

  ```
  Scenario: Perfect alignment (loss → 0)
    Tool: Bash (python)
    Preconditions: torch installed
    Steps:
      1. Create zI, zT, then zI_plus = zI + delta, zT_plus = zT + delta (same delta)
      2. Compute loss
      3. Verify loss ≈ 0.0
    Expected Result: Identical tangent directions → loss ≈ 0
    Failure Indicators: Loss not near 0
    Evidence: .sisyphus/evidence/task-7-perfect.txt

  Scenario: Orthogonal alignment (loss → 1)
    Tool: Bash (python)
    Preconditions: torch installed
    Steps:
      1. Create orthogonal delta_I and delta_T
      2. Compute loss
      3. Verify loss ≈ 1.0
    Expected Result: Orthogonal tangent directions → loss ≈ 1
    Failure Indicators: Loss not near 1
    Evidence: .sisyphus/evidence/task-7-orthogonal.txt

  Scenario: Gradient flow
    Tool: Bash (python)
    Preconditions: Loss implemented
    Steps:
      1. Create embeddings with requires_grad=True
      2. Compute loss, call backward()
      3. Verify all 4 input embeddings have gradients
    Expected Result: Gradients flow to all inputs
    Failure Indicators: Missing gradients on any input
    Evidence: .sisyphus/evidence/task-7-gradient.txt
  ```

  **Commit**: YES
  - Message: `feat: tangent alignment loss + combined loss wrapper`
  - Files: src/losses/tangent.py, src/losses/combined.py
  - Pre-commit: `python -c "from src.losses.tangent import TangentAlignmentLoss; from src.losses.combined import CombinedLoss"`

---

- [ ] 8. Experiment Logging Utilities

  **What to do**:
  - Create `src/utils/logging.py`:
    - `setup_logger(name, log_dir) -> logger` — File + console logging
    - `log_metrics(metrics: dict, step: int, logger)` — Formatted metric logging
    - JSON Lines logging to `results/{experiment}/metrics.jsonl`
  - Create `src/utils/checkpoint.py`:
    - `save_checkpoint(model, optimizer, epoch, metrics, path)` — Save training state
    - `load_checkpoint(path) -> dict` — Load training state
    - `find_best_checkpoint(checkpoint_dir, metric_name) -> path` — Find best by metric
  - Create `src/utils/seed.py`:
    - `set_seed(seed: int)` — Set random, numpy, torch, CUDA seeds for reproducibility

  **Must NOT do**:
  - Do not import wandb in logging (T20 handles W&B)
  - Do not create TensorBoard integration (keep simple JSON logging)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard utility functions
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T7)
  - **Blocks**: T20, T22
  - **Blocked By**: T1

  **References**:
  - `exp.md:69` — Step 6: monitor standard retrieval and factor-specific validation each epoch
  - `exp.md:77-83` — Checkpoint save/load for different training stages

  **Acceptance Criteria**:
  - [ ] Logger writes to both console and file
  - [ ] `save_checkpoint` creates .pt file with model state
  - [ ] `load_checkpoint` restores all training state
  - [ ] `set_seed` produces deterministic results

  **QA Scenarios**:

  ```
  Scenario: Checkpoint round-trip
    Tool: Bash (python)
    Preconditions: torch installed
    Steps:
      1. Create simple model, save checkpoint
      2. Create new model instance, load checkpoint
      3. Verify state dicts match
    Expected Result: Checkpoint save/load preserves model state
    Failure Indicators: State dict mismatch
    Evidence: .sisyphus/evidence/task-8-checkpoint.txt

  Scenario: Seed determinism
    Tool: Bash (python)
    Preconditions: torch installed
    Steps:
      1. set_seed(42), generate random tensor A
      2. set_seed(42), generate random tensor B
      3. Verify A == B
    Expected Result: Same seed produces same random state
    Failure Indicators: A != B
    Evidence: .sisyphus/evidence/task-8-seed.txt
  ```

  **Commit**: YES (grouped with T2)
  - Message: `chore: logging, checkpoint, and seed utilities`
  - Files: src/utils/logging.py, src/utils/checkpoint.py, src/utils/seed.py
  - Pre-commit: `python -c "from src.utils.checkpoint import save_checkpoint; from src.utils.seed import set_seed"`

---

### Wave 2: Training Pipeline + Evaluators + Baselines

- [ ] 9. Training Pipeline (Stage 1: Projection Heads)

  **What to do**:
  - Create `src/training/trainer.py` with `TangentTrainer` class:
    - `__init__(config, model, train_dataset, val_dataset)` — Setup optimizer, scheduler, loss
    - Optimizer: AdamW with configurable lr (default 1e-4 for projection-only)
    - Scheduler: Cosine annealing with warmup
    - Mixed precision: torch.cuda.amp (GradScaler + autocast)
    - `train_epoch(epoch) -> dict` — Single training epoch:
      1. Load batch: (x, x⁺, y, y⁺, factor_labels)
      2. Forward: zI = model.encode_image(x), zI⁺ = model.encode_image(x⁺), etc.
      3. Compute losses: combined_loss(zI, zI⁺, zT, zT⁺)
      4. Backward with AMP
      5. Gradient clipping
      6. Log per-factor losses
    - `validate() -> dict` — Run all evaluators on validation set
    - `train(num_epochs) -> list[dict]` — Full training loop with:
      - Epoch-level logging
      - Checkpoint saving (best by val metric)
      - Early stopping (optional)
    - Factor-aware batching: optionally sample balanced batches across factors
  - Create `src/training/train.py` — CLI entry point:
    - Parse config + CLI overrides
    - Load model, dataset, create trainer
    - Run training
    - Save final results JSON

  **Must NOT do**:
  - Do not implement Stage 2 (LoRA) or Stage 3 (full fine-tune) here (T16, T21)
  - Do not hardcode hyperparameters (all from config)
  - Do not skip gradient clipping (must be included)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Training loop requires careful handling of AMP, gradient clipping, factor-aware batching, and multi-loss coordination
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Wave 1)
  - **Blocks**: T16, T19
  - **Blocked By**: T2, T3, T4, T5, T6, T7

  **References**:
  - `exp.md:62-69` — Training flow steps 1-6 (batch prep, perturbation, embedding, loss, regularization, monitoring)
  - `exp.md:70-76` — Training schedule: Stage 1 projection-only, short run
  - `exp.md:30` — Total loss formula: L = L_clip + λ_tan * L_tan + λ_reg * L_reg
  - `config/stage1.yaml` — Config structure from T2

  **Acceptance Criteria**:
  - [ ] `python src/training/train.py --config config/stage1.yaml` starts training
  - [ ] Training loss decreases over epochs (verified with tiny config)
  - [ ] Checkpoints saved to `results/checkpoints/`
  - [ ] Metrics logged to `results/metrics.jsonl`
  - [ ] AMP works on GPU (falls back to FP32 on CPU)

  **QA Scenarios**:

  ```
  Scenario: Short training run (2 epochs, 4 samples)
    Tool: Bash (python)
    Preconditions: T1-T8 complete, small test data
    Steps:
      1. Create tiny dataset with 8 samples
      2. Run: `python src/training/train.py --config config/stage1.yaml --override training.max_epochs=2 data.batch_size=4`
      3. Verify loss printed for each epoch
      4. Verify checkpoint saved
      5. Verify metrics.jsonl has entries
    Expected Result: Training completes, loss decreases, artifacts saved
    Failure Indicators: Training crashes, loss NaN, no checkpoints
    Evidence: .sisyphus/evidence/task-9-training.txt

  Scenario: Factor-aware logging
    Tool: Bash (python)
    Preconditions: Training runs
    Steps:
      1. Run 1 epoch with mixed factors
      2. Check metrics.jsonl has per-factor loss breakdown
    Expected Result: Per-factor losses logged separately
    Failure Indicators: Only aggregate loss, no per-factor breakdown
    Evidence: .sisyphus/evidence/task-9-factor-log.txt
  ```

  **Commit**: YES
  - Message: `feat: Stage 1 training pipeline with AMP and factor-aware logging`
  - Files: src/training/trainer.py, src/training/train.py
  - Pre-commit: `python -c "from src.training.trainer import TangentTrainer"`

---

- [ ] 10. Standard Retrieval Evaluator (Recall@K)

  **What to do**:
  - Create `src/eval/retrieval.py` with `RetrievalEvaluator` class:
    - `evaluate(model, dataset) -> dict`:
      - Extract all image and text embeddings (batched, no_grad)
      - Compute similarity matrix (N_images × N_texts)
      - Image→Text Recall@1/5/10
      - Text→Image Recall@1/5/10
      - Per-factor breakdown: Recall for each perturbation factor separately
      - Median rank
    - Return dict: `{"I2T": {"R@1": ..., "R@5": ..., "R@10": ...}, "T2I": {...}, "per_factor": {...}}`
    - Support evaluation on subset (for speed during training)
  - Provide standalone CLI: `python src/eval/retrieval.py --checkpoint <path> --config config/eval.yaml`

  **Must NOT do**:
  - Do not include perturbation-specific metrics (that's T12)
  - Do not use batch size 1 for evaluation (use large batches for speed)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Retrieval evaluation requires efficient similarity computation and recall calculation across large sets
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T9, T11-T15)
  - **Blocks**: T19, T22
  - **Blocked By**: T3, T4

  **References**:
  - `exp.md:88` — Standard retrieval: I→T and T→I Recall@1/5/10
  - `exp.md:101-106` — Expected results: frozen CLIP < standard fine-tune on in-domain retrieval
  - `visualize/scripts/extract_embeddings.py:46-69` — Embedding extraction pattern to reuse

  **Acceptance Criteria**:
  - [ ] Evaluator computes Recall@1/5/10 for both directions
  - [ ] Per-factor breakdown included in output
  - [ ] Output saved as JSON to results directory
  - [ ] CLI entry point works standalone

  **QA Scenarios**:

  ```
  Scenario: Evaluate on small dataset
    Tool: Bash (python)
    Preconditions: Model wrapper + dataset working
    Steps:
      1. Create 20-sample dataset
      2. Run evaluator with pretrained CLIP
      3. Verify JSON output with R@1/5/10 for both directions
      4. Verify per_factor breakdown present
    Expected Result: Valid recall values in [0, 1], all factors present
    Failure Indicators: NaN, out-of-range, missing factors
    Evidence: .sisyphus/evidence/task-10-retrieval.json
  ```

  **Commit**: YES
  - Message: `feat: standard retrieval evaluator with per-factor Recall@K`
  - Files: src/eval/retrieval.py
  - Pre-commit: `python -c "from src.eval.retrieval import RetrievalEvaluator"`

---

- [ ] 11. Compositional Evaluator

  **What to do**:
  - Create `src/eval/compositional.py` with `CompositionalEvaluator` class:
    - `evaluate(model, dataset) -> dict`:
      - **Winoground-style scoring**: For each pair (x1, y1), (x2, y2) where captions are minimally different (e.g., "A red cube left of a blue sphere" vs "A blue cube left of a red sphere"):
        - Text score: sim(x1, y1) > sim(x1, y2) AND sim(x2, y2) > sim(x2, y1)
        - Image score: sim(x1, y1) > sim(x2, y1) AND sim(x2, y2) > sim(x1, y2)
        - Group score: both text AND image correct
      - **Relation accuracy**: For relation perturbation pairs, check if model ranks correct caption above perturbed caption
      - **Attribute accuracy**: For color/attribute pairs, check ranking
    - Return dict: `{"text_score": ..., "image_score": ..., "group_score": ..., "relation_acc": ..., "attribute_acc": ...}`
  - Auto-generate Winoground-style pairs from controlled dataset by pairing samples with same scene but different factors

  **Must NOT do**:
  - Do not assume Winoground dataset is available (generate pairs from controlled dataset)
  - Do not skip group score (most stringent metric)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Compositional evaluation requires careful pair construction and multiple scoring schemes
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T9-T10, T12-T15)
  - **Blocks**: T19, T22
  - **Blocked By**: T3, T4

  **References**:
  - `exp.md:89` — Winoground-style group scores, SugarCrepe subsets, relation accuracy
  - `exp.md:104` — Tangent fine-tuning should show more pronounced gains on relation-sensitive benchmarks
  - `dataset/build_controlled_dataset.py:362-415` — Hard negative generation pattern (related to compositional pairs)

  **Acceptance Criteria**:
  - [ ] Winoground-style text/image/group scores computed
  - [ ] Relation accuracy for relation perturbation pairs
  - [ ] Attribute accuracy for color perturbation pairs
  - [ ] Output JSON with all sub-metrics

  **QA Scenarios**:

  ```
  Scenario: Compositional scoring on synthetic pairs
    Tool: Bash (python)
    Preconditions: Model + dataset with relation + color factors
    Steps:
      1. Load dataset with mixed factors
      2. Run CompositionalEvaluator
      3. Verify text_score, image_score, group_score present
      4. Verify relation_acc and attribute_acc in [0, 1]
    Expected Result: All metrics computed, values in valid range
    Failure Indicators: Missing metrics, NaN, crash
    Evidence: .sisyphus/evidence/task-11-compositional.json
  ```

  **Commit**: YES
  - Message: `feat: compositional evaluator with Winoground-style scoring`
  - Files: src/eval/compositional.py
  - Pre-commit: `python -c "from src.eval.compositional import CompositionalEvaluator"`

---

- [ ] 12. Perturbation Consistency Evaluator

  **What to do**:
  - Create `src/eval/consistency.py` with `ConsistencyEvaluator` class:
    - `evaluate(model, dataset) -> dict`:
      - For each (x, x⁺, y, y⁺) perturbation pair:
        - Compute: sim(x, y), sim(x⁺, y), sim(x, y⁺), sim(x⁺, y⁺)
        - Consistency check: sim(x⁺, y⁺) > sim(x⁺, y) AND sim(x⁺, y⁺) > sim(x, y⁺)
        - Score: fraction of pairs where consistency holds
      - Per-factor consistency scores
      - Tangent alignment quality: compute actual ΔI, ΔT and their cosine similarity
    - Return dict: `{"overall_consistency": ..., "per_factor": {"relation": ..., "color": ...}, "mean_tangent_cosine": ...}`
    - This directly tests the first-order hypothesis from exp.md

  **Must NOT do**:
  - Do not confuse with augmentation consistency (that's T15 baseline)
  - Do not use this during training (only at evaluation time)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Custom evaluation metric requiring careful pair-wise similarity computation
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T9-T11, T13-T15)
  - **Blocks**: T19, T22
  - **Blocked By**: T3, T4, T5

  **References**:
  - `exp.md:90` — Perturbation consistency: correct score ordering under matched perturbations
  - `exp.md:102-106` — Expected: tangent fine-tune shows gains on controlled perturbation benchmarks
  - `exp.md:28-29` — ΔI, ΔT definitions for tangent alignment quality measurement

  **Acceptance Criteria**:
  - [ ] Overall consistency score computed
  - [ ] Per-factor consistency breakdown
  - [ ] Mean tangent cosine (measures how well tangent alignment works)
  - [ ] Output JSON with all metrics

  **QA Scenarios**:

  ```
  Scenario: Consistency evaluation
    Tool: Bash (python)
    Preconditions: Model + dataset
    Steps:
      1. Run evaluator on small dataset
      2. Verify consistency scores in [0, 1]
      3. Verify per_factor breakdown present
      4. Verify mean_tangent_cosine in [-1, 1]
    Expected Result: All metrics computed correctly
    Failure Indicators: Out-of-range values, missing metrics
    Evidence: .sisyphus/evidence/task-12-consistency.json
  ```

  **Commit**: YES
  - Message: `feat: perturbation consistency evaluator with tangent alignment quality`
  - Files: src/eval/consistency.py
  - Pre-commit: `python -c "from src.eval.consistency import ConsistencyEvaluator"`

---

- [ ] 13. Baseline: Frozen CLIP

  **What to do**:
  - Create `src/baselines/frozen_clip.py`:
    - Load pretrained CLIP, freeze all parameters
    - Run all 3 evaluators (retrieval, compositional, consistency) on test set
    - Output results to `results/baselines/frozen_clip.json`
    - This establishes the zero-shot starting point (exp.md: "reference zero-shot capabilities")
  - CLI: `python src/baselines/frozen_clip.py --config config/baselines.yaml`

  **Must NOT do**:
  - Do not fine-tune any parameters (everything frozen)
  - Do not skip any evaluator

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Needs to integrate model wrapper and all 3 evaluators
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T9-T12, T14-T15)
  - **Blocks**: T17, T22
  - **Blocked By**: T3, T4, T10, T11, T12

  **References**:
  - `exp.md:17` — Frozen CLIP: reference zero-shot capabilities, no task adaptation
  - `exp.md:102` — Expected: frozen CLIP < standard fine-tune on in-domain retrieval

  **Acceptance Criteria**:
  - [ ] Loads pretrained CLIP with all params frozen
  - [ ] Runs all 3 evaluators
  - [ ] Outputs JSON with retrieval, compositional, consistency metrics
  - [ ] CLI works standalone

  **QA Scenarios**:

  ```
  Scenario: Frozen CLIP baseline
    Tool: Bash (python)
    Preconditions: T10-T12 complete, test data available
    Steps:
      1. Run: `python src/baselines/frozen_clip.py --config config/baselines.yaml`
      2. Verify results/baselines/frozen_clip.json exists
      3. Verify JSON has retrieval, compositional, consistency sections
    Expected Result: Complete baseline results in JSON
    Failure Indicators: Missing sections, crash
    Evidence: .sisyphus/evidence/task-13-frozen.json
  ```

  **Commit**: YES
  - Message: `feat: frozen CLIP zero-shot baseline`
  - Files: src/baselines/frozen_clip.py
  - Pre-commit: `python -c "from src.baselines.frozen_clip import run_frozen_baseline"`

---

- [ ] 14. Baseline: Standard Fine-tune (No Tangent)

  **What to do**:
  - Create `src/baselines/standard_finetune.py`:
    - Same training pipeline as T9, but with λ_tan = 0 (only InfoNCE loss)
    - Use same training config (same data, same epochs, same optimizer)
    - Train, save checkpoint, run all 3 evaluators
    - Output results to `results/baselines/standard_finetune.json`
    - This is the PRIMARY comparison baseline (exp.md: "main baseline")
  - CLI: `python src/baselines/standard_finetune.py --config config/baselines.yaml`

  **Must NOT do**:
  - Do not use different hyperparameters than tangent training (must be identical except λ_tan=0)
  - Do not use different data augmentation

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Reuses training pipeline but with modified loss configuration
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T9-T13, T15)
  - **Blocks**: T17, T22
  - **Blocked By**: T3, T4, T6, T10, T11, T12

  **References**:
  - `exp.md:18` — Standard CLIP fine-tune: same data, same training plan, no tangent term
  - `exp.md:103` — Standard fine-tune and tangent fine-tune may be close on plain Recall@K

  **Acceptance Criteria**:
  - [ ] Trains with λ_tan=0, λ_reg=0 (pure InfoNCE)
  - [ ] Same hyperparameters as Stage 1 tangent training
  - [ ] Outputs JSON with all evaluator results
  - [ ] Checkpoint saved for comparison

  **QA Scenarios**:

  ```
  Scenario: Standard fine-tune baseline
    Tool: Bash (python)
    Preconditions: T9 working, T10-T12 complete
    Steps:
      1. Run: `python src/baselines/standard_finetune.py --config config/baselines.yaml`
      2. Verify training completes (loss decreases)
      3. Verify results/baselines/standard_finetune.json exists
      4. Compare R@1 with frozen_clip baseline (should be higher)
    Expected Result: Training completes, results better than frozen CLIP
    Failure Indicators: Training fails, results not better than frozen
    Evidence: .sisyphus/evidence/task-14-standard-ft.json
  ```

  **Commit**: YES
  - Message: `feat: standard CLIP fine-tune baseline (λ_tan=0)`
  - Files: src/baselines/standard_finetune.py
  - Pre-commit: `python -c "from src.baselines.standard_finetune import run_standard_baseline"`

---

- [ ] 15. Baseline: Augmentation Consistency

  **What to do**:
  - Create `src/baselines/augmentation_consistency.py`:
    - Same architecture as tangent training, but replace tangent loss with:
      - L_aug = 1 - (cos(fI(x), fI(x⁺)) + cos(fT(y), fT(y⁺))) / 2
      - This tests whether gains come from generic invariance vs cross-modal first-order matching
    - Use same training config, same data
    - Train, run all 3 evaluators
    - Output to `results/baselines/augmentation_consistency.json`
  - This is the CLOSEST alternative explanation baseline (exp.md CLIP+augmentation consistency)

  **Must NOT do**:
  - Do not confuse with perturbation consistency evaluator (T12)
  - Do not use cross-modal tangent vectors (that's our method)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires modified loss that replaces cross-modal tangent with intra-modal consistency
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T9-T14)
  - **Blocks**: T17, T22
  - **Blocked By**: T3, T4, T7, T10, T11, T12

  **References**:
  - `exp.md:19` — CLIP+augmentation consistency: tests whether gains from generic invariance vs first-order matching
  - `exp.md:106` — If augmentation consistency ≈ tangent method, need to refine perturbation design

  **Acceptance Criteria**:
  - [ ] Uses intra-modal consistency loss (not cross-modal tangent)
  - [ ] Same training setup as other baselines
  - [ ] Outputs JSON with all evaluator results
  - [ ] Results comparable to standard fine-tune (neither method's core hypothesis)

  **QA Scenarios**:

  ```
  Scenario: Augmentation consistency baseline
    Tool: Bash (python)
    Preconditions: T9 working, T10-T12 complete
    Steps:
      1. Run: `python src/baselines/augmentation_consistency.py --config config/baselines.yaml`
      2. Verify training completes
      3. Verify results/baselines/augmentation_consistency.json exists
      4. Compare with standard_finetune results
    Expected Result: Training completes, results in expected range
    Failure Indicators: Training fails, results diverge wildly
    Evidence: .sisyphus/evidence/task-15-aug-consistency.json
  ```

  **Commit**: YES
  - Message: `feat: augmentation consistency baseline (closest alternative explanation)`
  - Files: src/baselines/augmentation_consistency.py
  - Pre-commit: `python -c "from src.baselines.augmentation_consistency import run_aug_consistency_baseline"`

---

### Wave 3: Advanced Features + Experiment Management

- [ ] 16. LoRA Adapter (Stage 2 Training)

  **What to do**:
  - Create `src/models/lora_adapter.py`:
    - `apply_lora(model, config) -> model` — Apply LoRA to CLIP using peft library
    - LoRA targets: attention layers in both vision_model and text_model
    - Config: rank (default 16), alpha (default 32), dropout (default 0.1)
    - `merge_and_save(model, path)` — Merge LoRA weights and save full model
  - Create `config/stage2.yaml`:
    - Inherits stage1 config
    - Overrides: trainable_parts="lora", lr lower (1e-5), more epochs
    - LoRA-specific params: rank, alpha, dropout
  - Modify trainer to support loading stage1 checkpoint → unfreeze LoRA → continue training
  - Create `scripts/run_stage2.sh`: Load stage1 best checkpoint, train with LoRA

  **Must NOT do**:
  - Do not apply LoRA to embedding layers or projection heads
  - Do not modify trainer.py core loop (only add checkpoint loading)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: LoRA integration requires understanding of peft library, CLIP architecture injection points, and checkpoint loading between stages
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Wave 2)
  - **Blocks**: T21
  - **Blocked By**: T9, T3

  **References**:
  - `exp.md:35` — Stage 2: if stable, add LoRA or late-layer adapters
  - `exp.md:73-74` — Stage 2: projection head + late layers / LoRA, medium budget
  - peft library documentation — LoRA injection API

  **Acceptance Criteria**:
  - [ ] LoRA adapters injected into attention layers only
  - [ ] Stage 1 checkpoint loads correctly
  - [ ] Training continues with reduced lr
  - [ ] Merge produces standalone model

  **QA Scenarios**:

  ```
  Scenario: LoRA injection and training
    Tool: Bash (python)
    Preconditions: T9 working, stage1 checkpoint exists
    Steps:
      1. Load pretrained model, apply LoRA
      2. Verify LoRA modules present (print model)
      3. Run 1 epoch of training
      4. Verify loss decreases
      5. Merge LoRA and save
    Expected Result: LoRA trains, merges correctly
    Failure Indicators: LoRA not applied, training crashes
    Evidence: .sisyphus/evidence/task-16-lora.txt
  ```

  **Commit**: YES
  - Message: `feat: LoRA adapter for Stage 2 training`
  - Files: src/models/lora_adapter.py, config/stage2.yaml, scripts/run_stage2.sh
  - Pre-commit: `python -c "from src.models.lora_adapter import apply_lora"`

---

- [ ] 17. Ablation Experiment Runner

  **What to do**:
  - Create `src/experiments/ablation.py`:
    - Load ablation config: `config/ablation.yaml`
    - Each ablation is a set of overrides applied to base config:
      1. `lambda_tan=0`: No tangent term (same as standard fine-tune baseline)
      2. `augmentation_consistency`: Replace tangent with intra-modal consistency
      3. `broken_pairs`: Shuffle x⁺ to wrong y⁺ (break cross-modal pairing)
      4. `image_only_perturb`: Only image-side perturbation, text unchanged
      5. `text_only_perturb`: Only text-side perturbation, image unchanged
      6. `identity_preserving_only`: Only use identity-preserving perturbations
      7. `identity_changing_only`: Only use identity-changing perturbations
      8. `lambda_tan_sweep`: Test λ_tan ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0}
    - For each ablation:
      - Create modified config
      - Train model
      - Run all evaluators
      - Save results to `results/ablations/{name}.json`
    - Generate comparison table across all ablations

  **Must NOT do**:
  - Do not skip any ablation from exp.md Section 9
  - Do not run ablations sequentially if they can be parallelized (but note GPU memory constraints)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires careful config modification, training orchestration, and results aggregation across multiple experiments
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Wave 2)
  - **Blocks**: T22
  - **Blocked By**: T9, T13, T14, T15

  **References**:
  - `exp.md:93-99` — Complete ablation list (all 8 tests above)
  - `exp.md:101-106` — Expected result patterns that validate/invalidate hypothesis
  - `exp.md:60` — Critical rule: identity-preserving vs identity-changing separation

  **Acceptance Criteria**:
  - [ ] All 8 ablation variants defined in config
  - [ ] Each ablation trains and evaluates independently
  - [ ] Results saved to separate JSON files
  - [ ] Comparison table generated

  **QA Scenarios**:

  ```
  Scenario: Run one ablation variant
    Tool: Bash (python)
    Preconditions: T9 working
    Steps:
      1. Run: `python src/experiments/ablation.py --config config/ablation.yaml --ablations lambda_tan_zero`
      2. Verify results/ablations/lambda_tan_zero.json exists
      3. Verify JSON has retrieval, compositional, consistency metrics
    Expected Result: Ablation runs, results saved
    Failure Indicators: Crash, missing results
    Evidence: .sisyphus/evidence/task-17-ablation.json

  Scenario: Lambda_tan sweep
    Tool: Bash (python)
    Preconditions: T9 working
    Steps:
      1. Run sweep with 3 lambda values (for speed)
      2. Verify 3 result files produced
      3. Verify R@1 varies across lambda values
    Expected Result: Sweep produces comparable results
    Failure Indicators: All identical, crash
    Evidence: .sisyphus/evidence/task-17-sweep.json
  ```

  **Commit**: YES
  - Message: `feat: ablation experiment runner with all exp.md Section 9 tests`
  - Files: src/experiments/ablation.py
  - Pre-commit: `python -c "from src.experiments.ablation import AblationRunner"`

---

- [ ] 18. Training Run Scripts + Experiment Configs

  **What to do**:
  - Create shell scripts:
    - `scripts/run_stage1.sh`:
      ```bash
      #!/bin/bash
      python src/training/train.py --config config/stage1.yaml "$@"
      ```
    - `scripts/run_stage2.sh`:
      ```bash
      #!/bin/bash
      CHECKPOINT=$(find results/checkpoints -name "stage1_best.pt" | head -1)
      python src/training/train.py --config config/stage2.yaml --override training.resume_from="$CHECKPOINT" "$@"
      ```
    - `scripts/run_baselines.sh`:
      ```bash
      #!/bin/bash
      python src/baselines/frozen_clip.py --config config/baselines.yaml
      python src/baselines/standard_finetune.py --config config/baselines.yaml
      python src/baselines/augmentation_consistency.py --config config/baselines.yaml
      ```
    - `scripts/run_ablation.sh`:
      ```bash
      #!/bin/bash
      python src/experiments/ablation.py --config config/ablation.yaml "$@"
      ```
    - `scripts/run_all.sh`: Master script that runs everything in order
  - Finalize all YAML config files with documented defaults and comments

  **Must NOT do**:
  - Do not hardcode absolute paths (use relative paths or config)
  - Do not skip error checking in shell scripts

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell scripts and config finalization
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T16, T17, T19, T20)
  - **Blocks**: —
  - **Blocked By**: T2

  **References**:
  - `exp.md:70-76` — Training stages inform script structure
  - All config files created in T1

  **Acceptance Criteria**:
  - [ ] All 5 shell scripts are executable (chmod +x)
  - [ ] All YAML configs are complete with comments
  - [ ] `scripts/run_all.sh` runs full experiment pipeline

  **QA Scenarios**:

  ```
  Scenario: Script executability
    Tool: Bash (ls)
    Preconditions: Scripts created
    Steps:
      1. Check all scripts have execute permission
      2. Verify syntax: `bash -n scripts/run_stage1.sh`
    Expected Result: All scripts executable, no syntax errors
    Failure Indicators: Permission denied, syntax error
    Evidence: .sisyphus/evidence/task-18-scripts.txt
  ```

  **Commit**: YES
  - Message: `chore: training scripts and finalized experiment configs`
  - Files: scripts/, config/*.yaml
  - Pre-commit: `bash -n scripts/run_stage1.sh && bash -n scripts/run_all.sh`

---

- [ ] 19. Full Evaluation Pipeline Integration

  **What to do**:
  - Create `src/eval/pipeline.py` with `EvaluationPipeline` class:
    - `run(model, dataset, output_dir) -> dict`:
      - Run all 3 evaluators sequentially
      - Aggregate results into single JSON
      - Save individual evaluator results + aggregate
    - `compare(baseline_results_dir) -> dict`:
      - Load all baseline results
      - Generate comparison table
      - Compute deltas (tangent vs standard, tangent vs frozen)
      - Highlight wins/losses per metric
  - Create `src/eval/__init__.py` with unified exports
  - Integrate with trainer: after each epoch, run validation through pipeline

  **Must NOT do**:
  - Do not skip any evaluator
  - Do not modify evaluator implementations (just orchestrate)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration of 3 evaluators with comparison logic
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Wave 2)
  - **Blocks**: T21, T22
  - **Blocked By**: T9, T10, T11, T12

  **References**:
  - `exp.md:85-91` — All evaluation metrics to include
  - `exp.md:87-91` — Metric groups and their importance levels

  **Acceptance Criteria**:
  - [ ] Pipeline runs all 3 evaluators
  - [ ] Aggregate JSON has all metrics
  - [ ] Comparison table shows tangent vs baselines
  - [ ] Trainer calls pipeline for validation

  **QA Scenarios**:

  ```
  Scenario: Full evaluation pipeline
    Tool: Bash (python)
    Preconditions: T10-T12 complete
    Steps:
      1. Run pipeline on small dataset
      2. Verify aggregate JSON has retrieval + compositional + consistency sections
      3. Verify comparison function loads baselines and produces deltas
    Expected Result: Complete evaluation with comparison
    Failure Indicators: Missing sections, wrong deltas
    Evidence: .sisyphus/evidence/task-19-pipeline.json
  ```

  **Commit**: YES
  - Message: `feat: unified evaluation pipeline with baseline comparison`
  - Files: src/eval/pipeline.py, src/eval/__init__.py
  - Pre-commit: `python -c "from src.eval.pipeline import EvaluationPipeline"`

---

- [ ] 20. Experiment Tracking (W&B)

  **What to do**:
  - Create `src/experiments/tracking.py` with `ExperimentTracker` class:
    - `__init__(config, project_name, experiment_name)` — Init W&B run
    - `log_metrics(metrics, step)` — Log to W&B
    - `log_hyperparams(params)` — Log config
    - `log_image(image, caption, step)` — Log sample images
    - `finish()` — End W&B run
    - Fallback: if W&B not available, log to local JSON only
    - Support `--no_wandb` flag to disable W&B
  - Integrate with trainer: log train/val metrics per epoch
  - Log key visualizations:
    - Training curves (loss components)
    - Sample retrieval results (image + caption pairs)
    - Embedding space snapshots (periodic)

  **Must NOT do**:
  - Do not force W&B login (must work offline)
  - Do not log images every epoch (only periodically)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: W&B integration is well-documented
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T16-T19)
  - **Blocks**: —
  - **Blocked By**: T8

  **References**:
  - `exp.md:69` — Monitor standard retrieval and factor-specific validation each epoch
  - W&B documentation — Standard integration patterns

  **Acceptance Criteria**:
  - [ ] W&B logging works when API key present
  - [ ] Falls back to local JSON when W&B unavailable
  - [ ] Trainer logs metrics via tracker
  - [ ] `--no_wandb` flag disables W&B

  **QA Scenarios**:

  ```
  Scenario: Local fallback mode
    Tool: Bash (python)
    Preconditions: No W&B API key
    Steps:
      1. Create tracker with no_wandb=True
      2. Log some metrics
      3. Verify local JSON created
    Expected Result: Local logging works without W&B
    Failure Indicators: Crash on missing API key
    Evidence: .sisyphus/evidence/task-20-tracking.txt
  ```

  **Commit**: YES (grouped with T22)
  - Message: `feat: W&B experiment tracking with local fallback`
  - Files: src/experiments/tracking.py
  - Pre-commit: `python -c "from src.experiments.tracking import ExperimentTracker"`

---

### Wave 4: Integration + Polish + Documentation

- [ ] 21. Stage 3 Full Fine-tune Support

  **What to do**:
  - Create `config/stage3.yaml`:
    - trainable_parts="full" (all parameters trainable)
    - Lower learning rate (1e-6), more epochs
    - Stronger regularization (higher λ_reg)
    - Gradient checkpointing enabled (for memory efficiency)
  - Modify trainer to handle `trainable_parts="full"`:
    - Unfreeze all parameters
    - Load stage2 checkpoint (with merged LoRA)
    - Use gradient checkpointing for memory savings
    - Log total trainable parameters
  - Create `scripts/run_stage3.sh`: Load stage2 best checkpoint, train with full fine-tune
  - Add memory monitoring: log GPU memory usage, warn if OOM risk

  **Must NOT do**:
  - Do not run stage 3 by default (it's optional, only if compute allows — exp.md)
  - Do not skip gradient checkpointing (needed for full fine-tune memory)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Full fine-tuning requires careful memory management and training stability
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (after Wave 3)
  - **Blocks**: T22
  - **Blocked By**: T16, T19

  **References**:
  - `exp.md:36` — Stage 3: only if data and compute sufficient
  - `exp.md:74-75` — Stage 3: optional stronger adaptation
  - `exp.md:104` — Stress test: verify gains persist under stronger fine-tuning

  **Acceptance Criteria**:
  - [ ] Stage 3 config loads correctly
  - [ ] All parameters unfrozen
  - [ ] Gradient checkpointing enabled
  - [ ] GPU memory logged

  **QA Scenarios**:

  ```
  Scenario: Stage 3 config and memory
    Tool: Bash (python)
    Preconditions: T16 working, stage2 checkpoint
    Steps:
      1. Load model with trainable_parts="full"
      2. Verify all params trainable
      3. Enable gradient checkpointing
      4. Log param count and estimated memory
    Expected Result: Config correct, memory estimate reasonable
    Failure Indicators: Not all params trainable, memory estimation wrong
    Evidence: .sisyphus/evidence/task-21-stage3.txt
  ```

  **Commit**: YES
  - Message: `feat: Stage 3 full fine-tune with gradient checkpointing`
  - Files: config/stage3.yaml, scripts/run_stage3.sh, modifications to trainer.py
  - Pre-commit: `python -c "from src.training.trainer import TangentTrainer"`

---

- [ ] 22. Result Aggregation and Reporting

  **What to do**:
  - Create `src/experiments/report.py`:
    - `aggregate_results(results_dir) -> dict`:
      - Load all result JSONs (frozen, standard, tangent, aug_consistency, ablations)
      - Create comparison table:
        | Metric | Frozen | Standard FT | Tangent (Ours) | Aug Consistency |
        |--------|--------|-------------|----------------|-----------------|
        | I2T R@1 | ... | ... | ... | ... |
        | T2I R@1 | ... | ... | ... | ... |
        | Group Score | ... | ... | ... | ... |
        | Consistency | ... | ... | ... | ... |
      - Per-factor breakdowns
      - Ablation comparison table
      - Compute deltas and win/loss indicators
    - `generate_markdown_report(aggregated) -> str`:
      - Executive summary
      - Detailed tables
      - Key findings (does tangent method beat baselines on relation-sensitive cases?)
      - Hypothesis validation (support / refute / inconclusive)
    - `plot_training_curves(metrics_jsonl, output_path)`:
      - Loss curves (L_clip, L_tan, L_total)
      - Validation metric curves
    - CLI: `python src/experiments/report.py --results_dir results/ --output results/report.md`

  **Must NOT do**:
  - Do not cherry-pick metrics (include all metrics in tables)
  - Do not draw conclusions beyond what data supports

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Data aggregation, table generation, and markdown report creation
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (after Wave 3)
  - **Blocks**: T24
  - **Blocked By**: T10, T11, T12, T13, T14, T15, T17, T19

  **References**:
  - `exp.md:101-106` — Expected result patterns for hypothesis validation
  - `exp.md:117-118` — Paper claims to validate
  - `exp.md:85-91` — All metrics to include in report

  **Acceptance Criteria**:
  - [ ] Comparison table includes all baselines
  - [ ] Per-factor breakdown included
  - [ ] Ablation results summarized
  - [ ] Markdown report generated
  - [ ] Training curves plotted

  **QA Scenarios**:

  ```
  Scenario: Generate report from results
    Tool: Bash (python)
    Preconditions: At least frozen_clip and standard_finetune results exist
    Steps:
      1. Run: `python src/experiments/report.py --results_dir results/`
      2. Verify results/report.md exists
      3. Verify tables have all baselines
      4. Verify training curves PNG exists
    Expected Result: Complete report with tables and curves
    Failure Indicators: Missing sections, wrong data
    Evidence: .sisyphus/evidence/task-22-report.md
  ```

  **Commit**: YES
  - Message: `feat: result aggregation, comparison tables, and markdown report`
  - Files: src/experiments/report.py
  - Pre-commit: `python -c "from src.experiments.report import aggregate_results"`

---

- [ ] 23. Training Curve Visualization

  **What to do**:
  - Create `src/visualize/training_curves.py`:
    - `plot_loss_curves(metrics_jsonl_path, output_path)`:
      - L_clip (blue), L_tan (red), L_total (black) over epochs
      - Separate panels for train vs val
    - `plot_retrieval_curves(metrics_jsonl_path, output_path)`:
      - I2T R@1, T2I R@1 over epochs
      - Per-factor R@1 as dashed lines
    - `plot_ablation_comparison(ablation_results_dir, output_path)`:
      - Bar chart: each ablation's R@1 side by side
      - Color-coded: green for better than baseline, red for worse
    - `plot_tangent_cosine_distribution(model, dataset, output_path)`:
      - Histogram of cos(ΔI, ΔT) across test set
      - Separate by factor
  - Create CLI: `python src/visualize/training_curves.py --metrics results/metrics.jsonl --output results/plots/`

  **Must NOT do**:
  - Do not use 3D plots or overly complex visualizations
  - Do not generate plots that don't add clarity

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Data visualization with matplotlib, clear and publication-quality plots
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T22, T24)
  - **Blocks**: —
  - **Blocked By**: T20, T9

  **References**:
  - `visualize/scripts/plot_umap_analysis.py` — Existing matplotlib patterns to follow
  - `exp.md:115` — Save qualitative retrieval examples

  **Acceptance Criteria**:
  - [ ] Loss curves plot generated
  - [ ] Retrieval curves plot generated
  - [ ] Ablation comparison bar chart generated
  - [ ] Tangent cosine distribution histogram generated
  - [ ] All plots saved as PNG

  **QA Scenarios**:

  ```
  Scenario: Generate training curve plots
    Tool: Bash (python)
    Preconditions: Metrics JSONL with 5+ entries
    Steps:
      1. Run: `python src/visualize/training_curves.py --metrics results/metrics.jsonl --output results/plots/`
      2. Verify loss_curves.png, retrieval_curves.png exist
      3. Verify images are valid PNGs
    Expected Result: Plots generated correctly
    Failure Indicators: Missing files, corrupted PNGs
    Evidence: .sisyphus/evidence/task-23-plots/
  ```

  **Commit**: YES (grouped with T22)
  - Message: `feat: training curve and ablation comparison visualizations`
  - Files: src/visualize/training_curves.py
  - Pre-commit: `python -c "from src.visualize.training_curves import plot_loss_curves"`

---

- [ ] 24. README + Experiment Documentation

  **What to do**:
  - Create `README.md`:
    - Project overview and motivation (link to exp.md for full details)
    - Quick start: `pip install -r requirements.txt && bash scripts/run_all.sh`
    - Project structure overview
    - Configuration guide
    - Reproducing experiments
    - Expected results summary
    - Citation (if applicable)
  - Create `EXPERIMENT_LOG.md` template:
    - Experiment ID, date, config, results summary
    - Structured format for recording each run

  **Must NOT do**:
  - Do not duplicate exp.md content (reference it instead)
  - Do not include code that can't be run

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Technical documentation writing
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (after all results)
  - **Blocks**: —
  - **Blocked By**: T22

  **References**:
  - `exp.md` — Reference for full experiment details
  - `exp.md:117-118` — Paper claims for README motivation section

  **Acceptance Criteria**:
  - [ ] README.md exists with all required sections
  - [ ] All commands in README are valid
  - [ ] EXPERIMENT_LOG.md template exists
  - [ ] Quick start works end-to-end

  **QA Scenarios**:

  ```
  Scenario: README completeness
    Tool: Bash (grep)
    Preconditions: README created
    Steps:
      1. Verify sections: Overview, Quick Start, Structure, Config, Experiments, Results
      2. Verify code blocks are syntactically valid
      3. Verify file references exist
    Expected Result: All sections present, references valid
    Failure Indicators: Missing sections, broken references
    Evidence: .sisyphus/evidence/task-24-readme.txt
  ```

  **Commit**: YES
  - Message: `docs: README and experiment log template`
  - Files: README.md, EXPERIMENT_LOG.md
  - Pre-commit: `test -f README.md && test -f EXPERIMENT_LOG.md`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, check import). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check all 24 task deliverables exist. Verify tangent loss formula L_tan = 1 - cos(ΔI, ΔT) matches exp.md exactly. Verify dataset/ and visualize/ directories are unmodified.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [24/24] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m py_compile` on all source files. Check for: bare excepts, unused imports, hardcoded paths, print statements in library code. Verify type hints on public APIs (loss functions, evaluators, model wrapper). Check docstrings on loss functions and evaluators.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Integration Testing** — `unspecified-high`
  Run full pipeline end-to-end with tiny config: `pip install -r requirements.txt` → Stage1 training (2 epochs, 4 samples) → evaluate retrieval → run frozen baseline → run one ablation variant. Verify all JSON outputs produced. Verify loss decreases over training.
  Output: `Pipeline [PASS/FAIL] | Outputs [N/N] | Loss [decreasing/stuck] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual code. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance per task. Verify no cross-task contamination (Task N not touching Task M's files). Flag unaccounted changes.
  Output: `Tasks [24/24 compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1 complete**: `feat(infra): project config, model wrapper, dataset, losses` — src/, config/, requirements.txt
- **Wave 2 complete**: `feat(training): pipeline, evaluators, baselines` — src/training/, src/eval/, src/baselines/
- **Wave 3 complete**: `feat(experiments): LoRA, ablation, tracking, scripts` — src/experiments/, scripts/
- **Wave 4 complete**: `feat(report): visualization, docs, integration` — src/experiments/report.py, README.md

---

## Success Criteria

### Verification Commands
```bash
# 安装依赖
pip install -r requirements.txt

# Stage 1 训练（快速验证）
python src/training/train.py --config config/stage1.yaml --max_epochs 2 --batch_size 4

# 标准检索评估
python src/eval/retrieval.py --checkpoint results/checkpoints/stage1_best.pt

# 运行基线
python src/baselines/frozen_clip.py --config config/eval.yaml

# 消融实验
python src/experiments/ablation.py --config config/ablation.yaml

# 结果汇总
python src/experiments/report.py --results_dir results/
```

### Final Checklist
- [ ] 所有 "Must Have" 组件存在
- [ ] 所有 "Must NOT Have" 行为未出现
- [ ] 训练损失在小规模测试中下降
- [ ] 评估输出符合 JSON schema
- [ ] dataset/ 和 visualize/ 目录未被修改
