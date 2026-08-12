---
name: phenocv-phenotype-port
description: "Use this skill when porting a private computer-vision / plant-phenotyping pipeline into an open-source, agent-friendly toolkit, or when building a pluggable, fail-closed trait-extraction engine from scratch. It captures the reusable pattern behind PhenoCV's `phenocv.phenotypes` package and the `phenocv.thermal` (FLIR) module: a registry-based input-tier model (mask-only → mask+RGB → mask+depth+calib → mask+multispectral; and for thermal, mask + thermal + ambient temperature), a `TraitExtractor` extension point, data-agnostic core logic, and a strict fail-closed convention so another agent can compute traits from a mask / temperature deliverable without knowing internals. 本技能用于把私有 CV / 植物表型流水线移植为开源、智能体友好的工具包，或从零搭建可插拔、fail-closed 的表型提取引擎；沉淀 PhenoCV `phenocv.phenotypes` 与 `phenocv.thermal`（热红外 FLIR）的可复用模式：分层的输入注册表、TraitExtractor 扩展点、数据无关的核心逻辑、严格 fail-closed 约定（缺失/不可观测 → NaN + missing_reason）。"
agent_created: true
---

# PhenoCV-style Phenotype Port

Reusable blueprint for turning a monolithic CV/phenotyping script into an
open-source, extensible, *fail-closed* trait engine that other agents can drive.

## When to use

- Porting a private pipeline (e.g. soybean canopy segmentation → `PhenoCV`)
  into a public repo.
- A calculation module has grown `if/else` branches on "do we have RGB? depth?
  multispectral?" — replace with a **tier registry**.
- Porting a **thermal / FLIR** pipeline (temperature matrix + mask → canopy-layer
  traits, environment-sensor alignment, before/after stress analysis) into
  `phenocv.thermal` — same fail-closed + data-agnostic + lazy-import contract.
- You need a mask / temperature deliverable that *another* agent (or a human) can
  turn into a trait table without re-implementing the math.

## Core pattern: four input tiers

Model every trait by the **minimum inputs** it needs. Four natural tiers:

| Tier | Inputs | Example traits |
|------|--------|----------------|
| 1 | `mask` | area, bbox, centroid, solidity, circularity, aspect ratio |
| 2 | `mask`, `rgb` | normalized-RGB vegetation indices (ExG/ExGR/GLI/VARI/…) |
| 3 | `mask`, `depth`, `calibration` | plant height (mm, above fitted soil plane), projected area, envelope volume |
| 4 | `mask`, `multispectral` | NDVI/NDRE/GNDVI/… over calibrated reflectance bands |

Inputs beyond the mask are **optional and progressively richer**. A seedling
RGB frame runs only L1+L2; a full RGB-D frame adds L3; an MS400 frame adds L4.

## The registry extension point

Every trait lives in a `TraitExtractor` subclass with `name`, `requires`
(list of input tags, a *subset* of available), `tier`, and `extract(**inputs)`.
Decorate with `@register`. The orchestrator:

1. builds the set of available inputs from what the caller passed,
2. calls `available_for(inputs)` → every extractor whose `requires` ⊆ available,
   sorted by `tier`,
3. runs each `extract()` in a `try/except`, merging the returned dict,
4. records a `name_error` column (the extractor's registered name substituted
   for `name`) for any extractor that throws — **never aborts the row**.

Adding a new trait = write one class + `@register`. **Do not edit the
orchestrator.** Full sketch in `references/registry_pattern.py`.

## Fail-closed convention (non-negotiable)

- If an input is missing, the tier simply does not run (its columns are absent,
  not zeroed).
- If a value is unobservable (e.g. empty mask, degenerate soil plane), emit
  `NaN` plus a `missing_reason` string — **never a fabricated number**.
- Sign-preserving safe division: indices like VARI/WI legitimately have negative
  denominators — use a `_safe_divide` that returns `NaN` on singular, not raw `/`.

## Deliverable "four-piece" set (so other agents can compute)

When you publish masks, also publish:

1. a **manifest** with provenance (`frame`, source paths, `inputs_present`,
   `extractors_run`),
2. a **long CSV** of traits (one row per frame, all trait columns),
3. **per-frame JSON** (full row, for debugging / re-loading),
4. a **runnable `compute_phenotypes.py`** entry point (batch over a mask dir +
   optional `--rgb-dir/--depth-dir/--calibration/--ms-root`) that imports the
   engine and writes pieces 1–3. The engine does the math; the script only
   discovers frames and wires paths.

This is what lets "another智能体" recompute traits from your masks.

## Porting checklist (private → public)

- [ ] **Desensitize**: scan for secrets, personal paths (`/Volumes/`, `//10.`,
  `W:/`, `E:/Software`), and real data/weights; keep those git-ignored.
  (Companion skill: `opensource-desensitize-publish`.)
- [ ] **Lazy heavy imports**: `torch`/`sam2`/CUDA must import *inside* the
  function/class that needs them, so the core engine + CPU tests stay
  importable without GPU.
- [ ] **Data-source agnostic core**: never read a dataset layout in the engine;
  add an adapter under `adapters/` instead.
- [ ] **Bilingual thin docs**: `SKILL.md` (trigger + full body), `AGENTS.md` /
  `CLAUDE.md` as thin entry points pointing to `SKILL.md`. Keep the skill body
  in one place.
- [ ] **CPU tests**: cover each tier + fail-closed paths; run `pytest` before
  any push.

## References

- `references/registry_pattern.py` — minimal, runnable reference implementation
  of the tier registry + orchestrator (numpy-only, no domain code).
