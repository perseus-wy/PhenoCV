# phenocv-phenotype-port

A [WorkBuddy skill](https://www.workbuddy.cn) that captures the reusable pattern
behind PhenoCV's `phenocv.phenotypes` package: a registry-based **four-tier**
input model (mask-only → mask+RGB → mask+depth+calibration → mask+multispectral),
a `TraitExtractor` extension point, and a strict **fail-closed** convention so
another agent can compute traits from a mask deliverable without knowing the
internals.

Use it when porting a private CV / plant-phenotyping pipeline into an
open-source, agent-friendly toolkit, or building a pluggable trait engine from
scratch.

## Install as a WorkBuddy skill

Two options:

1. **Copy into your user skills folder** (recommended — persists across projects):

   ```powershell
   # Windows
   Copy-Item -Recurse .\skills\phenocv-phenotype-port `
     "$env:USERPROFILE\.workbuddy\skills\phenocv-phenotype-port"

   # macOS / Linux
   cp -r skills/phenocv-phenotype-port \
     ~/.workbuddy/skills/phenocv-phenotype-port
   ```

2. **Or load it from this repo on demand** — point WorkBuddy's skill loader at
   `skills/phenocv-phenotype-port/` and run `Skill` with
   `skill: "phenocv-phenotype-port"`.

The skill's `references/registry_pattern.py` is a minimal, numpy-only, runnable
reference of the tier registry + orchestrator (fail-closed, no fabricated
columns) — a starting point you can adapt.
