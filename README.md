# Pareto-Guided-Teacher-Alignment

## 1) Repository

This codebase supports:
- Base generation on controlled demographic grids
- Fairness audits (PBI, formality, emotion, lexical, NLI-personalization)
- Alignment data construction (self-revision, pair-aware, PGCD)
- LoRA SFT and optional DPO training
- Cross-domain evaluation on `climate` and `vaccination`

## 2) Main folders

- `configs/`: experiment configs (`climate.yaml`, `vaccination.yaml`, etc.)
- `fairness_alignment/`: generation, audits, dataset builders, training scripts
- `runs/`: model outputs, audits, and tables 


## 3) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using GPU, ensure compatible CUDA/PyTorch versions for your machine.

## 4) Generate base outputs

Climate:
```bash
python -m fairness_alignment.generate \
  --config configs/climate.yaml \
  --out runs/climate/orig.jsonl
```

Vaccination:
```bash
python -m fairness_alignment.generate \
  --config configs/vaccination.yaml \
  --out runs/vaccination/orig.jsonl
```

## 5) Run audits

Example (repeat for each run file):
```bash
python -m fairness_alignment.audit --config configs/climate.yaml --input runs/climate/orig.jsonl --out-dir runs/climate/orig_pbi
python -m fairness_alignment.formality_audit --input runs/climate/orig.jsonl --out-dir runs/climate/orig_formality
python -m fairness_alignment.emotion_audit --input runs/climate/orig.jsonl --out-dir runs/climate/orig_emotion
python -m fairness_alignment.lexical_audit --input runs/climate/orig.jsonl --out-dir runs/climate/orig_lexical
python -m fairness_alignment.nli_personalization_audit --input runs/climate/orig.jsonl --out-dir runs/climate/orig_nli
```

## 6) Alignment pipelines

- **SR-M2 / SR-M3**: pair-aware SFT dataset construction + LoRA SFT
- **T32-M\***: teacher-guided candidate generation and constrained selection
- **T32-Pref-\***: PGCD SFT warm start + DPO

Core builders/trainers:
- `fairness_alignment/build_pair_aware_sft.py`
- `fairness_alignment/build_pgcd_candidates_from_runs.py`
- `fairness_alignment/build_pgcd_dataset.py`
- `fairness_alignment/train_lora_sft.py`
- `fairness_alignment/train_pgcd_dpo.py`

## 7) Paper tables

- Main climate/vaccination comparison tables: from aggregated metrics in `runs/climate/*` and `runs/vaccination/*`
- New-family subset (Gemma9B student) tables: from corresponding `runs/*` outputs labeled `GM` in the paper
- Lexical WEAT metrics use:
  - gender: `career/family`, `power/support`
  - age: `innovation/tradition`, `energy/experience`

## 8) Reproducibility notes

- All compared runs should be evaluated on the same context grid.
- Report both fairness and personalization metrics; no single metric is sufficient.
- Use bounded-regression selection criteria for model choice (Pareto-style trade-offs).

