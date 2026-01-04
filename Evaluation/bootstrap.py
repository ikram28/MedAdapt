#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd


MODEL_SPECS: List[Dict[str, Optional[str]]] = [
    {"model_name": "BioMistral-7B", "original_qcm_dir": "BioMistral-7B", "original_qcmu_dir": "BioMistral-7B-qcmu", "swapped_dir": "BioMistral-7B-position-bias"},
    {"model_name": "BioMistral-7B-CPT", "original_qcm_dir": "BioMistral-7B-CPT", "original_qcmu_dir": "BioMistral-7B-CPT-qcmu", "swapped_dir": "BioMistral-7B-CPT-position-bias"},
    {"model_name": "gemma-3-4b-it", "original_qcm_dir": "gemma-3-4b-it", "original_qcmu_dir": "gemma-3-4b-it-qcmu", "swapped_dir": "gemma-3-4b-it-position-bias"},
    {"model_name": "Gemma3-4B-it-CPT", "original_qcm_dir": "Gemma3-4B-it-CPT", "original_qcmu_dir": "Gemma3-4B-it-CPT-qcmu", "swapped_dir": "Gemma3-4B-it-CPT-position-bias"},
    {"model_name": "Gemma3-4B-pt-CPT", "original_qcm_dir": "Gemma3-4B-pt-CPT", "original_qcmu_dir": "Gemma3-4B-pt-CPT-qcmu", "swapped_dir": "Gemma3-4B-pt-CPT-position-bias"},
    {"model_name": "gemma-3-4b-pt", "original_qcm_dir": "google/gemma-3-4b-pt", "original_qcmu_dir": "google/gemma-3-4b-pt-qcmu", "swapped_dir": "google/gemma-3-4b-pt-position-bias"},
    {"model_name": "medgemma-4b-pt", "original_qcm_dir": "google/medgemma-4b-pt", "original_qcmu_dir": "google/medgemma-4b-pt-qcmu", "swapped_dir": "google/medgemma-4b-pt-position-bias"},

    {"model_name": "Llama-2-7B-chat-CPT-Nachos", "original_qcm_dir": "Llama-2-7B-chat-CPT-Nachos", "original_qcmu_dir": None, "swapped_dir": "Llama-2-7B-chat-CPT-Nachos-position-bias"},
    {"model_name": "Llama-2-7b-chat-hf", "original_qcm_dir": "Llama-2-7b-chat-hf", "original_qcmu_dir": None, "swapped_dir": "Llama-2-7b-chat-hf-position-bias"},
    {"model_name": "Llama-2-7B-CPT-Nachos", "original_qcm_dir": "Llama-2-7B-CPT-Nachos", "original_qcmu_dir": None, "swapped_dir": "Llama-2-7B-CPT-Nachos-position-bias"},
    {"model_name": "Llama-2-7b-hf", "original_qcm_dir": "Llama-2-7b-hf", "original_qcmu_dir": None, "swapped_dir": "Llama-2-7b-hf-position-bias"},
    {"model_name": "Llama-2-13B-CPT-Nachos", "original_qcm_dir": "Llama-2-13B-CPT-Nachos", "original_qcmu_dir": None, "swapped_dir": "Llama-2-13B-CPT-Nachos-position-bias"},
    {"model_name": "LLama-7b-chat-SFT", "original_qcm_dir": "LLama-7b-chat-SFT", "original_qcmu_dir": None, "swapped_dir": "LLama-7b-chat-SFT-position-bias"},

    {"model_name": "MedGemma-4B-CPT", "original_qcm_dir": "MedGemma-4B-CPT", "original_qcmu_dir": "MedGemma-4B-CPT-qcmu", "swapped_dir": "MedGemma-4B-CPT-position-bias"},
    {"model_name": "meditron-7b", "original_qcm_dir": "meditron-7b", "original_qcmu_dir": None, "swapped_dir": "meditron-7b-position-bias"},
    {"model_name": "Meditron-7B-CPT-Nachos", "original_qcm_dir": "Meditron-7B-CPT-Nachos", "original_qcmu_dir": None, "swapped_dir": "Meditron-7B-CPT-Nachos-position-bias"},

    {"model_name": "Mistral-7B-Instruct-v0.1", "original_qcm_dir": "Mistral-7B-Instruct-v0.1", "original_qcmu_dir": "Mistral-7B-Instruct-v0.1-qcmu", "swapped_dir": "Mistral-7B-Instruct-v0.1-position-bias"},
    {"model_name": "Mistral-7B-Instruct-v0.1-CPT", "original_qcm_dir": "Mistral-7B-Instruct-v0.1-CPT", "original_qcmu_dir": "Mistral-7B-Instruct-v0.1-CPT-qcmu", "swapped_dir": "Mistral-7B-Instruct-v0.1-CPT-position-bias"},
    {"model_name": "Mistral-7B-v0.1", "original_qcm_dir": "Mistral-7B-v0.1", "original_qcmu_dir": "Mistral-7B-v0.1-qcmu", "swapped_dir": "Mistral-7B-v0.1-position-bias"},
    {"model_name": "Mistral-7B-v0.1-CPT", "original_qcm_dir": "Mistral-7B-v0.1-CPT", "original_qcmu_dir": "Mistral-7B-v0.1-CPT-qcmu", "swapped_dir": "Mistral-7B-v0.1-CPT-position-bias"},
    {"model_name": "Mistral-it-SFT", "original_qcm_dir": "Mistral-it-SFT", "original_qcmu_dir": None, "swapped_dir": "Mistral-it-SFT-position-bias"},

    {"model_name": "SFT-Biomistral-7B", "original_qcm_dir": "SFT-Biomistral-7B", "original_qcmu_dir": None, "swapped_dir": "SFT-Biomistral-7B-position-bias"},
    {"model_name": "SFT-Biomistral-7B-CPT", "original_qcm_dir": "SFT-Biomistral-7B-CPT", "original_qcmu_dir": None, "swapped_dir": "SFT-Biomistral-7B-CPT-position-bias"},
    {"model_name": "SFT-gemma-3-4b-CPT", "original_qcm_dir": "SFT-gemma-3-4b-CPT", "original_qcmu_dir": "SFT-gemma-3-4b-CPT-qcmu", "swapped_dir": "SFT-gemma-3-4b-CPT-position-bias"},
    {"model_name": "SFT-gemma-3-4b-pt", "original_qcm_dir": "SFT-gemma-3-4b-pt", "original_qcmu_dir": None, "swapped_dir": "SFT-gemma-3-4b-pt-position-bias"},
    {"model_name": "SFT-LLama-7b-Nachos", "original_qcm_dir": "SFT-LLama-7b-Nachos", "original_qcmu_dir": None, "swapped_dir": "SFT-LLama-7b-Nachos-position-bias"},
    {"model_name": "SFT-LLama-13b-chat", "original_qcm_dir": "SFT-LLama-13b-chat", "original_qcmu_dir": "SFT-LLama-13b-chat-qcmu", "swapped_dir": "SFT-LLama-13b-chat-position-bias"},

    {"model_name": "SFT-medgemma-4b", "original_qcm_dir": "SFT-medgemma-4b", "original_qcmu_dir": None, "swapped_dir": "SFT-medgemma-4b-position-bias"},
    {"model_name": "SFT-Meditron-7b", "original_qcm_dir": "SFT-Meditron-7b", "original_qcmu_dir": None, "swapped_dir": "SFT-Meditron-7b-position-bias"},
    {"model_name": "SFT-Meditron-Nachos", "original_qcm_dir": "SFT-Meditron-Nachos", "original_qcmu_dir": None, "swapped_dir": "SFT-Meditron-Nachos-position-bias"},
    {"model_name": "SFT-Mistral-7B", "original_qcm_dir": "SFT-Mistral-7B", "original_qcmu_dir": None, "swapped_dir": "SFT-Mistral-7B-position-bias"},
    {"model_name": "SFT-Mistral-7B-CPT", "original_qcm_dir": "SFT-Mistral-7B-CPT", "original_qcmu_dir": None, "swapped_dir": "SFT-Mistral-7B-CPT-position-bias"},
    {"model_name": "SFT-medgemma-4b-CPT", "original_qcm_dir": "SFT-medgemma-4b-CPT", "original_qcmu_dir": None, "swapped_dir": "SFT-medgemma-4b-CPT-position-bias"},
    {"model_name": "LLama-7b-chat-CPT-SFT", "original_qcm_dir": "LLama-7b-chat-CPT-SFT", "original_qcmu_dir": None, "swapped_dir": "LLama-7b-chat-CPT-SFT-position-bias"},
    {"model_name": "SFT-Mistral-instruct-CPT-7b-New", "original_qcm_dir": "SFT-Mistral-instruct-CPT-7b-New", "original_qcmu_dir": None, "swapped_dir": "SFT-Mistral-instruct-CPT-7b-New-position-bias"},
    {"model_name": "SFT-gemma-3-4b-it-CPT", "original_qcm_dir": "SFT-gemma-3-4b-it-CPT", "original_qcmu_dir": None, "swapped_dir": "SFT-gemma-3-4b-it-CPT-position-bias"},

    {"model_name": "chaoyi-wu_MedLLaMA_13B", "original_qcm_dir": "chaoyi-wu_MedLLaMA_13B", "original_qcmu_dir": "chaoyi-wu_MedLLaMA_13B-qcmu", "swapped_dir": "chaoyi-wu_MedLLaMA_13B-position-bias"},
    {"model_name": "LLaMA_13B-CHAT-CPT", "original_qcm_dir": "LLaMA_13B-CHAT-CPT", "original_qcmu_dir": None, "swapped_dir": "LLaMA_13B-CHAT-CPT-position-bias"},
    {"model_name": "MedLLaMA-13B-CPT", "original_qcm_dir": "MedLLaMA-13B-CPT", "original_qcmu_dir": "MedLLaMA-13B-CPT-qcmu", "swapped_dir": "MedLLaMA-13B-CPT-position-bias"},
    {"model_name": "meta-llama_Llama-2-13b-chat-hf", "original_qcm_dir": "meta-llama_Llama-2-13b-chat-hf", "original_qcmu_dir": "meta-llama_Llama-2-13b-chat-hf-qcmu", "swapped_dir": "meta-llama_Llama-2-13b-chat-hf-position-bias"},
    {"model_name": "meta-llama_Llama-2-13b-hf", "original_qcm_dir": "meta-llama_Llama-2-13b-hf", "original_qcmu_dir": "meta-llama_Llama-2-13b-hf-qcmu", "swapped_dir": "meta-llama_Llama-2-13b-hf-position-bias"},
    {"model_name": "SFT-LLama-13b", "original_qcm_dir": "SFT-LLama-13b", "original_qcmu_dir": "SFT-LLama-13b-qcmu", "swapped_dir": "SFT-LLama-13b-position-bias"},
    {"model_name": "SFT-LLama-13b-chat-CPT", "original_qcm_dir": "SFT-LLama-13b-chat-CPT", "original_qcmu_dir": None, "swapped_dir": "SFT-LLama-13b-chat-CPT-position-bias"},
    {"model_name": "SFT-MedLLama-13B", "original_qcm_dir": "SFT-MedLLama-13B", "original_qcmu_dir": None, "swapped_dir": "SFT-MedLLama-13B-position-bias"},
    {"model_name": "SFT-MedLLama-Nachos-13b", "original_qcm_dir": "SFT-MedLLama-Nachos-13b", "original_qcmu_dir": "SFT-MedLLama-Nachos-13b-qcmu", "swapped_dir": "SFT-MedLLama-Nachos-13b-position-bias"},
    {"model_name": "SFT-LLama-7b", "original_qcm_dir": "SFT-LLama-7b", "original_qcmu_dir": None, "swapped_dir": "SFT-LLama-7b-position-bias"},
    {"model_name": "SFT-gemma-3-4b-it", "original_qcm_dir": "SFT-gemma-3-4b-it", "original_qcmu_dir": None, "swapped_dir": "SFT-gemma-3-4b-it-position-bias"},
    {"model_name": "LLaMA_13B-CPT-SFT", "original_qcm_dir": "LLaMA_13B-CPT-SFT", "original_qcmu_dir": None, "swapped_dir": "LLaMA_13B-CPT-SFT-position-bias"},

]
# ---------------------------------------------------------------------------



def load_json_items(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def normalize_mcq_answer(ans: str | List[str]) -> List[str]:
    items = ans if isinstance(ans, list) else re.split(r"[\s,]+", str(ans))
    return [a.strip().upper() for a in items if a.strip()]

def em_score(true_labels: List[str], pred_labels) -> float:
    pred = pred_labels or []
    if not isinstance(pred, list):
        pred = normalize_mcq_answer(pred)
    else:
        pred = [str(x).strip().upper() for x in pred if str(x).strip()]
    return 1.0 if set(true_labels) == set(pred) else 0.0

def dataset_kind_from_filename(filename: str) -> str:
    name = filename.lower()
    if "qcmu" in name:
        return "qcmu"
    if "qcm" in name:
        return "qcm"
    return "unknown"

def base_dataset_name(name: str) -> str:
    name = name.strip()
    lower = name.lower()

    prefix = "copie de "
    if lower.startswith(prefix):
        name = name[len(prefix):].lstrip()
        lower = name.lower()

    qwen_suffix = ".qwen_eval"
    if lower.endswith(qwen_suffix):
        name = name[: -len(qwen_suffix)]
        lower = name.lower()

    for suf in ["_swap1", "_swap2"]:
        if lower.endswith(suf):
            name = name[: -len(suf)]
            lower = name.lower()

    return name



def build_swapped_paths_map(swapped_dir: str) -> Dict[Tuple[str, str], Dict[str, str]]:

    swapped_paths: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    if not os.path.isdir(swapped_dir):
        return swapped_paths

    files = [f for f in os.listdir(swapped_dir) if f.lower().endswith(".json")]
    for fname in files:
        lower = fname.lower()
        kind = dataset_kind_from_filename(fname)
        if kind not in {"qcm", "qcmu"}:
            continue
        if "swap1" not in lower and "swap2" not in lower:
            continue

        stem = os.path.splitext(fname)[0]
        base_name = base_dataset_name(stem)

        if "swap1" in lower:
            v = "swap1"
        elif "swap2" in lower:
            v = "swap2"
        else:
            continue

        swapped_paths[(base_name, kind)][v] = os.path.join(swapped_dir, fname)

    return swapped_paths


# ---------------------- EM ----------------------

def compute_micro_em_for_dataset_exact(
    base_name: str,
    kind: str,
    original_results_dir: str,
    swapped_paths_for_base: Dict[str, str],
) -> Optional[Dict[str, Any]]:
   
    paths: Dict[str, str] = {}

    orig_path = os.path.join(original_results_dir, base_name + ".qwen_eval.json")
    if os.path.exists(orig_path):
        paths["original"] = orig_path

    for v in ["swap1", "swap2"]:
        p = swapped_paths_for_base.get(v)
        if p and os.path.exists(p):
            paths[v] = p

    if len(paths) == 0:
        return None

    versions: Dict[str, List[dict]] = {vname: load_json_items(p) for vname, p in paths.items()}
    lengths = {v: len(recs) for v, recs in versions.items()}
    n_list = list(lengths.values())
    n = min(n_list) if n_list else 0
    if n == 0:
        return None

    vnames = sorted(versions.keys())
    V = len(vnames)

    total_em_greedy = 0.0
    total_em_constr = 0.0

    for i in range(n):
        first_v = vnames[0]
        gold = normalize_mcq_answer(versions[first_v][i].get("answer", ""))

        for v in vnames:
            rec = versions[v][i]
            preds_g = rec.get("prediction_greedy") or []
            preds_c = rec.get("prediction_constrained") or []
            total_em_greedy += em_score(gold, preds_g)
            total_em_constr += em_score(gold, preds_c)

    micro_em_greedy = total_em_greedy / (n * V)
    micro_em_constrained = total_em_constr / (n * V)

    return {
        "dataset_base": base_name,
        "type": kind,
        "n_versions": V,
        "n_items": n,
        "em_greedy": float(micro_em_greedy),
        "em_constrained": float(micro_em_constrained),
    }


# ---------------------- Collect per-dataset scores for a model ----------------------

def resolve_original_dir(model: Dict[str, Optional[str]], kind: str) -> Optional[str]:

    if kind == "qcm":
        return model.get("qcm")
    return model.get("qcmu") or model.get("qcm")

def collect_dataset_scores_for_model(model: Dict[str, Optional[str]]) -> Dict[str, Dict[str, Dict[str, float]]]:

    qcm_dir = model.get("qcm")
    qcmu_dir = model.get("qcmu")  # may be None
    swapped_dir = model.get("swapped")
    if not qcm_dir or not swapped_dir:
        return {"qcm": {}, "qcmu": {}}

    swapped_paths = build_swapped_paths_map(swapped_dir)

    out = {"qcm": {}, "qcmu": {}}

    for (base_name, kind), version_paths in sorted(swapped_paths.items()):
        if kind not in {"qcm", "qcmu"}:
            continue

        # prevent qcmu datasets from being treated as qcm (and vice versa)
        if dataset_kind_from_filename(base_name) != kind:
            continue

        orig_dir = qcm_dir if kind == "qcm" else (qcmu_dir or qcm_dir)

        # ensure the original exists in the chosen directory
        orig_path = os.path.join(orig_dir, base_name + ".qwen_eval.json")
        if not os.path.exists(orig_path):
            continue

        agg = compute_micro_em_for_dataset_exact(
            base_name=base_name,
            kind=kind,
            original_results_dir=orig_dir,
            swapped_paths_for_base=version_paths,
        )
        if agg is None:
            continue

        out[kind][base_name] = {
            "em_greedy": agg["em_greedy"],
            "em_constrained": agg["em_constrained"],
        }

    return out





# ---------------------- Bootstrap over DATASETS  ----------------------

def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float, float]:
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d_bs = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    delta_hat = float(a.mean() - b.mean())
    ci_low = float(np.quantile(d_bs, 0.025))
    ci_high = float(np.quantile(d_bs, 0.975))

    k = min(
        np.sum(d_bs <= 0.0),
        np.sum(d_bs >= 0.0)
    )

    p_two = 2.0 * (k + 1) / (len(d_bs) + 1)
    p_two = min(p_two, 1.0)

    return delta_hat, ci_low, ci_high, p_two

def bootstrap_both_avg_of_means(
    a_qcm: np.ndarray, b_qcm: np.ndarray,
    a_qcmu: np.ndarray, b_qcmu: np.ndarray,
    n_boot: int, seed: int
) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n1, n2 = a_qcm.size, a_qcmu.size
    idx1 = rng.integers(0, n1, size=(n_boot, n1))
    idx2 = rng.integers(0, n2, size=(n_boot, n2))

    a_bs = 0.5 * a_qcm[idx1].mean(axis=1) + 0.5 * a_qcmu[idx2].mean(axis=1)
    b_bs = 0.5 * b_qcm[idx1].mean(axis=1) + 0.5 * b_qcmu[idx2].mean(axis=1)
    d_bs = a_bs - b_bs

    delta_hat = float((0.5 * a_qcm.mean() + 0.5 * a_qcmu.mean()) - (0.5 * b_qcm.mean() + 0.5 * b_qcmu.mean()))
    ci_low = float(np.quantile(d_bs, 0.025))
    ci_high = float(np.quantile(d_bs, 0.975))

    k = min(np.sum(d_bs <= 0.0), np.sum(d_bs >= 0.0))
    p_two = 2.0 * (k + 1) / (len(d_bs) + 1)
    p_two = min(p_two, 1.0)
    return delta_hat, ci_low, ci_high, p_two



# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    # Resolve model paths
    models: Dict[str, Dict[str, Optional[str]]] = {}
    for spec in MODEL_SPECS:
        name = spec["model_name"]
        qcm = os.path.join(args.root_dir, spec["original_qcm_dir"]) if spec.get("original_qcm_dir") else None
        qcmu = os.path.join(args.root_dir, spec["original_qcmu_dir"]) if spec.get("original_qcmu_dir") else None
        swapped = os.path.join(args.root_dir, spec["swapped_dir"]) if spec.get("swapped_dir") else None

        if not qcm or not os.path.isdir(qcm):
            print(f"[WARN] skip {name}: missing qcm dir {qcm}")
            continue
        if qcmu and not os.path.isdir(qcmu):
            qcmu = None  # fallback to qcm
        if not swapped or not os.path.isdir(swapped):
            print(f"[WARN] skip {name}: missing swapped dir {swapped}")
            continue

        models[name] = {"qcm": qcm, "qcmu": qcmu, "swapped": swapped}

    names = sorted(models.keys())
    if len(names) < 2:
        raise SystemExit("Need at least 2 models.")

    # Cache: (model, kind) -> dataset_base -> metric -> value
    cache: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    def get_model_scores(model_name: str):

        if model_name in cache:
            return cache[model_name]

        scores = collect_dataset_scores_for_model(models[model_name])
        cache[model_name] = scores
        return scores

    rows: List[Dict[str, Any]] = []
    metrics = ["em_greedy", "em_constrained"]

    print("\n================ DATASET SANITY CHECK ================\n")
    for name in names:
        scores = get_model_scores(name)
        qcm_scores = scores["qcm"]
        qcmu_scores = scores["qcmu"]

        print(f"Model: {name}")
        print(f"  QCM datasets   : {len(qcm_scores)}")
        print(f"  QCMU datasets  : {len(qcmu_scores)}")

        print(f"  QCM dataset names  : {sorted(qcm_scores.keys())}")
        print(f"  QCMU dataset names : {sorted(qcmu_scores.keys())}")

        print("-" * 60)
    print("\n======================================================\n")


    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]

            A_scores = get_model_scores(A)
            B_scores = get_model_scores(B)

            for metric in metrics:
                # ---------- QCM ----------
                A_qcm = A_scores["qcm"]
                B_qcm = B_scores["qcm"]
                common_qcm = sorted(set(A_qcm.keys()) & set(B_qcm.keys()))
                a_qcm = np.array([A_qcm[d][metric] for d in common_qcm], dtype=np.float32)
                b_qcm = np.array([B_qcm[d][metric] for d in common_qcm], dtype=np.float32)

                if a_qcm.size > 0:
                    delta, lo, hi, p = paired_bootstrap_delta(
                        a_qcm, b_qcm, args.n_boot,
                        args.seed + (hash((A, B, metric, "qcm")) % 10_000_000)
                    )
                    rows.append({
                        "model_a": A, "model_b": B, "metric": metric, "scope": "qcm_dataset_weighted_exactmicro",
                        "n_datasets": int(a_qcm.size),
                        "mean_a": float(a_qcm.mean()), "mean_b": float(b_qcm.mean()),
                        "delta_a_minus_b": float(delta), "ci95_low": float(lo), "ci95_high": float(hi),
                        "p_two_sided": float(p),
                    })

                # ---------- QCMU ----------
                A_qcmu = A_scores["qcmu"]
                B_qcmu = B_scores["qcmu"]
                common_qcmu = sorted(set(A_qcmu.keys()) & set(B_qcmu.keys()))
                a_qcmu = np.array([A_qcmu[d][metric] for d in common_qcmu], dtype=np.float32)
                b_qcmu = np.array([B_qcmu[d][metric] for d in common_qcmu], dtype=np.float32)

                if a_qcmu.size > 0:
                    delta, lo, hi, p = paired_bootstrap_delta(
                        a_qcmu, b_qcmu, args.n_boot,
                        args.seed + (hash((A, B, metric, "qcmu")) % 10_000_000)
                    )
                    rows.append({
                        "model_a": A, "model_b": B, "metric": metric, "scope": "qcmu_dataset_weighted_exactmicro",
                        "n_datasets": int(a_qcmu.size),
                        "mean_a": float(a_qcmu.mean()), "mean_b": float(b_qcmu.mean()),
                        "delta_a_minus_b": float(delta), "ci95_low": float(lo), "ci95_high": float(hi),
                        "p_two_sided": float(p),
                    })

                # ---------- BOTH ----------
                if a_qcm.size > 0 and a_qcmu.size > 0:
                    delta, lo, hi, p = bootstrap_both_avg_of_means(
                        a_qcm, b_qcm, a_qcmu, b_qcmu,
                        args.n_boot,
                        args.seed + (hash((A, B, metric, "both")) % 10_000_000),
                    )
                    mean_a = 0.5 * float(a_qcm.mean()) + 0.5 * float(a_qcmu.mean())
                    mean_b = 0.5 * float(b_qcm.mean()) + 0.5 * float(b_qcmu.mean())
                    rows.append({
                        "model_a": A, "model_b": B, "metric": metric, "scope": "both_avg_of_means_exactmicro",
                        "n_qcm_datasets": int(a_qcm.size),
                        "n_qcmu_datasets": int(a_qcmu.size),
                        "mean_a": float(mean_a), "mean_b": float(mean_b),
                        "delta_a_minus_b": float(delta), "ci95_low": float(lo), "ci95_high": float(hi),
                        "p_two_sided": float(p),
                    })


    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("No pairwise comparisons produced (no overlapping datasets).")

    out_df = out_df.sort_values(["metric", "scope", "p_two_sided", "delta_a_minus_b"],
                                ascending=[True, True, True, False]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
    
