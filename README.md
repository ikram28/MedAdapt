# Medical Domain Adaptation for Large Language Models

This repository contains the **code, evaluation scripts, and resources** associated with the paper:

> Trade-offs in Medical LLM Adaptation: An Empirical Study in French QA

We present a controlled and statistically grounded study of **medical domain adaptation** for large language models, comparing **continual pretraining (CPT)**, **supervised fine-tuning (SFT)**, and their combination across multiple model families, initialization types, and decoding strategies.

---

## 🔍 What this repo contains

- Training scripts
- Evaluation pipeline  
  
## 🤗 Released models

Adapted checkpoints are released on Hugging Face :
- CPT models
- SFT models
- CPT+SFT models

👉 [MODELS](https://huggingface.co/medAdapt)

## 🤗 Data

- CPT training data 👉 [NACHOS](https://huggingface.co/datasets/Dr-BERT/NACHOS)
- SFT training data 👉 [MedInjection-FR-train-set](https://huggingface.co/datasets/MedInjection/ALL)
- Evaluation data 👉 [MedInjection-FR-Test-set](https://huggingface.co/datasets/MedInjection/ALL)


---

## 🧪 Experimental scope

- **Model families**: Gemma-4B, Mistral-7B, Llama-7B / 13B  
- **Initialization types**: General, Instruction-tuned, Medical  
- **Adaptation strategies**:  
  - CPT (full-parameter)
  - SFT (parameter-efficient (DoRA))
  - CPT + SFT
- **Evaluation size**:  
    - 14,533 native French medical questions  
    - 13,293 translated examples from established English benchmarks

All main MCQA results use **constrained decoding** to ensure fair format adherence across base and instruction-tuned models.

---

## 📊 Key takeaways

- **SFT on instruction-tuned models** offers the best performance–efficiency trade-off for medical MCQA.
- **CPT+SFT** yields the highest scores most often, but gains over SFT are usually small and not consistently significant.
- **CPT alone** is unstable for MCQA and mainly benefits OEQA overlap metrics.
- Medical initialization alone does **not** reliably improve downstream QA performance.
- OEQA results should be interpreted cautiously due to metric sensitivity and limited supervision.

Practical **adaptation guidelines** are summarized in the paper’s conclusion.

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{belmadani2026tradeoffsmedicalllmadaptation,
      title={Trade-offs in Medical LLM Adaptation: An Empirical Study in French QA}, 
      author={Ikram Belmadani and Oumaima El Khettari and Carlos Ramisch and Frederic Bechet and Richard Dufour and Benoit Favre},
      year={2026},
      eprint={2606.19266},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.19266}, 
}

