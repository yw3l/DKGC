# DKGC: A Context-Aware Framework for Knowledge Graph Completion via the Dynamic Fusion of Structure and Textual Semantics

This is the official implementation of **DKGC**, a dynamic fusion framework for Knowledge Graph Completion (KGC). DKGC adaptively balances structural embeddings (from **CompoundE**) and textual semantics (from **SimKGC**) at the triple level using a context-aware gating mechanism.

---

## 📝 Description

Knowledge Graph Completion (KGC) often struggles with either structural sparsity or textual ambiguity. While existing methods integrate these two modalities, they mostly rely on static fusion. **DKGC** introduces:
- A **Dynamic Gating Mechanism** that perceives local graph context to arbitrate between modalities.
- A **Nonlinear Interaction Module** to capture complex dependencies between geometric and semantic features.
- A **Two-Stage Training Strategy** to effectively align heterogeneous representations from pre-trained encoders.

---

## 📊 Dataset Information

DKGC is evaluated on two benchmark datasets:
- **WN18RR**: A sparse subset of WordNet, where textual information is critical.
- **FB15k-237**: A dense subset of Freebase, where structural connectivity provides strong signals.

Data should be organized as follows:
```text
SimKGC/data/
├── FB15k237/
│   ├── train.txt.json
│   ├── valid.txt.json
│   ├── test.txt.json
│   └── relations.json
└── WN18RR/
    ├── train.txt.json
    ├── ...
```

---

## 💻 Code Information

- `dkgc_model.py`: Core model architecture including the Dynamic Fusion Module (Projection, Gating, and Fusion MLP).
- `train_dkgc.py`: Script for training the fusion module (Stage 2 training).
- `evaluate_dkgc.py`: Evaluation script using a two-stage "Retrieve & Re-rank" strategy.
- `SimKGC/`: Submodule for textual encoding based on PLMs (BERT).
- `CompoundE/`: Submodule for structural encoding based on geometric transformations.

---

## 🚀 Usage Instructions

### 1. Pre-training Base Encoders (Stage 1)
Follow the instructions in `SimKGC/` and `CompoundE/` directories to train the base models independently and save their checkpoints.

### 2. Training the Fusion Module (Stage 2)
Train the gating and interaction modules while keeping the base encoders frozen:
```bash
python train_dkgc.py 
    --data_dir SimKGC/data/FB15k237 
    --simkgc_checkpoint path/to/simkgc_model.ckpt 
    --compounde_checkpoint path/to/compounde_dir 
    --batch_size 32 
    --epochs 20 
    --cuda
```

### 3. Evaluation
Evaluate the model using the re-ranking strategy:
```bash
python evaluate_dkgc.py 
    --data_dir SimKGC/data/FB15k237 
    --dkgc_checkpoint path/to/dkgc_model.ckpt 
    --rerank_k 100 
    --cuda
```

---

## 🛠 Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers (HuggingFace)
- Numpy, Scipy, tqdm

Install dependencies via:
```bash
pip install -r SimKGC/requirements.txt
pip install -r CompoundE/requirements.txt
```

---

## 🧠 Methodology

DKGC operates in a shared latent space ($d_p=256$).
1. **Geometric Query**: Uses CompoundE to model relations as Scaling, Rotation, and Translation.
2. **Semantic Query**: Uses SimKGC (BERT) to encode entity descriptions and relation text.
3. **Dynamic Fusion**: 
   - **Gate ($\alpha$)**: Predicts the reliability of text vs. structure based on the joint triple context.
   - **Score**: $S_{final} = [\alpha \cdot S_{text} + (1-\alpha) \cdot S_{struct}] + f_{fusion}(v_{joint})$.

---

## 📜 Citations

If you find this work useful, please cite our paper:

```bibtex
@article{wang2026dkgc,
  title={DKGC: A Context-Aware Framework for Knowledge Graph Completion via the Dynamic Fusion of Structure and Textual Semantics},
  author={Wang, Changlong and Li, Yawei and Cao, Jianlong and Guo, Wenzheng and Hu, Yaoyao and Liu, Yi and Hu, Jie},
  journal={College of Artificial Intelligence and Computer Science, Northwest Normal University},
  year={2026}
}
```

---

## ⚖️ License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
