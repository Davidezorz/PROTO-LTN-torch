# PROTOtypical Logic Tensor Networks (PROTO-LTN) - PyTorch Implementation

A PyTorch reimplementation of **PROTOtypical Logic Tensor Networks (PROTO-LTN)** for Zero-Shot Learning, with added support for CNN fine-tuning and Explainable AI analysis tools (GradCAM, TCAV, t-SNE visualization).

## 📄 Original Paper & Repository

This work is based on the following paper:
> **PROTOtypical Logic Tensor Networks (PROTO-LTN) for Zero Shot Learning**
> by Simone Martone, Francesco Manigrasso, Fabrizio Lamberti, Lia Morra

- **Original Repository:** [https://github.com/Frankkkko/PROTO-LTN](https://github.com/Frankkkko/PROTO-LTN)
- **This Repository:** [https://github.com/Davidezorz/PROTO-LTN-torch](https://github.com/Davidezorz/PROTO-LTN-torch)

## 🔧 What's New in This Implementation

This PyTorch-based version extends the original work with:

- **PyTorch Lightning** framework for cleaner training loops and logging
- **CNN Fine-tuning** with a regularization term to allow controlled fine-tuning of the backbone network
- **Explainable AI Analysis:**
  - **GradCAM** visualization for understanding model attention
  - **TCAV** (Testing with Concept Activation Vectors) for concept-based explanations
  - **t-SNE** visualization of learned embeddings

## 📁 Repository Structure

```
PROTO-LTN-torch/
├── config.py                    # All configuration parameters (hyperparameters, flags)
├── main.py                      # Entry point: loads config, builds data & model, runs training/XAI
├── data.py                      # ZSLDataModule (PyTorch Lightning DataModule)
├── lightning_model.py           # ZSLLightningModel (PyTorch Lightning wrapper)
├── model.py                     # PROTO-LTN architecture
├── utils.py                     # Utility functions (device selection, etc.)
├── gcam.py                      # GradCAM implementation
├── TCAV.py                      # TCAV analysis
├── prepare_data_tcav.py         # Data preparation for TCAV
├── visualization.py             # t-SNE and other visualizations
├── data/                        # Dataset directory (AwA2)
├── lightning_logs/              # Training logs (PyTorch Lightning)
└── results/                     # Output results and presentations
```

## 📊 Datasets

This implementation has been trained and evaluated on the **Animals with Attributes 2 (AwA2)** dataset. The original paper also evaluates on:

- **APY** – *Describing objects by their attributes* (Farhadi et al.)
- **AwA2** – *Zero-Shot Learning -- A Comprehensive Evaluation of the Good, the Bad and the Ugly* (Xian et al.)
- **CUB** – *Caltech-UCSD Birds-200-2011* (Wah et al.)
- **SUN** – *SUN Database: Large-scale Scene Recognition from Abbey to Zoo* (Xiao et al.)

Datasets can be obtained from [DeepEmbeddingModel_ZSL](https://github.com/lzrobots/DeepEmbeddingModel_ZSL.git). Place the dataset under `./data` (the `data.py` DataModule expects `base_path='./data'` and `dataset_name='AWA2'`).

## 🚀 Getting Started

### Prerequisites & Installation

This project relies on the **Logic Tensor Networks (`ltn`)** library, which must be installed in addition to the standard scientific Python stack.

```bash
# Core dependencies
pip install torch pytorch-lightning numpy scipy scikit-learn matplotlib

# Logic Tensor Networks
# Option A — pip package:
pip install ltn
# Option B — clone the reference implementation referenced by the original repo:
git clone https://github.com/ivanDonadello/logictensornetworks.git
```

> ⚠️ The original PROTO-LTN uses the cloned `logictensornetworks` repository by Serafini, Donadello & d'Avila Garcez (["Learning and Reasoning in Logic Tensor Networks"](https://github.com/ivanDonadello/logictensornetworks)). If you use Option B, make sure the package is on your `PYTHONPATH`.

### Configuration (no CLI arguments)

`main.py` **does not take command-line arguments** — all hyperparameters and switches live in `config.py`. Edit `config.py` before running. Key fields include:

| Field in `config` | Description |
|-------------------|-------------|
| `learning_rate` | Learning rate for the optimizer |
| `alpha` | Weight for the logical axioms |
| `regularization_parameter` | L2 regularization used for CNN fine-tuning |
| `negation_axioms` | Enable negation axioms |
| `compute_feature` | Whether raw images are used (vs. precomputed features) |
| `train_cnn` | Enable CNN fine-tuning |
| `checkpoint` | Path to a checkpoint to resume from (or `None`) |
| `epochs` | Number of training epochs |
| `hidden_dense_sizes` | Hidden layer sizes for the dense predicates |
| `use_ce_loss` | Use cross-entropy loss (used by GradCAM switch) |

### Running

```bash
python main.py
```

`main()` will:
1. Load `config()`, seed everything (`pl.seed_everything(0)`).
2. Build the `ZSLDataModule` for AwA2 (`data.ZSLDataModule(base_path='./data', dataset_name='AWA2', batch_size=32, use_raw_images=config.compute_feature)`).
3. Instantiate (or load from checkpoint) the `ZSLLightningModel`.
4. Run the selected analysis pipeline (GradCAM loop, t-SNE visualization) and/or training.

To **train**, uncomment the trainer block in `main.py`:

```python
trainer = pl.Trainer(max_epochs=config_file.epochs, accelerator=device)
trainer.fit(model, data_module)
```

## 🔍 Explainability Tools

The XAI analyses are **invoked from `main.py`**, not as standalone scripts. Their calls are toggled by commenting/uncommenting blocks in `main()`.

### GradCAM

Inside `main()`:

```python
for i in range(15, 25):
    gcam.plot_gcam(model, data_module, device,
                   use_logits=config_file.use_ce_loss, random_idx=i)
    pass
```

### TCAV

Also called from `main()`:

```python
# TCAV.run_tcav(model, data_module, target_class_name="tiger", attribute_name="stripes")
TCAV.run_tutorial_tcav_lib(model, data_module, target_class_name="tiger", use_logits=True)
```

Uncomment the desired call and set `target_class_name` / `attribute_name` as needed. `prepare_data_tcav.py` supplies the concept datasets used by TCAV.

### t-SNE Visualization

Enabled by default in `main()`:

```python
import visualization
visualization.plot_tsne_with_synthetic_vectors(model, data_module)
visualization.plot_image_embeddings_tsne(model, data_module)
visualization.plot_image_embeddings_with_centroids(model, data_module)
visualization.plot_image_embeddings_with_actual_centroids(model, data_module)
```

## 📚 Logic Tensor Networks Framework

The Logic Tensor Networks framework is cloned from:
> *Learning and Reasoning in Logic Tensor Networks: Theory and Application for Semantic Image Interpretation*
> by Serafini Luciano, Donadello Ivan, d'Avila Garcez Artur

Original repository: [https://github.com/ivanDonadello/logictensornetworks.git](https://github.com/ivanDonadello/logictensornetworks.git)

## 📖 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{martone2023prototypical,
  title={PROTOtypical Logic Tensor Networks (PROTO-LTN) for Zero Shot Learning},
  author={Martone, Simone and Manigrasso, Francesco and Lamberti, Fabrizio and Morra, Lia},
  year={2023}
}
```

## 👥 Contributors

- **Davide** – PyTorch implementation, CNN fine-tuning, XAI analysis ([GitHub](https://github.com/Davidezorz))

Original PROTO-LTN authors:
- Simone Martone
- Francesco Manigrasso
- Fabrizio Lamberti
- Lia Morra

## 📝 License

This project is for academic and research purposes. Please refer to the original repository for licensing information.

## 🙏 Acknowledgments

- Original PROTO-LTN implementation by the authors
- Logic Tensor Networks framework by Serafini, Donadello, and d'Avila Garcez
- Dataset providers for AwA2, APY, CUB, and SUN

---

### Changes made vs. the previous draft

- **Added `ltn` to requirements** with both install paths (pip `ltn` and the cloned `logictensornetworks` repo referenced by the original README).
- **Removed the CLI section** — replaced with a description of `config.py`, since `main.py` takes no arguments (e.g., `learning_rate`, `alpha`, `regularization_parameter`, `negation_axioms`, `train_cnn`, `checkpoint`, etc. all live there).
- **Rewrote GradCAM / TCAV / t-SNE docs** to reflect that they are called from `main()` (currently commented/uncommented blocks), not launched as standalone scripts. `prepare_data_tcav.py` remains a separate preparation step.
- **Added a "Running" section** matching `main()`'s actual flow (seeding, DataModule construction for AWA2, checkpoint loading, analysis/trainer toggle).