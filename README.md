# 🧬 Protein Function Classifier

This project aims to **predict the biological functional class of a protein based on its amino acid sequence** using a deep learning approach.

---
## 🚀 Live Demo

Try the interactive web app here:  
👉 [https://protein-sequence-classifier-pytorch-grace.streamlit.app/](https://protein-sequence-classifier-pytorch-grace.streamlit.app/)

## 🔁 Project Workflow

| Step | Description |
|------|-------------|
| 🔎 Load Sequence & Metadata | Includes PDB structure ID, amino acid sequences, and known labels (classification) |
| 🧼 Data Cleaning & Filtering | Select the top 10 most common classes as training targets |
| 🔢 Sequence Encoding | Each amino acid is mapped to an integer, then fed into an embedding layer |
| 🧠 Model Training | An LSTM model is trained to learn patterns in the sequence and predict its class |
| 📊 Prediction | Outputs the most probable functional classification of the given protein sequence |
| 📥 Evaluation | Compares predictions with the ground truth labels and computes accuracy |

---

## ❓Why Do Scientists Need This?

- 🧪 New protein sequences obtained from experiments often lack functional annotations
- 🧬 Predicting function computationally can guide future experiments or drug design
- 🧠 Some functional properties are hard to identify experimentally, but deep learning can learn patterns from large datasets

Typical applications in bioinformatics:

- Protein structural classification (e.g. CATH, SCOP)
- Gene Ontology (GO) term prediction
- Enzyme family inference
- Subcellular localization prediction

---

## 📐 Model Architecture

`ProteinClassifier` uses a typical **Embedding → LSTM → Fully Connected** structure. This architecture is inspired by **text classification in NLP**, such as sentiment analysis or language identification.

### 🧾 References:

- *Asgari & Mofrad (2015)*: [Continuous Distributed Representation of Biological Sequences](https://doi.org/10.1371/journal.pone.0141287)
- *BioSeq-BLM (2019)*: Applied BiLSTM to protein sequence-based function prediction
- Hugging Face RNN-based models
- PyTorch sentiment classification tutorials (e.g. IMDB)

---

## ⚙️ Training Optimization Tips

| Strategy | Description |
|----------|-------------|
| 🔻 Reduce `max_len` | Set `max_len=128` or `256` to speed up LSTM processing |
| 📉 Use fewer samples | Try 1,000–2,000 sequences for quick prototyping |
| 📦 Shrink the model | Reduce embedding/hidden dimensions |
| 🧪 Lower epochs | Set `epochs=1~3` to verify training loop |
| 📊 Add progress bar | Use `tqdm` to check if training is stuck |

---

## 📁 Data Schema

### Sequence Data (`pdb_data_seq.csv`)

| Column | Description |
|--------|-------------|
| `structureId` | Structure ID (PDB ID), e.g., `100D` |
| `chainId` | Chain ID, e.g., `A`, `B`, etc. |
| `sequence` | Protein or nucleic acid sequence (1-letter format) |
| `residueCount` | Number of residues (sequence length) |
| `macromoleculeType` | `Protein`, `DNA`, `RNA`, or `DNA/RNA Hybrid` |

### Metadata (`pdb_data_no_dups.csv`)

| Column | Description |
|--------|-------------|
| `structureId` | Unique ID for each protein structure |
| `classification` | Functional class or family label (target variable) |
| `experimentalTechnique` | Method used to determine structure (e.g., X-RAY DIFFRACTION) |
| `macromoleculeType` | Molecule type: Protein, DNA, RNA, Hybrid |
| `residueCount` | Sequence length |
| `resolution` | Structure resolution in Ångström |
| `structureMolecularWeight` | Molecular weight in Dalton |
| `crystallizationMethod` | Crystallization method used |
| `crystallizationTempK` | Temperature during crystallization (Kelvin) |
| `densityMatthews` | Matthews coefficient |
| `densityPercentSol` | Solvent content (%) |
| `pdbxDetails` | Additional experimental notes (e.g., pH, method) |
| `phValue` | pH value of crystallization condition |
| `publicationYear` | Year of structure publication |

---
## 📁 Dataset

Due to GitHub’s file size limit, the full dataset is **not included**.

We provide a lightweight subset of the original data for testing and development purposes.

To obtain the full dataset, please visit [Kaggle](https://www.kaggle.com/datasets/shahir/protein-data-set/code) or download it from the original source.
