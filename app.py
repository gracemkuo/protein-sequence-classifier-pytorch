import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd

# --- Load model class definition (simplified here) ---
class ProteinClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(21, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# --- Amino acid to integer encoding ---
amino_acids = 'ARNDCQEGHILKMFPSTWYV'
aa_dict = {aa: i + 1 for i, aa in enumerate(amino_acids)}  # Padding is 0

def encode_seq(seq, max_len=256):
    seq = seq[:max_len].ljust(max_len, 'X')
    return [aa_dict.get(aa, 0) for aa in seq]

# --- Load model ---
@st.cache_resource
def load_model():
    checkpoint = torch.load("models/best_protein_model.pth", map_location="cpu")
    label_dict = checkpoint["label_dict"]
    inv_label_dict = {v: k for k, v in label_dict.items()}
    model = ProteinClassifier(num_classes=len(label_dict))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, inv_label_dict, label_dict

model, inv_label_dict, label_dict = load_model()

# --- Optional: load a reference dataset to pick test samples ---
@st.cache_data
def load_sample_data():
    df = pd.read_csv("data/pdb_data_seq.csv")
    meta = pd.read_csv("data/pdb_data_no_dups.csv")
    df = pd.merge(df, meta[["structureId", "classification"]], on="structureId")
    top_classes = df["classification"].value_counts().nlargest(10).index
    df["label"] = df["classification"].apply(lambda x: x if x in top_classes else "others")
    return df[df['label'] != 'others']

# --- Streamlit UI ---
st.title("🔬 Protein Function Classifier")
st.markdown("Enter an amino acid sequence (A, R, N, D, ...), and predict its functional classification with confidence scores.")

# --- Sample Sequence Loader ---
sample_df = load_sample_data()
if st.button("🔁 Load a non-'others' example sequence"):
    row = sample_df.sample(1).iloc[0]
    st.session_state['example_seq'] = row['sequence']
    st.session_state['example_label'] = row['label']

user_input = st.text_input("Enter amino acid sequence:", value=st.session_state.get('example_seq', "ARNDCEQGHILKMFPSTWYV"))

if 'example_label' in st.session_state:
    st.info(f"📌 Original class of the example: **{st.session_state['example_label']}**")

if st.button("🚀 Predict"):
    encoded = encode_seq(user_input)
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).numpy().flatten()
        pred_class = logits.argmax(dim=1).item()
        pred_label = inv_label_dict[pred_class]

        st.success(f"✅ Predicted Class: **{pred_label}**")

        st.subheader("🔎 Prediction Confidence")
        for i, prob in enumerate(probs):
            label = inv_label_dict[i]
            st.write(f"{label:15s} : {prob:.4f}")
