import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np

# ----------------- Load Model and Vocab -----------------

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # use last time step
        return out

# Load vocab and config
load_model = pickle.load(open("E:/ML Project/Next Word Predictor/next_word_prediction.sav", "rb"))   

# prompt: /content/idx2word.txt and /content/word2idx.txt . I gave you two path so open both of it 

with open('E:/ML Project/Next Word Predictor/idx2word.txt', 'r' ,encoding="utf-8") as f1:
  idx2word_content = f1.read()


with open('E:/ML Project/Next Word Predictor/word2idx.txt', 'r',encoding="utf-8") as f2:
  word2idx_content = f2.read()
  
max_seq_len = 4  # max input length
# prompt: covert above file into dict

idx2word = {}
for line in idx2word_content.splitlines():
  key, value = line.split(': ')
  idx2word[int(key)] = value.strip()

word2idx = {}
for line in word2idx_content.splitlines():
  key, value = line.split(': ')
  word2idx[key.strip()] = int(value)

import torch
import torch.nn.functional as F

def preprocess_input(text, word2idx, max_seq_len):
    """
    Tokenizes and pads input text.

    Args:
    - text (str): raw input sequence
    - word2idx (dict): mapping from word to index
    - max_seq_len (int): max input length

    Returns:
    - torch.Tensor: tokenized and padded sequence
    """
    text = text.lower().strip().split()
    tokenized = [word2idx.get(word, word2idx["<unk>"]) for word in text]

    # Pad with zeros from the left
    if len(tokenized) < max_seq_len:
        tokenized = [word2idx["<pad>"]] * (max_seq_len - len(tokenized)) + tokenized
    else:
        tokenized = tokenized[-max_seq_len:]

    return torch.tensor(tokenized).unsqueeze(0)  # shape: (1, max_seq_len)
def predict_next_word_gru(input_text, model, word2idx, idx2word, max_seq_len, device="cpu"):
    """
    Predicts the next word using a trained GRU model.

    Args:
    - input_text (str): user-provided input sequence
    - model (torch.nn.Module): trained GRU model
    - word2idx (dict): word-to-index mapping
    - idx2word (dict): index-to-word mapping
    - max_seq_len (int): max input sequence length
    - device (str): 'cpu' or 'cuda'

    Returns:
    - str: predicted next word
    """

    model.eval()
    model.to(device)

    # 1. Preprocess and tokenize
    input_seq = preprocess_input(input_text, word2idx, max_seq_len).to(device)

    # 2. Run the model
    with torch.no_grad():
        output = load_model(input_seq)  # output shape: (batch_size, vocab_size)

    # 3. Pick the word with the highest score
    predicted_idx = torch.argmax(output, dim=-1).item()

    # 4. Convert index to word
    predicted_word = idx2word.get(predicted_idx, "<unk>")

    return predicted_word
# Example:
# input_text = "I like bike and what"
# predicted = predict_next_word_gru(input_text, load_model, word2idx, idx2word, max_seq_len)
# print("Predicted word:", predicted)

# ----------------- Streamlit App -----------------
st.title("Next Word Predictor")
# display the model summary
st.subheader("Model Summary")
st.write("This is a GRU-based model for next word prediction.")
st.write("The model was trained on a dataset of text sequences.")
# style the text
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f5;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write("Enter a sequence of words, and the model will predict the next word.")
# input text box style
st.markdown(
    """
    <style>
    .stTextInput {
        background-color: #fff;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
input_text = st.text_input("Input text", "I like bike and what")
if st.button("Predict"):
    if input_text:
        predicted_word = predict_next_word_gru(input_text, load_model, word2idx, idx2word, max_seq_len)
        st.write(f"Predicted next word: **{predicted_word}**")
    else:
        st.write("Please enter a valid input text.")