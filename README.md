🧠 Next Word Predictor using GRU
This project is a Next Word Prediction model built using PyTorch and trained on the text from Pride and Prejudice by Jane Austen. It leverages a Gated Recurrent Unit (GRU) based neural network to predict the next word in a given sentence fragment.

🚀 Features
Built using PyTorch with GRU-based architecture

Trained on classic English literature

Preprocessing includes cleaning, stopword removal, and tokenization

Saves the trained model and vocabulary for reuse

Calculates model perplexity

Simple CLI-based user input for next-word prediction

🏗️ Model Architecture
Embedding Layer: Converts word indices into dense vectors

GRU Layer: Two-layer GRU network (unidirectional) to model sequences

Fully Connected Layer: Predicts the next word from hidden state

Loss Function: CrossEntropyLoss

Optimizer: Adam

🧹 Preprocessing
Lowercased the text

Removed punctuation, numbers, and symbols

Removed English stopwords using NLTK

Tokenized into words and converted into indices

📊 Training Details
Input Size: Vocabulary size

Embedding Dimension: 128

Hidden Dimension: 256

Sequence Length: 4 input words, 1 target word

Epochs: 10

Batch Size: 64

Device: GPU/CPU (auto-detected)

📈 Evaluation
Model perplexity is calculated to evaluate how well the model predicts the sequence. A lower perplexity indicates better performance.

💾 Saved Files
next_word_prediction.sav: Trained model

word2idx.txt: Mapping from words to indices

idx2word.txt: Mapping from indices to words

📚 Acknowledgments
Jane Austen's Pride and Prejudice

PyTorch

NLTK

📄 License
This project is licensed under the MIT License.

Would you like me to also help you create the GitHub repository and push this project using Git commands or a step-by-step guide?









