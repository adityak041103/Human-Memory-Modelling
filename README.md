# 🧠 Human Memory Modelling (HMM)

This project simulates human-like memory by storing, generalizing, and decaying concepts from text input. It combines **NLP techniques**, **semantic understanding**, and **memory decay** to mimic how we remember and forget over time.

---

## 🔍 What it does

- Accepts sentences as input.
- Generalizes words using **WordNet hypernyms**.
- Stores these "supersets" with a **memory counter**.
- Applies **exponential decay** to simulate forgetting.
- Detects **emotions** in the input using Transformers.
- Converts high-repetition memory into **knowledge**.

---

## 🧱 Tech Stack

- **Python 3.8+**
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence Transformers](https://www.sbert.net/)
- [NLTK](https://www.nltk.org/) for tokenization and WordNet
- **JSON** file as local memory store

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/adityak041103/Human_Memory_Modelling.git
cd Human_Memory_Modelling

# Install dependencies
pip install -r requirements.txt
```
---
- If you don’t have a requirements.txt, install manually:
pip install transformers==4.30.2 sentence-transformers==2.2.2 huggingface_hub==0.14.1 nltk scikit-learn numpy
---

