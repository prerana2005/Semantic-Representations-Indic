# Indic-NLP-Explorer: Deep Semantic Representations & NLP for Hindi and Kannada

This repository contains an end-to-end exploration into capturing semantic meaning and performing high-level NLP tasks specifically for morphologically rich Indian languages (Hindi and Kannada). 

The project focuses on the transition from discrete text processing to continuous vector representations, evaluating everything from foundational "from-scratch" implementations to state-of-the-art Transformer architectures.

## 🚀 Overview

The goal was to bridge the "meaning gap" in vernacular data (using movie reviews and cultural reports as a dataset) by implementing and evaluating diverse embedding techniques. The project culminates in deploying specialized LLM pipelines for translation, summarization, and cross-lingual intelligence.

## 🛠️ Features & Implementations

### Part 1: Foundations of Vector Semantics
Implemented foundational algorithms from scratch to understand the mathematical underpinnings of language:
*   **Vocabulary Engineering:** Automated word-to-index mapping for Unicode-based scripts.
*   **Word2Vec (Skip-Gram & CBOW):** Custom PyTorch implementations with **Negative Sampling** to learn local word associations.
*   **GloVe (Global Vectors):** Construction of a weighted Global Co-occurrence Matrix to capture thematic relationships across the entire corpus.

### Part 2: Handling Morphological Complexity
Indian languages are highly agglutinative. This section explores advanced methods to handle complexity:
*   **FastText (Sub-word Embeddings):** Leveraging character n-grams to synthesize vectors for **Out-of-Vocabulary (OOV)** words and truncated roots.
*   **Contextualized Embeddings (IndicBERT):** Implementation of dynamic embeddings where word vectors shift mathematically based on the surrounding sentence context (Self-Attention).

### Part 3: Advanced NLP Task Pipelines
Deployed pre-trained SOTA models to solve real-world language challenges:
*   **Neural Machine Translation (NMT):** Benchmarking **Google Translate** against **NLLB-200 (1.3B)** with custom chunking wrappers to handle long-form cultural text.
*   **Multi-Paradigm Summarization:** Comparative analysis of **Extractive (LexRank)** vs. **Abstractive (BART-XSum & DistilBART)** summarization.
*   **Zero-Shot Classification:** Categorizing unseen text into dynamic labels (Action, Mythology, Finance) using **Natural Language Inference (NLI)**.
*   **Cross-Lingual Extractive QA:** Building a bridge to answer English queries based on Hindi/Kannada contexts using **RoBERTa-Base-SQuAD2**.

## 📊 Key Observations
*   **Static vs. Contextual:** Observed how BERT resolves polysemy (words with multiple meanings) compared to the fixed dictionary approach of GloVe.
*   **Extractive vs. Generative:** Analyzed the precision of Extractive QA models in factual retrieval versus the conversational flexibility of Generative models.
*   **Cultural Nuance:** Identified specific instances where NLLB-200 captured cultural terms like *"Daiva"* more precisely than standard statistical tools.

## 💻 Tech Stack
*   **Languages:** Python (3.10+)
*   **Frameworks:** PyTorch, Hugging Face Transformers
*   **Libraries:** Gensim, Indic-NLP-Library, Deep-Translator, Sumy, NLTK
*   **Environment:** Google Colab T4 GPU
