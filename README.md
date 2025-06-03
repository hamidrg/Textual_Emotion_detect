# Text Emotion Detection Project

This project explores and implements various machine learning and deep learning models for the task of emotion detection from textual data. The primary goal is to compare the performance of different approaches on the standard ISEAR dataset.

## Project Description

Emotion detection from text is a significant area within Natural Language Processing (NLP) that focuses on identifying and categorizing emotions expressed in text (such as joy, sadness, anger, etc.). This project, inspired by academic studies and reference papers, implements and evaluates several different methods.

## Dataset Used

All implementations in this project utilize the **ISEAR (International Survey on Emotion Antecedents and Reactions)** dataset. This dataset consists of English sentences labeled with **seven main emotion categories**:
* Joy
* Fear
* Anger
* Sadness
* Disgust
* Shame
* Guilt

## Models Implemented

This repository contains implementations of the following models in Jupyter Notebook format:

### 1. Traditional Machine Learning Models
   - **LinearSVC (Linear Support Vector Classifier)**: Using TF-IDF and N-gram features.
   - **SGD Classifier (Stochastic Gradient Descent Classifier)**: Using TF-IDF and N-gram features.
     * *Preprocessing*: These models used lemmatization with `WordNetLemmatizer` from the `nltk` library and stop-word removal.

### 2. RNN-based Deep Learning Models (Recurrent Neural Networks)
   - **LSTM (Long Short-Term Memory)**
   - **BiLSTM (Bidirectional LSTM)**
   - **GRU (Gated Recurrent Unit)**
     * *Preprocessing*: These models used lemmatization with the `spacy` library, stop-word removal, tokenization with Keras `Tokenizer`, and padding. The word embedding layer was trained from scratch.

### 3. Transformer-based Deep Learning Models
   - **BERT (Bidirectional Encoder Representations from Transformers)**: Fine-tuning the pre-trained `bert-base-uncased` model.
     * *Preprocessing*: Used `spacy` and the BERT tokenizer.
   - **RoBERTa (Robustly Optimized BERT Pretraining Approach)**: Fine-tuning the pre-trained `j-hartmann/emotion-english-distilroberta-base` model, which was previously trained for emotion detection.
     * *Preprocessing*: Used `spacy` and the RoBERTa tokenizer.

## General Methodology Overview

The general workflow for each model included the following steps:
1.  **Data Loading**: Reading the ISEAR dataset.
2.  **Text Preprocessing**: Including cleaning (e.g., removing URLs and user mentions), lowercasing, tokenization, stop-word removal, and word normalization (lemmatization or stemming, depending on the model).
3.  **Feature Extraction/Tokenization & Padding**:
    * For traditional models: Converting text to numerical vectors using TF-IDF and N-grams.
    * For RNN models: Tokenization, conversion to sequences, and padding; training an Embedding layer from scratch.
    * For Transformer models: Using specific BERT and RoBERTa tokenizers and preparing inputs in the required model format.
4.  **Model Building and Training**: Defining the model architecture and training it on the training data.
5.  **Model Evaluation**: Assessing model performance on the test data using metrics like Accuracy and F1-score.

## Key Findings

* The performance of different models varied on the 7-class ISEAR dataset.
* Among the models implemented, the fine-tuned base BERT model showed the best performance (with an accuracy of approximately 64%).
* The GRU model also performed relatively well among the RNN models (with an accuracy of around 59%).
* The traditional machine learning models (LinearSVC and SGD), with the preprocessing and feature extraction used in this project, achieved lower accuracies (around 56-58%).
* Utilizing pre-trained word embeddings or Transformer models pre-trained on emotion-related tasks (like `j-hartmann/emotion-english-distilroberta-base`) has the potential to improve performance, although fine-tuning results depend on various factors.

## How to Use

Each model is implemented in a separate Jupyter Notebook. To run the codes, the necessary requirements (such as `transformers`, `torch`, `tensorflow`, `sklearn`, `spacy`, `nltk` libraries) must be installed first. You can then run each notebook individually.

## References

This project was inspired by and based on methodologies and architectures explored in the following academic papers and other related works:

[1] Pokhun, L., & Chuttur, M. Y. (2020). Emotions in texts. *Bulletin of Social Informatics Theory and Application, 4*(2), 59-69.

[2] Kusal, S., Patil, S., Choudrie, J., Kotecha, K., Vora, D., & Pappas, I. (2022). A Review on Text-Based Emotion Detection - Techniques, Applications, Datasets, and Future Directions.

[3] Yohanes, D., Putra, J. S., Filbert, K., Suryaningrum, K. M., & Saputri, H. A. (2023). Emotion Detection in Textual Data using Deep Learning. *Procedia Computer Science, 227*, 464-473.

[4] Abas, A. R., Elhenawy, I., Zidan, M., & Othman, M. (2022). BERT-CNN: A Deep Learning Model for Detecting Emotions from Text. *Computers, Materials & Continua, 71*(2), 2943-2961.

[5] Saif, W. Q. A., Alshammari, M. K., Mohammed, B. A., & Sallam, A. A. (2024). Enhancing Emotion Detection in Textual Data: A Comparative Analysis of Machine Learning Models and Feature Extraction Techniques. *Engineering, Technology & Applied Science Research, 14*(5), 16471-16477.
