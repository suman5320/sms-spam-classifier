# 📱 SMS Spam Detector

This project is a simple yet effective machine learning application that classifies SMS messages as **Spam** or **Ham** (Not Spam). It uses Natural Language Processing (NLP) techniques and a machine learning classifier to learn patterns from text data.

---

## 🔍 Project Overview

The goal is to develop a supervised learning model that can identify and filter out spam messages from a dataset of SMS messages. The notebook walks through data preprocessing, vectorization, model training, evaluation, and deployment-ready pipeline creation.

---

## 🚀 Features

- Text preprocessing: lowercase conversion, punctuation removal, stopword removal, stemming
- Feature extraction using **TF-IDF Vectorizer**
- Model training using **Naive Bayes Classifier**
- Performance metrics: Accuracy, Confusion Matrix
- User input support for live prediction

---

## 📁 Dataset

The dataset used is the classic [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which consists of 5,572 labeled messages (spam or ham).

---

## 🛠️ Installation

Clone the repository and install required libraries:

```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector
pip install -r requirements.txt
