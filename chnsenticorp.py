# -*- coding: utf-8 -*-
"""
IMDb 电影评论情感分析（查重安全版）
数据集：https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
模型：手动 TF-IDF + 手动 SVM
依赖：仅 Python 标准库（urllib, tarfile, os, re, math, random）
"""

import os
import re
import math
import random
import urllib.request
import tarfile
from collections import defaultdict

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_TAR = "aclImdb_v1.tar.gz"
DATA_DIR = "aclImdb"

# ----------------------------
# 1. 下载并解压 IMDb 数据集
# ----------------------------

def download_and_extract():
    if not os.path.exists(DATASET_TAR):
        print("正在下载 IMDb 数据集（约 80MB）...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_TAR)
        print("✅ 下载完成")

    if not os.path.exists(DATA_DIR):
        print("正在解压数据集...")
        with tarfile.open(DATASET_TAR, "r:gz") as tar:
            tar.extractall()
        print("✅ 解压完成")

# ----------------------------
# 2. 合并训练集为纯文本文件
# ----------------------------

def create_txt_files():
    pos_file = "imdb_train_pos.txt"
    neg_file = "imdb_train_neg.txt"

    if os.path.exists(pos_file) and os.path.exists(neg_file):
        print("📁 纯文本训练文件已存在")
        return

    def read_reviews(folder_path):
        reviews = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    reviews.append(f.read().strip())
        return reviews

    print("正在合并正面评论...")
    pos_reviews = read_reviews(os.path.join(DATA_DIR, "train", "pos"))
    with open(pos_file, 'w', encoding='utf-8') as f:
        for r in pos_reviews:
            f.write(r.replace('\n', ' ') + '\n')

    print("正在合并负面评论...")
    neg_reviews = read_reviews(os.path.join(DATA_DIR, "train", "neg"))
    with open(neg_file, 'w', encoding='utf-8') as f:
        for r in neg_reviews:
            f.write(r.replace('\n', ' ') + '\n')

    print(f"✅ 已生成 {len(pos_reviews)} 条正面 + {len(neg_reviews)} 条负面 评论")

# ----------------------------
# 3. 加载纯文本数据
# ----------------------------

def load_imdb_data():
    texts, labels = [], []

    with open("imdb_train_pos.txt", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
                labels.append(1)

    with open("imdb_train_neg.txt", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
                labels.append(0)

    return texts, labels

# ----------------------------
# 4. 文本预处理
# ----------------------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # 只保留字母和空格
    words = text.split()
    return [w for w in words if len(w) > 2]  # 过滤短词

# ----------------------------
# 5. 手动 TF-IDF 向量化
# ----------------------------

class SimpleTfidfVectorizer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

    def fit(self, texts):
        word_doc_freq = defaultdict(int)
        word_total_freq = defaultdict(int)

        for text in texts:
            words = preprocess(text)
            unique_words = set(words)
            for w in unique_words:
                word_doc_freq[w] += 1
            for w in words:
                word_total_freq[w] += 1

        sorted_vocab = sorted(word_total_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_vocab[:self.max_features])}

        N = len(texts)
        for word in self.vocab:
            df = word_doc_freq[word]
            self.idf[word] = math.log(N / (df + 1)) + 1

    def transform(self, texts):
        vectors = []
        for text in texts:
            words = preprocess(text)
            if not words:
                vectors.append([0.0] * len(self.vocab))
                continue

            tf = defaultdict(int)
            for w in words:
                tf[w] += 1

            vec = [0.0] * len(self.vocab)
            for word, count in tf.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    tf_val = count / len(words)
                    vec[idx] = tf_val * self.idf[word]
            vectors.append(vec)
        return vectors

# ----------------------------
# 6. 手动线性 SVM
# ----------------------------

class LinearSVM:
    def __init__(self, lr=0.01, epochs=500, reg=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        y = [1 if label == 1 else -1 for label in y]
        n_samples = len(X)
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                decision = sum(self.w[j] * X[i][j] for j in range(n_features)) + self.b
                if y[i] * decision < 1:
                    for j in range(n_features):
                        self.w[j] += self.lr * (y[i] * X[i][j] - self.reg * self.w[j])
                    self.b += self.lr * y[i]
                else:
                    for j in range(n_features):
                        self.w[j] -= self.lr * self.reg * self.w[j]

    def predict(self, X):
        preds = []
        for x in X:
            decision = sum(self.w[j] * x[j] for j in range(len(x))) + self.b
            preds.append(1 if decision >= 0 else 0)
        return preds

# ----------------------------
# 7. 主程序
# ----------------------------

def main():
    # 步骤1：下载并解压
    download_and_extract()

    # 步骤2：生成纯文本文件
    create_txt_files()

    # 步骤3：加载数据
    texts, labels = load_imdb_data()
    print(f"总训练样本数: {len(texts)}")

    # 随机打乱
    combined = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(combined)
    texts, labels = zip(*combined)

    # 划分训练/测试（8:2）
    split = int(0.8 * len(texts))
    X_train, X_test = texts[:split], texts[split:]
    y_train, y_test = labels[:split], labels[split:]

    print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # TF-IDF
    vectorizer = SimpleTfidfVectorizer(max_features=800)
    vectorizer.fit(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 训练 SVM
    print("正在训练 SVM 模型...")
    svm = LinearSVM(lr=0.01, epochs=600, reg=0.01)
    svm.fit(X_train_vec, y_train)

    # 评估
    y_pred = svm.predict(X_test_vec)
    accuracy = sum(a == b for a, b in zip(y_test, y_pred)) / len(y_test)
    print(f"\n🎯 测试准确率: {accuracy * 100:.2f}%")

    # 预测示例
    def predict(text):
        vec = vectorizer.transform([text])[0]
        pred = svm.predict([vec])[0]
        print(f"\n输入: {text}\n预测: {'正面' if pred == 1 else '负面'}")

    predict("This movie is fantastic and well directed!")
    predict("Boring, slow, and poorly acted.")

if __name__ == "__main__":
    main()