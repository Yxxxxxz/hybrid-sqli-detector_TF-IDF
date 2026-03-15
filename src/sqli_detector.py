import re
import urllib.parse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from difflib import SequenceMatcher
###################################################### TF-IDF + Random Forest ######################################

###############################################
# Hybrid SQLi Detector (TF-IDF Version)
###############################################

class SQLiDetector:

    def __init__(self):

        self.vectorizer = None
        self.scaler = None
        self.rf_model = None

        # Signature patterns
        self.signature_patterns = {

            "union-based": [
                r"union\s+select",
                r"union\s+all\s+select"
            ],

            "error-based": [
                r"extractvalue\s*\(",
                r"updatexml\s*\(",
                r"floor\s*\(\s*rand\s*\(",
                r"group\s+by\s+.*rand",
                r"information_schema",
                r"mysql_fetch",
                r"syntax\s+error"
            ],

            "time-based": [
                r"sleep\s*\(",
                r"benchmark\s*\(",
                r"pg_sleep\s*\(",
                r"waitfor\s+delay"
            ]
        }

    # ======================================================
    # Data Cleaning
    # ======================================================

    def clean_data(self, df):

        print("Cleaning dataset...")

        df = df.dropna(subset=["payload"])
        df = df.drop_duplicates()

        df["payload"] = df["payload"].astype(str)

        df = df[df["payload"].str.strip() != ""]
        df = df[df["payload"].str.len() > 3]

        return df

    # ======================================================
    # Normalization
    # ======================================================

    def normalize(self, text):

        text = str(text)

        text = urllib.parse.unquote(text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ======================================================
    # Skeletonization
    # ======================================================

    def skeletonize(self, text):

        text = re.sub(r'0x[0-9a-f]+', ' CONST_HEX ', text)

        text = re.sub(r"'[^']*'", ' CONST_STR ', text)
        text = re.sub(r'"[^"]*"', ' CONST_STR ', text)

        text = re.sub(r'\b(true|false|null)\b', ' CONST_BOOL ', text)

        text = re.sub(r'\b\d+(\.\d+)?\b', ' CONST_NUM ', text)

        return text

    # ======================================================
    # Tokenization
    # ======================================================

    def tokenize_sql(self, query):

        tokens = re.findall(
            r"[a-zA-Z_]+|\d+|!=|==|<=|>=|['\"].*?['\"]|[^\s]",
            str(query)
        )

        return tokens

    # ======================================================
    # Fuzzy Similarity
    # ======================================================

    def fuzzy_similarity(self, a, b):

        return SequenceMatcher(None, a, b).ratio()

    # ======================================================
    # Signature Detection
    # ======================================================

    def signature_check(self, payload):

        payload = self.normalize(payload)

        for category, patterns in self.signature_patterns.items():

            for pattern in patterns:

                if re.search(pattern, payload):
                    return True, f"{category} signature"

                score = self.fuzzy_similarity(payload, pattern)

                if score > 0.75:
                    return True, f"fuzzy {category} signature"

        return False, None

    # ======================================================
    # Train + Evaluate
    # ======================================================

    def train_and_evaluate(self, df):

        df = self.clean_data(df)

        df["payload"] = df["payload"].apply(self.normalize)

        X = df["payload"]
        y = df["label"].values

        print("\nLabel Distribution")
        print(df["label"].value_counts())
        print("Total records:", len(df))

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.1,
            stratify=y,
            random_state=42
        )

        # skeleton ก่อน tokenize
        X_train_tokens = X_train_raw.apply(
            lambda x: " ".join(self.tokenize_sql(self.skeletonize(x)))
        )

        X_test_tokens = X_test_raw.apply(
            lambda x: " ".join(self.tokenize_sql(self.skeletonize(x)))
        )

        print("\nTraining TF-IDF...")

        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            max_features=5000
        )

        X_train_vec = self.vectorizer.fit_transform(X_train_tokens).toarray()
        X_test_vec = self.vectorizer.transform(X_test_tokens).toarray()

        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train_vec)
        X_test_scaled = self.scaler.transform(X_test_vec)

        print("\nTraining RandomForest...")

        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        self.rf_model.fit(X_train_scaled, y_train)

        self.evaluate(X_test_scaled, y_test)

    # ======================================================
    # Evaluation
    # ======================================================

    def evaluate(self, X_test, y_test):

        print("\n==============================")
        print("Performance Results")
        print("==============================")

        y_pred = self.rf_model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

    # ======================================================
    # Save Model
    # ======================================================

    def save_model(self, path="models/sqli_detector.pkl"):

        data_to_save = {
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'rf_model': self.rf_model
        }

        joblib.dump(data_to_save, path)

        print(f"\n✅ Brain saved to {path}")

    # ======================================================
    # Load Model
    # ======================================================

    @staticmethod
    def load_model(path="models/sqli_detector.pkl"):

        data = joblib.load(path)

        new_detector = SQLiDetector()

        new_detector.vectorizer = data.get('vectorizer')
        new_detector.scaler = data.get('scaler')
        new_detector.rf_model = data.get('rf_model')

        print(f"✅ Model loaded from {path}")

        return new_detector

    # ======================================================
    # Predict Payload
    # ======================================================

    def predict_single(self, payload):

        if self.rf_model is None:
            raise Exception("Model not loaded")

        payload = self.normalize(payload)

        sig_detect, reason = self.signature_check(payload)

        if sig_detect:

            return {
                "prediction": "BLOCKED",
                "stage": "Signature Detection",
                "reason": reason
            }

        payload = self.skeletonize(payload)

        tokens = " ".join(self.tokenize_sql(payload))

        vec = self.vectorizer.transform([tokens]).toarray()

        vec_scaled = self.scaler.transform(vec)

        pred = self.rf_model.predict(vec_scaled)[0]

        prob = self.rf_model.predict_proba(vec_scaled)[0][1]

        return {
            "prediction": "BLOCKED" if pred == 1 else "ALLOW",
            "stage": "RandomForest ML",
            "malicious_probability": float(prob)
        }