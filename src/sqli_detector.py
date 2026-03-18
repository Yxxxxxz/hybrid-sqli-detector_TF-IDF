import re
import urllib.parse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


######################################################
# SQLi Detector (FULL PIPELINE)
######################################################

class SQLiDetector:

    def __init__(self):

        self.vectorizer = None
        self.rf_model = None

        # Signature rules
        self.signature_patterns = {
            "union-based": [
                r"union\s+select",
                r"union\s+all\s+select"
            ],
            "error-based": [
                r"extractvalue\s*\(",
                r"updatexml\s*\(",
                r"information_schema",
                r"syntax\s+error"
            ],
            "time-based": [
                r"sleep\s*\(",
                r"benchmark\s*\(",
                r"pg_sleep\s*\(",
                r"waitfor\s+delay"
            ]
        }

    # =========================
    # LOAD MODEL
    # =========================
    @staticmethod
    def load(path="sqli_detector.pkl"):
        data = joblib.load(path)

        model = SQLiDetector()
        model.vectorizer = data["vectorizer"]
        model.rf_model = data["rf_model"]

        print("✅ Loaded model:", path)
        return model

    # =========================
    # AUTO DETECT COLUMN
    # =========================
    def detect_columns(self, df):

        df.columns = df.columns.str.strip()

        text_candidates = ["text", "payload", "query", "input"]
        label_candidates = ["label", "is_sqli", "target"]

        text_col = None
        label_col = None

        for c in text_candidates:
            if c in df.columns:
                text_col = c
                break

        for c in label_candidates:
            if c in df.columns:
                label_col = c
                break

        if text_col is None:
            raise Exception(f"❌ ไม่เจอ text column ใน {df.columns}")

        if label_col is None:
            raise Exception(f"❌ ไม่เจอ label column ใน {df.columns}")

        print(f"✅ Use text column: {text_col}")
        print(f"✅ Use label column: {label_col}")

        return text_col, label_col

    # =========================
    # TRAIN FROM CSV
    # =========================
    def train_from_csv(self, csv_path, model_path="sqli_detector.pkl"):

        print("📂 Loading dataset...")

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except:
            df = pd.read_csv(csv_path, encoding="latin-1")

        print("Columns:", df.columns.tolist())

        text_col, label_col = self.detect_columns(df)

        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()

        print("Total samples:", len(texts))
        print("Label distribution:\n", pd.Series(labels).value_counts())

        # =========================
        # PREPROCESS
        # =========================
        print("\n🔧 Preprocessing...")
        X_clean = [self.preprocess(t) for t in texts]

        # =========================
        # SPLIT 90/10
        # =========================
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, labels,
            test_size=0.1,
            random_state=42,
            stratify=labels
        )

        print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

        # =========================
        # TF-IDF
        # =========================
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3)
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # =========================
        # MODEL
        # =========================
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"   # 🔥 สำคัญมาก
        )

        print("\n🚀 Training model...")
        self.rf_model.fit(X_train_vec, y_train)

        # =========================
        # EVALUATE
        # =========================
        print("\n📊 Evaluating...")

        y_pred = self.rf_model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n=== METRICS ===")
        print("Accuracy :", acc)
        print("F1-score :", f1)

        print("\n=== CONFUSION MATRIX ===")
        print(confusion_matrix(y_test, y_pred))

        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_test, y_pred, digits=4))

        # =========================
        # SAVE MODEL
        # =========================
        print("\n💾 Saving model...")
        joblib.dump({
            "vectorizer": self.vectorizer,
            "rf_model": self.rf_model
        }, model_path)

        print("✅ Model saved:", model_path)

    # =========================
    # NORMALIZE
    # =========================
    def normalize(self, text):
        text = str(text)
        text = urllib.parse.unquote(text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # =========================
    # SKELETON
    # =========================
    def skeletonize(self, text):
        text = re.sub(r'0x[0-9a-fA-F]+', ' CONST_HEX ', text)
        text = re.sub(r"'[^']*'", ' CONST_STR ', text)
        text = re.sub(r'"[^"]*"', ' CONST_STR ', text)
        text = re.sub(r'\b(true|false|null)\b', ' CONST_BOOL ', text)
        text = re.sub(r'\b\d+(\.\d+)?\b', ' CONST_NUM ', text)
        return text

    # =========================
    # TOKENIZE
    # =========================
    def tokenize_sql(self, query):
        tokens = re.findall(
            r"[a-zA-Z0-9_]+|!=|==|<=|>=|--|/\*|\*/|[(),=*<>]|[^\s]",
            str(query)
        )
        return tokens if tokens else ["UNK"]

    # =========================
    # PREPROCESS
    # =========================
    def preprocess(self, text):
        text = self.normalize(text)
        text = self.skeletonize(text)
        tokens = self.tokenize_sql(text)
        return " ".join(tokens)

    # =========================
    # SIGNATURE CHECK
    # =========================
    def signature_check(self, payload):

        payload = self.normalize(payload)

        for category, patterns in self.signature_patterns.items():
            for pattern in patterns:
                if re.search(pattern, payload):
                    return True, category

        return False, None

    # =========================
    # PREDICT
    # =========================
    def predict(self, text):

        is_sqli, reason = self.signature_check(text)

        if is_sqli:
            return 1, 1.0, f"BLOCKED ({reason})"

        clean = self.preprocess(text)
        vec = self.vectorizer.transform([clean])

        pred = self.rf_model.predict(vec)[0]
        prob = self.rf_model.predict_proba(vec)[0][1]

        return pred, prob, "ML"
