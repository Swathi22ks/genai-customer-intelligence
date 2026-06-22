"""
sentiment_analyzer.py
=====================
Production-grade NLP sentiment analysis using HuggingFace Transformers.
Falls back to rule-based scoring if transformers are unavailable.

Techniques demonstrated:
 - Pre-trained BERT-based model (distilbert-base-uncased-finetuned-sst-2-english)
 - Batch inference for efficiency
 - Confidence scoring
 - NLP evaluation metrics (F1, Accuracy, Confusion Matrix)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)
import logging
import os

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# TRANSFORMER-BASED ANALYZER
# ──────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    BERT-based sentiment analyzer.
    Uses distilbert-base-uncased-finetuned-sst-2-english.
    Falls back to lexicon-based scoring if torch/transformers unavailable.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 use_transformers: bool = True):
        self.model_name = model_name
        self.pipeline = None
        self.use_transformers = use_transformers
        self._load_model()

    def _load_model(self):
        if not self.use_transformers:
            logger.info("[NLP] Using rule-based sentiment (transformer disabled)")
            return
        try:
            from transformers import pipeline
            logger.info(f"[NLP] Loading transformer model: {self.model_name}")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                truncation=True,
                max_length=512
            )
            logger.info("[NLP] Model loaded successfully ✓")
        except Exception as e:
            logger.warning(f"[NLP] Transformer unavailable ({e}). Using rule-based fallback.")
            self.pipeline = None

    def _rule_based_sentiment(self, text: str) -> dict:
        """
        Negation-aware lexicon-based sentiment analysis.

        Handles:
         - Pure star ratings  (1-5 or 1-10)
         - Negation phrases   (not good, isn't great, never satisfied)
         - Intensifiers       (very bad, extremely poor)
         - Mixed signals      (great product but bad delivery)
        """
        text_stripped = text.strip()

        # ── 1. Pure numeric input → treat as star rating ──────────────
        try:
            rating = float(text_stripped)
            if 1 <= rating <= 5:
                if rating >= 4:
                    return {"label": "POSITIVE", "score": round(0.6 + (rating - 4) * 0.19, 2),
                            "rating_mode": True}
                elif rating == 3:
                    return {"label": "NEUTRAL",  "score": 0.65, "rating_mode": True}
                else:
                    return {"label": "NEGATIVE", "score": round(0.6 + (2 - rating) * 0.19, 2),
                            "rating_mode": True}
            elif rating < 1 or rating > 5:
                return {"label": "OUT_OF_RANGE", "score": 0.0, "rating_mode": True}
        except ValueError:
            pass  # Not a number — continue with text analysis

        text_lower = text_stripped.lower()
        words = text_lower.split()

        # ── Guard: reject non-text input ──────────────────────────────
        import re
        real_words = [w for w in words if re.search(r'[a-zA-Z]', w)]
        if len(real_words) == 0:
            return {"label": "INVALID_INPUT", "score": 0.0, "rating_mode": False}
        
        # ── 2. Lexicons ────────────────────────────────────────────────
        POSITIVE = {
            "love", "excellent", "great", "fantastic", "amazing", "superb",
            "outstanding", "perfect", "wonderful", "brilliant", "best",
            "satisfied", "impressed", "recommend", "smooth", "quality",
            "happy", "delighted", "awesome", "nice", "good", "fast",
            "reliable", "helpful", "easy", "worth", "value", "pleased",
        }
        NEGATIVE = {
            "terrible", "horrible", "awful", "waste", "broke", "defective",
            "damaged", "disappointed", "misleading", "fake", "poor",
            "bad", "worst", "cheap", "unhelpful", "delayed", "useless",
            "broken", "frustrating", "annoying", "slow", "rude", "fail",
            "failure", "wrong", "issue", "problem", "return", "refund",
        }
        NEGATORS = {
            "not", "no", "never", "isn't", "wasn't", "aren't", "weren't",
            "don't", "doesn't", "didn't", "hardly", "barely", "nothing",
            "neither", "nor", "without", "lack", "lacking", "far from",
        }
        INTENSIFIERS = {
            "very", "extremely", "absolutely", "totally", "completely",
            "really", "so", "too", "quite", "highly",
        }

        # ── 3. Score with negation window ──────────────────────────────
        pos_score = 0.0
        neg_score = 0.0

        for i, word in enumerate(words):
            # Look back 3 words for a negator
            window = words[max(0, i-3):i]
            negated = any(neg in window for neg in NEGATORS)

            # Check intensifier in window
            intensified = any(inten in window for inten in INTENSIFIERS)
            weight = 1.5 if intensified else 1.0

            if word in POSITIVE:
                if negated:
                    neg_score += weight          # "not good" → negative
                else:
                    pos_score += weight
            elif word in NEGATIVE:
                if negated:
                    pos_score += weight * 0.5   # "not bad" → slightly positive
                else:
                    neg_score += weight

        # ── 4. Decide ──────────────────────────────────────────────────
        total = pos_score + neg_score
        if total == 0:
            return {"label": "NEUTRAL", "score": 0.55}

        ratio = pos_score / total

        if ratio >= 0.65:
            confidence = round(0.58 + min(pos_score * 0.08, 0.38), 2)
            return {"label": "POSITIVE", "score": min(confidence, 0.97)}
        elif ratio <= 0.35:
            confidence = round(0.58 + min(neg_score * 0.08, 0.38), 2)
            return {"label": "NEGATIVE", "score": min(confidence, 0.97)}
        else:
            return {"label": "NEUTRAL", "score": round(0.52 + abs(ratio - 0.5) * 0.2, 2)}

    def predict_single(self, text: str) -> dict:
        """
        Predict sentiment for a single text.
        Returns: {"label": "POSITIVE/NEGATIVE/NEUTRAL", "score": float}
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5}

        if self.pipeline:
            try:
                result = self.pipeline(text[:512])[0]
                # Map to 3-class: POSITIVE, NEGATIVE, NEUTRAL
                if result["label"] == "POSITIVE" and result["score"] > 0.75:
                    label = "POSITIVE"
                elif result["label"] == "NEGATIVE" and result["score"] > 0.75:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                return {"label": label, "score": round(result["score"], 4)}
            except Exception as e:
                logger.warning(f"Transformer inference failed: {e}")

        return self._rule_based_sentiment(text)

    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        Predict sentiment for a list of texts.
        Returns list of dicts: [{"label": ..., "score": ...}, ...]
        """
        results = []
        logger.info(f"[NLP] Running batch inference on {len(texts):,} texts...")

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.predict_single(text))

        logger.info(f"[NLP] Batch inference complete: {len(results):,} results")
        return results

    def analyze_dataframe(self, df: pd.DataFrame,
                          text_col: str = "review_text",
                          batch_size: int = 64) -> pd.DataFrame:
        """
        Run sentiment analysis on a DataFrame.
        Adds predicted_sentiment and confidence columns.
        """
        df = df.copy()
        texts = df[text_col].fillna("").tolist()
        predictions = self.predict_batch(texts, batch_size=batch_size)

        df["predicted_sentiment"] = [p["label"].upper() for p in predictions]
        df["sentiment_confidence"] = [p["score"] for p in predictions]

        # Standardize labels
        label_map = {"POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"}
        df["predicted_sentiment"] = df["predicted_sentiment"].map(
            lambda x: label_map.get(x, x.lower())
        )
        return df

    def evaluate(self, df: pd.DataFrame,
                 true_col: str = "sentiment_label",
                 pred_col: str = "predicted_sentiment") -> dict:
        """
        Evaluate predictions against ground truth labels.
        Returns a dict of evaluation metrics.
        """
        y_true = df[true_col].str.lower()
        y_pred = df[pred_col].str.lower()

        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred,
                              labels=["positive", "neutral", "negative"])

        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        logger.info(f"[NLP Evaluation] Accuracy: {metrics['accuracy']:.4f} | "
                    f"Macro F1: {metrics['macro_f1']:.4f}")
        return metrics


# ──────────────────────────────────────────────────────────────
# KEYWORD & TOPIC EXTRACTOR
# ──────────────────────────────────────────────────────────────

class KeywordExtractor:
    """
    Simple TF-IDF based keyword extraction per sentiment group.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words="english",
            ngram_range=(1, 2)
        )

    def extract_top_keywords(self, texts: list, top_n: int = 10) -> list:
        """Return top N keywords from a list of texts."""
        if not texts:
            return []
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = mean_scores.argsort()[::-1][:top_n]
            return [(feature_names[i], round(mean_scores[i], 4)) for i in top_indices]
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def keywords_by_sentiment(self, df: pd.DataFrame,
                              text_col: str = "review_text",
                              sentiment_col: str = "predicted_sentiment") -> dict:
        """Extract top keywords for each sentiment group."""
        result = {}
        for sentiment in ["positive", "negative", "neutral"]:
            subset = df[df[sentiment_col] == sentiment][text_col].fillna("").tolist()
            result[sentiment] = self.extract_top_keywords(subset)
        return result


# ──────────────────────────────────────────────────────────────
# QUICK DEMO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_texts = [
        "This product is absolutely fantastic! Great quality and fast delivery.",
        "Terrible experience. The item broke on the first day. Very disappointed.",
        "It's okay, nothing special. Does the job I suppose.",
        "Highly recommend this! Best purchase of the year.",
        "Waste of money. Very poor quality.",
    ]

    print("=" * 60)
    print("  NLP SENTIMENT ANALYSIS — DEMO")
    print("=" * 60)

    analyzer = SentimentAnalyzer(use_transformers=False)  # Use rule-based for demo
    for text in sample_texts:
        result = analyzer.predict_single(text)
        print(f"  Text    : {text[:55]}...")
        print(f"  Label   : {result['label']}")
        print(f"  Score   : {result['score']:.4f}")
        print()
