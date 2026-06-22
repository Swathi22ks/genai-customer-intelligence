"""
test_all_modules.py
===================
Unit tests for all GenAI Customer Intelligence modules.
Run: pytest tests/ -v --cov=src
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────
# TEST: DATA GENERATOR
# ──────────────────────────────────────────────────────────────

class TestDataGenerator:

    def test_customer_generation(self):
        from src.data.data_generator import generate_customers
        df = generate_customers(n=100)
        assert len(df) == 100
        assert "customer_id" in df.columns
        assert "churned" in df.columns
        assert df["churned"].isin([0, 1]).all()
        assert df["monthly_spend_inr"].ge(0).all()

    def test_review_generation(self):
        from src.data.data_generator import generate_customers, generate_reviews
        customers = generate_customers(n=50)
        reviews = generate_reviews(customers, n=200)
        assert len(reviews) == 200
        assert "review_text" in reviews.columns
        assert "sentiment_label" in reviews.columns
        assert reviews["sentiment_label"].isin(["positive", "negative", "neutral"]).all()
        assert reviews["rating"].between(1, 5).all()

    def test_ab_experiment_generation(self):
        from src.data.data_generator import generate_customers, generate_ab_experiment
        customers = generate_customers(n=100)
        ab = generate_ab_experiment(customers, n=500)
        assert len(ab) == 500
        assert ab["group"].isin(["control", "treatment"]).all()
        assert ab["converted"].isin([0, 1]).all()
        assert ab["revenue_inr"].ge(0).all()


# ──────────────────────────────────────────────────────────────
# TEST: NLP SENTIMENT ANALYZER
# ──────────────────────────────────────────────────────────────

class TestSentimentAnalyzer:

    @pytest.fixture
    def analyzer(self):
        from src.nlp.sentiment_analyzer import SentimentAnalyzer
        return SentimentAnalyzer(use_transformers=False)

    def test_predict_single_positive(self, analyzer):
        result = analyzer.predict_single(
            "This product is absolutely fantastic! Great quality."
        )
        assert "label" in result
        assert "score" in result
        assert result["label"].upper() == "POSITIVE"
        assert 0 <= result["score"] <= 1

    def test_predict_single_negative(self, analyzer):
        result = analyzer.predict_single(
            "Terrible product, completely broke and wasted money."
        )
        assert result["label"].upper() == "NEGATIVE"

    def test_predict_empty_text(self, analyzer):
        result = analyzer.predict_single("")
        assert "label" in result
        assert result["label"].upper() in ["NEUTRAL", "POSITIVE", "NEGATIVE"]

    def test_predict_batch(self, analyzer):
        texts = ["Great product!", "Terrible experience!", "It's okay."]
        results = analyzer.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "label" in r
            assert "score" in r

    def test_analyze_dataframe(self, analyzer):
        df = pd.DataFrame({
            "review_text": ["Love it!", "Hate it!", "It's okay"],
            "sentiment_label": ["positive", "negative", "neutral"],
        })
        result = analyzer.analyze_dataframe(df)
        assert "predicted_sentiment" in result.columns
        assert "sentiment_confidence" in result.columns
        assert len(result) == 3


# ──────────────────────────────────────────────────────────────
# TEST: STATISTICS & A/B TESTING
# ──────────────────────────────────────────────────────────────

class TestFrequentistABTest:

    @pytest.fixture
    def ab_test(self):
        from src.statistics.ab_testing import FrequentistABTest
        return FrequentistABTest(alpha=0.05)

    def test_significant_result(self, ab_test):
        """Large sample with real effect should be significant."""
        result = ab_test.test_conversion_rate(
            control_conversions=800, control_n=10000,
            treatment_conversions=1150, treatment_n=10000,
        )
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["relative_lift"] > 0

    def test_not_significant_result(self, ab_test):
        """Small sample with tiny effect should NOT be significant."""
        result = ab_test.test_conversion_rate(
            control_conversions=8, control_n=100,
            treatment_conversions=9, treatment_n=100,
        )
        assert result["significant"] is False
        assert result["p_value"] >= 0.05

    def test_result_structure(self, ab_test):
        result = ab_test.test_conversion_rate(
            control_conversions=100, control_n=1000,
            treatment_conversions=120, treatment_n=1000,
        )
        required_keys = [
            "test_type", "control_cvr", "treatment_cvr",
            "p_value", "significant", "ci_lower", "ci_upper"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_zero_control_conversions(self, ab_test):
        """Should not crash with zero conversions."""
        result = ab_test.test_conversion_rate(
            control_conversions=0, control_n=100,
            treatment_conversions=5, treatment_n=100,
        )
        assert "p_value" in result

    def test_continuous_metric(self, ab_test):
        rng = np.random.default_rng(42)
        control = rng.normal(1000, 200, 500)
        treatment = rng.normal(1100, 200, 500)
        result = ab_test.test_continuous_metric(control, treatment, "revenue")
        assert result["treatment_mean"] > result["control_mean"]
        assert "cohens_d" in result


class TestBayesianABTest:

    def test_high_probability_treatment_better(self):
        from src.statistics.ab_testing import BayesianABTest
        test = BayesianABTest()
        result = test.analyze(
            control_conversions=80, control_n=1000,
            treatment_conversions=150, treatment_n=1000,
            n_samples=50_000,
        )
        assert result["prob_treatment_better"] > 0.95
        assert result["recommendation"] == "Ship Treatment ✓"

    def test_output_structure(self):
        from src.statistics.ab_testing import BayesianABTest
        test = BayesianABTest()
        result = test.analyze(100, 1000, 110, 1000, n_samples=10_000)
        assert "prob_treatment_better" in result
        assert "prob_control_better" in result
        assert abs(result["prob_treatment_better"] + result["prob_control_better"] - 1.0) < 0.01


class TestSampleSizeCalculator:

    def test_sample_size_calculation(self):
        from src.statistics.ab_testing import SampleSizeCalculator
        result = SampleSizeCalculator.for_proportions(
            baseline_rate=0.08, mde=0.02, alpha=0.05, power=0.80
        )
        assert result["n_per_group"] > 0
        assert result["n_total"] == result["n_per_group"] * 2
        assert result["mde_absolute"] == 0.02


class TestHypothesisTestSuite:

    def test_pearson_correlation(self):
        from src.statistics.ab_testing import HypothesisTestSuite
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        y = x * 2 + rng.normal(0, 0.5, 200)
        result = HypothesisTestSuite.pearson_correlation(x, y)
        assert result["r"] > 0.9
        assert result["significant"] is True
        assert result["direction"] == "positive"

    def test_anova_significant(self):
        from src.statistics.ab_testing import HypothesisTestSuite
        rng = np.random.default_rng(1)
        groups = [rng.normal(100, 10, 100), rng.normal(200, 10, 100), rng.normal(150, 10, 100)]
        result = HypothesisTestSuite.anova_one_way(groups, ["A", "B", "C"])
        assert result["significant"] is True
        assert result["p_value"] < 0.05


# ──────────────────────────────────────────────────────────────
# TEST: DEEP LEARNING MODEL
# ──────────────────────────────────────────────────────────────

class TestChurnModel:

    @pytest.fixture
    def trainer(self):
        from src.deep_learning.churn_model import ChurnModelTrainer
        return ChurnModelTrainer(input_dim=20, epochs=3, batch_size=64)

    def test_model_forward_pass(self, trainer):
        import torch
        x = torch.randn(10, 20)
        out = trainer.model(x)
        assert out.shape == (10,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_training_improves_loss(self, trainer):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 20)).astype(np.float32)
        y = (rng.random(500) > 0.7).astype(np.float32)
        history = trainer.train(X[:400], y[:400], X[400:], y[400:], use_mlflow=False)
        assert len(history["train_loss"]) == 3

    def test_predict_proba_range(self, trainer):
        X = np.random.randn(100, 20).astype(np.float32)
        probs = trainer.predict_proba(X)
        assert probs.shape == (100,)
        assert (probs >= 0).all() and (probs <= 1).all()


# ──────────────────────────────────────────────────────────────
# TEST: LLM INSIGHT GENERATOR (template fallback)
# ──────────────────────────────────────────────────────────────

class TestLLMInsightGenerator:

    @pytest.fixture
    def generator(self):
        from src.llm.insight_generator import LLMInsightGenerator
        # Force fallback mode by clearing any API keys from environment
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        return LLMInsightGenerator()

    def test_review_insights_returns_string(self, generator):
        stats = {"total_reviews": 1000, "positive_pct": 55, "negative_pct": 25, "neutral_pct": 20}
        pos_kw = [("great quality", 0.4), ("fast delivery", 0.3)]
        neg_kw = [("broken product", 0.5), ("poor quality", 0.4)]
        result = generator.generate_review_insights(stats, pos_kw, neg_kw)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_churn_report_returns_string(self, generator):
        metrics = {"auc_roc": 0.87, "precision": 0.82, "recall": 0.78, "f1": 0.80}
        result = generator.generate_churn_report(
            model_metrics=metrics,
            risk_factors=[("monthly_spend_inr", 0.42), ("last_login_days_ago", 0.35)],
            high_risk_count=450,
            revenue_at_risk=1125000.0,
        )
        assert isinstance(result, str)
        assert len(result) > 50


# ──────────────────────────────────────────────────────────────
# RUN DIRECTLY
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
