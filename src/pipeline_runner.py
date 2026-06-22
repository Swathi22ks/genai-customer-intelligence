"""
pipeline_runner.py
==================
Master orchestrator for the GenAI Customer Intelligence Platform.
Runs the full end-to-end pipeline:
  1. Data Generation
  2. NLP Sentiment Analysis
  3. Deep Learning Churn Model
  4. A/B Testing & Hypothesis Testing
  5. LLM Insight Generation
  6. Export Results

Run: python src/pipeline_runner.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR    = os.path.join(os.path.dirname(__file__), "../data/raw")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "../outputs")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "../models")


def banner(title: str):
    width = 60
    logger.info("=" * width)
    logger.info(f"  {title}")
    logger.info("=" * width)


# ──────────────────────────────────────────────────────────────
# STEP 1 — DATA GENERATION
# ──────────────────────────────────────────────────────────────

def step_generate_data():
    banner("STEP 1: DATA GENERATION")
    from src.data.data_generator import main as generate
    generate()


# ──────────────────────────────────────────────────────────────
# STEP 2 — NLP SENTIMENT ANALYSIS
# ──────────────────────────────────────────────────────────────

def step_nlp_analysis() -> dict:
    banner("STEP 2: NLP SENTIMENT ANALYSIS")
    from src.data.preprocessor import ReviewDataPreprocessor
    from src.nlp.sentiment_analyzer import SentimentAnalyzer, KeywordExtractor

    preprocessor = ReviewDataPreprocessor()
    df_raw = preprocessor.load()
    df = preprocessor.preprocess(df_raw)

    logger.info(f"[NLP] Analyzing {len(df):,} reviews...")

    analyzer = SentimentAnalyzer(use_transformers=False)
    df = analyzer.analyze_dataframe(df, text_col="review_text")
    metrics = analyzer.evaluate(df, true_col="sentiment_label", pred_col="predicted_sentiment")

    logger.info(f"[NLP] Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")

    # Keyword extraction
    extractor = KeywordExtractor()
    keywords = extractor.keywords_by_sentiment(df)

    # Sentiment distribution
    sentiment_counts = df["predicted_sentiment"].value_counts()
    total = len(df)
    stats = {
        "total_reviews": total,
        "positive_pct": round(sentiment_counts.get("positive", 0) / total * 100, 1),
        "negative_pct": round(sentiment_counts.get("negative", 0) / total * 100, 1),
        "neutral_pct": round(sentiment_counts.get("neutral", 0) / total * 100, 1),
        "avg_rating": round(df["rating"].mean(), 2),
    }

    # Save enriched reviews
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/reviews_with_sentiment.csv", index=False)
    logger.info(f"[NLP] Saved enriched reviews → outputs/reviews_with_sentiment.csv")

    return {
        "nlp_metrics": metrics,
        "nlp_stats": stats,
        "keywords": keywords,
    }


# ──────────────────────────────────────────────────────────────
# STEP 3 — DEEP LEARNING CHURN MODEL
# ──────────────────────────────────────────────────────────────

def step_churn_model() -> dict:
    banner("STEP 3: DEEP LEARNING CHURN MODEL")
    from src.data.preprocessor import CustomerDataPreprocessor
    from src.deep_learning.churn_model import ChurnModelTrainer, compute_feature_importance_shap

    preprocessor = CustomerDataPreprocessor()
    df = preprocessor.load()
    X, y, feature_names = preprocessor.preprocess(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_and_scale(X, y)

    logger.info(f"[DL] Input dim: {X_train.shape[1]} | "
                f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    logger.info(f"[DL] Churn rate: {y_train.mean():.2%}")

    trainer = ChurnModelTrainer(
        input_dim=X_train.shape[1],
        lr=1e-3,
        epochs=25,
        batch_size=256,
    )

    history = trainer.train(X_train, y_train, X_val, y_val, use_mlflow=False)
    test_metrics = trainer.evaluate(X_test, y_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    trainer.save(f"{MODELS_DIR}/churn_model.pt")

    # Feature importance (SHAP or gradient proxy)
    importance = compute_feature_importance_shap(trainer, X_test, feature_names)

    logger.info("[DL] Top 5 churn risk factors:")
    for feat, score in importance[:5]:
        logger.info(f"    {feat}: {score:.4f}")

    # High-risk customers
    probs = trainer.predict_proba(X_test)
    high_risk_count = int((probs >= 0.7).sum())
    avg_spend = df["monthly_spend_inr"].mean()
    revenue_at_risk = high_risk_count * avg_spend

    # Save results
    results_df = pd.DataFrame({
        "churn_probability": probs,
        "predicted_class": (probs >= 0.5).astype(int),
        "risk_level": pd.cut(probs, bins=[0, 0.4, 0.7, 1.0],
                             labels=["Low", "Medium", "High"]),
        "true_label": y_test,
    })
    results_df.to_csv(f"{OUTPUT_DIR}/churn_predictions.csv", index=False)
    logger.info("[DL] Predictions saved → outputs/churn_predictions.csv")

    return {
        "dl_metrics": test_metrics,
        "feature_importance": importance,
        "high_risk_count": high_risk_count,
        "revenue_at_risk": round(revenue_at_risk, 2),
    }


# ──────────────────────────────────────────────────────────────
# STEP 4 — A/B TESTING & HYPOTHESIS TESTING
# ──────────────────────────────────────────────────────────────

def step_ab_testing() -> dict:
    banner("STEP 4: A/B TESTING & HYPOTHESIS TESTING")
    from src.statistics.ab_testing import (
        FrequentistABTest, BayesianABTest,
        HypothesisTestSuite, SampleSizeCalculator
    )

    df = pd.read_csv(f"{DATA_DIR}/ab_experiment_events.csv")
    logger.info(f"[A/B] Loaded {len(df):,} events | "
                f"Control: {(df.group=='control').sum():,} | "
                f"Treatment: {(df.group=='treatment').sum():,}")

    # Sample size check
    calc = SampleSizeCalculator()
    required = calc.for_proportions(
        baseline_rate=0.08, mde=0.03, alpha=0.05, power=0.80
    )
    logger.info(f"[A/B] Required sample size: {required['n_per_group']:,} per group "
                f"(have {(df.group=='control').sum():,})")

    # Frequentist test
    freq_test = FrequentistABTest(alpha=0.05)
    full_result = freq_test.run_full_analysis(df)

    logger.info(
        f"[A/B Frequentist] Control CVR: {full_result['control_cvr']:.2%} | "
        f"Treatment CVR: {full_result['treatment_cvr']:.2%} | "
        f"Uplift: {full_result['uplift']:.2%} | "
        f"p-value: {full_result['p_value']:.4f} | "
        f"{full_result['significance_label']}"
    )

    # Bayesian test
    control = df[df.group == "control"]
    treatment = df[df.group == "treatment"]
    bayes_test = BayesianABTest()
    bayes_result = bayes_test.analyze(
        control_conversions=control["converted"].sum(),
        control_n=len(control),
        treatment_conversions=treatment["converted"].sum(),
        treatment_n=len(treatment),
    )
    logger.info(
        f"[A/B Bayesian] P(Treatment > Control): {bayes_result['prob_treatment_better']:.1%} | "
        f"Recommendation: {bayes_result['recommendation']}"
    )

    # Additional hypothesis tests
    ht = HypothesisTestSuite()

    # ANOVA: revenue by device type
    device_groups = [
        df[df.device == d]["revenue_inr"].values
        for d in df["device"].unique() if len(df[df.device == d]) > 30
    ]
    device_names = [
        d for d in df["device"].unique() if len(df[df.device == d]) > 30
    ]
    anova_result = ht.anova_one_way(device_groups, device_names)
    logger.info(f"[HT] ANOVA (device vs revenue): {anova_result['interpretation']}")

    # Correlation: page views vs revenue
    corr_result = ht.pearson_correlation(
        df["page_views"].values,
        df["revenue_inr"].values,
        "page_views", "revenue_inr"
    )
    logger.info(f"[HT] Correlation: {corr_result['interpretation']}")

    # Save results
    def make_serializable(obj, _seen=None):
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            return None
        if isinstance(obj, dict):
            _seen.add(obj_id)
            return {k: make_serializable(v, _seen) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            _seen.add(obj_id)
            return [make_serializable(v, _seen) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    with open(f"{OUTPUT_DIR}/ab_test_results.json", "w") as f:
        json.dump(make_serializable({
            "frequentist": full_result,
            "bayesian": bayes_result,
            "anova_device": anova_result,
            "correlation": corr_result,
        }), f, indent=2)
    logger.info("[A/B] Results saved → outputs/ab_test_results.json")

    return {
        "ab_result": full_result,
        "bayesian_result": bayes_result,
    }


# ──────────────────────────────────────────────────────────────
# STEP 5 — LLM INSIGHT GENERATION
# ──────────────────────────────────────────────────────────────

def step_llm_insights(nlp_output: dict, dl_output: dict, ab_output: dict):
    banner("STEP 5: LLM INSIGHT GENERATION")
    from src.llm.insight_generator import LLMInsightGenerator

    generator = LLMInsightGenerator()

    # Review insights
    review_insight = generator.generate_review_insights(
        stats=nlp_output["nlp_stats"],
        positive_keywords=nlp_output["keywords"].get("positive", []),
        negative_keywords=nlp_output["keywords"].get("negative", []),
    )

    # A/B test interpretation
    ab_data = ab_output["ab_result"].copy()
    ab_data["experiment_name"] = ab_data.get("experiment_name", "personalized_recommendations_v2")
    ab_insight = generator.interpret_ab_result(ab_data)

    # Churn report
    churn_insight = generator.generate_churn_report(
        model_metrics={
            "auc_roc": dl_output["dl_metrics"]["auc_roc"],
            "precision": dl_output["dl_metrics"]["precision"],
            "recall": dl_output["dl_metrics"]["recall"],
            "f1": dl_output["dl_metrics"]["f1"],
        },
        risk_factors=dl_output["feature_importance"][:8],
        high_risk_count=dl_output["high_risk_count"],
        revenue_at_risk=dl_output["revenue_at_risk"],
    )

    # Save insights report
    report = f"""# GenAI Customer Intelligence — Executive Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Customer Review Analysis

{review_insight}

---

## 2. A/B Experiment Interpretation

{ab_insight}

---

## 3. Churn Prediction Analysis

{churn_insight}

---

## 4. Key Metrics Summary

| Metric | Value |
|--------|-------|
| NLP Accuracy | {nlp_output['nlp_metrics']['accuracy']:.4f} |
| NLP Macro F1 | {nlp_output['nlp_metrics']['macro_f1']:.4f} |
| Churn Model AUC-ROC | {dl_output['dl_metrics']['auc_roc']} |
| A/B Test Result | {ab_output['ab_result']['significance_label']} |
| Revenue Impact (est.) | ₹{ab_output['ab_result']['revenue_impact']:,.0f}/month |
| High-Risk Customers | {dl_output['high_risk_count']:,} |
| Revenue at Risk | ₹{dl_output['revenue_at_risk']:,.0f}/month |
"""

    with open(f"{OUTPUT_DIR}/executive_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("[LLM] Executive report saved → outputs/executive_report.md")
    return report


# ──────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now()
    banner("GenAI Customer Intelligence Platform — Full Pipeline")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Check if data exists
    if not os.path.exists(f"{DATA_DIR}/customers.csv"):
        step_generate_data()
    else:
        logger.info("[SKIP] Data already exists. Delete data/raw/ to regenerate.")

    nlp_output = step_nlp_analysis()
    dl_output  = step_churn_model()
    ab_output  = step_ab_testing()
    step_llm_insights(nlp_output, dl_output, ab_output)

    elapsed = (datetime.now() - start_time).seconds
    banner("PIPELINE COMPLETE")
    logger.info(f"✅ All steps completed in {elapsed}s")
    logger.info(f"📁 Results saved to: {os.path.abspath(OUTPUT_DIR)}")
    logger.info("🚀 Run the dashboard: streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()
