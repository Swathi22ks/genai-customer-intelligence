"""
insight_generator.py
====================
LLM-powered business insight generation using LangChain + OpenAI/Anthropic.
Falls back to template-based summaries if no API key is configured.

Skills demonstrated:
 - Prompt Engineering
 - LangChain chains
 - Structured output parsing
 - RAG-style context injection
 - Gen AI for business reporting
"""

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────────────────────

REVIEW_SUMMARY_PROMPT = """
You are a senior data analyst at an e-commerce company.
Analyze the following customer review statistics and key themes, then generate
a concise executive summary with actionable recommendations.

Review Statistics:
{stats}

Top Positive Keywords: {positive_keywords}
Top Negative Keywords: {negative_keywords}

Instructions:
1. Write a 3-sentence executive summary of customer sentiment.
2. Identify the top 2 strengths from positive reviews.
3. Identify the top 2 critical issues from negative reviews.
4. Provide 3 specific, actionable recommendations.
5. Format the output clearly with headers.

Be concise, professional, and data-driven.
"""

AB_TEST_INTERPRETATION_PROMPT = """
You are a Growth Analyst interpreting an A/B experiment result.

Experiment: {experiment_name}
Control Group: {control_n} users, {control_cvr:.1%} conversion rate
Treatment Group: {treatment_n} users, {treatment_cvr:.1%} conversion rate
Relative Uplift: {uplift:.1%}
Statistical Significance: p-value = {p_value:.4f} (alpha = 0.05)
Result: {significance_label}
Estimated Monthly Revenue Impact: ₹{revenue_impact:,.0f}

Write a 2-paragraph business interpretation:
Paragraph 1: What the result means for the business.
Paragraph 2: Whether to ship the treatment and what to do next.
Be specific and business-focused. Avoid statistical jargon.
"""

CHURN_REPORT_PROMPT = """
You are a Customer Success Manager interpreting a churn prediction model result.

Model Performance:
- AUC-ROC: {auc_roc}
- Precision: {precision}
- Recall: {recall}
- F1-Score: {f1}

Top Churn Risk Factors (from SHAP):
{risk_factors}

High-Risk Segment Stats:
- High-risk customers identified: {high_risk_count}
- Estimated monthly revenue at risk: ₹{revenue_at_risk:,.0f}

Write a 3-paragraph report:
1. Model performance interpretation (non-technical language).
2. Key churn risk factors and what they mean.
3. Recommended retention interventions for the high-risk segment.
"""


# ──────────────────────────────────────────────────────────────
# FALLBACK TEMPLATE GENERATOR
# ──────────────────────────────────────────────────────────────

def _template_review_summary(stats: dict, pos_kw: list, neg_kw: list) -> str:
    """Rule-based fallback when no LLM API key is present."""
    pos_pct = stats.get("positive_pct", 0)
    neg_pct = stats.get("negative_pct", 0)
    total = stats.get("total_reviews", 0)
    pos_words = ", ".join([k for k, _ in pos_kw[:5]]) if pos_kw else "quality, delivery"
    neg_words = ", ".join([k for k, _ in neg_kw[:5]]) if neg_kw else "delay, defective"

    return f"""
## Executive Summary — Customer Review Analysis

**Overall Sentiment**: {pos_pct:.0f}% of {total:,} reviews are positive,
while {neg_pct:.0f}% are negative. Customer satisfaction is
{"strong" if pos_pct > 60 else "moderate" if pos_pct > 45 else "poor"} overall.

### Strengths Identified
- Customers frequently praise: **{pos_words}**
- Positive reviews highlight fast delivery and product quality as key drivers.

### Critical Issues
- Recurring complaints focus on: **{neg_words}**
- Negative reviews most commonly report product defects and delivery delays.

### Recommendations
1. **Quality Control**: Implement stricter pre-shipment inspection for high-return categories.
2. **Logistics**: Partner with additional courier services to improve delivery SLA.
3. **Customer Service**: Create a 24h response SLA for customers with 1-2 star reviews.

*Note: This is an automated template summary. Connect a GROQ_API_KEY for LLM-powered insights.*
"""


# ──────────────────────────────────────────────────────────────
# LLM INSIGHT GENERATOR CLASS
# ──────────────────────────────────────────────────────────────

class LLMInsightGenerator:
    """
    Generates business insights from analytics data using LLM.
    Priority order:
      1. Groq / Llama3   (free tier — GROQ_API_KEY)
      2. Google Gemini   (free tier — GOOGLE_API_KEY)
      3. OpenAI GPT-4    (OPENAI_API_KEY)
      4. Anthropic Claude (ANTHROPIC_API_KEY)
      5. Template-based fallback (no key needed)
    """

    def __init__(self):
        self.llm           = None  # LangChain LLM (OpenAI / Anthropic)
        self.gemini_client = None  # Google Gemini client
        self.groq_client   = None  # Groq client
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM — tries Groq first (free), then Gemini, then OpenAI, then Anthropic."""

        # ── 1. Groq / Llama3 (free, no quota issues) ──────────
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key != "your_groq_api_key_here":
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                logger.info("[LLM] Using Groq / Llama3 (free tier)")
                return
            except Exception as e:
                logger.warning(f"[LLM] Groq init failed: {e}")

        # ── 2. Google Gemini (free tier) ──────────────────────
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key and google_key != "your_google_api_key_here":
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=google_key)
                logger.info("[LLM] Using Google Gemini (free tier)")
                return
            except Exception as e:
                logger.warning(f"[LLM] Gemini init failed: {e}")

        # ── 3. OpenAI GPT-4 ───────────────────────────────────
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "your_openai_api_key_here":
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    api_key=openai_key,
                    temperature=0.3,
                    max_tokens=800,
                )
                logger.info("[LLM] Using OpenAI GPT-4")
                return
            except Exception as e:
                logger.warning(f"[LLM] OpenAI init failed: {e}")

        # ── 4. Anthropic Claude ───────────────────────────────
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
            try:
                from langchain_community.chat_models import ChatAnthropic
                self.llm = ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    anthropic_api_key=anthropic_key,
                    temperature=0.3,
                    max_tokens=800,
                )
                logger.info("[LLM] Using Anthropic Claude")
                return
            except Exception as e:
                logger.warning(f"[LLM] Anthropic init failed: {e}")

        logger.info("[LLM] No API key found — using template-based generation.")

    def _call_llm(self, prompt: str) -> str:
        """Call whichever LLM backend is initialised."""

        # ── Groq path ─────────────────────────────────────────
        if self.groq_client is not None:
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"[LLM] Groq call failed: {e}")
                return None

        # ── Gemini path ───────────────────────────────────────
        if self.gemini_client is not None:
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                logger.error(f"[LLM] Gemini call failed: {e}")
                return None

        # ── LangChain path (OpenAI / Anthropic) ──────────────
        if self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                logger.error(f"[LLM] API call failed: {e}")
                return None

        return None

    def generate_review_insights(self,
                                  stats: dict,
                                  positive_keywords: list,
                                  negative_keywords: list) -> str:
        prompt = REVIEW_SUMMARY_PROMPT.format(
            stats=json.dumps(stats, indent=2),
            positive_keywords=", ".join([k for k, _ in positive_keywords[:8]]),
            negative_keywords=", ".join([k for k, _ in negative_keywords[:8]]),
        )
        result = self._call_llm(prompt)
        if result:
            return result
        return _template_review_summary(stats, positive_keywords, negative_keywords)

    def interpret_ab_result(self, experiment_data: dict) -> str:
        prompt = AB_TEST_INTERPRETATION_PROMPT.format(**experiment_data)
        result = self._call_llm(prompt)
        if result:
            return result
        sig    = experiment_data.get("significance_label", "NOT SIGNIFICANT")
        uplift = experiment_data.get("uplift", 0)
        return (
            f"## A/B Test Interpretation\n\n"
            f"The experiment shows a **{uplift:.1%} relative uplift** in conversion rate. "
            f"The result is **{sig}** at the 95% confidence level. "
            f"{'Recommend shipping the treatment.' if 'SIGNIFICANT' in sig else 'Recommend continuing the test or iterating.'}\n\n"
            f"*Connect a GROQ_API_KEY for LLM-powered insights.*"
        )

    def generate_churn_report(self, model_metrics: dict,
                               risk_factors: list,
                               high_risk_count: int,
                               revenue_at_risk: float) -> str:
        factors_str = "\n".join(
            [f"  - {f}: importance={s:.4f}" for f, s in risk_factors[:6]]
        )
        prompt = CHURN_REPORT_PROMPT.format(
            auc_roc=model_metrics.get("auc_roc", "N/A"),
            precision=model_metrics.get("precision", "N/A"),
            recall=model_metrics.get("recall", "N/A"),
            f1=model_metrics.get("f1", "N/A"),
            risk_factors=factors_str,
            high_risk_count=high_risk_count,
            revenue_at_risk=revenue_at_risk,
        )
        result = self._call_llm(prompt)
        if result:
            return result
        return (
            f"## Churn Prediction Report\n\n"
            f"**Model AUC-ROC**: {model_metrics.get('auc_roc', 'N/A')}\n\n"
            f"**High-Risk Customers**: {high_risk_count:,}\n\n"
            f"**Revenue at Risk**: ₹{revenue_at_risk:,.0f}/month\n\n"
            f"**Top Risk Factors**: {', '.join([f for f, _ in risk_factors[:3]])}\n\n"
            f"*Connect a GROQ_API_KEY for LLM-powered insights.*"
        )

    def answer_question(self, question: str, context: str) -> str:
        prompt = (
            f"You are a data analyst. Using the following data context, "
            f"answer the question clearly and concisely.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        result = self._call_llm(prompt)
        return result or "Please configure a GROQ_API_KEY in your .env file to enable Q&A."


# ──────────────────────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generator = LLMInsightGenerator()

    stats = {
        "total_reviews": 8000,
        "positive_pct": 55.2,
        "negative_pct": 20.3,
        "neutral_pct": 24.5,
        "avg_rating": 3.8,
    }
    pos_kw = [("excellent quality", 0.42), ("fast delivery", 0.38),
              ("great value", 0.31), ("highly recommend", 0.28)]
    neg_kw = [("broke quickly", 0.45), ("poor quality", 0.40),
              ("damaged product", 0.35), ("slow delivery", 0.29)]

    print("=" * 60)
    print("  LLM INSIGHT GENERATOR — DEMO")
    print("=" * 60)
    print(generator.generate_review_insights(stats, pos_kw, neg_kw))
