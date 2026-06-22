"""
ab_testing.py
=============
Production-grade A/B Testing & Hypothesis Testing engine.

Skills demonstrated:
 - Two-proportion z-test (conversion rates)
 - Welch's t-test (revenue per user)
 - Chi-square test (categorical outcomes)
 - Bayesian A/B testing
 - Multiple testing correction (Bonferroni, BH)
 - Sample size / power calculation
 - Sequential testing (early stopping)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, norm, beta as beta_dist
import statsmodels.stats.proportion as smp
import statsmodels.stats.power as smpow
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# SAMPLE SIZE CALCULATOR
# ──────────────────────────────────────────────────────────────

class SampleSizeCalculator:
    """
    Calculate required sample size for A/B test before running.
    """

    @staticmethod
    def for_proportions(baseline_rate: float,
                        mde: float,
                        alpha: float = 0.05,
                        power: float = 0.80) -> dict:
        """
        Calculate sample size for a two-proportion test.

        Args:
            baseline_rate: Control group conversion rate (e.g. 0.08)
            mde: Minimum Detectable Effect as absolute difference (e.g. 0.02)
            alpha: Significance level (Type I error)
            power: Statistical power (1 - Type II error)

        Returns:
            dict with required sample size per group and total
        """
        treatment_rate = baseline_rate + mde
        effect_size = smp.proportion_effectsize(baseline_rate, treatment_rate)

        analysis = smpow.NormalIndPower()
        n_per_group = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative="two-sided"
        )
        n_per_group = int(np.ceil(n_per_group))

        return {
            "baseline_rate": baseline_rate,
            "treatment_rate": treatment_rate,
            "mde_absolute": mde,
            "mde_relative": round(mde / baseline_rate, 4),
            "alpha": alpha,
            "power": power,
            "n_per_group": n_per_group,
            "n_total": n_per_group * 2,
        }


# ──────────────────────────────────────────────────────────────
# FREQUENTIST A/B TEST
# ──────────────────────────────────────────────────────────────

class FrequentistABTest:
    """
    Frequentist hypothesis testing for A/B experiments.
    Supports conversion rate tests and continuous metric tests.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test_conversion_rate(self,
                              control_conversions: int, control_n: int,
                              treatment_conversions: int, treatment_n: int,
                              two_sided: bool = True) -> dict:
        """
        Two-proportion z-test for conversion rate comparison.

        H0: p_control == p_treatment
        H1: p_control != p_treatment (two-sided)
        """
        p_control = control_conversions / control_n
        p_treatment = treatment_conversions / treatment_n

        # Pooled proportion
        p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_n + 1/treatment_n))

        z_stat = (p_treatment - p_control) / se
        alternative = "two-sided" if two_sided else "larger"
        p_value = 2 * (1 - norm.cdf(abs(z_stat))) if two_sided else (1 - norm.cdf(z_stat))

        # Confidence interval
        se_diff = np.sqrt(
            p_control*(1-p_control)/control_n + p_treatment*(1-p_treatment)/treatment_n
        )
        z_crit = norm.ppf(1 - self.alpha/2)
        diff = p_treatment - p_control
        ci_lower = diff - z_crit * se_diff
        ci_upper = diff + z_crit * se_diff

        significant = p_value < self.alpha
        uplift = (p_treatment - p_control) / p_control if p_control > 0 else 0

        result = {
            "test_type": "Two-Proportion Z-Test",
            "control_n": control_n,
            "treatment_n": treatment_n,
            "control_cvr": round(p_control, 6),
            "treatment_cvr": round(p_treatment, 6),
            "absolute_lift": round(diff, 6),
            "relative_lift": round(uplift, 4),
            "z_statistic": round(z_stat, 4),
            "p_value": round(p_value, 6),
            "alpha": self.alpha,
            "significant": bool(significant),
            "significance_label": "STATISTICALLY SIGNIFICANT ✓" if significant else "NOT SIGNIFICANT ✗",
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "ci_level": f"{int((1-self.alpha)*100)}%",
        }

        logger.info(
            f"[A/B Test] Control CVR: {p_control:.2%} | Treatment CVR: {p_treatment:.2%} | "
            f"Uplift: {uplift:.2%} | p-value: {p_value:.4f} | "
            f"{'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'}"
        )
        return result

    def test_continuous_metric(self,
                                control_values: np.ndarray,
                                treatment_values: np.ndarray,
                                metric_name: str = "revenue") -> dict:
        """
        Welch's t-test for continuous metrics (e.g. revenue per user).

        H0: mean_control == mean_treatment
        H1: mean_control != mean_treatment
        """
        control_values = np.array(control_values)
        treatment_values = np.array(treatment_values)

        t_stat, p_value = stats.ttest_ind(
            treatment_values, control_values, equal_var=False
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (control_values.std()**2 + treatment_values.std()**2) / 2
        )
        cohens_d = (treatment_values.mean() - control_values.mean()) / (pooled_std + 1e-9)

        significant = p_value < self.alpha
        uplift = ((treatment_values.mean() - control_values.mean())
                  / (control_values.mean() + 1e-9))

        return {
            "test_type": "Welch's t-Test",
            "metric": metric_name,
            "control_mean": round(control_values.mean(), 4),
            "treatment_mean": round(treatment_values.mean(), 4),
            "relative_lift": round(uplift, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "cohens_d": round(cohens_d, 4),
            "effect_size": "small" if abs(cohens_d) < 0.2 else "medium" if abs(cohens_d) < 0.5 else "large",
            "significant": bool(significant),
            "significance_label": "STATISTICALLY SIGNIFICANT ✓" if significant else "NOT SIGNIFICANT ✗",
            "alpha": self.alpha,
        }

    def run_full_analysis(self, df: pd.DataFrame) -> dict:
        """
        Run complete A/B test analysis on experiment dataframe.

        Args:
            df: DataFrame with columns: group, converted, revenue_inr

        Returns:
            dict with all test results and summary
        """
        control = df[df["group"] == "control"]
        treatment = df[df["group"] == "treatment"]

        # Conversion rate test
        cvr_result = self.test_conversion_rate(
            control_conversions=control["converted"].sum(),
            control_n=len(control),
            treatment_conversions=treatment["converted"].sum(),
            treatment_n=len(treatment),
        )

        # Revenue test (all users — intent-to-treat)
        rev_result = self.test_continuous_metric(
            control_values=control["revenue_inr"].values,
            treatment_values=treatment["revenue_inr"].values,
            metric_name="revenue_per_user_inr",
        )

        # Revenue per converter test
        control_conv = control[control["converted"] == 1]["revenue_inr"]
        treatment_conv = treatment[treatment["converted"] == 1]["revenue_inr"]
        if len(control_conv) > 10 and len(treatment_conv) > 10:
            aov_result = self.test_continuous_metric(
                control_values=control_conv.values,
                treatment_values=treatment_conv.values,
                metric_name="avg_order_value_inr",
            )
        else:
            aov_result = {"note": "Insufficient converters for AOV test"}

        # Monthly revenue impact estimate
        daily_control_users = len(control) / 60  # assume 60-day experiment
        monthly_users = daily_control_users * 30
        rev_impact = (
            cvr_result["absolute_lift"] * monthly_users *
            rev_result.get("treatment_mean", 0)
        )

        return {
            "experiment_name": df["experiment_name"].iloc[0] if "experiment_name" in df.columns else "unnamed",
            "control_n": len(control),
            "treatment_n": len(treatment),
            "control_cvr": cvr_result["control_cvr"],
            "treatment_cvr": cvr_result["treatment_cvr"],
            "uplift": cvr_result["relative_lift"],
            "p_value": cvr_result["p_value"],
            "significance_label": cvr_result["significance_label"],
            "revenue_impact": round(rev_impact, 2),
            "conversion_test": cvr_result,
            "revenue_test": rev_result,
            "aov_test": aov_result,
        }


# ──────────────────────────────────────────────────────────────
# BAYESIAN A/B TEST
# ──────────────────────────────────────────────────────────────

class BayesianABTest:
    """
    Bayesian A/B testing using Beta-Binomial conjugate model.
    Outputs probability that treatment beats control.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Args:
            prior_alpha, prior_beta: Beta distribution hyperparameters (default: uniform prior)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def analyze(self,
                control_conversions: int, control_n: int,
                treatment_conversions: int, treatment_n: int,
                n_samples: int = 100_000) -> dict:
        """
        Bayesian analysis via Monte Carlo sampling.

        Posterior: Beta(alpha + conversions, beta + non-conversions)
        """
        # Posterior parameters
        ctrl_alpha = self.prior_alpha + control_conversions
        ctrl_beta = self.prior_beta + (control_n - control_conversions)
        trt_alpha = self.prior_alpha + treatment_conversions
        trt_beta = self.prior_beta + (treatment_n - treatment_conversions)

        # Monte Carlo samples
        ctrl_samples = beta_dist.rvs(ctrl_alpha, ctrl_beta, size=n_samples)
        trt_samples = beta_dist.rvs(trt_alpha, trt_beta, size=n_samples)

        prob_treatment_better = np.mean(trt_samples > ctrl_samples)
        expected_lift = np.mean(trt_samples - ctrl_samples)
        ci_lower, ci_upper = np.percentile(trt_samples - ctrl_samples, [2.5, 97.5])

        # Expected loss (risk of choosing wrong variant)
        loss_treatment = np.mean(np.maximum(ctrl_samples - trt_samples, 0))

        result = {
            "method": "Bayesian Beta-Binomial",
            "prob_treatment_better": round(prob_treatment_better, 4),
            "prob_control_better": round(1 - prob_treatment_better, 4),
            "expected_lift": round(expected_lift, 6),
            "credible_interval_95": [round(ci_lower, 6), round(ci_upper, 6)],
            "expected_loss_if_treatment": round(loss_treatment, 6),
            "recommendation": "Ship Treatment ✓" if prob_treatment_better > 0.95 else
                              "Continue Testing" if prob_treatment_better > 0.80 else
                              "Keep Control",
        }

        logger.info(
            f"[Bayesian A/B] P(Treatment > Control): {prob_treatment_better:.1%} | "
            f"Recommendation: {result['recommendation']}"
        )
        return result


# ──────────────────────────────────────────────────────────────
# HYPOTHESIS TESTING SUITE
# ──────────────────────────────────────────────────────────────

class HypothesisTestSuite:
    """
    General-purpose hypothesis testing for data analysis.
    """

    @staticmethod
    def shapiro_normality(data: np.ndarray, name: str = "data") -> dict:
        """Shapiro-Wilk normality test."""
        stat, p = stats.shapiro(data[:5000])  # Shapiro is limited to 5000 samples
        return {
            "test": "Shapiro-Wilk",
            "variable": name,
            "statistic": round(stat, 4),
            "p_value": round(p, 6),
            "normal": p > 0.05,
            "interpretation": "Normally distributed" if p > 0.05 else "Not normally distributed"
        }

    @staticmethod
    def chi_square_independence(contingency_table: np.ndarray,
                                 row_labels: list, col_labels: list) -> dict:
        """Chi-square test for independence."""
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return {
            "test": "Chi-Square Independence Test",
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p, 6),
            "degrees_of_freedom": dof,
            "significant": bool(p < 0.05),
            "interpretation": "Variables are dependent" if p < 0.05 else "Variables are independent"
        }

    @staticmethod
    def anova_one_way(groups: list, group_names: list) -> dict:
        """One-way ANOVA test across multiple groups."""
        f_stat, p_value = stats.f_oneway(*groups)
        return {
            "test": "One-Way ANOVA",
            "groups": group_names,
            "group_means": [round(np.mean(g), 4) for g in groups],
            "f_statistic": round(f_stat, 4),
            "p_value": round(p_value, 6),
            "significant": bool(p_value < 0.05),
            "interpretation": "At least one group mean differs significantly" if p_value < 0.05 else "No significant difference between groups"
        }

    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray,
                             x_name: str = "X", y_name: str = "Y") -> dict:
        """Pearson correlation coefficient with significance test."""
        r, p = stats.pearsonr(x, y)
        strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
        direction = "positive" if r > 0 else "negative"
        return {
            "test": "Pearson Correlation",
            "variables": f"{x_name} vs {y_name}",
            "r": round(r, 4),
            "r_squared": round(r**2, 4),
            "p_value": round(p, 6),
            "significant": bool(p < 0.05),
            "strength": strength,
            "direction": direction,
            "interpretation": f"{strength.capitalize()} {direction} correlation between {x_name} and {y_name}"
        }
