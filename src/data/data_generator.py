"""
data_generator.py
=================
Generates a realistic synthetic e-commerce dataset for the platform.
Covers: customer reviews, purchase events, A/B experiment logs, churn labels.

Run: python src/data/data_generator.py
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
import os
import json
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
N_CUSTOMERS = 5_000
N_REVIEWS = 8_000
N_AB_EVENTS = 12_000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")

PRODUCT_CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen",
    "Sports", "Books", "Beauty", "Toys", "Grocery"
]

POSITIVE_REVIEWS = [
    "Absolutely love this product! Works perfectly and arrived on time.",
    "Exceeded my expectations. Great quality for the price.",
    "Very satisfied with the purchase. Fast delivery and excellent packaging.",
    "Five stars! This is exactly what I needed. Highly recommended.",
    "Outstanding product. The build quality is superb and it works flawlessly.",
    "Best purchase I've made this year. Would definitely buy again.",
    "Fantastic quality. My family loves it. Prompt delivery too!",
    "Impressed by the quality and attention to detail. Worth every rupee.",
    "Smooth transaction, product as described. Will shop again.",
    "Brilliant product! Setup was easy and the performance is excellent.",
]

NEGATIVE_REVIEWS = [
    "Completely disappointed. Product stopped working after two days.",
    "Terrible quality. Nothing like what was shown in the pictures.",
    "Waste of money. The item arrived damaged and customer support was unhelpful.",
    "Do not buy this! Broke on first use. Returning immediately.",
    "Very poor quality. Expected much better for this price.",
    "Product looks nothing like the description. Misleading listing.",
    "Horrible experience. Delivery was two weeks late and item was defective.",
    "Cheap material, falls apart easily. Would not recommend to anyone.",
    "Extremely disappointed. The product is fake and not original.",
    "Bad experience overall. Poor quality and terrible after-sales support.",
]

NEUTRAL_REVIEWS = [
    "Product is okay. Nothing special but does the job.",
    "Average quality. Could be better for the price.",
    "It works as described. Delivery was slightly delayed.",
    "Decent product. Some features are good, others not so much.",
    "Not bad but not great either. Just about meets expectations.",
    "Mediocre product. Satisfactory for occasional use.",
    "Item received in good condition. Quality is acceptable.",
    "Works fine for basic use. Would not recommend for heavy use.",
    "Standard product. Neither impressed nor disappointed.",
    "Okay for the price. Don't expect premium quality.",
]


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def random_date(start_days_ago=730, end_days_ago=0):
    """Return a random datetime within the given window."""
    days_offset = random.randint(end_days_ago, start_days_ago)
    return datetime.now() - timedelta(days=days_offset)


def generate_customers(n=N_CUSTOMERS):
    """Generate synthetic customer profiles."""
    print(f"[INFO] Generating {n} customer records...")
    data = []
    for _ in range(n):
        signup_date = random_date(730, 30)
        tenure_days = (datetime.now() - signup_date).days
        monthly_spend = max(0, np.random.normal(2500, 1200))
        orders = max(1, int(tenure_days / 30 * random.uniform(0.5, 2.5)))

        # Churn probability influenced by spend, orders, tenure
        churn_score = (
            0.4 * (1 - min(monthly_spend / 5000, 1)) +
            0.3 * (1 - min(orders / 20, 1)) +
            0.3 * (1 - min(tenure_days / 730, 1))
        )
        churned = int(random.random() < churn_score * 0.6)

        data.append({
            "customer_id": f"CUST{_:05d}",
            "name": fake.name(),
            "email": fake.email(),
            "city": fake.city(),
            "state": random.choice([
                "Karnataka", "Maharashtra", "Tamil Nadu", "Delhi",
                "Telangana", "Gujarat", "Rajasthan", "West Bengal"
            ]),
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "tenure_days": tenure_days,
            "monthly_spend_inr": round(monthly_spend, 2),
            "total_orders": orders,
            "avg_order_value": round(monthly_spend / max(orders / 12, 1), 2),
            "support_tickets": random.randint(0, 8),
            "last_login_days_ago": random.randint(1, 120),
            "email_open_rate": round(random.uniform(0.0, 1.0), 3),
            "app_sessions_monthly": random.randint(0, 50),
            "preferred_category": random.choice(PRODUCT_CATEGORIES),
            "payment_method": random.choice([
                "UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"
            ]),
            "churned": churned,
        })

    return pd.DataFrame(data)


def generate_reviews(customers_df, n=N_REVIEWS):
    """Generate synthetic product reviews with sentiment labels."""
    print(f"[INFO] Generating {n} review records...")
    customer_ids = customers_df["customer_id"].tolist()
    data = []

    for i in range(n):
        sentiment_roll = random.random()
        if sentiment_roll < 0.55:
            sentiment = "positive"
            rating = random.randint(4, 5)
            review_text = random.choice(POSITIVE_REVIEWS)
        elif sentiment_roll < 0.80:
            sentiment = "negative"
            rating = random.randint(1, 2)
            review_text = random.choice(NEGATIVE_REVIEWS)
        else:
            sentiment = "neutral"
            rating = 3
            review_text = random.choice(NEUTRAL_REVIEWS)

        review_date = random_date(365, 0)
        data.append({
            "review_id": f"REV{i:06d}",
            "customer_id": random.choice(customer_ids),
            "product_id": f"PROD{random.randint(1, 500):04d}",
            "category": random.choice(PRODUCT_CATEGORIES),
            "rating": rating,
            "review_text": review_text,
            "sentiment_label": sentiment,  # Ground truth for evaluation
            "review_date": review_date.strftime("%Y-%m-%d"),
            "helpful_votes": random.randint(0, 150),
            "verified_purchase": random.choice([True, False]),
        })

    return pd.DataFrame(data)


def generate_ab_experiment(customers_df, n=N_AB_EVENTS):
    """
    Generate A/B experiment data.
    Experiment: Does showing personalized recommendations (Treatment)
                increase conversion vs generic recommendations (Control)?
    """
    print(f"[INFO] Generating {n} A/B experiment event records...")
    customer_ids = customers_df["customer_id"].tolist()
    data = []

    # True uplift: treatment increases conversion by ~3.5%
    CONTROL_CVR = 0.08
    TREATMENT_CVR = 0.115

    for i in range(n):
        group = "treatment" if random.random() < 0.5 else "control"
        cvr = TREATMENT_CVR if group == "treatment" else CONTROL_CVR
        converted = int(random.random() < cvr)
        revenue = round(np.random.lognormal(7.0, 0.8), 2) if converted else 0.0

        data.append({
            "event_id": f"EVT{i:07d}",
            "customer_id": random.choice(customer_ids),
            "experiment_name": "personalized_recommendations_v2",
            "group": group,
            "timestamp": random_date(60, 0).strftime("%Y-%m-%d %H:%M:%S"),
            "page_views": random.randint(1, 15),
            "time_on_page_sec": round(np.random.exponential(45), 1),
            "converted": converted,
            "revenue_inr": revenue,
            "device": random.choice(["mobile", "desktop", "tablet"]),
            "source": random.choice([
                "organic", "paid_search", "email", "social", "direct"
            ]),
        })

    return pd.DataFrame(data)


def generate_feature_store(customers_df):
    """Create ML-ready feature store from customer data."""
    print("[INFO] Generating ML feature store...")
    df = customers_df.copy()

    # Engineered features
    df["spend_per_order"] = df["monthly_spend_inr"] / (df["total_orders"] + 1)
    df["tickets_per_order"] = df["support_tickets"] / (df["total_orders"] + 1)
    df["recency_score"] = 1 / (df["last_login_days_ago"] + 1)
    df["engagement_score"] = (
        df["email_open_rate"] * 0.3 +
        df["app_sessions_monthly"].clip(0, 50) / 50 * 0.4 +
        (1 - df["last_login_days_ago"].clip(0, 120) / 120) * 0.3
    )
    df["high_value_flag"] = (df["monthly_spend_inr"] > 3000).astype(int)
    df["tenure_segment"] = pd.cut(
        df["tenure_days"],
        bins=[0, 90, 180, 365, 730, float("inf")],
        labels=["new", "growing", "established", "loyal", "champion"]
    )

    return df


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    customers = generate_customers()
    reviews = generate_reviews(customers)
    ab_events = generate_ab_experiment(customers)
    features = generate_feature_store(customers)

    customers.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
    reviews.to_csv(f"{OUTPUT_DIR}/reviews.csv", index=False)
    ab_events.to_csv(f"{OUTPUT_DIR}/ab_experiment_events.csv", index=False)
    features.to_csv(f"{OUTPUT_DIR}/feature_store.csv", index=False)

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "customers": len(customers),
        "reviews": len(reviews),
        "ab_events": len(ab_events),
        "churn_rate": round(customers["churned"].mean(), 4),
        "review_sentiment_dist": reviews["sentiment_label"].value_counts().to_dict(),
        "ab_control_cvr": round(ab_events[ab_events.group == "control"]["converted"].mean(), 4),
        "ab_treatment_cvr": round(ab_events[ab_events.group == "treatment"]["converted"].mean(), 4),
    }
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Data generation complete!")
    print(f"   📁 Output directory : {os.path.abspath(OUTPUT_DIR)}")
    print(f"   👥 Customers        : {meta['customers']:,}")
    print(f"   📝 Reviews          : {meta['reviews']:,}")
    print(f"   🧪 A/B Events       : {meta['ab_events']:,}")
    print(f"   📉 Churn Rate       : {meta['churn_rate']:.1%}")
    print(f"   🔬 Control CVR      : {meta['ab_control_cvr']:.1%}")
    print(f"   🚀 Treatment CVR    : {meta['ab_treatment_cvr']:.1%}")


if __name__ == "__main__":
    main()
