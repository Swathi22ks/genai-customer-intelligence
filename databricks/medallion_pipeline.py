# Databricks notebook source
# ============================================================
# GenAI Customer Intelligence — Medallion Architecture Pipeline
# ============================================================
# Upload this file to your Databricks workspace and run as a job.
# Architecture: Bronze (raw) → Silver (cleaned) → Gold (features)
# ============================================================

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🥉 BRONZE LAYER — Raw Data Ingestion

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, current_timestamp, lit,
    when, isnan, count, mean, stddev, regexp_replace, trim
)
from pyspark.sql.types import DoubleType, IntegerType
from delta.tables import DeltaTable
import logging

# Initialize Spark (already available in Databricks — this is for local testing)
try:
    spark = SparkSession.getActiveSession()
    print("✓ Using existing Spark session")
except Exception:
    spark = SparkSession.builder \
        .appName("GenAI_Customer_Intelligence") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    print("✓ New Spark session created")

# COMMAND ----------
# Storage paths — update to your Databricks/ADLS/S3 paths
BASE_PATH = "/mnt/genai_customer_intelligence"   # Change to your mount point
BRONZE_PATH = f"{BASE_PATH}/bronze"
SILVER_PATH = f"{BASE_PATH}/silver"
GOLD_PATH   = f"{BASE_PATH}/gold"

# Source data paths
SOURCE_CUSTOMERS = f"{BASE_PATH}/source/customers.csv"
SOURCE_REVIEWS   = f"{BASE_PATH}/source/reviews.csv"
SOURCE_AB_EVENTS = f"{BASE_PATH}/source/ab_experiment_events.csv"

# COMMAND ----------
# MAGIC %md
# MAGIC ### Bronze: Ingest raw CSV data as-is with ingestion timestamp

def ingest_bronze(source_path: str, bronze_path: str, table_name: str):
    """Read source CSV and write to Bronze Delta table (raw, unmodified)."""
    print(f"[BRONZE] Ingesting {table_name}...")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(source_path)
    df = df.withColumn("_ingested_at", current_timestamp()) \
           .withColumn("_source_file", lit(source_path))

    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
      .save(f"{bronze_path}/{table_name}")

    count = df.count()
    print(f"  ✓ {table_name}: {count:,} records → {bronze_path}/{table_name}")
    return df

# Ingest all sources
bronze_customers = ingest_bronze(SOURCE_CUSTOMERS, BRONZE_PATH, "customers")
bronze_reviews   = ingest_bronze(SOURCE_REVIEWS,   BRONZE_PATH, "reviews")
bronze_ab_events = ingest_bronze(SOURCE_AB_EVENTS, BRONZE_PATH, "ab_events")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🥈 SILVER LAYER — Cleaned & Validated Data

def clean_customers_silver(bronze_df):
    """
    Silver transformation for customers:
    - Drop nulls, standardize types, validate ranges
    - Add data quality flags
    """
    print("[SILVER] Cleaning customers...")

    df = bronze_df \
        .dropDuplicates(["customer_id"]) \
        .filter(col("customer_id").isNotNull()) \
        .withColumn("monthly_spend_inr",
                    when(col("monthly_spend_inr") < 0, lit(0.0))
                    .otherwise(col("monthly_spend_inr").cast(DoubleType()))) \
        .withColumn("total_orders",
                    when(col("total_orders") < 0, lit(0))
                    .otherwise(col("total_orders").cast(IntegerType()))) \
        .withColumn("state", trim(col("state"))) \
        .withColumn("dq_valid",
                    when(col("monthly_spend_inr").isNull() |
                         col("total_orders").isNull() |
                         col("churned").isNull(), lit(False))
                    .otherwise(lit(True))) \
        .withColumn("_processed_at", current_timestamp())

    valid_count = df.filter(col("dq_valid")).count()
    total_count = df.count()
    print(f"  ✓ Customers silver: {valid_count:,}/{total_count:,} valid records")
    return df.filter(col("dq_valid"))


def clean_reviews_silver(bronze_df):
    """
    Silver transformation for reviews:
    - Remove HTML/special chars, validate rating range
    """
    print("[SILVER] Cleaning reviews...")

    df = bronze_df \
        .dropDuplicates(["review_id"]) \
        .filter(col("review_text").isNotNull() & (col("review_text") != "")) \
        .withColumn("review_text",
                    trim(regexp_replace(col("review_text"), "[^a-zA-Z0-9 .,!?']", " "))) \
        .withColumn("rating",
                    when((col("rating") < 1) | (col("rating") > 5), lit(3))
                    .otherwise(col("rating").cast(IntegerType()))) \
        .withColumn("review_date",
                    to_timestamp(col("review_date"), "yyyy-MM-dd")) \
        .withColumn("_processed_at", current_timestamp())

    print(f"  ✓ Reviews silver: {df.count():,} records")
    return df


def clean_ab_events_silver(bronze_df):
    """
    Silver transformation for A/B events:
    - Validate group assignments, remove invalid events
    """
    print("[SILVER] Cleaning A/B events...")

    df = bronze_df \
        .dropDuplicates(["event_id"]) \
        .filter(col("group").isin(["control", "treatment"])) \
        .withColumn("revenue_inr",
                    when(col("revenue_inr") < 0, lit(0.0))
                    .otherwise(col("revenue_inr").cast(DoubleType()))) \
        .withColumn("converted", col("converted").cast(IntegerType())) \
        .withColumn("timestamp", to_timestamp(col("timestamp"))) \
        .withColumn("_processed_at", current_timestamp())

    print(f"  ✓ AB events silver: {df.count():,} records")
    return df


# Run silver transformations
silver_customers = clean_customers_silver(bronze_customers)
silver_reviews   = clean_reviews_silver(bronze_reviews)
silver_ab_events = clean_ab_events_silver(bronze_ab_events)

# Write silver tables
for name, df in [("customers", silver_customers),
                  ("reviews", silver_reviews),
                  ("ab_events", silver_ab_events)]:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
      .save(f"{SILVER_PATH}/{name}")
    print(f"[SILVER] Written: {SILVER_PATH}/{name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🥇 GOLD LAYER — ML-Ready Feature Store

def build_gold_feature_store(customers_df, ab_events_df):
    """
    Gold layer: join & engineer ML-ready features.
    This is the feature store used by the DL churn model.
    """
    print("[GOLD] Building feature store...")

    # Aggregate A/B event features per customer
    ab_features = ab_events_df.groupBy("customer_id").agg(
        count("event_id").alias("total_ab_events"),
        mean("converted").alias("ab_conversion_rate"),
        mean("revenue_inr").alias("ab_avg_revenue"),
        mean("page_views").alias("ab_avg_page_views"),
    )

    # Join with customer features
    gold_df = customers_df.join(ab_features, on="customer_id", how="left")

    # Feature engineering
    gold_df = gold_df \
        .withColumn("spend_per_order",
                    col("monthly_spend_inr") / (col("total_orders") + 1)) \
        .withColumn("recency_score",
                    lit(1.0) / (col("last_login_days_ago") + 1)) \
        .withColumn("engagement_score",
                    col("email_open_rate") * 0.3 +
                    (col("app_sessions_monthly") / lit(50)).cast(DoubleType()) * 0.4 +
                    (lit(1) - col("last_login_days_ago") / lit(120)) * 0.3) \
        .withColumn("high_value_flag",
                    when(col("monthly_spend_inr") > 3000, 1).otherwise(0)) \
        .withColumn("_feature_timestamp", current_timestamp())

    print(f"  ✓ Gold feature store: {gold_df.count():,} records | {len(gold_df.columns)} features")
    return gold_df


gold_features = build_gold_feature_store(silver_customers, silver_ab_events)
gold_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .save(f"{GOLD_PATH}/feature_store")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📊 Data Quality Report

print("\n" + "="*60)
print("  MEDALLION PIPELINE — DATA QUALITY REPORT")
print("="*60)

for layer, name, path in [
    ("Bronze", "customers",  f"{BRONZE_PATH}/customers"),
    ("Silver", "customers",  f"{SILVER_PATH}/customers"),
    ("Gold",   "features",   f"{GOLD_PATH}/feature_store"),
    ("Bronze", "reviews",    f"{BRONZE_PATH}/reviews"),
    ("Silver", "reviews",    f"{SILVER_PATH}/reviews"),
]:
    try:
        df = spark.read.format("delta").load(path)
        print(f"  [{layer}] {name}: {df.count():,} rows × {len(df.columns)} cols")
    except Exception as e:
        print(f"  [{layer}] {name}: Could not read — {e}")

print("="*60)
print("  Pipeline complete ✓")
print("="*60)

# COMMAND ----------
# Register Gold tables in Hive Metastore (optional)
spark.sql("CREATE DATABASE IF NOT EXISTS genai_customer_intelligence")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS genai_customer_intelligence.gold_feature_store
    USING DELTA
    LOCATION '{GOLD_PATH}/feature_store'
""")
print("✓ Gold feature store registered in Hive Metastore")
