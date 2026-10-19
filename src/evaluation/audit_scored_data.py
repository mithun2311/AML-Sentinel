
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HDFS_PATH = (
    "hdfs://localhost:9000/"
    "aml-sentinel/results/scored_transactions"
)
METRICS_PATH = Path("outputs/evaluation_metrics.json")

spark = (
    SparkSession.builder
    .appName("AML-Sentinel-Scored-Data-Audit")
    .master("local[2]")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print("Reading existing scored transactions...")
    df = spark.read.parquet(HDFS_PATH)

    required = [
        "Amount",
        "rule_high_amount",
        "risk_score",
        "predicted_suspicious",
        "Is_laundering",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("\nRelevant schema:")
    df.select(*required).printSchema()

    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

    threshold = float(metrics["amount_threshold"])

    amount = F.col("Amount").cast("double")
    high_amount_rule = F.col("rule_high_amount").cast("int")

    print("\nChecking amount distribution...")
    quantiles = df.approxQuantile(
        "Amount", [0.50, 0.95, 0.99, 0.995, 0.999], 0.01
    )

    summary = df.agg(
        F.count("*").alias("rows"),
        F.sum(F.when(amount.isNull(), 1).otherwise(0)).alias("null_amounts"),
        F.min(amount).alias("min_amount"),
        F.max(amount).alias("max_amount"),
        F.sum(F.when(amount >= threshold, 1).otherwise(0))
            .alias("amount_ge_threshold"),
        F.sum(F.when(amount > threshold, 1).otherwise(0))
            .alias("amount_gt_threshold"),
        F.sum(F.when(high_amount_rule == 1, 1).otherwise(0))
            .alias("rule_flagged"),
        F.sum(
            F.when(
                (amount >= threshold) & (high_amount_rule != 1), 1
            ).otherwise(0)
        ).alias("expected_but_not_flagged"),
        F.sum(
            F.when(
                (amount < threshold) & (high_amount_rule == 1), 1
            ).otherwise(0)
        ).alias("flagged_below_threshold"),
    ).first().asDict()

    print("\n===== AMOUNT AUDIT =====")
    print(f"Threshold from metrics: {threshold:,.2f}")
    print(f"Approximate quantiles [50%, 95%, 99%, 99.5%, 99.9%]: {quantiles}")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n===== INTERPRETATION =====")
    expected = int(summary["amount_ge_threshold"] or 0)
    actual = int(summary["rule_flagged"] or 0)

    if expected == actual:
        print("Amount threshold count agrees with the saved rule.")
    else:
        print(
            "MISMATCH: the amount threshold count and saved rule count "
            "differ. Investigate the saved feature before tuning weights."
        )

    print(
        "\nThis audit does not change any data. It checks whether the "
        "stored amount rule agrees with the configured threshold."
    )

finally:
    spark.stop()
