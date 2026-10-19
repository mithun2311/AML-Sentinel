
import csv
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HDFS_PATH = (
    "hdfs://localhost:9000/"
    "aml-sentinel/results/scored_transactions"
)
EVALUATION_SPLIT = os.getenv("AML_EVALUATION_SPLIT", "VALIDATION").strip().upper()

FEATURES = {
    "Amount": [0.95, 0.99, 0.995, 0.999],
    "sender_tx_count": [0.50, 0.90, 0.95, 0.99, 0.995],
    "receiver_tx_count": [0.50, 0.90, 0.95, 0.99, 0.995],
    "sender_unique_receivers": [0.50, 0.90, 0.95, 0.99, 0.995],
    "receiver_unique_senders": [0.50, 0.90, 0.95, 0.99, 0.995],
    "sender_day_count": [0.50, 0.90, 0.95, 0.99, 0.995],
}

RULES = [
    "rule_high_amount",
    "rule_high_sender_activity",
    "rule_high_receiver_activity",
    "rule_fan_out",
    "rule_fan_in",
    "rule_transaction_burst",
    "rule_cash_related",
    "rule_cross_border",
    "rule_currency_difference",
    "rule_unusual_sender_amount",
]

spark = (
    SparkSession.builder
    .appName("AML-Sentinel-Deep-Feature-Audit")
    .master("local[2]")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print(f"Reading scored Parquet for {EVALUATION_SPLIT} diagnostics...")
    df = spark.read.parquet(HDFS_PATH)

    missing = [
        c for c in list(FEATURES) + RULES + ["Is_laundering", "evaluation_split"]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df.filter(F.col("evaluation_split") == EVALUATION_SPLIT)

    # Approximate quantiles with tighter rank error than the original
    # 1% relative error. This is still approximate, not exact.
    feature_rows = []

    print("\nCalculating feature quantiles...")

    for column, probabilities in FEATURES.items():
        values = df.approxQuantile(column, probabilities, 0.001)

        for probability, value in zip(probabilities, values):
            feature_rows.append({
                "feature": column,
                "quantile": probability,
                "value": value,
            })

        print(f"{column}: {dict(zip(probabilities, values))}")

    # Count active rules and labelled positives for each rule.
    # These are descriptive statistics, not independent test results.
    print("\nCalculating rule firing counts...")

    rule_expressions = [
        F.count("*").alias("total_transactions"),
        F.sum(
            F.when(F.col("Is_laundering") == 1, 1).otherwise(0)
        ).alias("labelled_laundering"),
    ]

    for rule in RULES:
        active = F.col(rule) == 1

        rule_expressions.extend([
            F.sum(F.when(active, 1).otherwise(0)).alias(f"{rule}_count"),
            F.sum(
                F.when(
                    active & (F.col("Is_laundering") == 1), 1
                ).otherwise(0)
            ).alias(f"{rule}_positive"),
        ])

    counts = df.agg(*rule_expressions).first().asDict()

    rule_rows = []

    for rule in RULES:
        total = int(counts[f"{rule}_count"] or 0)
        tp = int(counts[f"{rule}_positive"] or 0)

        rule_rows.append({
            "rule": rule,
            "flagged_transactions": total,
            "labelled_laundering_flagged": tp,
            "labelled_normal_flagged": total - tp,
            "precision_percent": 100 * tp / total if total else 0.0,
        })

    rule_rows.sort(
        key=lambda row: row["flagged_transactions"],
        reverse=True,
    )

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    with open(
        output / "deep_feature_quantiles.csv",
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=feature_rows[0].keys())
        writer.writeheader()
        writer.writerows(feature_rows)

    with open(
        output / "deep_rule_counts.csv",
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rule_rows[0].keys())
        writer.writeheader()
        writer.writerows(rule_rows)

    print("\n========== RULE FIRING COUNTS ==========")
    for row in rule_rows:
        print(
            f"{row['rule']:<32} "
            f"flagged={row['flagged_transactions']:>10,} "
            f"labelled_laundering={row['labelled_laundering_flagged']:>7,} "
            f"precision={row['precision_percent']:.3f}%"
        )

    print("\nSaved:")
    print("outputs/deep_feature_quantiles.csv")
    print("outputs/deep_rule_counts.csv")

finally:
    spark.stop()
