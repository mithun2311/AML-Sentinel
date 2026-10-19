import csv
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HDFS_PATH = (
    "hdfs://localhost:9000/"
    "aml-sentinel/results/scored_transactions"
)
EVALUATION_SPLIT = os.getenv("AML_EVALUATION_SPLIT", "VALIDATION").strip().upper()
if EVALUATION_SPLIT not in {"TRAIN", "VALIDATION", "TEST"}:
    raise ValueError("AML_EVALUATION_SPLIT must be TRAIN, VALIDATION, or TEST")

CANDIDATES = {
    "fan_out": {
        "column": "sender_unique_receivers",
        "thresholds": [10, 20, 30, 40, 50, 56, 57],
    },
    "fan_in": {
        "column": "receiver_unique_senders",
        "thresholds": [10, 15, 20, 22, 25, 26],
    },
    "transaction_burst": {
        "column": "sender_day_count",
        "thresholds": [20, 50, 100, 200, 250, 261],
    },
    "high_amount": {
        "column": "Amount",
        "thresholds": [45708.89, 67847.40, 12618498.40],
    },
}

spark = (
    SparkSession.builder
    .appName("AML-Sentinel-Rule-Calibration")
    .master("local[2]")
    .config(
        "spark.hadoop.dfs.client.use.datanode.hostname",
        "true",
    )
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print(f"Reading scored data for {EVALUATION_SPLIT} calibration...")
    df = spark.read.parquet(HDFS_PATH)

    required = [
        "Is_laundering",
        "Amount",
        "sender_unique_receivers",
        "receiver_unique_senders",
        "sender_day_count",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "evaluation_split" not in df.columns:
        raise ValueError("Scored data has no evaluation_split column")
    df = df.filter(F.col("evaluation_split") == EVALUATION_SPLIT)

    total_rows = df.count()
    total_positive = df.filter(
        F.col("Is_laundering") == 1
    ).count()

    print(f"Transactions: {total_rows:,}")
    print(f"Laundering-labelled transactions: {total_positive:,}")
    print("Testing candidate thresholds...")

    expressions = []

    for rule_name, config in CANDIDATES.items():
        column = config["column"]

        for threshold in config["thresholds"]:
            condition = F.col(column) >= F.lit(threshold)
            key = f"{rule_name}_{str(threshold).replace('.', '_')}"

            expressions.append(
                F.sum(
                    F.when(condition, 1).otherwise(0)
                ).alias(f"{key}_flagged")
            )

            expressions.append(
                F.sum(
                    F.when(
                        condition & (F.col("Is_laundering") == 1),
                        1,
                    ).otherwise(0)
                ).alias(f"{key}_tp")
            )

    # One aggregation evaluates all candidate thresholds.
    result = df.agg(*expressions).first().asDict()

    rows = []

    for rule_name, config in CANDIDATES.items():
        column = config["column"]

        for threshold in config["thresholds"]:
            key = f"{rule_name}_{str(threshold).replace('.', '_')}"

            flagged = int(result.get(f"{key}_flagged") or 0)
            tp = int(result.get(f"{key}_tp") or 0)
            fp = flagged - tp
            fn = total_positive - tp

            precision = tp / flagged if flagged else 0.0
            recall = tp / total_positive if total_positive else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            rows.append({
                "rule": rule_name,
                "feature": column,
                "threshold": threshold,
                "flagged_transactions": flagged,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision_percent": round(precision * 100, 4),
                "recall_percent": round(recall * 100, 4),
                "f1_percent": round(f1 * 100, 4),
                "alert_rate_percent": round(
                    flagged / total_rows * 100, 4
                ),
            })

    rows.sort(key=lambda r: (r["rule"], r["threshold"]))

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    output_file = output / "threshold_calibration.csv"

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== THRESHOLD COMPARISON ==========")

    for rule_name in CANDIDATES:
        print(f"\n{rule_name.upper()}")
        print(
            f"{'Threshold':>12} {'Flagged':>12} "
            f"{'Precision':>12} {'Recall':>10} {'F1':>10}"
        )

        rule_rows = [r for r in rows if r["rule"] == rule_name]

        for row in rule_rows:
            print(
                f"{row['threshold']:>12} "
                f"{row['flagged_transactions']:>12,} "
                f"{row['precision_percent']:>11.4f}% "
                f"{row['recall_percent']:>9.4f}% "
                f"{row['f1_percent']:>9.4f}%"
            )

    print(f"\nSaved: {output_file}")
    print(
        "\nNote: These are individual-rule results, not the "
        "performance of the combined detector."
    )
    print(
        "Threshold selection on these same labels is calibration, "
        "not independent evaluation."
    )

finally:
    spark.stop()