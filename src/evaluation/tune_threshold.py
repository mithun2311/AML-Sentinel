
import json
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HDFS_PATH = (
    "hdfs://localhost:9000/"
    "aml-sentinel/results/scored_transactions"
)

THRESHOLDS = [10, 15, 20, 22, 24, 26, 28, 30]
EVALUATION_SPLIT = os.getenv("AML_EVALUATION_SPLIT", "VALIDATION").strip().upper()
if EVALUATION_SPLIT not in {"VALIDATION", "TEST"}:
    raise ValueError("AML_EVALUATION_SPLIT must be VALIDATION or TEST")

spark = (
    SparkSession.builder
    .appName("AML-Sentinel-FanIn-Threshold-Analysis")
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
    print("Reading saved scored transactions...")
    df = spark.read.parquet(HDFS_PATH)

    required = [
        "evaluation_split",
        "receiver_unique_senders",
        "rule_currency_difference",
        "Is_laundering",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Explicitly select a real split in the current detector schema.
    # TEST has been examined previously; treat these results as exploratory.
    evaluation = df.filter(F.col("evaluation_split") == EVALUATION_SPLIT)
    label = F.col("Is_laundering") == 1
    fanin = F.col("receiver_unique_senders")

    expressions = [
        F.count("*").alias("total"),
        F.sum(F.when(label, 1).otherwise(0)).alias("positives"),
    ]

    for threshold in THRESHOLDS:
        condition = fanin >= threshold

        expressions.extend([
            F.sum(
                F.when(condition, 1).otherwise(0)
            ).alias(f"alerts_{threshold}"),
            F.sum(
                F.when(condition & label, 1).otherwise(0)
            ).alias(f"tp_{threshold}"),
        ])

    # Also measure the stricter fan-in plus currency condition.
    strict = (
        (fanin >= 26)
        & (F.col("rule_currency_difference") == 1)
    )
    expressions.extend([
        F.sum(F.when(strict, 1).otherwise(0)).alias("strict_alerts"),
        F.sum(F.when(strict & label, 1).otherwise(0)).alias("strict_tp"),
    ])

    row = evaluation.agg(*expressions).first().asDict()
    total = int(row["total"] or 0)
    positives = int(row["positives"] or 0)

    results = []

    for threshold in THRESHOLDS:
        alerts = int(row[f"alerts_{threshold}"] or 0)
        tp = int(row[f"tp_{threshold}"] or 0)
        fp = alerts - tp
        fn = positives - tp

        precision = tp / alerts if alerts else 0.0
        recall = tp / positives if positives else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results.append({
            "rule": f"fan_in_ge_{threshold}",
            "threshold": threshold,
            "alerts": alerts,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision_percent": round(precision * 100, 4),
            "recall_percent": round(recall * 100, 4),
            "f1_percent": round(f1 * 100, 4),
            "f2_percent": round((5 * precision * recall / (4 * precision + recall) * 100) if (4 * precision + recall) else 0.0, 4),
            "false_positive_rate_percent": round((fp / (fp + (total - positives - fp)) * 100) if (total - positives) > 0 else 0.0, 4),
            "alert_rate_percent": round(
                alerts / total * 100, 4
            ) if total else 0.0,
        })

    alerts = int(row["strict_alerts"] or 0)
    tp = int(row["strict_tp"] or 0)
    fp = alerts - tp
    fn = positives - tp
    precision = tp / alerts if alerts else 0.0
    recall = tp / positives if positives else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    results.append({
        "rule": "fan_in_ge_26_AND_currency_difference",
        "threshold": 26,
        "alerts": alerts,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_percent": round(precision * 100, 4),
        "recall_percent": round(recall * 100, 4),
        "f1_percent": round(f1 * 100, 4),
        "alert_rate_percent": round(
            alerts / total * 100, 4
        ) if total else 0.0,
    })

    results.sort(
        key=lambda item: item["f1_percent"],
        reverse=True,
    )

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"fanin_threshold_comparison_{EVALUATION_SPLIT.lower()}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump({
            "period": "HOLDOUT - exploratory, previously examined",
            "transactions": total,
            "actual_laundering": positives,
            "results": results,
            "warning": (
                "These results are exploratory diagnostics, not an independent benchmark. "
                "Choose thresholds on VALIDATION only; do not tune on TEST. Set "
                "AML_EVALUATION_SPLIT=VALIDATION for threshold selection."
            ),
        }, file, indent=2)

    print("\n========== FAN-IN THRESHOLD COMPARISON ==========")
    print(f"{EVALUATION_SPLIT} transactions: {total:,}")
    if total == 0:
        print(f"No rows found for split={EVALUATION_SPLIT}. Verify the scored Parquet split values.")
    print(f"Labelled laundering: {positives:,}")
    print(
        f"{'Rule':<38} {'Alerts':>9} {'TP':>6} "
        f"{'Precision':>11} {'Recall':>10} {'F1':>9}"
    )

    for item in results:
        print(
            f"{item['rule']:<38} "
            f"{item['alerts']:>9,} "
            f"{item['true_positives']:>6,} "
            f"{item['precision_percent']:>10.4f}% "
            f"{item['recall_percent']:>9.4f}% "
            f"{item['f1_percent']:>8.4f}%"
        )

    print(f"\nSaved: {output_path}")

finally:
    spark.stop()
