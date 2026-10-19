"""Compare individual rule flags on a chosen split without changing the detector.

Default: VALIDATION. Results are diagnostic and must not be represented as proof
that a combined strategy will perform equally well (rules overlap).
Run from the project root with the existing Spark/HDFS environment configured.
"""
import json
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HDFS_PATH = "hdfs://localhost:9000/aml-sentinel/results/scored_transactions"
SPLIT = os.getenv("AML_EVALUATION_SPLIT", "VALIDATION").strip().upper()
if SPLIT not in {"VALIDATION", "TEST"}:
    raise ValueError("AML_EVALUATION_SPLIT must be VALIDATION or TEST")

RULES = [
    "rule_high_amount",
    "rule_cross_border",
    "rule_cash_related",
    "rule_high_sender_activity",
    "rule_high_receiver_activity",
    "rule_fan_out",
    "rule_fan_in",
    "rule_transaction_burst",
    "rule_currency_difference",
    "rule_unusual_sender_amount",
]


def ratio(n, d):
    return float(n / d) if d else 0.0


spark = (
    SparkSession.builder.appName("AML-Sentinel-Rule-Ablation")
    .master("local[2]")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    df = spark.read.parquet(HDFS_PATH)
    required = ["evaluation_split", "Is_laundering", *RULES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in scored data: {missing}")

    selected = df.filter(F.col("evaluation_split") == SPLIT)
    total = selected.count()
    positives = int(selected.agg(F.sum(F.col("Is_laundering").cast("long")).alias("p")).first()["p"] or 0)
    if total == 0:
        raise ValueError(f"No rows found for split={SPLIT}; check evaluation_split values.")

    expressions = [F.count(F.lit(1)).alias("total")]
    for rule in RULES:
        pred = F.coalesce(F.col(rule).cast("int"), F.lit(0)) == 1
        label = F.col("Is_laundering").cast("int") == 1
        expressions.extend([
            F.sum(F.when(pred & label, 1).otherwise(0)).alias(f"{rule}_tp"),
            F.sum(F.when(pred & ~label, 1).otherwise(0)).alias(f"{rule}_fp"),
            F.sum(F.when(~pred & label, 1).otherwise(0)).alias(f"{rule}_fn"),
            F.sum(F.when(pred, 1).otherwise(0)).alias(f"{rule}_alerts"),
        ])
    row = selected.agg(*expressions).first().asDict()

    results = []
    for rule in RULES:
        tp = int(row[f"{rule}_tp"] or 0)
        fp = int(row[f"{rule}_fp"] or 0)
        fn = int(row[f"{rule}_fn"] or 0)
        alerts = int(row[f"{rule}_alerts"] or 0)
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        f1 = ratio(2 * precision * recall, precision + recall)
        f2 = ratio(5 * precision * recall, 4 * precision + recall)
        results.append({
            "rule": rule,
            "split": SPLIT,
            "transactions": total,
            "actual_laundering": positives,
            "alerts": alerts,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f2_score": f2,
            "alert_rate": ratio(alerts, total),
            "false_positive_rate": ratio(fp, total - positives),
        })
    results.sort(key=lambda item: (item["f2_score"], item["recall"], item["precision"]), reverse=True)

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"rule_ablation_{SPLIT.lower()}.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump({
            "split": SPLIT,
            "transactions": total,
            "actual_laundering": positives,
            "selection_warning": "Individual-rule metrics are not additive because rules overlap. Use combined-strategy validation comparisons for strategy selection.",
            "results": results,
        }, handle, indent=2)

    print(f"Rule ablation on {SPLIT}: {total:,} transactions, {positives:,} positives")
    print(f"{'Rule':34} {'Alerts':>10} {'Precision':>11} {'Recall':>11} {'F1':>10} {'F2':>10}")
    for item in results:
        print(f"{item['rule']:34} {item['alerts']:10,} {item['precision']:10.4%} {item['recall']:10.4%} {item['f1_score']:9.4%} {item['f2_score']:9.4%}")
    print(f"Saved: {destination}")
finally:
    spark.stop()
