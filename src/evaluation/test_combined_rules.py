"""Compare explainable AML rule combinations on the saved VALIDATION split.

This script reads already-scored HDFS Parquet. It does not rerun preprocessing,
recalculate account features, overwrite scored transactions, or inspect TEST labels
for rule selection. It is intended to guide the next detector revision cheaply.
"""
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HDFS_BASE = "hdfs://localhost:9000/aml-sentinel"
SCORED_PATH = f"{HDFS_BASE}/results/scored_transactions"
OUTPUT_FILE = Path("outputs") / "rule_combo_experiments.json"


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def metric_expressions(prediction, prefix):
    pred = prediction.cast("int")
    label = F.col("Is_laundering").cast("int")
    return [
        F.sum(F.when((pred == 1) & (label == 1), 1).otherwise(0)).alias(f"{prefix}_tp"),
        F.sum(F.when((pred == 1) & (label == 0), 1).otherwise(0)).alias(f"{prefix}_fp"),
        F.sum(F.when((pred == 0) & (label == 1), 1).otherwise(0)).alias(f"{prefix}_fn"),
        F.sum(F.when((pred == 0) & (label == 0), 1).otherwise(0)).alias(f"{prefix}_tn"),
        F.sum(pred).alias(f"{prefix}_alerts"),
    ]


def unpack(row, name, total):
    tp = int(row[f"{name}_tp"] or 0)
    fp = int(row[f"{name}_fp"] or 0)
    fn = int(row[f"{name}_fn"] or 0)
    tn = int(row[f"{name}_tn"] or 0)
    alerts = int(row[f"{name}_alerts"] or 0)
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)
    f2 = safe_rate(5 * precision * recall, 4 * precision + recall)
    return {
        "strategy": name,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "predicted_alerts": alerts,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "alert_rate": safe_rate(alerts, total),
    }


spark = (
    SparkSession.builder.appName("AML-Sentinel-Validation-Rule-Experiments")
    .master("local[2]")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    print(f"Reading existing scored Parquet: {SCORED_PATH}")
    saved = spark.read.parquet(SCORED_PATH)
    required = {
        "evaluation_split", "Is_laundering", "risk_score",
        "receiver_unique_senders", "rule_high_amount", "rule_cross_border",
        "rule_cash_related", "rule_currency_difference",
        "rule_unusual_sender_amount", "rule_fan_out", "rule_transaction_burst",
    }
    missing = sorted(required - set(saved.columns))
    if missing:
        raise ValueError(
            f"Scored Parquet is missing columns {missing}. "
            "Run the updated detect_transactions.py first."
        )

    validation = saved.filter(F.col("evaluation_split") == "VALIDATION")
    validation_rows = validation.agg(
        F.count("*").alias("total"),
        F.sum("Is_laundering").alias("positives"),
    ).first()
    total = int(validation_rows["total"] or 0)
    positives = int(validation_rows["positives"] or 0)
    if total == 0 or positives == 0:
        raise ValueError(
            f"Validation split is empty or has no positive labels: rows={total}, positives={positives}."
        )
    print(f"Validation transactions: {total:,}; labelled laundering: {positives:,}")
    print("Comparing candidate combinations on VALIDATION only...")

    fanin = F.coalesce(F.col("receiver_unique_senders"), F.lit(0))
    high_amount = F.col("rule_high_amount") == 1
    cross_border = F.col("rule_cross_border") == 1
    cash = F.col("rule_cash_related") == 1
    currency_diff = F.col("rule_currency_difference") == 1
    unusual_amount = F.col("rule_unusual_sender_amount") == 1
    fan_out = F.col("rule_fan_out") == 1
    burst = F.col("rule_transaction_burst") == 1
    risk = F.col("risk_score")

    candidates = {}
    for threshold in (1, 2, 3, 4, 5, 6):
        candidates[f"risk_score_ge_{threshold}"] = risk >= threshold

    for threshold in (20, 22, 24, 26, 28, 30):
        base_name = f"fan_in_ge_{threshold}"
        candidates[base_name] = fanin >= threshold
        if threshold in (24, 26, 28):
            candidates[f"{base_name}_OR_high_amount"] = (fanin >= threshold) | high_amount
            candidates[f"{base_name}_OR_unusual_amount"] = (fanin >= threshold) | unusual_amount
            candidates[f"{base_name}_OR_currency_difference"] = (fanin >= threshold) | currency_diff
            candidates[f"{base_name}_OR_cross_border"] = (fanin >= threshold) | cross_border
            candidates[f"{base_name}_OR_cash"] = (fanin >= threshold) | cash
            candidates[f"{base_name}_AND_context"] = (
                (fanin >= threshold) & (high_amount | currency_diff | cross_border | unusual_amount)
            )

    candidates["fan_in_ge_24_OR_risk_score_ge_3"] = (fanin >= 24) | (risk >= 3)
    candidates["fan_in_ge_26_OR_risk_score_ge_3"] = (fanin >= 26) | (risk >= 3)
    candidates["fan_in_ge_28_OR_risk_score_ge_3"] = (fanin >= 28) | (risk >= 3)
    candidates["fan_in_ge_24_AND_two_context_rules"] = (
        (fanin >= 24)
        & (
            high_amount.cast("int")
            + currency_diff.cast("int")
            + cross_border.cast("int")
            + unusual_amount.cast("int")
            + cash.cast("int")
            + fan_out.cast("int")
            + burst.cast("int")
            >= 2
        )
    )
    candidates["fan_in_ge_24_OR_high_amount_AND_currency"] = (
        (fanin >= 24) | (high_amount & currency_diff)
    )

    aggregate = [F.count("*").alias("total")]
    for name, prediction in candidates.items():
        aggregate.extend(metric_expressions(prediction, name))
    result_row = validation.agg(*aggregate).first().asDict()

    results = [
        unpack(result_row, name, total)
        for name in candidates
    ]
    results_by_f1 = sorted(
        results, key=lambda r: (r["f1_score"], r["precision"], r["recall"]), reverse=True
    )
    results_by_f2 = sorted(
        results, key=lambda r: (r["f2_score"], r["recall"], r["precision"]), reverse=True
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": SCORED_PATH,
        "evaluation_split": "VALIDATION",
        "validation_transactions": total,
        "validation_actual_laundering": positives,
        "candidate_count": len(results),
        "best_by_f1": results_by_f1[:10],
        "best_by_f2": results_by_f2[:10],
        "all_candidates_by_f1": results_by_f1,
        "note": (
            "These are exploratory rule comparisons on VALIDATION only. Labels are used "
            "to measure candidate quality, not as prediction features. Do not select rules "
            "using TEST metrics. Current features remain heuristic and do not prove laundering."
        ),
    }
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    def print_table(title, rows):
        print("\n" + title)
        print(f"{'Strategy':48} {'Prec%':>8} {'Recall%':>8} {'F1%':>8} {'F2%':>8} {'Alerts':>10} {'FP':>10}")
        print("-" * 108)
        for row in rows:
            print(
                f"{row['strategy'][:48]:48} "
                f"{row['precision'] * 100:8.2f} "
                f"{row['recall'] * 100:8.2f} "
                f"{row['f1_score'] * 100:8.2f} "
                f"{row['f2_score'] * 100:8.2f} "
                f"{row['predicted_alerts']:10,} "
                f"{row['false_positives']:10,}"
            )

    print_table("TOP 10 BY F1 — BALANCED PRECISION/RECALL", results_by_f1[:10])
    print_table("TOP 10 BY F2 — RECALL-LEANING", results_by_f2[:10])
    print(f"\nSaved comparison: {OUTPUT_FILE}")
    print("No HDFS data was overwritten.")
finally:
    spark.stop()
    print("Spark session stopped.")
