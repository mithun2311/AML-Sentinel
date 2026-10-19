"""Extend the existing rule experiment: select by VALIDATION F2, then evaluate
only the selected strategy and the existing fan_in_ge_26 baseline on TEST.

Reads the already-scored HDFS Parquet and does not overwrite any project outputs.
Do not use TEST results to pick a different candidate.
"""
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HDFS_BASE = "hdfs://localhost:9000/aml-sentinel"
SCORED_PATH = f"{HDFS_BASE}/results/scored_transactions"
OUTPUT_FILE = Path("outputs") / "selected_candidate_test_check.json"


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def metrics_for(frame, prediction, name):
    pred = prediction.cast("int")
    label = F.col("Is_laundering").cast("int")
    row = frame.agg(
        F.count("*").alias("total"),
        F.sum(F.when((pred == 1) & (label == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((pred == 1) & (label == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((pred == 0) & (label == 1), 1).otherwise(0)).alias("fn"),
        F.sum(F.when((pred == 0) & (label == 0), 1).otherwise(0)).alias("tn"),
        F.sum(pred).alias("alerts"),
    ).first().asDict()
    tp, fp, fn, tn = (int(row[k] or 0) for k in ("tp", "fp", "fn", "tn"))
    alerts, total = int(row["alerts"] or 0), int(row["total"] or 0)
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)
    f2 = safe_rate(5 * precision * recall, 4 * precision + recall)
    return {
        "split": name,
        "transactions": total,
        "actual_laundering": tp + fn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "alerts": alerts,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "alert_rate": safe_rate(alerts, total),
    }


spark = (
    SparkSession.builder.appName("AML-Sentinel-Selected-Candidate-Check")
    .master("local[2]")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.sql.shuffle.partitions", "32")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    print(f"Reading existing scored data: {SCORED_PATH}")
    df = spark.read.parquet(SCORED_PATH)
    required = {
        "evaluation_split", "Is_laundering", "receiver_unique_senders",
        "rule_high_amount", "rule_currency_difference", "predicted_suspicious",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required scored columns: {missing}")

    fanin = F.coalesce(F.col("receiver_unique_senders"), F.lit(0))
    high_amount = F.col("rule_high_amount") == 1
    currency_diff = F.col("rule_currency_difference") == 1

    # This candidate was selected by validation F2 in the previous experiment.
    selected_expr = (fanin >= 24) | (high_amount & currency_diff)
    baseline_expr = fanin >= 26

    validation = df.filter(F.col("evaluation_split") == "VALIDATION")
    test = df.filter(F.col("evaluation_split") == "TEST")

    # Validation confirms the selection criterion; TEST is only used once here
    # to estimate how the preselected candidate behaves on the later period.
    result = {
        "selected_by": "highest F2 among candidates in validation experiment",
        "selected_strategy": "fan_in_ge_24_OR_high_amount_AND_currency",
        "test_was_used_for_selection": False,
        "validation": {
            "selected_candidate": metrics_for(validation, selected_expr, "VALIDATION"),
            "fan_in_ge_26_baseline": metrics_for(validation, baseline_expr, "VALIDATION"),
        },
        "test": {
            "selected_candidate": metrics_for(test, selected_expr, "TEST"),
            "fan_in_ge_26_baseline": metrics_for(test, baseline_expr, "TEST"),
        },
        "note": (
            "The later TEST period overlaps data examined in prior experiments; this is not "
            "an untouched independent benchmark. This script does not modify HDFS or the detector."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    for split_name in ("validation", "test"):
        print(f"\n{split_name.upper()}")
        for strategy_name, values in result[split_name].items():
            print(
                f"{strategy_name}: precision={values['precision']:.4%}, "
                f"recall={values['recall']:.4%}, F1={values['f1_score']:.4%}, "
                f"F2={values['f2_score']:.4%}, alerts={values['alerts']:,}, "
                f"FP={values['false_positives']:,}, FN={values['false_negatives']:,}"
            )
    print(f"\nSaved: {OUTPUT_FILE}")
    print("No HDFS data was overwritten.")
finally:
    spark.stop()
    print("Spark session stopped.")
