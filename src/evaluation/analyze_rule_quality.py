
import csv
import json
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

spark = (
    SparkSession.builder
    .appName("AML-Sentinel-Rule-Quality-Analysis")
    .master("local[2]")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.sql.shuffle.partitions", "48")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print(f"Reading scored transactions for {EVALUATION_SPLIT} quality analysis...")
    df = spark.read.parquet(HDFS_PATH)

    required = ["risk_score", "Is_laundering", "evaluation_split"] + RULES
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in scored data: {missing}")

    df = df.filter(F.col("evaluation_split") == EVALUATION_SPLIT)

    df = df.select(
        *[F.col(c) for c in required]
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Risk-score distribution and laundering-label prevalence
    # --------------------------------------------------------

    print("Analyzing risk-score distribution...")

    score_rows = (
        df.groupBy("risk_score")
        .agg(
            F.count("*").alias("transactions"),
            F.sum(
                F.when(F.col("Is_laundering") == 1, 1).otherwise(0)
            ).alias("laundering_label_count"),
        )
        .orderBy("risk_score")
        .collect()
    )

    total_transactions = sum(int(r["transactions"]) for r in score_rows)
    total_labeled = sum(
        int(r["laundering_label_count"] or 0) for r in score_rows
    )

    score_results = []

    for row in score_rows:
        count = int(row["transactions"])
        positives = int(row["laundering_label_count"] or 0)

        score_results.append({
            "risk_score": int(row["risk_score"]),
            "transactions": count,
            "laundering_label_count": positives,
            "normal_label_count": count - positives,
            "label_prevalence_percent": (
                100 * positives / count if count else 0.0
            ),
            "transaction_share_percent": (
                100 * count / total_transactions
                if total_transactions else 0.0
            ),
            "share_of_labeled_laundering_percent": (
                100 * positives / total_labeled
                if total_labeled else 0.0
            ),
        })

    with open(
        output_dir / "risk_score_distribution.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=score_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(score_results)

    with open(
        output_dir / "risk_score_distribution.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(score_results, file, indent=2)

    # --------------------------------------------------------
    # 2. Measure each individual rule's alert quality
    # --------------------------------------------------------

    print("Measuring individual rule quality...")

    expressions = [
        F.count("*").alias("total_transactions"),
        F.sum(
            F.when(F.col("Is_laundering") == 1, 1).otherwise(0)
        ).alias("total_labeled_laundering"),
    ]

    for rule in RULES:
        active = F.col(rule) == 1

        expressions.extend([
            F.sum(
                F.when(active, 1).otherwise(0)
            ).alias(f"{rule}_alerts"),

            F.sum(
                F.when(
                    active & (F.col("Is_laundering") == 1),
                    1
                ).otherwise(0)
            ).alias(f"{rule}_true_positives"),
        ])

    summary = df.agg(*expressions).first().asDict()

    total = int(summary["total_transactions"] or 0)
    labeled_total = int(summary["total_labeled_laundering"] or 0)

    rule_results = []

    for rule in RULES:
        alerts = int(summary[f"{rule}_alerts"] or 0)
        true_positives = int(summary[f"{rule}_true_positives"] or 0)
        false_positives = alerts - true_positives

        rule_results.append({
            "rule": rule,
            "transactions_flagged": alerts,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision_percent": (
                100 * true_positives / alerts if alerts else 0.0
            ),
            "recall_percent": (
                100 * true_positives / labeled_total
                if labeled_total else 0.0
            ),
            "transaction_coverage_percent": (
                100 * alerts / total if total else 0.0
            ),
        })

    rule_results.sort(
        key=lambda item: item["precision_percent"],
        reverse=True
    )

    with open(
        output_dir / "rule_quality.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rule_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(rule_results)

    with open(
        output_dir / "rule_quality.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(rule_results, file, indent=2)

    # --------------------------------------------------------
    # 3. Display a compact report
    # --------------------------------------------------------

    print("\n========== RISK-SCORE DISTRIBUTION ==========")
    print(
        f"{'Score':>6} {'Transactions':>15} "
        f"{'Labeled AML':>13} {'AML prevalence':>15} "
        f"{'Share of AML':>14}"
    )

    for item in score_results:
        print(
            f"{item['risk_score']:>6} "
            f"{item['transactions']:>15,} "
            f"{item['laundering_label_count']:>13,} "
            f"{item['label_prevalence_percent']:>14.4f}% "
            f"{item['share_of_labeled_laundering_percent']:>13.2f}%"
        )

    print("\n========== INDIVIDUAL RULE QUALITY ==========")
    print(
        f"{'Rule':<32} {'Flagged':>12} {'True positives':>15} "
        f"{'Precision':>11} {'Recall':>10}"
    )

    for item in rule_results:
        print(
            f"{item['rule']:<32} "
            f"{item['transactions_flagged']:>12,} "
            f"{item['true_positives']:>15,} "
            f"{item['precision_percent']:>10.3f}% "
            f"{item['recall_percent']:>9.2f}%"
        )

    print("\nFiles saved:")
    print("outputs/risk_score_distribution.csv")
    print("outputs/risk_score_distribution.json")
    print("outputs/rule_quality.csv")
    print("outputs/rule_quality.json")

    print(
        "\nInterpretation note: individual rule results overlap; "
        "one transaction can trigger several rules. These results "
        "describe this labeled dataset and are not independent "
        "test-set performance."
    )

finally:
    spark.stop()
