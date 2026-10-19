
import csv
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HDFS_PATH = (
    "hdfs://localhost:9000/"
    "aml-sentinel/results/scored_transactions"
)

THRESHOLDS = [24, 26, 28, 30]
EVALUATION_SPLIT = os.getenv("AML_EVALUATION_SPLIT", "VALIDATION").strip().upper()
if EVALUATION_SPLIT not in {"VALIDATION", "TEST"}:
    raise ValueError("AML_EVALUATION_SPLIT must be VALIDATION or TEST")

spark = (
    SparkSession.builder
    .appName("AML-Sentinel-FanIn-Pattern-Coverage")
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
        "Is_laundering",
        "Laundering_type",
        "predicted_suspicious",
        "receiver_unique_senders",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # The detector emits TRAIN / VALIDATION / TEST, not HOLDOUT.
    # TEST has already been explored; report it only as an exploratory diagnostic.
    selected_split = df.filter(F.col("evaluation_split") == EVALUATION_SPLIT)
    labelled = selected_split.filter(F.col("Is_laundering") == 1)

    expressions = [
        F.count("*").alias("pattern_total"),
        F.sum(
            F.when(F.col("predicted_suspicious") == 1, 1)
            .otherwise(0)
        ).alias("current_caught"),
    ]

    for threshold in THRESHOLDS:
        expressions.append(
            F.sum(
                F.when(
                    F.col("receiver_unique_senders") >= threshold,
                    1,
                ).otherwise(0)
            ).alias(f"fanin_{threshold}_caught")
        )

    rows = (
        labelled.groupBy("Laundering_type")
        .agg(*expressions)
        .orderBy(F.desc("pattern_total"))
        .collect()
    )

    results = []

    for row in rows:
        item = row.asDict()
        total = int(item["pattern_total"] or 0)

        item["current_recall_percent"] = round(
            (int(item["current_caught"] or 0) / total * 100)
            if total else 0.0,
            2,
        )

        for threshold in THRESHOLDS:
            key = f"fanin_{threshold}_caught"
            item[f"fanin_{threshold}_recall_percent"] = round(
                (int(item[key] or 0) / total * 100)
                if total else 0.0,
                2,
            )

        results.append(item)

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    output_file = output / f"{EVALUATION_SPLIT.lower()}_fanin_pattern_coverage.csv"

    if results:
        with output_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    else:
        # Still create a valid CSV so downstream dashboard code can handle it.
        with output_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Laundering_type", "pattern_total", "current_caught"])

    print(f"\n========== {EVALUATION_SPLIT} PATTERN COVERAGE ==========")
    print(
        f"{'Pattern':<29} {'Total':>7} {'Current':>9} "
        f"{'Fan-in24':>9} {'Fan-in26':>9} "
        f"{'Fan-in28':>9} {'Fan-in30':>9}"
    )

    for item in results:
        print(
            f"{str(item['Laundering_type']):<29} "
            f"{item['pattern_total']:>7,} "
            f"{item['current_recall_percent']:>8.2f}% "
            f"{item['fanin_24_recall_percent']:>8.2f}% "
            f"{item['fanin_26_recall_percent']:>8.2f}% "
            f"{item['fanin_28_recall_percent']:>8.2f}% "
            f"{item['fanin_30_recall_percent']:>8.2f}%"
        )

    print(f"\nSaved: {output_file}")
    print(
        f"Rows evaluated: {labelled.count():,}. Results are diagnostics; "
        "do not use TEST to select thresholds. Set AML_EVALUATION_SPLIT=VALIDATION "
        "for validation-based comparison."
    )

finally:
    spark.stop()
