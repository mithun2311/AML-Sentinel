"""AML Sentinel: explainable PySpark rules with chronological train/validation/test evaluation.

Labels are used only for evaluation, never as detection features. Amount and account
statistics are fitted on the earliest period only. Validation selects a rule strategy;
the later test period evaluates that strategy. The test period has been explored in
earlier experiments, so its metrics are exploratory, not an untouched final benchmark.
Daily burst is a same-day batch feature, not transaction-by-transaction live scoring.
"""
import json
from datetime import date, timedelta
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType

HDFS_BASE = "hdfs://localhost:9000/aml-sentinel"
INPUT_PATH = f"{HDFS_BASE}/processed/transactions_parquet"
OUTPUT_PATH = f"{HDFS_BASE}/results/scored_transactions"

AMOUNT_QUANTILE = 0.995
AMOUNT_QUANTILE_RELATIVE_ERROR = 0.001
SENDER_ACTIVITY_THRESHOLD = 700.0
RECEIVER_ACTIVITY_THRESHOLD = 700.0
FANOUT_THRESHOLD = 55.0
FANIN_THRESHOLD = 25.0
BURST_THRESHOLD = 260.0

RISK_SCORE_CANDIDATES = [1, 2, 3, 4, 5, 6]
FANIN_CANDIDATES = [24, 26, 28, 30]


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def add_metric_expressions(prediction, prefix):
    """Aggregate confusion-matrix counts for a binary prediction expression."""
    pred = prediction.cast("int")
    label = F.col("Is_laundering").cast("int")
    return [
        F.sum(F.when((pred == 1) & (label == 1), 1).otherwise(0)).alias(f"{prefix}_tp"),
        F.sum(F.when((pred == 1) & (label == 0), 1).otherwise(0)).alias(f"{prefix}_fp"),
        F.sum(F.when((pred == 0) & (label == 1), 1).otherwise(0)).alias(f"{prefix}_fn"),
        F.sum(F.when((pred == 0) & (label == 0), 1).otherwise(0)).alias(f"{prefix}_tn"),
        F.sum(pred).alias(f"{prefix}_alerts"),
    ]


def unpack_metrics(row, prefix, total):
    tp = int(row[f"{prefix}_tp"] or 0)
    fp = int(row[f"{prefix}_fp"] or 0)
    fn = int(row[f"{prefix}_fn"] or 0)
    tn = int(row[f"{prefix}_tn"] or 0)
    alerts = int(row[f"{prefix}_alerts"] or 0)
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)
    beta_squared = 4.0
    f2 = safe_rate(
        (1 + beta_squared) * precision * recall,
        beta_squared * precision + recall,
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "predicted_alerts": alerts,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "accuracy": safe_rate(tp + tn, total),
        "false_positive_rate": safe_rate(fp, fp + tn),
        "alert_rate": safe_rate(alerts, total),
        "alert_rate_percent": 100.0 * safe_rate(alerts, total),
    }


spark = (
    SparkSession.builder.appName("AML-Sentinel-Rule-Optimization")
    .master("local[2]")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.sql.shuffle.partitions", "96")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    print("=" * 72)
    print("AML SENTINEL — CHRONOLOGICAL RULE OPTIMIZATION")
    print("=" * 72)
    print(f"Reading cleaned transactions: {INPUT_PATH}")
    df = spark.read.parquet(INPUT_PATH)

    required = [
        "Sender_account", "Receiver_account", "Amount", "Date",
        "Payment_currency", "Received_currency", "Sender_bank_location",
        "Receiver_bank_location", "Payment_type", "Is_laundering",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cleaned Parquet: {missing}")

    df = (
        df.withColumn("Amount", F.col("Amount").cast(DoubleType()))
        .withColumn("Is_laundering", F.col("Is_laundering").cast("int"))
        .withColumn("_event_date", F.to_date(F.col("Date").cast("string")))
    )

    date_stats = df.agg(
        F.count("*").alias("rows"),
        F.count("_event_date").alias("dated_rows"),
        F.countDistinct("_event_date").alias("distinct_dates"),
    ).first()
    total_rows = int(date_stats["rows"] or 0)
    dated_rows = int(date_stats["dated_rows"] or 0)
    distinct_dates = int(date_stats["distinct_dates"] or 0)

    if total_rows == 0:
        raise ValueError("The cleaned Parquet dataset is empty.")
    if dated_rows != total_rows:
        raise ValueError(
            f"{total_rows - dated_rows} rows have an unparseable Date. "
            "Fix Date parsing in preprocessing before detection."
        )
    if distinct_dates < 5:
        raise ValueError("At least five distinct dates are needed for chronological splitting.")

    # Keep each calendar date intact: earliest 60% train, next 20% validation,
    # latest 20% test. Thresholds/features are fitted only on TRAIN.
    offsets = df.select(
        F.datediff("_event_date", F.lit("1970-01-01")).alias("day_offset")
    ).distinct()
    cutoffs = offsets.agg(
        F.expr("percentile_approx(day_offset, 0.60, 10000)").alias("train_cutoff"),
        F.expr("percentile_approx(day_offset, 0.80, 10000)").alias("validation_cutoff"),
    ).first()
    train_cutoff = date(1970, 1, 1) + timedelta(days=int(cutoffs["train_cutoff"]))
    validation_cutoff = date(1970, 1, 1) + timedelta(days=int(cutoffs["validation_cutoff"]))

    df = df.withColumn(
        "evaluation_split",
        F.when(F.col("_event_date") <= F.lit(train_cutoff), "TRAIN")
         .when(F.col("_event_date") <= F.lit(validation_cutoff), "VALIDATION")
         .otherwise("TEST"),
    )
    split_counts = {
        row["evaluation_split"]: int(row["count"])
        for row in df.groupBy("evaluation_split").count().collect()
    }
    train_count = split_counts.get("TRAIN", 0)
    validation_count = split_counts.get("VALIDATION", 0)
    test_count = split_counts.get("TEST", 0)
    if min(train_count, validation_count, test_count) == 0:
        raise ValueError(
            f"Invalid split: train={train_count:,}, validation={validation_count:,}, "
            f"test={test_count:,}. Check Date values."
        )

    train = df.filter(F.col("evaluation_split") == "TRAIN")
    print(f"Total rows: {total_rows:,}")
    print(f"TRAIN through {train_cutoff}: {train_count:,}")
    print(f"VALIDATION through {validation_cutoff}: {validation_count:,}")
    print(f"TEST after {validation_cutoff}: {test_count:,}")
    print("Warning: this later period overlaps the holdout explored in earlier experiments.")

    print("Fitting amount threshold on TRAIN only...")
    amount_quantiles = train.approxQuantile(
        "Amount", [AMOUNT_QUANTILE], AMOUNT_QUANTILE_RELATIVE_ERROR
    )
    if not amount_quantiles or amount_quantiles[0] is None:
        raise ValueError("Could not calculate the training-period amount threshold.")
    amount_threshold = float(amount_quantiles[0])
    print(f"TRAIN 99.5th percentile amount: {amount_threshold:,.2f}")

    print("Fitting sender and receiver statistics on TRAIN only...")
    sender_stats = train.groupBy("Sender_account").agg(
        F.count("*").alias("sender_tx_count"),
        F.approx_count_distinct("Receiver_account", 0.02).alias("sender_unique_receivers"),
        F.avg("Amount").alias("sender_avg_amount"),
        F.stddev_pop("Amount").alias("sender_std_amount"),
    )
    receiver_stats = train.groupBy("Receiver_account").agg(
        F.count("*").alias("receiver_tx_count"),
        F.approx_count_distinct("Sender_account", 0.02).alias("receiver_unique_senders"),
    )

    # Batch feature across a calendar day; not true transaction-by-transaction streaming.
    print("Calculating same-day batch activity...")
    daily_activity = df.groupBy("Sender_account", "_event_date").agg(
        F.count("*").alias("sender_day_count")
    )
    enriched = (
        df.join(sender_stats, on="Sender_account", how="left")
        .join(receiver_stats, on="Receiver_account", how="left")
        .join(daily_activity, on=["Sender_account", "_event_date"], how="left")
    )

    sender_location = F.upper(F.trim(F.coalesce(F.col("Sender_bank_location").cast("string"), F.lit(""))))
    receiver_location = F.upper(F.trim(F.coalesce(F.col("Receiver_bank_location").cast("string"), F.lit(""))))
    payment_type = F.upper(F.coalesce(F.col("Payment_type").cast("string"), F.lit("")))
    payment_currency = F.upper(F.trim(F.coalesce(F.col("Payment_currency").cast("string"), F.lit(""))))
    received_currency = F.upper(F.trim(F.coalesce(F.col("Received_currency").cast("string"), F.lit(""))))

    print("Applying explainable rules...")
    scored = (
        enriched
        .withColumn("rule_high_amount", (F.col("Amount") >= F.lit(amount_threshold)).cast("int"))
        .withColumn("rule_cross_border", ((sender_location != "") & (receiver_location != "") & (sender_location != receiver_location)).cast("int"))
        .withColumn("rule_cash_related", F.when(payment_type.contains("CASH"), 1).otherwise(0))
        .withColumn("rule_high_sender_activity", (F.coalesce(F.col("sender_tx_count"), F.lit(0)) >= SENDER_ACTIVITY_THRESHOLD).cast("int"))
        .withColumn("rule_high_receiver_activity", (F.coalesce(F.col("receiver_tx_count"), F.lit(0)) >= RECEIVER_ACTIVITY_THRESHOLD).cast("int"))
        .withColumn("rule_fan_out", (F.coalesce(F.col("sender_unique_receivers"), F.lit(0)) >= FANOUT_THRESHOLD).cast("int"))
        .withColumn("rule_fan_in", (F.coalesce(F.col("receiver_unique_senders"), F.lit(0)) >= FANIN_THRESHOLD).cast("int"))
        .withColumn("rule_transaction_burst", (F.coalesce(F.col("sender_day_count"), F.lit(0)) >= BURST_THRESHOLD).cast("int"))
        .withColumn("rule_currency_difference", ((payment_currency != "") & (received_currency != "") & (payment_currency != received_currency)).cast("int"))
        .withColumn(
            "rule_unusual_sender_amount",
            ((F.coalesce(F.col("sender_tx_count"), F.lit(0)) >= 3)
             & F.col("sender_avg_amount").isNotNull()
             & (F.col("Amount") > (F.col("sender_avg_amount") + F.lit(3.0) * F.coalesce(F.col("sender_std_amount"), F.lit(0.0))))).cast("int"),
        )
        .withColumn(
            "risk_score",
            2 * F.col("rule_high_amount")
            + F.col("rule_cross_border")
            + F.col("rule_cash_related")
            + F.col("rule_high_sender_activity")
            + F.col("rule_high_receiver_activity")
            # Fan-out and same-day burst remain visible as diagnostic flags, but
            # are excluded from the score until validation ablation demonstrates
            # that they improve precision/recall. Previous runs showed severe noise.
            + 2 * F.col("rule_fan_in")
            + F.col("rule_currency_difference")
            + F.col("rule_unusual_sender_amount"),
        )
    )

    candidates = {}
    for threshold in RISK_SCORE_CANDIDATES:
        candidates[f"risk_score_ge_{threshold}"] = F.col("risk_score") >= threshold
    for threshold in FANIN_CANDIDATES:
        candidates[f"fan_in_ge_{threshold}"] = (
            F.coalesce(F.col("receiver_unique_senders"), F.lit(0)) >= threshold
        )
    candidates["fan_in_ge_26_AND_currency_difference"] = (
        (F.coalesce(F.col("receiver_unique_senders"), F.lit(0)) >= 26)
        & (F.col("rule_currency_difference") == 1)
    )
    candidates["fan_in_ge_26_AND_cross_border"] = (
        (F.coalesce(F.col("receiver_unique_senders"), F.lit(0)) >= 26)
        & (F.col("rule_cross_border") == 1)
    )
    candidates["risk_score_ge_2_OR_fan_in_ge_26"] = (
        (F.col("risk_score") >= 2)
        | (F.coalesce(F.col("receiver_unique_senders"), F.lit(0)) >= 26)
    )
    candidates["risk_score_ge_3_OR_fan_in_ge_26"] = (
        (F.col("risk_score") >= 3)
        | (F.coalesce(F.col("receiver_unique_senders"), F.lit(0)) >= 26)
    )

    validation_scored = scored.filter(F.col("evaluation_split") == "VALIDATION")
    validation_total = validation_scored.count()
    validation_positives = int(
        validation_scored.agg(F.sum("Is_laundering").alias("positives")).first()["positives"] or 0
    )
    if validation_positives == 0:
        raise ValueError("Validation period has no laundering positives; rule selection is impossible.")

    print("Comparing candidate strategies on VALIDATION...")
    expressions = [F.count("*").alias("total")]
    for name, prediction in candidates.items():
        expressions.extend(add_metric_expressions(prediction, name))
    validation_row = validation_scored.agg(*expressions).first().asDict()

    validation_results = []
    for name in candidates:
        validation_results.append({
            "strategy": name,
            **unpack_metrics(validation_row, name, validation_total),
        })
    # F2 weights recall more than precision (beta=2), appropriate for a
    # screening system where missed suspicious transactions are costly.
    # Keep precision and alert volume visible so a high-recall strategy is not
    # mistaken for a low-noise one.
    validation_results.sort(
        key=lambda item: (item["f2_score"], item["precision"], item["recall"]),
        reverse=True,
    )
    selected = validation_results[0]
    selected_name = selected["strategy"]
    selected_prediction = candidates[selected_name]
    print(f"Selected strategy by validation F2: {selected_name}")
    print(
        f"Validation precision={selected['precision']:.4%}; "
        f"recall={selected['recall']:.4%}; F1={selected['f1_score']:.4%}; "
        f"F2={selected['f2_score']:.4%}"
    )

    # Keep the established output location and dashboard-required column names.
    scored = (
        scored
        .withColumn("selected_strategy", F.lit(selected_name))
        .withColumn("predicted_suspicious", selected_prediction.cast("int"))
        .withColumn(
            "risk_level",
            F.when(
                F.col("predicted_suspicious") == 1,
                F.when(F.col("risk_score") >= 7, "HIGH").otherwise("MEDIUM"),
            ).otherwise("LOW"),
        )
        .withColumn(
            "alert_reasons",
            F.concat_ws(", ",
                F.when(F.col("rule_high_amount") == 1, "High transaction amount"),
                F.when(F.col("rule_cross_border") == 1, "Different sender/receiver locations"),
                F.when(F.col("rule_cash_related") == 1, "Cash-related payment"),
                F.when(F.col("rule_high_sender_activity") == 1, "High sender activity"),
                F.when(F.col("rule_high_receiver_activity") == 1, "High receiver activity"),
                F.when(F.col("rule_fan_out") == 1, "Fan-out transaction pattern"),
                F.when(F.col("rule_fan_in") == 1, "Fan-in transaction pattern"),
                F.when(F.col("rule_transaction_burst") == 1, "Daily transaction burst"),
                F.when(F.col("rule_currency_difference") == 1, "Payment/received currency difference"),
                F.when(F.col("rule_unusual_sender_amount") == 1, "Unusual amount for sender"),
            ),
        )
        .drop("_event_date")
    )

    print(f"Writing scored transactions to {OUTPUT_PATH} ...")
    scored.write.mode("overwrite").parquet(OUTPUT_PATH)
    print("Scored Parquet write completed.")

    print("Evaluating the selected strategy on TEST...")
    saved_test = spark.read.parquet(OUTPUT_PATH).filter(F.col("evaluation_split") == "TEST")
    test_total = saved_test.count()
    test_row_raw = saved_test.agg(
        F.sum(F.when((F.col("predicted_suspicious") == 1) & (F.col("Is_laundering") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("predicted_suspicious") == 1) & (F.col("Is_laundering") == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("predicted_suspicious") == 0) & (F.col("Is_laundering") == 1), 1).otherwise(0)).alias("fn"),
        F.sum(F.when((F.col("predicted_suspicious") == 0) & (F.col("Is_laundering") == 0), 1).otherwise(0)).alias("tn"),
        F.sum("predicted_suspicious").alias("alerts"),
        F.sum("Is_laundering").alias("actual_laundering"),
    ).first().asDict()
    test_row = {
        "test_tp": test_row_raw["tp"], "test_fp": test_row_raw["fp"],
        "test_fn": test_row_raw["fn"], "test_tn": test_row_raw["tn"],
        "test_alerts": test_row_raw["alerts"],
    }
    test_metrics = unpack_metrics(test_row, "test", test_total)
    test_actual_laundering = int(test_row_raw["actual_laundering"] or 0)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "rule_validation_comparison.json"
    with validation_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "selection_metric": "F2-score; ties broken by precision then recall",
        "note": "Fan-out and same-day burst remain diagnostic flags but are excluded from risk_score pending validation ablation.",
            "selected_strategy": selected_name,
            "validation_transactions": validation_total,
            "validation_actual_laundering": validation_positives,
            "candidate_results_sorted_by_f1": validation_results,
            "important_note": (
                "Candidate selection uses validation labels. The later TEST period overlaps "
                "data examined in previous experiments, so it is not an untouched final benchmark."
            ),
        }, handle, indent=2)

    metrics = {
        "full_dataset_transactions": total_rows,
        "training_transactions": train_count,
        "validation_transactions": validation_count,
        "test_transactions": test_total,
        "train_cutoff_date": train_cutoff.isoformat(),
        "validation_cutoff_date": validation_cutoff.isoformat(),
        "test_actual_laundering": test_actual_laundering,
        **test_metrics,
        "selected_strategy": selected_name,
        "selection_metric": "F2-score (recall-weighted); ties broken by precision then recall",
        "validation_precision": selected["precision"],
        "validation_recall": selected["recall"],
        "validation_f1_score": selected["f1_score"],
        "validation_f2_score": selected["f2_score"],
        "amount_quantile": AMOUNT_QUANTILE,
        "amount_quantile_relative_error": AMOUNT_QUANTILE_RELATIVE_ERROR,
        "amount_threshold": amount_threshold,
        "sender_activity_threshold": SENDER_ACTIVITY_THRESHOLD,
        "receiver_activity_threshold": RECEIVER_ACTIVITY_THRESHOLD,
        "fanout_threshold": FANOUT_THRESHOLD,
        "fanin_feature_threshold": FANIN_THRESHOLD,
        "burst_threshold": BURST_THRESHOLD,
        "score_excluded_rules": ["rule_fan_out", "rule_transaction_burst"],
        "prediction_rule": selected_name,
        "method": (
            "Explainable rule-based detection using amount, transaction context, "
            "training-period account behaviour and network-pattern proxies; no machine learning. "
            "Fan-out and same-day burst flags are diagnostic-only pending validation ablation."
        ),
        "evaluation_note": (
            "Threshold and account statistics are fitted on TRAIN; strategy is selected on "
            "VALIDATION and evaluated on later TEST. The TEST period overlaps data examined "
            "in earlier experiments, so metrics are exploratory. Daily burst is a same-day "
            "batch feature, not transaction-by-transaction live scoring."
        ),
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\n" + "=" * 72)
    print("AML SENTINEL — SELECTED STRATEGY RESULTS")
    print("=" * 72)
    print(f"Selected strategy: {selected_name}")
    print(f"TEST transactions: {test_total:,}")
    print(f"TEST actual laundering: {test_actual_laundering:,}")
    print(f"TEST precision: {test_metrics['precision']:.4%}")
    print(f"TEST recall: {test_metrics['recall']:.4%}")
    print(f"TEST F1: {test_metrics['f1_score']:.4%}")
    print(f"TEST F2: {test_metrics['f2_score']:.4%}")
    print(f"TEST alerts: {test_metrics['predicted_alerts']:,}")
    print(f"TEST false positives: {test_metrics['false_positives']:,}")
    print(f"TEST false negatives: {test_metrics['false_negatives']:,}")
    print(f"Metrics: {metrics_path}")
    print(f"Validation comparison: {validation_path}")
    print(f"Scored Parquet: {OUTPUT_PATH}")
    print("=" * 72)
finally:
    spark.stop()
    print("Spark session stopped.")
