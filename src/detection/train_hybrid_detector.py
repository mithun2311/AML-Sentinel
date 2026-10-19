"""Train and evaluate a causal behavioral AML candidate without replacing baseline data.

The candidate uses prior-event windows, TRAIN-only model fitting, and VALIDATION-only
threshold selection. TEST is reported for continuity but is already explored.
"""
import csv
import json
import math
import os
import time
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import DoubleType


HDFS_BASE = "hdfs://localhost:9000/aml-sentinel"
INPUT_PATH = os.getenv(
    "AML_INPUT_PATH", f"{HDFS_BASE}/processed/transactions_parquet"
)
BASELINE_PATH = os.getenv(
    "AML_BASELINE_PATH", f"{HDFS_BASE}/results/scored_transactions"
)
RUN_ID = os.getenv("AML_CANDIDATE_RUN_ID", "hybrid_v1")
CANDIDATE_PATH = os.getenv(
    "AML_CANDIDATE_PATH",
    f"{HDFS_BASE}/results/scored_transactions_{RUN_ID}",
)
MODEL_PATH = os.getenv(
    "AML_CANDIDATE_MODEL_PATH",
    f"{HDFS_BASE}/models/hybrid_{RUN_ID}",
)
OUTPUT_DIR = Path(os.getenv("AML_CANDIDATE_OUTPUT_DIR", "outputs")) / RUN_ID
ALERT_BUDGETS = (0.0002, 0.0005, 0.001)
RUN_ABLATIONS = os.getenv("AML_RUN_ABLATIONS", "true").strip().lower() in {
    "1", "true", "yes", "on"
}

FEATURES = [
    "log_amount",
    "log_sender_tx_1d",
    "log_sender_tx_7d",
    "log_sender_tx_30d",
    "log_sender_amount_1d",
    "log_sender_amount_7d",
    "log_sender_amount_30d",
    "log_receiver_tx_1d",
    "log_receiver_tx_30d",
    "log_receiver_amount_1d",
    "log_receiver_amount_30d",
    "new_counterparty",
    "log_new_counterparties_7d",
    "log_new_counterparties_30d",
    "new_receiver_sender",
    "log_new_senders_7d",
    "log_new_senders_30d",
    "amount_vs_sender_30d_mean",
    "rule_cross_border",
    "rule_currency_difference",
    "rule_cash_related",
    "rule_structuring",
]


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(frame, split, prediction):
    selected = frame.filter(F.col("evaluation_split") == split)
    pred = prediction.cast("int")
    label = F.col("Is_laundering").cast("int")
    row = selected.agg(
        F.count("*").alias("total"),
        F.sum(F.when(label == 1, 1).otherwise(0)).alias("positives"),
        F.sum(F.when((pred == 1) & (label == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((pred == 1) & (label == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((pred == 0) & (label == 1), 1).otherwise(0)).alias("fn"),
        F.sum(F.when((pred == 0) & (label == 0), 1).otherwise(0)).alias("tn"),
        F.sum(pred).alias("alerts"),
    ).first().asDict()

    tp, fp, fn, tn = (int(row[key] or 0) for key in ("tp", "fp", "fn", "tn"))
    alerts = int(row["alerts"] or 0)
    total = int(row["total"] or 0)
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    return {
        "transactions": total,
        "actual_laundering": int(row["positives"] or 0),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "alerts": alerts,
        "precision": precision,
        "recall": recall,
        "f1_score": safe_rate(2 * precision * recall, precision + recall),
        "f2_score": safe_rate(5 * precision * recall, 4 * precision + recall),
        "false_positive_rate": safe_rate(fp, fp + tn),
        "alert_rate": safe_rate(alerts, total),
    }


def budget_threshold(validation, budget):
    total = validation.count()
    max_alerts = max(1, math.floor(total * budget))
    ranked = (
        validation.select("model_probability")
        .orderBy(F.desc("model_probability"))
        .limit(max_alerts + 1)
        .collect()
    )
    if len(ranked) <= max_alerts:
        return 0.0
    # Strictly exclude the first score outside the budget, including ties.
    return math.nextafter(float(ranked[max_alerts]["model_probability"]), math.inf)


def pr_auc(frame, split):
    selected = frame.filter(F.col("evaluation_split") == split)
    evaluator = BinaryClassificationEvaluator(
        labelCol="Is_laundering",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )
    return float(evaluator.evaluate(selected))


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fit_model(training_frame, feature_columns, positive_weight):
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="_raw_features",
        handleInvalid="error",
    )
    scaler = StandardScaler(
        inputCol="_raw_features",
        outputCol="features",
        withMean=False,
        withStd=True,
    )
    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="_model_label",
        weightCol="_class_weight",
        maxIter=40,
        regParam=0.05,
        elasticNetParam=0.0,
        standardization=False,
    )
    return Pipeline(stages=[assembler, scaler, classifier]).fit(training_frame)


spark = (
    SparkSession.builder.appName("AML-Sentinel-Causal-Hybrid-Candidate")
    .master("local[2]")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
    .config("spark.sql.shuffle.partitions", "96")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
started_at = time.perf_counter()

try:
    hadoop_conf = spark._jsc.hadoopConfiguration()
    candidate_hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(CANDIDATE_PATH)
    model_hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(MODEL_PATH)
    if candidate_hdfs_path.getFileSystem(hadoop_conf).exists(candidate_hdfs_path):
        raise FileExistsError(
            f"Candidate output already exists: {CANDIDATE_PATH}. Set "
            "AML_CANDIDATE_RUN_ID to a new value to preserve previous runs."
        )
    if model_hdfs_path.getFileSystem(hadoop_conf).exists(model_hdfs_path):
        raise FileExistsError(
            f"Candidate model already exists: {MODEL_PATH}. Set "
            "AML_CANDIDATE_RUN_ID to a new value to preserve previous runs."
        )
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Candidate report directory already exists: {OUTPUT_DIR}. Set "
            "AML_CANDIDATE_RUN_ID to a new value to preserve previous runs."
        )

    print(f"Reading cleaned data: {INPUT_PATH}")
    data = spark.read.parquet(INPUT_PATH)
    required = {
        "Sender_account", "Receiver_account", "Amount", "Date",
        "Payment_type", "Payment_currency", "Received_currency",
        "Sender_bank_location", "Receiver_bank_location", "Is_laundering",
        "Laundering_type",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Cleaned Parquet is missing required columns: {missing}")

    if "Transaction_timestamp" in data.columns:
        event_ts = F.col("Transaction_timestamp").cast("timestamp")
    elif "Time" in data.columns:
        event_ts = F.to_timestamp(
            F.concat_ws(" ", F.col("Date").cast("string"), F.col("Time").cast("string")),
            "yyyy-MM-dd HH:mm:ss",
        )
    else:
        event_ts = F.to_timestamp(F.col("Date").cast("string"))

    data = (
        data.withColumn("Amount", F.col("Amount").cast(DoubleType()))
        .withColumn("Is_laundering", F.col("Is_laundering").cast("int"))
        .withColumn("_model_label", F.col("Is_laundering").cast("double"))
        .withColumn("_event_ts", event_ts)
        .withColumn("_event_date", F.to_date("_event_ts"))
        .withColumn("_event_epoch", F.col("_event_ts").cast("long"))
    )
    if data.filter(F.col("_event_epoch").isNull()).limit(1).count():
        raise ValueError("Transaction timestamps contain null/unparseable values.")

    offsets = data.select(
        F.datediff("_event_date", F.lit("1970-01-01")).alias("day_offset")
    ).distinct()
    cutoffs = offsets.agg(
        F.expr("percentile_approx(day_offset, 0.60, 10000)").alias("train_cutoff"),
        F.expr("percentile_approx(day_offset, 0.80, 10000)").alias("validation_cutoff"),
    ).first()
    train_cutoff = int(cutoffs["train_cutoff"])
    validation_cutoff = int(cutoffs["validation_cutoff"])
    data = data.withColumn(
        "evaluation_split",
        F.when(
            F.datediff("_event_date", F.lit("1970-01-01")) <= train_cutoff,
            "TRAIN",
        ).when(
            F.datediff("_event_date", F.lit("1970-01-01")) <= validation_cutoff,
            "VALIDATION",
        ).otherwise("TEST"),
    )

    print("Fitting structuring reference thresholds on TRAIN...")
    train = data.filter(F.col("evaluation_split") == "TRAIN")
    amount_quantile = train.approxQuantile("Amount", [0.995], 0.001)[0]
    daily_flow = (
        train.groupBy("Sender_account", "_event_date")
        .agg(F.sum("Amount").alias("_sender_daily_amount"))
    )
    daily_flow_quantile = daily_flow.approxQuantile(
        "_sender_daily_amount", [0.995], 0.001
    )[0]
    if amount_quantile is None or daily_flow_quantile is None:
        raise ValueError("Could not fit TRAIN-only structuring reference thresholds.")

    print("Building causal account and counterparty windows...")
    pair_first_seen = data.groupBy("Sender_account", "Receiver_account").agg(
        F.min("_event_epoch").alias("_pair_first_epoch")
    )
    receiver_pair_first_seen = data.groupBy(
        "Receiver_account", "Sender_account"
    ).agg(F.min("_event_epoch").alias("_receiver_pair_first_epoch"))
    for days in (7, 30):
        seconds = days * 24 * 60 * 60
        pair_first_seen = pair_first_seen.withColumn(
            f"new_counterparties_{days}d",
            F.count(F.lit(1)).over(
                Window.partitionBy("Sender_account")
                .orderBy("_pair_first_epoch")
                .rangeBetween(-seconds, -1)
            ),
        )
        receiver_pair_first_seen = receiver_pair_first_seen.withColumn(
            f"new_senders_{days}d",
            F.count(F.lit(1)).over(
                Window.partitionBy("Receiver_account")
                .orderBy("_receiver_pair_first_epoch")
                .rangeBetween(-seconds, -1)
            ),
        )
    features = (
        data.join(pair_first_seen, ["Sender_account", "Receiver_account"], "left")
        .join(
            receiver_pair_first_seen,
            ["Receiver_account", "Sender_account"],
            "left",
        )
        .withColumn(
            "new_counterparty",
            (F.col("_event_epoch") == F.col("_pair_first_epoch")).cast("int"),
        )
        .withColumn(
            "new_receiver_sender",
            (F.col("_event_epoch") == F.col("_receiver_pair_first_epoch")).cast("int"),
        )
    )

    amount = F.coalesce(F.col("Amount"), F.lit(0.0))
    for days in (1, 7, 30):
        seconds = days * 24 * 60 * 60
        sender_window = (
            Window.partitionBy("Sender_account")
            .orderBy("_event_epoch")
            .rangeBetween(-seconds, -1)
        )
        receiver_window = (
            Window.partitionBy("Receiver_account")
            .orderBy("_event_epoch")
            .rangeBetween(-seconds, -1)
        )
        features = (
            features
            .withColumn(
                f"sender_tx_{days}d",
                F.count(F.lit(1)).over(sender_window),
            )
            .withColumn(
                f"sender_amount_{days}d",
                F.coalesce(F.sum(amount).over(sender_window), F.lit(0.0)),
            )
        )
        if days in (1, 30):
            features = (
                features
                .withColumn(
                    f"receiver_tx_{days}d",
                    F.count(F.lit(1)).over(receiver_window),
                )
                .withColumn(
                    f"receiver_amount_{days}d",
                    F.coalesce(F.sum(amount).over(receiver_window), F.lit(0.0)),
                )
            )

    payment_type = F.upper(
        F.coalesce(F.col("Payment_type").cast("string"), F.lit(""))
    )
    sender_location = F.upper(
        F.trim(F.coalesce(F.col("Sender_bank_location").cast("string"), F.lit("")))
    )
    receiver_location = F.upper(
        F.trim(F.coalesce(F.col("Receiver_bank_location").cast("string"), F.lit("")))
    )
    payment_currency = F.upper(
        F.trim(F.coalesce(F.col("Payment_currency").cast("string"), F.lit("")))
    )
    received_currency = F.upper(
        F.trim(F.coalesce(F.col("Received_currency").cast("string"), F.lit("")))
    )
    features = (
        features
        .withColumn(
            "rule_cross_border",
            ((sender_location != "") & (receiver_location != "") &
             (sender_location != receiver_location)).cast("int"),
        )
        .withColumn("rule_cash_related", payment_type.contains("CASH").cast("int"))
        .withColumn(
            "rule_currency_difference",
            ((payment_currency != "") & (received_currency != "") &
             (payment_currency != received_currency)).cast("int"),
        )
        .withColumn(
            "rule_structuring",
            (
                (F.col("Amount") < F.lit(float(amount_quantile)))
                & (F.col("sender_tx_1d") >= 2)
                & ((F.col("sender_amount_1d") + amount) >= F.lit(float(daily_flow_quantile)))
            ).cast("int"),
        )
        .withColumn(
            "amount_vs_sender_30d_mean",
            F.log1p(amount)
            - F.log1p(
                F.col("sender_amount_30d")
                / F.greatest(F.col("sender_tx_30d").cast("double"), F.lit(1.0))
            ),
        )
    )

    log_sources = {
        "log_amount": "Amount",
        "log_sender_tx_1d": "sender_tx_1d",
        "log_sender_tx_7d": "sender_tx_7d",
        "log_sender_tx_30d": "sender_tx_30d",
        "log_sender_amount_1d": "sender_amount_1d",
        "log_sender_amount_7d": "sender_amount_7d",
        "log_sender_amount_30d": "sender_amount_30d",
        "log_receiver_tx_1d": "receiver_tx_1d",
        "log_receiver_tx_30d": "receiver_tx_30d",
        "log_receiver_amount_1d": "receiver_amount_1d",
        "log_receiver_amount_30d": "receiver_amount_30d",
        "log_new_counterparties_7d": "new_counterparties_7d",
        "log_new_counterparties_30d": "new_counterparties_30d",
        "log_new_senders_7d": "new_senders_7d",
        "log_new_senders_30d": "new_senders_30d",
    }
    for output, source in log_sources.items():
        features = features.withColumn(
            output,
            F.log1p(F.greatest(F.coalesce(F.col(source).cast("double"), F.lit(0.0)), F.lit(0.0))),
        )

    train_features = features.filter(F.col("evaluation_split") == "TRAIN")
    class_counts = train_features.agg(
        F.sum(F.when(F.col("Is_laundering") == 1, 1).otherwise(0)).alias("positive"),
        F.sum(F.when(F.col("Is_laundering") == 0, 1).otherwise(0)).alias("negative"),
    ).first()
    positives = int(class_counts["positive"] or 0)
    negatives = int(class_counts["negative"] or 0)
    if positives == 0:
        raise ValueError("TRAIN contains no positive labels; supervised fit is impossible.")
    positive_weight = min(100.0, negatives / positives)
    train_features = train_features.withColumn(
        "_class_weight",
        F.when(F.col("Is_laundering") == 1, F.lit(positive_weight)).otherwise(F.lit(1.0)),
    )

    print(
        f"Fitting class-weighted LogisticRegression on TRAIN only "
        f"({positives:,} positive, {negatives:,} negative; positive weight={positive_weight:.2f})..."
    )
    model = fit_model(train_features, FEATURES, positive_weight)

    print("Scoring all chronological splits...")
    scored = model.transform(features).withColumn(
        "model_probability", vector_to_array("probability")[1]
    )
    validation = scored.filter(F.col("evaluation_split") == "VALIDATION")
    if validation.filter(F.col("Is_laundering") == 1).limit(1).count() == 0:
        raise ValueError("VALIDATION contains no positive labels; threshold selection is impossible.")

    budget_results = []
    for budget in ALERT_BUDGETS:
        threshold = budget_threshold(validation, budget)
        validation_result = confusion_metrics(
            scored, "VALIDATION", F.col("model_probability") > F.lit(threshold)
        )
        test_result = confusion_metrics(
            scored, "TEST", F.col("model_probability") > F.lit(threshold)
        )
        budget_results.append({
            "validation_alert_budget": budget,
            "validation_threshold": threshold,
            "validation": validation_result,
            "test": test_result,
        })

    selected_budget = max(
        budget_results,
        key=lambda item: (
            item["validation"]["f2_score"],
            item["validation"]["precision"],
            -item["validation"]["alerts"],
        ),
    )
    selected_threshold = selected_budget["validation_threshold"]
    ablation_rows = [{
        "ablation": "full_model",
        "feature_count": len(FEATURES),
        "validation_threshold": selected_threshold,
        "validation_alert_budget": selected_budget["validation_alert_budget"],
        **selected_budget["validation"],
        "pr_auc": pr_auc(
            validation.select("evaluation_split", "Is_laundering", "rawPrediction"),
            "VALIDATION",
        ),
    }]
    full_budget_threshold = budget_threshold(validation, 0.001)
    ablation_rows.append({
        "ablation": "full_model_at_0.1pct_budget",
        "feature_count": len(FEATURES),
        "validation_threshold": full_budget_threshold,
        "validation_alert_budget": 0.001,
        **confusion_metrics(
            scored,
            "VALIDATION",
            F.col("model_probability") > F.lit(full_budget_threshold),
        ),
        "pr_auc": pr_auc(
            validation.select("evaluation_split", "Is_laundering", "rawPrediction"),
            "VALIDATION",
        ),
    })
    ablation_features = {
        "without_temporal_behavior": [
            "log_amount", "rule_cross_border", "rule_currency_difference",
            "rule_cash_related",
        ],
        "without_new_counterparty_features": [
            feature for feature in FEATURES
            if feature not in {
                "new_counterparty", "log_new_counterparties_7d",
                "log_new_counterparties_30d", "new_receiver_sender",
                "log_new_senders_7d", "log_new_senders_30d",
            }
        ],
        "without_structuring_rule": [
            feature for feature in FEATURES if feature != "rule_structuring"
        ],
    }
    if RUN_ABLATIONS:
        for ablation_name, feature_columns in ablation_features.items():
            print(f"Retraining validation ablation: {ablation_name}...")
            ablation_model = fit_model(
                train_features, feature_columns, positive_weight
            )
            ablation_validation = ablation_model.transform(
                features.filter(F.col("evaluation_split") == "VALIDATION")
            ).withColumn(
                "model_probability", vector_to_array("probability")[1]
            )
            ablation_threshold = budget_threshold(ablation_validation, 0.001)
            ablation_rows.append({
                "ablation": ablation_name,
                "feature_count": len(feature_columns),
                "validation_threshold": ablation_threshold,
                "validation_alert_budget": 0.001,
                **confusion_metrics(
                    ablation_validation,
                    "VALIDATION",
                    F.col("model_probability") > F.lit(ablation_threshold),
                ),
                "pr_auc": pr_auc(ablation_validation, "VALIDATION"),
            })
    scored = (
        scored
        .withColumn(
            "predicted_suspicious",
            (F.col("model_probability") > F.lit(selected_threshold)).cast("int"),
        )
        .withColumn("selected_strategy", F.lit("hybrid_logistic_validation_f2"))
        .withColumn(
            "risk_score",
            F.col("rule_structuring") * 3
            + F.col("new_counterparty") * 2
            + F.col("rule_cash_related")
            + F.col("rule_cross_border")
            + F.col("rule_currency_difference")
            + (F.col("log_new_senders_7d") >= F.lit(math.log1p(3))).cast("int")
            + (F.col("log_new_counterparties_7d") >= F.lit(math.log1p(3))).cast("int"),
        )
        .withColumn(
            "risk_level",
            F.when(F.col("predicted_suspicious") == 0, "LOW")
            .when(F.col("model_probability") >= 0.5, "HIGH")
            .otherwise("MEDIUM"),
        )
        .withColumn(
            "alert_reasons",
            F.concat_ws(", ",
                F.when(F.col("rule_structuring") == 1, "Repeated sub-threshold transfers approaching high daily flow"),
                F.when(F.col("new_counterparty") == 1, "First observed sender-counterparty relationship"),
                F.when(F.col("log_new_counterparties_7d") >= math.log1p(3), "Several new counterparties in prior 7 days"),
                F.when(F.col("log_new_senders_7d") >= math.log1p(3), "Several new senders in prior 7 days"),
                F.when(F.col("rule_cash_related") == 1, "Cash-related payment context"),
                F.when(F.col("rule_cross_border") == 1, "Different sender/receiver locations"),
                F.when(F.col("rule_currency_difference") == 1, "Payment/received currency difference"),
                F.when(
                    F.col("predicted_suspicious") == 1,
                    "Hybrid model exceeded validation-selected score cutoff",
                ),
            ),
        )
    )

    print(f"Writing candidate scored data to new HDFS path: {CANDIDATE_PATH}")
    scored.write.mode("errorifexists").parquet(CANDIDATE_PATH)
    print(f"Saving fitted model to new HDFS path: {MODEL_PATH}")
    model.write().save(MODEL_PATH)

    baseline = spark.read.parquet(BASELINE_PATH)
    baseline_required = {
        "evaluation_split", "Is_laundering", "predicted_suspicious", "risk_score"
    }
    baseline_missing = sorted(baseline_required - set(baseline.columns))
    if baseline_missing:
        raise ValueError(f"Baseline scored data is missing columns: {baseline_missing}")

    selected_metrics = {
        "VALIDATION": confusion_metrics(
            scored, "VALIDATION", F.col("predicted_suspicious") == 1
        ),
        "TEST": confusion_metrics(
            scored, "TEST", F.col("predicted_suspicious") == 1
        ),
    }
    baseline_metrics = {
        split: confusion_metrics(
            baseline, split, F.col("predicted_suspicious") == 1
        )
        for split in ("VALIDATION", "TEST")
    }

    candidate_with_probability = scored.select(
        "evaluation_split", "Is_laundering", "rawPrediction"
    )
    pr_auc_values = {
        split: pr_auc(candidate_with_probability, split)
        for split in ("VALIDATION", "TEST")
    }
    baseline_with_score = baseline.withColumn(
        "rawPrediction",
        array_to_vector(
            F.array(F.lit(0.0), F.col("risk_score").cast("double"))
        ),
    )
    baseline_pr_auc = {
        split: pr_auc(baseline_with_score, split)
        for split in ("VALIDATION", "TEST")
    }

    typology_rows = []
    for detector, detector_frame in (("baseline", baseline), ("hybrid_candidate", scored)):
        for split in ("VALIDATION", "TEST"):
            grouped = (
                detector_frame.filter(F.col("evaluation_split") == split)
                .groupBy("Laundering_type")
                .agg(
                    F.count("*").alias("transactions"),
                    F.sum(F.when(F.col("Is_laundering") == 1, 1).otherwise(0)).alias("actual_positive"),
                    F.sum(F.when((F.col("Is_laundering") == 1) & (F.col("predicted_suspicious") == 1), 1).otherwise(0)).alias("true_positives"),
                    F.sum(F.when((F.col("Is_laundering") == 0) & (F.col("predicted_suspicious") == 1), 1).otherwise(0)).alias("false_positives"),
                )
                .collect()
            )
            for row in grouped:
                actual = int(row["actual_positive"] or 0)
                tp = int(row["true_positives"] or 0)
                fp = int(row["false_positives"] or 0)
                typology_rows.append({
                    "detector": detector,
                    "split": split,
                    "Laundering_type": row["Laundering_type"],
                    "transactions": int(row["transactions"]),
                    "actual_positive": actual,
                    "true_positives": tp,
                    "false_positives": fp,
                    "precision": safe_rate(tp, tp + fp),
                    "recall": safe_rate(tp, actual),
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(
        OUTPUT_DIR / "typology_recall.csv",
        typology_rows,
        ["detector", "split", "Laundering_type", "transactions", "actual_positive", "true_positives", "false_positives", "precision", "recall"],
    )
    budget_rows = []
    for item in budget_results:
        for split in ("VALIDATION", "TEST"):
            budget_rows.append({
                "validation_alert_budget_percent": item["validation_alert_budget"] * 100,
                "validation_selected_threshold": item["validation_threshold"],
                "split": split,
                **item[split.lower()],
            })
    write_csv(
        OUTPUT_DIR / "alert_budget_comparison.csv",
        budget_rows,
        ["validation_alert_budget_percent", "validation_selected_threshold", "split", "transactions", "actual_laundering", "true_positives", "false_positives", "false_negatives", "true_negatives", "alerts", "precision", "recall", "f1_score", "f2_score", "false_positive_rate", "alert_rate"],
    )

    write_csv(
        OUTPUT_DIR / "feature_ablation_validation.csv",
        ablation_rows,
        ["ablation", "feature_count", "validation_threshold", "validation_alert_budget", "transactions", "actual_laundering", "true_positives", "false_positives", "false_negatives", "true_negatives", "alerts", "precision", "recall", "f1_score", "f2_score", "false_positive_rate", "alert_rate", "pr_auc"],
    )

    report = {
        "run_id": RUN_ID,
        "candidate_hdfs_path": CANDIDATE_PATH,
        "candidate_model_hdfs_path": MODEL_PATH,
        "baseline_hdfs_path": BASELINE_PATH,
        "input_hdfs_path": INPUT_PATH,
        "fit_split": "TRAIN",
        "threshold_selection_split": "VALIDATION",
        "selected_validation_budget": selected_budget["validation_alert_budget"],
        "selected_threshold": selected_threshold,
        "selected_strategy": "hybrid_logistic_validation_f2",
        "class_weight_positive": positive_weight,
        "train_structuring_amount_quantile": float(amount_quantile),
        "train_daily_flow_quantile": float(daily_flow_quantile),
        "feature_columns": FEATURES,
        "baseline": baseline_metrics,
        "candidate": selected_metrics,
        "candidate_pr_auc": pr_auc_values,
        "baseline_pr_auc": baseline_pr_auc,
        "alert_budget_comparison": budget_results,
        "test_was_used_for_selection": False,
        "test_limitation": (
            "TEST results are exploratory because this period was examined in prior experiments. "
            "No threshold, feature, or model choice was made using TEST metrics."
        ),
        "ablation_note": (
            "Feature-group rows are separately retrained on TRAIN and measured on VALIDATION at a "
            "0.1% validation alert budget. The full model is evaluated at the budget selected by "
            "VALIDATION F2. Set AML_RUN_ABLATIONS=false to skip the additional fits."
        ),
        "ablations_run": RUN_ABLATIONS,
        "runtime_seconds": time.perf_counter() - started_at,
        "feature_causality_note": (
            "Rolling windows exclude the current event timestamp and include only events in the prior "
            "1, 7, or 30 days. First-seen counterpart flags are derived from pair minimum timestamps; "
            "equal-timestamp transactions are treated as a simultaneous batch."
        ),
    }
    with (OUTPUT_DIR / "candidate_evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Baseline VALIDATION F2: {baseline_metrics['VALIDATION']['f2_score']:.4%}")
    print(f"Candidate VALIDATION F2: {selected_metrics['VALIDATION']['f2_score']:.4%}")
    print(f"Baseline TEST F2 (exploratory): {baseline_metrics['TEST']['f2_score']:.4%}")
    print(f"Candidate TEST F2 (exploratory): {selected_metrics['TEST']['f2_score']:.4%}")
    print(f"Candidate PR-AUC: {pr_auc_values}")
    print(f"Model: {MODEL_PATH}")
    print(f"Reports: {OUTPUT_DIR}")
finally:
    spark.stop()