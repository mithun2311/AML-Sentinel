
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    regexp_replace,
    to_date,
    to_timestamp,
    concat_ws,
    when,
    sum as spark_sum,
    count,
)
from pyspark.sql.types import DoubleType, IntegerType

HDFS_INPUT = "hdfs://localhost:9000/aml-sentinel/raw/SAML-D.csv"

HDFS_CLEAN = (
    "hdfs://localhost:9000/aml-sentinel/processed/transactions_parquet"
)
HDFS_REJECTED = (
    "hdfs://localhost:9000/aml-sentinel/processed/rejected_records"
)


def main():
    spark = (
        SparkSession.builder
        .appName("AML-Sentinel-Preprocessing")
        .master("local[*]")
        .config(
            "spark.hadoop.dfs.client.use.datanode.hostname",
            "true",
        )
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("\n[1/6] Reading the original dataset from HDFS...")

        raw = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(HDFS_INPUT)
        )

        print("\nOriginal schema:")
        raw.printSchema()

        print("\n[2/6] Cleaning and converting data types...")

        cleaned = (
            raw
            .withColumn("Time", trim(col("Time")))
            .withColumn("Date", to_date(trim(col("Date")), "yyyy-MM-dd"))
            .withColumn(
                "Amount",
                regexp_replace(
                    trim(col("Amount")), ",", ""
                ).cast(DoubleType()),
            )
            .withColumn(
                "Is_laundering",
                trim(col("Is_laundering")).cast(IntegerType()),
            )
            .withColumn("Sender_account", trim(col("Sender_account")))
            .withColumn("Receiver_account", trim(col("Receiver_account")))
            .withColumn("Payment_type", trim(col("Payment_type")))
            .withColumn("Laundering_type", trim(col("Laundering_type")))
            .withColumn(
                "Transaction_timestamp",
                to_timestamp(
                    concat_ws(" ", col("Date"), col("Time")),
                    "yyyy-MM-dd HH:mm:ss",
                ),
            )
        )

        missing_sender = (
            col("Sender_account").isNull()
            | (col("Sender_account") == "")
        )
        missing_receiver = (
            col("Receiver_account").isNull()
            | (col("Receiver_account") == "")
        )

        invalid_condition = (
            missing_sender
            | missing_receiver
            | col("Amount").isNull()
            | (col("Amount") < 0)
            | col("Is_laundering").isNull()
            | (~col("Is_laundering").isin(0, 1))
            | col("Date").isNull()
            | col("Transaction_timestamp").isNull()
        )

        rejected = (
            cleaned
            .filter(invalid_condition)
            .withColumn(
                "Rejection_reason",
                when(missing_sender, "Missing sender account")
                .when(missing_receiver, "Missing receiver account")
                .when(col("Amount").isNull(), "Invalid amount")
                .when(col("Amount") < 0, "Negative amount")
                .when(
                    col("Is_laundering").isNull()
                    | (~col("Is_laundering").isin(0, 1)),
                    "Invalid laundering label",
                )
                .when(col("Date").isNull(), "Invalid date")
                .otherwise("Invalid transaction timestamp"),
            )
        )

        valid = cleaned.filter(~invalid_condition)

        print("\n[3/6] Saving valid transactions as Parquet...")
        (
            valid.write
            .mode("overwrite")
            .parquet(HDFS_CLEAN)
        )

        print("\n[4/6] Saving rejected records separately...")
        (
            rejected.write
            .mode("overwrite")
            .parquet(HDFS_REJECTED)
        )

        print("\n[5/6] Verifying processed output...")

        result = spark.read.parquet(HDFS_CLEAN)

        summary = result.agg(
            count("*").alias("total_transactions"),
            spark_sum(
                when(col("Is_laundering") == 1, 1).otherwise(0)
            ).alias("laundering_transactions"),
            spark_sum(
                when(col("Is_laundering") == 0, 1).otherwise(0)
            ).alias("normal_transactions"),
            spark_sum("Amount").alias("total_transaction_amount"),
        )

        summary.show(truncate=False)

        print("\nTransactions by laundering label:")
        (
            result.groupBy("Is_laundering")
            .count()
            .orderBy("Is_laundering")
            .show()
        )

        print("\n[6/6] Preprocessing completed successfully.")
        print(f"Clean Parquet path: {HDFS_CLEAN}")
        print(f"Rejected records path: {HDFS_REJECTED}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
