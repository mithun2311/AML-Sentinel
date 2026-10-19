
from pyspark.sql import SparkSession

HDFS_INPUT = "hdfs://localhost:9000/aml-sentinel/raw/SAML-D.csv"


def main():
    spark = (
        SparkSession.builder
        .appName("AML-Sentinel-HDFS-Test")
        .master("local[2]")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("\nConnecting to SAML-D in HDFS...")

        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(HDFS_INPUT)
        )

        print("\nSchema:")
        df.printSchema()

        print("\nFirst five transactions from HDFS:")
        df.show(5, truncate=False)

        print("\nHDFS read test successful.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
