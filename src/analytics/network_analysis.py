
import json
import os

import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum

HDFS_SCORED = os.getenv(
    "AML_SCORED_PATH",
    "hdfs://localhost:9000/aml-sentinel/results/scored_transactions",
)
ANALYSIS_SPLIT = os.getenv("AML_NETWORK_SPLIT", "VALIDATION").strip().upper()
if ANALYSIS_SPLIT not in {"TRAIN", "VALIDATION", "TEST", "ALL"}:
    raise ValueError("AML_NETWORK_SPLIT must be TRAIN, VALIDATION, TEST, or ALL")


def main():
    os.makedirs("outputs", exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("AML-Sentinel-Network-Analytics")
        .master("local[*]")
        .config(
            "spark.hadoop.dfs.client.use.datanode.hostname", "true"
        )
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("[1/5] Loading scored transactions...")
        df = spark.read.parquet(HDFS_SCORED)
        if ANALYSIS_SPLIT != "ALL":
            if "evaluation_split" not in df.columns:
                raise ValueError("Scored data has no evaluation_split column")
            df = df.filter(col("evaluation_split") == ANALYSIS_SPLIT)

        print("[2/5] Aggregating suspicious account connections...")
        edges = (
            df.filter(col("predicted_suspicious") == 1)
            .groupBy("Sender_account", "Receiver_account")
            .agg(
                count("*").alias("transaction_count"),
                spark_sum("Amount").alias("total_amount"),
                spark_sum("risk_score").alias("combined_risk_score"),
            )
            .orderBy(col("combined_risk_score").desc())
            .limit(50000)
        )

        edge_rows = edges.collect()

        print(f"Account-pair records selected: {len(edge_rows)}")

        print("[3/5] Building directed transaction graph...")
        graph = nx.DiGraph()

        for row in edge_rows:
            sender = str(row["Sender_account"])
            receiver = str(row["Receiver_account"])

            graph.add_edge(
                sender,
                receiver,
                transaction_count=int(row["transaction_count"]),
                total_amount=float(row["total_amount"] or 0),
                weight=float(row["transaction_count"]),
            )

        print("[4/5] Calculating network metrics...")
        degree_centrality = nx.degree_centrality(graph)

        # PageRank can take time on large graphs; use the selected graph.
        pagerank = (
            nx.pagerank(graph, weight="weight", max_iter=50)
            if graph.number_of_nodes() > 0 else {}
        )

        account_rows = []
        for account in graph.nodes():
            account_rows.append({
                "account": account,
                "in_degree": int(graph.in_degree(account)),
                "out_degree": int(graph.out_degree(account)),
                "degree_centrality": float(
                    degree_centrality.get(account, 0.0)
                ),
                "pagerank": float(pagerank.get(account, 0.0)),
            })

        account_rows.sort(
            key=lambda item: item["pagerank"], reverse=True
        )

        accounts_path = "outputs/network_top_accounts.csv"
        with open(accounts_path, "w", newline="", encoding="utf-8") as file:
            import csv
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "account", "in_degree", "out_degree",
                    "degree_centrality", "pagerank",
                ],
            )
            writer.writeheader()
            writer.writerows(account_rows[:1000])

        summary = {
            "network_type": "directed suspicious-transaction account network",
            "suspicious_account_pair_records": len(edge_rows),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "top_accounts": account_rows[:20],
            "note": (
                "Network is built from rule-engine alerts, limited to the "
                "50,000 highest combined-risk account pairs. It is not the "
                "complete transaction network."
            ),
        }

        with open(
            "outputs/network_summary.json", "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)

        print("\nTop 10 accounts by PageRank:")
        for account in account_rows[:10]:
            print(
                f"{account['account']}: "
                f"PageRank={account['pagerank']:.6f}, "
                f"in={account['in_degree']}, "
                f"out={account['out_degree']}"
            )

        print("\n[5/5] Network analysis completed.")
        print("Account results: outputs/network_top_accounts.csv")
        print("Network summary: outputs/network_summary.json")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
