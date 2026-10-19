
import csv
import json
import os
from pathlib import Path

import networkx as nx
from pyspark.sql import SparkSession, functions as F

HDFS_PATH = (
    os.getenv(
        "AML_SCORED_PATH",
        "hdfs://localhost:9000/aml-sentinel/results/scored_transactions",
    )
)

MAX_EDGES = 50_000
ANALYSIS_SPLIT = os.getenv("AML_NETWORK_SPLIT", "VALIDATION").strip().upper()
if ANALYSIS_SPLIT not in {"TRAIN", "VALIDATION", "TEST", "ALL"}:
    raise ValueError("AML_NETWORK_SPLIT must be TRAIN, VALIDATION, TEST, or ALL")

spark = (
    SparkSession.builder
    .appName("AML-Sentinel-Pattern-Graph-Analysis")
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
    print("Reading scored transactions from HDFS...")
    df = spark.read.parquet(HDFS_PATH)

    required = [
        "Sender_account",
        "Receiver_account",
        "Amount",
        "risk_score",
        "predicted_suspicious",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "evaluation_split" in df.columns and ANALYSIS_SPLIT != "ALL":
        df = df.filter(F.col("evaluation_split") == ANALYSIS_SPLIT)
    elif ANALYSIS_SPLIT != "ALL" and "evaluation_split" not in df.columns:
        raise ValueError("Scored data has no evaluation_split column; set AML_NETWORK_SPLIT=ALL only if pooled analysis is intentional.")

    print(f"Aggregating suspicious transfers for split={ANALYSIS_SPLIT}...")
    edges_df = (
        df.filter(
            (F.col("predicted_suspicious") == 1)
            & F.col("Sender_account").isNotNull()
            & F.col("Receiver_account").isNotNull()
            & (F.col("Sender_account") != F.col("Receiver_account"))
        )
        .groupBy("Sender_account", "Receiver_account")
        .agg(
            F.count("*").alias("transaction_count"),
            F.sum("Amount").alias("total_amount"),
            F.max("risk_score").alias("max_risk_score"),
            F.avg("risk_score").alias("avg_risk_score"),
        )
        .orderBy(
            F.desc("max_risk_score"),
            F.desc("transaction_count"),
        )
        .limit(MAX_EDGES)
    )

    edge_rows = edges_df.collect()

    graph = nx.DiGraph()

    for row in edge_rows:
        graph.add_edge(
            str(row["Sender_account"]),
            str(row["Receiver_account"]),
            transaction_count=int(row["transaction_count"]),
            total_amount=float(row["total_amount"] or 0),
            max_risk_score=float(row["max_risk_score"] or 0),
            avg_risk_score=float(row["avg_risk_score"] or 0),
        )

    print(
        f"Graph built from {len(edge_rows):,} selected account pairs; "
        f"{graph.number_of_nodes():,} nodes."
    )

    print("Computing graph metrics...")
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph, max_iter=100)

    # Reciprocal relationships: A -> B and B -> A both exist.
    reciprocal_pairs = [
        (u, v)
        for u, v in graph.edges()
        if u < v and graph.has_edge(v, u)
    ]

    # Strongly connected components reveal groups with directed
    # reachability in both directions, not proof of laundering.
    components = [
        component
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1
    ]
    components.sort(key=len, reverse=True)

    component_by_account = {}
    for component_id, component in enumerate(components, start=1):
        for account in component:
            component_by_account[account] = component_id

    accounts = []

    for account in graph.nodes():
        incoming = list(graph.in_edges(account, data=True))
        outgoing = list(graph.out_edges(account, data=True))

        incoming_amount = sum(
            edge[2]["total_amount"] for edge in incoming
        )
        outgoing_amount = sum(
            edge[2]["total_amount"] for edge in outgoing
        )

        accounts.append({
            "account": account,
            "in_degree": in_degree[account],
            "out_degree": out_degree[account],
            "total_degree": in_degree[account] + out_degree[account],
            "pagerank": pagerank[account],
            "incoming_amount": round(incoming_amount, 2),
            "outgoing_amount": round(outgoing_amount, 2),
            "net_flow_out": round(outgoing_amount - incoming_amount, 2),
            "strong_component_id": component_by_account.get(account, ""),
            "in_suspicious_pairs": len(incoming),
            "out_suspicious_pairs": len(outgoing),
        })

    accounts.sort(
        key=lambda row: (
            row["total_degree"],
            row["pagerank"],
        ),
        reverse=True,
    )

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    account_file = output / "network_pattern_accounts.csv"
    with account_file.open("w", newline="", encoding="utf-8") as file:
        if accounts:
            writer = csv.DictWriter(file, fieldnames=accounts[0].keys())
            writer.writeheader()
            writer.writerows(accounts)
        else:
            writer = csv.writer(file)
            writer.writerow(["account", "in_degree", "out_degree", "total_degree", "pagerank"])

    edge_file = output / "network_pattern_edges.csv"
    with edge_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "sender", "receiver", "transaction_count",
            "total_amount", "max_risk_score", "avg_risk_score",
            "reciprocal_relationship",
        ])

        for sender, receiver, data in graph.edges(data=True):
            writer.writerow([
                sender,
                receiver,
                data["transaction_count"],
                round(data["total_amount"], 2),
                data["max_risk_score"],
                round(data["avg_risk_score"], 3),
                graph.has_edge(receiver, sender),
            ])

    summary = {
        "analysis": "Directed suspicious-transaction network",
        "evaluation_split": ANALYSIS_SPLIT,
        "edge_selection_limit": MAX_EDGES,
        "selected_account_pairs": len(edge_rows),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "reciprocal_directed_pairs": len(reciprocal_pairs),
        "strongly_connected_components_size_gt_1": len(components),
        "accounts_in_nontrivial_components": sum(
            len(component) for component in components
        ),
        "largest_component_size": (
            len(components[0]) if components else 0
        ),
        "top_accounts_by_degree": accounts[:20],
        "interpretation_note": (
            "Graph metrics are risk indicators, not proof of money laundering. "
            "By default the graph uses TEST alerts only; set AML_NETWORK_SPLIT=VALIDATION "
            "or ALL explicitly to change the scope. It uses up to 50,000 selected "
            "suspicious account pairs, not the complete transaction network."
        ),
    }

    with (output / "network_pattern_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n========== NETWORK ANALYSIS SUMMARY ==========")
    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Directed edges: {graph.number_of_edges():,}")
    print(f"Reciprocal pairs: {len(reciprocal_pairs):,}")
    print(f"Nontrivial strongly connected components: {len(components):,}")
    print(
        "Accounts in these components: "
        f"{summary['accounts_in_nontrivial_components']:,}"
    )
    print(f"Largest component: {summary['largest_component_size']:,}")

    print("\nTop 10 accounts by degree:")
    for account in accounts[:10]:
        print(
            f"{account['account']}: "
            f"in={account['in_degree']}, "
            f"out={account['out_degree']}, "
            f"PageRank={account['pagerank']:.6f}, "
            f"component={account['strong_component_id']}"
        )

    print("\nSaved:")
    print(account_file)
    print(edge_file)
    print(output / "network_pattern_summary.json")

finally:
    spark.stop()
