
import argparse
import csv
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kafka import KafkaProducer
from kafka.errors import KafkaError


DEFAULT_TOPIC = "aml-transactions"

BANKS = ["UK", "US", "IN", "DE", "FR", "SG"]
CURRENCIES = ["GBP", "USD", "INR", "EUR", "SGD"]
PAYMENT_TYPES = [
    "Credit card",
    "Debit card",
    "ACH",
    "Wire",
    "Cash Deposit",
    "Cash Withdrawal",
    "Cheque",
]

# Repeated accounts create realistic-looking network hubs
# for demonstrating the streaming rules.
SENDERS = [f"S{i:04d}" for i in range(1, 21)]
RECEIVERS = [f"R{i:04d}" for i in range(1, 21)]


def synthetic_transactions(count, seed):
    rng = random.Random(seed)

    for index in range(count):
        sender = rng.choice(SENDERS)
        receiver = rng.choice(RECEIVERS)

        # Include some deliberately unusual demo events.
        if index % 8 == 0:
            amount = round(rng.uniform(70000, 150000), 2)
        else:
            amount = round(rng.lognormvariate(7.5, 1.0), 2)

        sender_bank = rng.choice(BANKS)
        receiver_bank = (
            rng.choice([b for b in BANKS if b != sender_bank])
            if index % 3 == 0
            else sender_bank
        )

        payment_currency = rng.choice(CURRENCIES)
        received_currency = (
            rng.choice([
                c for c in CURRENCIES
                if c != payment_currency
            ])
            if index % 4 == 0
            else payment_currency
        )

        yield {
            "transaction_id": str(uuid.uuid4()),
            "event_time": datetime.now(timezone.utc).isoformat(),
            "Sender_account": sender,
            "Receiver_account": receiver,
            "Amount": amount,
            "Payment_currency": payment_currency,
            "Received_currency": received_currency,
            "Sender_bank_location": sender_bank,
            "Receiver_bank_location": receiver_bank,
            "Payment_type": (
                rng.choice(["Cash Deposit", "Cash Withdrawal"])
                if index % 5 == 0
                else rng.choice(PAYMENT_TYPES)
            ),
            "source": "synthetic_demo",
        }


def csv_transactions(path, limit):
    with Path(path).open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        required = [
            "Sender_account",
            "Receiver_account",
            "Amount",
        ]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}"
            )

        for index, row in enumerate(reader):
            if limit and index >= limit:
                break

            # Intentionally omit Is_laundering and Laundering_type.
            event = {
                key: row.get(key)
                for key in [
                    "Time",
                    "Date",
                    "Sender_account",
                    "Receiver_account",
                    "Payment_currency",
                    "Received_currency",
                    "Sender_bank_location",
                    "Receiver_bank_location",
                    "Payment_type",
                ]
            }

            event["transaction_id"] = str(uuid.uuid4())
            event["event_time"] = datetime.now(
                timezone.utc
            ).isoformat()
            event["Amount"] = float(row["Amount"])
            event["source"] = "csv_demo"

            yield event


def main():
    parser = argparse.ArgumentParser(
        description="Publish AML Sentinel transaction events to Kafka."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--bootstrap-server", default="localhost:9092")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.interval < 0:
        parser.error("--interval cannot be negative")

    events = (
        csv_transactions(args.csv, args.count)
        if args.csv
        else synthetic_transactions(args.count, args.seed)
    )

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_server,
        value_serializer=lambda value: json.dumps(
            value, ensure_ascii=False
        ).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
        acks="all",
        retries=5,
    )

    sent = 0

    try:
        print(f"Publishing to topic: {args.topic}")

        for event in events:
            sender = str(event.get("Sender_account") or "unknown")
            future = producer.send(
                args.topic,
                key=sender,
                value=event,
            )

            metadata = future.get(timeout=20)
            sent += 1

            print(
                f"[{sent}] topic={metadata.topic} "
                f"partition={metadata.partition} "
                f"offset={metadata.offset} "
                f"sender={sender} "
                f"receiver={event.get('Receiver_account')} "
                f"amount={event.get('Amount')}"
            )

            if args.interval:
                time.sleep(args.interval)

        producer.flush()
        print(f"\nSuccessfully published {sent} events.")

    except (KafkaError, OSError, ValueError) as exc:
        print(f"Producer failed: {exc}")
        raise
    finally:
        producer.close()


if __name__ == "__main__":
    main()
