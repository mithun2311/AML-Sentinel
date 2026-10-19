
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "sample" / "transactions.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Is_laundering",
    "Laundering_type",
]

accounts = [f"ACC{number:04d}" for number in range(1, 51)]
locations = ["India", "UAE", "UK", "Singapore", "USA"]
currencies = ["INR", "USD", "AED", "GBP", "SGD"]
payment_types = [
    "Cash Deposit",
    "Credit Card",
    "Debit Card",
    "Wire Transfer",
    "ACH",
]

rows = []
start_time = datetime(2026, 1, 1, 8, 0, 0)

for i in range(200):
    sender = random.choice(accounts)
    receiver = random.choice(accounts)

    while receiver == sender:
        receiver = random.choice(accounts)

    # Most transactions are normal.
    # Every 20th transaction is deliberately marked suspicious.
    is_laundering = 1 if i % 20 == 0 else 0

    if is_laundering:
        amount = round(random.uniform(75000, 250000), 2)
        sender_location = random.choice(["India", "UAE"])
        receiver_location = random.choice(["UK", "Singapore", "USA"])
        laundering_type = random.choice([
            "Layering",
            "Cross-border transfer",
            "Smurfing",
        ])
    else:
        amount = round(random.uniform(500, 30000), 2)
        sender_location = random.choice(locations)
        receiver_location = random.choice(locations)
        laundering_type = "Normal"

    transaction_time = start_time + timedelta(
        minutes=random.randint(1, 50000)
    )

    rows.append({
        "Time": transaction_time.strftime("%H:%M:%S"),
        "Date": transaction_time.strftime("%Y-%m-%d"),
        "Sender_account": sender,
        "Receiver_account": receiver,
        "Amount": amount,
        "Payment_currency": random.choice(currencies),
        "Received_currency": random.choice(currencies),
        "Sender_bank_location": sender_location,
        "Receiver_bank_location": receiver_location,
        "Payment_type": random.choice(payment_types),
        "Is_laundering": is_laundering,
        "Laundering_type": laundering_type,
    })

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print("Sample transaction dataset created successfully.")
print(f"File: {OUTPUT_FILE}")
print(f"Total transactions: {len(rows)}")
print(f"Marked suspicious: {sum(row['Is_laundering'] == 1 for row in rows)}")
print(f"Marked normal: {sum(row['Is_laundering'] == 0 for row in rows)}")
