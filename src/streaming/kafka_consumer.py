
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kafka import KafkaConsumer
from kafka.errors import KafkaError


DEFAULT_TOPIC = "aml-transactions"
AMOUNT_THRESHOLD = 67847.40
ALERT_THRESHOLD = 3

OUTPUT_DIR = Path("outputs")
EVENTS_FILE = OUTPUT_DIR / "streaming_events.jsonl"
ALERTS_FILE = OUTPUT_DIR / "streaming_alerts.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "streaming_summary.json"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def assess_event(event, state):
    sender = str(event.get("Sender_account") or "unknown")
    receiver = str(event.get("Receiver_account") or "unknown")

    # Update in-memory behavioural state for this running demo.
    state["sender_count"][sender] += 1
    state["receiver_count"][receiver] += 1
    state["sender_receivers"][sender].add(receiver)
    state["receiver_senders"][receiver].add(sender)

    amount = safe_float(event.get("Amount"))
    sender_bank = str(event.get("Sender_bank_location") or "").strip()
    receiver_bank = str(event.get("Receiver_bank_location") or "").strip()
    payment_currency = str(event.get("Payment_currency") or "").strip()
    received_currency = str(event.get("Received_currency") or "").strip()
    payment_type = str(event.get("Payment_type") or "").lower()

    score = 0
    reasons = []

    if amount >= AMOUNT_THRESHOLD:
        score += 2
        reasons.append("high_transaction_amount")

    if sender_bank and receiver_bank and sender_bank != receiver_bank:
        score += 1
        reasons.append("cross_border_transfer")

    if (
        payment_currency
        and received_currency
        and payment_currency != received_currency
    ):
        score += 1
        reasons.append("currency_difference")

    if any(word in payment_type for word in [
        "cash", "withdraw", "deposit"
    ]):
        score += 1
        reasons.append("cash_related_payment")

    if state["sender_count"][sender] >= 4:
        score += 1
        reasons.append("repeated_sender_activity")

    if state["receiver_count"][receiver] >= 4:
        score += 1
        reasons.append("repeated_receiver_activity")

    if len(state["sender_receivers"][sender]) >= 3:
        score += 2
        reasons.append("sender_fan_out_pattern")

    if len(state["receiver_senders"][receiver]) >= 3:
        score += 2
        reasons.append("receiver_fan_in_pattern")

    risk_level = (
        "HIGH" if score >= 6
        else "MEDIUM" if score >= ALERT_THRESHOLD
        else "LOW"
    )

    result = {
        **event,
        "stream_processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "stream_risk_score": score,
        "stream_risk_level": risk_level,
        "stream_alert": score >= ALERT_THRESHOLD,
        "stream_alert_reasons": reasons,
        "stream_rule_version": "demo-v1",
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Consume AML transaction events and score them."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--bootstrap-server", default="localhost:9092"
    )
    parser.add_argument(
        "--group-id", default="aml-sentinel-stream-consumer-v1"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "sender_count": defaultdict(int),
        "receiver_count": defaultdict(int),
        "sender_receivers": defaultdict(set),
        "receiver_senders": defaultdict(set),
    }

    processed = 0
    alerts = 0
    high_risk = 0
    medium_risk = 0

    consumer = None

    try:
        consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.bootstrap_server,
            group_id=args.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda raw: json.loads(
                raw.decode("utf-8")
            ),
            consumer_timeout_ms=1000,
        )

        print(f"Listening on Kafka topic: {args.topic}")
        print("Press Ctrl+C to stop the consumer.")

        with (
            EVENTS_FILE.open("a", encoding="utf-8") as event_file,
            ALERTS_FILE.open("a", encoding="utf-8") as alert_file,
        ):
            while True:
                records = consumer.poll(
                    timeout_ms=1000,
                    max_records=100,
                )

                for _, messages in records.items():
                    for message in messages:
                        try:
                            event = message.value
                            if not isinstance(event, dict):
                                raise ValueError(
                                    "Kafka message must contain a JSON object"
                                )

                            result = assess_event(event, state)
                            line = json.dumps(
                                result, ensure_ascii=False
                            )

                            event_file.write(line + "\n")
                            event_file.flush()
                            processed += 1

                            if result["stream_alert"]:
                                alert_file.write(line + "\n")
                                alert_file.flush()
                                alerts += 1

                                if result["stream_risk_level"] == "HIGH":
                                    high_risk += 1
                                elif result["stream_risk_level"] == "MEDIUM":
                                    medium_risk += 1

                                print(
                                    f"[ALERT] "
                                    f"{result['stream_risk_level']} "
                                    f"score={result['stream_risk_score']} "
                                    f"{result.get('Sender_account')} -> "
                                    f"{result.get('Receiver_account')} | "
                                    f"{', '.join(result['stream_alert_reasons'])}"
                                )
                            else:
                                print(
                                    f"[OK] score={result['stream_risk_score']} "
                                    f"{result.get('Sender_account')} -> "
                                    f"{result.get('Receiver_account')}"
                                )

                        except (ValueError, TypeError, KeyError) as exc:
                            print(f"Skipping invalid event: {exc}")

    except KeyboardInterrupt:
        print("\nStopping consumer...")
    except (KafkaError, OSError) as exc:
        print(f"Consumer failed: {exc}")
        raise
    finally:
        if consumer is not None:
            consumer.close()

        summary = {
            "topic": args.topic,
            "consumer_group": args.group_id,
            "processed_this_run": processed,
            "alerts_this_run": alerts,
            "high_risk_alerts_this_run": high_risk,
            "medium_risk_alerts_this_run": medium_risk,
            "alert_threshold": ALERT_THRESHOLD,
            "amount_threshold": AMOUNT_THRESHOLD,
            "rule_version": "demo-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "limitations": [
                "Demo rule engine, separate from historical Spark detector.",
                "Account behaviour state is held in memory and resets on restart.",
                "Thresholds are illustrative and not production-calibrated.",
                "An alert is a risk signal, not proof of money laundering.",
            ],
        }

        SUMMARY_FILE.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\n========== STREAMING SUMMARY ==========")
        print(json.dumps(summary, indent=2))
        print(f"Events:  {EVENTS_FILE}")
        print(f"Alerts:  {ALERTS_FILE}")
        print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
