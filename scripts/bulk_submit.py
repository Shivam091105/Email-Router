"""
Bulk-submits emails through the running API's POST /emails endpoint, so
you get more data in the emails/routing_results/reviews tables to look at
in the Streamlit Analytics tab and Human Review tab, without typing
anything into the database by hand.

Usage:
    python -m scripts.bulk_submit                      # submits data/sample_emails.json
    python -m scripts.bulk_submit --source evaluation   # submits data/evaluation_emails.json instead
    python -m scripts.bulk_submit --url http://localhost:8000

Requires the API to already be running (uvicorn app.main:app) and the
RAG index to already be built (scripts/build_index.py), same as any
normal POST /emails call.
"""

import argparse
import json
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_emails(source: str) -> list[dict]:
    if source == "sample":
        raw = json.loads((DATA_DIR / "sample_emails.json").read_text())
        return [{"sender": e["sender"], "subject": e["subject"], "body": e["body"]} for e in raw]

    if source == "evaluation":
        raw = json.loads((DATA_DIR / "evaluation_emails.json").read_text())
        # evaluation_emails.json stores "Subject: ...\n\nBody..." combined --
        # split it back out just for a nicer subject line in the demo.
        results = []
        for item in raw:
            text = item["email"]
            if text.startswith("Subject:"):
                subject_line, _, body = text.partition("\n\n")
                subject = subject_line.removeprefix("Subject:").strip()
            else:
                subject, body = "", text
            results.append({"sender": "eval@example.com", "subject": subject, "body": body})
        return results

    raise ValueError(f"Unknown source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sample", "evaluation"], default="sample")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=None, help="Only submit the first N emails")
    args = parser.parse_args()

    emails = load_emails(args.source)
    if args.limit:
        emails = emails[: args.limit]

    print(f"Submitting {len(emails)} emails from '{args.source}' to {args.url}/emails ...")

    for i, email in enumerate(emails, start=1):
        try:
            resp = requests.post(f"{args.url}/emails", json=email, timeout=10)
            if resp.status_code == 201:
                print(f"[{i}/{len(emails)}] OK  id={resp.json()['id']}  subject={email['subject'][:50]!r}")
            else:
                print(f"[{i}/{len(emails)}] FAILED {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            print(f"[{i}/{len(emails)}] ERROR: {exc}")
            break

        time.sleep(0.3)  # be gentle on the Hugging Face Inference API rate limits

    print("\nDone. Background processing may take a few more seconds to finish for the last few emails.")
    print(f"Check results: {args.url}/emails  or  {args.url}/analytics")


if __name__ == "__main__":
    main()