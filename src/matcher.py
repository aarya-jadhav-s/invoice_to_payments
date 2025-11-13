import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


@dataclass
class Match:
    payment_id: str
    invoice_id: str
    confidence: float
    rationale: str


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df


def write_out(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def baseline_normalize_name(name: Optional[str]) -> str:
    if not isinstance(name, str):
        return ""
    return (
        name.strip()
        .replace("Private Limited", "Pvt Ltd")
        .replace("Pvt. Ltd", "Pvt Ltd")
        .replace("Limited", "Ltd")
        .replace("Ltd.", "Ltd")
        .lower()
    )


def match_records(invoices: pd.DataFrame, payments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    invoices.columns = invoices.columns.str.strip().str.lower()
    payments.columns = payments.columns.str.strip().str.lower()

    invoices = invoices.rename(columns={
        "invoice_amount": "amount",
        "invoice_date": "date"
    })
    payments = payments.rename(columns={
        "payment_amount": "amount",
        "payment_date": "date"
    })

    invoices["amount"] = pd.to_numeric(invoices["amount"], errors="coerce")
    payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce")

    invoices["date"] = pd.to_datetime(invoices["date"], errors="coerce")
    payments["date"] = pd.to_datetime(payments["date"], errors="coerce")

    matches = []

    # --- Direct match by invoice_id in memo/reference ---
    for _, pay in payments.iterrows():
        memo_text = str(pay.get("memo", "")) + " " + str(pay.get("reference_number", ""))
        for _, inv in invoices.iterrows():
            if str(inv["invoice_id"]) in memo_text:
                matches.append({
                    "payment_id": pay["payment_id"],
                    "invoice_id": inv["invoice_id"],
                    "confidence": 1.0,
                    "rationale": "Direct invoice_id found in memo/reference"
                })

    # --- Match by exact amount and near-date ---
    window_days = 7
    for _, pay in payments.iterrows():
        for _, inv in invoices.iterrows():
            if (
                inv["currency"] == pay["currency"]
                and abs((pay["date"] - inv["date"]).days) <= window_days
                and abs(inv["amount"] - pay["amount"]) < 1e-2
            ):
                matches.append({
                    "payment_id": pay["payment_id"],
                    "invoice_id": inv["invoice_id"],
                    "confidence": 0.8,
                    "rationale": "Exact amount and near-date (±7 days)"
                })

    # --- Name normalization and amount logic ---
    for _, pay in payments.iterrows():
        pay_name = baseline_normalize_name(pay.get("payer_name"))
        for _, inv in invoices.iterrows():
            inv_name = baseline_normalize_name(inv.get("customer_name"))
            if pay_name == inv_name:
                amount_diff = pay["amount"] - inv["amount"]
                if abs(amount_diff) < 1e-2:
                    conf = 0.7
                    reason = "Name-normalized and amount match"
                elif pay["amount"] < inv["amount"]:
                    conf = 0.5
                    reason = "Partial payment - lower amount"
                elif pay["amount"] > inv["amount"]:
                    conf = 0.4
                    reason = "Overpayment - higher amount"
                else:
                    continue
                matches.append({
                    "payment_id": pay["payment_id"],
                    "invoice_id": inv["invoice_id"],
                    "confidence": conf,
                    "rationale": reason
                })

    # --- Convert to DataFrame ---
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df = matches_df.sort_values("confidence", ascending=False)
        matches_df = matches_df.drop_duplicates(subset=["payment_id"], keep="first")

    matched_payment_ids = set(matches_df["payment_id"]) if not matches_df.empty else set()
    matched_invoice_ids = set(matches_df["invoice_id"]) if not matches_df.empty else set()

    unmatched_payments = payments[~payments["payment_id"].isin(matched_payment_ids)]
    unmatched_invoices = invoices[~invoices["invoice_id"].isin(matched_invoice_ids)]

    # --- NEW TASK: Create remaining_payments.csv for partial payments ---
    remaining_records = []

    for _, row in matches_df.iterrows():
        pay = payments[payments["payment_id"] == row["payment_id"]].iloc[0]
        inv = invoices[invoices["invoice_id"] == row["invoice_id"]].iloc[0]
        if pay["amount"] < inv["amount"]:
            remaining = inv["amount"] - pay["amount"]
            remaining_records.append({
                "invoice_id": inv["invoice_id"],
                "customer_name": inv["customer_name"],
                "invoice_amount": inv["amount"],
                "payment_id": pay["payment_id"],
                "payment_amount": pay["amount"],
                "remaining_amount": remaining,
                "currency": inv["currency"]
            })

    remaining_df = pd.DataFrame(remaining_records)
    if not remaining_df.empty:
        os.makedirs("out", exist_ok=True)
        remaining_df.to_csv("out/remaining_payments.csv", index=False)
        print("\n💰 Remaining payments saved to: out/remaining_payments.csv")
    else:
        print("\n✅ No remaining payments found (all invoices fully paid).")

    return matches_df, unmatched_payments, unmatched_invoices


def main():
    parser = argparse.ArgumentParser(description="Invoice ↔ Payment matcher with remaining amount tracking")
    parser.add_argument("--invoices", required=True, help="path to invoices.csv")
    parser.add_argument("--payments", required=True, help="path to payments.csv")
    parser.add_argument("--out", default="out/", help="output directory (default: out/)")
    args = parser.parse_args()

    invoices = load_csv(args.invoices)
    payments = load_csv(args.payments)

    matches, u_pay, u_inv = match_records(invoices, payments)

    os.makedirs(args.out, exist_ok=True)
    write_out(matches, os.path.join(args.out, "matches.csv"))
    write_out(u_pay, os.path.join(args.out, "unmatched_payments.csv"))
    write_out(u_inv, os.path.join(args.out, "unmatched_invoices.csv"))

    summary = {
        "matches": len(matches),
        "unmatched_payments": len(u_pay),
        "unmatched_invoices": len(u_inv),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

