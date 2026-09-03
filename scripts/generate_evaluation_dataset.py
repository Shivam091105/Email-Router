"""
Generates data/evaluation_emails.json — a labeled, synthetic evaluation
dataset used by the Phase 11 evaluation scripts.

This is explicitly synthetic (per project requirements, that's fine as
long as it's realistic and diverse) rather than scraped or copied from a
real dataset. Each entry is built from hand-written subject/body
fragments per team, categorized by difficulty:

- straightforward   : clearly matches one team, close to its example phrasing
- reworded           : same underlying issue, different words entirely
- ambiguous          : plausibly fits more than one team; labeled with the
                        best-fit team, so a model choosing a "reasonable"
                        alternative should not be scored as wildly wrong,
                        but IS scored as wrong for strict accuracy
- multi_issue        : mentions two problems; labeled by the dominant one
- confusable         : deliberately shares vocabulary with a *different*,
                        similar team (e.g. IT Support vs Infrastructure)

Re-run this script any time you want to regenerate/expand the dataset:
    python -m scripts.generate_evaluation_dataset
"""

import itertools
import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation_emails.json"

# Each entry: (department, team, team_id, difficulty, subject, body)
RAW_EXAMPLES: list[tuple[str, str, int, str, str, str]] = [
    # ---------------- IT Support (101) — straightforward ----------------
    ("IT", "IT Support", 101, "straightforward", "Can't log in", "I can't log into my account, it says my password is incorrect even though I'm sure it's right."),
    ("IT", "IT Support", 101, "straightforward", "Password reset broken", "The password reset link you emailed me doesn't work, it just shows a blank page."),
    ("IT", "IT Support", 101, "straightforward", "Need admin rights", "Could someone give me admin access on my laptop so I can install the design software?"),
    ("IT", "IT Support", 101, "straightforward", "Laptop won't boot", "My laptop won't turn on this morning, I've tried holding the power button but nothing happens."),
    ("IT", "IT Support", 101, "straightforward", "VPN keeps dropping", "My VPN connection keeps dropping every few minutes when I work from home."),
    ("IT", "IT Support", 101, "reworded", "Locked out", "I think my account got locked after too many failed attempts, how do I get back in?"),
    ("IT", "IT Support", 101, "reworded", "Software install request", "Is there someone who can help me get Photoshop installed? I don't have permission on this machine."),
    ("IT", "IT Support", 101, "reworded", "Screen issue", "My monitor stopped displaying anything after the latest Windows update, could be a driver issue?"),
    ("IT", "IT Support", 101, "reworded", "MFA not working", "The two-factor authentication code never arrives on my phone when I try to sign in."),
    ("IT", "IT Support", 101, "reworded", "New hire setup", "I'm starting Monday and need my laptop and email account set up before then."),

    # ---------------- Infrastructure & DevOps (102) — straightforward ----------------
    ("IT", "Infrastructure & DevOps", 102, "straightforward", "Production down", "The production website has been returning 500 errors for the last 20 minutes, this is urgent."),
    ("IT", "Infrastructure & DevOps", 102, "straightforward", "API errors", "Our API integration is getting intermittent 502 responses since this afternoon."),
    ("IT", "Infrastructure & DevOps", 102, "straightforward", "Staging environment", "Can we get a new staging environment provisioned for the upcoming release testing?"),
    ("IT", "Infrastructure & DevOps", 102, "straightforward", "Deployment failing", "The CI/CD pipeline is failing at the build step on the latest commit to main."),
    ("IT", "Infrastructure & DevOps", 102, "reworded", "Server load", "Our monitoring dashboard shows CPU usage pinned at 100% on the main app server since noon."),
    ("IT", "Infrastructure & DevOps", 102, "reworded", "SSL certificate expiring", "Heads up, the SSL cert on api.ourapp.com expires in 3 days and needs renewing."),
    ("IT", "Infrastructure & DevOps", 102, "reworded", "Database migration", "We need a database migration run against production during the next maintenance window."),
    ("IT", "Infrastructure & DevOps", 102, "confusable", "Can't connect to server", "I can't SSH into the build server, connection just times out — is it down?"),

    # ---------------- Billing & Payments (201) — straightforward ----------------
    ("Finance", "Billing & Payments", 201, "straightforward", "Double charged", "I was charged twice for my subscription this month, can I get one of those refunded?"),
    ("Finance", "Billing & Payments", 201, "straightforward", "Missing transaction", "I am unable to access my online banking account and several transactions are missing."),
    ("Finance", "Billing & Payments", 201, "straightforward", "Need an invoice copy", "Could you send me a copy of last month's invoice? I need it for expense reporting."),
    ("Finance", "Billing & Payments", 201, "straightforward", "Card declined", "My payment method was declined at checkout but I definitely have funds available."),
    ("Finance", "Billing & Payments", 201, "straightforward", "Refund request", "I'd like to request a refund, I accidentally purchased the annual plan instead of monthly."),
    ("Finance", "Billing & Payments", 201, "reworded", "Wrong amount charged", "The amount on my statement doesn't match what I was quoted at signup, can someone check?"),
    ("Finance", "Billing & Payments", 201, "reworded", "Update card on file", "How do I update the credit card on file for my subscription?"),
    ("Finance", "Billing & Payments", 201, "reworded", "Subscription still active after cancel", "I cancelled last week but was still billed today, please look into this."),

    # ---------------- Accounts Payable (202) — straightforward ----------------
    ("Finance", "Accounts Payable", 202, "straightforward", "Vendor invoice unpaid", "Our invoice to your company from last month hasn't been paid yet, can you check on it?"),
    ("Finance", "Accounts Payable", 202, "straightforward", "Update banking details", "We need to update our vendor banking details for future payments."),
    ("Finance", "Accounts Payable", 202, "straightforward", "PO status", "When will purchase order #7788 be processed? We're waiting to ship the order."),
    ("Finance", "Accounts Payable", 202, "reworded", "Contractor payment", "As a contractor, I haven't received payment for last month's invoice yet."),
    ("Finance", "Accounts Payable", 202, "confusable", "Payment not received", "We are a supplier and have not received payment for our March invoice."),

    # ---------------- HR Operations (301) — straightforward ----------------
    ("HR", "HR Operations", 301, "straightforward", "Missing paycheck", "I haven't received my paycheck this month, could someone look into this?"),
    ("HR", "HR Operations", 301, "straightforward", "Benefits enrollment", "How do I enroll in the company health plan? I just started last week."),
    ("HR", "HR Operations", 301, "straightforward", "Parental leave request", "I'd like to formally request parental leave starting next month."),
    ("HR", "HR Operations", 301, "straightforward", "Employment verification", "I need an employment verification letter for a mortgage application."),
    ("HR", "HR Operations", 301, "reworded", "PTO balance", "Can someone tell me how many vacation days I have left this year?"),
    ("HR", "HR Operations", 301, "reworded", "Address change", "I moved recently and need to update my home address in the payroll system."),
    ("HR", "HR Operations", 301, "reworded", "401k question", "How do I change my 401k contribution percentage?"),

    # ---------------- Recruiting (302) — straightforward ----------------
    ("HR", "Recruiting", 302, "straightforward", "Application status", "I applied for a position two weeks ago and haven't heard back, any update?"),
    ("HR", "Recruiting", 302, "straightforward", "Reschedule interview", "Could we reschedule my interview to next week? Something came up."),
    ("HR", "Recruiting", 302, "straightforward", "Is role still open", "Is the Senior Engineer role still open? I'd like to apply."),
    ("HR", "Recruiting", 302, "reworded", "Referral question", "I want to refer a friend for an open position, what's the process?"),
    ("HR", "Recruiting", 302, "confusable", "Offer letter question", "I received an offer letter but have a question about the start date listed."),

    # ---------------- Account Management (401) — straightforward ----------------
    ("Sales", "Account Management", 401, "straightforward", "Upgrade to enterprise", "We'd like to upgrade our existing account to the enterprise plan."),
    ("Sales", "Account Management", 401, "straightforward", "Renewal questions", "Our contract renewal date is coming up and we have a few questions before signing."),
    ("Sales", "Account Management", 401, "straightforward", "Custom pricing", "Can we get a custom pricing quote for 500 seats under our existing contract?"),
    ("Sales", "Account Management", 401, "reworded", "Downgrade plan", "We'd like to downgrade our current plan starting next billing cycle."),
    ("Sales", "Account Management", 401, "reworded", "Add seats", "We need to add 20 more seats to our current enterprise agreement."),

    # ---------------- New Business (402) — straightforward ----------------
    ("Sales", "New Business", 402, "straightforward", "Interested in your product", "I'm interested in your product for my company, could someone reach out?"),
    ("Sales", "New Business", 402, "straightforward", "Free trial question", "Do you offer a free trial before committing to a paid plan?"),
    ("Sales", "New Business", 402, "straightforward", "Plan differences", "What's the difference between your Basic and Pro plans?"),
    ("Sales", "New Business", 402, "reworded", "Demo request", "Could we schedule a product demo for our team sometime this week?"),
    ("Sales", "New Business", 402, "confusable", "Considering switching providers", "We currently use a competitor's product but are considering switching, what's the pricing?"),

    # ---------------- Contracts & Compliance (501) — straightforward ----------------
    ("Legal", "Contracts & Compliance", 501, "straightforward", "DPA required", "We need a signed Data Processing Agreement before proceeding with integration."),
    ("Legal", "Contracts & Compliance", 501, "straightforward", "Contract review", "Can your legal team review this MSA redline we sent over last week?"),
    ("Legal", "Contracts & Compliance", 501, "straightforward", "GDPR compliance", "Are you GDPR compliant? We need documentation for our records."),
    ("Legal", "Contracts & Compliance", 501, "reworded", "NDA request", "Before we share our roadmap, we'll need a mutual NDA signed."),
    ("Legal", "Contracts & Compliance", 501, "reworded", "SOC 2 report", "Could you share your latest SOC 2 Type II report for our vendor security review?"),

    # ---------------- Product Support (601) — straightforward ----------------
    ("Customer Support", "Product Support", 601, "straightforward", "Export to CSV", "How do I export my data to CSV? I can't find the option anywhere."),
    ("Customer Support", "Product Support", 601, "straightforward", "Bug report", "I found a bug where the save button doesn't work on mobile Safari."),
    ("Customer Support", "Product Support", 601, "straightforward", "Bulk delete", "Is there a way to bulk-delete old records instead of one at a time?"),
    ("Customer Support", "Product Support", 601, "straightforward", "Feature request", "Could you add a dark mode option? Would really help with eye strain."),
    ("Customer Support", "Product Support", 601, "reworded", "How to use a feature", "I can't figure out how to set up recurring reports, is there a guide?"),
    ("Customer Support", "Product Support", 601, "reworded", "App crashing", "The mobile app crashes every time I try to open the settings page."),
    ("Customer Support", "Product Support", 601, "reworded", "Where is my data", "I imported a CSV yesterday but don't see the records anywhere in the dashboard."),

    # ---------------- Ambiguous (deliberately hard) ----------------
    ("Finance", "Billing & Payments", 201, "ambiguous", "Account problem", "There's something wrong with my account, I think it might be a billing issue but I'm not totally sure."),
    ("IT", "IT Support", 101, "ambiguous", "Can't access anything", "I can't access anything today — not sure if it's my login or something on your end."),
    ("Sales", "Account Management", 401, "ambiguous", "Question about our plan", "We have some questions about our current plan and what our options are going forward."),
    ("Customer Support", "Product Support", 601, "ambiguous", "Something's not working", "Something isn't working right in the app today, not sure exactly what's wrong."),
    ("HR", "HR Operations", 301, "ambiguous", "Payroll and benefits question", "I have a question that touches both payroll and my benefits enrollment, who should I talk to?"),

    # ---------------- Multi-issue (dominant issue labeled) ----------------
    ("IT", "IT Support", 101, "multi_issue", "Login issue and also a billing question", "I can't log into my account, and separately I also noticed I was charged twice this month."),
    ("Finance", "Billing & Payments", 201, "multi_issue", "Refund and also a bug", "I'd like a refund for this charge, and by the way the export button is also broken."),
    ("Sales", "Account Management", 401, "multi_issue", "Upgrade and contract question", "We want to upgrade our plan, and also had a question about the contract renewal terms."),
    ("HR", "Recruiting", 302, "multi_issue", "Interview reschedule and offer question", "Could we reschedule my interview, and I also had a question about the offer letter timeline."),

    # ---------------- Confusable pairs (similar teams, different label) ----------------
    ("IT", "Infrastructure & DevOps", 102, "confusable", "Whole platform is slow", "The entire platform has been extremely slow for all users since this morning, seems like a systemic issue."),
    ("IT", "IT Support", 101, "confusable", "My account specifically is slow", "The app is fine for my coworkers but very slow specifically on my laptop only."),
    ("Finance", "Accounts Payable", 202, "confusable", "We haven't been paid", "As your vendor, we have not received payment for services rendered last quarter."),
    ("Finance", "Billing & Payments", 201, "confusable", "I haven't been charged", "I expected to be charged for this month's subscription but wasn't, is something wrong?"),
]


def build_dataset() -> list[dict]:
    dataset = []
    for i, (department, team, team_id, difficulty, subject, body) in enumerate(RAW_EXAMPLES, start=1):
        dataset.append(
            {
                "id": i,
                "email": f"Subject: {subject}\n\n{body}",
                "expected_department": department,
                "expected_team": team,
                "expected_team_id": team_id,
                "difficulty": difficulty,
            }
        )
    return dataset


def main() -> None:
    dataset = build_dataset()
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"Wrote {len(dataset)} labeled examples to {OUTPUT_PATH}")

    by_difficulty = {}
    for item in dataset:
        by_difficulty.setdefault(item["difficulty"], 0)
        by_difficulty[item["difficulty"]] += 1
    print("Breakdown by difficulty:", by_difficulty)


if __name__ == "__main__":
    main()
