"""Question generators and independent checkers.

Tier 1: synthetic documents with code-planted facts; checkers re-derive the answer from the state text.
Tier 2: constraint problems; checkers are brute-force solvers independent of the planting code.
Tier 3: factual questions from a frontier model, cross-checked by a second frontier model (API; cached).

Usage:
  python -m src.generate            # build data/questions.jsonl from tiers 1+2 (+ tier 3 cache if present)
  python -m src.generate --tier3    # call the API to build the tier-3 cache, then rebuild
  python -m src.generate --validate # run every checker against data/questions.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from .common import (DATA_DIR, QUESTIONS_PATH, SCORE_LEVELS, Question, load_config, read_jsonl,
                     stable_hash, write_jsonl)

# --------------------------------------------------------------------------------------
# Text pools
# --------------------------------------------------------------------------------------
FIRST = ["Dana", "Jordan", "Priya", "Theo", "Maya", "Sam", "Elena", "Marcus", "Aisha", "Tomas", "Ingrid",
         "Kenji", "Rosa", "Felix", "Nadia", "Omar", "Lena", "Victor", "Hana", "Gabriel"]
LAST = ["Whitfield", "Reyes", "Natarajan", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Novak",
        "Beaumont", "Castillo", "Schreiber", "Mensah", "Petrov", "Fontaine", "Iyer", "Kowalski", "Duarte"]
COMPANIES = [("Acme Corp", "acme-corp.com"), ("Northwind Traders", "northwind.io"), ("Bluepeak Logistics", "bluepeak.co"),
             ("Helios Energy", "helios-energy.com"), ("Ledgerly", "ledgerly.app"), ("Marlowe & Finch LLP", "marlowefinch.law"),
             ("Cobalt Systems", "cobaltsys.net"), ("Juniper Health", "juniperhealth.org"), ("Orion Freight", "orionfreight.com"),
             ("Saffron Studio", "saffron.studio"), ("Tidewater Analytics", "tidewater.ai"), ("Vantage Retail", "vantageretail.com")]
DOMAIN_TO_COMPANY = {d: c for c, d in COMPANIES}
DEPARTMENTS = ["Finance", "Legal", "Human Resources", "Facilities", "Engineering", "Marketing", "Procurement",
               "Compliance", "Customer Support", "Operations", "Clinical Services", "Information Security",
               "Pharmacy", "Research"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PLACES = ["Lupa", "the taco place", "Bao House", "Nora's", "the pier", "Giulio's", "the rooftop bar"]
PRODUCTS = {
    "business": ["Ledgerly Payroll", "Vantage POS", "Northwind Procure", "Tidewater Reports"],
    "technical": ["Cobalt API Gateway", "Orion CLI", "Helios Metrics SDK", "Acme Build Cloud"],
    "casual": ["Saffron Photo App", "Juniper Fitness", "Bluepeak Rides", "Marlowe Reader"],
}
# Filler sentences. They must not contain any ticket-topic keyword (see TOPICS); test_generate checks this.
FILLER = {
    "business": [
        "Our finance team closes the books on the fifth business day of each month.",
        "The quarterly review has been moved to the second week of the month.",
        "We have three regional offices and one shared services centre.",
        "The procurement portal will be unavailable this weekend for maintenance.",
        "Please keep the reference number in the subject line of any reply.",
        "The account manager assigned to us changed in January.",
        "Our fiscal year ends in June, which affects when approvals are granted.",
    ],
    "technical": [
        "We run the service in two availability zones behind a load balancer.",
        "The staging environment mirrors production except for the database size.",
        "Our deploy pipeline builds a container image and runs the integration suite.",
        "The error budget for this quarter is already half consumed.",
        "Logs are retained for thirty days and metrics for ninety.",
        "The SDK is pinned to version 4.2 in our lock file.",
        "The on-call rotation changes every Tuesday at noon.",
    ],
    "casual": [
        "I mostly use the app on my phone during my commute.",
        "My partner signed up last month after I recommended it.",
        "The new icon looks nice, by the way.",
        "I have been a member since the beta a couple of years ago.",
        "Sorry for the long message, I wanted to include everything.",
        "The notifications come through at odd hours but that is a minor thing.",
        "I usually log in from a tablet in the evenings.",
    ],
}
GENERIC_NOTES = [
    "First response SLA: 1 business day.", "Linked knowledge base article: KB-1042.",
    "Customer tier: standard.", "Previous contacts this quarter: 2.", "Sentiment flag: neutral.",
]

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
Rng = random.Random


def _name(r: Rng) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def _email(r: Rng, name: str, domain: str) -> str:
    f, l = name.lower().split()
    return f"{f}.{l}@{domain}"


def _date(r: Rng, start=date(2026, 1, 5), span=270) -> date:
    return start + timedelta(days=r.randrange(span))


def _mk_options(r: Rng, must: list[str], pool: list[str], n: int) -> list[str]:
    """n options containing everything in `must` plus distractors from `pool`, shuffled."""
    opts = list(dict.fromkeys(must))
    extra = [p for p in pool if p not in opts]
    r.shuffle(extra)
    opts += extra[: max(0, n - len(opts))]
    r.shuffle(opts)
    return opts


def _choice_q(qid, tier, state, question, options, answer_text, domain, gen, seed, difficulty=None, meta=None):
    idx = options.index(answer_text)
    return Question(id=qid, tier=tier, format="choice", state=state, question=question, options=options,
                    ground_truth=idx, ground_truth_text=answer_text, domain=domain, generator=gen,
                    generator_seed=seed, difficulty=difficulty, meta=meta or {})


def _yesno_q(qid, tier, state, question, answer_yes: bool, domain, gen, seed, difficulty=None, meta=None):
    gt = "yes" if answer_yes else "no"
    return Question(id=qid, tier=tier, format="yesno", state=state, question=question, options=None,
                    ground_truth=gt, ground_truth_text=gt, domain=domain, generator=gen,
                    generator_seed=seed, difficulty=difficulty, meta=meta or {})


def _score_q(qid, tier, state, question, levels: list[str], level_idx: int, domain, gen, seed, difficulty=None, meta=None):
    assert len(levels) == SCORE_LEVELS and 0 <= level_idx < SCORE_LEVELS
    return Question(id=qid, tier=tier, format="score", state=state, question=question, options=levels,
                    ground_truth=level_idx, ground_truth_text=levels[level_idx], domain=domain, generator=gen,
                    generator_seed=seed, difficulty=difficulty, meta=meta or {})


COUNT_LEVELS = ["1", "2", "3", "4", "5"]

# ======================================================================================
# TIER 1 — read off the page
# ======================================================================================

# ---- t1_ticket_topic (yesno) -----------------------------------------------------------
TOPICS = {
    "refund": dict(kw=r"\brefund(?:ed|s|ing)?\b", q="Does this ticket mention a refund?", sents=[
        "I would like a refund for order {order}.", "Can I get a refund on the extra seat we were charged for?",
        "I am not asking for a refund, I just want the item replaced.", "Please process the refund to the original card.",
        "Your agent promised a partial refund last week and nothing has arrived."]),
    "password": dict(kw=r"\bpassword(?:s)?\b", q="Does this ticket mention a password?", sents=[
        "I reset my password twice and the new one is still rejected.", "The password rules changed without notice.",
        "Nobody asked me for a password, which seems wrong.", "My team shares one password for the admin console, which I know is bad."]),
    "cancellation": dict(kw=r"\bcancel(?:led|lation|ling|s)?\b", q="Does this ticket mention a cancellation?", sents=[
        "Please cancel my subscription at the end of the month.", "The cancellation went through but I was billed again.",
        "I do not want to cancel, I just need the plan downgraded.", "Someone cancelled our order without telling us."]),
    "invoice": dict(kw=r"\binvoice(?:s|d)?\b", q="Does this ticket mention an invoice?", sents=[
        "The invoice from March lists the wrong company name.", "Could you resend the invoice as a PDF?",
        "We were invoiced twice for the same period.", "I have not received any invoice at all this year."]),
    "shipping": dict(kw=r"\bshipp?(?:ing|ed|ment)\b", q="Does this ticket mention shipping or a shipment?", sents=[
        "The shipment left the warehouse but the tracking page is blank.", "Shipping took three weeks instead of five days.",
        "The parcel was shipped to my old address.", "Is express shipping available for replacement units?"]),
    "discount": dict(kw=r"\bdiscount(?:s|ed)?\b", q="Does this ticket mention a discount?", sents=[
        "The annual plan discount was not applied at checkout.", "We were promised a volume discount for twenty seats.",
        "I am not looking for a discount, I want the bug fixed.", "The discount code expired a day earlier than advertised."]),
    "upgrade": dict(kw=r"\bupgrade(?:d|s)?\b", q="Does this ticket mention an upgrade?", sents=[
        "After the upgrade to the pro tier, the dashboard stopped loading.", "Can I upgrade mid-cycle and pay the difference?",
        "We upgraded last week and the dashboard still shows the old limits.", "I never asked for an upgrade but my bill went up."]),
    "export": dict(kw=r"\bexport(?:ed|s|ing)?\b", q="Does this ticket mention exporting data?", sents=[
        "I need to export all of our records before the audit.", "The CSV export stops at exactly 10,000 rows.",
        "Exporting to PDF produces an empty file.", "Is there a way to export the audit trail?"]),
}


def _ticket_header(r: Rng, domain: str) -> str:
    name = _name(r)
    comp, dom = r.choice(COMPANIES)
    return "\n".join([
        f"Ticket #{r.randrange(10000, 99999)}",
        f"Product: {r.choice(PRODUCTS[domain])}",
        f"Customer: {name} ({_email(r, name, dom)})",
        f"Channel: {r.choice(['email', 'web form', 'chat transcript'])}",
        f"Submitted: {_date(r).isoformat()} {r.randrange(8, 19):02d}:{r.randrange(60):02d}",
        f"Priority: {r.choice(['Low', 'Medium', 'High'])}",
    ])


def gen_t1_ticket_topic(r: Rng, qid: str, seed: int) -> Question:
    domain = r.choice(["business", "technical", "casual"])
    target = r.choice(sorted(TOPICS))
    answer_yes = r.random() < 0.5
    others = [t for t in TOPICS if t != target]
    r.shuffle(others)
    n_other = r.randrange(1, 4)
    body = [r.choice(TOPICS[t]["sents"]).format(order=f"#{r.randrange(100000, 999999)}") for t in others[:n_other]]
    if answer_yes:
        body += [r.choice(TOPICS[target]["sents"]).format(order=f"#{r.randrange(100000, 999999)}")]
    body += r.sample(FILLER[domain], r.randrange(2, 5))
    r.shuffle(body)
    subject = r.choice(["Problem with my account", "Question about last month", "Issue after recent change",
                        "Need help", "Follow-up on earlier contact"])
    state = (_ticket_header(r, domain) + f"\n\nSubject: {subject}\n\n" + " ".join(body) +
             "\n\n--\nInternal notes: " + " ".join(r.sample(GENERIC_NOTES, 2)))
    return _yesno_q(qid, 1, state, TOPICS[target]["q"], answer_yes, domain, "t1_ticket_topic", seed,
                    meta={"doc_type": "support_ticket", "topic": target})


def check_t1_ticket_topic(q: Question) -> bool:
    topic = next(t for t, d in TOPICS.items() if d["q"] == q.question)
    found = re.search(TOPICS[topic]["kw"], q.state, re.I) is not None
    return (q.ground_truth == "yes") == found


# ---- t1_memo_department (choice) -------------------------------------------------------
MEMO_SUBJECTS = ["Badge access changes for the north wing", "Vendor onboarding checklist update",
                 "Revised travel reimbursement timeline", "Audit preparation for Q3", "Printer fleet replacement",
                 "Patient intake form revision", "Holiday coverage schedule", "Data retention policy reminder"]
MEMO_BODY = [
    "Please note that {d1} has already reviewed the attached draft and raised no objections.",
    "The timeline was discussed with {d2} last week, and their comments are incorporated below.",
    "If you have questions about the budget line, {d3} is the right point of contact.",
    "This supersedes the memo circulated by {d1} in the spring.",
    "A copy has also gone to {d2} for their records.",
    "We expect to brief {d3} once the first phase is complete.",
]


def gen_t1_memo_department(r: Rng, qid: str, seed: int) -> Question:
    domain = r.choice(["business", "legal", "medical", "technical"])
    depts = r.sample(DEPARTMENTS, 5)
    to, cc, d1, d2, d3 = depts
    n_opts = r.randrange(4, 9)
    options = _mk_options(r, [to, cc, d1, d2, d3][: min(5, n_opts)], DEPARTMENTS, n_opts)
    body = [s.format(d1=d1, d2=d2, d3=d3) for s in r.sample(MEMO_BODY, 3)]
    r.shuffle(body)
    state = "\n".join([
        "MEMORANDUM",
        f"TO: {to} Department",
        f"FROM: {_name(r)}, Office of the {r.choice(['COO', 'CFO', 'General Counsel', 'Medical Director', 'CTO'])}",
        f"CC: {cc} Department",
        f"DATE: {_date(r).isoformat()}",
        f"RE: {r.choice(MEMO_SUBJECTS)}",
        "",
        " ".join(body),
        "",
        r.choice(FILLER["business"]) + " " + r.choice(FILLER["business"]),
    ])
    return _choice_q(qid, 1, state, "Which department is this memo addressed to?", options, to, domain,
                     "t1_memo_department", seed, meta={"doc_type": "memo"})


def check_t1_memo_department(q: Question) -> bool:
    m = re.search(r"^TO:\s*(.+?) Department\s*$", q.state, re.M)
    return m is not None and q.options[q.ground_truth] == m.group(1)


# ---- t1_fields_present (score, count 1-5 of five required fields) ----------------------
FIELD_SETS = {
    "medical": (["patient_id", "date_of_birth", "insurer", "policy_number", "primary_physician"],
                ["referral_source", "phone", "preferred_language", "notes"]),
    "technical": (["service_name", "owner_email", "on_call_rota", "runbook_url", "tier"],
                  ["tags", "repository", "last_deploy", "notes"]),
    "business": (["vendor_name", "tax_id", "bank_iban", "contact_email", "payment_terms"],
                 ["website", "notes", "category", "account_manager"]),
}
SAMPLE_VALUES = {
    "patient_id": "MRN-{n}", "date_of_birth": "19{y}-0{m}-1{d}", "insurer": "Juniper Health Plan", "policy_number": "JH-{n}",
    "primary_physician": "Dr. {name}", "referral_source": "self", "phone": "+1-555-01{n2}", "preferred_language": "en", "notes": "{filler}",
    "service_name": "checkout-api", "owner_email": "{email}", "on_call_rota": "team-payments", "runbook_url": "https://runbooks.internal/{n}",
    "tier": "tier-2", "tags": "payments,critical", "repository": "git@internal:payments/checkout-api", "last_deploy": "2026-04-0{d}",
    "vendor_name": "{company}", "tax_id": "TX-{n}", "bank_iban": "GB29NWBK601613{n}", "contact_email": "{email}",
    "payment_terms": "net 30", "website": "https://{dom}", "category": "services", "account_manager": "{name}",
}


def _fill_value(r: Rng, key: str) -> str:
    comp, dom = r.choice(COMPANIES)
    name = _name(r)
    return SAMPLE_VALUES[key].format(n=r.randrange(10000, 99999), y=r.randrange(50, 99), m=r.randrange(1, 9), d=r.randrange(1, 9),
                                     name=name, n2=r.randrange(10, 99), filler=r.choice(FILLER["business"]), email=_email(r, name, dom),
                                     company=comp, dom=dom)


def gen_t1_fields_present(r: Rng, qid: str, seed: int) -> Question:
    domain = r.choice(sorted(FIELD_SETS))
    required, optional = FIELD_SETS[domain]
    n_present = r.randrange(1, 6)
    present = set(r.sample(required, n_present))
    record: dict = {}
    for k in required:
        if k in present:
            record[k] = _fill_value(r, k)
        else:
            mode = r.choice(["absent", "null", "empty", "blank"])
            if mode == "null":
                record[k] = None
            elif mode == "empty":
                record[k] = ""
            elif mode == "blank":
                record[k] = "   "
    for k in r.sample(optional, r.randrange(2, 4)):
        record[k] = _fill_value(r, k)
    # distractor: a nested object with look-alike keys
    record["_meta"] = {f"{r.choice(required)}_hint": "see attached", "source": "intake form v3"}
    keys = list(record.items())
    r.shuffle(keys)
    record = dict(keys)
    as_text = r.random() < 0.4
    if as_text:
        lines = []
        for k, v in record.items():
            if k == "_meta":
                continue
            lines.append(f"{k}: {'' if v is None else v}")
        state = f"RECORD ({domain} intake)\n" + "\n".join(lines) + "\n\n_meta: " + json.dumps(record["_meta"])
    else:
        state = record
    question = ("How many of these five required fields are present with a non-empty value: " +
                ", ".join(required) + "? (Null, missing, empty or whitespace-only values do not count.)")
    return _score_q(qid, 1, state, question, COUNT_LEVELS, n_present - 1, domain, "t1_fields_present", seed,
                    meta={"doc_type": "record_text" if as_text else "record_json", "required": required})


def _parse_record(q: Question) -> dict:
    if isinstance(q.state, dict):
        return q.state
    rec = {}
    for line in q.state.splitlines():
        m = re.match(r"^([a-z_]+):\s?(.*)$", line)
        if m and m.group(1) != "_meta":
            rec[m.group(1)] = m.group(2)
    return rec


def check_t1_fields_present(q: Question) -> bool:
    m = re.search(r"non-empty value: (.+?)\?", q.question)
    required = [x.strip() for x in m.group(1).split(",")]
    rec = _parse_record(q)
    n = sum(1 for k in required if rec.get(k) is not None and not (isinstance(rec[k], str) and rec[k].strip() == ""))
    return n >= 1 and q.ground_truth == n - 1


# ---- t1_invoice_status (yesno) ---------------------------------------------------------
SERVICES = ["Consulting services", "Cloud hosting", "Support retainer", "Training workshop", "Data migration",
            "Design review", "Security audit", "License renewal", "Hardware lease", "Translation services"]
STATUSES = ["PAID", "UNPAID", "OVERDUE", "PARTIALLY PAID", "VOID"]


def gen_t1_invoice_status(r: Rng, qid: str, seed: int) -> Question:
    domain = r.choice(["business", "legal"])
    status = r.choice(STATUSES)
    items = r.sample(SERVICES, r.randrange(2, 5))
    lines, subtotal = [], 0.0
    for i, s in enumerate(items, 1):
        qty, rate = r.randrange(1, 40), r.choice([75.0, 120.0, 150.0, 240.0, 995.0])
        amt = qty * rate
        subtotal += amt
        lines.append(f"{i}. {s} ........ {qty} x ${rate:,.2f} = ${amt:,.2f}")
    tax = round(subtotal * 0.08, 2)
    sender, _ = r.choice(COMPANIES)
    billto, _ = r.choice(COMPANIES)
    not_items = [s for s in SERVICES if s not in items]
    note_service = r.choice(not_items)
    notes = [
        f"Payment for invoice INV-2026-{r.randrange(100, 999)} was received on {_date(r).isoformat()}, thank you.",
        f"{note_service} will be invoiced separately next month.",
        "Late payments accrue 1.5% monthly interest.",
        f"Status of the previous invoice: {r.choice([s for s in STATUSES if s != status])}.",
    ]
    r.shuffle(notes)
    issued = _date(r)
    state = "\n".join([
        "INVOICE", f"Invoice number: INV-2026-{r.randrange(1000, 9999)}",
        f"Issued: {issued.isoformat()}   Due: {(issued + timedelta(days=30)).isoformat()}",
        f"Bill to: {billto}", f"From: {sender}", f"Status: {status}", "", "Line items:", *lines,
        f"Subtotal: ${subtotal:,.2f}", f"Tax (8%): ${tax:,.2f}", f"Total due: ${subtotal + tax:,.2f}", "",
        "Notes: " + " ".join(notes[:3]),
    ])
    if r.random() < 0.5:
        question = "Is this invoice marked as PAID (status exactly PAID)?"
        answer = status == "PAID"
        kind = "status"
    else:
        if r.random() < 0.5:
            svc, answer = r.choice(items), True
        else:
            svc, answer = note_service, False
        question = f'Does this invoice include a line item for "{svc}"?'
        kind = "item"
    return _yesno_q(qid, 1, state, question, answer, domain, "t1_invoice_status", seed,
                    meta={"doc_type": "invoice", "kind": kind})


def check_t1_invoice_status(q: Question) -> bool:
    if q.question.startswith("Is this invoice marked as PAID"):
        m = re.search(r"^Status: (.+)$", q.state, re.M)
        return (m.group(1).strip() == "PAID") == (q.ground_truth == "yes")
    svc = re.search(r'line item for "(.+?)"', q.question).group(1)
    section = q.state.split("Line items:")[1].split("Subtotal:")[0]
    found = re.search(r"^\d+\. " + re.escape(svc) + r" ", section, re.M) is not None
    return found == (q.ground_truth == "yes")


# ---- t1_email_sender_company (choice) --------------------------------------------------
EMAIL_BODY = [
    "I spoke with the team at {c1} yesterday and they are comfortable with the revised dates.",
    "Our partners at {c2} asked whether the same terms could apply to them.",
    "As {c3} mentioned in their last note, the onboarding window closes at the end of the quarter.",
    "The draft still references {c1} in section two, which we should fix before it goes out.",
    "Please loop in {c2} if the pricing question comes up again.",
]
TITLES = ["Partnerships Lead", "Account Director", "Senior Counsel", "Operations Manager", "Product Manager", "Clinical Liaison"]


def gen_t1_email_sender_company(r: Rng, qid: str, seed: int) -> Question:
    domain = r.choice(["business", "legal", "casual"])
    comps = r.sample(COMPANIES, 6)
    (sc, sd), (tc, td), (cc, cd), (c1, _), (c2, _), (c3, _) = comps
    sender, to, ccn = _name(r), _name(r), _name(r)
    n_opts = r.randrange(4, 9)
    options = _mk_options(r, [sc, tc, cc, c1, c2, c3][: min(6, n_opts)], [c for c, _ in COMPANIES], n_opts)
    body = [s.format(c1=c1, c2=c2, c3=c3) for s in r.sample(EMAIL_BODY, 3)]
    state = "\n".join([
        f"From: {sender} <{_email(r, sender, sd)}>", f"To: {to} <{_email(r, to, td)}>",
        f"Cc: {ccn} <{_email(r, ccn, cd)}>", f"Date: {_date(r).isoformat()}",
        f"Subject: {r.choice(['Re: revised terms', 'Next steps', 'Quick question', 'Follow-up from Tuesday'])}", "",
        f"Hi {to.split()[0]},", "", " ".join(body), "", r.choice(FILLER['business']), "",
        r.choice(["Best,", "Thanks,", "Regards,"]), sender, f"{r.choice(TITLES)}, {sc}",
    ])
    return _choice_q(qid, 1, state, "Which company does the sender of this email work for?", options, sc, domain,
                     "t1_email_sender_company", seed, meta={"doc_type": "email"})


def check_t1_email_sender_company(q: Question) -> bool:
    m = re.search(r"^From: .+<[^@]+@([^>]+)>", q.state, re.M)
    by_domain = DOMAIN_TO_COMPANY.get(m.group(1)) if m else None
    sig = q.state.rstrip().splitlines()[-1]
    by_sig = sig.split(", ", 1)[1] if ", " in sig else None
    ans = q.options[q.ground_truth]
    return by_domain == ans and by_sig == ans


# ---- t1_log_event_level (choice) and t1_log_count_level (score) -------------------------
LEVELS = ["TRACE", "DEBUG", "INFO", "NOTICE", "WARN", "ERROR", "CRITICAL", "FATAL"]
COMPONENTS = ["auth", "cart", "payments", "search", "scheduler", "mailer", "gateway", "cache"]
LOG_MSGS = [
    "request completed in {ms} ms", "error rate within normal bounds", "warning threshold adjusted to {ms}",
    "cache miss for key user:{n}", "retrying upstream call ({k}/3)", "connection pool exhausted",
    "token refreshed for session {n}", "critical path latency {ms} ms", "fatal signal handler registered",
    "scheduled job {n} finished", "debug sampling enabled", "info: config reloaded",
]


def _log_state(r: Rng, n_events: int, level_pool: list[str]) -> dict:
    events, used_ids = [], set()
    base_ts = r.randrange(1_700_000_000, 1_790_000_000)
    for i in range(n_events):
        while True:
            eid = f"evt-{r.randrange(0x1000, 0xffff):04x}"
            if eid not in used_ids:
                used_ids.add(eid)
                break
        events.append({
            "id": eid, "ts": base_ts + i * r.randrange(1, 90), "level": r.choice(level_pool),
            "component": r.choice(COMPONENTS),
            "message": r.choice(LOG_MSGS).format(ms=r.randrange(5, 900), n=r.randrange(1000, 9999), k=r.randrange(1, 4)),
        })
    return {"service": r.choice(["checkout-api", "search-indexer", "billing-worker", "edge-gateway"]),
            "environment": r.choice(["prod", "staging"]), "events": events}


def gen_t1_log_event_level(r: Rng, qid: str, seed: int) -> Question:
    n_opts = r.randrange(4, 9)
    pool = r.sample(LEVELS, n_opts)
    state = _log_state(r, r.randrange(6, 13), pool)
    target = r.choice(state["events"])
    # near-miss id distractor
    near = target["id"][:-1] + ("a" if target["id"][-1] != "a" else "b")
    state["events"].append({"id": near, "ts": target["ts"] + 1, "level": r.choice([l for l in pool if l != target["level"]]),
                            "component": target["component"], "message": "duplicate suppressed"})
    r.shuffle(state["events"])
    options = list(pool)
    r.shuffle(options)
    return _choice_q(qid, 1, state, f"What is the level of the event with id {target['id']}?", options, target["level"],
                     "technical", "t1_log_event_level", seed, meta={"doc_type": "json_log"})


def check_t1_log_event_level(q: Question) -> bool:
    eid = re.search(r"id (evt-[0-9a-f]{4})\?", q.question).group(1)
    ev = [e for e in q.state["events"] if e["id"] == eid]
    return len(ev) == 1 and q.options[q.ground_truth] == ev[0]["level"]


def gen_t1_log_count_level(r: Rng, qid: str, seed: int) -> Question:
    target = r.choice(LEVELS)
    count = r.randrange(1, 6)
    others = [l for l in LEVELS if l != target]
    n_events = r.randrange(count + 3, count + 9)
    state = _log_state(r, n_events, others)
    for e in r.sample(state["events"], count):
        e["level"] = target
    # distractor messages mentioning the level word
    e = r.choice(state["events"])
    e["message"] = f"{target.lower()} count reset for component {e['component']}"
    return _score_q(qid, 1, state, f"How many events in this log have level {target}?", COUNT_LEVELS, count - 1,
                    "technical", "t1_log_count_level", seed, meta={"doc_type": "json_log"})


def check_t1_log_count_level(q: Question) -> bool:
    lvl = re.search(r"have level ([A-Z]+)\?", q.question).group(1)
    n = sum(1 for e in q.state["events"] if e["level"] == lvl)
    return 1 <= n <= 5 and q.ground_truth == n - 1


# ---- t1_contract_term (choice) ---------------------------------------------------------
JURISDICTIONS = ["the State of New York", "the State of Delaware", "England and Wales", "the State of California",
                 "the Province of Ontario", "Singapore", "the State of Texas", "Ireland"]
TERM_MONTHS = [3, 6, 12, 18, 24, 36, 48, 60]
NOTICE_DAYS = [15, 30, 45, 60, 90, 120, 180]


def gen_t1_contract_term(r: Rng, qid: str, seed: int) -> Question:
    term, renew = r.sample(TERM_MONTHS, 2)
    notice, cure = r.sample(NOTICE_DAYS, 2)
    law, other_law = r.sample(JURISDICTIONS, 2)
    vendor, _ = r.choice(COMPANIES)
    client, _ = r.choice([c for c in COMPANIES if c[0] != vendor])
    state = "\n".join([
        f"SERVICES AGREEMENT between {vendor} (\"Vendor\") and {client} (\"Client\")", "",
        f"1. Term. The initial term of this Agreement is {term} months from the Effective Date. Thereafter the Agreement "
        f"renews automatically for successive periods of {renew} months unless either party gives notice of non-renewal.",
        f"2. Termination. Either party may terminate for convenience on {notice} days' written notice. A party in material "
        f"breach shall have {cure} days from written notice to cure the breach.",
        f"3. Fees. Client shall pay the fees set out in Schedule A within 30 days of invoice.",
        f"4. Governing Law. This Agreement is governed by the laws of {law}, notwithstanding that Vendor is organised under "
        f"the laws of {other_law}.",
        f"5. Entire Agreement. This Agreement supersedes all prior agreements between the parties.",
    ])
    kind = r.choice(["term", "notice", "law"])
    n_opts = r.randrange(4, 9)
    if kind == "term":
        question, ans = "What is the initial term of this agreement?", f"{term} months"
        options = _mk_options(r, [ans, f"{renew} months"], [f"{m} months" for m in TERM_MONTHS], n_opts)
    elif kind == "notice":
        question, ans = "How many days' written notice are required to terminate for convenience?", f"{notice} days"
        options = _mk_options(r, [ans, f"{cure} days"], [f"{d} days" for d in NOTICE_DAYS], min(n_opts, len(NOTICE_DAYS)))
    else:
        question, ans = "Which jurisdiction's law governs this agreement?", law
        options = _mk_options(r, [ans, other_law], JURISDICTIONS, n_opts)
    return _choice_q(qid, 1, state, question, options, ans, "legal", "t1_contract_term", seed,
                     meta={"doc_type": "contract", "kind": kind})


def check_t1_contract_term(q: Question) -> bool:
    ans = q.options[q.ground_truth]
    if q.question.startswith("What is the initial term"):
        m = re.search(r"initial term of this Agreement is (\d+) months", q.state)
        return ans == f"{m.group(1)} months"
    if q.question.startswith("How many days"):
        m = re.search(r"terminate for convenience on (\d+) days", q.state)
        return ans == f"{m.group(1)} days"
    m = re.search(r"governed by the laws of (.+?), notwithstanding", q.state)
    return ans == m.group(1)


# ---- t1_ticket_priority (score) --------------------------------------------------------
PRIORITY_LEVELS = ["Low", "Medium", "High", "Urgent", "Critical"]
URGENCY_TEXT = ["This is extremely urgent, our whole team is blocked!!!", "No rush on this one, whenever you get to it.",
                "I would call this critical for us.", "Honestly it is a minor annoyance but worth logging.",
                "Please treat this as high priority from our side."]


def gen_t1_ticket_priority(r: Rng, qid: str, seed: int) -> Question:
    domain = r.choice(["technical", "business"])
    level = r.randrange(5)
    header = _ticket_header(r, domain)
    header = re.sub(r"^Priority: .*$", f"Priority: {PRIORITY_LEVELS[level]}", header, flags=re.M)
    body = [r.choice(URGENCY_TEXT)] + r.sample(FILLER[domain], 3)
    r.shuffle(body)
    state = header + "\n\nSubject: Issue report\n\n" + " ".join(body) + "\n\n--\nInternal notes: " + \
        f"Customer-stated urgency: {r.choice(PRIORITY_LEVELS).lower()}. " + r.choice(GENERIC_NOTES)
    return _score_q(qid, 1, state, "What priority level is assigned to this ticket in its header (the Priority field)?",
                    PRIORITY_LEVELS, level, domain, "t1_ticket_priority", seed, meta={"doc_type": "support_ticket"})


def check_t1_ticket_priority(q: Question) -> bool:
    m = re.search(r"^Priority: (\w+)$", q.state, re.M)
    return m is not None and q.options[q.ground_truth] == m.group(1)


# ---- t1_medical_allergy (yesno) --------------------------------------------------------
DRUGS = ["penicillin", "sulfa drugs", "ibuprofen", "codeine", "latex", "aspirin", "amoxicillin", "contrast dye"]
MEDS = ["metformin 500 mg twice daily", "lisinopril 10 mg daily", "atorvastatin 20 mg nightly",
        "levothyroxine 75 mcg daily", "omeprazole 20 mg daily"]


def gen_t1_medical_allergy(r: Rng, qid: str, seed: int) -> Question:
    target = r.choice(DRUGS)
    answer_yes = r.random() < 0.5
    others = [d for d in DRUGS if d != target]
    listed = r.sample(others, r.randrange(0, 3))
    if answer_yes:
        listed.append(target)
        r.shuffle(listed)
    allergies = ", ".join(f"{d} ({r.choice(['rash', 'hives', 'anaphylaxis', 'nausea'])})" for d in listed) if listed else "NKDA"
    distractors = []
    if not answer_yes:
        distractors.append(r.choice([f"Family history: mother allergic to {target}.",
                                     f"Patient asked whether {target} would be appropriate; deferred.",
                                     f"Tolerated {target} in the past without reaction."]))
    state = "\n".join([
        "CLINIC NOTE - Internal Medicine",
        f"Patient: {r.choice('ABCDEFGHJKLMNPRSTW')}.{r.choice('ABCDEFGHJKLMNPRSTW')}., {r.randrange(19, 88)}{r.choice('MF')}   "
        f"MRN: {r.randrange(10000000, 99999999)}   Visit: {_date(r).isoformat()}",
        f"Chief complaint: {r.choice(['follow-up of hypertension', 'intermittent headaches', 'annual physical', 'knee pain'])}",
        f"Allergies: {allergies}",
        f"Medications: {'; '.join(r.sample(MEDS, 2))}",
        f"Assessment: {r.choice(['stable', 'improving', 'unchanged from last visit'])}. " + " ".join(distractors),
        f"Plan: {r.choice(['continue current regimen', 'labs in 3 months', 'referral placed'])}.",
    ])
    return _yesno_q(qid, 1, state, f"Does this note list an allergy to {target}?", answer_yes, "medical",
                    "t1_medical_allergy", seed, meta={"doc_type": "clinic_note"})


def check_t1_medical_allergy(q: Question) -> bool:
    drug = re.search(r"allergy to (.+?)\?", q.question).group(1)
    line = re.search(r"^Allergies: (.*)$", q.state, re.M).group(1)
    return (drug in line) == (q.ground_truth == "yes")


# ---- t1_casual_plan_day (choice) -------------------------------------------------------
SETTLE_TEMPLATES = ["ok locking in {day} {time} at {place}", "{day} it is then, {time} at {place}", "final: {day} {time}, {place}"]
SETTLE_RE = re.compile(r"(?:locking in|it is then|final:) ?(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?", re.I)


def gen_t1_casual_plan_day(r: Rng, qid: str, seed: int) -> Question:
    days = r.sample(DAYS, 4)
    final, bad, maybe, alt = days
    names = r.sample(FIRST, 3)
    t = r.randrange(17, 21)
    msgs = [
        f"{names[0]}: ok so dinner this week?",
        f"{names[1]}: not {bad}, I have a thing",
        f"{names[0]}: {maybe} works for me",
        f"{names[2]}: {maybe} is bad, how about {alt}",
        f"{names[1]}: {alt} is tight, {final}?",
        f"{names[2]}: {final} works",
        f"{names[0]}: " + r.choice(SETTLE_TEMPLATES).format(day=final, time=f"{t}:00", place=r.choice(PLACES)),
        f"{names[2]}: " + r.choice(["see you then", "perfect", "bringing dessert"]),
    ]
    state = "\n".join(f"[{t - 2}:{i * 2 + r.randrange(0, 2):02d}] {m}" for i, m in enumerate(msgs))
    n_opts = r.randrange(4, 8)
    options = _mk_options(r, days, DAYS, n_opts)
    return _choice_q(qid, 1, state, "Which day did the group finally settle on for dinner?", options, final, "casual",
                     "t1_casual_plan_day", seed, meta={"doc_type": "chat"})


def check_t1_casual_plan_day(q: Question) -> bool:
    found = None
    for line in q.state.splitlines():
        m = re.search(r"(?:locking in|final:) (\w+day)|(\w+day) it is then", line)
        if m:
            found = m.group(1) or m.group(2)
    return found is not None and q.options[q.ground_truth] == found


# ---- t1_json_flag (yesno) --------------------------------------------------------------
FLAGS = ["dark_mode", "new_cart", "fast_checkout", "beta_search", "inline_help", "price_alerts", "bulk_export", "sso_login"]


def gen_t1_json_flag(r: Rng, qid: str, seed: int) -> Question:
    names = r.sample(FLAGS, r.randrange(4, 7))
    target = r.choice(names)
    features = {}
    for n in names:
        features[n] = {"enabled": r.random() < 0.5, "rollout_percent": r.choice([0, 5, 20, 50, 100]),
                       "owner": r.choice(COMPONENTS)}
    if r.random() < 0.5:
        features[target]["note"] = r.choice(["enable after QA sign-off", "was enabled in v3", "disabled during incident, re-enable later"])
    state = {"service": r.choice(["checkout-api", "search-indexer", "mobile-app"]), "env": r.choice(["prod", "staging"]),
             "features": features, "deprecated_flags": [f"{target}_v1", f"{r.choice(FLAGS)}_legacy"]}
    return _yesno_q(qid, 1, state, f"Is the feature flag '{target}' enabled (its enabled field is true)?",
                    features[target]["enabled"], "technical", "t1_json_flag", seed, meta={"doc_type": "json_config"})


def check_t1_json_flag(q: Question) -> bool:
    name = re.search(r"flag '(\w+)'", q.question).group(1)
    return (q.state["features"][name]["enabled"] is True) == (q.ground_truth == "yes")


# ---- t1_review_rating (score) ----------------------------------------------------------
STAR_LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
REVIEW_TEXT = ["Honestly a bit disappointed with the battery, but support was great.", "Does exactly what it says.",
               "Took two weeks to arrive and the box was dented.", "Would buy again without hesitation.",
               "My previous one lasted five years; this one feels flimsier.", "Fine for the price, nothing special."]


def gen_t1_review_rating(r: Rng, qid: str, seed: int) -> Question:
    level = r.randrange(5)
    other = r.choice([x for x in range(1, 6) if x != level + 1])
    state = "\n".join([
        f"Product: {r.choice(['Wireless earbuds', 'Standing desk', 'Espresso grinder', 'Trail running shoes'])}",
        f"Reviewer: {r.choice(FIRST)} ({r.choice(['verified purchase', 'unverified'])})",
        f"Rating: {level + 1}/5",
        f"Helpful votes: {r.randrange(0, 40)}",
        "", " ".join(r.sample(REVIEW_TEXT, 2)) + f" My sister gave hers {other} stars, for what it is worth.",
    ])
    return _score_q(qid, 1, state, "What star rating did the reviewer give?", STAR_LEVELS, level, "casual",
                    "t1_review_rating", seed, meta={"doc_type": "review"})


def check_t1_review_rating(q: Question) -> bool:
    m = re.search(r"^Rating: (\d)/5$", q.state, re.M)
    return m is not None and q.ground_truth == int(m.group(1)) - 1


# ======================================================================================
# TIER 2 — reason over the state
# ======================================================================================

# ---- t2_meeting_rooms --------------------------------------------------------------------
MEETING_NAMES = ["Budget review", "Design sync", "Hiring panel", "Vendor call", "Sprint planning", "All-hands prep",
                 "Security briefing", "Customer demo", "Board prep", "Retro"]
ROOM_RULES = ("An assignment is valid if (1) no two meetings assigned to the same room overlap in time (a meeting that "
              "ends at 10:00 and one that starts at 10:00 do not overlap), and (2) each meeting's attendee count is at "
              "most its room's capacity.")


def _hm(slot: int) -> str:
    return f"{8 + slot // 2:02d}:{'30' if slot % 2 else '00'}"


def _slot(hm: str) -> int:
    h, m = hm.split(":")
    return (int(h) - 8) * 2 + (1 if m == "30" else 0)


def meeting_violations(state: dict) -> tuple[set[str], set[str]]:
    """Independent checker: returns (violating meeting names, rooms containing a violation)."""
    cap = {rm["name"]: rm["capacity"] for rm in state["rooms"]}
    ms = {m["name"]: m for m in state["meetings"]}
    bad_m, bad_r = set(), set()
    for name, room in state["assignment"].items():
        if ms[name]["attendees"] > cap[room]:
            bad_m.add(name)
            bad_r.add(room)
    names = list(state["assignment"])
    for a, b in itertools.combinations(names, 2):
        if state["assignment"][a] != state["assignment"][b]:
            continue
        sa, ea = _slot(ms[a]["start"]), _slot(ms[a]["end"])
        sb, eb = _slot(ms[b]["start"]), _slot(ms[b]["end"])
        if sa < eb and sb < ea:
            bad_m.update({a, b})
            bad_r.add(state["assignment"][a])
    return bad_m, bad_r


def _build_rooms_state(r: Rng, n_meet: int, n_rooms: int) -> dict | None:
    rooms = [{"name": f"Room {chr(65 + i)}", "capacity": r.choice([4, 6, 8, 10, 12])} for i in range(n_rooms)]
    meetings = []
    for name in r.sample(MEETING_NAMES, n_meet):
        s = r.randrange(0, 18)
        d = r.choice([1, 2, 3, 4])
        meetings.append({"name": name, "start": _hm(s), "end": _hm(min(s + d, 20)), "attendees": r.randrange(2, 13)})
    # backtracking for a valid assignment
    order = list(range(n_meet))
    assign: dict[str, str] = {}

    def ok(i, room):
        m = meetings[i]
        if m["attendees"] > rooms[room]["capacity"]:
            return False
        for j in range(i):
            if assign[meetings[j]["name"]] == rooms[room]["name"]:
                a, b = meetings[j], m
                if _slot(a["start"]) < _slot(b["end"]) and _slot(b["start"]) < _slot(a["end"]):
                    return False
        return True

    def bt(i):
        if i == n_meet:
            return True
        rs = list(range(n_rooms))
        r.shuffle(rs)
        for room in rs:
            if ok(i, room):
                assign[meetings[i]["name"]] = rooms[room]["name"]
                if bt(i + 1):
                    return True
                del assign[meetings[i]["name"]]
        return False

    if not bt(0):
        return None
    return {"rooms": rooms, "meetings": meetings, "assignment": assign}


def _plant_room_violation(r: Rng, state: dict, kind: str) -> bool:
    ms = state["meetings"]
    if kind == "capacity":
        m = r.choice(ms)
        room = next(rm for rm in state["rooms"] if rm["name"] == state["assignment"][m["name"]])
        if m["attendees"] < 12 and all(x["attendees"] <= 12 for x in ms):
            m["attendees"] = room["capacity"] + r.randrange(1, 4)
            return True
        return False
    # overlap: move a meeting into a room where it overlaps another
    r.shuffle(ms)
    for m in ms:
        for other in ms:
            if other is m or state["assignment"][other["name"]] == state["assignment"][m["name"]]:
                continue
            if _slot(m["start"]) < _slot(other["end"]) and _slot(other["start"]) < _slot(m["end"]):
                room = state["assignment"][other["name"]]
                cap = next(rm["capacity"] for rm in state["rooms"] if rm["name"] == room)
                if m["attendees"] <= cap:
                    state["assignment"][m["name"]] = room
                    return True
    return False


def _rooms_state_with(r: Rng, want_violations: int, unique_room: bool) -> dict:
    for _ in range(200):
        n_meet, n_rooms = r.randrange(4, 9), r.randrange(2, 6)
        st = _build_rooms_state(r, n_meet, n_rooms)
        if st is None:
            continue
        for _ in range(want_violations * 2):
            bad_m, bad_r = meeting_violations(st)
            if len(bad_m) >= want_violations:
                break
            _plant_room_violation(r, st, r.choice(["capacity", "overlap"]))
        bad_m, bad_r = meeting_violations(st)
        if want_violations == 0 and bad_m:
            continue
        if want_violations and len(bad_m) != want_violations:
            continue
        if unique_room and len(bad_r) > 1:
            continue
        st["meetings"].sort(key=lambda m: m["name"])
        st["assignment"] = dict(sorted(st["assignment"].items()))
        return st
    raise RuntimeError("could not build rooms state")


def gen_t2_meeting_rooms(r: Rng, qid: str, seed: int, fmt: str) -> Question:
    if fmt == "yesno":
        valid = r.random() < 0.5
        st = _rooms_state_with(r, 0 if valid else r.randrange(1, 4), unique_room=False)
        st["rules"] = ROOM_RULES
        q = "Is this room assignment valid according to the rules?"
        return _yesno_q(qid, 2, st, q, valid, "business", "t2_meeting_rooms", seed, difficulty=len(st["meetings"]))
    if fmt == "choice":
        none = r.random() < 0.25
        st = _rooms_state_with(r, 0 if none else r.randrange(1, 4), unique_room=True)
        st["rules"] = ROOM_RULES
        options = [rm["name"] for rm in st["rooms"]] + ["No violation"]
        while len(options) < 4:
            options.insert(len(options) - 1, f"Room {chr(65 + len(options) - 1)}")
        bad_m, bad_r = meeting_violations(st)
        ans = "No violation" if none else next(iter(bad_r))
        q = "Which room contains a violation of the rules (an overlap or an over-capacity meeting)? If none, answer 'No violation'."
        return _choice_q(qid, 2, st, q, options, ans, "business", "t2_meeting_rooms", seed, difficulty=len(st["meetings"]))
    k = r.randrange(1, 6)
    st = _rooms_state_with(r, k, unique_room=False)
    st["rules"] = ROOM_RULES
    q = ("How many meetings are involved in at least one violation of the rules (overlapping another meeting in the "
         "same room, or exceeding the room's capacity)?")
    return _score_q(qid, 2, st, q, COUNT_LEVELS, k - 1, "business", "t2_meeting_rooms", seed, difficulty=len(st["meetings"]))


def check_t2_meeting_rooms(q: Question) -> bool:
    bad_m, bad_r = meeting_violations(q.state)
    if q.format == "yesno":
        return (len(bad_m) == 0) == (q.ground_truth == "yes")
    if q.format == "choice":
        ans = q.options[q.ground_truth]
        if ans == "No violation":
            return not bad_r
        return bad_r == {ans}
    return 1 <= len(bad_m) <= 5 and q.ground_truth == len(bad_m) - 1


# ---- t2_expense_policy -------------------------------------------------------------------
RULE_OPTIONS = ["Rule 1 (meal limit)", "Rule 2 (receipt)", "Rule 3 (alcohol)", "Rule 4 (pre-approval)", "Rule 5 (trip dates)", "No rule"]
EXP_DESC = {"meal": ["Team dinner", "Client lunch", "Breakfast meeting", "Working lunch"],
            "alcohol": ["Bar tab", "Wine for client", "Drinks after panel"],
            "flight": ["Flight to summit", "Return flight", "Regional flight"],
            "hotel": ["Hotel, 2 nights", "Hotel, 1 night"],
            "taxi": ["Taxi to airport", "Ride to venue", "Airport transfer"],
            "supplies": ["Printing", "Adapter", "Office supplies"]}


def _expense_policy(r: Rng) -> dict:
    start = _date(r)
    return {"meal_max_per_person": r.choice([30, 40, 50, 60]), "receipt_required_over": r.choice([20, 25, 50, 75]),
            "alcohol_reimbursable": False, "travel_requires_preapproval": True,
            "trip": {"start": start.isoformat(), "end": (start + timedelta(days=r.randrange(1, 5))).isoformat()}}


def _rules_text(p: dict) -> list[str]:
    return [f"Rule 1: Meals may not exceed ${p['meal_max_per_person']} per person per meal (amount divided by number of people).",
            f"Rule 2: Any item with an amount over ${p['receipt_required_over']} must have a receipt.",
            "Rule 3: Alcohol is not reimbursable.",
            "Rule 4: Flights and hotels must be pre-approved.",
            "Rule 5: Expense dates must fall within the trip dates, inclusive."]


def item_violations(policy: dict, item: dict) -> list[int]:
    """Independent checker: which rule numbers (1-5) the item violates."""
    v = []
    if item["category"] == "meal" and item["amount"] / max(1, item.get("people", 1)) > policy["meal_max_per_person"]:
        v.append(1)
    if item["amount"] > policy["receipt_required_over"] and not item["receipt"]:
        v.append(2)
    if item["category"] == "alcohol":
        v.append(3)
    if item["category"] in ("flight", "hotel") and not item["preapproved"]:
        v.append(4)
    if not (policy["trip"]["start"] <= item["date"] <= policy["trip"]["end"]):
        v.append(5)
    return v


def _clean_item(r: Rng, p: dict, i: int) -> dict:
    cat = r.choice(["meal", "flight", "hotel", "taxi", "supplies", "meal", "taxi"])
    start, end = date.fromisoformat(p["trip"]["start"]), date.fromisoformat(p["trip"]["end"])
    d = start + timedelta(days=r.randrange(0, (end - start).days + 1))
    people = r.randrange(1, 5) if cat == "meal" else 1
    if cat == "meal":
        amount = round(people * r.uniform(8, p["meal_max_per_person"]), 2)
    elif cat in ("flight", "hotel"):
        amount = round(r.uniform(120, 900), 2)
    else:
        amount = round(r.uniform(5, 80), 2)
    item = {"id": f"E{i}", "date": d.isoformat(), "category": cat, "description": r.choice(EXP_DESC[cat]), "amount": amount,
            "people": people, "receipt": True if amount > p["receipt_required_over"] else r.random() < 0.6,
            "preapproved": True if cat in ("flight", "hotel") else r.random() < 0.3}
    assert not item_violations(p, item)
    return item


def _break_item(r: Rng, p: dict, item: dict, rule: int) -> None:
    start, end = date.fromisoformat(p["trip"]["start"]), date.fromisoformat(p["trip"]["end"])
    if rule == 1:
        item["category"], item["description"] = "meal", r.choice(EXP_DESC["meal"])
        item["people"] = r.randrange(1, 5)
        item["amount"] = round(item["people"] * p["meal_max_per_person"] + r.uniform(1, 40), 2)
        item["receipt"], item["preapproved"] = True, item["preapproved"]
    elif rule == 2:
        if item["amount"] <= p["receipt_required_over"]:
            item["amount"] = round(p["receipt_required_over"] + r.uniform(1, 30), 2)
            if item["category"] == "meal":
                item["amount"] = min(item["amount"], item["people"] * p["meal_max_per_person"])
                if item["amount"] <= p["receipt_required_over"]:
                    item["category"], item["description"], item["people"] = "taxi", r.choice(EXP_DESC["taxi"]), 1
                    item["amount"] = round(p["receipt_required_over"] + r.uniform(1, 30), 2)
        item["receipt"] = False
    elif rule == 3:
        item["category"], item["description"], item["people"] = "alcohol", r.choice(EXP_DESC["alcohol"]), 1
        item["amount"] = round(r.uniform(10, p["receipt_required_over"]), 2)
        item["receipt"] = True
    elif rule == 4:
        item["category"] = r.choice(["flight", "hotel"])
        item["description"] = r.choice(EXP_DESC[item["category"]])
        item["people"], item["amount"], item["receipt"], item["preapproved"] = 1, round(r.uniform(120, 900), 2), True, False
    elif rule == 5:
        item["date"] = (start - timedelta(days=r.randrange(1, 5))).isoformat() if r.random() < 0.5 else \
            (end + timedelta(days=r.randrange(1, 5))).isoformat()


def _expense_state(r: Rng, n_items: int, broken: dict[int, int]) -> dict:
    """broken: item index -> rule number to violate (exactly that one rule)."""
    for _ in range(100):
        p = _expense_policy(r)
        items = [_clean_item(r, p, i + 1) for i in range(n_items)]
        for idx, rule in broken.items():
            _break_item(r, p, items[idx], rule)
        if all(item_violations(p, items[idx]) == [rule] for idx, rule in broken.items()) and \
                all(not item_violations(p, it) for i, it in enumerate(items) if i not in broken):
            return {"policy": p, "rules": _rules_text(p), "items": items}
    raise RuntimeError("could not build expense state")


def gen_t2_expense_policy(r: Rng, qid: str, seed: int, fmt: str) -> Question:
    n_items = r.randrange(4, 9)
    if fmt == "yesno":
        violates = r.random() < 0.5
        broken = {r.randrange(n_items): r.randrange(1, 6)} if violates else {}
        st = _expense_state(r, n_items, broken)
        q = "Does this expense report violate at least one of the five policy rules?"
        return _yesno_q(qid, 2, st, q, violates, "business", "t2_expense_policy", seed, difficulty=n_items)
    if fmt == "choice":
        idx = r.randrange(n_items)
        rule = r.choice([1, 2, 3, 4, 5, 0])  # 0 = no rule
        # add 0-2 other broken items as distractors
        broken = {idx: rule} if rule else {}
        for j in r.sample([k for k in range(n_items) if k != idx], r.randrange(0, 3)):
            broken[j] = r.randrange(1, 6)
        st = _expense_state(r, n_items, broken)
        ans = RULE_OPTIONS[rule - 1] if rule else "No rule"
        q = f"Which policy rule does item E{idx + 1} violate? If it violates none of the rules, answer 'No rule'."
        return _choice_q(qid, 2, st, q, list(RULE_OPTIONS), ans, "business", "t2_expense_policy", seed, difficulty=n_items)
    k = r.randrange(1, min(6, n_items + 1))
    broken = {j: r.randrange(1, 6) for j in r.sample(range(n_items), k)}
    st = _expense_state(r, n_items, broken)
    q = "How many line items in this report violate at least one of the five policy rules?"
    return _score_q(qid, 2, st, q, COUNT_LEVELS, k - 1, "business", "t2_expense_policy", seed, difficulty=n_items)


def check_t2_expense_policy(q: Question) -> bool:
    p, items = q.state["policy"], q.state["items"]
    viol = {it["id"]: item_violations(p, it) for it in items}
    if q.format == "yesno":
        return any(viol.values()) == (q.ground_truth == "yes")
    if q.format == "choice":
        iid = re.search(r"item (E\d+) violate", q.question).group(1)
        v = viol[iid]
        ans = q.options[q.ground_truth]
        if not v:
            return ans == "No rule"
        return len(v) == 1 and ans == RULE_OPTIONS[v[0] - 1]
    n = sum(1 for v in viol.values() if v)
    return 1 <= n <= 5 and q.ground_truth == n - 1


# ---- t2_zebra ----------------------------------------------------------------------------
PETS = ["dog", "cat", "fish", "parrot", "rabbit"]
DRINKS = ["tea", "coffee", "milk", "juice", "water"]


def zebra_solve(n: int, names: list[str], pets: list[str], drinks: list[str], clues: list[dict]) -> list[tuple]:
    """Independent brute-force solver. Returns all (pet_perm, drink_perm) satisfying every clue."""
    sols = []
    for pp in itertools.permutations(pets, n):
        if not all(_clue_ok(c, names, pp, None, partial=True) for c in clues):
            continue
        for dp in itertools.permutations(drinks, n):
            if all(_clue_ok(c, names, pp, dp) for c in clues):
                sols.append((pp, dp))
    return sols


def _clue_ok(c: dict, names, pp, dp, partial=False) -> bool:
    """Evaluate one clue. With partial=True, drink clues are skipped (dp unknown)."""
    t = c["t"]
    pos = {nm: i for i, nm in enumerate(names)}
    if t == "has_pet":
        return pp[pos[c["name"]]] == c["pet"]
    if t == "not_pet":
        return pp[pos[c["name"]]] != c["pet"]
    if t == "pet_left_of_pet":   # pet a is immediately left of pet b
        return pp.index(c["a"]) + 1 == pp.index(c["b"])
    if t == "house_not_pet":
        return pp[c["house"] - 1] != c["pet"]
    if partial:
        return True
    if t == "drinks":
        return dp[pos[c["name"]]] == c["drink"]
    if t == "not_drink":
        return dp[pos[c["name"]]] != c["drink"]
    if t == "pet_drinks":        # the person with pet drinks drink
        return dp[pp.index(c["pet"])] == c["drink"]
    if t == "drink_left_of_pet": # drinker is immediately left of pet owner
        return dp.index(c["drink"]) + 1 == pp.index(c["pet"])
    if t == "house_drinks":
        return dp[c["house"] - 1] == c["drink"]
    raise ValueError(t)


def _clue_text(c: dict) -> str:
    t = c["t"]
    return {
        "has_pet": lambda: f"{c['name']} has the {c['pet']}.",
        "not_pet": lambda: f"{c['name']} does not have the {c['pet']}.",
        "pet_left_of_pet": lambda: f"The {c['a']} owner lives immediately to the left of the {c['b']} owner.",
        "house_not_pet": lambda: f"The person in house {c['house']} does not have the {c['pet']}.",
        "drinks": lambda: f"{c['name']} drinks {c['drink']}.",
        "not_drink": lambda: f"{c['name']} does not drink {c['drink']}.",
        "pet_drinks": lambda: f"The person with the {c['pet']} drinks {c['drink']}.",
        "drink_left_of_pet": lambda: f"The {c['drink']} drinker lives immediately to the left of the {c['pet']} owner.",
        "house_drinks": lambda: f"The person in house {c['house']} drinks {c['drink']}.",
    }[t]()


def _random_true_clue(r: Rng, names, pp, dp) -> dict:
    n = len(names)
    i = r.randrange(n)
    t = r.choice(["has_pet", "not_pet", "pet_left_of_pet", "house_not_pet", "drinks", "not_drink", "pet_drinks",
                  "drink_left_of_pet", "house_drinks", "not_pet", "not_drink", "pet_drinks"])
    if t == "has_pet":
        return {"t": t, "name": names[i], "pet": pp[i]}
    if t == "not_pet":
        return {"t": t, "name": names[i], "pet": r.choice([p for p in pp if p != pp[i]])}
    if t == "pet_left_of_pet":
        j = r.randrange(n - 1)
        return {"t": t, "a": pp[j], "b": pp[j + 1]}
    if t == "house_not_pet":
        return {"t": t, "house": i + 1, "pet": r.choice([p for p in pp if p != pp[i]])}
    if t == "drinks":
        return {"t": t, "name": names[i], "drink": dp[i]}
    if t == "not_drink":
        return {"t": t, "name": names[i], "drink": r.choice([d for d in dp if d != dp[i]])}
    if t == "pet_drinks":
        return {"t": t, "pet": pp[i], "drink": dp[i]}
    if t == "drink_left_of_pet":
        j = r.randrange(n - 1)
        return {"t": t, "drink": dp[j], "pet": pp[j + 1]}
    return {"t": t, "house": i + 1, "drink": dp[i]}


def _zebra_puzzle(r: Rng, n: int):
    names = r.sample(FIRST, n)
    pp = tuple(r.sample(PETS, n))
    dp = tuple(r.sample(DRINKS, n))
    clues: list[dict] = []
    for _ in range(60):
        c = _random_true_clue(r, names, pp, dp)
        if c in clues:
            continue
        clues.append(c)
        sols = zebra_solve(n, names, list(pp), list(dp), clues)
        assert (pp, dp) in sols
        if len(sols) == 1:
            break
    else:
        raise RuntimeError("zebra: no unique solution")
    # drop clues that are redundant (keeps the puzzle minimal-ish)
    for c in list(clues):
        rest = [x for x in clues if x != c]
        if len(zebra_solve(n, names, list(pp), list(dp), rest)) == 1:
            clues = rest
    r.shuffle(clues)
    intro = (f"{n} friends live in a row of houses numbered 1 to {n} from left to right. "
             + "; ".join(f"house {i + 1}: {nm}" for i, nm in enumerate(names)) + ". "
             f"Each has exactly one pet from {{{', '.join(sorted(pp))}}} and drinks exactly one drink from "
             f"{{{', '.join(sorted(dp))}}}; no two friends share a pet or a drink.")
    text = intro + "\n\nClues:\n" + "\n".join(f"{i + 1}. {_clue_text(c)}" for i, c in enumerate(clues))
    return names, pp, dp, clues, text


def gen_t2_zebra(r: Rng, qid: str, seed: int, fmt: str) -> Question:
    n = r.choice([4, 4, 5])
    names, pp, dp, clues, text = _zebra_puzzle(r, n)
    if fmt == "choice":
        if r.random() < 0.5:
            pet = r.choice(pp)
            q, ans, options = f"Who has the {pet}?", names[pp.index(pet)], list(names)
        else:
            nm = r.choice(names)
            q, ans, options = f"What does {nm} drink?", dp[names.index(nm)], list(dp)
        options = list(options)
        r.shuffle(options)
        return _choice_q(qid, 2, text, q, options, ans, "casual", "t2_zebra", seed, difficulty=len(clues))
    nm = r.choice(names)
    truth = r.random() < 0.5
    if r.random() < 0.5:
        pet = pp[names.index(nm)] if truth else r.choice([p for p in pp if p != pp[names.index(nm)]])
        q = f"Does {nm} have the {pet}?"
    else:
        drink = dp[names.index(nm)] if truth else r.choice([d for d in dp if d != dp[names.index(nm)]])
        q = f"Does {nm} drink {drink}?"
    return _yesno_q(qid, 2, text, q, truth, "casual", "t2_zebra", seed, difficulty=len(clues))


def _parse_zebra(state: str):
    m = re.search(r"(\d+) friends live", state)
    n = int(m.group(1))
    names = re.findall(r"house \d+: (\w+)", state)
    pets = re.search(r"one pet from \{(.+?)\}", state).group(1).split(", ")
    drinks = re.search(r"one drink from \{(.+?)\}", state).group(1).split(", ")
    clues = []
    for line in state.split("Clues:\n")[1].splitlines():
        s = re.sub(r"^\d+\. ", "", line.strip())
        clues.append(_parse_clue(s))
    return n, names, pets, drinks, clues


def _parse_clue(s: str) -> dict:
    pats = [
        (r"^(\w+) has the (\w+)\.$", lambda m: {"t": "has_pet", "name": m[1], "pet": m[2]}),
        (r"^(\w+) does not have the (\w+)\.$", lambda m: {"t": "not_pet", "name": m[1], "pet": m[2]}),
        (r"^The (\w+) owner lives immediately to the left of the (\w+) owner\.$", lambda m: {"t": "pet_left_of_pet", "a": m[1], "b": m[2]}),
        (r"^The person in house (\d+) does not have the (\w+)\.$", lambda m: {"t": "house_not_pet", "house": int(m[1]), "pet": m[2]}),
        (r"^(\w+) drinks (\w+)\.$", lambda m: {"t": "drinks", "name": m[1], "drink": m[2]}),
        (r"^(\w+) does not drink (\w+)\.$", lambda m: {"t": "not_drink", "name": m[1], "drink": m[2]}),
        (r"^The person with the (\w+) drinks (\w+)\.$", lambda m: {"t": "pet_drinks", "pet": m[1], "drink": m[2]}),
        (r"^The (\w+) drinker lives immediately to the left of the (\w+) owner\.$", lambda m: {"t": "drink_left_of_pet", "drink": m[1], "pet": m[2]}),
        (r"^The person in house (\d+) drinks (\w+)\.$", lambda m: {"t": "house_drinks", "house": int(m[1]), "drink": m[2]}),
    ]
    for pat, fn in pats:
        m = re.match(pat, s)
        if m:
            return fn(m)
    raise ValueError(s)


def check_t2_zebra(q: Question) -> bool:
    n, names, pets, drinks, clues = _parse_zebra(q.state)
    sols = zebra_solve(n, names, pets, drinks, clues)
    if len(sols) != 1:
        return False
    pp, dp = sols[0]
    if q.format == "choice":
        ans = q.options[q.ground_truth]
        m = re.match(r"Who has the (\w+)\?", q.question)
        if m:
            return names[pp.index(m.group(1))] == ans
        m = re.match(r"What does (\w+) drink\?", q.question)
        return dp[names.index(m.group(1))] == ans
    m = re.match(r"Does (\w+) have the (\w+)\?", q.question)
    if m:
        truth = pp[names.index(m.group(1))] == m.group(2)
    else:
        m = re.match(r"Does (\w+) drink (\w+)\?", q.question)
        truth = dp[names.index(m.group(1))] == m.group(2)
    return truth == (q.ground_truth == "yes")


# ---- t2_event_order ----------------------------------------------------------------------
EVENT_SETS = {
    "technical": ["Build 412 compiled", "Canary deploy started", "Alert fired", "Rollback executed", "Incident declared",
                  "Postmortem scheduled", "Config change merged", "Traffic shifted"],
    "legal": ["Notice served", "Response filed", "Hearing held", "Motion granted", "Settlement signed",
              "Appeal lodged", "Discovery closed", "Judgment entered"],
}
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]


def _order_ok(c: dict, order: list[str]) -> bool:
    p = {e: i for i, e in enumerate(order)}
    if c["t"] == "before":
        return p[c["a"]] < p[c["b"]]
    if c["t"] == "after":
        return p[c["a"]] > p[c["b"]]
    if c["t"] == "imm_before":
        return p[c["a"]] + 1 == p[c["b"]]
    raise ValueError(c)


def _constraint_text(c: dict) -> str:
    if c["t"] == "before":
        return f"'{c['a']}' happened before '{c['b']}'."
    if c["t"] == "after":
        return f"'{c['a']}' happened after '{c['b']}'."
    return f"'{c['a']}' happened immediately before '{c['b']}'."


def _parse_constraint(s: str) -> dict:
    m = re.match(r"^'(.+?)' happened immediately before '(.+?)'\.$", s)
    if m:
        return {"t": "imm_before", "a": m[1], "b": m[2]}
    m = re.match(r"^'(.+?)' happened before '(.+?)'\.$", s)
    if m:
        return {"t": "before", "a": m[1], "b": m[2]}
    m = re.match(r"^'(.+?)' happened after '(.+?)'\.$", s)
    if m:
        return {"t": "after", "a": m[1], "b": m[2]}
    raise ValueError(s)


def consistent_orders(events: list[str], cons: list[dict]) -> list[tuple]:
    return [p for p in itertools.permutations(events) if all(_order_ok(c, list(p)) for c in cons)]


def _random_constraint(r: Rng, order: list[str]) -> dict:
    n = len(order)
    t = r.choice(["before", "before", "after", "imm_before"])
    if t == "imm_before":
        i = r.randrange(n - 1)
        return {"t": t, "a": order[i], "b": order[i + 1]}
    i, j = sorted(r.sample(range(n), 2))
    return {"t": t, "a": order[i], "b": order[j]} if t == "before" else {"t": t, "a": order[j], "b": order[i]}


def _order_puzzle(r: Rng, n: int, unique: bool):
    domain = r.choice(sorted(EVENT_SETS))
    events = r.sample(EVENT_SETS[domain], n)
    true_order = list(events)
    r.shuffle(true_order)
    cons: list[dict] = []
    for _ in range(40):
        c = _random_constraint(r, true_order)
        if c in cons:
            continue
        cons.append(c)
        sols = consistent_orders(events, cons)
        if unique and len(sols) == 1:
            break
        if not unique and len(sols) <= max(2, n):
            break
    if unique:
        for c in list(cons):
            rest = [x for x in cons if x != c]
            if len(consistent_orders(events, rest)) == 1:
                cons = rest
    r.shuffle(cons)
    listed = sorted(events)
    text = (f"Events (listed in no particular order): " + "; ".join(listed) + ".\n\nStatements:\n" +
            "\n".join(f"{i + 1}. {_constraint_text(c)}" for i, c in enumerate(cons)))
    return domain, events, true_order, cons, text


def gen_t2_event_order(r: Rng, qid: str, seed: int, fmt: str) -> Question:
    n = r.choice([5, 5, 6, 7])
    if fmt == "choice":
        domain, events, true_order, cons, text = _order_puzzle(r, n, unique=True)
        k = r.randrange(n)
        q = f"Given the statements, which event happened {ORDINALS[k]}?"
        options = sorted(events)
        return _choice_q(qid, 2, text, q, options, true_order[k], domain, "t2_event_order", seed, difficulty=len(cons))
    if fmt == "yesno":
        domain, events, true_order, cons, text = _order_puzzle(r, n, unique=False)
        consistent = r.random() < 0.5
        if consistent:
            cand = list(r.choice(consistent_orders(events, cons)))
        else:
            for _ in range(100):
                cand = list(true_order)
                i, j = r.sample(range(n), 2)
                cand[i], cand[j] = cand[j], cand[i]
                if not all(_order_ok(c, cand) for c in cons):
                    break
            else:
                raise RuntimeError("could not build inconsistent order")
        text += "\n\nProposed timeline:\n" + "\n".join(f"{i + 1}. {e}" for i, e in enumerate(cand))
        q = "Is the proposed timeline consistent with all of the statements?"
        return _yesno_q(qid, 2, text, q, consistent, domain, "t2_event_order", seed, difficulty=len(cons))
    # score: how many of five claims are guaranteed true in every consistent order
    for _ in range(100):
        domain, events, true_order, cons, text = _order_puzzle(r, n, unique=False)
        sols = consistent_orders(events, cons)
        claims = []
        seen = set()
        while len(claims) < 5:
            a, b = r.sample(events, 2)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            claims.append({"t": "before", "a": a, "b": b})
        guaranteed = sum(1 for c in claims if all(_order_ok(c, list(p)) for p in sols))
        if 1 <= guaranteed <= 5:
            break
    else:
        raise RuntimeError("could not build score order puzzle")
    text += "\n\nClaims:\n" + "\n".join(f"({chr(97 + i)}) {_constraint_text(c)}" for i, c in enumerate(claims))
    q = "How many of the five claims are guaranteed to be true in every ordering consistent with the statements?"
    return _score_q(qid, 2, text, q, COUNT_LEVELS, guaranteed - 1, domain, "t2_event_order", seed, difficulty=len(cons))


def _parse_order_state(state: str):
    head, rest = state.split("\n\nStatements:\n", 1)
    events = head[len("Events (listed in no particular order): "):].rstrip(".").split("; ")
    parts = re.split(r"\n\n(Proposed timeline|Claims):\n", rest)
    stmts = [_parse_constraint(re.sub(r"^\d+\. ", "", l)) for l in parts[0].splitlines() if l.strip()]
    extra = {}
    if len(parts) == 3:
        extra[parts[1]] = parts[2]
    return events, stmts, extra


def check_t2_event_order(q: Question) -> bool:
    events, cons, extra = _parse_order_state(q.state)
    sols = consistent_orders(events, cons)
    if q.format == "choice":
        if len(sols) != 1:
            return False
        k = ORDINALS.index(re.search(r"happened (\w+)\?", q.question).group(1))
        return q.options[q.ground_truth] == sols[0][k]
    if q.format == "yesno":
        cand = [re.sub(r"^\d+\. ", "", l) for l in extra["Proposed timeline"].splitlines() if l.strip()]
        ok = sorted(cand) == sorted(events) and all(_order_ok(c, cand) for c in cons)
        return ok == (q.ground_truth == "yes")
    claims = [_parse_constraint(re.sub(r"^\(\w\) ", "", l)) for l in extra["Claims"].splitlines() if l.strip()]
    g = sum(1 for c in claims if all(_order_ok(c, list(p)) for p in sols))
    return 1 <= g <= 5 and q.ground_truth == g - 1


# ======================================================================================
# Registry and assembly
# ======================================================================================
# generator name -> (fn(r, qid, seed[, fmt]), checker, needs_fmt)
REGISTRY: dict[str, tuple[Callable, Callable, bool]] = {
    "t1_ticket_topic": (gen_t1_ticket_topic, check_t1_ticket_topic, False),
    "t1_invoice_status": (gen_t1_invoice_status, check_t1_invoice_status, False),
    "t1_medical_allergy": (gen_t1_medical_allergy, check_t1_medical_allergy, False),
    "t1_json_flag": (gen_t1_json_flag, check_t1_json_flag, False),
    "t1_memo_department": (gen_t1_memo_department, check_t1_memo_department, False),
    "t1_email_sender_company": (gen_t1_email_sender_company, check_t1_email_sender_company, False),
    "t1_log_event_level": (gen_t1_log_event_level, check_t1_log_event_level, False),
    "t1_contract_term": (gen_t1_contract_term, check_t1_contract_term, False),
    "t1_casual_plan_day": (gen_t1_casual_plan_day, check_t1_casual_plan_day, False),
    "t1_fields_present": (gen_t1_fields_present, check_t1_fields_present, False),
    "t1_log_count_level": (gen_t1_log_count_level, check_t1_log_count_level, False),
    "t1_ticket_priority": (gen_t1_ticket_priority, check_t1_ticket_priority, False),
    "t1_review_rating": (gen_t1_review_rating, check_t1_review_rating, False),
    "t2_meeting_rooms": (gen_t2_meeting_rooms, check_t2_meeting_rooms, True),
    "t2_expense_policy": (gen_t2_expense_policy, check_t2_expense_policy, True),
    "t2_zebra": (gen_t2_zebra, check_t2_zebra, True),
    "t2_event_order": (gen_t2_event_order, check_t2_event_order, True),
}

# (generator, format, count). Tier 1: 80 yesno / 80 choice / 40 score. Tier 2: same.
PLAN: list[tuple[str, str, int]] = [
    ("t1_ticket_topic", "yesno", 30), ("t1_invoice_status", "yesno", 20), ("t1_medical_allergy", "yesno", 15), ("t1_json_flag", "yesno", 15),
    ("t1_memo_department", "choice", 20), ("t1_email_sender_company", "choice", 15), ("t1_log_event_level", "choice", 15),
    ("t1_contract_term", "choice", 15), ("t1_casual_plan_day", "choice", 15),
    ("t1_fields_present", "score", 14), ("t1_log_count_level", "score", 10), ("t1_ticket_priority", "score", 8), ("t1_review_rating", "score", 8),
    ("t2_meeting_rooms", "yesno", 25), ("t2_expense_policy", "yesno", 25), ("t2_zebra", "yesno", 10), ("t2_event_order", "yesno", 20),
    ("t2_meeting_rooms", "choice", 20), ("t2_expense_policy", "choice", 20), ("t2_zebra", "choice", 25), ("t2_event_order", "choice", 15),
    ("t2_meeting_rooms", "score", 12), ("t2_expense_policy", "score", 14), ("t2_event_order", "score", 14),
]


def check_question(q: Question) -> bool:
    """Run the question's own checker. Tier 3 questions carry a cross-check flag instead."""
    if q.tier == 3:
        return bool(q.meta.get("cross_checked")) and q.format in ("yesno", "choice", "score")
    return REGISTRY[q.generator][1](q)


def generate_tiers_1_2(global_seed: int) -> list[Question]:
    out: list[Question] = []
    for gen_name, fmt, count in PLAN:
        fn, _, needs_fmt = REGISTRY[gen_name]
        tier = int(gen_name[1])
        for i in range(count):
            seed = stable_hash(global_seed, gen_name, fmt, i)
            r = random.Random(seed)
            qid = f"t{tier}-{gen_name[3:]}-{fmt}-{i:03d}"
            q = fn(r, qid, seed, fmt) if needs_fmt else fn(r, qid, seed)
            assert q.format == fmt, (gen_name, fmt, q.format)
            assert check_question(q), f"checker failed for {qid}"
            out.append(q)
    return out


# ======================================================================================
# TIER 3 — outside knowledge (frontier generator + independent frontier cross-check)
# ======================================================================================
TIER3_CACHE = DATA_DIR / "tier3_candidates.json"
TIER3_STATE = "No supporting document is provided for this question. Answer from general knowledge."

T3_GEN_PROMPT = """Generate {n} factual quiz questions as a JSON array. Requirements:
- Each question must have a single, unambiguous, verifiable answer that has been stable since before 2024 (no recent events, no opinions, no trivia that depends on the date).
- Spread across domains: geography, history, science, mathematics, literature, music, sports, technology, biology, astronomy, economics, art.
- Mix difficulty: about a third easy, a third medium, a third hard (hard = specific facts a well-read person might not know).
- Formats and the exact counts: {n_yesno} "yesno", {n_choice} "choice", {n_score} "score".
  - "yesno": a statement question answerable yes or no; "answer" is "yes" or "no". Make about half yes and half no.
  - "choice": 4 to 8 plausible options; "options" is a list of strings; "answer" is the 0-based index of the correct option.
  - "score": an ordinal question with exactly 5 ordered bins from low to high (for example year ranges, magnitude bins, count bins); "options" is the 5 bin labels; "answer" is the 0-based index of the correct bin.
- Fields per item: {{"format": ..., "domain": ..., "question": ..., "options": [...] or null, "answer": ...}}
Return only the JSON array, no prose."""

T3_CHECK_PROMPT = """You are verifying a quiz question. Answer from your own knowledge only.
Question: {question}
{options_block}
Reply with only the answer: for yes/no questions reply "yes" or "no"; for option questions reply with the 0-based index of the correct option as an integer. If the question is ambiguous or you are unsure, reply "unsure"."""


def _t3_options_block(c: dict) -> str:
    if not c.get("options"):
        return ""
    return "Options:\n" + "\n".join(f"{i}: {o}" for i, o in enumerate(c["options"]))


def generate_tier3(cfg: dict) -> dict:
    """Calls the API (generator + checker). Writes the cache and raw responses. Returns the cache dict."""
    import httpx
    from .common import RAW_DIR, get_api_key, scrub_secret
    key = get_api_key(cfg)
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set; tier 3 needs it.")
    t3 = cfg["tier3"]
    n = t3["candidates"]
    n_yesno, n_choice = round(n * 0.4), round(n * 0.4)
    n_score = n - n_yesno - n_choice
    base = cfg["openrouter"]["base_url"] + cfg["openrouter"]["chat_path"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **cfg["openrouter"].get("app_headers", {})}
    raw_dir = RAW_DIR / "tier3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def chat(model: str, prompt: str, tag: str, max_tokens: int) -> str:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "reasoning": {"effort": "medium"}}
        with httpx.Client(timeout=600) as client:
            resp = client.post(base, headers=headers, json=body)
        data = resp.json()
        (raw_dir / f"{tag}.json").write_text(json.dumps(scrub_secret({"request": body, "status": resp.status_code, "response": data},
                                                                     key), indent=2), encoding="utf-8")
        resp.raise_for_status()
        return data["choices"][0]["message"]["content"]

    gen_text = chat(t3["generator_slug"], T3_GEN_PROMPT.format(n=n, n_yesno=n_yesno, n_choice=n_choice, n_score=n_score),
                    "generate", 16000)
    m = re.search(r"\[.*\]", gen_text, re.S)
    cands = json.loads(m.group(0))
    out = []
    for i, c in enumerate(cands):
        c = dict(c)
        c["checker_raw"] = chat(t3["checker_slug"], T3_CHECK_PROMPT.format(question=c["question"], options_block=_t3_options_block(c)),
                                f"check_{i:03d}", 2000).strip()
        raw = c["checker_raw"].strip().strip('."').lower()
        if c["format"] == "yesno":
            c["checker_answer"] = raw if raw in ("yes", "no") else None
            c["agree"] = c["checker_answer"] == str(c["answer"]).lower()
        else:
            mm = re.match(r"^-?\d+", raw)
            c["checker_answer"] = int(mm.group(0)) if mm else None
            c["agree"] = c["checker_answer"] == int(c["answer"])
        out.append(c)
    cache = {"generator": t3["generator_slug"], "checker": t3["checker_slug"], "generated_at": date.today().isoformat(),
             "candidates": out}
    TIER3_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"tier 3: {sum(c['agree'] for c in out)}/{len(out)} candidates agreed; cache -> {TIER3_CACHE}")
    return cache


def load_tier3(global_seed: int, keep: int) -> list[Question]:
    if not TIER3_CACHE.exists():
        return []
    cache = json.loads(TIER3_CACHE.read_text(encoding="utf-8"))
    agreed = [c for c in cache["candidates"] if c.get("agree")]
    want = {"yesno": round(keep * 0.4), "choice": round(keep * 0.4)}
    want["score"] = keep - want["yesno"] - want["choice"]
    out, taken = [], {f: 0 for f in want}
    for c in agreed:
        f = c["format"]
        if f not in want or taken[f] >= want[f]:
            continue
        if f == "choice" and not (4 <= len(c["options"]) <= 8):
            continue
        if f == "score" and len(c["options"]) != 5:
            continue
        i = taken[f]
        qid = f"t3-knowledge-{f}-{i:03d}"
        meta = {"cross_checked": True, "generator_model": cache["generator"], "checker_model": cache["checker"],
                "ground_truth_note": "agreement of two frontier models, not verified against a source"}
        if f == "yesno":
            q = _yesno_q(qid, 3, TIER3_STATE, c["question"], str(c["answer"]).lower() == "yes", c.get("domain", "knowledge"),
                         "t3_knowledge", stable_hash(global_seed, "t3", i), meta=meta)
        elif f == "choice":
            q = _choice_q(qid, 3, TIER3_STATE, c["question"], list(c["options"]), c["options"][int(c["answer"])],
                          c.get("domain", "knowledge"), "t3_knowledge", stable_hash(global_seed, "t3", i), meta=meta)
        else:
            q = _score_q(qid, 3, TIER3_STATE, c["question"], list(c["options"]), int(c["answer"]), c.get("domain", "knowledge"),
                         "t3_knowledge", stable_hash(global_seed, "t3", i), meta=meta)
        out.append(q)
        taken[f] += 1
    return out


def select_labelswap(qs: list[Question], global_seed: int, n: int) -> None:
    r = random.Random(stable_hash(global_seed, "labelswap"))
    for tier in (1, 2):
        pool = sorted(q.id for q in qs if q.tier == tier and q.format == "yesno")
        chosen = set(r.sample(pool, n // 2))
        for q in qs:
            if q.id in chosen:
                q.in_labelswap = True


def build(cfg: dict) -> list[Question]:
    seed = cfg["seed"]
    qs = generate_tiers_1_2(seed)
    t3 = load_tier3(seed, cfg["counts"]["tier3"])
    qs += t3
    select_labelswap(qs, seed, cfg["counts"]["labelswap"])
    write_jsonl(QUESTIONS_PATH, [q.to_dict() for q in qs])
    return qs


def summarize(qs: list[Question]) -> str:
    from collections import Counter
    lines = [f"questions: {len(qs)}"]
    for t in (1, 2, 3):
        c = Counter(q.format for q in qs if q.tier == t)
        d = Counter(q.domain for q in qs if q.tier == t)
        lines.append(f"  tier {t}: {sum(c.values())}  formats={dict(c)}  domains={dict(d)}")
    lines.append(f"  labelswap: {sum(q.in_labelswap for q in qs)}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier3", action="store_true", help="call the API to (re)build the tier-3 cache, then rebuild")
    ap.add_argument("--validate", action="store_true", help="run every checker on data/questions.jsonl")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.validate:
        qs = [Question.from_dict(d) for d in read_jsonl(QUESTIONS_PATH)]
        bad = [q.id for q in qs if not check_question(q)]
        print(f"validated {len(qs)} questions; failures: {bad}")
        return 1 if bad else 0
    if args.tier3:
        generate_tier3(cfg)
    qs = build(cfg)
    print(summarize(qs))
    if not any(q.tier == 3 for q in qs):
        print("note: tier 3 not present (run `python -m src.generate --tier3` with OPENROUTER_API_KEY set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
