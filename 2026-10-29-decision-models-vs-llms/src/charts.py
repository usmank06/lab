"""Build the SVG figures for the write-up from results/*.csv.

Writes the SVGs to website/decision-models-vs-llms/; the page inlines them.
Run from the lab folder: python -m src.charts Colours follow the site's
palette: one blue accent for decision models, orange for instant LLMs, green for the
reasoning model, zinc greys for chrome. Marker shape doubles as the identity channel
(circle / square / triangle) so colour is never the only cue.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE.parent.parent / "website" / "decision-models-vs-llms"
OUT.mkdir(parents=True, exist_ok=True)

FONT = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
INK = "#111111"
INK2 = "#3f3f46"
MUTED = "#71717a"
GRID = "#efeff1"
AXIS = "#d4d4d8"
SURFACE = "#ffffff"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#16a34a"
BLUE_SOFT = "#93c5fd"

KIND = {
    "jev-1.14": "decision", "decider-v1.2": "decision", "elephant-1": "decision",
    "microsoft-decision-1": "decision", "d1-ultra": "decision",
    "qwen3.5-9b": "instant", "qwen3.5-27b": "instant", "gpt-6.1-sol": "reasoning",
    "qwen3.5-9b-reasoning": "reasoning",   # follow-up control (HANDOFF-2.md): same weights as qwen3.5-9b, thinking on
}
MAIN_ARMS = ["jev-1.14", "decider-v1.2", "elephant-1", "microsoft-decision-1", "d1-ultra", "qwen3.5-9b", "qwen3.5-27b",
             "gpt-6.1-sol"]   # the main-run arms; the scatter (figure 1) keeps to these
COLOR = {"decision": BLUE, "instant": ORANGE, "reasoning": GREEN}
LABEL = {
    "jev-1.14": "Jev 1.14", "decider-v1.2": "Decider V1.2", "elephant-1": "Elephant 1",
    "microsoft-decision-1": "Microsoft-Decision-1", "d1-ultra": "d1-ultra",
    "qwen3.5-9b": "Qwen3.5-9B", "qwen3.5-27b": "Qwen3.5-27B", "gpt-6.1-sol": "GPT-6.1 Sol",
    "qwen3.5-9b-reasoning": "Qwen3.5-9B, thinking",
}
PAIRS = [("decider-v1.2", "qwen3.5-27b"), ("elephant-1", "qwen3.5-9b"), ("microsoft-decision-1", "qwen3.5-9b")]
REASONING_BASE = {"qwen3.5-9b": "qwen3.5-9b-reasoning"}   # one-token base -> the same weights with reasoning on
PROMPT_ARMS = [("qwen3.5-9b", "a", "original"), ("qwen3.5-9b-prompt-b", "b", "terse"), ("qwen3.5-9b-prompt-c", "c", "careful")]


def rows(name):
    with open(RES / name, newline="") as f:
        return list(csv.DictReader(f))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fnum(s) -> float | None:
    """A CSV number, or None when the cell is empty / nan (a tier with no data)."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def marker(kind, x, y, r=5.5, fill=None, extra=""):
    """A mark with a 2px surface ring. Shape by kind."""
    c = fill or COLOR[kind]
    ring = f'stroke="{SURFACE}" stroke-width="2"'
    if kind == "decision":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}" {ring} {extra}/>'
    if kind == "instant":
        s = r * 1.8
        return f'<rect x="{x - s/2:.1f}" y="{y - s/2:.1f}" width="{s:.1f}" height="{s:.1f}" rx="1.5" fill="{c}" {ring} {extra}/>'
    h = r * 2.1
    pts = f"{x:.1f},{y - h*0.6:.1f} {x - h*0.58:.1f},{y + h*0.4:.1f} {x + h*0.58:.1f},{y + h*0.4:.1f}"
    return f'<polygon points="{pts}" fill="{c}" {ring} stroke-linejoin="round" {extra}/>'


def text(x, y, s, size=12, fill=INK2, anchor="start", weight=400, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {extra}>{esc(s)}</text>')


def svg_open(w, h, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-labelledby="{title[0]} {title[1]}" font-family=\'{FONT}\' '
            f'style="font-variant-numeric: tabular-nums; max-width: 100%; height: auto;">'
            f'<title id="{title[0]}">{esc(title[1])}</title><desc id="{desc[0]}">{esc(desc[1])}</desc>')


def legend(x, y, items):
    """items: list of (kind, label) or (kind, label, fill). Returns svg + width."""
    out, cx = [], x
    for item in items:
        kind, lab = item[0], item[1]
        fill = item[2] if len(item) > 2 else None
        out.append(marker(kind, cx + 6, y - 4, r=5, fill=fill))
        out.append(text(cx + 18, y, lab, 12, INK2))
        cx += 18 + len(lab) * 6.6 + 22
    return "".join(out)


# ---------------------------------------------------------------- figure 1: accuracy vs cost

def fig_accuracy_vs_cost():
    mt = {(r["arm"], r["tier"]): r for r in rows("metrics_by_tier.csv")}
    tier = "2"
    W, H = 680, 440
    pad_l, pad_r, pad_t, pad_b = 48, 40, 64, 54
    pw = W - pad_l - pad_r
    ph = H - pad_t - pad_b
    xmin, xmax = math.log10(0.004), math.log10(3.0)
    ymin, ymax = 0.3, 1.0

    def X(cost):
        return pad_l + (math.log10(cost) - xmin) / (xmax - xmin) * pw

    def Y(acc):
        return pad_t + (ymax - acc) / (ymax - ymin) * ph

    offs = {
        "decider-v1.2": (12, 4, "start"), "d1-ultra": (12, 4, "start"),
        "elephant-1": (12, 4, "start"), "microsoft-decision-1": (-12, 4, "end"),
        "jev-1.14": (12, 4, "start"), "qwen3.5-9b": (-12, 4, "end"),
        "qwen3.5-27b": (12, 4, "start"), "gpt-6.1-sol": (-14, 4, "end"),
    }
    s = [svg_open(W, H, ("f1t", "Tier 2 accuracy against cost per 1,000 decisions"),
                  ("f1d", "Scatter on a log cost axis. Decision models sit top-left: cheapest and more accurate than the raw Qwen models they are built from. The reasoning model is top-right: perfect, at 100 to 200 times the cost."))]
    s.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
    for yv in (0.4, 0.6, 0.8, 1.0):
        y = Y(yv)
        s.append(f'<line x1="{pad_l}" x2="{pad_l + pw:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(text(pad_l - 8, y + 4, f"{yv:.1f}", 11, MUTED, "end"))
    for c in (0.01, 0.1, 1.0):
        x = X(c)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{pad_t + ph:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(text(x, pad_t + ph + 18, {0.01: "$0.01", 0.1: "$0.10", 1.0: "$1"}[c], 11, MUTED, "middle"))
    s.append(f'<line x1="{pad_l}" x2="{pad_l + pw:.1f}" y1="{pad_t + ph:.1f}" y2="{pad_t + ph:.1f}" stroke="{AXIS}" stroke-width="1"/>')
    b = float(mt[("baseline", tier)]["accuracy"])
    yb = Y(b)
    s.append(f'<line x1="{pad_l}" x2="{pad_l + pw:.1f}" y1="{yb:.1f}" y2="{yb:.1f}" stroke="{AXIS}" stroke-width="1" stroke-dasharray="3 4"/>')
    s.append(text(pad_l + pw, yb - 6, f"majority-class baseline {b:.2f}", 11, MUTED, "end"))
    for a, bse in PAIRS:
        ra, rb = mt[(a, tier)], mt[(bse, tier)]
        s.append(f'<line x1="{X(float(ra["cost_per_1k_actual"])):.1f}" y1="{Y(float(ra["accuracy"])):.1f}" '
                 f'x2="{X(float(rb["cost_per_1k_actual"])):.1f}" y2="{Y(float(rb["accuracy"])):.1f}" '
                 f'stroke="{AXIS}" stroke-width="1"/>')
    for arm in MAIN_ARMS:
        kind = KIND[arm]
        r = mt[(arm, tier)]
        cost, acc = float(r["cost_per_1k_actual"]), float(r["accuracy"])
        x, y = X(cost), Y(acc)
        tip = f'<title>{LABEL[arm]}: accuracy {acc:.3f}, ${cost:.4f} per 1,000 decisions, p50 {float(r["latency_p50_ms"]):.0f} ms</title>'
        s.append(f'<g>{tip}{marker(kind, x, y, r=6)}</g>')
        dx, dy, anc = offs[arm]
        s.append(text(x + dx, y + dy, LABEL[arm], 12, INK2, anc))
    s.append(text(pad_l + pw / 2, H - 14, "cost per 1,000 decisions, USD, log scale", 11, MUTED, "middle"))
    s.append(text(6, pad_t - 14, "accuracy", 11, MUTED, "start"))
    s.append(legend(pad_l, 22, [("decision", "decision model"), ("instant", "LLM, one token"), ("reasoning", "LLM, reasoning")]))
    s.append(f'<line x1="{W - 150}" x2="{W - 128}" y1="18" y2="18" stroke="{AXIS}" stroke-width="1"/>')
    s.append(text(W - 122, 22, "same base model", 12, INK2))
    s.append("</svg>")
    (OUT / "fig-accuracy-vs-cost.svg").write_text("".join(s))


# ---------------------------------------------------------------- figure 2: matched pairs (dumbbells)

def fig_matched_pairs():
    mp = rows("matched_pair_deltas.csv")
    acc = {(r["pair_arm"], r["base_arm"], r["tier"]): r for r in mp if r["metric"] == "accuracy"}
    W = 680
    pad_l, pad_r, pad_t, pad_b = 160, 150, 70, 36
    row_h = 44
    panel_gap = 48
    panel_h = 3 * row_h + 30
    H = pad_t + 2 * panel_h + panel_gap + pad_b - 30
    pw = W - pad_l - pad_r
    xmin, xmax = 0.5, 1.0

    def X(v):
        return pad_l + (v - xmin) / (xmax - xmin) * pw

    # the reasoning-on base (follow-up) where it was run: its tier accuracy per 9B pair
    reas = {}
    for a, b in PAIRS:
        rb = REASONING_BASE.get(b)
        for tier in ("1", "2"):
            if rb and (a, rb, tier) in acc and fnum(acc[(a, rb, tier)]["base_value"]) is not None:
                reas[(a, tier)] = (rb, acc[(a, rb, tier)])
    desc = ("Dumbbell chart, one panel per tier. On tier 1 the gaps are a few points. On tier 2, Decider is 31 points above "
            "Qwen3.5-27B and the two 9B heads are 10 to 15 points above Qwen3.5-9B.")
    if reas:
        parts = []
        for a, b in PAIRS:
            if (a, "2") in reas:
                rb, r = reas[(a, "2")]
                parts.append(f"{LABEL[a]} is {float(r['delta']):+.2f} against {LABEL[rb]} ({float(r['base_value']):.2f})")
        desc += " A grey triangle marks the same 9B base with reasoning on: " + "; ".join(parts) + "."
    s = [svg_open(W, H, ("f2t", "Matched pairs: each decision model against the model it was built from"), ("f2d", desc))]
    s.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
    names = {"decider-v1.2": ("Decider V1.2", "Qwen3.5-27B"), "elephant-1": ("Elephant 1", "Qwen3.5-9B"),
             "microsoft-decision-1": ("Microsoft-Decision-1", "Qwen3.5-9B")}
    for i, tier in enumerate(("1", "2")):
        py0 = pad_t + i * (panel_h + panel_gap)
        title = "Tier 1 · answer is in the text" if tier == "1" else "Tier 2 · answer needs reasoning"
        s.append(text(pad_l, py0 - 18, title, 13, INK, weight=600))
        for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            x = X(v)
            s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{py0 - 4}" y2="{py0 + 3 * row_h - 10:.1f}" stroke="{GRID}" stroke-width="1"/>')
            s.append(text(x, py0 + 3 * row_h + 4, f"{v:.1f}", 11, MUTED, "middle"))
        for j, (a, b) in enumerate(PAIRS):
            y = py0 + j * row_h + 12
            r = acc[(a, b, tier)]
            pa, pb = float(r["pair_value"]), float(r["base_value"])
            d, lo, hi = float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"])
            s.append(text(pad_l - 16, y - 2, names[a][0], 12, INK, "end", 500))
            s.append(text(pad_l - 16, y + 13, f"vs {names[a][1]}", 11, MUTED, "end"))
            xa, xb = X(pa), X(pb)
            s.append(f'<line x1="{xb:.1f}" x2="{xa:.1f}" y1="{y}" y2="{y}" stroke="{AXIS}" stroke-width="2" stroke-linecap="round"/>')
            if (a, tier) in reas:
                rb, rr = reas[(a, tier)]
                pr = float(rr["base_value"])
                s.append(f'<g><title>{names[a][1]} with reasoning on: {pr:.3f} ({names[a][0]} minus it {float(rr["delta"]):+.3f} '
                         f'[{float(rr["ci_lo"]):+.2f}, {float(rr["ci_hi"]):+.2f}])</title>{marker("reasoning", X(pr), y, r=5, fill=MUTED)}</g>')
            s.append(f'<g><title>{names[a][1]} (raw, one token): {pb:.3f}</title>{marker("instant", xb, y, r=5)}</g>')
            s.append(f'<g><title>{names[a][0]} (decision model): {pa:.3f}</title>{marker("decision", xa, y, r=5.5)}</g>')
            s.append(text(pad_l + pw + 14, y + 4, f"{d:+.3f}", 12, INK, "start", 600))
            s.append(text(pad_l + pw + 62, y + 4, f"[{lo:+.2f}, {hi:+.2f}]", 11, MUTED, "start"))
    s.append(text(W / 2, H - 6, "accuracy over 5 runs · delta with bootstrap 95% CI over questions", 11, MUTED, "middle"))
    s.append(legend(pad_l, 22, [("instant", "base, one token"), ("decision", "decision model built on it")]))
    if reas:
        s.append(legend(pad_l + 330, 22, [("reasoning", "same base, thinking on", MUTED)]))
    s.append("</svg>")
    (OUT / "fig-matched-pairs.svg").write_text("".join(s))


# ---------------------------------------------------------------- figure 3: routing frontier

def fig_routing():
    rt = rows("routing.csv")
    mt = {(r["arm"], r["tier"]): r for r in rows("metrics_by_tier.csv")}
    W, H = 680, 400
    pad_l, pad_r, pad_t, pad_b = 48, 150, 64, 54
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    xmin, xmax = math.log10(0.006), math.log10(2.5)
    ymin, ymax = 0.55, 1.0

    def X(c):
        return pad_l + (math.log10(c) - xmin) / (xmax - xmin) * pw

    def Y(a):
        return pad_t + (ymax - a) / (ymax - ymin) * ph

    s = [svg_open(W, H, ("f3t", "Routing on confidence, tier 2: decision model answers when sure, GPT-6.1 Sol otherwise"),
                  ("f3d", "Two curves from the decision model alone on the left to GPT alone on the right. Decider reaches 0.98 accuracy at under half of GPT's cost; Jev needs to route three quarters of questions to get there, at 80% of GPT's cost."))]
    s.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
    for yv in (0.6, 0.7, 0.8, 0.9, 1.0):
        y = Y(yv)
        s.append(f'<line x1="{pad_l}" x2="{pad_l + pw:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(text(pad_l - 8, y + 4, f"{yv:.1f}", 11, MUTED, "end"))
    for c in (0.01, 0.1, 1.0):
        x = X(c)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{pad_t + ph:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(text(x, pad_t + ph + 18, {0.01: "$0.01", 0.1: "$0.10", 1.0: "$1"}[c], 11, MUTED, "middle"))
    s.append(f'<line x1="{pad_l}" x2="{pad_l + pw:.1f}" y1="{pad_t + ph:.1f}" y2="{pad_t + ph:.1f}" stroke="{AXIS}" stroke-width="1"/>')

    gpt = mt[("gpt-6.1-sol", "2")]
    gx, gy = X(float(gpt["cost_per_1k_actual"])), Y(float(gpt["accuracy"]))

    series = [("decider-v1.2", BLUE, "Decider V1.2 → GPT", (0.8, 0.96)),
              ("jev-1.14", BLUE_SOFT, "Jev 1.14 → GPT", (0.8,))]
    lab_off = {("decider-v1.2", 0.8): (0, 18, "middle"), ("decider-v1.2", 0.96): (-8, -12, "end"),
               ("jev-1.14", 0.8): (14, 34, "start")}
    for arm, col, lab, taus in series:
        pts = [r for r in rt if r["arm"] == arm and r["tier"] == "2"]
        pts.sort(key=lambda r: float(r["tau"]))
        alone = mt[(arm, "2")]
        path = [(X(float(alone["cost_per_1k_actual"])), Y(float(alone["accuracy"])))]
        path += [(X(float(r["cost_per_1k"])), Y(float(r["accuracy"]))) for r in pts if float(r["tau"]) < 1.0]
        path.append((gx, gy))
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in path)
        s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        # start marker + label
        x0, y0 = path[0]
        s.append(f'<g><title>{LABEL[arm]} alone: {float(alone["accuracy"]):.3f}, ${float(alone["cost_per_1k_actual"]):.4f}/1k</title>{marker("decision", x0, y0, fill=col)}</g>')
        s.append(text(x0 + (10 if arm == "decider-v1.2" else -10), y0 + (4 if arm == "decider-v1.2" else 16), f"{LABEL[arm]} alone", 11, INK2, "start" if arm == "decider-v1.2" else "end"))
        for r in pts:
            t = float(r["tau"])
            if t in taus:
                x, y = X(float(r["cost_per_1k"])), Y(float(r["accuracy"]))
                s.append(f'<g><title>τ = {t:.2f}: routes {float(r["frac_routed"]) * 100:.0f}% to GPT, accuracy {float(r["accuracy"]):.3f}, ${float(r["cost_per_1k"]):.2f}/1k</title>{marker("decision", x, y, r=4, fill=col)}</g>')
                dx, dy, anc = lab_off[(arm, t)]
                s.append(text(x + dx, y + dy, f"τ {t:.2f} · {float(r['frac_routed']) * 100:.0f}% routed · ${float(r['cost_per_1k']):.2f}", 11, INK2 if arm == "decider-v1.2" else MUTED, anc))
    s.append(f'<g><title>GPT-6.1 Sol alone: {float(gpt["accuracy"]):.3f}, ${float(gpt["cost_per_1k_actual"]):.2f}/1k</title>{marker("reasoning", gx, gy)}</g>')
    s.append(text(gx + 14, gy + 4, "GPT-6.1 Sol alone", 12, INK2))
    s.append(text(gx + 14, gy + 19, f"${float(gpt['cost_per_1k_actual']):.2f} per 1,000", 11, MUTED))
    s.append(text(pad_l + pw / 2, H - 14, "cost per 1,000 decisions, USD, log scale · each point is a confidence threshold τ", 11, MUTED, "middle"))
    s.append(text(6, pad_t - 14, "accuracy", 11, MUTED, "start"))
    # legend (line keys)
    lx = pad_l
    for col, lab in ((BLUE, "Decider V1.2, then GPT below τ"), (BLUE_SOFT, "Jev 1.14, then GPT below τ")):
        s.append(f'<line x1="{lx}" x2="{lx + 22}" y1="18" y2="18" stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
        s.append(text(lx + 30, 22, lab, 12, INK2))
        lx += 30 + len(lab) * 6.6 + 24
    s.append("</svg>")
    (OUT / "fig-routing.svg").write_text("".join(s))


# ---------------------------------------------------------------- figure 4: tier 2 leaderboard with flips

def fig_tier2_bars():
    mt = {(r["arm"], r["tier"]): r for r in rows("metrics_by_tier.csv")}
    arms = [a for a in KIND if (a, "2") in mt and fnum(mt[(a, "2")]["accuracy"]) is not None]   # an arm not yet run is left out
    arms = sorted(arms, key=lambda a: -float(mt[(a, "2")]["accuracy"]))
    W = 680
    row_h = 34
    pad_l, pad_r, pad_t, pad_b = 150, 240, 64, 36
    H = pad_t + len(arms) * row_h + pad_b
    pw = W - pad_l - pad_r

    def X(v):
        return pad_l + v * pw

    order = ", ".join(f"{LABEL[a]} {float(mt[(a, '2')]['accuracy']):.2f}" for a in arms)
    s = [svg_open(W, H, ("f4t", "Tier 2 accuracy by model, with run-to-run flips and cost"),
                  ("f4d", f"Horizontal bars sorted by accuracy: {order}."))]
    s.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        x = X(v)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 8}" y2="{pad_t + len(arms) * row_h - 8:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(text(x, pad_t + len(arms) * row_h + 8, f"{v:.2f}", 11, MUTED, "middle"))
    s.append(text(X(0) + pw + 60, pad_t - 16, "flips", 11, MUTED, "start"))
    s.append(text(X(0) + pw + 115, pad_t - 16, "$ / 1,000", 11, MUTED, "start"))
    s.append(text(X(0) + pw + 185, pad_t - 16, "p50 ms", 11, MUTED, "start"))
    bl = float(mt[("baseline", "2")]["accuracy"])
    xb = X(bl)
    s.append(f'<line x1="{xb:.1f}" x2="{xb:.1f}" y1="{pad_t - 8}" y2="{pad_t + len(arms) * row_h - 8:.1f}" stroke="{AXIS}" stroke-width="1" stroke-dasharray="3 4"/>')
    s.append(text(xb, pad_t - 16, f"majority-class baseline {bl:.2f}", 11, MUTED, "middle"))
    for j, arm in enumerate(arms):
        r = mt[(arm, "2")]
        acc, flip = float(r["accuracy"]), float(r["flip_rate"])
        y = pad_t + j * row_h
        kind = KIND[arm]
        s.append(text(pad_l - 14, y + 15, LABEL[arm], 12, INK, "end", 500))
        w = X(acc) - X(0)
        s.append(f'<g><title>{LABEL[arm]}: accuracy {acc:.3f}, flip rate {flip:.3f}, ${float(r["cost_per_1k_actual"]):.4f} per 1,000, p50 {float(r["latency_p50_ms"]):.0f} ms</title>'
                 f'<path d="M{X(0):.1f},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v14 a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z" fill="{COLOR[kind]}"/></g>')
        s.append(text(X(acc) + 8, y + 15, f"{acc:.3f}", 12, INK, "start", 600))
        s.append(text(X(0) + pw + 60, y + 15, f"{flip * 100:.1f}%" if flip else "0", 12, INK2 if flip else MUTED))
        s.append(text(X(0) + pw + 115, y + 15, f"${float(r['cost_per_1k_actual']):.3f}", 12, INK2))
        s.append(text(X(0) + pw + 185, y + 15, f"{float(r['latency_p50_ms']):.0f}", 12, INK2))
    s.append(legend(pad_l, 22, [("decision", "decision model"), ("instant", "LLM, one token"), ("reasoning", "LLM, reasoning")]))
    s.append("</svg>")
    (OUT / "fig-tier2.svg").write_text("".join(s))


# ---------------------------------------------------------------- figure 5: prompt sensitivity of the one-token control

def fig_prompts():
    """Three horizontal bars: the one-token Qwen3.5-9B control under system prompts a / b / c, tier 2 accuracy."""
    try:
        ps = rows("prompt_sensitivity.csv")
    except FileNotFoundError:
        return
    by_arm = {r["arm"]: r for r in ps if r["scope"] == "all"}
    if not all(a in by_arm and fnum(by_arm[a]["accuracy"]) is not None for a, _, _ in PROMPT_ARMS):
        return
    mt = {(r["arm"], r["tier"]): r for r in rows("metrics_by_tier.csv")}
    W = 680
    row_h = 34
    pad_l, pad_r, pad_t, pad_b = 150, 190, 64, 36
    H = pad_t + len(PROMPT_ARMS) * row_h + pad_b
    pw = W - pad_l - pad_r

    def X(v):
        return pad_l + v * pw

    rng = by_arm.get("range")
    accs = {a: float(by_arm[a]["accuracy"]) for a, _, _ in PROMPT_ARMS}
    desc = ("Three horizontal bars, one per system prompt, for the same one-token Qwen3.5-9B on tier 2: " +
            ", ".join(f"prompt {v} ({name}) {accs[a]:.3f}" for a, v, name in PROMPT_ARMS) + "." +
            (f" Range {float(rng['accuracy']):.3f} with bootstrap 95% CI [{float(rng['ci_lo']):.3f}, {float(rng['ci_hi']):.3f}]." if rng else ""))
    s = [svg_open(W, H, ("f5t", "The one-token control under three system prompts, tier 2 accuracy"), ("f5d", desc))]
    s.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
    n = len(PROMPT_ARMS)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        x = X(v)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 8}" y2="{pad_t + n * row_h - 8:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(text(x, pad_t + n * row_h + 8, f"{v:.2f}", 11, MUTED, "middle"))
    s.append(text(X(0) + pw + 60, pad_t - 16, "95% CI", 11, MUTED, "start"))
    bl = float(mt[("baseline", "2")]["accuracy"])
    xb = X(bl)
    s.append(f'<line x1="{xb:.1f}" x2="{xb:.1f}" y1="{pad_t - 8}" y2="{pad_t + n * row_h - 8:.1f}" stroke="{AXIS}" stroke-width="1" stroke-dasharray="3 4"/>')
    s.append(text(xb, pad_t - 16, f"majority-class baseline {bl:.2f}", 11, MUTED, "middle"))
    for j, (arm, variant, name) in enumerate(PROMPT_ARMS):
        r = by_arm[arm]
        acc, lo, hi = float(r["accuracy"]), float(r["ci_lo"]), float(r["ci_hi"])
        y = pad_t + j * row_h
        s.append(text(pad_l - 14, y + 15, f"prompt {variant} · {name}", 12, INK, "end", 500))
        w = X(acc) - X(0)
        s.append(f'<g><title>Qwen3.5-9B, one token, system prompt {variant} ({name}): accuracy {acc:.3f} [{lo:.3f}, {hi:.3f}], '
                 f'Brier {float(r["brier"]):.3f}, flip rate {float(r["flip_rate"]):.3f}</title>'
                 f'<path d="M{X(0):.1f},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v14 a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z" fill="{COLOR["instant"]}"/></g>')
        s.append(text(X(acc) + 8, y + 15, f"{acc:.3f}", 12, INK, "start", 600))
        s.append(text(X(0) + pw + 60, y + 15, f"[{lo:.2f}, {hi:.2f}]", 12, INK2))
    s.append(legend(pad_l, 22, [("instant", "Qwen3.5-9B, one token, same weights and provider throughout")]))
    s.append("</svg>")
    (OUT / "fig-prompts.svg").write_text("".join(s))


if __name__ == "__main__":
    fig_accuracy_vs_cost()
    fig_matched_pairs()
    fig_routing()
    fig_tier2_bars()
    fig_prompts()
    for p in sorted(OUT.glob("*.svg")):
        print(p.name, p.stat().st_size)
