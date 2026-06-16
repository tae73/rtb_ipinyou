"""Portfolio concept diagrams — hand-built SVGs (GitHub camo-safe: inline attributes only,
no <style>/<filter>/<marker>/<script>). Generates EN + KO variants with identical geometry so the
bilingual READMEs stay in visual parity.

Usage:
    python scripts/portfolio/make_diagrams.py

Writes:
    assets/funnel_selection_bias.svg        + .ko.svg
    assets/escm2wc_architecture.svg         + .ko.svg
    assets/falsification_arc.svg            + .ko.svg
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets"

# palette (matches results/figures/portfolio)
NEURAL = "#4F46E5"
LGB = "#0D9488"
LR = "#6B7280"
POS = "#16A34A"
NEG = "#DC2626"
NS = "#9CA3AF"
INK = "#111827"
MUTE = "#374151"
GRIDBG = "#F9FAFB"
CARD = "#FFFFFF"
FONT = "DejaVu Sans, Segoe UI, Apple SD Gothic Neo, Malgun Gothic, sans-serif"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x, y, w, h, fill=CARD, stroke=INK, sw=1.5, rx=10, opacity=1.0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')


def text(x, y, s, size=15, fill=INK, weight="normal", anchor="middle", italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>')


def lines(x, y, rows, size=13, fill=INK, weight="normal", anchor="middle", lh=18, italic=False):
    return "".join(text(x, y + i * lh, r, size, fill, weight, anchor, italic)
                   for i, r in enumerate(rows))


def arrow(x1, y1, x2, y2, color=INK, width=2.2, head=11, halfw=5.5, dash=None):
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    p2 = (bx - halfw * math.sin(ang), by + halfw * math.cos(ang))
    p3 = (bx + halfw * math.sin(ang), by - halfw * math.cos(ang))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round"{da}/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}" '
            f'fill="{color}"/>')


def svg(w, h, body) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" font-family="{FONT}">'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{GRIDBG}"/>{body}</svg>')


def panel(x, y, w, h, fill=CARD, stroke=INK, sw=1.4, rx=10, shadow=True):
    """A card with a soft offset drop-shadow (no <filter>; GitHub-safe)."""
    s = rect(x + 3, y + 4, w, h, fill="#0F172A", stroke="none", sw=0, rx=rx, opacity=0.10) if shadow else ""
    return s + rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx)


def chip(x, y, w, h, label, fill="#F1F5F9", stroke="#94A3B8", tsize=10.5, tcol=INK):
    return rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.0, rx=6) + \
        text(x + w / 2, y + h / 2 + tsize * 0.35, label, tsize, tcol)


def nodec(cx, cy, r, glyph, stroke, fill="#FFFFFF", gsize=13):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.6"/>') + text(cx, cy + gsize * 0.34, glyph, gsize, stroke, "bold")


# ==================================================================================================
# Diagram 1 — Bid -> Win -> Click funnel with win selection bias
# ==================================================================================================
def funnel(L) -> str:
    W, H = 920, 430
    b = []
    b.append(text(W / 2, 40, L["title"], 21, INK, "bold"))
    b.append(text(W / 2, 66, L["subtitle"], 13, MUTE))

    # three funnel stages (boxes shrinking), connected by arrows
    stages = [
        (60, 120, 230, 150, LR, L["bids"], "129.5M", L["bids_note"]),
        (345, 140, 215, 130, LGB, L["wins"], "30.6M", L["wins_note"]),
        (615, 165, 245, 100, NEURAL, L["clicks"], "23K", L["clicks_note"]),
    ]
    for (x, y, w, h, col, name, n, note) in stages:
        b.append(rect(x, y, w, h, fill=CARD, stroke=col, sw=2.5))
        b.append(text(x + w / 2, y + 36, name, 17, col, "bold"))
        b.append(text(x + w / 2, y + 70, n, 26, INK, "bold"))
        b.append(text(x + w / 2, y + 95, note, 11.5, MUTE))

    # arrows + gate labels
    b.append(arrow(292, 195, 343, 205, INK, 2.6))
    b.append(text(318, 178, L["gate1"], 12, INK, "bold"))
    b.append(arrow(562, 205, 613, 215, INK, 2.6))
    b.append(text(588, 188, L["gate2"], 12, INK, "bold"))

    # censoring callout under the bid->win gate
    cy = 320
    b.append(rect(60, cy, 500, 86, fill="#FEF2F2", stroke=NEG, sw=1.6))
    b.append(text(80, cy + 28, L["censor_h"], 14, NEG, "bold", "start"))
    b.append(lines(80, cy + 50, L["censor_body"], 12.5, MUTE, anchor="start", lh=17))

    # the key insight callout (right)
    b.append(rect(590, cy, 270, 86, fill="#EEF2FF", stroke=NEURAL, sw=1.6))
    b.append(text(605, cy + 28, L["insight_h"], 13.5, NEURAL, "bold", "start"))
    b.append(lines(605, cy + 49, L["insight_body"], 12, MUTE, anchor="start", lh=16))
    return svg(W, H, "".join(b))


# ==================================================================================================
# Diagram 2 — ESCM2-WC 3-tower architecture (DR debiasing + dual-purpose Win Tower)
# ==================================================================================================
def architecture(L) -> str:
    W, H = 1180, 760
    PW, PC, PI = "#0D9488", "#4F46E5", "#9333EA"        # win / ctr / imputation
    ALLc, WONc, LOSTc, AMBER = "#6B7280", "#0D9488", "#D97706", "#D97706"
    pw, pc, pb, dh, ddr = "p̂_win", "p̂_ctr", "p̂_bid", "δ̂", "Δ_DR"
    b = []
    b.append(text(W / 2, 36, L["title"], 22, INK, "bold"))
    b.append(text(W / 2, 61, L["subtitle"], 13, MUTE))

    # ---------------- A. input features ----------------
    ax, ay, aw, ah = 32, 150, 150, 196
    b.append(panel(ax, ay, aw, ah, stroke=INK, sw=1.4))
    b.append(text(ax + aw / 2, ay + 22, L["feat_h"], 11.5, INK, "bold"))
    for i, g in enumerate(L["feat_groups"]):
        b.append(chip(ax + 13, ay + 33 + i * 21, aw - 26, 17, g, tsize=9.3))
    b.append(lines(ax + aw / 2, ay + ah - 22, L["feat_note"], 8.2, MUTE, lh=10))
    b.append(arrow(ax + aw, ay + ah / 2, 197, ay + ah / 2, INK, 2.0))

    # ---------------- B. shared trunk ----------------
    bx, by, bw, bh = 200, 158, 168, 180
    b.append(panel(bx, by, bw, bh, fill="#EEF2FF", stroke=PC, sw=1.6))
    b.append(text(bx + bw / 2, by + 22, L["trunk_h"], 11.5, PC, "bold"))
    for i in range(6):
        b.append(rect(bx + 20 + i * 8, by + 34, 6, 26, fill=PC, stroke="none", sw=0,
                      rx=1, opacity=0.25 + 0.12 * (i % 3)))
    b.append(text(bx + bw / 2 + 16, by + 50, L["embed_l"], 9, INK, anchor="start"))
    b.append(text(bx + bw / 2, by + 76, "Embed 30×32 = [B, 960]", 9, MUTE))
    b.append(chip(bx + 14, by + 88, bw - 28, 22, L["fm_l"], "#FFFFFF", PC, 8.4, MUTE))
    b.append(text(bx + bw / 2, by + 128, "⊕ concat", 10, PC, "bold"))
    b.append(rect(bx + 36, by + 138, bw - 72, 26, fill="#FFFFFF", stroke=PC, sw=1.4, rx=8))
    b.append(text(bx + bw / 2, by + 155, "z ∈ ℝ⁹⁹²", 12, INK, "bold"))

    # ---------------- C. three towers (horizontal MLP pipelines) ----------------
    b.append(text(620, 96, L["tower_note"], 9.3, MUTE))
    rows = [
        (160, PW, L["win_name"], L["win_role"], [992, 64, 32, 1], "σ", pw),
        (250, PC, L["ctr_name"], L["ctr_role"], [992, 128, 64, 1], "σ", pc),
        (340, PI, L["imp_name"], L["imp_role"], [992, 128, 64, 1], L["lin"], dh),
    ]
    zx, zy = bx + bw, by + 90  # trunk z exit point
    for (cy, col, name, role, dims, act, sym) in rows:
        b.append(text(410, cy - 21, f"{name}  ·  {role}", 10.2, col, "bold", "start"))
        b.append(arrow(zx, zy, 406, cy, col, 1.5, head=8, halfw=4))
        cx = 410
        for j, d in enumerate(dims):
            faded = j == 0
            b.append(rect(cx, cy - 15, 46, 30, fill="#E2E8F0" if faded else "#F8FAFC",
                          stroke="#94A3B8" if faded else col, sw=1.1, rx=5))
            b.append(text(cx + 23, cy + 4, str(d), 10.5, MUTE if faded else INK,
                          "normal" if faded else "bold"))
            if j < len(dims) - 1:
                b.append(arrow(cx + 46, cy, cx + 54, cy, col, 1.2, head=5, halfw=2.5))
            cx += 54
        b.append(arrow(cx - 8, cy, 632, cy, col, 1.3, head=6, halfw=3))
        b.append(nodec(645, cy, 13, act, col, gsize=11))            # activation
        b.append(arrow(658, cy, 663, cy, col, 1.3, head=6, halfw=3))
        b.append(rect(664, cy - 14, 74, 28, fill=col, stroke=col, sw=1.2, rx=14, opacity=0.16))
        b.append(text(701, cy + 4, sym, 13, col, "bold"))
        b.append(arrow(738, cy, 815, 360, col, 1.4, head=7, halfw=3.5))  # -> objective

    # dual-purpose Win Tower callout (top-right, level with Win row)
    dx, dy, dw, dh2 = 860, 120, 288, 116
    b.append(panel(dx, dy, dw, dh2, fill="#FFF7ED", stroke=AMBER, sw=1.5))
    b.append(text(dx + dw / 2, dy + 22, L["dual_h"], 12, AMBER, "bold"))
    b.append(lines(dx + 14, dy + 44, L["dual_a"], 9.6, MUTE, anchor="start", lh=14))
    b.append(lines(dx + 14, dy + 82, L["dual_b"], 9.6, MUTE, anchor="start", lh=14))
    b.append(arrow(740, 160, dx, dy + 28, AMBER, 1.5, dash="4,3"))

    # merge label
    b.append(text(828, 352, L["outputs"], 9, MUTE, anchor="start"))
    b.append(arrow(815, 360, 815, 386, INK, 2.2))

    # ---------------- D. training objective panel ----------------
    px, py, pwid, phgt = 250, 388, 900, 214
    b.append(panel(px, py, pwid, phgt, fill="#FFFFFF", stroke=INK, sw=1.5))
    b.append(text(px + pwid / 2, py + 22, L["obj_h"], 12.5, INK, "bold"))
    # the three combine formulas
    fchips = [
        (262, 296, L["w_h"], f"ŵ = clip( win / clip({pw}, 0.05, 1), 0, 10 )", L["w_sub"]),
        (566, 296, L["dr_h2"], f"{ddr} = {dh} + ŵ·(y − {pc} − {dh})", L["dr_sub"]),
        (870, 264, L["esmm_h2"], f"{pb} = {pw} × {pc}", L["esmm_sub"]),
    ]
    for (fx, fwid, h_, eq_, sub_) in fchips:
        b.append(rect(fx, py + 38, fwid, 46, fill="#F8FAFC", stroke="#CBD5E1", sw=1.1, rx=8))
        b.append(text(fx + 10, py + 55, h_, 9, MUTE, "bold", "start"))
        b.append(text(fx + fwid / 2, py + 72, eq_, 10, INK))
        b.append(text(fx + fwid / 2, py + 80, sub_, 7.8, MUTE))
    # five loss terminals
    losses = [
        ("ℒ_win", f"BCE({pw}, win)", L["dom_all"], ALLc),
        ("ℒ_ctr", f"MSE({ddr})  · DR", L["dom_all"], ALLc),
        ("ℒ_imp", f"MSE({dh}, y−{pc})", L["dom_won"], WONc),
        ("ℒ_joint", f"BCE({pb}, click)", L["dom_all"], ALLc),
        ("ℒ_cfr", f"⟨(1−win)·{dh}²⟩", L["dom_lost"], LOSTc),
    ]
    lw, lx = 170, 262
    for nm, eq_, dom, dc in losses:
        b.append(panel(lx, py + 100, lw - 8, 56, fill="#FFFFFF", stroke=dc, sw=1.3, shadow=False))
        b.append(text(lx + (lw - 8) / 2, py + 120, nm, 12, dc, "bold"))
        b.append(text(lx + (lw - 8) / 2, py + 136, eq_, 8.6, INK))
        b.append(rect(lx + (lw - 8) / 2 - 32, py + 142, 64, 13, fill=dc, stroke="none",
                      sw=0, rx=6, opacity=0.16))
        b.append(text(lx + (lw - 8) / 2, py + 151, dom, 7.6, dc, "bold"))
        lx += lw
    b.append(text(px + pwid / 2, py + phgt - 12,
                  "ℒ = ℒ_win + ( ℒ_ctr + 0.5·ℒ_imp ) + ℒ_joint + 0.1·ℒ_cfr",
                  12, INK, "bold"))

    # ---------------- E. legend ----------------
    ly = 640
    b.append(text(40, ly, L["leg_dom"] + ":", 9.5, INK, "bold", "start"))
    for i, (lab, c) in enumerate([(L["dom_all"], ALLc), (L["dom_won"], WONc), (L["dom_lost"], LOSTc)]):
        gx = 150 + i * 150
        b.append(rect(gx, ly - 9, 12, 12, fill=c, stroke="none", sw=0, rx=3, opacity=0.55))
        b.append(text(gx + 18, ly, lab, 9.2, MUTE, anchor="start"))
    b.append(text(640, ly, L["leg_fwd"], 9.2, MUTE, anchor="start"))
    b.append(f'<line x1="600" y1="{ly - 4}" x2="636" y2="{ly - 4}" stroke="{INK}" stroke-width="2"/>')
    b.append(text(870, ly, L["leg_sg"], 9.2, MUTE, anchor="start"))
    b.append(f'<line x1="830" y1="{ly - 4}" x2="866" y2="{ly - 4}" stroke="{AMBER}" '
             f'stroke-width="2" stroke-dasharray="4,3"/>')
    return svg(W, H, "".join(b))


# ==================================================================================================
# Diagram 3 — falsification arc (the honest research story)
# ==================================================================================================
def arc(L) -> str:
    W, H = 1000, 340
    b = []
    b.append(text(W / 2, 38, L["title"], 21, INK, "bold"))

    nodes = [
        (30, NS, L["n1_h"], L["n1"], True),
        (226, MUTE, L["n2_h"], L["n2"], False),
        (422, LGB, L["n3_h"], L["n3"], False),
        (618, NEURAL, L["n4_h"], L["n4"], False),
        (814, POS, L["n5_h"], L["n5"], False),
    ]
    y, w, h = 90, 176, 150
    for i, (x, col, head, body, struck) in enumerate(nodes):
        b.append(rect(x, y, w, h, fill=CARD, stroke=col, sw=2.2))
        b.append(text(x + w / 2, y + 26, f"{i + 1}", 13, col, "bold"))
        hcol = NEG if struck else col
        b.append(lines(x + w / 2, y + 50, head, 12.5, hcol, "bold", lh=15))
        b.append(lines(x + w / 2, y + 96, body, 11, MUTE, lh=14.5))
        if struck:  # small "retracted" badge instead of a strikethrough
            b.append(text(x + w / 2, y + h - 12, L["retracted"], 10, NEG, "bold"))
        if i < len(nodes) - 1:
            b.append(arrow(x + w, y + h / 2, x + w + 20, y + h / 2, INK, 2.4))

    b.append(rect(30, 268, 940, 50, fill="#EEF2FF", stroke=NEURAL, sw=1.6))
    b.append(text(W / 2, 298, L["verdict"], 13.5, NEURAL, "bold"))
    return svg(W, H, "".join(b))


# --------------------------------------------------------------------------------------------------
# Label sets (EN + KO) — identical geometry, different strings
# --------------------------------------------------------------------------------------------------
FUNNEL_EN = dict(
    title="Win selection bias in RTB: the Bid → Win → Click funnel",
    subtitle="Clicks are observable only where the bid won the auction",
    bids="ALL BIDS", bids_note="every auction entered",
    wins="WINS (impressions)", wins_note="win rate ≈ 24%",
    clicks="CLICKS", clicks_note="CTR ≈ 0.075% of wins",
    gate1="win auction", gate2="user clicks",
    censor_h="Lost bids are censored",
    censor_body=["For bids that LOSE, the click outcome is never observed.",
                 "CTR trained on winners only ⇒ P(click|win) ≠ P(click) — selection bias."],
    insight_h="Why it matters",
    insight_body=["Biased pCTR ⇒ biased value ⇒",
                  "biased bids. Debiasing aims",
                  "to recover unbiased pCTR."],
)
FUNNEL_KO = dict(
    title="RTB의 승리 선택 편향: Bid → Win → Click 퍼널",
    subtitle="클릭은 경매에서 낙찰된(win) 입찰에서만 관측된다",
    bids="전체 입찰", bids_note="참여한 모든 경매",
    wins="낙찰 (노출)", wins_note="낙찰률 ≈ 24%",
    clicks="클릭", clicks_note="낙찰 대비 CTR ≈ 0.075%",
    gate1="경매 낙찰", gate2="사용자 클릭",
    censor_h="패찰 입찰은 검열(censored)된다",
    censor_body=["패찰(lose)한 입찰은 클릭 여부를 전혀 관측할 수 없다.",
                 "낙찰자만으로 CTR 학습 ⇒ P(click|win) ≠ P(click) — 선택 편향."],
    insight_h="왜 중요한가",
    insight_body=["편향된 pCTR ⇒ 편향된 가치 ⇒",
                  "편향된 입찰. 디바이어싱은",
                  "비편향 pCTR 복원이 목표."],
)

ARCH_EN = dict(
    title="ESCM²-WC (DR): a 3-tower entire-space model with a dual-purpose Win Tower",
    subtitle="Shared embedding → Win / CTR / Imputation towers · doubly-robust click loss + ESMM constraint",
    feat_h="Input features (30)",
    feat_groups=["time", "geo · region/city", "slot / ad unit", "domain / creative",
                 "advertiser", "price / exchange"],
    feat_note=["categorical → Embed(32)", "numeric → Linear(1→32)"],
    trunk_h="Shared representation",
    embed_l="per-feature",
    fm_l="+ FM  ½((Σe)²−Σe²) = [B,32]",
    tower_note="each tower:  Dense → ReLU → Dropout 0.3        σ = sigmoid  ·  lin = identity",
    win_name="Win Tower", win_role="propensity  P(win|x)",
    ctr_name="CTR Tower", ctr_role="debiased  P(click|win,x)",
    imp_name="Imputation Tower", imp_role="CTR-error control variate",
    lin="lin",
    dual_h="Dual-purpose Win Tower",
    dual_a=["(a) training — DR propensity ŵ = win / P̂(win)", "      debiases the CTR loss"],
    dual_b=["(b) serving — win-rate model for", "      first-price bid shading (AUC ≈ 0.91)"],
    outputs="model outputs  p̂_win, p̂_ctr, δ̂",
    obj_h="Training objective — doubly-robust click loss + ESMM entire-space constraint",
    w_h="propensity weight", w_sub="self-normalized · p̂_win stop-grad",
    dr_h2="doubly-robust residual", dr_sub="unbiased if propensity OR imputation is correct",
    esmm_h2="ESMM constraint", esmm_sub="entire-space",
    dom_all="all bids", dom_won="won only", dom_lost="lost only",
    leg_dom="sample domain", leg_fwd="forward pass", leg_sg="stop-grad / weight feed",
)
ARCH_KO = dict(
    title="ESCM²-WC (DR): Win Tower를 이중 활용하는 3-tower 전체공간(entire-space) 모델",
    subtitle="공유 임베딩 → Win / CTR / Imputation 타워 · 이중 강건(DR) 클릭 손실 + ESMM 제약",
    feat_h="입력 피처 (30개)",
    feat_groups=["시간", "지역 · region/city", "슬롯 / 광고면", "도메인 / 크리에이티브",
                 "광고주", "가격 / 거래소"],
    feat_note=["범주형 → Embed(32)", "수치형 → Linear(1→32)"],
    trunk_h="공유 표현(representation)",
    embed_l="피처별",
    fm_l="+ FM  ½((Σe)²−Σe²) = [B,32]",
    tower_note="각 타워:  Dense → ReLU → Dropout 0.3        σ = 시그모이드  ·  lin = 항등",
    win_name="Win 타워", win_role="성향점수  P(win|x)",
    ctr_name="CTR 타워", ctr_role="비편향  P(click|win,x)",
    imp_name="Imputation 타워", imp_role="CTR 오차 제어변량(control variate)",
    lin="lin",
    dual_h="Win 타워 이중 활용",
    dual_a=["(a) 학습 — DR 성향점수 ŵ = win / P̂(win)", "      가 CTR 손실을 디바이어싱"],
    dual_b=["(b) 서빙 — 1차가격 비드 셰이딩용", "      낙찰률 모델 (AUC ≈ 0.91)"],
    outputs="모델 출력  p̂_win, p̂_ctr, δ̂",
    obj_h="학습 목적함수 — 이중 강건 클릭 손실 + ESMM 전체공간 제약",
    w_h="성향 가중치", w_sub="자기정규화 · p̂_win stop-grad",
    dr_h2="이중 강건 잔차", dr_sub="성향 또는 대체 모델 중 하나만 맞아도 비편향",
    esmm_h2="ESMM 제약", esmm_sub="전체공간(entire-space)",
    dom_all="전체 입찰", dom_won="낙찰만", dom_lost="패찰만",
    leg_dom="표본 영역", leg_fwd="순전파", leg_sg="stop-grad / 가중치 입력",
)

ARC_EN = dict(
    title="The honest research arc — falsification first",
    n1_h=["Headline:", "neural beats LR", "on AUC"], n1=["a tempting", "first result"],
    n2_h=["Root-cause", "audit"], n2=["it was a", "split artifact", "(self-caught)"],
    n3_h=["Fair split", "retrain"], n3=["winners-AUC", "0.658 > 0.632", "> 0.554"],
    n4_h=["Calibration", "solved"], n4=["IEB 0.597→0", "residual", "0.226→0.0006"],
    n5_h=["Decision", "test"], n5=["robust vs LR;", "NOT vs LGB", "(I²=0.82)"],
    retracted="✗ retracted",
    verdict="Honest verdict: debiasing helps bidding vs a linear model — not (robustly) vs a strong GBM.",
)
ARC_KO = dict(
    title="정직한 연구 아크 — 반증(falsification) 우선",
    n1_h=["헤드라인:", "neural이 AUC에서", "LR을 이긴다"], n1=["그럴듯한", "첫 결과"],
    n2_h=["근본원인", "감사"], n2=["사실은", "split 아티팩트", "(자가 적발)"],
    n3_h=["공정 split", "재학습"], n3=["winners-AUC", "0.658 > 0.632", "> 0.554"],
    n4_h=["캘리브레이션", "해결"], n4=["IEB 0.597→0", "잔차", "0.226→0.0006"],
    n5_h=["의사결정", "검정"], n5=["LR 대비 강건;", "LGB 대비 아님", "(I²=0.82)"],
    retracted="✗ 철회됨",
    verdict="정직한 결론: 디바이어싱은 선형 모델 대비 입찰에 도움 — 강한 GBM 대비로는 (강건하게) 아니다.",
)

DIAGRAMS = [
    ("funnel_selection_bias", funnel, FUNNEL_EN, FUNNEL_KO),
    ("escm2wc_architecture", architecture, ARCH_EN, ARCH_KO),
    ("falsification_arc", arc, ARC_EN, ARC_KO),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn, en, ko in DIAGRAMS:
        (OUT / f"{name}.svg").write_text(fn(en), encoding="utf-8")
        (OUT / f"{name}.ko.svg").write_text(fn(ko), encoding="utf-8")
        print(f"  wrote assets/{name}.svg + .ko.svg")


if __name__ == "__main__":
    main()
