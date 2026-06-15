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


def arrow(x1, y1, x2, y2, color=INK, width=2.2, head=11, halfw=5.5):
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    p2 = (bx - halfw * math.sin(ang), by + halfw * math.cos(ang))
    p3 = (bx + halfw * math.sin(ang), by - halfw * math.cos(ang))
    return (f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round"/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}" '
            f'fill="{color}"/>')


def svg(w, h, body) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" font-family="{FONT}">'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{GRIDBG}"/>{body}</svg>')


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
    W, H = 940, 560
    b = []
    b.append(text(W / 2, 38, L["title"], 21, INK, "bold"))
    b.append(text(W / 2, 62, L["subtitle"], 13, MUTE))

    # input + shared trunk (left)
    b.append(rect(40, 250, 150, 70, fill=CARD, stroke=INK, sw=1.6))
    b.append(lines(115, 282, L["input"], 13, INK, "bold", lh=17))
    b.append(arrow(190, 285, 232, 285, INK))
    b.append(rect(232, 235, 150, 100, fill="#EEF2FF", stroke=NEURAL, sw=2))
    b.append(lines(307, 272, L["trunk"], 13, NEURAL, "bold", lh=17))

    # three towers (middle column)
    towers = [
        (435, 120, LGB, L["win_tower"], L["win_out"]),
        (435, 250, NEURAL, L["ctr_tower"], L["ctr_out"]),
        (435, 380, "#9333EA", L["imp_tower"], L["imp_out"]),
    ]
    for (x, y, col, name, out) in towers:
        b.append(rect(x, y, 200, 95, fill=CARD, stroke=col, sw=2.2))
        b.append(text(x + 100, y + 32, name, 14.5, col, "bold"))
        b.append(lines(x + 100, y + 54, out, 11.5, MUTE, lh=15))
        b.append(arrow(382, 285, x, y + 47, col, 1.8))

    # DR weighting: win tower propensity -> ctr loss
    b.append(arrow(535, 215, 535, 248, MUTE, 1.8))
    b.append(text(545, 236, L["dr_w"], 10, MUTE, "italic", "start", italic=True))

    # ESMM joint constraint bracket linking win + ctr
    b.append(rect(435, 478, 200, 56, fill="#F0FDF4", stroke=POS, sw=1.6))
    b.append(text(535, 500, L["esmm_h"], 12.5, POS, "bold"))
    b.append(text(535, 520, L["esmm_eq"], 12, MUTE))

    # dual-purpose Win Tower (right): two outputs
    b.append(rect(700, 95, 210, 70, fill="#ECFEFF", stroke=LGB, sw=1.8))
    b.append(text(805, 122, L["use_a_h"], 12.5, LGB, "bold"))
    b.append(text(805, 142, L["use_a"], 11, MUTE))
    b.append(arrow(635, 150, 700, 130, LGB, 1.8))

    b.append(rect(700, 185, 210, 70, fill="#ECFEFF", stroke=LGB, sw=1.8))
    b.append(text(805, 212, L["use_b_h"], 12.5, LGB, "bold"))
    b.append(text(805, 232, L["use_b"], 11, MUTE))
    b.append(arrow(635, 165, 700, 215, LGB, 1.8))
    b.append(text(805, 78, L["dual"], 12, INK, "bold"))
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
    title="ESCM²-WC (DR): 3-tower debiasing with a dual-purpose Win Tower",
    subtitle="Shared trunk → Win / CTR / Imputation towers; doubly-robust correction + ESMM constraint",
    input=["Features x", "(30 fields)"],
    trunk=["Shared", "embedding + MLP", "trunk"],
    win_tower="Win Tower", win_out=["P(win | x)", "propensity"],
    ctr_tower="CTR Tower", ctr_out=["P(click | win, x)", "debiased pCTR"],
    imp_tower="Imputation Tower", imp_out=["δ̂ = imputed", "CTR error"],
    dr_w="DR weight: w = win / P(win)",
    esmm_h="ESMM joint constraint",
    esmm_eq="P(click,win) = P(win) × P(click|win)",
    dual="Dual-purpose Win Tower",
    use_a_h="(a) Debiasing", use_a="propensity for DR / IPW",
    use_b_h="(b) Bid shading", use_b="win-rate model (AUC ≈ 0.91)",
)
ARCH_KO = dict(
    title="ESCM²-WC (DR): Win Tower를 이중 활용하는 3-tower 디바이어싱",
    subtitle="공유 트렁크 → Win / CTR / Imputation 타워; 이중 강건(DR) 보정 + ESMM 제약",
    input=["피처 x", "(30개)"],
    trunk=["공유", "임베딩 + MLP", "트렁크"],
    win_tower="Win 타워", win_out=["P(win | x)", "성향점수(propensity)"],
    ctr_tower="CTR 타워", ctr_out=["P(click | win, x)", "비편향 pCTR"],
    imp_tower="Imputation 타워", imp_out=["δ̂ = 대체(impute)된", "CTR 오차"],
    dr_w="DR: w = win / P(win)",
    esmm_h="ESMM 결합 제약",
    esmm_eq="P(click,win) = P(win) × P(click|win)",
    dual="Win 타워 이중 활용",
    use_a_h="(a) 디바이어싱", use_a="DR / IPW 성향점수",
    use_b_h="(b) 비드 셰이딩", use_b="낙찰률 모델 (AUC ≈ 0.91)",
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
