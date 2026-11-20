#!/usr/bin/env python3
"""
Build report.pdf for HPML HW3 from the outputs of hw3_report_runs.ipynb (lab-manual layout).

Usage:
    pip install reportlab matplotlib
    python build_report.py --in ../report_out --out report.pdf

Every number is parsed from the program logs in report_out/logs, and every screenshot is
report_out/shots/<log>.png, an image of that same log, so text, tables and screenshots agree.
"""
import argparse
import re
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (CondPageBreak, HRFlowable, Image, KeepTogether, ListFlowable,
                                ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

# ---------------------------------------------------------------- edit me
NAME = "Hardik Saurabh Gupta"
UNI = "hg2770"
COURSE = "COMS 6998 High Performance Machine Learning, Fall 2026"
AI_DISCLOSURE = (
    "I used Claude Code (Anthropic's AI coding assistant) as a helper, not for the entire assignment: "
    "for parts of the code where I was stuck, and to help write the documentation and this report "
    "from my own measured outputs, which I reviewed and verified."
)
# ------------------------------------------------------------------------

RESULT_RE = re.compile(r"^RESULT,(.+)$")
FMT_RE = re.compile(r"^(-?\d+\.\d{3}),(\d+\.\d{3})$")  # C1..C3 "checksum,time" line


# ------------------------------------------------------------------ data
def parse_results(text):
    rows = []
    for line in text.splitlines():
        m = RESULT_RE.match(line.strip())
        if not m:
            continue
        d = {}
        for kv in m.group(1).split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                for cast in (int, float):
                    try:
                        d[k] = cast(v)
                        break
                    except ValueError:
                        d[k] = v
        rows.append(d)
    return rows


class Run:
    def __init__(self, root):
        self.root = Path(root)
        self.logs = {p.stem: p.read_text() for p in (self.root / "logs").glob("*.txt")}
        self.res = {k: parse_results(v) for k, v in self.logs.items()}

    def find(self, log, **kw):
        for r in self.res.get(log, []):
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    def t(self, log, **kw):
        r = self.find(log, **kw)
        return r["time_ms"] if r else None


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ratio(a, b):
    return (a / b) if (a and b) else None


def fx(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}x"


def code(s):
    return f"<font face='Courier'>{s}</font>"


# ------------------------------------------------------------------ styles
NAVY = colors.HexColor("#1f3b63")
RULE = colors.HexColor("#9fb0c6")
ss = getSampleStyleSheet()
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=12.4,
                      spaceAfter=3)
H1 = ParagraphStyle("H1", parent=BODY, fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                    textColor=NAVY, spaceBefore=10, spaceAfter=2)
H2 = ParagraphStyle("H2", parent=BODY, fontName="Helvetica-Bold", fontSize=10.6, leading=14,
                    textColor=NAVY, spaceBefore=8, spaceAfter=3)
LABEL = ParagraphStyle("L", parent=BODY, fontName="Helvetica-Bold", fontSize=9.4, spaceBefore=3, spaceAfter=1)
CAP = ParagraphStyle("C", parent=BODY, fontSize=8, leading=10, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=8)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
TITLE = ParagraphStyle("T", parent=BODY, fontName="Helvetica-Bold", fontSize=17, leading=21,
                       alignment=TA_CENTER, textColor=NAVY)
SUB = ParagraphStyle("Su", parent=BODY, alignment=TA_CENTER, fontSize=10, textColor=colors.HexColor("#444444"))


def P(text, st=BODY):
    return Paragraph(text, st)


class Counter:
    sec = 0
    sub = 0
    fig = 0
    tab = 0


def section(title):
    Counter.sec += 1
    Counter.sub = 0
    return [CondPageBreak(1.6 * inch), P(f"{Counter.sec}&nbsp;&nbsp;{title}", H1),
            HRFlowable(width="100%", thickness=0.8, color=NAVY, spaceBefore=1, spaceAfter=5)]


def subsection(title):
    Counter.sub += 1
    return [CondPageBreak(2.4 * inch), P(f"{Counter.sec}.{Counter.sub}&nbsp;&nbsp;{title}", H2)]


def bullets(items, label=None):
    out = [P(label, LABEL)] if label else []
    out.append(ListFlowable([ListItem(P(i), leftIndent=12) for i in items],
                            bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8))
    out.append(Spacer(1, 3))
    return out


def table(header, rows, caption=None, widths=None):
    data = [[P(f"<b>{h}</b>", CELL) for h in header]] + [[P(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf4")),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, NAVY),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    out = []
    if caption:
        Counter.tab += 1
        out.append(P(f"<b>Table {Counter.tab}.</b> {caption}", ParagraphStyle("tc", parent=CAP, alignment=0,
                                                                                spaceAfter=2)))
    return [KeepTogether(out + [t]), Spacer(1, 7)]


# ------------------------------------------------------------------ images
WRAP = 108  # wrap log lines longer than this so screenshots keep a readable, uniform font size


def log_image(run, name):
    """The Colab-rendered screenshot, or (if the log has very long lines, e.g. a long shell
    command) the same log re-rendered with those lines wrapped. Content is unchanged."""
    shot = run.root / "shots" / f"{name}.png"
    log = run.root / "logs" / f"{name}.txt"
    if not log.exists():
        return shot if shot.exists() else None
    txt = log.read_text().rstrip("\n").expandtabs(4)
    if shot.exists() and max((len(l) for l in txt.splitlines()), default=0) <= WRAP:
        return shot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = []
    for l in txt.splitlines():
        lines += textwrap.wrap(l, WRAP, subsequent_indent="    ", break_on_hyphens=False) or [""]
    fs = 8.5
    cw, lh = fs * 0.62 / 72, fs * 1.25 / 72
    W, H = max(60, max(len(l) for l in lines)) * cw + 0.4, len(lines) * lh + 0.4
    fig = plt.figure(figsize=(W, H), dpi=160)
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.2 / W, 1 - 0.2 / H, "\n".join(lines), family="DejaVu Sans Mono", fontsize=fs,
             color="#e8e8e8", va="top", ha="left", linespacing=1.25)
    d = run.root / "shots_wrapped"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.png"
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    return p


def figure(path, caption, max_w=6.9 * inch, max_h=8.4 * inch, natural=False):
    if path is None or not Path(path).exists():
        return [P(f"<font color='#b00020'>[missing: {escape(caption)}]</font>")]
    iw, ih = ImageReader(str(path)).getSize()
    s = min(max_w / iw, max_h / ih, 72 / 160 if natural else 10.0)
    Counter.fig += 1
    return [KeepTogether([Image(str(path), width=iw * s, height=ih * s),
                          P(f"<b>Figure {Counter.fig}.</b> {caption}", CAP)])]


def shot(run, name, caption):
    return figure(log_image(run, name), caption, natural=True)


# ------------------------------------------------------------------ header
def header_block(run):
    s = run.logs.get("00_setup", "")
    gpu = re.search(r"^name, driver_version.*\n([^\n]+)", s, re.M)
    gp = [p.strip() for p in gpu.group(1).split(",")] if gpu else []
    rows = [["Name", NAME, "UNI", UNI],
            ["Course", COURSE, "Assignment", "Homework 3: CUDA Kernels"],
            ["GPU", f"{gp[0]} ({gp[2]})" if len(gp) > 2 else "see Fig. 1", "Platform", "Google Colab"]]
    t = Table([[P(f"<b>{a}</b>", CELL), P(b, CELL), P(f"<b>{c}</b>", CELL), P(d, CELL)] for a, b, c, d in rows],
              colWidths=[0.8 * inch, 3.35 * inch, 0.95 * inch, 1.8 * inch])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.0, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e6ecf4")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#e6ecf4")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    disc = Table([[P(f"<b>AI-use disclosure.</b> {escape(AI_DISCLOSURE)}", CELL)]], colWidths=[6.9 * inch])
    disc.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE),
                              ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfbf3")),
                              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return [P("Lab Report: Homework 3, CUDA Kernels", TITLE), P(COURSE, SUB), Spacer(1, 8), t,
            Spacer(1, 6), disc, Spacer(1, 4)]


# ------------------------------------------------------------------ 1 setup
def setup_section(run):
    s = run.logs.get("00_setup", "")
    gpu = re.search(r"^name, driver_version.*\n([^\n]+)", s, re.M)
    gp = [p.strip() for p in gpu.group(1).split(",")] if gpu else ["?"] * 4
    nvcc = re.search(r"release ([\d.]+)", s)
    cpu = re.search(r"Model name:\s*(.+)", s)
    ncpu = re.search(r"^CPU\(s\):\s*(\d+)", s, re.M)
    os_ = re.search(r'PRETTY_NAME="([^"]+)"', s)
    kern = re.search(r"^(\d+\.\d+\.\d+\S*)$", s, re.M)
    out = section("Experimental Setup")
    out += table(["Component", "Value"], [
        ["GPU", f"{gp[0]}, {gp[2]}, compute capability {gp[3]}" if len(gp) > 3 else "see Fig. 1"],
        ["NVIDIA driver / CUDA", f"{gp[1] if len(gp) > 1 else '?'} / nvcc {nvcc.group(1) if nvcc else '?'}"],
        ["CPU", (cpu.group(1).strip() if cpu else "?") + (f", {ncpu.group(1)} vCPUs" if ncpu else "")],
        ["OS", (os_.group(1) if os_ else "?") + (f", Linux {kern.group(1)}" if kern else "")],
        ["Build", "make ARCH=sm_75 (nvcc -O3), unmodified starter Makefiles"],
    ], "Hardware and software used for all measurements.", [1.7 * inch, 5.2 * inch])
    out += bullets([
        "<b>Kernel-only timing</b>, in ms with 3 decimals; allocation, data generation, host-device copies and "
        "cleanup are excluded.",
        f"Part A/B: CUDA events ({code('GpuTimer')}) around the launches. Part C: CUDA events around kernels / "
        "cuBLAS calls. CPU baseline: " + code("std::chrono") + " around the add loop only. C4: "
        + code("triton.testing.do_bench") + ".",
        "<b>Single run</b> per data point (no averaging). Only warm-up: one untimed C3 pass (both GEMMs + softmax).",
        "Part B Q3 (unified memory) numbers are <b>first-touch</b>: page migration happens inside the timed region.",
        "All outputs below come from one Colab session.",
    ], "Methodology")
    out += shot(run, "00_setup", "GPU, driver, CUDA, CPU and OS")
    return out


# ------------------------------------------------------------------ 2 part A
def part_a(run, bdata):
    out = section("Part A: CUDA Matrix Operations")
    vpts = (500, 1000, 2000)
    v0 = {v: run.find("A_Q1_vecadd00", values_per_thread=v) for v in vpts}
    v1 = {v: run.find("A_Q2_vecadd01", values_per_thread=v) for v in vpts}
    bw = lambda r: 3 * r["N"] * 4 / (r["time_ms"] / 1e3) / 1e9 if r else None

    # Q1
    out += subsection("Q1: vecadd00 (non-coalesced baseline)")
    out += table(["Values / thread", "N (floats)", "Time (ms)", "Bandwidth", "Result"],
                 [[v, v0[v]["N"] if v0[v] else "n/a", f3(v0[v] and v0[v]["time_ms"]),
                   f"{bw(v0[v]):.1f} GB/s" if v0[v] else "n/a",
                   "PASSED" if v0[v] and v0[v].get("correct") == 1 else "n/a"] for v in vpts],
                 "vecadd00 kernel time.")
    out += bullets([
        f"Thread {code('tid')} handles a contiguous run starting at {code('tid*valuesPerThread')}.",
        "Neighbouring threads in a warp are " + code("valuesPerThread*4") + " bytes apart, so each warp access "
        "splits into 32 separate memory transactions.",
        "Time scales linearly with the work (2x data gives about 2x time).",
        f"Effective bandwidth is only ~{bw(v0[2000]):.0f} GB/s, a few percent of the T4's ~320 GB/s peak."
        if v0[2000] else "Effective bandwidth is far below the T4 peak.",
    ], "Observations")
    out += shot(run, "A_Q1_vecadd00", "vecadd00 at 500, 1000, 2000 values per thread")

    # Q2
    out += subsection("Q2: vecadd01 (coalesced)")
    sp = {v: ratio(v0[v] and v0[v]["time_ms"], v1[v] and v1[v]["time_ms"]) for v in vpts}
    out += table(["Values / thread", "vecadd00 (ms)", "vecadd01 (ms)", "Speedup", "vecadd01 bandwidth"],
                 [[v, f3(v0[v] and v0[v]["time_ms"]), f3(v1[v] and v1[v]["time_ms"]), fx(sp[v], 1),
                   f"{bw(v1[v]):.0f} GB/s" if v1[v] else "n/a"] for v in vpts],
                 "Coalesced vs non-coalesced vector add (same work).")
    lo, hi = min(x for x in sp.values() if x), max(x for x in sp.values() if x)
    out += bullets([
        f"New mapping: at loop step {code('j')}, thread {code('tid')} handles element "
        f"{code('j*totalThreads + tid')}.",
        "Same number of elements per thread; consecutive threads now touch consecutive floats.",
        "Each warp access becomes one 128-byte transaction per array.",
        f"Speedup: <b>{lo:.1f}x&ndash;{hi:.1f}x</b>; vecadd01 now runs close to DRAM bandwidth (bandwidth-bound).",
    ], "Implementation and results")
    out += shot(run, "A_Q2_vecadd01", "vecadd01 at 500, 1000, 2000 values per thread")

    # Q3
    sizes = (256, 512, 1024)
    m0 = {s: run.t("A_Q3_matmult00", size=s) for s in sizes}
    m1 = {s: run.t("A_Q4_matmult01", size=s) for s in sizes}
    gf = lambda s, t: f"{2 * s ** 3 / (t / 1e3) / 1e9:.0f}" if t else "n/a"
    out += subsection("Q3: matmult00 (FOOTPRINT_SIZE = 16)")
    out += table(["Matrix size", "Argument", "Time (ms)", "GFLOP/s"],
                 [[f"{s} x {s}", s // 16, f3(m0[s]), gf(s, m0[s])] for s in sizes],
                 "Baseline tiled matmul, one output per thread.")
    out += bullets([
        "16x16 thread block; each step loads a 16x16 tile of A and B into shared memory.",
        "Each thread computes one element of C.",
        f"64x more work (256 &rarr; 1024) costs only {m0[1024] / m0[256]:.1f}x more time."
        if m0[256] and m0[1024] else "Throughput rises with matrix size.",
        "At 256 there are only 256 blocks and the kernel is very short, so latency and launch overhead "
        "dominate. Larger matrices fill all 40 SMs.",
    ], "Observations")
    out += shot(run, "A_Q3_matmult00", "matmult00 at 256, 512, 1024")

    # Q4
    out += subsection("Q4: matmult01 (FOOTPRINT_SIZE = 32, 2x2 per thread)")
    spm = {s: ratio(m0[s], m1[s]) for s in sizes}
    out += table(["Matrix size", "matmult00 (ms)", "matmult01 (ms)", "Speedup", "matmult01 GFLOP/s"],
                 [[f"{s} x {s}", f3(m0[s]), f3(m1[s]), fx(spm[s]), gf(s, m1[s])] for s in sizes],
                 "Register-blocked kernel vs baseline.")
    out += bullets([
        "Thread block stays 16x16; each block now computes a 32x32 tile of C.",
        "Each thread computes 4 outputs at rows {ty, ty+16} and cols {tx, tx+16}.",
        "<b>Coalesced:</b> loads and stores use stride 16, so consecutive " + code("threadIdx.x")
        + " values hit consecutive addresses. Shared-memory accesses are conflict-free.",
        "<b>Register blocking:</b> 4 FMAs per 4 shared loads (1:1), vs 1:2 in matmult00.",
        "<b>Reuse:</b> each global value is used 32 times instead of 16, so global traffic per FLOP halves.",
        "<b>Unrolled:</b> " + code("#pragma unroll") + " on the 32-step inner loop and the 2x2 load loops.",
    ], "Implementation")
    res = [f"Speedup at 1024: <b>{fx(spm[1024])}</b>."]
    slow = [s for s in sizes if spm[s] and spm[s] < 1]
    if slow:
        res.append(f"Slower at {', '.join(map(str, slow))}: a 32x32 tile leaves too few blocks "
                   f"({(slow[0] // 32) ** 2}) to fill 40 SMs.")
    else:
        res.append(f"Faster at every size ({', '.join(fx(spm[s]) for s in sizes)}).")
    res.append("The gain comes from more data reuse and more FMAs per memory access.")
    out += bullets(res, "Results")
    out += shot(run, "A_Q4_matmult01", "matmult01 at 256, 512, 1024")

    # Q5
    Ks, cpu, q2, q3 = bdata
    out += subsection("Q5: Rules of thumb")
    va, vb = run.t("A_Q1_vecadd00", values_per_thread=2000), run.t("A_Q2_vecadd01", values_per_thread=2000)
    rows = [
        ["Coalesce global memory accesses", f"vecadd 2000 v/t: {f3(va)} &rarr; {f3(vb)} ms ({fx(ratio(va, vb), 1)})"],
        ["Few large launches, not many tiny ones",
         f"K=10: scenario 1 {f3(q2.get((1, 10)))} ms vs scenario 3 {f3(q2.get((3, 10)))} ms "
         f"({fx(ratio(q2.get((1, 10)), q2.get((3, 10))), 0)})"],
        ["Give each launch enough threads to fill the GPU",
         f"K=100: scenario 2 {f3(q2.get((2, 100)))} ms vs scenario 3 {f3(q2.get((3, 100)))} ms "
         f"({fx(ratio(q2.get((2, 100)), q2.get((3, 100))), 0)})"],
        ["Reuse data from shared memory and registers",
         f"matmult 1024: {f3(m0[1024])} &rarr; {f3(m1[1024])} ms ({fx(spm[1024])})"],
        ["Keep data resident on the GPU (avoid paging)",
         f"K=100 single launch: {f3(q2.get((3, 100)))} ms explicit vs {f3(q3.get((3, 100)))} ms unified"],
    ]
    out += table(["Rule", "Supporting measurement"], rows, "Rules of thumb with evidence from Parts A and B.",
                 [2.6 * inch, 4.3 * inch])
    if slow:
        s0 = slow[0]
        nh = [f"FOOTPRINT_SIZE 16 &rarr; 32 at {s0}x{s0}: {f3(m0[s0])} &rarr; {f3(m1[s0])} ms ({fx(spm[s0])}).",
              "Cause: too few blocks to occupy all SMs at that size."]
    else:
        nh = [f"Switching the single-launch vector add to {code('cudaMallocManaged')}: "
              f"{f3(q2.get((3, 100)))} &rarr; {f3(q3.get((3, 100)))} ms at K = 100 "
              f"({fx(ratio(q3.get((3, 100)), q2.get((3, 100))), 0)} slower).",
              "Cause: first-touch page migration happens inside the kernel."]
    out += bullets(nh, "A change that did not help")
    return out


# ------------------------------------------------------------------ 3 part B
def part_b_data(run):
    Ks = (1, 5, 10, 50, 100)
    cpu = {K: run.t("B_Q1_cpu", K=K) for K in Ks}
    q2 = {(s, K): run.t("B_Q2_malloc", scenario=s, K=K) for s in (1, 2, 3) for K in Ks}
    q3 = {(s, K): run.t("B_Q3_unified", scenario=s, K=K) for s in (1, 2, 3) for K in Ks}
    return Ks, cpu, q2, q3


def part_b(run, bdata):
    Ks, cpu, q2, q3 = bdata
    out = section("Part B: CUDA Unified Memory")

    out += subsection("Q1: CPU baseline")
    out += table(["K (million)"] + [str(K) for K in Ks], [["CPU time (ms)"] + [f3(cpu[K]) for K in Ks]],
                 "CPU vector add, z[i] = x[i] + y[i].")
    out += bullets(["Plain C++ loop; memory freed with " + code("free()") + ".",
                    "Linear in K, about 3 ms per million elements."])
    out += shot(run, "B_Q1_cpu", "q1_cpu for K = 1, 5, 10, 50, 100")

    def scen_table(d, cap):
        rows = []
        for s, lab in ((1, "S1: n launches &lt;&lt;&lt;1,1&gt;&gt;&gt;"),
                       (2, "S2: n/256 launches &lt;&lt;&lt;1,256&gt;&gt;&gt;"),
                       (3, "S3: 1 launch &lt;&lt;&lt;ceil(n/256),256&gt;&gt;&gt;")):
            rows.append([lab] + [("capped" if (s == 1 and K > 10) else f3(d[(s, K)])) for K in Ks])
        return table(["Scenario \\ K (million)"] + [str(K) for K in Ks], rows, cap,
                     [2.4 * inch] + [0.9 * inch] * len(Ks))

    out += subsection("Q2: Explicit cudaMalloc / cudaMemcpy")
    out += scen_table(q2, "Kernel time (ms), explicit device memory.")
    out += bullets([
        "Data copied with " + code("cudaMemcpy") + " <i>before</i> the timer starts.",
        "One timer wraps all launches of a scenario, so S1's launch overhead is included.",
        "S2 adds one smaller launch for the remainder (n mod 256); the kernel guards i &lt; n.",
        "S1 is capped at K &le; 10.",
    ])
    out += shot(run, "B_Q2_malloc", "q2_malloc, all K, all three scenarios")

    out += subsection("Q3: Unified memory (cudaMallocManaged)")
    out += scen_table(q3, "Kernel time (ms), unified memory, first-touch.")
    out += bullets([
        code("cudaMallocManaged") + " for x, y, z; initialised on the host; freed with " + code("cudaFree") + ".",
        "No explicit copies: pages migrate to the GPU on first access, <i>inside</i> the timed region.",
        "So Q2 and Q3 are not a head-to-head comparison: Q2 excludes the transfer, Q3 includes it.",
    ])
    out += shot(run, "B_Q3_unified", "q3_unified, all K, all three scenarios")

    out += subsection("Q4: Predictions, charts and analysis")
    out += bullets([
        "<b>S1:</b> pure launch overhead, linear in K, slowest by far (seconds at K = 1).",
        "<b>S2:</b> 256x fewer launches, so ~2 orders of magnitude below S1; still above the CPU.",
        "<b>S3 (explicit):</b> fastest; bandwidth-bound, linear in K, far below the CPU.",
        "<b>Unified memory:</b> S1 and S2 roughly unchanged; S3 much slower (page migration), near the CPU.",
        "<b>Order:</b> S1 &gt; S2 &gt; CPU &gt; S3.",
    ], "Predictions")
    for f, cap in (("partb_without_unified.png", "Q2, explicit copies: time vs K, log-log."),
                   ("partb_with_unified.png", "Q3, unified memory: time vs K, log-log.")):
        out += figure(run.root / "plots" / f, cap, max_w=4.8 * inch, max_h=3.1 * inch)
    out += [P("Charts generated by the provided " + code("q4_plot.py") + ", which re-runs the binaries in the same "
              "session, so values differ slightly from the screenshots (run-to-run variation).", CAP)]

    obs = []
    s1, s10 = q2.get((1, 1)), q2.get((1, 10))
    if s1 and s10:
        obs.append(f"<b>S1:</b> {f3(s1)} &rarr; {f3(s10)} ms for K = 1 &rarr; 10 (linear), about "
                   f"{s10 * 1e3 / 1e7:.2f} &micro;s per launch. Unified memory barely changes it "
                   f"({f3(q3.get((1, 10)))} ms).")
    if q2.get((2, 100)) and cpu.get(100):
        obs.append(f"<b>S2:</b> above the CPU at every K ({f3(q2[(2, 100)])} vs {f3(cpu[100])} ms at K = 100), "
                   f"about {q2[(2, 100)] * 1e3 / (1e8 / 256):.2f} &micro;s per launch, so also launch-bound.")
    if q2.get((3, 100)) and cpu.get(100):
        obs.append(f"<b>S3 explicit:</b> {f3(q2[(3, 1)])} &rarr; {f3(q2[(3, 100)])} ms, "
                   f"{cpu[100] / q2[(3, 100)]:.0f}x faster than the CPU at K = 100 "
                   f"(~{1.2e9 / (q2[(3, 100)] / 1e3) / 1e9:.0f} GB/s, bandwidth-bound).")
    if q3.get((3, 100)) and q2.get((3, 100)):
        obs.append(f"<b>S3 unified:</b> {q3[(3, 100)] / q2[(3, 100)]:.0f}x slower than explicit at K = 100 "
                   f"({f3(q3[(3, 100)])} ms), close to the CPU ({f3(cpu[100])} ms) because of page faults. "
                   "Prefetching (" + code("cudaMemPrefetchAsync") + ") would remove most of this.")
    bad = [K for K in Ks if (lambda seq: any(a <= b for a, b in zip(seq, seq[1:])))(
        [t for t in (q2.get((1, K)), q2.get((2, K)), cpu.get(K), q2.get((3, K))) if t is not None])]
    obs.append("<b>vs prediction:</b> ordering " + ("held at every K." if not bad else
                                                    f"broke at K = {', '.join(map(str, bad))}."))
    g3, g3u = ratio(q2.get((3, 10)), q2.get((3, 1))), ratio(q3.get((3, 10)), q3.get((3, 1)))
    if g3 and g3u:
        obs.append(f"<b>Difference:</b> for 10x data S3 grew {g3:.1f}x (explicit) and {g3u:.1f}x (unified). "
                   + ("Under 10x means a fixed setup / page-fault cost matters at small K, so the curve is flatter "
                      "at the left than predicted." if min(g3, g3u) < 7 else
                      "Close to 10x, i.e. bandwidth-bound already at K = 1."))
    out += bullets(obs, "Observed vs predicted")
    return out


# ------------------------------------------------------------------ 4 part C
def part_c(run):
    out = section("Part C: Attention in CUDA")
    fmt = {}
    for log in ("C1_naive", "C2_tiled", "C3_cublas"):
        for line in run.logs.get(log, "").splitlines():
            m = FMT_RE.match(line.strip())
            if m:
                fmt[log] = f"{m.group(1)},{m.group(2)}"
    c4 = run.logs.get("C4_triton", "")
    c4_fmt = next((l.strip() for l in reversed(c4.splitlines()) if re.fullmatch(r"\d+\.\d{3}", l.strip())), "n/a")
    lines = [fmt.get(k, "n/a") for k in ("C1_naive", "C2_tiled", "C3_cublas")] + [c4_fmt]
    out += [P("<b>Required program output</b> (C1, C2, C3: checksum,time; C4: time), N = 2048, d = 64:")]
    out += [P(code("<br/>".join(escape(l) for l in lines)),
              ParagraphStyle("pre", parent=BODY, leftIndent=14, backColor=colors.HexColor("#f2f4f7"),
                             borderPadding=5, spaceBefore=3, spaceAfter=10))]
    c = {k: run.find(k, N=2048) for k in ("C1_naive", "C2_tiled", "C3_cublas")}
    rows = [[n, c[k]["checksum"] if c[k] else "n/a", f3(c[k] and c[k]["time_ms"])]
            for k, n in (("C1_naive", "C1 naive"), ("C2_tiled", "C2 tiled"), ("C3_cublas", "C3 cuBLAS"))]
    out += table(["Version", "Checksum", "Kernel time (ms)"], rows, "Attention, fp64, N = 2048, d = 64.")
    out += bullets(["All three checksums are identical.",
                    "Outputs also match an independent reference element by element at N, d = (8, 4), (129, 4), "
                    "(257, 8)."])

    # C1
    out += subsection("C1: Naive attention")
    out += bullets([
        "<b>Kernel 1, scores:</b> 16x16 blocks, one thread per S[i,j]; j on threadIdx.x so writes are coalesced.",
        "<b>Kernel 2, safe softmax:</b> one 256-thread block per row; shared-memory reduction for the max, "
        "then exp(S &minus; max), a second reduction for the sum, then normalise in place.",
        "<b>Kernel 3, output:</b> one thread per O[i,t] = &Sigma;<sub>j</sub> P[i,j]&middot;V[j,t].",
        "The full N x N matrix S (32 MiB at N = 2048) is stored in global memory.",
    ], "Design")
    out += shot(run, "C1_naive", "c1_naive, N = 2048, d = 64")

    # C2
    out += subsection("C2: Tiled (FlashAttention-style) attention")
    out += bullets([
        "One 128-thread block per query row; the query row is held in shared memory.",
        "K and V are processed in tiles of 16 rows, loaded coalesced into shared memory (padded to d+1).",
        "Per tile: 16 scores go to shared memory, then the running max m, denominator l and accumulator "
        "acc[d] (in registers) are updated.",
        code("__syncthreads()") + " separates every shared-memory write from the reads that depend on it.",
        "Output O[i,:] = acc / l. The N x N scores are never written to global memory.",
    ], "Design")
    out += bullets([
        "acc and l are sums of exp(s &minus; m) terms, all relative to the <i>current</i> max m.",
        "When a new tile raises the max to m', new terms use exp(s &minus; m').",
        "Old sums must be multiplied by &alpha; = exp(m &minus; m'): l &larr; &alpha;l + &Sigma;exp(s &minus; m'), "
        "acc &larr; &alpha;acc + &Sigma;exp(s &minus; m')V.",
        "<b>Without rescaling:</b> earlier keys are over-weighted by exp(m' &minus; m) &gt; 1, so the result is "
        "not a correctly normalised softmax average (wrong output, biased to early keys).",
    ], "Why the accumulator is rescaled")
    rows = []
    for N in (1024, 2048, 4096):
        t = [run.t(k, N=N, d=64) or run.t("C_sizes", program=p, N=N, d=64)
             for k, p in (("C1_naive", "c1_naive"), ("C2_tiled", "c2_tiled"), ("C3_cublas", "c3_cublas"))]
        rows.append([N] + [f3(x) for x in t] + [fx(ratio(t[1], t[0]), 1)])
    out += table(["N", "C1 naive (ms)", "C2 tiled (ms)", "C3 cuBLAS (ms)", "C2 / C1"], rows,
                 "Kernel time vs sequence length (d = 64).")
    t1, t2 = run.t("C1_naive", N=2048, d=64), run.t("C2_tiled", N=2048, d=64)
    perf = ["C2 avoids the N x N HBM traffic: C1 makes several passes over a 32&ndash;128 MiB S matrix; C2 "
            "streams only K and V (2 MiB at N = 2048, fits in the 4 MiB L2)."]
    if t1 and t2 and t2 > t1:
        perf += [f"My C2 is still slower than C1 ({f3(t2)} vs {f3(t1)} ms), for three reasons:",
                 "<b>Redundant work:</b> every thread recomputes the tile max and all 16 exps: 128x redundant fp64 "
                 "exp() (fp64 runs at 1/32 rate on a T4).",
                 "<b>Low utilisation:</b> only 16 of 128 threads compute scores; only 64 accumulate (d = 64).",
                 "<b>No K/V reuse:</b> one row per block, so a K/V tile is reused by just one query row.",
                 "<b>Fix:</b> give each block Br query rows (32&ndash;64), so each K/V tile is shared and each reduction "
                 "is done once per row."]
    out += bullets(perf, "Performance")
    out += shot(run, "C2_tiled", "c2_tiled, N = 2048, d = 64")
    out += shot(run, "C_sizes", "C1, C2, C3 at N = 1024 and N = 4096 (d = 64)")

    # C3
    out += subsection("C3: Attention with cuBLAS")
    out += table(["Call", "Computes (col-major)", "trans a,b", "m, n, k", "A (lda)", "B (ldb)", "C (ldc)",
                  "&alpha;, &beta;"],
                 [["1: S = &alpha;QK<super>T</super>", "S<super>T</super> = K&middot;Q<super>T</super>",
                   "T, N", "N, N, d", "dK (d)", "dQ (d)", "dS (N)", "1/&radic;d, 0"],
                  ["2: O = PV", "O<super>T</super> = V<super>T</super>&middot;P<super>T</super>",
                   "N, N", "d, N, N", "dV (d)", "dS (N)", "dO (d)", "1, 0"]],
                 "cublasDgemm calls (all buffers row-major).",
                 [1.0 * inch, 1.25 * inch, 0.6 * inch, 0.7 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch])
    out += bullets([
        "A row-major r x c buffer, read column-major, is its transpose (c x r, ld = c).",
        "<b>Call 1:</b> the K buffer is K<super>T</super> (d x N); OP_T gives K. The result "
        "K&middot;Q<super>T</super> = S<super>T</super> is column-major with ld N, which is S in row-major.",
        "<b>Call 2:</b> V<super>T</super>&middot;P<super>T</super> = O<super>T</super>, column-major with ld d, "
        "which is O in row-major.",
        "No explicit transposes. The softmax between the calls is the same kernel as C1.",
        "One untimed warm-up pass (QK<super>T</super> GEMM, softmax, PV GEMM) absorbs cuBLAS initialisation; every QK<super>T</super> GEMM is followed by the PV GEMM that reads its S.",
    ], "Layout mapping")
    out += shot(run, "C3_cublas", "c3_cublas, N = 2048, d = 64")

    # C4
    out += subsection("C4 (bonus): Attention with OpenAI Triton")
    tri = re.search(r"Triton attention:\s*([\d.]+)", c4)
    sd = re.search(r"PyTorch SDPA:\s*([\d.]+)", c4)
    err = re.search(r"max abs error vs reference:\s*([\deE.+-]+)", c4)
    out += bullets([
        "One program per BM = 64 query rows; the Q tile is kept in registers.",
        "Loops over K/V in tiles of BN = 64; scores via " + code("tl.dot(q, tl.trans(k))") + ".",
        "Keys past N are masked to &minus;inf; same online softmax as C2.",
        code("my_attention") + ": grid of ceil(N/64) programs, tensor strides passed in. fp32, N = 2048, d = 64.",
    ], "Design")
    if tri and sd:
        t, s = float(tri.group(1)), float(sd.group(1))
        out += table(["Implementation", "Time (ms, do_bench)"], [["Triton attn_kernel", f"{t:.3f}"],
                                                                 ["PyTorch SDPA (fused)", f"{s:.3f}"]],
                     "Triton vs PyTorch scaled_dot_product_attention.")
        e = float(err.group(1)) if err else None
        res = [f"Max abs error vs PyTorch reference: {e:.2e} (check: &lt; 1e-2)." if e is not None else
               "Correctness check passed."]
        if s < t:
            res += [f"SDPA is {t / s:.1f}x faster at this size.",
                    "It uses a tuned fused backend (block sizes, pipelining).",
                    "On the T4 (no TF32), my fp32 " + code("tl.dot") + " runs on regular FP32 units, and my "
                    "64x64 tiles are not autotuned."]
        else:
            res += [f"Triton is {s / t:.1f}x faster at this size.",
                    "At N = 2048, SDPA's dispatch overhead is a large share of its time.",
                    "The tuned backend would be expected to win at larger N."]
        out += bullets(res, "Results")
    out += shot(run, "C4_triton", "executed c4_triton.ipynb outputs")
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(Path(__file__).parent / "report_out"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "report.pdf"))
    a = ap.parse_args()
    run = Run(a.inp)
    missing = [k for k in ("00_setup", "A_Q1_vecadd00", "A_Q2_vecadd01", "A_Q3_matmult00", "A_Q4_matmult01",
                           "B_Q1_cpu", "B_Q2_malloc", "B_Q3_unified", "C1_naive", "C2_tiled", "C3_cublas",
                           "C_sizes", "C4_triton") if k not in run.logs]
    if missing:
        print("WARNING: missing logs:", ", ".join(missing))

    bdata = part_b_data(run)
    story = header_block(run) + setup_section(run) + part_a(run, bdata) + part_b(run, bdata) + part_c(run)

    def decorate(c, d):
        c.saveState()
        c.setStrokeColor(RULE)
        c.setLineWidth(0.5)
        c.line(0.75 * inch, letter[1] - 0.55 * inch, letter[0] - 0.75 * inch, letter[1] - 0.55 * inch)
        c.line(0.75 * inch, 0.62 * inch, letter[0] - 0.75 * inch, 0.62 * inch)
        c.setFont("Helvetica", 7.8)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(0.75 * inch, letter[1] - 0.5 * inch, "HPML Homework 3 - CUDA Kernels - Lab Report")
        c.drawRightString(letter[0] - 0.75 * inch, letter[1] - 0.5 * inch, f"{NAME} ({UNI})")
        c.drawRightString(letter[0] - 0.75 * inch, 0.45 * inch, f"Page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(a.out, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                            title="HPML HW3 Lab Report", author=NAME)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
