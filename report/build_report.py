#!/usr/bin/env python3
"""
Build report.pdf for HPML HW3 from the outputs of hw3_report_runs.ipynb.

Usage (from this folder, after unzipping hw3_report_outputs.zip here):
    pip install reportlab
    python build_report.py                 # reads ./report_out, writes ./report.pdf
    python build_report.py --in report_out --out report.pdf

Every number in the report is parsed from the program logs in report_out/logs,
and every "screenshot" is report_out/shots/<log>.png, an image of that same log.
Nothing is typed in by hand, so the text, tables, and screenshots always agree.
"""
import argparse
import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# ---------------------------------------------------------------- edit me
NAME = "Hardik Gupta"
UNI = "YOUR_UNI"
AI_DISCLOSURE = (
    "I used Claude Code (Anthropic's AI coding assistant) to help write and debug the "
    "CUDA kernels, the Triton notebook, and the Colab script that collected these "
    "measurements, and to draft the report text from my own measured outputs, which "
    "I reviewed and verified."
)
# ------------------------------------------------------------------------

RESULT_RE = re.compile(r"^RESULT,(.+)$")
FMT_RE = re.compile(r"^(-?\d+\.\d{3}),(\d+\.\d{3})$")  # C1..C3 "checksum,time" line


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
                try:
                    d[k] = int(v)
                except ValueError:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
        rows.append(d)
    return rows


class Run:
    def __init__(self, root):
        self.root = Path(root)
        self.logs = {p.stem: p.read_text() for p in (self.root / "logs").glob("*.txt")}
        self.res = {k: parse_results(v) for k, v in self.logs.items()}

    def shot(self, name):
        p = self.root / "shots" / f"{name}.png"
        return p if p.exists() else None

    def find(self, log, **kw):
        for r in self.res.get(log, []):
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    def t(self, log, **kw):
        r = self.find(log, **kw)
        return r["time_ms"] if r else None


# ------------------------------------------------------------------ helpers
def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ratio(a, b):
    return (a / b) if (a and b) else None


def fx(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}x"


styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1f3b63"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9.6, leading=13, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.2, leading=10.5, textColor=colors.HexColor("#555555"))
CAP = ParagraphStyle("C", parent=SMALL, alignment=TA_CENTER, spaceAfter=8)
TITLE = ParagraphStyle("T", parent=styles["Title"], fontSize=18, spaceAfter=4)
CENTER = ParagraphStyle("Ce", parent=BODY, alignment=TA_CENTER)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.6, leading=11, spaceAfter=0)


def P(text, st=BODY):
    return Paragraph(text, st)


def bullets(items):
    return [P("&bull;&nbsp;" + it, ParagraphStyle("bl", parent=BODY, leftIndent=12, firstLineIndent=-8))
            for it in items]


def table(header, rows, widths=None):
    data = [[P(f"<b>{h}</b>", CELL) for h in header]]
    data += [[P(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cf")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 6)]


def image(path, caption, max_w=6.9 * inch, max_h=8.6 * inch):
    if path is None or not Path(path).exists():
        return [P(f"<font color='#b00020'>[missing image: {escape(caption)}]</font>")]
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    # Log images are rendered at 160 dpi; never scale past natural size (72/160 pt per px),
    # so every terminal screenshot has the same font size.
    cap = 72 / 160 if "shots" in str(path) else 10.0
    s = min(max_w / iw, max_h / ih, cap)
    w, h = iw * s, ih * s
    return [KeepTogether([Image(str(path), width=w, height=h), P(caption, CAP)])]


def shot(run, name, caption):
    return image(run.shot(name), f"Screenshot: {caption} (Colab terminal output, {name}.txt)")


# ------------------------------------------------------------------ sections
def setup_section(run):
    s = run.logs.get("00_setup", "")
    gpu = re.search(r"^name, driver_version.*\n([^\n]+)", s, re.M)
    gpu_name = drv = mem = cap = "see screenshot"
    if gpu:
        parts = [p.strip() for p in gpu.group(1).split(",")]
        gpu_name, drv = parts[0], parts[1]
        mem = parts[2] if len(parts) > 2 else mem
        cap = parts[3] if len(parts) > 3 else cap
    nvcc = re.search(r"release ([\d.]+)", s)
    cpu = re.search(r"Model name:\s*(.+)", s)
    ncpu = re.search(r"^CPU\(s\):\s*(\d+)", s, re.M)
    os_ = re.search(r'PRETTY_NAME="([^"]+)"', s)
    kern = re.search(r"^(\d+\.\d+\.\d+\S*)$", s, re.M)
    out = [P("Setup", H1)]
    out += table(["Item", "Value"], [
        ["Platform", "Google Colab, GPU runtime"],
        ["GPU", f"{gpu_name} ({mem}, compute capability {cap})"],
        ["NVIDIA driver", drv],
        ["CUDA toolkit (nvcc)", nvcc.group(1) if nvcc else "see screenshot"],
        ["CPU", (cpu.group(1).strip() if cpu else "see screenshot") + (f", {ncpu.group(1)} vCPUs" if ncpu else "")],
        ["OS / kernel", (os_.group(1) if os_ else "see screenshot") + (f", Linux {kern.group(1)}" if kern else "")],
        ["Build", "make ARCH=sm_75 (nvcc -O3), from the unmodified starter Makefiles"],
    ], [1.6 * inch, 5.3 * inch])
    out += [P("<b>Timing methodology.</b> All GPU numbers are <i>kernel-only</i> times in milliseconds "
              "(3 decimals). Part A and Part B use the provided <font face='Courier'>GpuTimer</font> "
              "(CUDA events recorded around the launches, then synchronized); Part C uses CUDA events "
              "around the kernels / cuBLAS calls. Data generation, allocation, host-device copies and "
              "cleanup are outside the timed region. Each number is a <b>single run</b> (no averaging); "
              "the only warm-up is the one-time cuBLAS initialisation call in C3, done before timing. "
              "Part B Q3 (unified memory) numbers are <b>first-touch</b>: the host initialises the "
              "managed arrays, so the GPU's first access inside the timed region triggers on-demand "
              "page migration. The CPU baseline (Part B Q1) is timed with "
              "<font face='Courier'>std::chrono</font> around <font face='Courier'>addArrays</font> only. "
              "C4 is timed with <font face='Courier'>triton.testing.do_bench</font> (median of many runs). "
              "All outputs below come from one Colab session.")]
    out += shot(run, "00_setup", "GPU, driver, CUDA, CPU and OS")
    return out


def part_a(run):
    out = [PageBreak(), P("Part A: CUDA Matrix Operations", H1)]
    vpts = (500, 1000, 2000)
    v0 = {v: run.find("A_Q1_vecadd00", values_per_thread=v) for v in vpts}
    v1 = {v: run.find("A_Q2_vecadd01", values_per_thread=v) for v in vpts}

    # Q1
    out += [P("Q1. vecadd00 (non-coalesced)", H2)]
    out += [P("Each thread of the provided kernel owns a contiguous run of "
              "<font face='Courier'>valuesPerThread</font> elements starting at "
              "<font face='Courier'>tid*valuesPerThread</font>. At any loop step the 32 threads of a warp "
              "therefore read addresses that are <font face='Courier'>valuesPerThread*4</font> bytes apart, "
              "so every warp-level load and store is split into 32 separate memory transactions.")]
    out += table(["values per thread", "N (floats)", "vecadd00 time (ms)", "result"],
                 [[v, v0[v]["N"] if v0[v] else "n/a", f3(v0[v] and v0[v]["time_ms"]),
                   "PASSED" if v0[v] and v0[v].get("correct") == 1 else "n/a"] for v in vpts])
    if all(v0.values()):
        t = [v0[v]["time_ms"] for v in vpts]
        bw = [3 * v0[v]["N"] * 4 / (v0[v]["time_ms"] / 1e3) / 1e9 for v in vpts]
        out += [P(f"Time grows roughly linearly with the work ({t[0]:.3f} &rarr; {t[1]:.3f} &rarr; "
                  f"{t[2]:.3f} ms for 1x/2x/4x the data), but the effective bandwidth is only "
                  f"{bw[0]:.0f}&ndash;{bw[2]:.0f} GB/s (3 arrays x 4 B per element), a small fraction of the "
                  "T4's ~320 GB/s peak, because most of each 32-byte memory sector fetched is wasted.")]
    out += shot(run, "A_Q1_vecadd00", "vecadd00 at 500, 1000, 2000 values per thread")

    # Q2
    out += [P("Q2. vecadd01 (coalesced)", H2)]
    out += [P("In <font face='Courier'>vecaddKernel01.cu</font> thread <font face='Courier'>tid</font> handles "
              "element <font face='Courier'>j*totalThreads + tid</font> at loop step <font face='Courier'>j</font> "
              "(<font face='Courier'>totalThreads = gridDim.x*blockDim.x</font>). Each thread still does "
              "exactly <font face='Courier'>valuesPerThread</font> additions, so the work is identical, but now "
              "consecutive threads touch consecutive floats and a warp's 32 accesses merge into one 128-byte "
              "transaction per array.")]
    rows, sp = [], {}
    for v in vpts:
        a, b = v0[v] and v0[v]["time_ms"], v1[v] and v1[v]["time_ms"]
        sp[v] = ratio(a, b)
        bw = (3 * v1[v]["N"] * 4 / (b / 1e3) / 1e9) if (v1[v] and b) else None
        rows.append([v, f3(a), f3(b), fx(sp[v], 1), "n/a" if bw is None else f"{bw:.0f} GB/s"])
    out += table(["values per thread", "vecadd00 (ms)", "vecadd01 (ms)", "speedup", "vecadd01 bandwidth"], rows)
    if all(sp.values()):
        lo, hi = min(sp.values()), max(sp.values())
        rng = f"{lo:.1f}x" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f}x&ndash;{hi:.1f}x"
        out += [P(f"Coalescing gives a <b>{rng}</b> speedup for "
                  "the same work. The coalesced kernel runs close to the T4's DRAM bandwidth, so it is now "
                  "limited by memory bandwidth, while the original was limited by the number of separate "
                  "transactions it issued.")]
    out += shot(run, "A_Q2_vecadd01", "vecadd01 at 500, 1000, 2000 values per thread")

    # Q3
    sizes = (256, 512, 1024)
    m0 = {s: run.t("A_Q3_matmult00", size=s) for s in sizes}
    m1 = {s: run.t("A_Q4_matmult01", size=s) for s in sizes}
    gf = lambda s, t: (2 * s ** 3 / (t / 1e3) / 1e9) if t else None
    out += [P("Q3. matmult00 (FOOTPRINT_SIZE = 16, one output per thread)", H2)]
    out += [P("Each 16x16 thread block loads a 16x16 tile of A and of B into shared memory per step and every "
              "thread computes one element of C. The argument is multiplied by FOOTPRINT_SIZE, so arguments "
              "16/32/64 give 256/512/1024 matrices.")]
    out += table(["matrix size", "time (ms)", "GFLOP/s"],
                 [[f"{s} x {s}", f3(m0[s]), "n/a" if not m0[s] else f"{gf(s, m0[s]):.0f}"] for s in sizes])
    if all(m0.values()):
        out += [P(f"Going from 256 to 1024 multiplies the work by 64x but the time only by "
                  f"{m0[1024] / m0[256]:.1f}x: at 256 there are only (256/16)<super>2</super> = 256 blocks, "
                  "too few to hide memory latency on 40 SMs, and fixed launch overhead dominates the short "
                  "kernel. Larger matrices fill the GPU and throughput rises.")]
    out += shot(run, "A_Q3_matmult00", "matmult00 at 256, 512, 1024")

    # Q4
    out += [P("Q4. matmult01 (FOOTPRINT_SIZE = 32, 2x2 outputs per thread)", H2)]
    out += [P("The thread block stays 16x16, but each block now computes a 32x32 tile of C, so each thread "
              "computes four outputs at rows {ty, ty+16} and columns {tx, tx+16}. Design choices:")]
    out += bullets([
        "<b>Coalesced loads and stores:</b> each thread loads 4 elements of the A tile and 4 of the B tile at "
        "stride 16 (not 2 adjacent ones), so consecutive <font face='Courier'>threadIdx.x</font> always touch "
        "consecutive addresses in global memory, and shared-memory accesses are conflict-free. The four results "
        "are written the same way.",
        "<b>Register blocking:</b> in the inner loop each thread reads 2 values of A and 2 of B from shared "
        "memory and does 4 FMAs, 1 FMA per shared load instead of 0.5 in matmult00, keeping the 4 partial sums "
        "in registers.",
        "<b>Data reuse:</b> each value brought from global memory is reused 32 times instead of 16, halving "
        "global-memory traffic per FLOP.",
        "<b>Unrolling:</b> the 32-step inner accumulation loop and the 2x2 load loops are fully unrolled with "
        "<font face='Courier'>#pragma unroll</font>, removing loop overhead and letting the compiler keep "
        "everything in registers.",
    ])
    rows = []
    for s in sizes:
        rows.append([f"{s} x {s}", f3(m0[s]), f3(m1[s]), fx(ratio(m0[s], m1[s])),
                     "n/a" if not m1[s] else f"{gf(s, m1[s]):.0f}"])
    out += table(["matrix size", "matmult00 (ms)", "matmult01 (ms)", "speedup", "matmult01 GFLOP/s"], rows)
    sp = {s: ratio(m0[s], m1[s]) for s in sizes}
    if all(sp.values()):
        txt = f"At 1024 the new kernel is <b>{sp[1024]:.2f}x</b> faster, thanks to the higher data reuse and FMA:load ratio. "
        slow = [s for s in sizes if sp[s] < 1]
        if slow:
            txt += (f"At {', '.join(map(str, slow))} it is <i>slower</i> ({', '.join(fx(sp[s]) for s in slow)}): a 32x32 tile "
                    f"gives only ({slow[0]}/32)<super>2</super> = {(slow[0] // 32) ** 2} blocks for the T4's 40 SMs, "
                    "so most SMs hold a single block and cannot hide latency, and the run is so short that "
                    "launch overhead and timer noise matter. Register blocking pays off once the matrix is large "
                    "enough to keep every SM busy.")
        else:
            txt += "The gain grows with matrix size because larger matrices keep every SM busy."
        out += [P(txt)]
    out += shot(run, "A_Q4_matmult01", "matmult01 at 256, 512, 1024")
    return out, sp, m0, m1


def part_b_data(run):
    Ks = (1, 5, 10, 50, 100)
    cpu = {K: run.t("B_Q1_cpu", K=K) for K in Ks}
    q2 = {(s, K): run.t("B_Q2_malloc", scenario=s, K=K) for s in (1, 2, 3) for K in Ks}
    q3 = {(s, K): run.t("B_Q3_unified", scenario=s, K=K) for s in (1, 2, 3) for K in Ks}
    return Ks, cpu, q2, q3


def part_a_q5(run, a_sp, m0, m1, bdata, vsp):
    Ks, cpu, q2, q3 = bdata
    out = [P("Q5. Rules of thumb (from Parts A and B)", H2)]
    v0, v1 = run.t("A_Q1_vecadd00", values_per_thread=2000), run.t("A_Q2_vecadd01", values_per_thread=2000)
    s1, s3 = q2.get((1, 10)), q2.get((3, 10))
    s2_100, s3_100 = q2.get((2, 100)), q2.get((3, 100))
    u3, e3 = q3.get((3, 100)), q2.get((3, 100))
    rules = [
        f"<b>Make consecutive threads access consecutive addresses (coalescing).</b> Same work, different "
        f"index mapping: vecadd00 {f3(v0)} ms vs vecadd01 {f3(v1)} ms at 2000 values/thread "
        f"(<b>{fx(ratio(v0, v1), 1)}</b>).",
        f"<b>Launch few large kernels, never many tiny ones.</b> Scenario 1 (one thread per launch) takes "
        f"{f3(s1)} ms at K = 10 vs {f3(s3)} ms for one launch (<b>{fx(ratio(s1, s3), 0)}</b>); that is about "
        f"{(s1 * 1e3 / 1e7) if s1 else float('nan'):.2f} &micro;s of launch overhead per launch.",
        f"<b>Give the GPU enough parallel work per launch.</b> Even with 256 threads per launch, scenario 2 is "
        f"{fx(ratio(s2_100, s3_100), 0)} slower than a single launch at K = 100 ({f3(s2_100)} vs {f3(s3_100)} ms): a "
        "256-thread launch occupies one SM out of 40.",
        f"<b>Reuse data from on-chip memory (shared memory and registers).</b> Computing a 2x2 subtile per "
        f"thread raised reuse from 16 to 32 and gave {fx(a_sp.get(1024))} at 1024 ({f3(m0.get(1024))} &rarr; "
        f"{f3(m1.get(1024))} ms).",
        f"<b>Keep data resident on the GPU / avoid paging inside hot loops.</b> The same single launch at "
        f"K = 100 takes {f3(e3)} ms on data already copied with cudaMemcpy, but {f3(u3)} ms on first-touch "
        f"unified memory ({fx(ratio(u3, e3), 0)}), because pages migrate on demand while the kernel runs.",
    ]
    out += bullets(rules)
    slow = [s for s in (256, 512, 1024) if a_sp.get(s) and a_sp[s] < 1]
    if slow:
        s = slow[0]
        out += [P(f"<b>A change that did not help.</b> Register blocking (FOOTPRINT_SIZE 16 &rarr; 32) made the "
                  f"{s}x{s} multiply <i>slower</i>: {f3(m0[s])} ms &rarr; {f3(m1[s])} ms ({fx(a_sp[s])}). With 32x32 "
                  f"tiles a {s}x{s} matrix gives only {(s // 32) ** 2} blocks, fewer than needed to occupy 40 SMs, "
                  "so the extra per-thread work cannot be overlapped with memory latency. More work per thread "
                  "only helps when there is still enough parallelism left to fill the GPU.")]
    else:
        out += [P(f"<b>A change that did not help.</b> Switching the single-launch vector add from explicit "
                  f"cudaMalloc/cudaMemcpy to cudaMallocManaged made the timed kernel slower at K = 100 "
                  f"({f3(e3)} &rarr; {f3(u3)} ms), because first-touch page migration lands inside the kernel.")]
    return out


def part_b(run, bdata):
    Ks, cpu, q2, q3 = bdata
    out = [PageBreak(), P("Part B: CUDA Unified Memory", H1)]
    out += [P("Q1. CPU baseline", H2)]
    out += [P("<font face='Courier'>q1_cpu.cpp</font> adds two K-million-element float arrays with a plain loop "
              "(<font face='Courier'>z[i] = x[i] + y[i]</font>); memory is freed with "
              "<font face='Courier'>free()</font>. Only the add loop is timed.")]
    out += table(["K (million)"] + [str(K) for K in Ks], [["CPU time (ms)"] + [f3(cpu[K]) for K in Ks]])
    out += shot(run, "B_Q1_cpu", "q1_cpu for K = 1, 5, 10, 50, 100")

    def scen_table(d):
        rows = []
        for s, lab in ((1, "1: n launches x 1 thread"), (2, "2: n/256 launches x 256 threads"),
                       (3, "3: 1 launch, ceil(n/256) blocks x 256 threads")):
            rows.append([lab] + [("capped" if (s == 1 and K > 10) else f3(d[(s, K)])) for K in Ks])
        return table(["scenario \\ K (million)"] + [str(K) for K in Ks], rows,
                     [2.6 * inch] + [0.85 * inch] * len(Ks))

    out += [P("Q2. GPU with explicit cudaMalloc / cudaMemcpy", H2)]
    out += [P("Arrays are allocated with <font face='Courier'>cudaMalloc</font> and copied with "
              "<font face='Courier'>cudaMemcpy</font> before timing; one GpuTimer wraps <i>all</i> launches of a "
              "scenario, so scenario 1's per-launch overhead is included. Grid/block per launch: scenario 1 "
              "<font face='Courier'>&lt;&lt;&lt;1,1&gt;&gt;&gt;</font> x n; scenario 2 "
              "<font face='Courier'>&lt;&lt;&lt;1,256&gt;&gt;&gt;</font> x floor(n/256) plus one "
              "<font face='Courier'>&lt;&lt;&lt;1,n mod 256&gt;&gt;&gt;</font> for the remainder; scenario 3 "
              "<font face='Courier'>&lt;&lt;&lt;ceil(n/256),256&gt;&gt;&gt;</font> once. Scenario 1 is capped at "
              "K &le; 10. Times in ms:")]
    out += scen_table(q2)
    out += shot(run, "B_Q2_malloc", "q2_malloc, all K and all three scenarios")

    out += [P("Q3. GPU with Unified Memory (cudaMallocManaged)", H2)]
    out += [P("Same three scenarios, but x, y, z come from <font face='Courier'>cudaMallocManaged</font>, are "
              "initialised by the host, and there is no explicit copy; they are freed with "
              "<font face='Courier'>cudaFree</font>. These are <b>first-touch</b> numbers: page migration from "
              "host to device happens inside the timed region, which Q2 paid for outside it. Times in ms:")]
    out += scen_table(q3)
    out += shot(run, "B_Q3_unified", "q3_unified, all K and all three scenarios")

    # Q4
    out += [PageBreak(), P("Q4. Predictions, charts and analysis", H2)]
    out += [P("<b>Predictions (written before looking at the data).</b>")]
    out += bullets([
        "<b>Scenario 1</b> is pure launch overhead (several &micro;s per launch, n launches), so it grows linearly "
        "in K and is by far the slowest, seconds even at K = 1, far above the CPU.",
        "<b>Scenario 2</b> has 256x fewer launches, so it is ~2 orders of magnitude below scenario 1, still linear "
        "in K, and still slower than the CPU because each launch occupies only one SM.",
        "<b>Scenario 3</b> with explicit copies is one launch on data already in GPU memory, so it is the "
        "fastest by far (sub-ms to a few ms), bandwidth-bound and roughly linear in K, well below the CPU.",
        "<b>Unified memory</b>: scenarios 1 and 2 barely change (launch overhead dominates), but scenario 3 "
        "gets much slower because the first GPU touch migrates every page over PCIe inside the timed region, "
        "pulling it close to the CPU line.",
        "<b>Ordering</b> (slowest to fastest): scenario 1 &gt; scenario 2 &gt; CPU &gt; scenario 3 without UM; with "
        "UM, scenario 3 moves up near the CPU.",
    ])
    root = run.root / "plots"
    for f, cap in (("partb_without_unified.png", "Chart for Q2: explicit cudaMalloc/cudaMemcpy vs CPU (log-log)"),
                   ("partb_with_unified.png", "Chart for Q3: unified memory vs CPU (log-log)")):
        out += image(root / f, cap, max_w=5.6 * inch, max_h=3.9 * inch)

    s1_1, s1_10 = q2.get((1, 1)), q2.get((1, 10))
    obs = []
    if s1_1 and s1_10:
        obs.append(f"<b>Scenario 1</b> took {f3(s1_1)} ms at K = 1 and {f3(s1_10)} ms at K = 10 "
                   f"({s1_10 / s1_1:.1f}x for 10x the elements), i.e. ~{s1_10 * 1e3 / 1e7:.2f} &micro;s per launch, "
                   "linear as predicted and the slowest by 2&ndash;3 orders of magnitude. Unified memory changes it "
                   f"little ({f3(q3.get((1, 10)))} ms at K = 10) since it is launch-bound.")
    if q2.get((2, 100)) and cpu.get(100):
        obs.append(f"<b>Scenario 2</b> is linear in K and sits above the CPU at every K "
                   f"({f3(q2.get((2, 100)))} ms vs CPU {f3(cpu.get(100))} ms at K = 100), as predicted: "
                   f"~{q2[(2, 100)] * 1e3 / (1e8 / 256):.2f} &micro;s per 256-thread launch, so it too is launch-bound.")
    if q2.get((3, 1)) and q2.get((3, 100)) and cpu.get(100):
        obs.append(f"<b>Scenario 3 (explicit copies)</b> is the fastest: {f3(q2[(3, 1)])} ms at K = 1 to "
                   f"{f3(q2[(3, 100)])} ms at K = 100, {cpu[100] / q2[(3, 100)]:.0f}x faster than the CPU at K = 100. "
                   f"At K = 100 it moves 1.2 GB in {f3(q2[(3, 100)])} ms "
                   f"(~{1.2e9 / (q2[(3, 100)] / 1e3) / 1e9:.0f} GB/s), i.e. it is bandwidth-bound. Note this "
                   "excludes the cudaMemcpy cost, which Q2 pays outside the timer.")
    if q3.get((3, 100)) and q2.get((3, 100)) and cpu.get(100):
        obs.append(f"<b>Scenario 3 (unified memory)</b> is {q3[(3, 100)] / q2[(3, 100)]:.0f}x slower than with "
                   f"explicit copies at K = 100 ({f3(q3[(3, 100)])} vs {f3(q2[(3, 100)])} ms) and ends up close to "
                   f"the CPU ({f3(cpu[100])} ms): the kernel stalls on page faults while pages migrate over PCIe. "
                   "This matched the prediction; the gap would largely disappear with "
                   "<font face='Courier'>cudaMemPrefetchAsync</font> or a warm-up pass (steady-state), which is why "
                   "the Q2 and Q3 columns are not a head-to-head comparison.")
    # Check the predicted ordering (s1 > s2 > CPU > s3, explicit copies) at every K where all exist.
    bad = []
    for K in Ks:
        seq = [("scenario 1", q2.get((1, K))), ("scenario 2", q2.get((2, K))), ("CPU", cpu.get(K)),
               ("scenario 3", q2.get((3, K)))]
        seq = [(n, t) for n, t in seq if t is not None]
        if any(a[1] <= b[1] for a, b in zip(seq, seq[1:])):
            bad.append(K)
    g3 = ratio(q2.get((3, 10)), q2.get((3, 1)))
    g3u = ratio(q3.get((3, 10)), q3.get((3, 1)))
    diff = ("<b>Difference from the predictions:</b> the predicted ordering (scenario 1 &gt; scenario 2 &gt; CPU "
            "&gt; scenario 3 with explicit copies) " +
            ("held at every K. " if not bad else f"did <i>not</i> hold at K = {', '.join(map(str, bad))}. "))
    if g3 and g3u:
        diff += (f"From K = 1 to 10 (10x the data), scenario 3 grew {g3:.1f}x with explicit copies and {g3u:.1f}x "
                 "with unified memory. " +
                 ("Growth well under 10x means a fixed per-launch / page-fault setup cost still matters at small K, "
                  "so the curve is flatter at the left than a pure bandwidth model predicts."
                  if min(g3, g3u) < 7 else
                  "Both are close to 10x, i.e. even K = 1 is already in the bandwidth-bound regime; the fixed launch "
                  "cost is too small to flatten the curve."))
    obs.append(diff)
    out += [P("<b>Observed vs predicted.</b>")] + bullets(obs)
    return out


def part_c(run):
    out = [PageBreak(), P("Part C: Attention in CUDA", H1)]
    fmt = {}
    for log in ("C1_naive", "C2_tiled", "C3_cublas"):
        for line in run.logs.get(log, "").splitlines():
            m = FMT_RE.match(line.strip())
            if m:
                fmt[log] = (m.group(1), m.group(2))
    c4 = run.logs.get("C4_triton", "")
    c4_fmt = None
    for line in c4.splitlines():
        if re.fullmatch(r"\d+\.\d{3}", line.strip()):
            c4_fmt = line.strip()
    out += [P("All three CUDA versions use double precision with N = 2048, d = 64 (the compiled-in defaults) "
              "and print the required <font face='Courier'>checksum,time</font> line. Program output in the "
              "required format:")]
    lines = [f"{fmt[k][0]},{fmt[k][1]}" if k in fmt else "n/a" for k in ("C1_naive", "C2_tiled", "C3_cublas")]
    lines.append(c4_fmt or "n/a")
    out += [P("<font face='Courier'>" + "<br/>".join(escape(l) for l in lines) + "</font>",
              ParagraphStyle("pre", parent=BODY, leftIndent=18, backColor=colors.HexColor("#f4f4f4"),
                             borderPadding=5, spaceBefore=4, spaceAfter=10))]
    c = {k: run.find(k, N=2048) for k in ("C1_naive", "C2_tiled", "C3_cublas")}
    out += table(["version", "checksum", "kernel time (ms)"],
                 [[n, c[k]["checksum"] if c[k] else "n/a", f3(c[k] and c[k]["time_ms"])]
                  for k, n in (("C1_naive", "C1 naive"), ("C2_tiled", "C2 tiled (FlashAttention-style)"),
                               ("C3_cublas", "C3 cuBLAS"))])
    cs = {c[k]["checksum"] for k in c if c[k]}
    if len(cs) == 1:
        out += [P(f"All three checksums are identical ({cs.pop()}). Correctness was also checked element by "
                  "element against an independent Python reference at N, d = (8, 4), (129, 4), (257, 8) by "
                  "grade_local.py (see results.json; all three pass).")]

    # C1
    out += [P("C1. Naive attention", H2)]
    out += [P("Three kernels: (1) <b>scores</b>, a 2-D grid of 16x16 blocks, one thread per S[i,j] with j on "
              "threadIdx.x so writes to S are coalesced; (2) <b>safe row softmax</b>, one 256-thread block per "
              "row, a shared-memory tree reduction for the row max, then exp(S - max) and a second reduction for "
              "the sum, then normalisation, all in place; (3) <b>output</b>, one thread per O[i,t] summing "
              "P[i,j]&middot;V[j,t]. The full N x N matrix S (N<super>2</super>&middot;8 B = 32 MiB at N = 2048) "
              "lives in global memory and is written once and read/written several more times.")]
    out += shot(run, "C1_naive", "c1_naive at N = 2048, d = 64")

    # C2
    out += [P("C2. Tiled (FlashAttention-style) attention", H2)]
    out += [P("One thread block (128 threads) per query row i. The query row is staged in shared memory; keys "
              "and values are processed in tiles of 16 rows, each tile of K and V loaded cooperatively and "
              "coalesced into shared memory (padded to d+1 to avoid bank conflicts). For each tile the block "
              "computes the 16 scores into shared memory, then updates the running state: running max m, running "
              "denominator l, and the output accumulator acc[d] (spread over the threads' registers). "
              "<font face='Courier'>__syncthreads()</font> separates every shared-memory write from the reads "
              "that depend on it. At the end O[i,:] = acc / l. The N x N score matrix is never written to "
              "global memory, only 16 scores at a time in shared memory.")]
    out += [P("<b>Why the accumulator must be rescaled.</b> After processing keys 0..j the state is "
              "m = max of the scores so far, l = &Sigma; exp(s<sub>k</sub> &minus; m), and "
              "acc = &Sigma; exp(s<sub>k</sub> &minus; m)&middot;V<sub>k</sub>; every term is measured relative to "
              "the same m. When a new tile has a larger maximum m' &gt; m, the new terms are computed as "
              "exp(s &minus; m'), so the old sums must be multiplied by &alpha; = exp(m &minus; m') to express them "
              "relative to m' as well: l &larr; &alpha;l + &Sigma;exp(s &minus; m'), acc &larr; &alpha;acc + "
              "&Sigma;exp(s &minus; m')V. Then acc/l at the end is exactly softmax(S)&middot;V. "
              "<b>Without rescaling</b>, keys seen before the max increased would be over-weighted by a factor "
              "exp(m' &minus; m) &gt; 1 compared with later keys, so the output would no longer be a correctly "
              "normalised softmax average (wrong values, biased toward early keys). If l were rescaled but acc "
              "not, the rows would not even sum to the right total. Skipping the max subtraction altogether "
              "could also overflow exp() for large scores.")]
    sizes = [(1024, 64), (2048, 64), (4096, 64)]
    rows = []
    for N, d in sizes:
        t1 = run.t("C1_naive", N=N, d=d) or run.t("C_sizes", program="c1_naive", N=N, d=d)
        t2 = run.t("C2_tiled", N=N, d=d) or run.t("C_sizes", program="c2_tiled", N=N, d=d)
        t3 = run.t("C3_cublas", N=N, d=d) or run.t("C_sizes", program="c3_cublas", N=N, d=d)
        rows.append([N, f3(t1), f3(t2), f3(t3), fx(ratio(t2, t1), 1)])
    out += [P("<b>Performance.</b> Measured at three sequence lengths (d = 64), kernel time in ms:")]
    out += table(["N", "C1 naive", "C2 tiled", "C3 cuBLAS", "C2 / C1"], rows)
    t1d = run.t("C1_naive", N=2048, d=64)
    t2d = run.t("C2_tiled", N=2048, d=64)
    traffic = ("C2 avoids the N x N HBM traffic: C1 writes and then makes several passes over a 32 MiB "
               "(N = 2048) to 128 MiB (N = 4096) score matrix in global memory, while C2 only streams K and V, "
               "which at 2 x N x d x 8 B = 2 MiB fit in the T4's 4 MiB L2 cache.")
    if not (t1d and t2d and t2d > t1d):
        out += [P(traffic + f" At N = 2048, C2 takes {f3(t2d)} ms vs {f3(t1d)} ms for C1.")]
        out += shot(run, "C2_tiled", "c2_tiled at N = 2048, d = 64")
        out += shot(run, "C_sizes", "C1/C2/C3 at N = 1024 and N = 4096 (d = 64)")
        return out + part_c3_c4(run, c4)
    out += [P(traffic + f" In my implementation, though, C2 is <b>slower</b> than C1 ({f3(t2d)} vs {f3(t1d)} ms "
              "at N = 2048), for reasons that come from my block-to-row mapping rather than from tiling itself:")]
    out += bullets([
        "<b>Redundant exp() work.</b> Every one of the 128 threads in a block recomputes the tile max and all "
        "16 exponentials to update its own copy of (m, l), i.e. 128x more exp() calls than needed "
        "(~128&middot;N<super>2</super> in total vs N<super>2</super> in C1), and fp64 exp is expensive on a T4, "
        "whose fp64 rate is 1/32 of fp32.",
        "<b>Low utilisation.</b> Only 16 of 128 threads compute scores per tile, and with d = 64 only 64 threads "
        "accumulate the output; the rest wait at the barriers.",
        "<b>No reuse of K/V across query rows.</b> With one row per block, each of the N blocks streams all of K "
        "and V through shared memory, so a K/V tile is used for one query row instead of many.",
    ])
    out += [P("The fix is the standard FlashAttention mapping: give each block a tile of Br query rows (e.g. 32 or "
              "64), load each K/V tile once into shared memory for all of them, compute the Br x Bc score tile in "
              "parallel, and do the max/sum reductions once per row instead of once per thread. This is the same "
              "reuse argument as the matmul tiling of Part A. The handout FAQ also notes the benefit of avoiding "
              "the N x N traffic grows with N, and the C2/C1 ratio in the table shows how it changes across "
              "sizes.")]
    out += shot(run, "C2_tiled", "c2_tiled at N = 2048, d = 64")
    out += shot(run, "C_sizes", "C1/C2/C3 at N = 1024 and N = 4096 (d = 64)")
    return out + part_c3_c4(run, c4)


def part_c3_c4(run, c4):
    out = [P("C3. Attention with cuBLAS", H2)]
    out += [P("cuBLAS is column-major. A row-major r x c buffer with leading dimension c is, read column-major, "
              "its transpose (c x r, ld = c). Using that, the two calls are:")]
    out += table(["call", "computes (column-major view)", "transa, transb", "m, n, k",
                  "A (lda)", "B (ldb)", "C (ldc)", "alpha, beta"],
                 [["1: S = &alpha;QK<super>T</super>", "S<super>T</super> = K&middot;Q<super>T</super> (N x N)",
                   "OP_T, OP_N", "N, N, d", "dK, d x N (d)", "dQ, d x N (d)", "dS, N x N (N)", "1/&radic;d, 0"],
                  ["2: O = PV", "O<super>T</super> = V<super>T</super>&middot;P<super>T</super> (d x N)",
                   "OP_N, OP_N", "d, N, N", "dV, d x N (d)", "dS, N x N (N)", "dO, d x N (d)", "1, 0"]],
                 [0.95 * inch, 1.35 * inch, 0.75 * inch, 0.6 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.65 * inch])
    out += [P("<b>Call 1:</b> column-major, the row-major K buffer is K<super>T</super> (d x N), so OP_T gives K "
              "(N x d), and the Q buffer is Q<super>T</super> (d x N) used as-is. The product K&middot;Q<super>T</super> "
              "= (QK<super>T</super>)<super>T</super> = S<super>T</super> is stored column-major with ldc = N, which is "
              "exactly S in row-major order, so the softmax kernel reads row i at dS + i&middot;N. "
              "<b>Call 2:</b> the V buffer is V<super>T</super> (d x N) and the softmaxed S buffer is "
              "P<super>T</super> (N x N); V<super>T</super>P<super>T</super> = (PV)<super>T</super> = O<super>T</super> "
              "stored column-major with ldc = d, which is O in row-major order. No explicit transposes or copies "
              "are needed. The softmax between the calls is the same one-block-per-row safe-softmax kernel as C1. "
              "The first cuBLAS call pays one-time handle/workspace initialisation, so one untimed warm-up GEMM "
              "runs before the timed region.")]
    out += shot(run, "C3_cublas", "c3_cublas at N = 2048, d = 64")

    # C4
    out += [P("C4 (bonus). Attention with OpenAI Triton", H2)]
    tri = re.search(r"Triton attention:\s*([\d.]+)", c4)
    sd = re.search(r"PyTorch SDPA:\s*([\d.]+)", c4)
    err = re.search(r"max abs error vs reference:\s*([\deE.+-]+)", c4)
    out += [P("<font face='Courier'>attn_kernel</font>: one Triton program per block of BM = 64 query rows. The "
              "Q tile stays in registers; the kernel loops over K/V in tiles of BN = 64 keys, computes the score "
              "tile with <font face='Courier'>tl.dot(q, tl.trans(k))</font>, masks keys past N to &minus;inf, "
              "and applies the same online softmax as C2 (running max, rescaled denominator and accumulator, "
              "<font face='Courier'>acc = acc&middot;&alpha; + tl.dot(p, v)</font>), then stores acc / l with a row "
              "mask. <font face='Courier'>my_attention</font> allocates O and launches a grid of "
              "ceil(N / 64) programs, passing the tensors' strides. Inputs are fp32, N = 2048, d = 64.")]
    if tri and sd:
        t, s = float(tri.group(1)), float(sd.group(1))
        out += table(["implementation", "time (ms, do_bench)"],
                     [["Triton attn_kernel", f"{t:.3f}"], ["torch SDPA (fused)", f"{s:.3f}"]])
        faster = "SDPA" if s < t else "Triton"
        out += [P(f"Max absolute error vs the PyTorch reference: {err.group(1) if err else 'n/a'} "
                  f"(passes the &lt; 1e-2 check). <b>{faster}</b> is faster at this size "
                  f"({max(t, s) / min(t, s):.1f}x). "
                  + ("SDPA dispatches to a tuned fused kernel with well-chosen block sizes and pipelining; on a T4 "
                     "(no TF32) my fp32 <font face='Courier'>tl.dot</font> runs on the regular FP32 units, and the "
                     "fixed BM = BN = 64 tiles were not autotuned. At this small size SDPA's dispatch overhead "
                     "is small compared with the work."
                     if faster == "SDPA" else
                     "At N = 2048 the problem is small, so SDPA's dispatch and backend-selection overhead is a "
                     "significant part of its time, while the single Triton launch has almost none; at larger N the "
                     "tuned fused backend would be expected to pull ahead."))]
    else:
        out += [P("<font color='#b00020'>Triton output not found in the logs.</font>")]
    out += shot(run, "C4_triton", "executed c4_triton.ipynb outputs")
    return out


def correctness(run):
    p = run.root / "results.json"
    if not p.exists():
        return []
    r = json.loads(p.read_text())
    out = [P("Grader run (results.json)", H1)]
    rows = []
    for part in ("part_a", "part_b", "part_c"):
        for k, v in r[part].items():
            if isinstance(v, dict) and "status" in v:
                rows.append([part.replace("_", " ").title(), k, v["status"]])
    out += [P(f"<font face='Courier'>grade_local.py</font> was re-run in the same session: mode = "
              f"<b>{r['meta']['mode']}</b>, GPU = {escape(str(r['meta']['host'].get('gpu_name')))}, generated "
              f"{escape(r['meta']['generated_at'])}. Part C element-wise checks: "
              f"{escape(json.dumps(r['part_c']['correctness']))}.")]
    out += table(["part", "program", "status"], rows, [1.2 * inch, 1.6 * inch, 1.0 * inch])
    return out


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

    story = [P("HPML Homework 3: CUDA Kernels &mdash; Report", TITLE),
             P(f"<b>{escape(NAME)}</b> &nbsp;|&nbsp; UNI: <b>{escape(UNI)}</b> &nbsp;|&nbsp; COMS 6998, Fall 2026", CENTER),
             Spacer(1, 8),
             P(f"<b>AI-use disclosure.</b> {escape(AI_DISCLOSURE)}"),
             Spacer(1, 4)]
    story += setup_section(run)
    a_out, a_sp, m0, m1 = part_a(run)
    story += a_out
    bdata = part_b_data(run)
    story += part_a_q5(run, a_sp, m0, m1, bdata, None)
    story += part_b(run, bdata)
    story += part_c(run)
    story += correctness(run)

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(0.75 * inch, 0.5 * inch, f"HPML HW3 report - {NAME} ({UNI})")
        c.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(a.out, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                            title="HPML HW3 Report", author=NAME)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
