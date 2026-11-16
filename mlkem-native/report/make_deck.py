"""Build report/ML-KEM_on_LEO-NG_IPU.pptx from results/*.json, the IPU kernel sources (ipu/) and mlkem-native's
C source (ref/mlkem-native, commit 948be12). Formulas are native PowerPoint equations (OMML).

Run after ./run_all.sh:   <python with python-pptx> report/make_deck.py [--eq-docx FILE]
--eq-docx also writes every equation into a .docx (same OMML), to check that the equations render.
"""
import copy
import glob
import json
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REF = ROOT / "ref" / "mlkem-native" / "mlkem" / "src"
MLKEM_NATIVE = "948be12"
EMULATOR = "4ebd3918"
CLOCK_HZ = 1e9                       # the IPU's target clock

poly = json.load(open(RES / "nx8_polyops.json"))
io = json.load(open(RES / "nx8_io.json"))
main = json.load(open(RES / "kem_ipu_main.json"))
seeds = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "kem_ipu_seeds*.json")))]
bench = json.load(open(RES / "mlkem_native_bench.json"))

# ================================================================= theme
INK = RGBColor(0x1B, 0x24, 0x30)
MUTED = RGBColor(0x5A, 0x65, 0x72)
ACCENT = RGBColor(0x0F, 0x6E, 0x8C)
OKC = RGBColor(0x1A, 0x7F, 0x37)
BAD = RGBColor(0xB0, 0x2A, 0x2A)
LINE = RGBColor(0xC9, 0xD1, 0xD9)
CODEBG = RGBColor(0xF4, 0xF6, 0xF8)
HDRBG = RGBColor(0x24, 0x31, 0x40)
ZEBRA = RGBColor(0xF2, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BODY, MONO = "Calibri", "Consolas"
CODE_PT = 11

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333

# ================================================================= OMML equations
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS_A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class E:
    """An OMML fragment (xml, with {RPI}/{RPP} run-property placeholders) and its plain-text fallback."""

    def __init__(self, xml="", txt=""):
        self.xml, self.txt = xml, txt

    def __add__(self, o):
        o = m(o) if isinstance(o, str) else o
        return E(self.xml + o.xml, self.txt + o.txt)

    def __radd__(self, o):
        return m(o) + self


def _run(t, italic):
    # words: normal text (upright, text spacing); operators and digits: upright math; one-letter names: italic
    word = len(t.strip()) > 1 and t.strip().isalpha()
    sty = "" if italic else ("<m:rPr><m:nor/></m:rPr>" if word else '<m:rPr><m:sty m:val="p"/></m:rPr>')
    return f'<m:r>{sty}{{{"RPI" if italic else "RPP"}}}<m:t xml:space="preserve">{_esc(t)}</m:t></m:r>'


_TOK = re.compile(r"[A-Za-zΑ-ω]+|[0-9][0-9.,]*|\s+|.")


def m(s):
    """Linear text to OMML: one-letter names italic; words, numbers and operators upright; spaces kept only
    next to words (so 'x mod q' keeps its spaces)."""
    toks = _TOK.findall(s)
    xml = []
    for i, t in enumerate(toks):
        if t.isspace():
            nb = [toks[j] for j in (i - 1, i + 1) if 0 <= j < len(toks)]
            if any(len(x) > 1 and x.isalpha() for x in nb):
                xml.append(_run(" ", False))
            continue
        italic = len(t) == 1 and t.isalpha()
        xml.append(_run(t, italic))
    return E("".join(xml), s)


def _e(x):
    return m(x) if isinstance(x, str) else x


def _wrap_txt(t):
    return t if len(t) <= 1 or re.fullmatch(r"[A-Za-z0-9]+", t) else f"({t})"


def sup(b, e):
    b, e = _e(b), _e(e)
    return E(f"<m:sSup><m:e>{b.xml}</m:e><m:sup>{e.xml}</m:sup></m:sSup>", f"{b.txt}^{_wrap_txt(e.txt)}")


def sub(b, s):
    b, s = _e(b), _e(s)
    return E(f"<m:sSub><m:e>{b.xml}</m:e><m:sub>{s.xml}</m:sub></m:sSub>", f"{b.txt}_{_wrap_txt(s.txt)}")


def subsup(b, s, e):
    b, s, e = _e(b), _e(s), _e(e)
    return E(f"<m:sSubSup><m:e>{b.xml}</m:e><m:sub>{s.xml}</m:sub><m:sup>{e.xml}</m:sup></m:sSubSup>",
             f"{b.txt}_{_wrap_txt(s.txt)}^{_wrap_txt(e.txt)}")


def fr(n, d):
    n, d = _e(n), _e(d)
    return E(f"<m:f><m:num>{n.xml}</m:num><m:den>{d.xml}</m:den></m:f>", f"{_wrap_txt(n.txt)}/{_wrap_txt(d.txt)}")


def dl(x, a, b):
    x = _e(x)
    return E(f'<m:d><m:dPr><m:begChr m:val="{a}"/><m:endChr m:val="{b}"/></m:dPr><m:e>{x.xml}</m:e></m:d>', f"{a}{x.txt}{b}")


def fl(x):
    return dl(x, "⌊", "⌋")


def rnd(x):                       # FIPS 203's rounding to the nearest integer
    return dl(x, "⌈", "⌋")


def par(x):
    return dl(x, "(", ")")


def brk(x):
    return dl(x, "[", "]")


def ab(x):
    return dl(x, "|", "|")


def sm(lo, hi, body):
    lo = _e(lo) if lo else E()
    hi = _e(hi) if hi else E()
    body = _e(body)
    hide = ("" if lo.xml else '<m:subHide m:val="1"/>') + ("" if hi.xml else '<m:supHide m:val="1"/>')
    return E(f'<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr>'
             f"<m:sub>{lo.xml}</m:sub><m:sup>{hi.xml}</m:sup><m:e>{body.xml}</m:e></m:nary>",
             f"sum{'_' + _wrap_txt(lo.txt) if lo.txt else ''}{'^' + _wrap_txt(hi.txt) if hi.txt else ''} {body.txt}")


def acc(x, ch, name):
    x = _e(x)
    return E(f'<m:acc><m:accPr><m:chr m:val="{ch}"/></m:accPr><m:e>{x.xml}</m:e></m:acc>', f"{x.txt}{name}")


def hat(x):
    return acc(x, "̂", "^")


def bar(x):
    return acc(x, "̅", "bar")


EQS = []          # every equation placed, for --eq-docx


def omml_runs(e, size, color, italic_font="Cambria Math"):
    col = f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
    rp = lambda i: (f'<a:rPr lang="en-US" sz="{int(size * 100)}" b="0" i="{i}">{col}'   # noqa: E731
                    f'<a:latin typeface="{italic_font}"/><a:cs typeface="{italic_font}"/></a:rPr>')
    return e.xml.replace("{RPI}", rp(1)).replace("{RPP}", rp(0))


# ================================================================= drawing helpers
MATH_TOKENS = {}


def text(slide, x, y, w, h, paras, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, bullets=False, space=6, name=None):
    """paras: str, or list of paragraphs; a paragraph is a str, an E (equation), or a list of str / (str, opts) / E."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    has_math = False
    paras = paras if isinstance(paras, list) else [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space)
        segs = p if isinstance(p, list) else [p]
        if bullets:
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.22))))
            pPr.set("indent", str(-int(Inches(0.2))))
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
        for sgm in segs:
            if isinstance(sgm, E):
                tok = f"{len(MATH_TOKENS)}"
                MATH_TOKENS[tok] = (sgm, size, color)
                EQS.append(sgm)
                has_math = True
                r = para.add_run()
                r.text = tok
                r.font.size = Pt(size)
                continue
            t, o = (sgm, {}) if isinstance(sgm, str) else sgm
            r = para.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.name = o.get("font", font)
            f.color.rgb = o.get("color", color)
    if has_math:
        tb.name = "MathText"
    elif name:
        tb.name = name
    return tb


def to_alternate_content(sp):
    """Wrap a text shape holding equation tokens as PowerPoint does: mc:AlternateContent whose Choice
    (Requires a14) has the a14:m equations and whose Fallback has the same shape with plain text."""
    choice, fall = copy.deepcopy(sp), copy.deepcopy(sp)
    A = f"{{{NS_A}}}"
    for para in choice.iter(A + "p"):
        runs = para.findall(A + "r")
        real = [r for r in runs if (r.findtext(A + "t") or "").strip()]
        for r in runs:
            t = r.findtext(A + "t") or ""
            if t not in MATH_TOKENS:
                continue
            e, size, color = MATH_TOKENS[t]
            body = omml_runs(e, size, str(color))
            if len(real) == 1:
                inner = (f'<m:oMathPara><m:oMathParaPr><m:jc m:val="left"/></m:oMathParaPr>'
                         f"<m:oMath>{body}</m:oMath></m:oMathPara>")
            else:
                inner = f"<m:oMath>{body}</m:oMath>"
            el = etree.fromstring(f'<a14:m xmlns:a14="{NS_A14}" xmlns:m="{NS_M}" xmlns:a="{NS_A}">{inner}</a14:m>')
            r.addprevious(el)
            para.remove(r)
    for r in fall.iter(A + "r"):
        t = r.find(A + "t")
        if t is not None and t.text in MATH_TOKENS:
            e, size, color = MATH_TOKENS[t.text]
            t.text = e.txt
            rPr = r.find(A + "rPr")
            rPr.set("i", "0")
            sf = etree.SubElement(rPr, A + "solidFill")
            etree.SubElement(sf, A + "srgbClr").set("val", str(color))
            etree.SubElement(rPr, A + "latin").set("typeface", "Cambria Math")
    ac = etree.fromstring(f'<mc:AlternateContent xmlns:mc="{NS_MC}"><mc:Choice xmlns:a14="{NS_A14}" Requires="a14"/>'
                          f"<mc:Fallback/></mc:AlternateContent>")
    ac[0].append(choice)
    ac[1].append(fall)
    sp.addprevious(ac)
    sp.getparent().remove(sp)


def title(slide, t, sub_=None):
    text(slide, 0.5, 0.28, W - 1.0, 0.7, t, size=28, bold=True, color=INK)
    if sub_:
        text(slide, 0.5, 0.92, W - 1.0, 0.4, sub_, size=14, color=MUTED)


def box(slide, x, y, w, h, fill=CODEBG, line=LINE):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    s.line.width = Pt(0.75)
    s.shadow.inherit = False
    return s


def label(slide, x, y, w, t, color=ACCENT):
    text(slide, x, y, w, 0.3, t, size=11, bold=True, color=color)


def code(slide, x, y, w, h, hdr, lines, size=CODE_PT):
    label(slide, x, y, w, hdr)
    box(slide, x, y + 0.3, w, h - 0.3)
    text(slide, x + 0.08, y + 0.36, w - 0.16, h - 0.42, lines, size=size, font=MONO, space=0)


def table(slide, x, y, w, rows, colw, size=12, rowh=0.32, bold_first=False, right_from=1):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(rowh * nr)).table
    for j, cw in enumerate(colw):
        gt.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        gt.rows[i].height = Inches(rowh)
        for j, val in enumerate(row):
            c = gt.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            col = INK
            if isinstance(val, tuple):
                val, col = val
            lines = str(val).split("\n")
            for k, ln in enumerate(lines):
                p = c.text_frame.paragraphs[0] if k == 0 else c.text_frame.add_paragraph()
                r = p.add_run()
                r.text = ln
                r.font.size = Pt(size if k == 0 else size - 1.5)
                r.font.name = BODY
                if i == 0:
                    r.font.bold = True
                    r.font.color.rgb = WHITE
                else:
                    r.font.color.rgb = col if k == 0 else MUTED
                    r.font.bold = bold_first and j == 0
                if j >= right_from and i > 0 and re.fullmatch(r"[-+0-9.,]+( \S+)?|[0-9.,]+ [µm]?s|—", ln.strip()):
                    p.alignment = PP_ALIGN.RIGHT
            c.fill.solid()
            c.fill.fore_color.rgb = HDRBG if i == 0 else (ZEBRA if i % 2 == 0 else WHITE)
    return gt


def fmt(n):
    return f"{n:,.0f}"


def at1g(c):
    """Time at the 1 GHz target clock."""
    s = c / CLOCK_HZ
    if s >= 1e-4:
        return f"{s * 1e6:,.1f} µs"
    if s >= 1e-6:
        return f"{s * 1e6:.2f} µs"
    return f"{s * 1e9:.0f} ns"


def cyc(c):
    return f"{fmt(c)} cycles ({at1g(c)})"


# ================================================================= source excerpts
WRAP_C, WRAP_PY, WRAP_FIPS = 75, 80, 54
C_PT = 10.5


def wrap_line(l, width):
    """Break a long code line at a space where possible; continuation lines are indented."""
    l = l.rstrip()
    if len(l) <= width:
        return [l]
    lead = len(l) - len(l.lstrip())
    ind = " " * (lead + 4)
    out = []
    while len(l) > width:
        cut = l.rfind(" ", lead + 4, width + 1)
        if cut <= lead:
            cut = width
        out.append(l[:cut].rstrip())
        l = ind + l[cut:].lstrip()
    out.append(l)
    return out


def c_func(rel, pattern, keep=None, nmax=30):
    """An mlkem-native function from the source (contracts, comments and blank lines dropped).
    keep: list of (first marker, last marker) line ranges of the body to show; the rest becomes '…'."""
    lines = open(REF / rel).read().split("\n")
    start = next(i for i, l in enumerate(lines) if re.search(pattern, l) and not l.rstrip().endswith(";"))
    out, depth, seen, inspec, i = [], 0, False, 0, start
    while i < len(lines):
        l = lines[i]
        s = l.strip()
        if s.startswith(("__contract__", "__loop__")):
            inspec = 1
        if inspec:
            inspec += l.count("(") - l.count(")")
            if inspec <= 1 and s.endswith(")"):
                inspec = 0
            i += 1
            continue
        if s.startswith(("mlk_assert", "MLK_CT_TESTING")):      # debug/CT-test statements, possibly multi-line
            while not lines[i].rstrip().endswith(";"):
                i += 1
            i += 1
            continue
        depth += l.count("{") - l.count("}")
        seen = seen or "{" in l
        l = re.sub(r"\s*/\*.*?\*/", "", l.replace("MLK_STATIC_TESTABLE ", ""))
        if not (l.strip() == "" or s.startswith(("/*", "*", "//", "#"))):
            out.append(l.rstrip())
        if seen and depth <= 0:
            break
        i += 1
    # clang-format aligns continuation lines far to the right: indent them 4 past their statement instead
    norm, stmt_ind, cont = [], 0, False
    for l in out:
        ind = len(l) - len(l.lstrip())
        if cont and not l.lstrip().startswith(("{", "}")):
            l = " " * (stmt_ind + 4) + l.lstrip()
        else:
            stmt_ind = ind
        norm.append(l)
        cont = not l.rstrip().endswith((";", "{", "}", ")"))
    out = norm
    if keep:
        body_start = next(k for k, l in enumerate(out) if l.strip() == "{") + 1
        sel = list(range(body_start))
        for a, b in keep:
            ka = next(k for k, l in enumerate(out) if k >= body_start and a in l)
            kb = next(k for k, l in enumerate(out) if k >= ka and b in l) if b else ka
            if sel and sel[-1] != ka - 1:
                sel.append(None)
            sel.extend(range(ka, kb + 1))
        if sel[-1] != len(out) - 2:
            sel.append(None)
        sel.append(len(out) - 1)
        res = []
        for k, n in enumerate(sel):
            if n is None:
                nxt = next((out[q] for q in sel[k + 1:] if q is not None), "")
                res.append(" " * (len(nxt) - len(nxt.lstrip())) + "…")
            else:
                res.append(out[n])
        out = res
    wrapped = [w for l in out for w in wrap_line(l, WRAP_C)]
    if len(wrapped) > nmax:
        wrapped = wrapped[:nmax - 2] + ["  …", "}"]
    return f"mlkem-native {rel}:{start + 1}", wrapped


def py_src(rel, pieces, nmax=29):
    """Lines of this repository's source. pieces: (first marker, last marker or line count); the function's
    docstring is dropped, comments are kept; gaps between pieces become '…'."""
    lines = open(ROOT / rel).read().split("\n")
    out, first_no, pos, funcs = [], None, 0, []
    for pi, (a, b) in enumerate(pieces):
        ka = next(k for k, l in enumerate(lines) if k >= pos and a in l)
        ind0 = len(lines[ka]) - len(lines[ka].lstrip())
        for k in range(ka, -1, -1):                  # the function this piece belongs to
            mm = re.match(r"(\s*)def (\w+)", lines[k])
            if mm and (k == ka or len(mm.group(1)) < ind0):
                if mm.group(2) not in funcs:
                    funcs.append(mm.group(2))
                break
        if isinstance(b, int):
            kb = ka + b - 1
        else:
            kb = next(k for k, l in enumerate(lines) if k > ka and b in l)
        first_no = ka + 1 if first_no is None else first_no
        pos = kb + 1
        seg, indoc = [], False
        for l in lines[ka:kb + 1]:
            st_ = l.strip()
            if st_.startswith('"""') or indoc:
                if st_.count('"""') == 1:
                    indoc = not indoc
                continue
            if st_:
                seg.append(l.rstrip())
        if out:
            nxt = seg[0]
            out.append(" " * (len(nxt) - len(nxt.lstrip())) + "…")
        out.extend(seg)
    ind = min(len(l) - len(l.lstrip()) for l in out if l.strip() and l.strip() != "…")
    out = [l[ind:] if len(l) - len(l.lstrip()) >= ind else l.lstrip() for l in out]
    wrapped = [w for l in out for w in wrap_line(l, WRAP_PY)]
    if len(wrapped) > nmax:
        raise SystemExit(f"excerpt too long for a slide: {rel} {pieces[0][0][:40]} ({len(wrapped)} lines)")
    return f"{rel}:{first_no}", wrapped, funcs


def fips(lines):
    return [w for l in lines for w in wrap_line(l, WRAP_FIPS)]


# ================================================================= results
OPS = ("keygen", "encaps", "decaps")
DISP = {"keygen": "KeyGen", "encaps": "Encaps", "decaps": "Decaps"}
TOT = {op: main["ops"][op]["total"] for op in OPS}
CAT = {op: main["ops"][op]["cycles"] for op in OPS}
LAUNCH = {op: main["ops"][op]["launches"] for op in OPS}
PK = {op: main["ops"][op]["per_kernel"] for op in OPS}
allres = dict(main["results"])
for s_ in seeds:
    allres.update(s_["results"])
NSEEDS = sum(1 for k in allres if k.startswith("seed"))
ok_poly, ok_io = poly["results"], io["results"]
assert all(allres.values()) and all(ok_poly.values()) and all(ok_io.values()), "a check failed: run ./run_all.sh"
assert bench["ipu"] == TOT, "results/mlkem_native_bench.json IPU row differs from results/kem_ipu_main.json"


def kern(op, *names, prefix=False, contains=False):
    """(cycles, launches, largest program words) of kernels of one operation."""
    c = n = w = 0
    for k, (cy, la, cat, wd) in PK[op].items():
        if (any(k.startswith(x) for x in names) if prefix else any(x in k for x in names) if contains else k in names):
            c, n, w = c + cy, n + la, max(w, wd)
    assert n, (op, names)
    return c, n, w


def nperm(op):
    return kern(op, "permute")[1]


PERM = kern("keygen", "permute")
PERM1 = PERM[0] // PERM[1]
assert all(kern(op, "permute")[0] == PERM1 * nperm(op) for op in OPS)


def result_box(slide, x, y, w, h, exact, compared, kem_line, words):
    box(slide, x, y, w, h, fill=WHITE, line=ACCENT)
    text(slide, x + 0.1, y + 0.05, w - 0.2, h - 0.1, [
        [("Bit-exact: ", {"bold": True}), ("yes" if exact else "NO", {"bold": True, "color": OKC if exact else BAD}),
         (f" ({compared})", {})],
        [("In the KEM: ", {"bold": True}), (kem_line, {})],
        [("Program: ", {"bold": True}), (words, {})]], size=11.5, space=3)


def step(name, sub_a, fips_hdr, fips_lines, spec_math, cref, ipu, method, res, sub_b=None):
    """Two slides per ML-KEM step: (1) FIPS 203 pseudocode + its formulas, mlkem-native C;
    (2) the IPU kernel source, the method's formulas, and the measured result."""
    s = prs.slides.add_slide(BLANK)
    title(s, name, sub_a)
    fl_ = fips(fips_lines)
    hf = 0.45 + 0.2 * len(fl_)
    code(s, 0.5, 1.4, 5.3, hf, fips_hdr, fl_)
    if spec_math:
        label(s, 0.5, 1.55 + hf, 5.3, "Formulas")
        text(s, 0.5, 1.87 + hf, 5.3, 5.4 - hf, spec_math, size=16, space=8)
    code(s, 5.95, 1.4, 6.9, 5.85, f"{cref[0]} (commit {MLKEM_NATIVE})", cref[1], size=C_PT)
    s = prs.slides.add_slide(BLANK)
    if sub_b is None:
        fns = ipu[2]
        sub_b = f"Kernel builder{'s' if len(fns) > 1 else ''} {', '.join(fns)} in {ipu[0].split(':')[0]}"
    title(s, f"{name} on the IPU", sub_b)
    code(s, 0.5, 1.4, 7.85, 5.85, ipu[0], ipu[1])
    label(s, 8.55, 1.4, 4.3, "Method")
    text(s, 8.55, 1.72, 4.3, 3.6, method, size=13, space=6)
    result_box(s, 8.55, 5.4, 4.3, 1.85, *res)


# ================================================================= 1 title
s = prs.slides.add_slide(BLANK)
box(s, 0, 0, W, 7.5, fill=HDRBG, line=HDRBG)
text(s, 0.8, 2.0, 11.7, 1.0, "ML-KEM-768 on the LEO-NG IPU", size=44, bold=True, color=WHITE)
text(s, 0.8, 3.1, 11.7, 1.0, "KeyGen, Encaps and Decaps run entirely on the IPU in NX8, bit-exact against mlkem-native",
     size=22, color=RGBColor(0xF0, 0xB4, 0x4C))
text(s, 0.8, 4.6, 11.7, 1.4, [
    f"Cycles: bit-accurate IPU emulator (ipu-emulator {EMULATOR}). Times: the IPU's target clock, 1 GHz.",
    f"Reference: mlkem-native {MLKEM_NATIVE} (github.com/pq-code-package/mlkem-native). Specification: FIPS 203.",
    "Reproduce: mlkem-native/run_all.sh (10-15 minutes, CPU only)."], size=15, color=RGBColor(0xDD, 0xE3, 0xEA), space=6)

# ================================================================= 2 status and results
s = prs.slides.add_slide(BLANK)
title(s, "Status and results", "ML-KEM-768; cycles on the emulated IPU, times at the 1 GHz target clock")
table(s, 0.5, 1.45, 12.33, [
    ["Implementation", "Reference", "Status"],
    ["Unprotected ML-KEM-768", f"mlkem-native {MLKEM_NATIVE}\ngithub.com/pq-code-package/mlkem-native",
     ("Done: KeyGen, Encaps and Decaps entirely on the IPU in NX8 under the host rule; bit-exact on every test", OKC)]], [3.0, 4.6, 4.73], size=13, rowh=0.62, bold_first=True, right_from=9)
rows = [["Operation", "Cycles", "Time at 1 GHz", "Kernel launches", "Keccak-f[1600] permutations", "Keccak share of cycles"]]
for op in OPS:
    rows.append([DISP[op], fmt(TOT[op]), at1g(TOT[op]), str(LAUNCH[op]), str(nperm(op)),
                 f"{100 * CAT[op]['keccak'] / TOT[op]:.0f}%"])
table(s, 0.5, 3.75, 12.33, rows, [2.0, 1.8, 2.0, 2.0, 2.6, 1.93], size=14, rowh=0.45, bold_first=True)
text(s, 0.5, 5.75, 12.33, 1.5, [
    f"Correctness: the main mlkem-native vector, {NSEEDS} random seeds, 3 crafted ciphertexts (implicit rejection) and 2 invalid inputs "
    "(ek with a coefficient ≥ q, dk with a wrong H(ek)): every output byte and the status byte match mlkem-native.",
    "Host rule: the host writes constants and input bytes before the first kernel launch and reads output bytes after the last; "
    "between launches it only loads programs and registers. Measured host XMEM accesses between launches: "
    f"{main['host_log_counts']['during']}."], size=13, bullets=True, space=5)

# ================================================================= 3 where the cycles go
s = prs.slides.add_slide(BLANK)
title(s, "Where the cycles go", "Cycles per operation by kernel category (results/kem_ipu_main.json)")
CATS = [("keccak", "Keccak: permutations, absorb, squeeze"), ("sampling", "Sampling: SampleNTT, CBD, centring of Â"),
        ("encode", "Encoding and checks: bytes, bits, fields, ek/dk checks, status"), ("poly", "Polynomial arithmetic"),
        ("fo", "FO compare and select (Decaps)")]
rows = [["Category"] + [f"{DISP[o]} cycles" for o in OPS] + [f"{DISP[o]} at 1 GHz" for o in OPS]]
for k, nm in CATS:
    rows.append([nm] + [fmt(CAT[o].get(k, 0)) if CAT[o].get(k) else "—" for o in OPS]
                + [at1g(CAT[o][k]) if CAT[o].get(k) else "—" for o in OPS])
rows.append(["Total"] + [fmt(TOT[o]) for o in OPS] + [at1g(TOT[o]) for o in OPS])
table(s, 0.5, 1.45, 12.33, rows, [4.53, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], size=13, rowh=0.48, bold_first=True)
text(s, 0.5, 5.0, 12.33, 2.2, [
    f"One Keccak-f[1600] permutation takes {fmt(PERM1)} cycles ({at1g(PERM1)}): {nperm('keygen')}, {nperm('encaps')} and "
    f"{nperm('decaps')} permutations make up most of KeyGen, Encaps and Decaps.",
    "SampleNTT always squeezes 5 SHAKE128 blocks per matrix entry (FIPS 203 Appendix B bound), so the cycle count does not "
    f"depend on the input: the {NSEEDS} random seeds give the same totals as the main vector.",
    f"Kernels: {main['kernels']} programs, the largest {max(main['words'].values())} of 1,024 instruction words."],
    size=13, bullets=True, space=6)

# ================================================================= 4 host rule
s = prs.slides.add_slide(BLANK)
title(s, "The host rule", "What the host may do around one operation, and how it is enforced")
hl = main["host_log_counts"]
table(s, 0.5, 1.45, 12.33, [
    ["Phase", "Host access to XMEM", "Main test run (8 operations)"],
    ["Before the first launch", "Write every kernel's constant rows (independent of the input) and the input bytes (d, z, m, ek, dk, ct) as U8 rows",
     f"{fmt(hl['before'])} row writes"],
    ["Between launches", "None. Only the next program (IMEM), its control registers (CRs) and loop registers (LRs) are loaded",
     f"{hl['during']}"],
    ["After the last launch", "Read the output bytes (ek, dk, ct, ss) and one status byte", f"{fmt(hl['after'])} row reads"]],
    [2.6, 7.13, 2.6], size=13, rowh=0.62, bold_first=True, right_from=9)
text(s, 0.5, 4.2, 12.33, 3.0, [
    "Enforcement: emu.StrictSession logs every host read and write of XMEM with its phase and raises an error for any access "
    "between launches. test_kem_ipu.py asserts the count is 0.",
    "Every intermediate value (bits, hashes, samples, polynomials, the FO check and the input checks) is produced by IPU kernels.",
    "Branches and addresses come only from CRs and LRs, never from data. Data-dependent choices are arithmetic: compare with "
    "a constant, multiply, select.",
    "The 8 operations of the main test: KeyGen, Encaps, Decaps, Decaps of 3 crafted ciphertexts, Encaps with an invalid ek, "
    "Decaps with an invalid dk."], size=13, bullets=True, space=6)

# ================================================================= 5 correctness
s = prs.slides.add_slide(BLANK)
title(s, "Correctness checks", "Reference values: mlkem-native compiled for the host (refgen/), and hashlib for SHA-3")
BX = ("Bit-exact", OKC)
table(s, 0.5, 1.45, 12.33, [
    ["Check", "What is compared", "Result"],
    ["Polynomial kernels", f"{len(ok_poly)} checks: mlkem-native vectors (NTT, NTT⁻¹, reduce, tomont, CBD: 16 each; add, sub: 8; basemul: 4; "
     "mulcache: 12 polynomials); all 65,536 int16 (Barrett); all 3,329 inputs (compress); inputs ±(q−1); zero inputs", BX],
    ["I/O, hash and sampling kernels", f"{len(ok_io)} checks: BytesToBits, ByteDecode, ByteEncode (d = 4, 10, 12), BitsToBytes, CBD (η = 2, 3), "
     "SHA3-256/512 and SHAKE128/256 (hashlib), SampleNTT (30 random streams, 5 crafted)", BX],
    ["Keccak-f[1600]", "4 mlkem-native permutation vectors, as 24 one-round programs and as one looped program", BX],
    ["ML-KEM-768 main vector", "ek, dk, ct, ss from Encaps, ss from Decaps, status byte", BX],
    ["Crafted ciphertexts", "flipped bit, c₁ = 0, all-zero ct: implicit-rejection key from Decaps", BX],
    ["Invalid inputs", "ek with a coefficient ≥ q (mlkem-native returns −4), dk with a wrong H(ek) (−5): outputs 0, status 0", BX],
    ["Random seeds", f"{NSEEDS} seeds: KeyGen, Encaps, Decaps; expected values from mlkem-native", BX],
    ["Host rule", "host XMEM accesses between launches", (f"{hl['during']}", OKC)]],
    [2.8, 8.03, 1.5], size=12, rowh=0.5, bold_first=True, right_from=9)

# ================================================================= 6 the IPU in NX8
s = prs.slides.add_slide(BLANK)
title(s, "The IPU and the NX8 format", "A 128-lane VLIW multiply-accumulate engine: one instruction word per cycle, six slots per word")
table(s, 0.5, 1.45, 7.2, [
    ["Slot", "What it does"],
    ["COND", "branch and HALT; loops are fixed (LR counters), never data-dependent"],
    ["LR", "3 loop registers: INC, DEC, SET; supply addresses and element indices"],
    ["LOAD", "one row per word: LD.MULT to R0/R1, LD.MULT.CYCLIC to a slot of R_CYCLIC, LD.MASK"],
    ["MULT", "128 products: a window of R_CYCLIC (RC) or R0/R1 times a row (VV), a CR constant (VE) or itself (VS); EE: one element times a CR constant, to any lanes"],
    ["ACC", "R_ACC (128 FP32) += / −= products; AGG.SUM adds all 128 products into one element"],
    ["AaQ", "activation (identity, relu, relu6), quantise R_ACC to one NX8 row and store it"]],
    [1.0, 6.2], size=12, rowh=0.5, bold_first=True, right_from=9)
table(s, 7.95, 1.45, 4.88, [
    ["Row format", "Element"],
    ["U8", "unsigned integer 0..255, exact: bytes"],
    ["E4 (E4M3)", "4-bit exponent: bits 0/1 and weights 2ᵗ"],
    ["S8", "k·2^(u−6), |k| ≤ 127, 2^u = binade of the row's largest |x|: limb rows"]],
    [1.25, 3.63], size=12, rowh=0.62, bold_first=True, right_from=9)
text(s, 7.95, 4.15, 4.88, 1.2, [
    "R_CYCLIC: 4 slots of 128 elements read as a ring of 512; a MULT reads a 128-element window at any offset (shifts are free).",
    "XMEM: 16,384 NX8 rows (2 MB). IMEM: 1,024 instruction words."], size=12, bullets=True, space=4)
label(s, 0.5, 5.5, 12.33, "Not available, and what replaces it")
text(s, 0.5, 5.8, 12.33, 1.6, [
    "No integer, bitwise, shift, rotate or compare instructions on vectors: integers are exact FP32 sums; XOR is a square (next slides); "
    "shifts and rotations are window offsets; comparisons are roundings of scaled values.",
    "No path from R_ACC into a multiplier: a value computed in R_ACC is stored as NX8 rows and reloaded before it is multiplied.",
    "No data-indexed addressing: selection and compaction (SampleNTT, FO select) are arithmetic."], size=12.5, bullets=True, space=4)

# ================================================================= 7 exact integer arithmetic
s = prs.slides.add_slide(BLANK)
title(s, "Exact integer arithmetic in NX8", "How every kernel gets exact integer results from 8-bit rows and an FP32 accumulator")
X = [
    [("Exact sums. ", {"bold": True}), "An 8-bit × 8-bit product is exact in FP32; a sum stays exact while every partial sum is an integer below ",
     sup("2", "24"), ". Kernel builders assert this bound for every accumulation."],
    [("Split store (integers wider than 8 bits). ", {"bold": True}), "A value is held as limb rows whose sum is the value:"],
    m("x = ") + sub("ℓ", "1") + m(" + ") + sub("ℓ", "2") + m(" + …,   ") + sub("ℓ", "1") + m(" = Q(x),   ")
    + sub("ℓ", "2") + m(" = Q(x − ") + sub("ℓ", "1") + m("),   Q(x) = k·") + sup("2", "u−6") + m(",  ") + ab("k") + m(" ≤ 127"),
    ["Q is the store's own rounding to the row grid. ", ab("x") + m(" < ") + sup("2", "14"), " needs 2 limbs, ",
     ab("x") + m(" < ") + sup("2", "24"), " needs 4. Each extra limb costs one MULT and one reload."],
    [("Rounding to an integer. ", {"bold": True}), "With M = 1.5·", sup("2", "23"), " and ", ab("y") + m(" < ") + sup("2", "22"), ":"],
    rnd("y") + m(" = (y + M) − M"),
    [("Quotient and residue. ", {"bold": True}), "Reduction mod q, Barrett, compress and the NTT layers all compute"],
    m("K = ") + rnd(sm("i", "", sub("d", "i") + fr(sub("c", "i"), "q")) + m(" + β")) + m(",     r = y − qK,     y = ")
    + sm("i", "", sub("d", "i") + sub("c", "i")),
    ["with the constants ", fr(sub("c", "i"), "q"), " split into 8-bit limbs. The build bounds the error ε of the rounded sum: ",
     ab("ε") + m(" < ") + fr("1", "4q"), " where the rounding decides the result (q is odd: no ties), ",
     ab("ε") + m(" < 0.02"), " where K only keeps values bounded. If one pass cannot meet ", fr("1", "4q"), ":"],
    m("c = ") + sub("K", "0") + m(" + ") + rnd(fr(sub("r", "0"), "q")) + m(",     ") + sub("r", "0") + m(" = y − q") + sub("K", "0")
    + m("  (exact),  ") + sub("K", "0") + m(" approximate"),
]
text(s, 0.5, 1.4, 12.33, 5.9, X, size=14, space=5)

# ================================================================= 8 bits and Keccak state
s = prs.slides.add_slide(BLANK)
title(s, "Bits without bitwise instructions", "Bit rows, XOR as a square, and the Keccak state encoding")
X = [
    [("Bit rows. ", {"bold": True}), "A bit string is stored one bit per element (E4 rows, 128 bits per row), in FIPS 203 bit order."],
    [("Boolean operations on bits ", {"bold": True}), "a, b ∈ {0, 1}:"],
    m("a ⊕ b = ") + sup("(a − b)", "2") + m(",     a ∧ b = a·b,     ¬ a = 1 − a"),
    [("Keccak state. ", {"bold": True}), "15 rows hold the 1,600 state bits (lane (x, y) in row 3x + ⌊y/2⌋, half y mod 2). "
     "Each element stores S with the state bit A = ", sup("S", "2"), ", S ∈ {−1, 0, 1}. A MULT.VS squares a shifted, masked window "
     "(rotation and lane moves are window offsets), so XOR costs one product:"],
    m("absorb:   S′ = ") + sup("S", "2") + m(" − m   ⇒   ") + sup("S′", "2") + m(" = A ⊕ m"),
    m("χ:   S′ = ") + sub("b", "0") + m(" − ") + sub("b", "2") + m(" + ") + sub("b", "1") + sub("b", "2") + m("   ⇒   ")
    + sup("S′", "2") + m(" = ") + sub("b", "0") + m(" ⊕ (¬ ") + sub("b", "1") + m(" ∧ ") + sub("b", "2") + m(")"),
    [("Bit of an integer. ", {"bold": True}), "For y = B·", sup("2", "−t"), " (one product with a weight row), with ",
     sub("round", "2"), " = rounding to an even integer (magic constant 1.5·", sup("2", "24"), "):"],
    m("−2") + fl(fr("y", "2")) + m(" = ") + sub("round", "2") + par(m("−y + 1 − ") + sup("2", "−9"))
    + m(",     r = y − 2") + fl(fr("y", "2")) + m(",     bit = ") + rnd(m("r − ") + fr("1", "2") + m(" + ") + sup("2", "−9")),
    [("Data-dependent flags. ", {"bold": True}), "A comparison is a rounding of a scaled value, e.g. for 0 ≤ c < 4,096:"],
    brk(m("c ≥ q")) + m(" = ") + fl(fr("c + 767", "4096")),
]
text(s, 0.5, 1.4, 12.33, 5.9, X, size=14, space=5)

# ================================================================= per-step slides
PS, IS = poly["stats"], io["stats"]

# 1 Keccak-f
step("Keccak-f[1600]", "FIPS 202 permutation, used by every FIPS 203 hash function (Section 4.1)",
     "FIPS 202 Keccak-p[1600, 24] (3.2.1-3.2.5)", [
         "for ir = 0 .. 23:",
         "  θ:    C[x] ← A[x,0] ⊕ A[x,1] ⊕ … ⊕ A[x,4]",
         "        D[x] ← C[x−1] ⊕ rot(C[x+1], 1)",
         "        A[x,y] ← A[x,y] ⊕ D[x]",
         "  ρ, π: B[y, 2x+3y] ← rot(A[x,y], r[x,y])",
         "  χ:    A[x,y] ← B[x,y] ⊕ (¬B[x+1,y] ∧ B[x+2,y])",
         "  ι:    A[0,0] ← A[0,0] ⊕ RC[ir]"],
     [m("χ:  ") + sub("a", "x,y") + m(" = ") + sub("b", "x,y") + m(" ⊕ (¬ ") + sub("b", "x+1,y") + m(" ∧ ") + sub("b", "x+2,y") + m(")"),
      "State: 25 lanes of 64 bits; rot is a cyclic shift of a lane."],
     c_func("fips202/keccakf1600.c", r"void mlk_keccakf1600_permute_c\(",
            keep=[("for (round = 0", "Du = BCo ^"), ("Aba ^= Da;", "Eba ^= (uint64_t)mlk_KeccakF_RoundConstants[round];")]),
     py_src("ipu/kernels/keccak.py", [("    # ---- theta: C", "        T3[x] = t3"), ("    # ---- chi (+ iota on (0,0))", 12)]),
     [["State in 15 rows, element S with A = ", sup("S", "2"), " (slide “Bits without bitwise instructions”). One program is one round; "
       "a hardware loop of 24 iterations is the permutation, reading ι's round-constant row with stride 1."],
      ["θ and ρ+π: masked, shifted MULT.VS squares: ", m("a ⊕ b = ") + sup("(a − b)", "2"), "."],
      ["χ: 3 products per row (MULT.RC.VE, MULT.VE, MULT.RC.VV):"],
      m("S′ = ") + sub("b", "0") + m(" − ") + sub("b", "2") + m(" + ") + sub("b", "1") + sub("b", "2")],
     (True, "4 mlkem-native permutation vectors; every hash of the KEM",
      f"{cyc(PERM1)} per permutation; {PERM[1]} permutations in KeyGen", f"{PERM[2]} instruction words (one round, looped 24 times)"))

# 2 absorb
ab_c, ab_n, ab_w = kern("encaps", ".absorb", contains=True)
step("Sponge: absorb and padding", "FIPS 202 sponge; inputs of FIPS 203 G, H, J, PRF and XOF (Section 4.1)",
     "FIPS 202 SPONGE[f, pad10*1, r] / FIPS 203 (4.2)-(4.6)", [
         "P ← M ‖ pad10*1  (SHA3: 01, SHAKE: 1111 before pad)",
         "for each r-bit block P_i of P:",
         "    S ← f(S ⊕ (P_i ‖ 0^(1600−r)))",
         "inputs:  G(d ‖ k)          H(ek)",
         "         PRF_η(σ, N)       XOF(ρ ‖ j ‖ i)",
         "         G(m ‖ H(ek))       J(z ‖ c)"],
     [m("S ← f(S ⊕ ") + sub("P", "i") + m(")"),
      m("rate r = 1088 (SHA3-256, SHAKE256), 576 (SHA3-512), 1344 (SHAKE128) bits")],
     c_func("fips202/fips202.c", r"static void mlk_keccak_absorb_once\("),
     py_src("ipu/kernels/nx8io.py", [("def absorb(scratch, state_base, lanes):", "    return nb.kernel()")]),
     [["Input: whole bit rows (every variable part of a hash input is a multiple of 128 bits) and constant rows for "
       "the domain bytes (k, N, j, i), the domain-separation bits and the final pad bit, built before the first launch."],
      ["Per state row: MULT.VS squares the state window, each 64-bit message half is subtracted with a masked MULT.VE:"],
      m("S′ = ") + sup("S", "2") + m(" − m   ⇒   ") + sup("S′", "2") + m(" = A ⊕ m"),
      ["A hash is a fixed kernel sequence: zero the state, then absorb and permute per block."]],
     (True, "SHA3-256/512, SHAKE128/256 against hashlib; every hash of the KEM",
      f"{ab_n} absorb launches in Encaps, {fmt(ab_c)} cycles in total ({at1g(ab_c)})", f"{ab_w} instruction words at most"))

# 3 squeeze
ex_c, ex_n, ex_w = kern("encaps", ".extract", contains=True)
step("Sponge: squeeze and the hash functions", "FIPS 202 squeeze; FIPS 203 H, G, J, PRF, XOF (Section 4.1)",
     "FIPS 203 (4.2)-(4.6) / FIPS 202 squeeze", [
         "Z ← first r bits of S",
         "while |Z| < ℓ:  S ← f(S);  Z ← Z ‖ first r bits of S",
         "",
         "H(s)        = SHA3-256(s)",
         "G(c)        = SHA3-512(c), two 32-byte halves",
         "J(s)        = SHAKE256(s, 8·32)",
         "PRF_η(s, b) = SHAKE256(s ‖ b, 8·64η)",
         "XOF         = SHAKE128 (SampleNTT reads 840 bytes)"],
     [m("Z = ") + sub("S", "0..r−1") + m(" ‖ ") + sub("f(S)", "0..r−1") + m(" ‖ …")],
     c_func("fips202/fips202.c", r"static void mlk_keccak_squeezeblocks\("),
     py_src("ipu/kernels/nx8io.py", [("def extract(scratch, state_base, outs):", "    return nb.kernel()"),
                                    ("def sponge_specs(plan, state_base, out_base):", '        specs.append(("permute", None))')]),
     [["Output bit row g/2, half g mod 2, takes state lane g of the current block: one masked MULT.VS stores ",
       sup("S", "2"), ", the plain bit."],
      ["All output lengths are fixed: H and G one block each; J 32 bytes; PRF 128 bytes (one block of 136); "
       "XOF 840 bytes = 5 blocks of 168. Each further block costs one permutation."]],
     (True, "SHA3-256/512, SHAKE128/256 against hashlib; every hash of the KEM",
      f"{ex_n} squeeze launches in Encaps, {fmt(ex_c)} cycles in total ({at1g(ex_c)})", f"{ex_w} instruction words at most"))

# 4 BytesToBits
bb = kern("decaps", "unpack_dk")
step("BytesToBits", "FIPS 203 Algorithm 4; the inputs d, z, m, ek, dk, ct arrive as bytes",
     "FIPS 203 Algorithm 4 BytesToBits(B)", [
         "1: C ← B",
         "2: for (i ← 0; i < ℓ; i++)",
         "3:   for (j ← 0; j < 8; j++)",
         "4:     b[8i + j] ← C[i] mod 2",
         "5:     C[i] ← C[i] / 2",
         "8: return b"],
     [sub("b", "8i+j") + m(" = ") + fl(fr(sub("B", "i"), sup("2", "j"))) + m(" mod 2"),
      "mlkem-native has no separate BytesToBits: the Keccak state is XORed with bytes, and the decoders unpack bits with shifts and masks."],
     c_func("fips202/keccakf1600.c", r"void mlk_keccakf1600_xor_bytes\("),
     py_src("ipu/kernels/nx8io.py", [("def _bits_from_rep(nb, reps, dst):", "    nb.store(dst, st_qf=QFE)"),
                                    ("def bytes_to_bits(scratch, byte_base, n_byte_rows, bit_base):", 1),
                                    ("    for q in range(8):", "    return nb.kernel()")]),
     [["1) 16 masked MULT.EE copy byte 16q+g of a U8 row to elements 8g..8g+7 (q = 0..7 covers 128 bytes)."],
      ["2) bit t of each byte, t = element mod 8, with y = B·", sup("2", "−t"), ":"],
      m("bit = ") + rnd(m("r − ") + fr("1", "2") + m(" + ") + sup("2", "−9")) + m(",  r = y − 2") + fl(fr("y", "2")),
      ["(", m("−2") + fl(fr("y", "2")), " by rounding to even; slide “Bits without bitwise instructions”)."]],
     (True, "384 random bytes against numpy unpackbits; every KEM input",
      f"unpacking dk (2,400 bytes) takes {cyc(bb[0])}", f"{bb[2]} instruction words"))

# 5 ByteDecode
bd = kern("encaps", "decodeten")
step("ByteDecode_d", "FIPS 203 Algorithm 6: d = 12 for t̂ and ŝ, 10 for u, 4 for v, 12-bit candidates of SampleNTT",
     "FIPS 203 Algorithm 6 ByteDecode_d(B)", [
         "1: b ← BytesToBits(B)",
         "2: for (i ← 0; i < 256; i++)",
         "3:   F[i] ← Σ_{j=0}^{d−1} b[i·d + j] · 2^j  mod m",
         "4: end for",
         "5: return F          (m = 2^d if d < 12, q if d = 12)"],
     [sub("F", "i") + m(" = ") + sm("j=0", "d−1", sub("b", "id+j") + sup("2", "j")) + m("  mod m")],
     c_func("compress.c", r"void mlk_poly_frombytes_c\("),
     py_src("ipu/kernels/nx8io.py", [("def bits_to_fields(scratch, bit_base, d, n_out_rows, out_refs, weights, out_kind=\"limbs\"):",
                                     "        nb.agg_vv((d * f) % 512"), ("        if f % 128 == 127:", 4), ("    k = nb.kernel()", 3)]),
     [["One AGG.SUM per field: the MULT.RC.VV window that starts at bit d·f times the weight row (E4, exact), "
       "summed into element f mod 128 of R_ACC:"],
      sub("F", "f") + m(" = ") + sm("t=0", "d−1", sub("w", "t") + m("·") + sub("bit", "df+t")) + m(",   ") + sub("w", "t") + m(" = ") + sup("2", "t"),
      ["The 4 slots of R_CYCLIC hold 4 consecutive bit rows, so a field that spans two rows is one window. After 128 fields a split store "
       "writes the row. The same kernel with other weights is CBD and BitsToBytes."]],
     (True, "random bytes against reference ByteDecode, d = 4, 10, 12",
      f"decoding t̂ of ek (3 polynomials) takes {cyc(bd[0])}", f"{bd[2]} instruction words (d = 12)"))

# 6 ByteEncode
f2b = kern("keygen", "f2b12t0")
pk_ = kern("keygen", "packdk")
step("ByteEncode_d and BitsToBytes", "FIPS 203 Algorithms 5 and 3: the output bytes ek, dk, ct and ss",
     "FIPS 203 Algorithm 5 ByteEncode_d(F); Algorithm 3", [
         "1: for (i ← 0; i < 256; i++)",
         "2:   a ← F[i]",
         "3:   for (j ← 0; j < d; j++)",
         "4:     b[i·d + j] ← a mod 2",
         "5:     a ← (a − b[i·d + j]) / 2",
         "8: B ← BitsToBytes(b)",
         "Alg. 3: B[⌊i/8⌋] ← B[⌊i/8⌋] + b[i] · 2^(i mod 8)"],
     [sub("b", "id+j") + m(" = ") + fl(fr(sub("F", "i"), sup("2", "j"))) + m(" mod 2"),
      sub("B", "i") + m(" = ") + sm("j=0", "7", sub("b", "8i+j") + sup("2", "j"))],
     c_func("compress.c", r"void mlk_poly_tobytes_c\("),
     py_src("ipu/kernels/nx8io.py", [("def fields_to_bits(scratch, digits, d, bit_base):", "    nb = NB(scratch, loop=2 * d // P)"),
                                    ("        wl[q] = np.where", 2), ("    reps = [R(nb.alloc()) for _ in range(4)]", "    return nb.kernel()")]),
     [["A field x < 4,096 is first split into byte digits (to_digits kernel):"],
      m("hi = ") + fl(fr("x", "256")) + m(",   lo = x − 256·hi"),
      ["Per output bit row, masked MULT.EE copy each field's digit to its d elements; element j gets weight ",
       sup("2", "−t"), " (t = j mod d) and floor mod 2 gives the bit."],
      ["BitsToBytes is the ByteDecode kernel with d = 8 and weights 1, 2, …, 128: one cycle per byte."]],
     (True, "reference ByteEncode d = 4, 10, 12 and packbits; ek, dk, ct",
      f"one polynomial, d = 12: {cyc(f2b[0])}; packing dk (2,400 bytes): {cyc(pk_[0])}", f"{f2b[2]} instruction words (d = 12)"))

# 7 Compress / Decompress
cu, cvv, du = kern("encaps", "compress_uen"), kern("encaps", "compress_ven"), kern("decaps", "decompress_u")
step("Compress_d and Decompress_d", "FIPS 203 (4.7), (4.8): d = 10 for u, 4 for v, 1 for the message",
     "FIPS 203 (4.7), (4.8)", [
         "Compress_d   : Z_q → Z_{2^d}",
         "    x ↦ ⌈(2^d / q) · x⌋ mod 2^d",
         "Decompress_d : Z_{2^d} → Z_q",
         "    y ↦ ⌈(q / 2^d) · y⌋",
         "(⌈·⌋: rounding to the nearest integer, half up)"],
     [sub("Compress", "d") + m("(x) = ") + rnd(fr(sup("2", "d"), "q") + m("·x")) + m(" mod ") + sup("2", "d"),
      sub("Decompress", "d") + m("(y) = ") + rnd(fr("q", sup("2", "d")) + m("·y"))],
     ("mlkem-native compress.h:225, compress.h:259", c_func("compress.h", r"static MLK_INLINE uint16_t mlk_scalar_compress_d10\(")[1]
      + [""] + c_func("compress.h", r"static MLK_INLINE int16_t mlk_scalar_decompress_d10\(")[1]),
     py_src("ipu/kernels/nx8poly.py", [("def compress(scratch: int, src, dst, d: int, n: int = 1, out_qf=None) -> dict:", "    return nb.kernel()"),
                                      ("def decompress(scratch: int, src, dst, d: int, n: int = 1, src_digits: bool = False, src_qf=None) -> dict:", 1),
                                      ("        nb.const(2.0 ** -(d + 1))", 4)]),
     [["Compress: exact quotient (d = 10 needs the corrected form: ", sub("K", "0"), " approximate, ",
       sub("r", "0"), " = ", sup("2", "d"), "x − q", sub("K", "0"), "):"],
      m("c = ") + sub("K", "0") + m(" + ") + rnd(fr(sub("r", "0"), "q")) + m(",   out = c − ") + sup("2", "d") + m("·relu(c − ") + sup("2", "d") + m(" + 1)"),
      ["Decompress: ", m("c·q/") + sup("2", "d") + m(" + ") + sup("2", "−(d+1)"), " is exact in FP32 and never a tie, so"],
      m("x = ") + rnd(fr("c·q", sup("2", "d")) + m(" + ") + sup("2", "−(d+1)"))],
     (True, "Compress: all 3,329 inputs; Decompress: all 2^d inputs; d = 1, 4, 5, 10, 11",
      f"Compress_10 of u (3 polynomials): {cyc(cu[0])}; Compress_4 of v: {cyc(cvv[0])}; Decompress_10 of u: {cyc(du[0])}",
      f"compress d = 10: {cu[2]} words; decompress d = 10: {du[2]} words"))

# 8 SampleNTT
SAMPK = ("parse", "prep_first", "prep", "stage", "centre_A")
sp_c, sp_n, sp_w = kern("keygen", *SAMPK, prefix=True)
xof_c = sum(cy for k, (cy, la, cat, wd) in PK["keygen"].items() if k.startswith("xof"))
xof_perm = 9 * 5
step("SampleNTT", "FIPS 203 Algorithm 7, with the loop bound of Appendix B (280 iterations = 5 SHAKE128 blocks)",
     "FIPS 203 Algorithm 7 SampleNTT(B)", [
         "4: while j < 256",
         "5:   (ctx, C) ← XOF.Squeeze(ctx, 3)",
         "6:   d1 ← C[0] + 256 · (C[1] mod 16)",
         "7:   d2 ← ⌊C[1]/16⌋ + 16 · C[2]",
         "8:   if d1 < q:  â[j] ← d1;  j ← j + 1",
         "12:  if d2 < q and j < 256:  â[j] ← d2;  j ← j + 1",
         "Appendix B: the loop may stop after 280 iterations",
         "(probability of reaching it: about 2^−261)"],
     [sub("d", "1") + m(" = ") + sub("C", "0") + m(" + 256·(") + sub("C", "1") + m(" mod 16),   ")
      + sub("d", "2") + m(" = ") + fl(fr(sub("C", "1"), "16")) + m(" + 16·") + sub("C", "2"),
      "Keep the candidates d < q, in order, until 256 are accepted."],
     c_func("sampling.c", r"MLK_STATIC_TESTABLE unsigned mlk_rej_uniform_c\("),
     py_src("ipu/kernels/nx8io.py", [("        # r = floor((c + 767) / 4096): exact", 5),
                                    ("def sample_stage(scratch, b, Vin, Vout, Y, W, zero_ref, rows_out):", "        nb.split_store(W[k])")]),
     [["Fixed schedule: 5 SHAKE128 blocks = 560 candidates of 12 bits, decoded by the ByteDecode kernel (d = 12). Per candidate c:"],
      m("r = ") + fl(fr("c + 767", "4096")) + m(" = ") + brk(m("c ≥ q")) + m(",   V = (z + 512c)(1 − r)"),
      ["z = rejections before c (prefix count in 7 shifted adds). 9 stages b = 0..8 move each accepted c down by z lanes, ",
       sup("2", "b"), " per stage:"],
      m("m = V mod 2,  Y = ") + fr("m", "2") + m(",  W = (V − 1)Y"),
      m("V′ = ") + fr("V", "2") + m(" − W − Y + shift(W, ") + sup("2", "b") + m(")"),
      ["Fewer than 256 accepted (z > 304): all outputs 0, status 0 (Appendix B)."]],
     (True, "30 random streams; 5 crafted (largest shift, exactly 256, all, limit reached)",
      f"the 9 entries of Â: {cyc(sp_c)} plus {xof_perm} permutations of SHAKE128",
      f"{max(main['words'][k] for k in main['words'] if k.startswith(('parse', 'prep', 'stage')))} instruction words at most"))

# 9 CBD
cb = kern("keygen", "cbdkg")
step("SamplePolyCBD_η", "FIPS 203 Algorithm 8 (η = 2 for ML-KEM-768; η = 3 also tested)",
     "FIPS 203 Algorithm 8 SamplePolyCBD_η(B)", [
         "1: b ← BytesToBits(B)",
         "2: for (i ← 0; i < 256; i++)",
         "3:   x ← Σ_{j=0}^{η−1} b[2iη + j]",
         "4:   y ← Σ_{j=0}^{η−1} b[2iη + η + j]",
         "5:   f[i] ← x − y mod q",
         "7: return f"],
     [sub("f", "i") + m(" = ") + sm("j=0", "η−1", sub("b", "2iη+j")) + m(" − ")
      + sm("j=0", "η−1", sub("b", "2iη+η+j"))],
     c_func("sampling.c", r"void mlk_poly_cbd2\("),
     py_src("ipu/mlkem768_ipu.py", [("        def prf_cbd(tag, seed_rows, n):", "            return seq")]),
     [["The ByteDecode kernel with d = 2η and weights (1, 1, −1, −1) for η = 2 (1, 1, 1, −1, −1, −1 for η = 3), "
       "one AGG.SUM per coefficient:"],
      sub("f", "i") + m(" = ") + sub("b", "4i") + m(" + ") + sub("b", "4i+1") + m(" − ") + sub("b", "4i+2") + m(" − ") + sub("b", "4i+3"),
      ["Stored as one S8 row per 128 coefficients (values −2..2); the NTT takes them as signed integers."]],
     (True, "2 mlkem-native vectors each for η = 2 and η = 3; every KEM test",
      f"6 polynomials (s, e) in KeyGen: {cyc(cb[0])}, plus 6 PRF hashes", f"{cb[2]} instruction words"),
     sub_b="ipu/mlkem768_ipu.py: the PRF hashes, then the ByteDecode kernel (bits_to_fields) with CBD weights")

# 10 NTT
nt = kern("keygen", "ntt_se")
step("NTT", "FIPS 203 Algorithm 9; mlkem-native: mlk_poly_ntt (Montgomery multiplication by ζ)",
     "FIPS 203 Algorithm 9 NTT(f)", [
         "2: i ← 1",
         "3: for (len ← 128; len ≥ 2; len ← len/2)",
         "4:   for (start ← 0; start < 256; start ← start + 2·len)",
         "5:     zeta ← ζ^BitRev7(i) mod q;  i ← i + 1",
         "7:     for (j ← start; j < start + len; j++)",
         "8:       t ← zeta · f̂[j + len]",
         "9:       f̂[j + len] ← f̂[j] − t",
         "10:      f̂[j] ← f̂[j] + t"],
     [sub("f", "j") + m(" ← ") + sub("f", "j") + m(" + ζ·") + sub("f", "j+len"),
      sub("f", "j+len") + m(" ← ") + sub("f", "j") + m(" − ζ·") + sub("f", "j+len"),
      m("ζ = 17,  q = 3329")],
     c_func("poly.c", r"static void mlk_ntt_butterfly_block\("),
     py_src("ipu/kernels/nx8poly.py", [("def ntt_vecs(layer_: int, full: bool):", "    return out"),
                                      ("def ntt(scratch: int, src, dst, n: int = 1, stride: int = 0, bound: float = Q) -> dict:", 1),
                                      ("    for lay in range(1, 8):", 4)]),
     [["One generic layer for all 7 layers (ℓ = 256 >> layer), on R_CYCLIC windows shifted by ±ℓ:"],
      sub("out", "j") + m(" = ") + sub("A", "j") + sub("p", "j") + m(" + ") + sub("B", "j") + sub("p", "j+ℓ") + m(" + ")
      + sub("C", "j") + sub("p", "j−ℓ") + m(" − q") + sub("K", "j"),
      ["Top lanes: A = 1, B = ζ, C = 0. Bottom lanes: A = −ζ, B = 0, C = 1. ", sub("K", "j"),
       " (approximate quotient, slide “Exact integer arithmetic”) is applied at layers 1, 4, 7 only; between them the "
       "build-time bounds keep partial sums below ", sup("2", "24"), "."]],
     (True, "16 mlkem-native vectors (mod q), 8 inputs ±(q−1), zero inputs",
      f"NTT of s and e (6 polynomials) in KeyGen: {cyc(nt[0])}", f"{nt[2]} instruction words (7 layers, looped over the polynomials)"))

# 11 inverse NTT
it = kern("encaps", "intt_uven")
step("Inverse NTT", "FIPS 203 Algorithm 10; mlkem-native: mlk_poly_invntt_tomont (output in Montgomery form)",
     "FIPS 203 Algorithm 10 NTT⁻¹(f̂)", [
         "2: i ← 127",
         "3: for (len ← 2; len ≤ 128; len ← 2·len)",
         "4:   for (start ← 0; start < 256; start ← start + 2·len)",
         "5:     zeta ← ζ^BitRev7(i) mod q;  i ← i − 1",
         "7:     for (j ← start; j < start + len; j++)",
         "8:       t ← f[j]",
         "9:       f[j] ← t + f[j + len]",
         "10:      f[j + len] ← zeta · (f[j + len] − t)",
         "14: f ← f · 3303 mod q          (3303 = 128⁻¹ mod q)"],
     [sub("f", "j") + m(" ← ") + sub("f", "j") + m(" + ") + sub("f", "j+len") + m(",   ")
      + sub("f", "j+len") + m(" ← ζ·(") + sub("f", "j+len") + m(" − ") + sub("f", "j") + m(")"),
      m("f ← 3303·f mod q")],
     c_func("poly.c", r"static void mlk_invntt_layer\("),
     py_src("ipu/kernels/nx8poly.py", [("def intt_vecs(layer_: int, scale: int, full: bool):", "    return rows"),
                                      ("def intt(scratch: int, src, dst, n: int = 1, scale: int = F_INTT, bound: float = Q) -> dict:", 1),
                                      ("    for i, lay in enumerate(range(7, 0, -1)):", 4)]),
     [["Gentleman-Sande butterflies in the same generic layer:"],
      m("top: s(") + sub("p", "j") + m(" + ") + sub("p", "j+ℓ") + m("),   bottom: ") + sub("ζ", "s") + m("(") + sub("p", "j") + m(" − ")
      + sub("p", "j−ℓ") + m(")"),
      ["The final scale s is folded into the constants of the last layer: ", m("s = ") + sup("128", "−1") + m(" mod q"),
       " in the KEM, ", m("s = ") + sup("2", "16") + m("/128 mod q = 512"), " for mlkem-native's invntt_tomont in the kernel test. "
       "The quotient is applied at layers 7, 5, 3, 1."]],
     (True, "16 mlkem-native vectors (mod q), 8 inputs ±(q−1), zero inputs",
      f"NTT⁻¹ of u and v (4 polynomials) in Encaps: {cyc(it[0])}", f"{it[2]} instruction words"))

# 12 MultiplyNTTs
bm, mc = kern("keygen", "basemul_As"), kern("keygen", "mulcache_s")
step("MultiplyNTTs and BaseCaseMultiply", "FIPS 203 Algorithms 11 and 12; mlkem-native precomputes a mulcache (γ·b₁)",
     "FIPS 203 Algorithms 11, 12", [
         "Alg. 11 MultiplyNTTs(f̂, ĝ):",
         "  for (i ← 0; i < 128; i++)",
         "    (ĥ[2i], ĥ[2i+1]) ← BaseCaseMultiply(f̂[2i],",
         "        f̂[2i+1], ĝ[2i], ĝ[2i+1], ζ^(2BitRev7(i)+1))",
         "Alg. 12 BaseCaseMultiply(a0, a1, b0, b1, γ):",
         "  c0 ← a0·b0 + a1·b1·γ",
         "  c1 ← a0·b1 + a1·b0"],
     [sub("c", "0") + m(" = ") + sub("a", "0") + sub("b", "0") + m(" + ") + sub("a", "1") + sub("b", "1") + m("γ,   ")
      + sub("c", "1") + m(" = ") + sub("a", "0") + sub("b", "1") + m(" + ") + sub("a", "1") + sub("b", "0"),
      m("γ = ") + sup("ζ", "2BitRev7(i)+1")],
     c_func("poly_k.c", r"MLK_STATIC_TESTABLE void mlk_polyvec_basemul_acc_montgomery_cached_c\("),
     py_src("ipu/kernels/nx8poly.py", [("def basemul_acc(scratch: int, out, terms, n: int = 1, montgomery_fold: bool = False) -> dict:",
                                       "    return nb.kernel()")]),
     [["Mulcache rows, exactly centred (|V| ≤ 1,664): ", sub("V1", "j"), " = ", sub("b", "j"), " (even j), ", sub("b", "j−1"),
       " (odd j); ", sub("V2", "j"), " = γ", sub("b", "j+1"), " (even), ", sub("b", "j"), " (odd). Then, with the window shifted +1 on even "
       "and −1 on odd lanes:"],
      sub("c", "j") + m(" = ") + sm("k", "", sub("a", "k,j") + sub("V1", "k,j") + m(" + ") + sub("a", "k,j±1") + sub("V2", "k,j")),
      ["Every partial sum stays below ", m("6·") + sup("1672", "2") + m(" = 16,773,504 < ") + sup("2", "24"),
       " (k = 3), then one approximate-quotient reduction."]],
     (True, "4 mlkem-native vectors, 12 mulcache polynomials, |a| = |V| = 1,664",
      f"Â∘ŝ (3 output polynomials) in KeyGen: {cyc(bm[0])}; mulcache of ŝ: {cyc(mc[0])}",
      f"basemul {bm[2]} words, mulcache {mc[2]} words"))

# 13 reductions
rt, ce = kern("keygen", "reduce_t"), kern("keygen", "centre_A0")
step("Modular reduction", "FIPS 203 Section 2.3 (arithmetic mod q); mlkem-native's Barrett and Montgomery reductions",
     "FIPS 203 Section 2.3 / mlkem-native", [
         "x mod q: the representative in [0, q)",
         "Barrett   : a − q·round(a/q)  in [−(q−1)/2, (q−1)/2]",
         "canonical : x − q·⌊x/q⌋    in [0, q)",
         "Montgomery: a · 2^−16 mod q   (mlkem-native fqmul,",
         "            basemul; folded into constants here)"],
     [m("t = ") + fl(fr(m("20159·a + ") + sup("2", "25"), sup("2", "26"))) + m(",   a − tq   (Barrett)"),
      m("t = a·") + sup("q", "−1") + m(" mod± ") + sup("2", "16") + m(",   ") + fr("a − tq", sup("2", "16")) + m(" ≡ a·")
      + sup("2", "−16") + m(" (Montgomery)")],
     ("mlkem-native poly.c:75, poly.h:58", c_func("poly.c", r"static MLK_INLINE int16_t mlk_barrett_reduce\(")[1] + [""]
      + c_func("poly.h", r"static MLK_ALWAYS_INLINE int16_t mlk_montgomery_reduce\(")[1]),
     py_src("ipu/kernels/nx8poly.py", [("def residue(cx: Ctx, terms, kterms, dst, exact: bool, bias: float = 0.0, clean: bool = False):", 1),
                                      ("    budget = EXACT_BUDGET if exact else APPROX_BUDGET", 2),
                                      ("    if err < budget:", 5),
                                      ("    plan, btol, err, cm = plan_k([(r0, 0, 1.0 / Q, rb)], bias, EXACT_BUDGET)", 5),
                                      ("def reduce_canonical(scratch: int, src, dst, n: int = 1, bound: float = 2 ** 15) -> dict:",
                                       "    return nb.kernel()")]),
     [["All reductions are one residue: ", m("r = y − qK,  K = ") + rnd(m("y/q + β")), " with exact K (error < 1/(4q)):"],
      m("Barrett:  β = 0   ⇒   r ∈ [−1664, 1664]"),
      m("canonical:  β = −") + fr("q − 1", "2q") + m("   ⇒   r ∈ [0, q)"),
      ["Montgomery is not a kernel: the factors ", sup("2", "±16"), " are folded into the constants (ζ, γ, the final INTT scale)."]],
     (True, "Barrett on all 65,536 int16; 16 poly_reduce, 16 tomont vectors",
      f"canonical reduction of t̂ (3 polynomials) in KeyGen: {cyc(rt[0])}; centring one entry of Â: {cyc(ce[0])}",
      f"canonical {rt[2]} words, centring {ce[2]} words"))

# 14 add / sub
ad, sb = kern("keygen", "add_te"), kern("decaps", "sub_w")
step("Polynomial add and subtract", "FIPS 203 Algorithms 13-15 (t̂ = Â∘ŝ + ê, u, v, w); no reduction, as in mlkem-native",
     "FIPS 203 (inside Algorithms 13-15)", [
         "13.18: t̂ ← Â ∘ ŝ + ê",
         "14.19: u ← NTT⁻¹(Âᵀ ∘ ŷ) + e1",
         "14.21: v ← NTT⁻¹(t̂ᵀ ∘ ŷ) + e2 + μ",
         "15.6:  w ← v′ − NTT⁻¹(ŝᵀ ∘ NTT(u′))"],
     [sub("h", "i") + m(" = ") + sub("f", "i") + m(" ± ") + sub("g", "i") + m("   (i = 0..255, no reduction)")],
     ("mlkem-native poly.c:231, poly.c:251", c_func("poly.c", r"^void mlk_poly_add\(")[1] + [""] + c_func("poly.c", r"^void mlk_poly_sub\(")[1]),
     py_src("ipu/kernels/nx8poly.py", [("def emit_exact(nb, terms, K=None, kmax=0.0, sign_k=-1):", "        nb.term(K, 0, float(Q), sign=sign_k)"),
                                      ("def add_sub(scratch: int, triples, sub: bool = False, bound: float = 2 ** 13) -> dict:",
                                       "    return nb.kernel()")]),
     [["Each limb row of f and g is added (ACC.ADD) or subtracted (ACC.SUB) exactly in R_ACC; a split store writes 2 limbs. "
       "Inputs may have any number of limbs (CBD outputs have 1)."],
      sub("h", "i") + m(" = ") + sm("l", "", sub("f", "i,l")) + m(" ± ") + sm("l", "", sub("g", "i,l"))],
     (True, "8 mlk_poly_add and 8 mlk_poly_sub vectors",
      f"t̂ = Â∘ŝ + ê (3 polynomials): {cyc(ad[0])}; w = v′ − … in Decaps: {cyc(sb[0])}",
      f"{ad[2]} instruction words"))

# 15 KeyGen
step("KeyGen", "FIPS 203 Algorithms 16 (ML-KEM.KeyGen_internal) and 13 (K-PKE.KeyGen); input d, z; output ek, dk",
     "FIPS 203 Algorithms 13 and 16", [
         "13.1:     (ρ, σ) ← G(d ‖ k)",
         "13.3-7:   Â[i,j] ← SampleNTT(ρ ‖ j ‖ i)",
         "13.8-15:  s[i], e[i] ← SamplePolyCBD_η1(PRF_η1(σ, N))",
         "13.16-17: ŝ ← NTT(s);  ê ← NTT(e)",
         "13.18:    t̂ ← Â ∘ ŝ + ê",
         "13.19:    ek_PKE ← ByteEncode12(t̂) ‖ ρ",
         "13.20:    dk_PKE ← ByteEncode12(ŝ)",
         "16.3:     dk ← dk_PKE ‖ ek ‖ H(ek) ‖ z"],
     [hat("t") + m(" = ") + hat("A") + m("∘") + hat("s") + m(" + ") + hat("e")],
     c_func("indcpa.c", r"^int mlk_indcpa_keypair_derand\(",
            keep=[("mlk_memcpy(coins_with_domain_separator", "mlk_hash_g("), ("mlk_gen_matrix(", "mlk_pack_pk(")]),
     py_src("ipu/mlkem768_ipu.py", [("        # ---------------- keygen", "        self.seq_keygen = kg")]),
     [["The whole operation is one fixed list of kernel launches, built once; the host only launches them in order."],
      [f"{LAUNCH['keygen']} launches, {nperm('keygen')} Keccak-f permutations. The bit rows of dk are laid out as "
       "s | t | ρ | h | z, so G writes ρ into place, H(ek) hashes bit rows 72..145, and one pack produces ek, another dk."],
      ["ok_kg: status 1, or every output byte 0 and status 0 if a SampleNTT reached its bound."]],
     (allres["keygen: ek, dk bit-exact, status 1"], f"main vector and {NSEEDS} random seeds: ek, dk",
      f"{cyc(TOT['keygen'])} per KeyGen", f"{len(PK['keygen'])} kernels, the largest {kern('keygen', *PK['keygen'])[2]} instruction words"),
     sub_b="ipu/mlkem768_ipu.py: the launch sequence of KeyGen")

# 16 Encaps
rc_ = kern("encaps", "ek_range_check")
step("Encaps", "FIPS 203 Algorithms 17 (ML-KEM.Encaps_internal) and 14 (K-PKE.Encrypt); Section 7.2 input check",
     "FIPS 203 Algorithms 14, 17 and Section 7.2", [
         "17.1:     (K, r) ← G(m ‖ H(ek))",
         "14.2:     t̂ ← ByteDecode12(ek[0 : 384k])",
         "14.4-8:   Â[i,j] ← SampleNTT(ρ ‖ j ‖ i)",
         "14.9-17:  y, e1, e2 ← SamplePolyCBD(PRF(r, N))",
         "14.18-19: u ← NTT⁻¹(Âᵀ ∘ NTT(y)) + e1",
         "14.20-21: v ← NTT⁻¹(t̂ᵀ ∘ NTT(y)) + e2 + Decompress1(m)",
         "14.22-23: c ← ByteEncode10(Compress10(u))",
         "            ‖ ByteEncode4(Compress4(v))",
         "7.2:      ek = ByteEncode12(ByteDecode12(ek))"],
     [m("u = ") + sup("NTT", "−1") + m("(") + sup(hat("A"), "T") + m("∘") + hat("y") + m(") + ") + sub("e", "1"),
      m("v = ") + sup("NTT", "−1") + m("(") + sup(hat("t"), "T") + m("∘") + hat("y") + m(") + ") + sub("e", "2") + m(" + μ")],
     c_func("indcpa.c", r"^int mlk_indcpa_enc\(",
            keep=[("mlk_unpack_pk(", "mlk_pack_ciphertext(")]),
     py_src("ipu/mlkem768_ipu.py", [("        def encrypt(tag, coins_rows, mbits_rows, encode_out):", 8),
                                   ("        # ---------------- encaps", "        self.seq_encaps = en")]),
     [["Section 7.2 check on the IPU, per coefficient of the decoded t̂ (ek_range_check):"],
      brk(m("t ≥ q")) + m(" = ") + fl(fr("t + 767", "4096")) + m(",   F ← F + ") + brk(sm("", "", brk(m("t ≥ q"))) + m(" ≥ 1")),
      ["If F > 0 (invalid ek or a SampleNTT bound), ct and ss are all 0 and the status is 0, where mlkem-native returns −4."],
      [f"{LAUNCH['encaps']} launches, {nperm('encaps')} Keccak-f permutations; the check costs {cyc(rc_[0])}."]],
     (allres["encaps: ct, ss bit-exact, status 1"], f"main vector and {NSEEDS} random seeds; invalid ek gives status 0",
      f"{cyc(TOT['encaps'])} per Encaps", f"{len(PK['encaps'])} kernels, the largest {kern('encaps', *PK['encaps'])[2]} instruction words"),
     sub_b="ipu/mlkem768_ipu.py: K-PKE.Encrypt (shared with Decaps) and the launch sequence of Encaps")

# 17 Decaps
fo = kern("decaps", "fo_select")
step("Decaps", "FIPS 203 Algorithms 18 (ML-KEM.Decaps_internal) and 15 (K-PKE.Decrypt); Section 7.3 input check",
     "FIPS 203 Algorithms 15, 18 and Section 7.3", [
         "15.3-4: u′ ← Decompress10(ByteDecode10(c1))",
         "        v′ ← Decompress4(ByteDecode4(c2))",
         "15.5:   ŝ ← ByteDecode12(dk_PKE)",
         "15.6:   w ← v′ − NTT⁻¹(ŝᵀ ∘ NTT(u′))",
         "15.7:   m′ ← ByteEncode1(Compress1(w))",
         "18.7:   (K′, r′) ← G(m′ ‖ h)",
         "18.8:   K̄ ← J(z ‖ c)",
         "18.9:   c′ ← K-PKE.Encrypt(ek_PKE, m′, r′)",
         "18.10:  if c ≠ c′ then K′ ← K̄",
         "7.3:    H(dk[384k : 768k + 32]) = h"],
     [m("w = v′ − ") + sup("NTT", "−1") + m("(") + sup(hat("s"), "T") + m("∘NTT(u′))"),
      m("K = ") + brk(m("c = c′")) + m("·K′ + ") + brk(m("c ≠ c′")) + m("·") + bar("K")],
     c_func("kem.c", r"^int mlk_kem_dec\(",
            keep=[("ret = mlk_kem_check_sk(", "goto cleanup;"), ("ret = mlk_indcpa_dec(", "ret = mlk_indcpa_dec("),
                  ("mlk_memcpy(buf + MLKEM_SYMBYTES", "ret = mlk_indcpa_enc("), ("fail = mlk_ct_memcmp(", "mlk_memcpy(ss, buf")]),
     py_src("ipu/mlkem768_ipu.py", [("        # ---------------- decaps", 3), ('        de += self.sponge("G_de"', "        self.seq_decaps = de")]),
     [["FO compare and select (fo_select), per compressed value, d = c′ − c over N = 1,024 values:"],
      m("e = relu6(1 − d)·relu6(1 + d) = ") + brk(m("d = 0")),
      m("eq = relu(") + sm("", "", "e") + m(" − N + 1),   K = ") + bar("K") + m(" + eq·(K′ − ") + bar("K") + m(")"),
      ["Section 7.3 (dk_hash_check): ", sm("", "", sup("(a − b)", "2")), " over the bits of H(ek) and h; nonzero ⇒ status 0 "
       "(mlkem-native returns −5)."]],
     (all(v for k, v in main["results"].items() if "decaps" in k), f"main vector, {NSEEDS} random seeds, 3 crafted ct; invalid dk gives status 0",
      f"{cyc(TOT['decaps'])} per Decaps; FO select {cyc(fo[0])}", f"{len(PK['decaps'])} kernels, the largest {kern('decaps', *PK['decaps'])[2]} instruction words"),
     sub_b="ipu/mlkem768_ipu.py: the launch sequence of Decaps")

# ================================================================= mlkem-native benchmarks
BOARDS = ("Mac Mini", "Cortex-A76", "ppc64le", "Cortex-A72", "Cortex-A55", "SpacemiT", "Cortex-M55")
CLOCK_FIX = {"Mac Mini": "not published", "Cortex-A55": "not published", "SpacemiT": "not published",
             "ppc64le": "2.45-4.15 GHz, depends on system"}
CLOCK_GHZ = {"Graviton5": 3.3, "Graviton4": 2.8, "Graviton3": 2.6, "Graviton2": 2.5, "Cortex-A76": 2.4,
             "Cortex-A72": 1.5, "Cortex-M55": 0.8}


def clock_of(r):
    """A single published clock for a row that counts core cycles; None otherwise (no time is given)."""
    if "rdtsc" in r["counter"]:
        return None
    return next((v for k_, v in CLOCK_GHZ.items() if k_ in r["platform"]), None)


def bench_rows(sel):
    rows = [["Platform", "Build", "Clock", "KeyGen", "Encaps", "Decaps"],
            [("IPU (this work, emulated)", ACCENT), ("NX8, host rule", ACCENT), ("1 GHz (target)", ACCENT)]
            + [(f"{fmt(bench['ipu'][o])}\n{at1g(bench['ipu'][o])}", ACCENT) for o in OPS]]
    for r in bench["rows"]:
        if not sel(r):
            continue
        nm = r["platform"].replace(" benchmarks", "")
        if "rdtsc" in r["counter"]:
            nm += " †"
        if r["commit"] == "9345dfb":
            nm += " ‡"
        clk = next((v for k_, v in CLOCK_FIX.items() if k_ in r["platform"]), r["clock"])
        g = clock_of(r)
        rows.append([nm, r["build"], clk] + [f"{fmt(r[o])}" + (f"\n{at1g(r[o] / g)}" if g else "") for o in OPS])
    return rows


BFOOT = ("Cells: cycles, and below them the time where the row counts core cycles at one published clock (time = cycles / clock). "
         "† x86 instances: mlkem-native reports rdtsc TSC ticks, not core cycles, so no time is derived. "
         "‡ CI commit 9345dfb; other rows 76b6587. Source: https://pq-code-package.github.io/mlkem-native/dev/bench/, fetched 2026-11-15 "
         "(results/mlkem_native_bench.json); clocks from the sources named there. IPU: emulator cycles at the 1 GHz target clock.")


def is_board(r):
    return any(b in r["platform"] for b in BOARDS)


s = prs.slides.add_slide(BLANK)
title(s, "mlkem-native benchmarks: boards", "ML-KEM-768 from mlkem-native's CI benchmark page, next to the IPU")
rows = bench_rows(is_board)
table(s, 0.5, 1.4, 12.33, rows, [3.6, 1.9, 2.53, 1.45, 1.45, 1.4], size=12, rowh=0.55, right_from=3)
m55 = next(r for r in bench["rows"] if "Cortex-M55" in r["platform"])
text(s, 0.5, 6.35, 12.33, 1.0, BFOOT, size=10, color=MUTED)

s = prs.slides.add_slide(BLANK)
title(s, "mlkem-native benchmarks: cloud instances", "ML-KEM-768 from mlkem-native's CI benchmark page, next to the IPU")
rows = bench_rows(lambda r: not is_board(r))
table(s, 0.5, 1.3, 12.33, rows, [3.6, 1.75, 2.98, 1.35, 1.35, 1.3], size=9.5, rowh=0.255, right_from=3)
text(s, 0.5, 6.85, 12.33, 0.6, BFOOT, size=8.5, color=MUTED)

# ================================================================= caveats
s = prs.slides.add_slide(BLANK)
title(s, "What the numbers mean", "Scope of the cycle counts, times and checks")
text(s, 0.5, 1.4, 12.33, 5.9, [
    [("Emulator cycles. ", {"bold": True}), f"Cycles come from the bit-accurate emulator (ipu-emulator {EMULATOR}): one instruction word per "
     f"cycle. Program loads into IMEM and launch overhead ({min(LAUNCH.values())}-{max(LAUNCH.values())} launches per operation) are not counted."],
    [("Times. ", {"bold": True}), "Time = cycles / 1 GHz, the IPU's target clock."],
    [("SampleNTT bound. ", {"bold": True}), "Each SampleNTT reads exactly 5 SHAKE128 blocks (280 iterations), the bound FIPS 203 Appendix B "
     "permits (probability of reaching it about 2⁻²⁶¹). If it is reached, every output byte is 0 and the status byte is 0, "
     "as Appendix B prescribes."],
    [("All-zero rows. ", {"bold": True}), "An all-zero NX8 row decodes as 2⁻¹²⁷ in every element, not 0. Kernels that "
     "multiply by large constants round their outputs to integers, and zero inputs (c₁ = 0, all-zero ct) are tested end to end."],
    [("Exactness rests on bounds. ", {"bold": True}), "Limb splits, quotient error budgets and the 2²⁴ partial-sum limit are asserted "
     "when each kernel is built, and tested on all int16 inputs (Barrett), all 3,329 inputs (compress) and ±(q−1) inputs (NTT, basemul)."],
    [("Representatives. ", {"bold": True}), "Inside the NTT domain a coefficient may be any representative within its bound; "
     "those are compared mod q. Every byte output (ek, dk, ct, ss) is compared exactly."],
    [("mlkem-native figures. ", {"bold": True}), "They are mlkem-native's CI results on other processors, with different counters; "
     "times are given only where a single clock is published and the counter counts core cycles."],
], size=14, space=8)

# ================================================================= test output
s = prs.slides.add_slide(BLANK)
title(s, "Test output", "results/run_all.log: the ML-KEM-768 checks under the host rule")
log = open(RES / "run_all.log").read().split("\n")
i0 = next(i for i, l in enumerate(log) if l.startswith("== test_kem_ipu main"))
kem_log = [l for l in log[i0:] if l.strip() and not re.match(r"(keygen|encaps|decaps): total", l)]
kem_log = [w for l in kem_log for w in wrap_line(l, 120)]
assert len(kem_log) <= 31, len(kem_log)
box(s, 0.5, 1.4, 12.33, 5.85)
text(s, 0.58, 1.46, 12.17, 5.75, kem_log, size=10.5, font=MONO, space=0)

# ================================================================= equations as OMML, then save
for sl in prs.slides:
    for sp in list(sl.shapes._spTree.iterchildren(qn("p:sp"))):
        if sp.find(qn("p:nvSpPr")).find(qn("p:cNvPr")).get("name") == "MathText":
            to_alternate_content(sp)
out = ROOT / "report" / "ML-KEM_on_LEO-NG_IPU.pptx"
prs.save(out)
print("wrote", out, len(prs.slides), "slides,", len(EQS), "equations")

if "--eq-docx" in sys.argv:
    dst = sys.argv[sys.argv.index("--eq-docx") + 1]
    rp = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:sz w:val="28"/></w:rPr>'
    paras = "".join(f'<w:p><m:oMathPara><m:oMathParaPr><m:jc m:val="left"/></m:oMathParaPr><m:oMath>'
                    f'{e.xml.replace("{RPI}", rp).replace("{RPP}", rp)}</m:oMath></m:oMathPara></w:p>'
                    f'<w:p><w:r><w:rPr><w:sz w:val="16"/></w:rPr><w:t xml:space="preserve">{_esc(e.txt)}</w:t></w:r></w:p>' for e in EQS)
    doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{NS_W}" xmlns:m="{NS_M}">'
           f"<w:body>{paras}<w:sectPr/></w:body></w:document>")
    with zipfile.ZipFile(dst, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" '
                   'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml", doc)
    print("wrote", dst)
