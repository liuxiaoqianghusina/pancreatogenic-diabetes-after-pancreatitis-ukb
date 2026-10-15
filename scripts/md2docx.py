# Markdown -> Word converter in Diabetes Care style (12 pt Times New Roman, double-spaced, justified, display items 10 pt, italic P).
# Usage: python md2docx.py input.md output.docx; DOC_SPACING sets line spacing; DC_HEADINGS=caps sets level-1 headings in capitals.
# **bold**, *italic*, ~subscript~, and 【placeholder】 highlighted in yellow).
# Table titles, table text, table notes and figure legends are set in 10 pt, with P values in italics.
import re, sys, os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

src, dst = sys.argv[1], sys.argv[2]
doc = Document()
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Cm(2.54); sec.top_margin = sec.bottom_margin = Cm(2.54)   # 1-inch margins
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)   # Diabetes Care：12 pt
st.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
st.paragraph_format.line_spacing = float(os.environ.get("DOC_SPACING", "2.0")); st.paragraph_format.space_after = Pt(0)   # double spacing by default
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY   # justified text; only the title is centred
for lvl, size in [(1, 12), (2, 12), (3, 12)]:
    h = doc.styles[f"Heading {lvl}"]; h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = False   # no bold headings
    h.font.color.rgb = RGBColor(0, 0, 0); h.element.rPr.rFonts.set(qn("w:eastAsia"), "SimHei")
    h.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    h.paragraph_format.keep_with_next = False; h.paragraph_format.keep_together = False   # no black square markers

TOK = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|~[^~]+?~|\^[^^]+?\^|【[^】]*】)")
P_RE = re.compile(r"((?<![A-Za-z*])P(?![A-Za-z*]))")   # P-value symbol, set in italics in tables and legends
def add_runs(par, text, p_italic=False):
    if p_italic:   # mark bare P values as *P* so the italic branch below handles them
        text = "".join(t if TOK.fullmatch(t) else P_RE.sub(r"*\1*", t) for t in TOK.split(text) if t)
    for part in TOK.split(text):
        if not part: continue
        if part.startswith("**"): r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("*"): r = par.add_run(part[1:-1]); r.italic = True
        elif part.startswith("~"): r = par.add_run(part[1:-1]); r.font.subscript = True
        elif part.startswith("^"): r = par.add_run(part[1:-1]); r.font.superscript = True
        elif part.startswith("【"): r = par.add_run(part); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else: r = par.add_run(part.replace("\\*", "*"))
        r.font.name = "Times New Roman"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")

DISPLAY = {"Tables", "Figure Legends", "Figure legends"}
REFS = {"References"}   # references single-spaced and justified
refs = False   # display items stay 10 pt   # level-1 sections whose paragraphs are display-item text
DISPLAY_PT = Pt(10)
display = False
title_done = False
lines = open(src, encoding="utf-8").read().split("\n")
i = 0
while i < len(lines):
    ln = lines[i]
    if not ln.strip(): i += 1; continue
    mi = re.match(r"^!\[(.*?)\]\((.+?)\)", ln.strip())
    if mi:   # image: ![caption](path)
        doc.add_picture(mi.group(2), width=Cm(16)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; i += 1; continue
    if ln.strip() == "\\pagebreak":
        doc.add_page_break(); i += 1; continue
    m = re.match(r"^(#{1,3})\s+(.*)", ln)
    if m:
        if len(m.group(1)) == 1:
            display = m.group(2).strip() in DISPLAY; refs = m.group(2).strip() in REFS
        htxt = m.group(2)
        caps = os.environ.get("DC_HEADINGS") == "caps" and title_done   # level-1 headings in capitals, level-2 in italics, no bold
        if caps and len(m.group(1)) == 1: htxt = htxt.upper()
        hp = doc.add_heading(level=len(m.group(1))); add_runs(hp, htxt); i += 1
        if caps and len(m.group(1)) >= 2:
            for r_ in hp.runs: r_.italic = True
        if len(m.group(1)) == 1 and not title_done:   # first level-1 heading is the paper title
            hp.alignment = WD_ALIGN_PARAGRAPH.CENTER; title_done = True
        continue
    if ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip(" \t") for c in lines[i].strip().strip("|").split("|")]   # keep em-space indents
            if not all(re.fullmatch(r"-+", c) for c in cells): rows.append(cells)
            i += 1
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))   # three-line table: rules above/below header, below body
        for r, row in enumerate(rows):
            for c, txt in enumerate(row):
                cell = t.cell(r, c); cell.text = ""; p = cell.paragraphs[0]
                p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(0)
                add_runs(p, txt, p_italic=True)
                for run in p.runs:
                    run.font.size = DISPLAY_PT
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT   # table columns left-aligned
                edges = (("top", "bottom") if r == 0 else ()) + (("bottom",) if r == len(rows) - 1 else ())
                tcPr = cell._tc.get_or_add_tcPr(); bd = OxmlElement("w:tcBorders")
                for e in ("top", "left", "bottom", "right"):
                    el = OxmlElement(f"w:{e}"); el.set(qn("w:val"), "single" if e in edges else "nil")
                    if e in edges:
                        el.set(qn("w:sz"), "8"); el.set(qn("w:color"), "000000")
                    bd.append(el)
                tcPr.append(bd)
        doc.add_paragraph(); continue
    m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", ln)
    if m:
        depth = len(m.group(1)) // 2
        style = "List Number" if m.group(2)[0].isdigit() else "List Bullet"
        if depth: style += f" {min(depth + 1, 3)}"
        add_runs(doc.add_paragraph(style=style), m.group(3)); i += 1; continue
    p = doc.add_paragraph(); add_runs(p, ln.strip(), p_italic=display); i += 1
    if refs: p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(6)
    if display:
        for run in p.runs:
            run.font.size = DISPLAY_PT
for sec in doc.sections:                       # continuous line numbering (journal requirement)
    ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous")
    sec._sectPr.append(ln)
doc.save(dst)
print("saved", dst)
