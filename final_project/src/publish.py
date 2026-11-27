"""Build the report and landscape poster from editable prose and data-driven panels."""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from matplotlib import get_data_path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from settings import RESULTS, ROOT


INK = colors.HexColor("#192e41")
TEAL = colors.HexColor("#10695f")
MUTED = colors.HexColor("#516679")
BLUE = colors.HexColor("#214d76")
RULE = colors.HexColor("#d5e1e6")
PALE = colors.HexColor("#f0f7f7")
WHITE = colors.white
AUTHORS = "Yu-Hsin Chen, Ziyi Huang, and Wenxin Zhang"
TITLE = "When Do Structural Features Complement Protein Language Models in Predicting Mutational Effects?"


def register_fonts() -> None:
    target = Path("/usr/share/fonts/truetype/lato")
    sources = {"Paper": "Regular", "PaperBold": "Bold", "PaperItalic": "Italic", "PaperBoldItalic": "BoldItalic"}
    use_lato = all((target / f"Lato-{variant}.ttf").is_file() for variant in sources.values())
    for name, style in sources.items():
        if use_lato:
            path = target / f"Lato-{style}.ttf"
        else:
            suffix = {"Regular": "", "Bold": "-Bold", "Italic": "-Oblique", "BoldItalic": "-BoldOblique"}[style]
            path = Path(get_data_path()) / "fonts" / "ttf" / f"DejaVuSans{suffix}.ttf"
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily("Paper", normal="Paper", bold="PaperBold", italic="PaperItalic", boldItalic="PaperBoldItalic")


def rich(text: str) -> str:
    text = escape(text.strip())
    text = re.sub(r"\[([^\[\]]+)\]\((https://[^\s)]+)\)", r'<link href="\2" color="#10695f">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)


def style(name: str, size: float, leading: float, *, font="Paper", color=INK, after=0, before=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading, textColor=color,
                          spaceAfter=after, spaceBefore=before, keepWithNext=name in {"title", "h1", "h2"})


def paper_header(pdf, doc) -> None:
    width, height = letter
    pdf.setStrokeColor(RULE)
    pdf.line(53, height - 44, width - 53, height - 44)
    pdf.setFillColor(TEAL)
    pdf.setFont("PaperBold", 8)
    pdf.drawString(53, height - 35, "COBB 2035   /   STRUCTURE AND MUTATIONAL EFFECTS")
    pdf.setFillColor(MUTED)
    pdf.setFont("Paper", 8)
    pdf.drawString(53, 31, "Chen, Huang, and Zhang")
    pdf.drawRightString(width - 53, 31, str(doc.page))


def paper_figure(source: str, caption: str, caption_style):
    path = ROOT / source
    file = ImageReader(str(path))
    width, height = file.getSize()
    max_height = {"fig01": 184, "fig02": 211, "fig03": 142, "fig04": 176, "fig05": 153}
    scale = min(498 / width, max_height[path.name[:5]] / height)
    panel = Image(str(path), width=width * scale, height=height * scale, hAlign="CENTER")
    return KeepTogether([Spacer(1, 3), panel, Spacer(1, 3), Paragraph(rich(caption), caption_style)])


def paper_table(chunk: str, cell_style):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for i, line in enumerate(chunk.splitlines()) if i != 1]
    cells = [[Paragraph(rich(cell), cell_style if i else style("tablehead", 9.1, 10.7, font="PaperBold", color=WHITE))
              for cell in row] for i, row in enumerate(rows)]
    table = Table(cells, colWidths=[97, 106, 89, 92, 112], hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), TEAL),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, WHITE]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE)]))
    return table


def build_report() -> None:
    destination = RESULTS / "final_report.pdf"
    document = SimpleDocTemplate(str(destination), pagesize=letter, leftMargin=53, rightMargin=53,
                                 topMargin=62, bottomMargin=49, title=TITLE, author=AUTHORS)
    styles = {"title": style("title", 26, 30.5, font="PaperBold", after=10),
              "author": style("author", 10.2, 15, color=TEAL, after=10),
              "h1": style("h1", 15.7, 19, font="PaperBold", color=TEAL, before=8, after=6),
              "h2": style("h2", 11.6, 14.2, font="PaperBold", before=6, after=5),
              "body": style("body", 10.3, 13.5, after=7),
              "caption": style("caption", 8.55, 11.4, color=MUTED, after=6),
              "reference": style("reference", 8.45, 10.6, after=2),
              "cell": style("cell", 9.1, 10.7)}
    story = []
    for chunk in re.split(r"\n\s*\n", (ROOT / "REPORT.md").read_text().strip()):
        if chunk == "<!-- pagebreak -->":
            story.append(PageBreak())
            continue
        if chunk.startswith("|"):
            story.append(paper_table(chunk, styles["cell"]))
            continue
        if match := re.fullmatch(r"!\[(.*)\]\(([^()]+)\)", chunk):
            story.append(paper_figure(match[2], match[1], styles["caption"]))
            continue
        if re.match(r"^1\. ", chunk):
            story.extend(Paragraph(rich(line), styles["reference"]) for line in chunk.splitlines())
            continue
        role, text = "body", chunk
        if chunk.startswith(("# ", "## ", "### ")):
            level, text = chunk.split(" ", 1)
            role = {"#": "title", "##": "h1", "###": "h2"}[level]
        elif chunk.startswith(AUTHORS):
            role = "author"
        story.append(Paragraph(rich(text), styles[role]))
    document.build(story, onFirstPage=paper_header, onLaterPages=paper_header)
    print(f"Wrote {destination}")


class Poster:
    width, height = 48 * inch, 36 * inch

    def __init__(self, path: Path):
        self.pdf = canvas.Canvas(str(path), pagesize=(self.width, self.height))
        self.pdf.setTitle(TITLE)
        self.pdf.setAuthor(AUTHORS)

    def paragraph(self, text, x, top, width, size=32, leading=43, *, font="Paper", color=INK, align=0):
        options = style("poster", size, leading, font=font, color=color)
        options.alignment = align
        block = Paragraph(rich(text), options)
        _, height = block.wrap(width, self.height)
        block.drawOn(self.pdf, x, self.height - top - height)
        return top + height

    def line(self, x, top, width, color=RULE, thickness=2.2):
        self.pdf.setStrokeColor(color)
        self.pdf.setLineWidth(thickness)
        self.pdf.line(x, self.height - top, x + width, self.height - top)

    def rect(self, x, top, width, height, color, radius=0):
        self.pdf.setFillColor(color)
        if radius:
            self.pdf.roundRect(x, self.height - top - height, width, height, radius, fill=True, stroke=False)
        else:
            self.pdf.rect(x, self.height - top - height, width, height, fill=True, stroke=False)

    def section(self, text, x, top, width):
        top = self.paragraph(text, x, top, width, 46, 52, font="PaperBold", color=TEAL)
        self.line(x, top + 9, width)
        return top + 27

    def figure(self, name, caption, x, top, width):
        asset = ImageReader(str(RESULTS / "figures" / f"{name}.png"))
        pixel_width, pixel_height = asset.getSize()
        height = width * pixel_height / pixel_width
        self.pdf.drawImage(asset, x, self.height - top - height, width, height)
        return self.paragraph(caption, x + 8, top + height + 9, width - 16, 27, 34, color=MUTED)


def build_poster() -> None:
    destination = RESULTS / "poster_48x36in.pdf"
    poster = Poster(destination)
    margin, x2, x3 = 125, 1132, 2359
    w1, w2, w3 = 925, 1145, 972
    poster.rect(0, 0, poster.width, 18, TEAL)
    poster.rect(0, 18, poster.width, 366, PALE)
    poster.paragraph("COBB 2035  /  FALL 2026 PROJECT", margin, 52, 3000, 29, 36, font="PaperBold", color=TEAL)
    title_bottom = poster.paragraph(TITLE, margin, 107, 3130, 78, 84, font="PaperBold")
    poster.paragraph(AUTHORS, margin, title_bottom + 24, 3000, 38, 42, color=MUTED)
    poster.line(margin, 430, poster.width - 2 * margin, TEAL, 4)

    y = poster.section("Research question", margin, 480, w1)
    y = poster.paragraph("When does wild-type structure improve a sequence language model’s prediction of single-substitution effects? We vary the predictor, number of measured training variants, and difficulty of the test split.", margin, y, w1)
    y = poster.section("Four ProteinGym assays", margin, y + 63, w1)
    assay_table = pd.read_csv(RESULTS / "tables" / "assay_metadata.csv")
    for row in assay_table.itertuples(index=False):
        poster.rect(margin, y + 3, w1, 76, PALE, radius=10)
        poster.paragraph(row.label, margin + 19, y + 19, 130, 32, 37, font="PaperBold")
        poster.paragraph(row.readout, margin + 148, y + 19, 525, 31, 36)
        poster.paragraph(f"{row.mutated_positions} sites", margin + 700, y + 20, 200, 28, 35, color=MUTED)
        y += 86
    y = poster.paragraph("1,500 observed single substitutions per assay. Reference geometry came from the ProteinGym AlphaFold2 structures.", margin, y + 14, w1, 30, 40)

    y = poster.section("Representations", margin, y + 61, w1)
    descriptions = [
        ("D", "Amino-acid and substitution descriptors"),
        ("E", "Frozen ESM-2 35M mutant-minus-WT embedding"),
        ("P", "Frozen SaProt 35M embedding with WT 3Di states"),
        ("S", "Solvent accessibility, secondary structure, and Cα contacts"),
    ]
    for symbol, description in descriptions:
        poster.rect(margin, y + 2, 54, 50, TEAL, radius=8)
        poster.paragraph(symbol, margin, y + 9, 54, 30, 33, font="PaperBold", color=WHITE, align=TA_CENTER)
        y = poster.paragraph(description, margin + 75, y + 5, w1 - 75, 30, 40) + 20

    y = poster.section("Evaluation", margin, y + 33, w1)
    y = poster.paragraph("Ridge, gradient-boosted trees, and an MLP with dropout and weight decay. We fit each assay independently and reused five seeded test splits across all models and feature sets.", margin, y, w1, 30, 40)
    y = poster.paragraph("We compared random variant testing with whole residue positions held out. Tuning and scaling stayed inside each training budget. Scores are mean test Spearman ρ.", margin, y + 23, w1, 30, 40)

    y2 = poster.section("Predictive accuracy", x2, 480, w2)
    y2 = poster.figure("fig01_performance", "Figure 1. Performance at 500 labels. GBT with D+E+S reached 0.403 on unseen positions. The GBT random-split leader used D+S (0.537).", x2, y2, w2)
    y2 = poster.section("Sensitivity to the label budget", x2, y2 + 53, w2)
    y2 = poster.figure("fig02_learning_curves", "Figure 2. Paired change in Spearman ρ relative to D+E. Shading shows one SD across five seed-level assay averages for conventional structure and SaProt.", x2, y2, w2)
    y2 = poster.paragraph("Conventional structure raised held-out scores at every tested budget. SaProt gains averaged near zero on held-out positions at 500 labels, with larger gains under random testing.", x2, y2 + 23, w2, 31, 41)

    y3 = poster.section("Assay and site differences", x3, 480, w3)
    y3 = poster.paragraph("At 500 labels on held-out positions, adding conventional structure gained **+0.237 to +0.267** in EstA across the three models. TrpC showed a small loss across all three.", x3, y3, w3, 31, 42)
    y3 = poster.figure("fig03_environments", "Figure 3. Buried versus exposed sites at 500 labels. This comparison uses only folds with enough positions in both groups. TrpC had too few exposed test positions.", x3, y3 + 30, w3)
    y3 = poster.section("Two buried EstA substitutions", x3, y3 + 43, w3)
    y3 = poster.figure("fig05_structural_cases", "Figure 4. Adding structure improved the prediction for S158Q and worsened it for G135M. Ranks are means across two held-out appearances.", x3, y3, w3)
    y3 += 47
    poster.rect(x3, y3, w3, 358, PALE, radius=21)
    y3 = poster.paragraph("Conventional structure added the most at unseen sites", x3 + 30, y3 + 23, w3 - 60,
                          36, 42, font="PaperBold", color=TEAL)
    y3 = poster.paragraph("+0.082 to +0.101 Δρ across heads at 500 labels. Adding SaProt to D+E gave −0.009 to +0.009. The estimates apply to these four assays and this frozen-embedding setup.",
                          x3 + 30, y3 + 20, w3 - 60, 29, 38)

    if max(y, y2, y3) > 2290:
        raise ValueError(f"Poster content extends below the body area: {y:.0f}, {y2:.0f}, {y3:.0f}")
    poster.line(margin, 2372, poster.width - 2 * margin, TEAL, 3)
    poster.paragraph("References  ·  Notin et al., ProteinGym (NeurIPS 2023)  ·  Lin et al., ESM-2 (Science 2023)  ·  Su et al., SaProt (ICLR 2024)  ·  van Kempen et al., Foldseek (Nat. Biotechnol. 2024)",
                     margin, 2410, poster.width - 2 * margin, 26, 35, color=MUTED)
    poster.paragraph("Codex assisted with code and drafting. Panels were generated from data and reference coordinates using the reproducible project scripts.",
                     margin, 2467, poster.width - 2 * margin, 24, 33, color=MUTED)
    poster.pdf.showPage()
    poster.pdf.save()
    print(f"Wrote {destination}")


def main() -> None:
    register_fonts()
    build_report()
    build_poster()


if __name__ == "__main__":
    main()
