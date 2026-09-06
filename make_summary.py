"""
One-page concept summary PDF for DataForge 2026 Pathway Track.
Topic: Linear Attention / Recurrent State vs. KV Cache
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

W, H = A4
MARGIN = 18 * mm

# ── Palette ──────────────────────────────────────────────
DARK   = colors.HexColor('#0B0E1C')
MUTED  = colors.HexColor('#5A637A')
LIN    = colors.HexColor('#1D6ED0')   # blue  — linear attention
SOFT   = colors.HexColor('#C74826')   # orange — softmax
BDH    = colors.HexColor('#5E31BC')   # violet — BDH
BORDER = colors.HexColor('#C6CCEA')
LIGHT  = colors.HexColor('#EFF1F9')

# ── Styles ───────────────────────────────────────────────
def style(name, **kw):
    base = dict(fontName='Helvetica', fontSize=9, leading=13,
                textColor=DARK, spaceAfter=0, spaceBefore=0)
    base.update(kw)
    return ParagraphStyle(name, **base)

S = {
    'eyebrow': style('eyebrow', fontName='Helvetica', fontSize=7,
                     textColor=MUTED, letterSpacing=1.0,
                     spaceAfter=3),
    'title':   style('title', fontName='Helvetica-Bold', fontSize=18,
                     leading=22, spaceAfter=6),
    'claim':   style('claim', fontName='Helvetica-Oblique', fontSize=9.5,
                     leading=14, textColor=MUTED, spaceAfter=0),
    'h2':      style('h2', fontName='Helvetica-Bold', fontSize=8.5,
                     textColor=LIN, letterSpacing=0.5,
                     spaceBefore=8, spaceAfter=3),
    'body':    style('body', fontSize=8.5, leading=13,
                     alignment=TA_JUSTIFY, spaceAfter=3),
    'bullet':  style('bullet', fontSize=8.5, leading=13,
                     leftIndent=10, firstLineIndent=-7, spaceAfter=2),
    'eq':      style('eq', fontName='Courier', fontSize=9,
                     leading=14, textColor=BDH, spaceAfter=2),
    'caption': style('caption', fontSize=7.5, textColor=MUTED,
                     alignment=TA_CENTER),
    'cite':    style('cite', fontSize=7, leading=10, textColor=MUTED,
                     spaceAfter=1),
    'thhdr':   style('thhdr', fontName='Helvetica-Bold', fontSize=7.5,
                     leading=11, textColor=DARK, alignment=TA_CENTER),
    'td':      style('td', fontSize=7.5, leading=11,
                     textColor=DARK, alignment=TA_CENTER),
    'tdl':     style('tdl', fontSize=7.5, leading=11,
                     textColor=DARK, alignment=TA_LEFT),
}

# ── Document ─────────────────────────────────────────────
doc = SimpleDocTemplate(
    '/mnt/user-data/outputs/concept_summary.pdf',
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=14*mm, bottomMargin=14*mm,
    title='Memory vs. State — Concept Summary',
    author='DataForge 2026 Submission',
)

story = []

# ─── Header ───────────────────────────────────────────────
story.append(Paragraph('DATAFORGE 2026 · PATHWAY TRACK · CONCEPT SUMMARY', S['eyebrow']))
story.append(Paragraph('Memory vs. State', S['title']))
story.append(Paragraph(
    'A fixed-size recurrent state can process a sequence of any length without growing its '
    'memory — but it pays by losing the ability to perfectly retrieve any past token on demand.',
    S['claim']
))
story.append(Spacer(1, 4*mm))
story.append(HRFlowable(width='100%', thickness=1, color=BORDER))
story.append(Spacer(1, 3*mm))

# ─── The Problem ──────────────────────────────────────────
story.append(Paragraph('THE PROBLEM', S['h2']))
story.append(Paragraph(
    'Every standard Transformer stores a key vector and value vector for every token it has '
    'processed. That key-value (KV) cache grows linearly: a sequence of n tokens requires '
    'O(n·d) memory and O(n) time per new token to attend over. At long contexts this becomes '
    'the dominant memory cost and a hard engineering constraint for deployment.',
    S['body']
))

# ─── The Mechanism ────────────────────────────────────────
story.append(Paragraph('THE MECHANISM: LINEAR ATTENTION', S['h2']))
story.append(Paragraph(
    'Linear attention replaces the softmax kernel exp(q·k) with a feature map '
    'phi(q)·phi(k), decoupling the query-key product so the sum over tokens can be '
    'computed incrementally. The resulting recurrence:',
    S['body']
))
story.append(Paragraph('S  &lt;-  S + k (x) v', S['eq']))
story.append(Paragraph('o  =  (S · q) / (z · q),   where z &lt;- z + k', S['eq']))
story.append(Paragraph(
    'collapses the entire context into a single d x d state matrix S and a d-dimensional '
    'normalizer z. Memory cost is O(d<super>2</super>) — constant, regardless of sequence '
    'length. Each new token is a Hebbian write (outer-product addition); each read is a '
    'linear associative retrieval.',
    S['body']
))

# ─── Architecture comparison table ────────────────────────
story.append(Paragraph('ARCHITECTURAL COMPARISON', S['h2']))

hdr = ['', 'Softmax Attention', 'Linear Attention', 'BDH / BDH-CQ']
rows = [
    ['Memory per token',  'O(n·d) — grows',  'O(d²) — fixed',  'O(d²) — fixed'],
    ['Retrieval quality', 'Exact',            'Approximate',     'Approximate + sparse'],
    ['Inference cost',    'O(n) per step',    'O(1) per step',   'O(1) per step'],
    ['Long-context',      'Expensive',        'Free',            'Free'],
    ['Test-time adapt',   'No',               'No',              'Yes (BDH-CQ, recurrent)'],
    ['Kernel',            'exp(q·k / sqrt(d))','q·k (linear)',   'q·k + sparse ReLU-low-rank'],
]

col_w = [28*mm, 37*mm, 37*mm, 37*mm]
tbl_data = [[Paragraph(c, S['thhdr']) for c in hdr]]
for i, row in enumerate(rows):
    tbl_data.append([Paragraph(cell, S['tdl'] if j == 0 else S['td'])
                     for j, cell in enumerate(row)])

tbl = Table(tbl_data, colWidths=col_w, repeatRows=1)
tbl.setStyle(TableStyle([
    ('BACKGROUND',  (0,0), (-1,0),  LIGHT),
    ('BACKGROUND',  (2,1), (2,-1),  colors.HexColor('#EBF3FF')),
    ('BACKGROUND',  (3,1), (3,-1),  colors.HexColor('#F0EBFF')),
    ('GRID',        (0,0), (-1,-1), 0.4, BORDER),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7F8FC')]),
    ('TOPPADDING',  (0,0), (-1,-1), 4),
    ('BOTTOMPADDING',(0,0),(-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('RIGHTPADDING',(0,0), (-1,-1), 5),
    ('TEXTCOLOR',   (0,0), (-1,0),  DARK),
    ('TEXTCOLOR',   (1,1), (1,-1),  SOFT),
    ('TEXTCOLOR',   (2,1), (2,-1),  LIN),
    ('TEXTCOLOR',   (3,1), (3,-1),  BDH),
    ('FONTNAME',    (0,1), (0,-1),  'Helvetica-Bold'),
    ('FONTSIZE',    (0,1), (0,-1),  7.5),
]))
story.append(tbl)
story.append(Spacer(1, 3*mm))

# ─── BDH ──────────────────────────────────────────────────
story.append(Paragraph('ROLE OF BDH AND BDH-CQ', S['h2']))
story.append(Paragraph(
    '<b>Dragon Hatchling (BDH)</b> is Pathway\'s Post-Transformer architecture that treats '
    'the recurrent state as the primary computational substrate, not an optimization. '
    'Key distinctions from vanilla linear attention:',
    S['body']
))
bullets = [
    '<b>Sparse non-negative activations.</b> Roughly 5% of neurons fire per token, '
    'reducing write density and mitigating interference in the shared state matrix.',
    '<b>Monosemantic synapses.</b> Connections reportedly encode single semantic concepts '
    'rather than distributed representations, improving state interpretability.',
    '<b>BDH-CQ</b> removes evaluation-task demonstrations from training and performs '
    'adaptation entirely through the recurrent state at inference — no backward pass, '
    'no parameter update. The state is the adapter.',
    '<b>Scale evidence.</b> Publicly reported pretraining experiments range from 1B to 600B '
    'parameters; GPU-friendly formulation (BDH-GPU) uses ReLU-low-rank transformations '
    'with linear attention, deployed via Amazon SageMaker HyperPod.',
]
for b in bullets:
    story.append(Paragraph(u'•  ' + b, S['bullet']))
story.append(Spacer(1, 1*mm))

# ─── Limitations ──────────────────────────────────────────
story.append(Paragraph('LIMITATIONS AND OPEN QUESTIONS', S['h2']))
story.append(Paragraph(
    'Interference accumulates: when many tokens share similar keys, writes collide in the '
    'shared matrix and retrieval degrades. BDH\'s sparse activation reduces but does not '
    'eliminate this. State capacity is fundamentally bounded by d<super>2</super> — '
    'sufficiently dense or long sequences will saturate it. '
    'Cross-session durable learning (consolidating fast state into slow weights) remains '
    'unsolved. BDH\'s reported results are developer-reported; no independent reproduction '
    'of the full system is publicly available as of this submission.',
    S['body']
))

story.append(Spacer(1, 2*mm))
story.append(HRFlowable(width='100%', thickness=0.5, color=BORDER))
story.append(Spacer(1, 2*mm))

# ─── Citations ────────────────────────────────────────────
story.append(Paragraph('PRIMARY SOURCES', S['h2']))
cites = [
    '[1] Katharopoulos et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020. arXiv:2006.16236.',
    '[2] Pathway (2024). Dragon Hatchling (BDH): A Brain-Inspired Post-Transformer Architecture. Pathway Technical Report.',
    '[3] Pathway (2024). BDH-CQ: In-Context Learning via Recurrent State. Pathway Technical Report.',
    '[4] Dao & Gu (2024). Transformers are SSMs: Generalized Models with Structured State Spaces. NeurIPS 2024. arXiv:2405.21060.',
    '[5] Schlag et al. (2021). Linear Transformers are Secretly Fast Weight Programmers. ICML 2021. arXiv:2102.11174.',
    '[6] Sun et al. (2023). Retentive Network: A Successor to Transformer for LLMs. arXiv:2307.08621.',
]
for c in cites:
    story.append(Paragraph(c, S['cite']))

# ─── Build ────────────────────────────────────────────────
doc.build(story)
print("PDF written to /mnt/user-data/outputs/concept_summary.pdf")
