"""
Blog post PDF for DataForge 2026 Pathway Track.
Title: What Your Transformer Is Forgetting (And Why That's Actually Fine)
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

W, H = A4
MARGIN = 22 * mm

DARK   = colors.HexColor('#0B0E1C')
MUTED  = colors.HexColor('#5A637A')
LIN    = colors.HexColor('#1D6ED0')
SOFT   = colors.HexColor('#C74826')
BDH    = colors.HexColor('#5E31BC')
BORDER = colors.HexColor('#C6CCEA')
LIGHT  = colors.HexColor('#EFF1F9')

def style(name, **kw):
    base = dict(fontName='Helvetica', fontSize=9.5, leading=15,
                textColor=DARK, spaceAfter=0, spaceBefore=0,
                alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)

S = {
    'eyebrow': style('eyebrow', fontName='Helvetica', fontSize=7,
                     textColor=MUTED, letterSpacing=1.0,
                     alignment=TA_LEFT, spaceAfter=4),
    'title':   style('title', fontName='Helvetica-Bold', fontSize=20,
                     leading=25, alignment=TA_LEFT, spaceAfter=5),
    'deck':    style('deck', fontName='Helvetica-Oblique', fontSize=10.5,
                     leading=16, textColor=MUTED,
                     alignment=TA_LEFT, spaceAfter=0),
    'byline':  style('byline', fontSize=8, textColor=MUTED,
                     alignment=TA_LEFT, spaceAfter=0),
    'h2':      style('h2', fontName='Helvetica-Bold', fontSize=10,
                     textColor=LIN, spaceBefore=10, spaceAfter=4,
                     alignment=TA_LEFT),
    'body':    style('body', fontSize=9.5, leading=15,
                     alignment=TA_JUSTIFY, spaceAfter=6),
    'pull':    style('pull', fontName='Helvetica-Oblique', fontSize=11,
                     leading=17, textColor=BDH,
                     leftIndent=12, rightIndent=12,
                     spaceBefore=6, spaceAfter=6,
                     alignment=TA_JUSTIFY),
    'eq':      style('eq', fontName='Courier', fontSize=9.5,
                     leading=15, textColor=BDH,
                     leftIndent=20, spaceAfter=3,
                     alignment=TA_LEFT),
    'cite':    style('cite', fontSize=7.5, leading=11, textColor=MUTED,
                     spaceAfter=2, alignment=TA_LEFT),
}

doc = SimpleDocTemplate(
    '/mnt/user-data/outputs/blog_post.pdf',
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=16*mm, bottomMargin=16*mm,
    title='What Your Transformer Is Forgetting',
    author='DataForge 2026 Submission',
)

story = []

# Header
story.append(Paragraph('DATAFORGE 2026 · PATHWAY TRACK · BLOG POST', S['eyebrow']))
story.append(Paragraph('What Your Transformer Is Forgetting\n(And Why That\'s Actually Fine)', S['title']))
story.append(Paragraph(
    'Standard attention remembers everything perfectly — and that perfectionism is '
    'quietly breaking your GPU budget.',
    S['deck']
))
story.append(Spacer(1, 3*mm))
story.append(Paragraph('DataForge 2026 Submission — Memory vs. State', S['byline']))
story.append(Spacer(1, 4*mm))
story.append(HRFlowable(width='100%', thickness=1, color=BORDER))
story.append(Spacer(1, 4*mm))

# Section 1
story.append(Paragraph('The filing cabinet that never stops growing', S['h2']))
story.append(Paragraph(
    'Imagine a student who writes down every word of every lecture, word for word, '
    'in a separate notebook. Their notes are perfect. Any question you ask, they can '
    'flip back and find the exact sentence. The problem is that by week ten, they need '
    'a wheelbarrow to carry their notebooks to class.',
    S['body']
))
story.append(Paragraph(
    'This is a reasonable description of what a standard Transformer does with its '
    'KV cache. For every token in the sequence, the model stores a key vector and a '
    'value vector. Ask it to process a document with n tokens and it holds n pairs in '
    'memory — growing linearly, with no upper bound. At 100k-token contexts, the cache '
    'can consume more GPU RAM than the model weights themselves.',
    S['body']
))
story.append(Paragraph(
    'This is not a bug. It is the direct consequence of the exact retrieval guarantee '
    'that makes softmax attention so accurate. The question is whether we actually need '
    'that guarantee all the time — or whether a student who remembers the gist perfectly '
    'is good enough for most exams.',
    S['body']
))

# Section 2
story.append(Paragraph('Collapsing the filing cabinet into a single page', S['h2']))
story.append(Paragraph(
    'Linear attention\'s insight is deceptively simple: the reason the KV cache grows '
    'is that softmax couples every query to every key. If we replace the softmax kernel '
    'with a linear one, the sum over all past tokens can be computed incrementally — '
    'and the result is a fixed-size state matrix.',
    S['body']
))
story.append(Paragraph(
    'Each new token writes to this matrix via an outer product:',
    S['body']
))
story.append(Paragraph('S  <-  S + k (x) v', S['eq']))
story.append(Paragraph(
    'and any query retrieves from it with a simple dot product:',
    S['body']
))
story.append(Paragraph('o  =  (S * q) / (z * q),   z  <-  z + k', S['eq']))
story.append(Paragraph(
    'The state matrix S has shape d x d. Its memory cost is O(d<super>2</super>) — '
    'constant, no matter how many tokens have been processed. The wheelbarrow is gone. '
    'In its place is a single dense page of notes.',
    S['body']
))

story.append(Paragraph(
    'The write operation is Hebbian: each token adds its key-value contribution to the '
    'shared matrix. The read operation is associative: a query key selects a weighted '
    'blend of everything written so far. Neuroscientists will recognize this as a '
    'content-addressable memory — the same principle that underlies Hopfield networks '
    'and, according to Pathway\'s interpretation, how biological synapses store '
    'episodic associations.',
    S['body']
))

# Pull quote
story.append(Paragraph(
    '"The state is not a cache. It is a compression — and like all compressions, '
    'it introduces a loss."',
    S['pull']
))

# Section 3
story.append(Paragraph('What the page cannot hold', S['h2']))
story.append(Paragraph(
    'The loss is interference. If two tokens share similar key vectors — say, the '
    'tokens "orange" and "yellow" in a sequence — their outer-product writes overlap '
    'in the state matrix. A later query for "orange" retrieves a blend: part orange, '
    'part yellow, with weights proportional to their key similarity. The student\'s '
    'single-page notes have conflated two different lecture points.',
    S['body']
))
story.append(Paragraph(
    'Softmax attention avoids this because the exponential function amplifies '
    'differences between scores. Two keys that are 95% similar get scores with a much '
    'larger relative gap after exponentiation. The highest-scoring key wins '
    'overwhelmingly. Linear attention has no such sharpening — it reports exact '
    'proportions, so similar keys genuinely interfere.',
    S['body']
))
story.append(Paragraph(
    'State capacity is the second limit. A d x d matrix can store at most d<super>2</super> '
    'independent pieces of information. Write more than that — with enough distinct, '
    'orthogonal tokens — and earlier writes are progressively overwritten. In practice, '
    'real sequences are far from maximally orthogonal, so the effective capacity is '
    'higher, but it remains bounded.',
    S['body']
))

# Section 4
story.append(Paragraph('Dragon Hatchling: when the state becomes the architecture', S['h2']))
story.append(Paragraph(
    'Pathway\'s Dragon Hatchling (BDH) does not treat the recurrent state as an '
    'optimization of a Transformer. It treats the state as the primary computational '
    'substrate — the thing the model IS, rather than a side effect of how it runs.',
    S['body']
))
story.append(Paragraph(
    'Two design decisions address the interference problem directly. First, sparse '
    'non-negative activations: roughly 5% of neurons fire on any given token. This '
    'reduces write density dramatically — most of the state matrix stays untouched '
    'for most tokens, which means collisions between similar keys are much rarer. '
    'Second, monosemantic synapses: connections in BDH reportedly encode single '
    'semantic concepts rather than the superposed, distributed representations '
    'typical of Transformer MLP layers. This further separates the storage regions '
    'for different concepts.',
    S['body']
))
story.append(Paragraph(
    'BDH-CQ pushes this further: it performs in-context learning — adapting its '
    'behavior based on examples provided at inference time — entirely through the '
    'recurrent state, with no gradient update. The state is the adapter. This is a '
    'meaningful departure from standard in-context learning, where the model\'s '
    'parameters never change but the KV cache carries the demonstration examples. '
    'In BDH-CQ, the demonstrations are compressed into state and consumed; the '
    'inference loop sees only the final query.',
    S['body']
))

# Section 5
story.append(Paragraph('What remains open', S['h2']))
story.append(Paragraph(
    'Linear attention\'s core trade-off — constant memory against approximate retrieval '
    '— is well characterized. BDH\'s architectural choices (sparsity, monosemanticity) '
    'are theoretically motivated, but independent reproduction of the full system at '
    'the reported scales (1B to 600B parameters) is not yet publicly available. '
    'Cross-session consolidation — moving fast, recurrent state into slow, durable '
    'weights — remains an unsolved problem for all state-space and linear attention '
    'architectures. And the right balance between state size and sequence length '
    'for production deployments is still an active empirical question.',
    S['body']
))
story.append(Paragraph(
    'None of this diminishes the core insight: for many tasks, the student with one '
    'well-organized page of notes outperforms the student dragging a wheelbarrow — '
    'not because they remember more, but because they show up.',
    S['body']
))

story.append(Spacer(1, 3*mm))
story.append(HRFlowable(width='100%', thickness=0.5, color=BORDER))
story.append(Spacer(1, 3*mm))

story.append(Paragraph('REFERENCES', style('refhdr', fontName='Helvetica-Bold',
             fontSize=7.5, textColor=LIN, letterSpacing=0.5,
             alignment=TA_LEFT, spaceAfter=3)))

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

doc.build(story)
print("PDF written to /mnt/user-data/outputs/blog_post.pdf")
