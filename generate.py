from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

OUTPUT = "/mnt/user-data/outputs/ML_Assignment_Report.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm
)

W, H = A4
styles = getSampleStyleSheet()

# ── Custom styles ────────────────────────────────────────────
BLUE       = colors.HexColor("#1A56DB")
DARK       = colors.HexColor("#111827")
GRAY       = colors.HexColor("#6B7280")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
GREEN      = colors.HexColor("#065F46")
GREEN_BG   = colors.HexColor("#D1FAE5")
RED_BG     = colors.HexColor("#FEE2E2")
AMBER_BG   = colors.HexColor("#FEF3C7")
BORDER     = colors.HexColor("#E5E7EB")

def s(name, **kw):
    base = styles[name]
    return ParagraphStyle(name+"_c", parent=base, **kw)

title_style = ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold",
    textColor=DARK, spaceAfter=4, leading=24)
subtitle_style = ParagraphStyle("subtitle", fontSize=11, fontName="Helvetica",
    textColor=GRAY, spaceAfter=2)
meta_style = ParagraphStyle("meta", fontSize=9, fontName="Helvetica",
    textColor=GRAY, spaceAfter=0)
h1_style = ParagraphStyle("h1", fontSize=13, fontName="Helvetica-Bold",
    textColor=BLUE, spaceBefore=14, spaceAfter=6, leading=16)
h2_style = ParagraphStyle("h2", fontSize=11, fontName="Helvetica-Bold",
    textColor=DARK, spaceBefore=8, spaceAfter=4, leading=14)
body_style = ParagraphStyle("body", fontSize=9.5, fontName="Helvetica",
    textColor=DARK, spaceAfter=4, leading=14)
small_style = ParagraphStyle("small", fontSize=8.5, fontName="Helvetica",
    textColor=GRAY, spaceAfter=2, leading=12)
code_style = ParagraphStyle("code", fontSize=8.5, fontName="Courier",
    textColor=colors.HexColor("#1F2937"), backColor=LIGHT_GRAY,
    spaceAfter=4, leading=13, leftIndent=8, rightIndent=8,
    spaceBefore=4, borderPadding=4)
bullet_style = ParagraphStyle("bullet", fontSize=9.5, fontName="Helvetica",
    textColor=DARK, spaceAfter=3, leading=14, leftIndent=12,
    bulletIndent=0)
link_style = ParagraphStyle("link", fontSize=9.5, fontName="Helvetica",
    textColor=BLUE, spaceAfter=3, leading=14)

def HR():
    return HRFlowable(width="100%", thickness=0.5,
                      color=BORDER, spaceAfter=8, spaceBefore=4)

def section(title):
    return [Spacer(1, 6), Paragraph(title, h1_style), HR()]

def bullet(text):
    return Paragraph(f"• {text}", bullet_style)

def tag_table(items, bg):
    """Render a row of badge-style tags."""
    cells = [[Paragraph(i, ParagraphStyle("tag", fontSize=8,
        fontName="Helvetica-Bold", textColor=DARK))] for i in items]
    t = Table([cells], colWidths=[3.2*cm]*len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("ROUNDEDCORNERS", [4,4,4,4]),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("GRID", (0,0), (-1,-1), 0.5, BORDER),
    ]))
    return t

story = []

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
story += [
    Paragraph("Real-Time Conversational Query Expansion", title_style),
    Paragraph("& Hierarchical Topic Classification", title_style),
    Spacer(1, 4),
    Paragraph("ML Engineering Assignment — Submission Report", subtitle_style),
    Paragraph("May 2026", meta_style),
    Spacer(1, 8),
    HR(),
]

# ════════════════════════════════════════════════════════════
# LINKS
# ════════════════════════════════════════════════════════════
story += section("1.  Submission Links")

links_data = [
    ["Resource", "Link"],
    ["Google Colab Notebook", "________________"],
    ["Dataset (CSV)", "________________"],
    ["Streamlit App (Live)", "________________"],
    ["Demo Recording", "________________"],
    ["GitHub Repository", "________________"],
]
links_table = Table(links_data, colWidths=[5*cm, 11.5*cm])
links_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), BLUE),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 9),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT_GRAY]),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
]))
story += [links_table, Spacer(1, 8)]

# ════════════════════════════════════════════════════════════
# PROBLEM STATEMENT
# ════════════════════════════════════════════════════════════
story += section("2.  Problem Statement")
story += [
    Paragraph(
        "For an ongoing dialogue exchange between a user and a bot, perform "
        "real-time expansion of the user query and tag it to the correct "
        "2-level hierarchical topic — all within the context of the last "
        "20 messages (10 dialogue exchanges).",
        body_style),
    Spacer(1, 4),
    Paragraph("Key requirements:", h2_style),
    bullet("Expand ambiguous / incomplete user queries into standalone questions using conversation context"),
    bullet("Classify each query into Level 1 (vertical) and Level 2 (subdomain) topics"),
    bullet("Handle general messages (greetings, interruptions) without misclassification"),
    bullet("Maintain a sliding window of last 20 messages for context"),
    Spacer(1, 6),
]

# ════════════════════════════════════════════════════════════
# SYSTEM ARCHITECTURE
# ════════════════════════════════════════════════════════════
story += section("3.  System Architecture")
story += [
    Paragraph(
        "The system is designed as a sequential pipeline with a clear separation "
        "between the generative (LLM) and discriminative (trained classifier) components.",
        body_style),
    Spacer(1, 6),
]

arch_data = [
    ["Component", "Technology", "Responsibility"],
    ["User Interface", "Streamlit", "Chat window, displays expanded query + topic tag inline"],
    ["Conversation Manager", "SQLite + st.session_state", "Stores all messages, manages session"],
    ["Sliding Window", "Python list (last 20 msgs)", "Builds context for query expander"],
    ["Query Expander", "Claude API (LLM)", "Resolves pronouns, expands ambiguous queries"],
    ["Level 1 Classifier", "DistilBERT (fine-tuned)", "Predicts vertical: Politics / Sports / Tech / Finance / General"],
    ["Level 2 Classifier", "DistilBERT (fine-tuned)", "Predicts subdomain (41 classes)"],
    ["Confidence Gate", "Softmax threshold 0.75", "Falls back to General if L1 confidence < 0.75"],
    ["Taxonomy Gating", "taxonomy.json", "Ensures L2 label belongs to predicted L1 vertical"],
    ["Response Aggregator", "Python dict", "Combines original query, expanded query, L1, L2, confidence"],
]
arch_table = Table(arch_data, colWidths=[3.8*cm, 4.2*cm, 8.5*cm])
arch_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), BLUE),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 8.5),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT_GRAY]),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ("WORDWRAP",     (0,0), (-1,-1), True),
]))
story += [arch_table, Spacer(1, 8)]

story += [
    Paragraph("Pipeline Flow", h2_style),
    Paragraph(
        "User Message → Conversation Manager (SQLite) → Sliding Window Context (last 20 msgs) "
        "→ Query Expander (LLM) → Expanded Query → [L1 Classifier → L2 Classifier] "
        "→ Confidence Check → Response Aggregator → Streamlit UI",
        code_style),
    Spacer(1, 4),
    Paragraph("Key design decision — LLM is called exactly once per message (query expansion only). "
              "Classification uses fast fine-tuned DistilBERT models, making the system "
              "production-grade in terms of latency and cost.", body_style),
    Spacer(1, 6),
]

# Alternative approaches
story += [Paragraph("Alternative Approaches Evaluated", h2_style)]
alt_data = [
    ["Approach", "Why Rejected"],
    ["Pure LLM Classification", "Inconsistent outputs, hallucinations, high latency, expensive per call"],
    ["Multiple LLM Calls", "Better accuracy possible but not production-grade — cost and latency prohibitive"],
    ["Single Flat Classifier", "Poor scalability — 50+ label flat classification loses hierarchical signal"],
    ["Parallel L1+L2 Pipeline", "Weak for incomplete conversational queries; L1 must gate L2 selection"],
]
alt_table = Table(alt_data, colWidths=[5*cm, 11.5*cm])
alt_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#374151")),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 8.5),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT_GRAY]),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
]))
story += [alt_table, Spacer(1, 8)]

# ════════════════════════════════════════════════════════════
# TOPIC TAXONOMY
# ════════════════════════════════════════════════════════════
story += section("4.  Topic Taxonomy")
story += [
    Paragraph("6 verticals, 55 total subdomains. Pre-defined before training — "
              "the classifier only predicts within these known labels.", body_style),
    Spacer(1, 6),
]

tax_data = [
    ["Vertical", "Subdomains (10 each)"],
    ["Politics",
     "Indian Politics · UK Politics · USA Politics · International Relations · Elections & Voting · "
     "Government Policy · Political Parties · Defence & Military · Geopolitics · Judiciary & Law"],
    ["Sports",
     "Cricket · Football · Tennis · Athletics & Olympics · Basketball · Kabaddi & Wrestling · "
     "Badminton · Formula 1 & Motorsport · Hockey · Sports Business"],
    ["Technology",
     "Artificial Intelligence · Smartphones & Gadgets · Space & Exploration · Cybersecurity · "
     "Software & Apps · Social Media & Internet · EV & Clean Tech · Big Tech · Semiconductors & Chips · Telecom & Networks"],
    ["Health / Medicine",
     "Diseases & Conditions · Mental Health · Drugs & Medications · Nutrition & Diet · "
     "Fitness & Exercise · Public Health · Surgery & Procedures · Maternal & Child Health · "
     "Alternative Medicine · Health Policy"],
    ["Finance",
     "Stock Market · Cryptocurrency · Banking & Loans · Personal Finance · Mutual Funds & SIP · "
     "Insurance · Real Estate · Taxation · Global Economy · Startups & VC"],
    ["General",
     "Greetings · Chitchat · Interruptions · Acknowledgements · Out of Scope"],
]
tax_table = Table(tax_data, colWidths=[3.2*cm, 13.3*cm])
tax_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), BLUE),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 8),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),
     [colors.HexColor("#EFF6FF"), colors.HexColor("#F0FDF4"),
      colors.HexColor("#FFF7ED"), colors.HexColor("#FFF1F2"),
      colors.HexColor("#FEFCE8"), LIGHT_GRAY]),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 6),
    ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "TOP"),
]))
story += [tax_table, Spacer(1, 8)]

# ════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════
story += section("5.  Dataset Preparation")
story += [
    Paragraph("Dataset was created from scratch as required by the assignment. "
              "A two-phase approach was used:", body_style),
    Spacer(1, 4),
    Paragraph("Phase 1 — Base Dataset (LLM-generated + manual)", h2_style),
    bullet("Used LLM (Claude + ChatGPT) to generate declarative statements for each subdomain"),
    bullet("Manually wrote and verified examples for ambiguous boundary cases"),
    bullet("Target: ~100 examples per subdomain across all verticals"),
    Spacer(1, 4),
    Paragraph("Phase 2 — Query Augmentation (critical fix)", h2_style),
    bullet("Added 152 real conversational queries (questions, not just declarative statements)"),
    bullet("Reason: initial model predicted 'General' for question-format inputs because "
           "dataset was 97.9% declarative sentences — users type questions"),
    bullet("Added WHO / WHAT / HOW / WHEN style queries for all major subdomains"),
    Spacer(1, 4),
]

ds_data = [
    ["Metric", "Value"],
    ["Total rows (after cleaning)", "~2,900"],
    ["Unique texts", "~2,850 (duplicates removed)"],
    ["Question-format queries", "~250 (augmented)"],
    ["Verticals", "5 (Health / Medicine excluded — no data available)"],
    ["Subdomains", "41 active subdomains"],
    ["Split strategy", "GroupShuffleSplit — 80 / 10 / 10 (train/val/test)"],
    ["Leakage prevention", "Near-duplicates grouped by first-8-word prefix"],
    ["Class imbalance handling", "Weighted CrossEntropyLoss (inverse frequency weights)"],
]
ds_table = Table(ds_data, colWidths=[6.5*cm, 10*cm])
ds_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#374151")),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 9),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT_GRAY]),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
]))
story += [ds_table, Spacer(1, 6)]

story += [
    Paragraph("Important dataset decisions:", h2_style),
    bullet("Removed 901 LLM-generated garbage rows containing filler phrases like "
           "'strongly during' and 'prepared for strongly' — these added noise not signal"),
    bullet("Fixed 1 conflicting label row (same text, 2 different subdomain labels)"),
    bullet("Fixed typo label 'Telecom & Networks3'"),
    bullet("Corrected ChatGPT / OpenAI rows mislabelled as 'Software & Apps' → 'Artificial Intelligence'"),
    Spacer(1, 6),
]

# ════════════════════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════════════════════
story += section("6.  Model Training")
story += [
    Paragraph("Two separate DistilBERT models trained — one for each level of the hierarchy.", body_style),
    Spacer(1, 4),
]

train_data = [
    ["Parameter", "Value"],
    ["Base model", "distilbert-base-uncased"],
    ["Max token length", "64"],
    ["Batch size", "32"],
    ["Epochs", "6 (with EarlyStopping patience=2)"],
    ["Learning rate", "2e-5"],
    ["Warmup ratio", "0.1"],
    ["Weight decay", "0.01"],
    ["Loss function", "CrossEntropyLoss with class weights"],
    ["Best model selection", "Highest F1 Macro on validation set"],
    ["Training platform", "Google Colab (T4 GPU)"],
    ["Training time", "~15 minutes for both models"],
]
train_table = Table(train_data, colWidths=[6*cm, 10.5*cm])
train_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), BLUE),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 9),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT_GRAY]),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
]))
story += [train_table, Spacer(1, 8)]

# ════════════════════════════════════════════════════════════
# ACCURACY
# ════════════════════════════════════════════════════════════
story += section("7.  Accuracy & Results")

acc_data = [
    ["Model", "Accuracy", "F1 Macro", "Classes", "Test Samples"],
    ["Level 1 — Vertical Classifier", "98.62%", "0.9839", "5", "290"],
    ["Level 2 — Subdomain Classifier", "95.52%", "0.9294", "41", "290"],
]
acc_table = Table(acc_data, colWidths=[5.5*cm, 2.5*cm, 2.5*cm, 2*cm, 4*cm])
acc_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), BLUE),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 9),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica-Bold"),
    ("BACKGROUND",   (0,1), (-1,1), GREEN_BG),
    ("BACKGROUND",   (0,2), (-1,2), GREEN_BG),
    ("TEXTCOLOR",    (0,1), (-1,-1), GREEN),
    ("ALIGN",        (1,0), (-1,-1), "CENTER"),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 6),
    ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
]))
story += [acc_table, Spacer(1, 8)]

story += [
    Paragraph("Per-class breakdown (Level 1):", h2_style),
]
l1_data = [
    ["Class", "Precision", "Recall", "F1", "Support"],
    ["Finance",    "0.98", "0.95", "0.96", "57"],
    ["General",    "0.94", "1.00", "0.97", "17"],
    ["Politics",   "0.98", "1.00", "0.99", "84"],
    ["Sports",     "1.00", "1.00", "1.00", "18"],
    ["Technology", "1.00", "0.99", "1.00", "114"],
    ["macro avg",  "0.98", "0.99", "0.98", "290"],
]
l1_table = Table(l1_data, colWidths=[4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
l1_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#374151")),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 9),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("FONTNAME",     (0,-1), (-1,-1), "Helvetica-Bold"),
    ("ROWBACKGROUNDS",(0,1),(-1,-2),[colors.white, LIGHT_GRAY]),
    ("BACKGROUND",   (0,-1), (-1,-1), LIGHT_GRAY),
    ("ALIGN",        (1,0), (-1,-1), "CENTER"),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
]))
story += [l1_table, Spacer(1, 8)]

# ════════════════════════════════════════════════════════════
# INFERENCE TEST CASES
# ════════════════════════════════════════════════════════════
story += section("8.  Sample Inference Test Cases")
story += [
    Paragraph("All test cases run after full training. Format: expanded_query [SEP] bot_answer → predicted topic.", body_style),
    Spacer(1, 6),
]

tc_data = [
    ["Input Query", "Predicted", "Conf.", ""],
    ["Who is the Prime Minister of UK and what are his duties?",
     "Politics > UK Politics", "99.3%", "✓"],
    ["What are the latest developments in Formula 1 this season?",
     "Sports > Formula 1 & Motorsport", "99.3%", "✓"],
    ["Narendra Modi announced new infrastructure projects in Parliament.",
     "Politics > Indian Politics", "97.8%", "✓"],
    ["Virat Kohli scored a century in the Test match.",
     "Sports > Cricket", "99.3%", "✓"],
    ["OpenAI released a new version of ChatGPT with improved reasoning.",
     "Technology > Artificial Intelligence", "98.7%", "✓"],
    ["The RBI increased interest rates to control inflation.",
     "Finance > Banking & Loans", "99.4%", "✓"],
    ["Hey, give me a minute I will be right back.",
     "General > General Greetings", "99.4%", "✓"],
    ["What affects stock market volatility?",
     "Finance > Stock Market", "99.4%", "✓"],
    ["How does an AI chatbot work?",
     "Technology > Artificial Intelligence", "98.4%", "✓"],
    ["What is the role of the Labour Party in Britain?",
     "Politics > UK Politics", "99.4%", "✓"],
    ["Why are NVIDIA chips important for AI companies?",
     "Technology > Semiconductors & Chips", "96.7%", "✓"],
    ["Hello ChatGPT, how are you doing today?",
     "General > General Greetings", "99.4%", "✓"],
]
tc_table = Table(tc_data, colWidths=[7.5*cm, 5.5*cm, 1.5*cm, 1*cm])
tc_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), BLUE),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 8),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT_GRAY]),
    ("TEXTCOLOR",    (3,1), (3,-1), GREEN),
    ("FONTNAME",     (3,1), (3,-1), "Helvetica-Bold"),
    ("ALIGN",        (2,0), (-1,-1), "CENTER"),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
]))
story += [tc_table, Spacer(1, 8)]

# ════════════════════════════════════════════════════════════
# KEY ENGINEERING DECISIONS
# ════════════════════════════════════════════════════════════
story += section("9.  Key Engineering Decisions")
decisions = [
    ("LLM only for expansion, not classification",
     "LLMs hallucinate topic labels and are inconsistent across runs. "
     "DistilBERT with supervised training gives deterministic, fast, cheap inference."),
    ("Two-stage hierarchy (L1 → L2) not flat",
     "A flat 41-class classifier loses inter-class signal. "
     "L1 acts as a router — only L2 labels belonging to the predicted vertical are considered valid (taxonomy gating)."),
    ("GroupShuffleSplit for train/val/test split",
     "Near-duplicate sentences (same template, different entities) must stay in the same split. "
     "Standard StratifiedShuffleSplit caused data leakage — inflated accuracy to 99.7%."),
    ("Weighted CrossEntropyLoss",
     "Class imbalance ratio of 140x between subdomains (Football=140 rows, Personal Finance=40 rows). "
     "Weighted loss ensures rare classes are not ignored during training."),
    ("Confidence threshold at 0.75",
     "If L1 softmax confidence < 0.75, fall back to General > Miscellaneous. "
     "Prevents overconfident wrong predictions on truly out-of-domain queries."),
    ("SQLite for session storage",
     "Zero config, file-based, works perfectly on Streamlit Cloud. "
     "No external DB required — conversation history persists across page reloads."),
]
for title, desc in decisions:
    story += [
        Paragraph(f"<b>{title}</b>", body_style),
        Paragraph(desc, small_style),
        Spacer(1, 4),
    ]

# ════════════════════════════════════════════════════════════
# KNOWN LIMITATIONS + HOW TO IMPROVE
# ════════════════════════════════════════════════════════════
story += section("10.  Known Limitations & How to Improve")

story += [
    Paragraph("Current limitations:", h2_style),
    bullet("Health / Medicine vertical not trained — 0 rows in dataset. "
           "Any health query will be misclassified as General or Technology."),
    bullet("L2 confidence is low (12-50%) on many correct predictions — "
           "the model is right but uncertain, suggesting more training data needed per subdomain."),
    bullet("Dataset is LLM-generated — may not fully represent real user conversational patterns. "
           "Human-annotated data would significantly improve robustness."),
    Spacer(1, 6),
    Paragraph("How to push accuracy further:", h2_style),
    bullet("Add Health / Medicine data (~500 rows) and retrain — will raise L1 to 6 verticals"),
    bullet("Increase subdomain data from 40-100 rows to 200+ rows for all thin classes"),
    bullet("Use sentence-transformers or RoBERTa base instead of DistilBERT for better semantic understanding"),
    bullet("Add hard negative examples — boundary cases like 'Elon Musk Tesla stock' "
           "(Finance > Stock Market, not Technology)"),
    bullet("Implement active learning — collect real user queries from the Streamlit app, "
           "label them, add to training data iteratively"),
    Spacer(1, 6),
]

# ════════════════════════════════════════════════════════════
# TECH STACK SUMMARY
# ════════════════════════════════════════════════════════════
story += section("11.  Tech Stack")
stack_data = [
    ["Layer", "Technology"],
    ["UI", "Streamlit"],
    ["LLM / Query Expansion", "Anthropic Claude API (claude-sonnet-4-20250514)"],
    ["Classifier", "HuggingFace Transformers — DistilBERT"],
    ["Training", "HuggingFace Trainer API + Google Colab T4 GPU"],
    ["Database", "SQLite (conversations.db)"],
    ["Data processing", "pandas, scikit-learn, numpy"],
    ["Model serialization", "safetensors + pickle (LabelEncoders)"],
    ["Deployment", "Streamlit Cloud"],
]
stack_table = Table(stack_data, colWidths=[4*cm, 12.5*cm])
stack_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), BLUE),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 9),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT_GRAY]),
    ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
    ("TOPPADDING",   (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
]))
story += [stack_table, Spacer(1, 8)]

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
story += [
    HR(),
    Paragraph(
        "This report was prepared as part of the AI/ML Engineering Assignment. "
        "All code, dataset, and models were built from scratch. "
        "The system is designed to be production-oriented — "
        "not just a proof of concept.",
        small_style),
]

doc.build(story)
print(f"PDF saved → {OUTPUT}")