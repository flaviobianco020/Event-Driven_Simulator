#!/usr/bin/env python3
"""Piano formale Fase 4 — supervisore LLM (SLM) sul percorso lento."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

OUT = "/Users/flaviobianco/Desktop/Piano_Fase4_Supervisore_LLM.pdf"
W, H = A4
base = getSampleStyleSheet()


def s_(n, p="Normal", **k):
    return ParagraphStyle(n, parent=base[p], **k)


TITLE = s_("T", "Title", fontSize=19, spaceAfter=8, leading=25, textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
SUB = s_("S", "Normal", fontSize=12, spaceAfter=4, textColor=colors.HexColor("#444466"), alignment=TA_CENTER)
AUTH = s_("A", "Normal", fontSize=10.5, spaceAfter=3, leading=15, textColor=colors.HexColor("#333355"), alignment=TA_CENTER)
H1 = s_("H1", "Heading1", fontSize=15, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1a1a2e"))
H2 = s_("H2", "Heading2", fontSize=12.5, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#2d2d5e"))
BODY = s_("B", "Normal", fontSize=10.5, leading=16, spaceAfter=7, alignment=TA_JUSTIFY)
LI = s_("LI", "Normal", fontSize=10.5, leading=15, spaceAfter=3, leftIndent=14, alignment=TA_JUSTIFY)
MATH = s_("M", "Normal", fontSize=9.5, leading=13, spaceAfter=7, fontName="Courier", leftIndent=24, textColor=colors.HexColor("#222244"))
CAP = s_("C", "Normal", fontSize=8.5, spaceAfter=10, alignment=TA_CENTER, textColor=colors.HexColor("#666666"))
REF = s_("R", "Normal", fontSize=9, leading=13, spaceAfter=5, leftIndent=18, firstLineIndent=-18)
ABS = s_("AB", "Normal", fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY, leftIndent=20, rightIndent=20, textColor=colors.HexColor("#333333"))
KEYS = s_("K", "Normal", fontSize=10, leading=15, spaceAfter=8, alignment=TA_JUSTIFY, leftIndent=16, rightIndent=10,
          borderWidth=0.8, borderColor=colors.HexColor("#4a8a6a"), borderPadding=7, backColor=colors.HexColor("#f0f8f3"))


def sp(h=6): return Spacer(1, h)
def hr(): return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cccccc"), spaceAfter=6, spaceBefore=2)
def P(t, st=BODY): return Paragraph(t, st)
def KEY(t): return Paragraph(t, KEYS)


def tstyle():
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d2d5e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])


def tbl(data, widths):
    from reportlab.platypus import Paragraph as PP
    cell = ParagraphStyle("cell", parent=base["Normal"], fontSize=8.5, leading=11)
    hc = ParagraphStyle("hc", parent=base["Normal"], fontSize=8.5, leading=11, fontName="Helvetica-Bold", textColor=colors.white)
    wr = [[PP(str(c).replace("\n", "<br/>"), hc if r == 0 else cell) for c in row] for r, row in enumerate(data)]
    return Table(wr, colWidths=widths, style=tstyle(), repeatRows=1)


st = []

# COPERTINA
st += [
    sp(48),
    P("Fase 4 — Supervisore LLM per la Gestione Autonoma della Congestione", TITLE),
    sp(8), P("Piano tecnico: architettura ibrida a due livelli con Small Language Model", SUB),
    sp(18), hr(), sp(8),
    P("<b>Autore:</b> Flavio Bianco", AUTH),
    P("<b>Tesi:</b> <i>Towards Agentic Networks: Autonomous Congestion Management</i>", AUTH),
    P("<b>Anno Accademico 2025/2026</b>", AUTH),
    sp(20), hr(), sp(8),
    P("<b>Abstract</b>", s_("AH", "Normal", fontSize=10.5, alignment=TA_CENTER, spaceAfter=6)),
    P("La Fase 3 ha prodotto una politica MAPPO validata su hardware, che supera il controllo a "
      "regole. I suoi limiti — politica congelata, assenza di spiegabilita', fragilita' fuori dalla "
      "distribuzione di addestramento — motivano la Fase 4. Si propone un'architettura <b>ibrida a due "
      "livelli</b>: la policy MAPPO resta il percorso veloce (decisione per secondo, deterministica, "
      "deployata come artefatto leggero), mentre un <b>supervisore basato su Small Language Model</b> "
      "opera sul percorso lento (monitoraggio, spiegazione, correzione dei casi fuori distribuzione). "
      "L'LLM non entra mai nel ciclo per-secondo: non puo' quindi degradare le prestazioni gia' "
      "verificate. Si adotta un modello piccolo (1.5-3B, locale, riproducibile) con constrained "
      "decoding, e la dimensione del modello diventa una variabile sperimentale — un'ablation che "
      "risponde alla domanda 'qual e' il supervisore piu' piccolo sufficiente?', allineata alla "
      "letteratura sugli Small Language Model per l'AI agentica.", ABS),
    PageBreak(),
]

# 1. MOTIVAZIONE
st += [
    P("1. Motivazione: i limiti della Fase 3", H1),
    P("La politica MAPPO della Fase 3 e' efficace <b>dentro</b> la sua distribuzione di addestramento, "
      "ma presenta tre limiti strutturali, intrinseci al paradigma dell'apprendimento per rinforzo:"),
    P("• <b>Politica congelata</b>: dopo l'addestramento i pesi sono fissi; non si adatta a scenari "
      "radicalmente nuovi senza un nuovo training.", LI),
    P("• <b>Assenza di spiegabilita'</b>: e' una rete neurale opaca; non puo' giustificare le proprie "
      "decisioni a un operatore.", LI),
    P("• <b>Generalizzazione limitata</b>: su situazioni fuori distribuzione (traffico, guasti, "
      "topologie mai visti) la policy congelata puo' comportarsi male.", LI),
    KEY("Punto chiave: questi limiti NON si risolvono con un altro algoritmo di apprendimento per "
        "rinforzo — sono del paradigma, non dell'algoritmo. Richiedono capacita' di natura diversa "
        "(ragionamento, spiegazione, adattamento senza retraining). E' precisamente cio' che un agente "
        "linguistico fornisce. La Fase 4 non sostituisce la Fase 3: la <b>estende</b>."),
]

# 2. ARCHITETTURA
st += [
    P("2. Architettura ibrida a due livelli", H1),
    P("L'architettura separa nettamente due percorsi, secondo l'analogia System 1 / System 2 "
      "(Kahneman): il riflesso veloce e il ragionamento lento."),
    tbl([
        ["", "Percorso VELOCE (System 1)", "Percorso LENTO (System 2)"],
        ["Chi", "Actor MAPPO (Fase 3)", "Supervisore LLM (Fase 4)"],
        ["Cadenza", "ogni 1 secondo", "ogni ~30 s o su anomalia"],
        ["Natura", "deterministica, congelata", "ragionamento, adattiva"],
        ["Costo", "trascurabile (24 KB, Python puro)", "un'inferenza LLM al tick lento"],
        ["Ruolo", "decide la compressione", "monitora, spiega, corregge OOD"],
        ["Impatto Fase 4", "INTATTO", "nuovo livello supervisorio"],
    ], [2.6*cm, 6.6*cm, 6.8*cm]),
    P("Tabella 1 — I due livelli. L'LLM non entra mai nel ciclo per-secondo.", CAP),
    KEY("Garanzia fondamentale: poiche' l'LLM opera solo sul percorso lento e il percorso veloce "
        "resta quello della Fase 3, <b>nessuna scelta di modello LLM puo' degradare</b> la latenza "
        "(447 ms) o la consegna (PDR 0,943) gia' misurate. L'LLM gestisce il caso comune con "
        "'endorse' (nessun intervento) e agisce solo sull'eccezione."),
    PageBreak(),
]

# 3. IL SUPERVISORE
st += [
    P("3. Il supervisore: azioni e deliverable", H1),
    P("3.1 Spazio di azione vincolato", H2),
    P("Il supervisore riceve solo <b>metriche aggregate</b> (mai il contenuto dei pacchetti — cosi' "
      "non esiste superficie di prompt injection) e sceglie una di cinque azioni:"),
    tbl([
        ["Azione", "Effetto", "Percorso veloce?"],
        ["endorse", "MAPPO ok, nessun intervento (caso comune)", "no"],
        ["override_state(k, durata)", "forza uno stato di compressione per una finestra breve", "SI (vincolato)"],
        ["flag_retrain", "segnala regime fuori distribuzione per il retrain offline", "no"],
        ["explain", "sola giustificazione in linguaggio naturale", "no"],
        ["coordinate(msg)", "messaggio a un peer router (topologie mesh)", "no"],
    ], [4.6*cm, 8.6*cm, 2.8*cm]),
    P("Tabella 2 — Le cinque azioni supervisorie. Solo override_state tocca il percorso veloce, in "
      "modo limitato nel tempo e reversibile.", CAP),
    P("3.2 I due deliverable", H2),
    P("<b>Spiegabilita' (contributo qualitativo).</b> MAPPO e' una scatola nera: 'ha scelto ESCALATE'. "
      "Il supervisore produce: <i>«il flusso video e' salito 3x negli ultimi 20 s mentre il controllo "
      "e' stabile; escalo la compressione per proteggere la classe di controllo ed evitare la "
      "saturazione della coda.»</i> Un log leggibile per l'operatore — cio' che la policy neurale da "
      "sola non puo' dare."),
    P("<b>Robustezza fuori distribuzione (contributo quantitativo).</b> Il risultato misurabile. Si "
      "progettano scenari <b>non presenti in addestramento</b> e si confronta MAPPO-da-solo contro "
      "MAPPO+supervisore: la policy congelata degrada, il supervisore che ragiona recupera. E' il "
      "numero che dimostra il valore aggiunto della fase."),
]

# 4. MODELLO SLM
st += [
    P("4. Scelta del modello: Small Language Model + ablation", H1),
    P("Il compito del supervisore e' <b>stretto e strutturato</b>: leggere ~7 metriche, scegliere una "
      "di cinque azioni, produrre una breve giustificazione, su cadenza lenta. Questo profilo e' il "
      "caso ideale per uno Small Language Model, coerente con la tesi di Belcak et al. (2024), "
      "<i>Small Language Models are the Future of Agentic AI</i>: i sotto-compiti agentici ripetitivi "
      "e ben delimitati non richiedono modelli frontier."),
    tbl([
        ["Componente del compito", "Effetto della dimensione del modello"],
        ["Output strutturato (scegliere l'azione)", "Nessun problema a scendere CON constrained decoding: anche un 1.5B emette sempre un'azione valida"],
        ["Reasoning fuori distribuzione", "Pavimento: sotto ~1.5B il ragionamento sull'anomalia diventa inaffidabile"],
        ["Qualita' della spiegazione (prosa)", "Tetto: piu' grande aiuta — un 3B e' decente, un 7B piu' chiaro"],
    ], [6.2*cm, 9.8*cm]),
    P("Tabella 3 — Il punto dolce e' 1.5-3B: piccolo abbastanza da essere deployabile e riproducibile, "
      "grande abbastanza da ragionare e spiegare.", CAP),
    KEY("Il <b>constrained decoding</b> (schema JSON / grammatica) elimina il rischio principale dello "
        "scendere di dimensione: il modello piccolo non puo' sbagliare il FORMATO dell'azione, sceglie "
        "solo QUALE. Questo separa la robustezza del controllo (garantita) dalla qualita' della prosa "
        "(che degrada dolcemente con la dimensione)."),
    P("<b>La dimensione come esperimento.</b> Invece di fissare un modello, la dimensione diventa una "
      "variabile: stesso supervisore con Qwen2.5 0,5B / 1,5B / 3B / 7B (piu' Claude Haiku come tetto). "
      "Si misura: validita' dell'output strutturato, recupero fuori distribuzione, qualita' della "
      "spiegazione. Questo trasforma la domanda «quanto piccolo basta?» in un contributo di tesi "
      "pubblicabile. Modello primario: <b>Qwen2.5-3B</b> locale via Ollama (riproducibile, offline — "
      "l'argomento di credibilita'); interfaccia model-agnostic, cosi' l'ablation e' un semplice cambio "
      "di nome modello."),
    PageBreak(),
]

# 5. GUARDRAIL
st += [
    P("5. Sicurezza: guardrail e kill switch", H1),
    P("Il supervisore <b>suggerisce</b>; un guardrail decide se applicare. Una decisione errata o "
      "allucinata del modello non puo' danneggiare il sistema:"),
    P("• Azioni solo dallo spazio vincolato (garantito dallo schema, ri-validato dal guardrail).", LI),
    P("• Override limitato nel tempo (max 120 s) e reversibile.", LI),
    P("• Protezione dura: se il PDR e' gia' sotto una soglia critica, l'override viene rifiutato.", LI),
    P("• Kill switch: se attivo, ogni azione di controllo e' ignorata e resta il MAPPO puro.", LI),
    P("• Fallback fail-safe: se il backend LLM va in errore, si ripiega su 'endorse' — il percorso "
      "veloce non viene mai bloccato.", LI),
    KEY("Nel peggiore dei casi un override sbagliato tiene uno stato di compressione errato per una "
        "finestra breve, poi decade automaticamente: danno limitato e reversibile. Il percorso veloce "
        "(MAPPO) non viene mai spento — l'override sostituisce solo l'azione per la finestra."),
    P("6. Piano incrementale (milestone)", H1),
    tbl([
        ["Milestone", "Contenuto", "Rischio", "Deliverable"],
        ["M1", "LLM read-only explainer: solo commento in linguaggio naturale sulle decisioni MAPPO, nessuna autorita' di controllo", "nullo", "spiegabilita' (qualitativo), subito"],
        ["M2", "LLM supervisore con override vincolato + guardrail + kill switch", "basso", "controllo sicuro"],
        ["M3", "Valutazione fuori distribuzione: scenari nuovi, MAPPO-solo vs MAPPO+LLM; ablation sulla dimensione del modello", "medio", "il risultato quantitativo + il capitolo SLM"],
        ["M4 (stretch)", "Coordinamento LLM multi-agente su topologia mesh", "alto", "estensione"],
    ], [2.4*cm, 8.2*cm, 1.8*cm, 3.6*cm]),
    P("Tabella 4 — M1 e' il cancello di de-risking: si valida che il ragionamento del modello sia sano "
      "PRIMA di dargli qualsiasi autorita' di controllo (M2).", CAP),
]

# 7. VALUTAZIONE + INTEGRAZIONE
st += [
    P("7. Valutazione e scenari fuori distribuzione", H1),
    P("Il criterio per uno scenario OOD valido: la policy MAPPO congelata deve <b>fallire davvero</b> "
      "(essere fuori dal suo training) e un ragionatore deve poter plausibilmente aiutare. Lo scenario "
      "primario e' il <b>mix di traffico inedito</b> — chiaramente fuori distribuzione (MAPPO ha visto "
      "solo i sei scenari canonici), costruibile sulla topologia esistente, controllabile. Il "
      "<b>guasto multi-link correlato</b> e' il secondo (richiede la mesh, va con M4). Metriche di "
      "confronto: PDR, latenza, drop, piu' — per il supervisore — validita' dell'output strutturato e "
      "qualita' della spiegazione (giudicata)."),
    P("8. Integrazione e stato dello scheletro", H1),
    P("Lo scheletro e' gia' realizzato su un <b>branch dedicato e scartabile</b> "
      "(<font face='Courier'>phase4-llm-supervisor</font>), <b>disaccoppiato dal motore</b> (nessuna "
      "modifica a core.py — additivo). Moduli in <font face='Courier'>simulator/supervisor/</font>:"),
    P("actions.py   — 5 azioni + schema JSON (constrained decoding)\n"
      "backend.py   — interfaccia model-agnostic: Mock / Ollama (SLM) / Anthropic (Haiku)\n"
      "guardrail.py — override reversibile+limitato, kill switch, PDR floor, fail-safe\n"
      "controller.py— tick lento, costruzione prompt, decisione, log spiegabilita'\n"
      "examples/run_supervisor.py — demo (gira con MockBackend, nessun modello richiesto)\n"
      "tests/test_phase4_skeleton.py — 13 test; 94/94 totali passano, zero regressioni", MATH),
    P("Il MockBackend fa girare l'intero flusso senza modelli installati, ed e' anche la baseline "
      "'nessun ragionamento' dell'ablation. Il passaggio a un modello reale e' un cambio di backend.", BODY),
    P("9. Rischi e mitigazioni", H1),
    tbl([
        ["Rischio", "Mitigazione"],
        ["Latenza dell'LLM", "tick lento (~30 s): una risposta di 5 s e' irrilevante"],
        ["Allucinazione", "spazio di azione vincolato + guardrail bloccano le azioni dannose"],
        ["Non-determinismo", "temperature 0 + seed → riproducibile per la tesi"],
        ["Prompt injection", "l'LLM riceve solo metriche aggregate, mai il payload"],
        ["Divario sim→real", "validazione su emulatore, come gia' fatto in Fase 3"],
    ], [5*cm, 11*cm]),
    sp(8), hr(),
    P("Riferimenti bibliografici", H1),
    P("[1] Belcak, P., et al. (2024). <i>Small Language Models are the Future of Agentic AI.</i> "
      "NVIDIA Research. arXiv:2506.02153.", REF),
    P("[2] Yu, C., et al. (2022). <i>The Surprising Effectiveness of PPO in Cooperative Multi-Agent "
      "Games.</i> NeurIPS 35. arXiv:2103.01955. (Fase 3.)", REF),
    P("[3] Kahneman, D. (2011). <i>Thinking, Fast and Slow.</i> (Analogia System 1 / System 2.)", REF),
    P("[4] Abate, M., Sacco, A., Fiore, M., &amp; Esposito, F. <i>eFRAC: Elastic Flow-Rate Adaptive "
      "Compression for Network Congestion Management.</i> (Fase 2.)", REF),
]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2*cm, 1.2*cm, "Piano Fase 4 — Supervisore LLM  |  Towards Agentic Networks")
    canvas.drawRightString(W - 2*cm, 1.2*cm, f"Pag. {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#dddddd"))
    canvas.line(2*cm, 1.6*cm, W - 2*cm, 1.6*cm)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.3*cm, rightMargin=2.3*cm,
                        topMargin=2.2*cm, bottomMargin=2.2*cm,
                        title="Piano Fase 4 — Supervisore LLM", author="Flavio Bianco")
doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF generato: {OUT}")
