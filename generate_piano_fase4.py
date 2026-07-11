#!/usr/bin/env python3
"""Fase 4 — Supervisore LLM: progetto E risultati (M1-M3 completati).

v2: aggiorna il piano originale con i risultati sperimentali reali delle
milestone M1 (explainer SLM), M2 (controllo attivo), M3 (valutazione OOD).
"""
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
WARNS = s_("W", "Normal", fontSize=10, leading=15, spaceAfter=8, alignment=TA_JUSTIFY, leftIndent=16, rightIndent=10,
           borderWidth=0.8, borderColor=colors.HexColor("#aa6644"), borderPadding=7, backColor=colors.HexColor("#fbf3ee"))


def sp(h=6): return Spacer(1, h)
def hr(): return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cccccc"), spaceAfter=6, spaceBefore=2)
def P(t, st=BODY): return Paragraph(t, st)
def KEY(t): return Paragraph(t, KEYS)
def WARN(t): return Paragraph(t, WARNS)


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
    cell = ParagraphStyle("cell", parent=base["Normal"], fontSize=8.5, leading=11)
    hc = ParagraphStyle("hc", parent=base["Normal"], fontSize=8.5, leading=11,
                        fontName="Helvetica-Bold", textColor=colors.white)
    wr = [[Paragraph(str(c).replace("\n", "<br/>"), hc if r == 0 else cell) for c in row]
          for r, row in enumerate(data)]
    return Table(wr, colWidths=widths, style=tstyle(), repeatRows=1)


st = []

# ── COPERTINA ──
st += [
    sp(44),
    P("Fase 4 — Supervisore LLM per la Gestione Autonoma della Congestione", TITLE),
    sp(8), P("Progetto e risultati sperimentali (milestone M1–M3 completate)", SUB),
    sp(16), hr(), sp(8),
    P("<b>Autore:</b> Flavio Bianco", AUTH),
    P("<b>Tesi:</b> <i>Towards Agentic Networks: Autonomous Congestion Management</i>", AUTH),
    P("<b>Anno Accademico 2025/2026 — aggiornato con i risultati M1–M3</b>", AUTH),
    sp(18), hr(), sp(8),
    P("<b>Abstract</b>", s_("AH", "Normal", fontSize=10.5, alignment=TA_CENTER, spaceAfter=6)),
    P("La Fase 4 estende la politica MAPPO validata su hardware (Fase 3) con un <b>supervisore "
      "basato su Small Language Model</b> sul percorso lento: monitoraggio, spiegazione in "
      "linguaggio naturale, intervento vincolato. Questo documento presenta il progetto E i "
      "risultati delle tre milestone completate. Gli esiti principali, in parte inattesi: "
      "(1) la separazione fra <b>decisione deterministica</b> e <b>spiegazione LLM</b> e' "
      "necessaria — un modello da 3B non sa confrontare numeri, ma spiega bene un verdetto "
      "gia' calcolato; (2) la policy MAPPO e' risultata <b>robusta su tre assi fuori "
      "distribuzione</b>, ridimensionando il ruolo correttivo del supervisore; (3) gli override "
      "aggressivi <b>danneggiano</b> (una regola di escalation e' stata provata e rimossa) e i "
      "<b>guardrail</b> si sono rivelati il componente critico del progetto. Il valore robusto "
      "del supervisore e': spiegabilita', stabilizzazione marginale, monitoraggio — con il "
      "principio operativo «first, do no harm».", ABS),
    PageBreak(),
]

# ── 1. MOTIVAZIONE ──
st += [
    P("1. Motivazione: i limiti della Fase 3", H1),
    P("La politica MAPPO della Fase 3 supera il controllo a regole su hardware reale, ma ha tre "
      "limiti strutturali del paradigma: <b>politica congelata</b> (nessun adattamento senza "
      "retraining), <b>assenza di spiegabilita'</b> (rete neurale opaca), <b>generalizzazione "
      "incerta</b> fuori dalla distribuzione di addestramento. Questi limiti non si risolvono con "
      "un altro algoritmo RL — richiedono capacita' di natura diversa (linguaggio, ragionamento, "
      "monitoraggio). La Fase 4 non sostituisce la Fase 3: la <b>estende</b> con un livello "
      "supervisorio."),

    P("2. Architettura ibrida a due livelli", H1),
    tbl([
        ["", "Percorso VELOCE (System 1)", "Percorso LENTO (System 2)"],
        ["Chi", "Actor MAPPO (Fase 3)", "Supervisore (Fase 4)"],
        ["Cadenza", "ogni 1 secondo", "ogni ~30 s o su anomalia"],
        ["Natura", "deterministica, congelata", "regola deterministica + spiegazione LLM"],
        ["Ruolo", "decide la compressione", "monitora, spiega, interviene con parsimonia"],
        ["Impatto", "INTATTO", "livello supervisorio additivo"],
    ], [2.4*cm, 6.6*cm, 7*cm]),
    P("Tabella 1 — I due livelli. L'LLM non entra mai nel ciclo per-secondo: nessuna scelta di "
      "modello puo' degradare le prestazioni validate della Fase 3.", CAP),
    KEY("<b>Evoluzione architetturale emersa in M1</b> — il supervisore stesso e' diventato "
        "ibrido: la DECISIONE (endorse / override + stato target) e' calcolata da una regola "
        "deterministica (assess: soglie su PDR e drop rate); l'LLM fornisce SOLO la spiegazione "
        "in linguaggio naturale e puo' segnalare anomalie (flag_retrain). Motivo empirico: un "
        "modello da 3B non sa confrontare numeri in modo affidabile (dichiarava «basso» "
        "un PDR di 0,997 e «alto» un drop di 0,000, ignorando la regola esplicita nel "
        "prompt). La separazione usa la forza dell'LLM (linguaggio) e ne evita la debolezza "
        "(aritmetica) — e rafforza l'argomento SLM: 3B basta per spiegare, non gli si chiede di "
        "calcolare."),

    P("3. Il supervisore: azioni, sicurezza, deliverable", H1),
    tbl([
        ["Azione", "Effetto", "Percorso veloce?"],
        ["endorse", "nessun intervento (caso comune)", "no"],
        ["override_state(3, 30s)", "forza la compressione massima senza scarti per una finestra", "SI (vincolato)"],
        ["flag_retrain", "segnala regime anomalo per retrain offline", "no"],
        ["explain", "sola giustificazione in linguaggio naturale", "no"],
        ["coordinate(msg)", "messaggio a un peer (mesh, futuro)", "no"],
    ], [4.2*cm, 9*cm, 2.8*cm]),
    P("Tabella 2 — Spazio di azione vincolato. Il target dell'override e' SEMPRE lo stato 3: "
      "l'escalation automatica allo stato 4 e' stata provata e rimossa (sez. 6.3).", CAP),
    P("Guardrail (tutti verificati sperimentalmente): azioni solo dallo spazio vincolato "
      "(constrained decoding + rivalidazione); override limitato nel tempo (max 120 s) e "
      "reversibile; <b>revoca</b> se il PDR di finestra scende sotto il floor; rifiuto di nuovi "
      "override sotto il floor; kill switch (riporta a M1); fail-safe sul backend (errore LLM → "
      "endorse, il percorso veloce non si blocca mai)."),
    PageBreak(),
]

# ── 4. MODELLO SLM ──
st += [
    P("4. Modello: SLM locale e ablation sulla dimensione", H1),
    P("Il compito del supervisore e' stretto e strutturato — il profilo ideale per uno Small "
      "Language Model (Belcak et al., 2024). Primario: <b>Qwen2.5-3B via Ollama</b> (locale, "
      "offline, riproducibile con temperature 0 e seed; tag <font face='Courier'>qwen2.5:3b</font>). "
      "Interfaccia model-agnostic (Mock / Ollama / Anthropic) con constrained decoding: lo schema "
      "JSON garantisce output valido a qualunque dimensione."),
    P("4.1 Risultato ablation (0.5B / 1.5B / 3B / 7B, scenario 3)", H2),
    tbl([
        ["Modello", "Comportamento osservato sulla spiegazione"],
        ["Qwen 0.5B", "PAROTA la valutazione iniettata nel prompt: zero errori ma zero interpretazione"],
        ["Qwen 1.5B", "coerente ma confuso nei dettagli («la forza di una rete di addestramento»)"],
        ["Qwen 3B", "interpreta la traiettoria (coglie l'anomalia) ma scivola sulla direzione («PDR elevati» con PDR basso)"],
        ["Qwen 7B", "il migliore: direzione corretta + lettura della traiettoria + rimedio proposto"],
    ], [2.6*cm, 13.4*cm]),
    P("Tabella 3 — Ablation sulla dimensione. La qualita' interpretativa scala con la dimensione; "
      "la metrica automatica «errori di direzione» premia il parroting (0.5B = 0 errori "
      "perche' non interpreta nulla) e va usata con cautela.", CAP),
    KEY("Lezioni di metodo dall'ablation: (1) valutare i supervisori SLM sulla sola correttezza "
        "di direzione e' fuorviante — serve una metrica del valore interpretativo aggiunto; "
        "(2) «il piu' piccolo sufficiente» dipende dalla barra: per eco della valutazione "
        "basta 0.5B, per interpretazione genuina serve 7B; il 3B e' il compromesso pratico "
        "(interpreta, con residui di prosa). (3) L'iper-ingegneria del prompt DANNEGGIA i modelli "
        "piccoli: flag qualitativi per-metrica hanno causato override spuri (il 3B seguiva ogni "
        "«CRITICO» ciecamente); la soluzione e' la regola giusta, non piu' flag."),

    P("5. Milestone: piano ed esito", H1),
    tbl([
        ["Milestone", "Contenuto", "Esito"],
        ["M1", "Explainer read-only (kill switch): spiegazioni NL sul loop MAPPO reale",
         "COMPLETATA — azioni deterministiche corrette, spiegazioni grounded con Qwen-3B; 3 iterazioni di prompt documentate"],
        ["M2", "Controllo attivo: override applicato al percorso veloce + revoca",
         "COMPLETATA — scenario 3: PDR +0.008, drop −8, transizioni −20% (5 seed)"],
        ["M3", "Valutazione OOD su 3 assi + gruppo di controllo",
         "COMPLETATA — esito inatteso e istruttivo (sez. 6)"],
        ["M4 (stretch)", "Coordinamento multi-agente su mesh", "non avviata"],
    ], [1.9*cm, 6.3*cm, 7.8*cm]),
    P("Tabella 4 — Stato delle milestone.", CAP),
    PageBreak(),
]

# ── 6. RISULTATI M3 ──
st += [
    P("6. Risultati M3: valutazione fuori distribuzione", H1),
    P("6.1 Protocollo", H2),
    P("Tre scenari OOD costruiti ad hoc (mai visti in addestramento), ciascuno su un asse di "
      "novita' diverso; per ogni seed lo stesso episodio gira due volte (MAPPO solo / MAPPO + "
      "supervisore); gruppo di controllo in-distribution (scenario 3 canonico); 5 seed; KPI "
      "misurati dal simulatore, indipendenti dal backend LLM (la decisione e' deterministica)."),
    tbl([
        ["Scenario OOD", "Asse di novita'"],
        ["video_flood", "mix inedito: solo VIDEO, 17 pkt/s, nessuna classe protetta in coda (feature di priorita' mai viste)"],
        ["pulsed", "asse temporale: surge on/off ogni 10 s (in training il carico e' stazionario o cambia una volta)"],
        ["capacity_collapse", "collasso permanente del collo di bottiglia 10→2 pkt/s (minimo visto: 4, transitorio); metrica aggiuntiva: consegna del flusso CONTROL"],
    ], [3.4*cm, 12.6*cm]),
    P("Tabella 5 — I tre assi fuori distribuzione.", CAP),

    P("6.2 Risultati (5 seed, media)", H2),
    tbl([
        ["Scenario", "KPI chiave", "MAPPO solo", "+ supervisore", "Lettura"],
        ["video_flood", "PDR", "0,932", "0,934", "MAPPO robusto; supervisore ≈ neutro (drop −3,6)"],
        ["pulsed", "PDR / trans.", "0,874 / 29,6", "0,874 / 24,8", "pari sul PDR; supervisore stabilizza (−16% trans., −12 ms)"],
        ["capacity_collapse", "consegna CONTROL", "0,887", "0,706", "MAPPO protegge da solo (usa lo stato 4); l'override a 3 DANNEGGIA (−0,18), danno contenuto dai guardrail"],
        ["scenario 3 (controllo)", "PDR / drop", "0,865 / 142", "0,873 / 134", "in-distribution il supervisore aiuta (drop −8, trans. −20%)"],
    ], [3.1*cm, 2.9*cm, 2.5*cm, 2.6*cm, 4.9*cm]),
    P("Tabella 6 — M3: MAPPO-solo contro MAPPO+supervisore sui quattro blocchi.", CAP),

    P("6.3 L'esperimento fallito (documentato): escalation automatica allo stato 4", H2),
    WARN("Per il collasso di capacita' e' stata provata una regola di escalation: se il sistema "
         "e' critico e la compressione e' gia' attiva, forzare lo stato 4 (scarto attivo delle "
         "priorita' basse). Esito: <b>fallimento su entrambi i fronti</b>. Nel collasso non "
         "scattava mai (bloccata dal PDR floor del guardrail); sul degrado transitorio dello "
         "scenario 3 forzava scarti attivi devastanti (PDR 0,865→0,690, drop +189). La regola e' "
         "stata rimossa e il principio opposto e' stato adottato e fissato nei test: "
         "<b>«first, do no harm»</b> — mai lo stato 4 da soglie statiche. Nota "
         "metodologica: il confine fra oscillazione-da-stabilizzare e uso-necessario dello stato 4 "
         "NON e' separabile con le metriche di finestra disponibili (PDR e drop quasi identici nei "
         "due casi) — servono metriche per-classe o un supervisore piu' capace (lavoro futuro)."),

    P("6.4 Conclusioni della Fase 4", H2),
    KEY("(1) <b>MAPPO e' robusto</b> sui tre assi OOD testati — un risultato pro-Fase 3: la "
        "strategia appresa (compressione massima civile, stato 4 solo quando serve) e' "
        "quasi-ottima per qualunque sovraccarico su questa topologia. (2) Il <b>valore robusto del "
        "supervisore</b> e': spiegabilita' (M1, con SLM 3B locale), stabilizzazione marginale "
        "(transizioni −8/−20% ovunque, drop −8 in-distribution), monitoraggio. (3) Il "
        "<b>controllo</b> va esercitato con parsimonia: gli override aggressivi danneggiano, la "
        "policy appresa spesso ne sa di piu'. (4) I <b>guardrail sono il componente critico</b>: "
        "revoca + PDR floor hanno trasformato un potenziale disastro in un −0,18 contenuto. Un "
        "esito parzialmente negativo, riportato con protocollo rigoroso: e' la conclusione "
        "difendibile della fase."),

    P("7. Stato del codice", H1),
    P("Branch <font face='Courier'>phase4-llm-supervisor</font> (pushato su GitHub). Moduli: "
      "supervisor/{actions, backend, guardrail, controller, ood}.py; runner examples/"
      "{run_supervisor, run_m1_explainer, run_m2_supervisor, run_m3_ood, run_ablation}.py; "
      "111 test totali (nessuna modifica alle Fasi 1-3, tutto additivo)."),
    P("python3.12 examples/run_m1_explainer.py --scenario 3 --backend ollama\n"
      "python3.12 examples/run_m2_supervisor.py --scenario 3 --compare\n"
      "python3.12 examples/run_m3_ood.py --ood capacity_collapse --seeds 5\n"
      "python3.12 examples/run_ablation.py --scenario 3", MATH),

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
    canvas.drawString(2*cm, 1.2*cm, "Fase 4 — Supervisore LLM: progetto e risultati  |  Towards Agentic Networks")
    canvas.drawRightString(W - 2*cm, 1.2*cm, f"Pag. {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#dddddd"))
    canvas.line(2*cm, 1.6*cm, W - 2*cm, 1.6*cm)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.3*cm, rightMargin=2.3*cm,
                        topMargin=2.2*cm, bottomMargin=2.2*cm,
                        title="Fase 4 — Supervisore LLM: progetto e risultati",
                        author="Flavio Bianco")
doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF generato: {OUT}")
