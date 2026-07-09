#!/usr/bin/env python3
"""Report tesi DEFINITIVO e coerente — modello cloud (marl/), numeri reali sim+emulatore.

Sostituisce i 3 PDF incoerenti (Report_EDS_Tesi, Documentazione_Completa,
MAPPO_Guida) che descrivevano il modello PyTorch scartato. Qui: implementazione
fedele al documento tecnico + penalita' di stabilita', validata su hardware.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

OUT = "/Users/flaviobianco/Desktop/Report_EDS_Definitivo.pdf"
W, H = A4
base = getSampleStyleSheet()


def st(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)


TITLE = st("T", "Title", fontSize=19, spaceAfter=8, leading=25,
           textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
SUB = st("S", "Normal", fontSize=12, spaceAfter=4, textColor=colors.HexColor("#444466"), alignment=TA_CENTER)
AUTH = st("A", "Normal", fontSize=10.5, spaceAfter=3, leading=15, textColor=colors.HexColor("#333355"), alignment=TA_CENTER)
H1 = st("H1", "Heading1", fontSize=15, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1a1a2e"))
H2 = st("H2", "Heading2", fontSize=12.5, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#2d2d5e"))
H3 = st("H3", "Heading3", fontSize=11, spaceBefore=9, spaceAfter=4, textColor=colors.HexColor("#3a3a6a"))
BODY = st("B", "Normal", fontSize=10.5, leading=16, spaceAfter=7, alignment=TA_JUSTIFY)
LI = st("LI", "Normal", fontSize=10.5, leading=15, spaceAfter=3, leftIndent=14, alignment=TA_JUSTIFY)
MATH = st("M", "Normal", fontSize=9.5, leading=13, spaceBefore=2, spaceAfter=7, fontName="Courier", leftIndent=24, textColor=colors.HexColor("#222244"))
CAP = st("C", "Normal", fontSize=8.5, spaceAfter=10, alignment=TA_CENTER, textColor=colors.HexColor("#666666"))
REF = st("R", "Normal", fontSize=9, leading=13, spaceAfter=5, leftIndent=18, firstLineIndent=-18)
ABS = st("AB", "Normal", fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY, leftIndent=20, rightIndent=20, textColor=colors.HexColor("#333333"))
KEYS = st("K", "Normal", fontSize=10, leading=15, spaceAfter=8, alignment=TA_JUSTIFY, leftIndent=16, rightIndent=10,
          borderWidth=0.8, borderColor=colors.HexColor("#4a8a6a"), borderPadding=7, backColor=colors.HexColor("#f0f8f3"))


def sp(h=6): return Spacer(1, h)
def hr(): return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cccccc"), spaceAfter=6, spaceBefore=2)
def P(t, s=BODY): return Paragraph(t, s)
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
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])


def tbl(data, widths):
    return Table(data, colWidths=widths, style=tstyle(), repeatRows=1)


s = []

# ── COPERTINA ──
s += [
    sp(48),
    P("Gestione Autonoma della Congestione di Rete:<br/>dal Simulatore a Eventi Discreti alla Validazione su Hardware", TITLE),
    sp(8),
    P("Report tecnico definitivo — Fasi 1–3 del progetto Event-Driven Simulator", SUB),
    sp(18), hr(), sp(8),
    P("<b>Autore:</b> Flavio Bianco", AUTH),
    P("<b>Tesi:</b> <i>Towards Agentic Networks: Autonomous Congestion Management</i>", AUTH),
    P("<b>Anno Accademico 2025/2026</b>", AUTH),
    sp(20), hr(), sp(8),
    P("<b>Abstract</b>", st("AH", "Normal", fontSize=10.5, alignment=TA_CENTER, spaceAfter=6)),
    P("Questo documento descrive la progettazione, l'implementazione e la validazione di un sistema "
      "di gestione autonoma della congestione di rete basato su compressione semantica adattiva. Il "
      "lavoro procede in tre fasi: (1) un simulatore a eventi discreti fedele alla dinamica di una rete "
      "con collo di bottiglia; (2) un controllo a regole basato sul modello eFRAC, usato come baseline "
      "competitiva; (3) una politica appresa con apprendimento per rinforzo multi-agente (MAPPO), "
      "implementata fedelmente alla specifica tecnica e raffinata con una penalita' di stabilita'. La "
      "politica appresa e' stata poi <b>distribuita su un emulatore di rete hardware</b> (ContainerLab, "
      "con compressione reale dei pacchetti via iptables NFQUEUE) e validata: MAPPO supera il controllo "
      "a regole su hardware reale, con un divario simulazione-realta' trascurabile. Tutti i valori "
      "numerici sono ottenuti da esecuzioni reali e riproducibili.", ABS),
    PageBreak(),
]

# ── 1. INTRODUZIONE ──
s += [
    P("1. Introduzione e Obiettivi", H1),
    P("La congestione — quando il traffico verso un nodo eccede la capacita' del collegamento in "
      "uscita — degrada latenza, consegna ed equita'. Le contromisure classiche (congestion control "
      "TCP, AQM come RED o CoDel) sono basate su regole fisse, robuste ma incapaci di adattarsi. Questa "
      "tesi esplora se una politica <b>appresa</b> possa eguagliare o superare una politica a regole, "
      "restando <b>distribuibile su hardware reale</b>. Il meccanismo di mitigazione e' la "
      "<b>compressione semantica adattiva al middlebox</b> (modello eFRAC): al crescere della "
      "congestione il router applica livelli crescenti di compressione, preservando l'informazione e "
      "aumentando il throughput effettivo."),
    tbl([
        ["Fase", "Contenuto", "Stato"],
        ["1", "Simulatore a eventi discreti (ambiente controllato)", "Completata"],
        ["2", "Controllo a regole eFRAC (baseline competitiva)", "Completata"],
        ["3", "MAPPO — politica appresa, validata su emulatore hardware", "Completata"],
        ["4", "Agenti LLM (spiegabilita', adattamento)", "Pianificata"],
    ], [1.2*cm, 11.3*cm, 3*cm]),
    P("Tabella 1 — Le fasi del progetto.", CAP),
    P("La progressione e' cumulativa: ogni fase produce una baseline misurabile per la successiva. La "
      "Fase 2 fornisce il termine di paragone per giudicare la Fase 3; i limiti della Fase 3 (politica "
      "congelata, assenza di spiegabilita') motiveranno la Fase 4.", BODY),
]

# ── 2. FASE 1 ──
s += [
    P("2. Fase 1 — Il simulatore a eventi discreti", H1),
    P("Il simulatore adotta il paradigma della simulazione a eventi discreti: lo stato evolve tramite "
      "una sequenza di eventi ordinati su una coda a priorita', anziche' a passi temporali fissi. "
      "Vantaggi: efficienza (si elabora solo cio' che accade), precisione (istanti di tempo continuo), "
      "naturalezza (generazione, arrivo, accodamento, servizio, guasto sono eventi puntuali). Sono "
      "definiti 14 tipi di evento; il cuore e' uno scheduler a heap che li esegue in ordine temporale."),
    P("Il modello di servizio delle code e' <b>M/D/1</b> (arrivi poissoniani, servizio deterministico, "
      "servente singolo), con politica tail-drop e tempo di servizio <b>byte-aware</b>: la compressione "
      "riduce la dimensione effettiva dei pacchetti e quindi il tempo di trasmissione. La topologia di "
      "riferimento e' il collo di bottiglia singolo. La congestione e' governata da una macchina a "
      "cinque stati (NORMAL, HEADER, DELTA, INCREMENTAL, DROP_LOW_PRIORITY) con soglie di occupazione "
      "crescenti (50/70/85/95%). La Fase 1 e' validata da una suite di test unitari."),
]

# ── 3. FASE 2 ──
s += [
    P("3. Fase 2 — Il controllo a regole (eFRAC)", H1),
    P("La Fase 2 implementa la baseline: compressione semantica adattiva fedele al paper eFRAC. A ogni "
      "coppia (stato, priorita') e' associato un rapporto di compressione dalla Tabella 1 del paper. "
      "Due meccanismi stabilizzano il controllo: la <b>media mobile esponenziale (EWMA, α=0,125)</b> "
      "sull'occupazione, che filtra i picchi transitori, e l'<b>isteresi asimmetrica</b> (salita dopo "
      "1,5 s sostenuti, discesa dopo 4,5 s): l'escalation e' un'emergenza, la de-escalation richiede "
      "prova robusta di recupero. Le transizioni avvengono un livello alla volta."),
    P("Questo controllo e' <b>reattivo</b>: reagisce dopo il superamento delle soglie, quindi lascia "
      "riempire i buffer prima di intervenire. E' una baseline forte ma con questo limite strutturale, "
      "che la Fase 3 supera comprimendo in modo proattivo.", BODY),
    PageBreak(),
]

# ── 4. FASE 3 MODEL ──
s += [
    P("4. Fase 3 — Apprendimento per rinforzo multi-agente (MAPPO)", H1),
    P("4.1 Formulazione e scelta dell'algoritmo", H2),
    P("Il controllo distribuito della congestione e' un <b>Dec-POMDP</b> (processo decisionale di "
      "Markov decentralizzato e parzialmente osservabile): ogni router decide sulla sola coda locale, "
      "coopera a un obiettivo comune, e non comunica a runtime. Tra gli algoritmi MARL cooperativi si e' "
      "scelto <b>MAPPO</b> per: gestione della non-stazionarieta' tramite critico centralizzato "
      "(CTDE), esecuzione decentralizzata senza comunicazione (deployabilita' su hardware), stabilita' "
      "del clipping PPO, e coerenza con la Fase 4."),

    P("4.2 Architettura — implementazione in NumPy puro", H2),
    P("Attore  : Input(7) → LayerNorm → Linear(7→64) → Tanh → Linear(64→64) → Tanh "
      "→ Linear(64→3) → Softmax", MATH),
    P("Critico : Input(11) → LayerNorm → Linear(11→128) → Tanh → Linear(128→128) → "
      "Tanh → Linear(128→1)", MATH),
    KEY("L'intera implementazione (forward, backpropagation, ottimizzatore Adam, LayerNorm non-affine) "
        "e' in <b>NumPy puro</b>, senza framework di deep learning. Scelta decisiva per il deployment: "
        "il checkpoint e' esportato in JSON e l'inferenza gira in <b>Python puro (sola libreria "
        "standard)</b> dentro il container del router, con parita' numerica verificata. Inizializzazione "
        "ortogonale (guadagno √2 sugli strati nascosti), attivazione Tanh, LayerNorm senza parametri "
        "apprendibili."),

    P("4.3 Osservazione, azione, ricompensa", H2),
    P("L'osservazione locale ha 7 feature (occupazione EWMA, stato normalizzato, quote di traffico ad "
      "alta/bassa priorita' in coda, drop rate di finestra, utilizzo del collegamento, tempo nello "
      "stato). Le azioni sono 3: ESCALATE, MAINTAIN, DE-ESCALATE. La ricompensa condivisa segue la "
      "specifica tecnica (§6):"),
    P("r_t = PDR_t − 0,3·drop_rate_t − 0,2·(latenza_t / 2 s) + 0,2·Jain_t", MATH),
    P("A questa si aggiunge un unico raffinamento: una <b>penalita' di stabilita'</b> che sottrae un "
      "termine proporzionale al numero di transizioni di stato nel passo (peso 0,1). Motivazione "
      "diretta: oscillare tra livelli di compressione ha un costo reale; penalizzarlo spinge la policy "
      "a stabilizzarsi sul livello ottimo e a mantenerlo, riducendo il flapping."),
    tbl([
        ["Iperparametro", "Valore", "Fonte"],
        ["Sconto γ / GAE λ", "0,99 / 0,95", "doc §3.3"],
        ["Clipping PPO ε", "0,2", "doc §3.2"],
        ["Learning rate attore / critico", "3·10⁻⁴ / 10⁻³", "doc Tab. 4"],
        ["Epoche K / minibatch", "10 / 256", "doc Tab. 4"],
        ["Gradient clipping (norma)", "10", "doc Tab. 4"],
        ["Coefficiente entropia", "0,01", "PPO standard"],
        ["Penalita' di stabilita'", "0,1 per transizione", "raffinamento"],
    ], [6.6*cm, 4.6*cm, 4.3*cm]),
    P("Tabella 2 — Iperparametri (fedeli alla specifica tecnica + penalita' di stabilita').", CAP),

    P("4.4 Addestramento e deployment", H2),
    P("L'addestramento gira su <b>tutti e sei gli scenari canonici</b> (scenario campionato a caso a "
      "ogni episodio), per 500 episodi, con valutazione deterministica (argmax) periodica e salvataggio "
      "del checkpoint al miglioramento del reward. Il checkpoint JSON contiene i pesi in forma "
      "leggibile; viene copiato nel container del router dell'emulatore, dove l'attore in Python puro "
      "pilota la macchina a stati sulla rete reale. Unica dipendenza dell'intera Fase 3: NumPy (solo "
      "per l'addestramento; l'inferenza non ne ha bisogno)."),
    PageBreak(),
]

# ── 5. VALIDAZIONE HARDWARE ──
s += [
    P("5. Validazione su hardware emulato (ContainerLab)", H1),
    P("La sola simulazione non basta per la credibilita' scientifica. La politica appresa e' stata "
      "distribuita su un <b>emulatore ContainerLab</b>: container Linux reali, stack di rete del "
      "kernel, shaping con tc, agente di traffico UDP che riproduce gli stessi modelli di flusso del "
      "simulatore. La topologia e gli scenari sono identici a quelli simulati."),
    P("5.1 Compressione reale al middlebox", H2),
    P("Il punto critico: nell'emulatore i byte sono reali. La compressione avviene al router tramite "
      "<b>iptables NFQUEUE</b> — il traffico inoltrato viene intercettato in spazio utente, il payload "
      "viene troncato al rapporto corrispondente allo stato di congestione corrente (letto da file "
      "condiviso, scritto dal control-plane), e il pacchetto viene ricostruito (aggiornamento di "
      "lunghezza IP, ricalcolo del checksum). I byte sul collegamento collo di bottiglia sono cosi' "
      "<b>realmente ridotti</b>, e la compression ratio e' misurata sui byte effettivi, non stimata. "
      "Un flag di fail-open garantisce che, se il compressore cade, il traffico passi non compresso "
      "invece di essere bloccato."),
    P("5.2 Protocollo di benchmark", H2),
    P("Un benchmark automatico esegue, <b>sullo stesso ferro</b> e sugli stessi scenari, tutti e tre i "
      "controllori (Fase 1 istantanea, Fase 2 a regole, Fase 3 MAPPO) e ne misura i KPI, mediando su "
      "piu' ripetizioni per gestire il rumore reale. La Fase 3 carica il checkpoint JSON del "
      "simulatore: <b>lo stesso modello</b> validato in simulazione gira sulla rete reale.", BODY),
]

# ── 6. RISULTATI ──
s += [
    P("6. Risultati", H1),
    P("6.1 Confronto per scenario (simulazione, media su 5 semi)", H2),
    tbl([
        ["Scenario", "PDR F2", "PDR F3", "Lat F2", "Lat F3", "Compr F2", "Compr F3"],
        ["1 single bottleneck", "95,6%", "99,6%", "774ms", "158ms", "1,20×", "1,49×"],
        ["2 flash crowd", "97,9%", "99,6%", "573ms", "185ms", "1,09×", "1,37×"],
        ["3 bandwidth degr.", "83,6%", "87,3%", "1220ms", "604ms", "1,24×", "1,50×"],
        ["4 link fail/recov.", "71,9%", "72,0%", "295ms", "120ms", "1,00×", "1,39×"],
        ["5 persistent overload", "93,3%", "99,4%", "706ms", "168ms", "1,34×", "1,59×"],
        ["6 mixed traffic", "98,5%", "99,8%", "1058ms", "111ms", "1,05×", "1,54×"],
    ], [4.2*cm, 1.9*cm, 1.9*cm, 1.9*cm, 1.9*cm, 2.1*cm, 2.1*cm]),
    P("Tabella 3 — Fase 2 vs Fase 3 per scenario (simulazione). Fase 3 vince sul PDR in tutti e sei.", CAP),
    P("La Fase 3 migliora il PDR su ogni scenario e riduce la latenza ovunque. Lo scenario 4 (guasto "
      "del collegamento) e' l'unico dove il PDR resta basso per entrambi (≈72%): e' il <b>tetto "
      "fisico</b> — con il link giu' nessuna politica puo' consegnare quei pacchetti.", BODY),

    P("6.2 Simulazione vs hardware (media globale, 6 scenari)", H2),
    tbl([
        ["KPI", "Fase 2", "Fase 3 sim.", "Fase 3 emulatore"],
        ["Packet Delivery Ratio", "90,12%", "92,95%", "93,00%"],
        ["Latenza end-to-end", "771 ms", "224 ms", "264 ms"],
        ["Compression ratio (byte reali)", "1,15×", "1,48×", "1,49×"],
        ["Transizioni di stato", "11", "9", "10"],
        ["Jain fairness", "0,935", "0,936", "0,933"],
    ], [5.6*cm, 3*cm, 3.6*cm, 3.8*cm]),
    P("Tabella 4 — Il risultato centrale: i numeri di simulazione e quelli misurati sull'emulatore "
      "hardware coincidono quasi esattamente.", CAP),
    KEY("Il divario simulazione-realta' e' <b>trascurabile</b>: PDR 92,95% (sim) contro 93,00% "
        "(hardware), compressione 1,48× contro 1,49×, latenza 224 contro 264 ms (piccolo overhead "
        "reale). La baseline Fase 2 e' 90,12% in entrambi gli ambienti — controllo di sanita' "
        "superato. La politica appresa in simulazione riproduce fedelmente il proprio comportamento su "
        "stack di rete Linux reale con byte realmente compressi. Questo chiude il ciclo "
        "simulazione → addestramento → validazione hardware."),

    P("6.3 Interpretazione", H2),
    P("La spiegazione e' meccanicistica: la policy appresa comprime in modo <b>proattivo</b> e mantiene "
      "le code corte; code corte significano meno latenza e meno perdite, quindi piu' consegna. Il "
      "controllo a regole, reattivo per costruzione, reagisce solo dopo il superamento delle soglie e "
      "arriva in ritardo. La penalita' di stabilita' contribuisce riducendo il flapping: la policy si "
      "posiziona sul livello di compressione adeguato e lo mantiene (poche transizioni), invece di "
      "oscillare. L'equita' resta sostanzialmente invariata rispetto alla baseline.", BODY),
    PageBreak(),
]

# ── 7. CONCLUSIONI ──
s += [
    P("7. Conclusioni e prossimi passi", H1),
    P("7.1 Risultati raggiunti", H2),
    P("Il progetto ha conseguito gli obiettivi delle prime tre fasi: un simulatore a eventi discreti "
      "validato; un controllo a regole eFRAC come baseline; una politica MAPPO appresa, fedele alla "
      "specifica tecnica e raffinata con una penalita' di stabilita', che <b>supera la baseline su "
      "hardware reale</b>. Il risultato centrale e' la coerenza simulazione-realta': la stessa policy "
      "consegna il 93% dei pacchetti con compressione reale 1,49× e latenza ridotta del 66% rispetto "
      "al controllo a regole, sia in simulazione sia sull'emulatore ContainerLab."),
    P("7.2 Note di riproducibilita'", H2),
    P("L'implementazione della Fase 3 e' in NumPy puro (unica dipendenza), con suite di test unitari "
      "sull'intera pipeline. L'addestramento e' riproducibile da riga di comando; il checkpoint JSON "
      "e' auto-descrittivo e caricabile in Python puro per il deployment. I comandi:", BODY),
    P("python3 examples/train_mappo.py --stability-penalty 0.1\n"
      "python3 examples/compare_phase2_phase3.py --ckpt checkpoints/mappo_best_stab.json --seeds 5\n"
      "python3 emulator/benchmark.py --mappo-ckpt ../checkpoints/mappo_best_stab.json", MATH),
    P("7.3 Prossimi passi — Fase 4", H2),
    P("I limiti di MAPPO — politica congelata dopo l'addestramento, assenza di spiegabilita', "
      "generalizzazione limitata alla distribuzione di addestramento — costituiscono la motivazione "
      "della Fase 4: un <b>livello supervisore basato su LLM</b> sopra la policy MARL. L'architettura "
      "prevista e' ibrida a due livelli: la policy MAPPO resta il percorso veloce (decisione per "
      "secondo, deterministica, deployata come file leggero); un agente LLM opera sul percorso lento "
      "(supervisione, spiegazione delle decisioni, riconoscimento di scenari fuori distribuzione, "
      "ritaratura). Questa struttura mantiene i vantaggi verificati della Fase 3 e vi aggiunge "
      "adattamento e trasparenza — senza sacrificare la velocita' e la deployabilita' gia' dimostrate "
      "su hardware."),
    sp(8), hr(),
    P("Riferimenti bibliografici", H1),
    P("[1] Yu, C., et al. (2022). <i>The Surprising Effectiveness of PPO in Cooperative Multi-Agent "
      "Games.</i> NeurIPS 35. arXiv:2103.01955.", REF),
    P("[2] Schulman, J., et al. (2017). <i>Proximal Policy Optimization Algorithms.</i> arXiv:1707.06347.", REF),
    P("[3] Schulman, J., et al. (2016). <i>High-Dimensional Continuous Control Using Generalized "
      "Advantage Estimation.</i> ICLR. arXiv:1506.02438.", REF),
    P("[4] Abate, M., Sacco, A., Fiore, M., &amp; Esposito, F. <i>eFRAC: Elastic Flow-Rate Adaptive "
      "Compression for Network Congestion Management.</i> (Riferimento della Fase 2.)", REF),
    P("[5] Jacobson, V. (1988). <i>Congestion Avoidance and Control.</i> SIGCOMM '88. (Costante EWMA α=1/8.)", REF),
    P("[6] Bernstein, D. S., et al. (2002). <i>The Complexity of Decentralized Control of Markov "
      "Decision Processes.</i> Math. Oper. Res., 27(4).", REF),
]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2*cm, 1.2*cm, "Report EDS — Towards Agentic Networks: Autonomous Congestion Management")
    canvas.drawRightString(W - 2*cm, 1.2*cm, f"Pag. {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#dddddd"))
    canvas.line(2*cm, 1.6*cm, W - 2*cm, 1.6*cm)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.3*cm, rightMargin=2.3*cm,
                        topMargin=2.2*cm, bottomMargin=2.2*cm,
                        title="Report EDS Definitivo", author="Flavio Bianco")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF generato: {OUT}")
