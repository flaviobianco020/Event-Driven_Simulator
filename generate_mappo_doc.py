#!/usr/bin/env python3
"""Genera il documento accademico su MAPPO per la tesi EDS."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

OUTPUT = "/Users/flaviobianco/Desktop/MAPPO_Phase3_EDS.pdf"

W, H = A4

# ── stili ──────────────────────────────────────────────────────────────────────

base = getSampleStyleSheet()

def style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

TITLE     = style("TITLE",  "Title",   fontSize=18, spaceAfter=6,  textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
SUBTITLE  = style("SUB",    "Normal",  fontSize=12, spaceAfter=4,  textColor=colors.HexColor("#444466"), alignment=TA_CENTER, italic=True)
AUTH      = style("AUTH",   "Normal",  fontSize=11, spaceAfter=2,  textColor=colors.HexColor("#333355"), alignment=TA_CENTER)
DATE      = style("DATE",   "Normal",  fontSize=10, spaceAfter=20, textColor=colors.HexColor("#888888"), alignment=TA_CENTER)

H1        = style("H1",     "Heading1", fontSize=13, spaceBefore=18, spaceAfter=6,
                  textColor=colors.HexColor("#1a1a2e"), borderPad=0)
H2        = style("H2",     "Heading2", fontSize=11, spaceBefore=12, spaceAfter=4,
                  textColor=colors.HexColor("#2d2d5e"))

BODY      = style("BODY",   "Normal",  fontSize=10.5, leading=16, spaceAfter=6,
                  alignment=TA_JUSTIFY)
BODY_SM   = style("BSML",   "Normal",  fontSize=9.5,  leading=14, spaceAfter=4,
                  alignment=TA_JUSTIFY)
ABSTRACT  = style("ABS",    "Normal",  fontSize=10,   leading=15, spaceAfter=6,
                  alignment=TA_JUSTIFY, leftIndent=30, rightIndent=30,
                  textColor=colors.HexColor("#333333"), italic=True)
MATH      = style("MATH",   "Normal",  fontSize=10,   leading=14, spaceAfter=4,
                  fontName="Courier", leftIndent=40)
CAPTION   = style("CAP",    "Normal",  fontSize=9,    spaceAfter=8,
                  alignment=TA_CENTER, textColor=colors.HexColor("#555555"), italic=True)
REF       = style("REF",    "Normal",  fontSize=9.5,  leading=14, spaceAfter=4,
                  leftIndent=20, firstLineIndent=-20)

# ── helper ─────────────────────────────────────────────────────────────────────

def sp(h=6):
    return Spacer(1, h)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=4)

def table_style(header_bg="#2d2d5e", row_alt="#f0f0f8"):
    return TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor(header_bg)),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(row_alt)]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
    ])

# ── contenuto ──────────────────────────────────────────────────────────────────

story = []

# ── copertina ──────────────────────────────────────────────────────────────────
story += [
    sp(60),
    Paragraph("Multi-Agent Proximal Policy Optimization (MAPPO)", TITLE),
    sp(8),
    Paragraph("Architettura di Apprendimento per Rinforzo Multi-Agente<br/>"
              "per il Controllo Autonomo della Congestione di Rete", SUBTITLE),
    sp(16),
    hr(),
    sp(10),
    Paragraph("Documento tecnico — Fase 3 del progetto<br/>"
              "<i>Towards Agentic Networks: Autonomous Congestion Management</i>", AUTH),
    sp(6),
    Paragraph("Flavio Bianco &nbsp;&nbsp;|&nbsp;&nbsp; Anno Accademico 2025/2026", AUTH),
    sp(6),
    Paragraph("Giugno 2026", DATE),
    sp(40),
    hr(),
    sp(12),
    Paragraph("<b>Abstract</b>", style("AH", "Normal", fontSize=10, alignment=TA_CENTER, spaceBefore=0, spaceAfter=4)),
    Paragraph(
        "Il presente documento illustra la scelta e il funzionamento di MAPPO "
        "(Multi-Agent Proximal Policy Optimization) come algoritmo di "
        "apprendimento per rinforzo multi-agente adottato nella Fase 3 del "
        "simulatore Event-Driven Simulator (EDS). Dopo aver inquadrato il problema "
        "come un Decentralized Partially Observable Markov Decision Process "
        "(Dec-POMDP), si analizzano le motivazioni che hanno portato alla "
        "selezione di MAPPO rispetto ad algoritmi alternativi, si descrive "
        "l'architettura delle reti neurali Actor e Critic, e si dettaglia "
        "l'integrazione con il simulatore. Il documento si conclude delineando "
        "la transizione naturale verso la Fase 4, basata su agenti LLM autonomi.",
        ABSTRACT),
    PageBreak(),
]

# ── 1. inquadramento ───────────────────────────────────────────────────────────
story += [
    Paragraph("1. Inquadramento del Problema", H1),
    Paragraph(
        "Il controllo della congestione in una rete con molteplici router "
        "costituisce un problema intrinsecamente distribuito: ciascun nodo deve "
        "prendere decisioni locali in tempo reale, senza poter osservare "
        "direttamente lo stato dell'intera infrastruttura. Tale classe di "
        "problemi è formalizzata nella letteratura come "
        "<b>Decentralized Partially Observable Markov Decision Process (Dec-POMDP)</b>.",
        BODY),
    Paragraph(
        "Formalmente, un Dec-POMDP è definito dalla tupla "
        "(S, {A<sub>i</sub>}, T, R, {O<sub>i</sub>}, {Z<sub>i</sub>}, gamma), dove:",
        BODY),
]

dec_data = [
    ["Simbolo", "Significato nel contesto EDS"],
    ["S", "Stato globale: occupancy di tutte le code, rate di tutti i flussi, stato dei link"],
    ["A_i", "Azione dell'agente i: {ESCALATE, MAINTAIN, DE-ESCALATE} sullo stato di compressione"],
    ["T", "Funzione di transizione: evoluzione della rete dopo le azioni degli agenti"],
    ["R", "Reward condiviso: funzione di PDR, latenza, Jain Fairness Index, drop rate"],
    ["O_i", "Osservazione locale dell'agente i: ewma_occupancy, drop rate, link utilization, ..."],
    ["Z_i", "Funzione di osservazione: mappa stato globale -> osservazione parziale locale"],
    ["gamma", "Fattore di sconto: gamma = 0.99 (orizzonte lungo, reward futuro quasi equivalente)"],
]
story.append(sp(4))
story.append(Table(dec_data, colWidths=[3*cm, 13.5*cm], style=table_style()))
story.append(CAPTION and Paragraph("Tabella 1 — Componenti del Dec-POMDP nel sistema EDS.", CAPTION))

story += [
    sp(6),
    Paragraph(
        "La caratteristica fondamentale che rende il problema complesso è la "
        "<b>osservabilità parziale</b>: ogni router vede soltanto la propria coda "
        "locale, ignorando lo stato degli altri nodi. Questo crea una dipendenza "
        "implicita tra le decisioni degli agenti: la scelta di escalare la "
        "compressione su un nodo influenza il traffico sugli altri, ma tale "
        "influenza non è direttamente osservabile localmente.",
        BODY),
]

# ── 2. motivazione ─────────────────────────────────────────────────────────────
story += [
    Paragraph("2. Motivazione della Scelta di MAPPO", H1),
    Paragraph(
        "La selezione dell'algoritmo per la Fase 3 è avvenuta valutando quattro "
        "classi principali di algoritmi multi-agente cooperativi, analizzandone "
        "la compatibilità con i requisiti del sistema EDS: stabilità del training, "
        "sample efficiency, deployabilità su hardware reale e coerenza con la "
        "narrativa della tesi.",
        BODY),
]

comp_data = [
    ["Algoritmo", "Paradigma", "Criticità nel contesto EDS", "Esito"],
    ["IQL\n(Ind. Q-Learning)", "Decentralizzato\nQ-learning",
     "Non-stazionarietà: ogni agente percepisce un ambiente\nnon-MDP perché gli altri agenti cambiano policy\nconcorrentemente. Convergenza non garantita.",
     "Scartato"],
    ["MADDPG", "CTDE\nPolicy gradient\ndeterministico",
     "Progettato per action space continuo. Il replay buffer\nriduce la sample efficiency in ambienti simulati.\nL'action space discreto (3 azioni) non è il caso d'uso ottimale.",
     "Scartato"],
    ["QMIX", "CTDE\nValue decomposition",
     "L'ipotesi di monotonic mixing Q_tot = f(Q_1,...,Q_N)\nlimita l'espressivita' della cooperazione. Non\ngiustificata teoricamente per task di compressione.",
     "Scartato"],
    ["IPPO", "Decentralizzato\nPPO indipendente",
     "Ogni agente allena PPO separatamente senza\nosservare gli altri: stessa non-stazionarieta' di IQL,\nmitigata solo dalla stabilita' del clipping PPO.",
     "Alternativa\ndebolе"],
    ["MAPPO", "CTDE\nPPO multi-agente",
     "Risolve la non-stazionarieta' tramite critic centralizzato.\nEsecuzione decentralizzata: nessun coordinator a runtime.\nStabilita' PPO. SOTA su benchmark cooperativi (SMACv2).",
     "SCELTO"],
]
story.append(Table(comp_data,
    colWidths=[2.8*cm, 3*cm, 8.5*cm, 2.2*cm],
    style=table_style()))
story.append(Paragraph("Tabella 2 — Confronto algoritmi MARL cooperativi.", CAPTION))

story += [
    sp(6),
    Paragraph(
        "MAPPO è risultato l'unico algoritmo a soddisfare simultaneamente tutti "
        "i requisiti: (1) gestione della non-stazionarietà tramite il paradigma "
        "CTDE, (2) stabilità garantita dal clipping PPO, (3) esecuzione a runtime "
        "senza comunicazione inter-agente — proprietà fondamentale per il deployment "
        "su hardware reale nell'emulatore ContainerLab — e (4) struttura "
        "multi-agente che anticipa e prepara la transizione alla Fase 4 basata "
        "su agenti LLM.",
        BODY),
    Paragraph(
        "Un ulteriore vantaggio di MAPPO in questo contesto è il "
        "<b>parameter sharing</b>: poiché tutti i router svolgono il medesimo "
        "ruolo funzionale (nodi omogenei), è possibile condividere i pesi "
        "dell'actor tra tutti gli agenti. Ciò equivale a moltiplicare per N la "
        "quantità di esperienza disponibile per lo stesso numero di parametri, "
        "aumentando sensibilmente la sample efficiency.",
        BODY),
    PageBreak(),
]

# ── 3. background PPO ──────────────────────────────────────────────────────────
story += [
    Paragraph("3. Background: Proximal Policy Optimization (PPO)", H1),
    Paragraph("3.1  Policy Gradient e il Problema della Stabilità", H2),
    Paragraph(
        "I metodi di policy gradient aggiornano i parametri della policy "
        "theta nella direzione del gradiente del return atteso:",
        BODY),
    Paragraph(
        "    theta  <--  theta + alpha * grad_theta E[ log pi_theta(a|s) * A(s,a) ]",
        MATH),
    Paragraph(
        "dove A(s,a) è la funzione di vantaggio (advantage), che misura quanto "
        "l'azione a sia migliore rispetto alla baseline media nello stato s. "
        "Il problema fondamentale di questo aggiornamento è che un learning rate "
        "troppo elevato produce aggiornamenti destabilizzanti della policy, dai "
        "quali il processo di training non si riprende. Non esiste un valore "
        "universale di alpha che garantisca stabilità.",
        BODY),

    Paragraph("3.2  La Soluzione PPO: Clipping del Probability Ratio", H2),
    Paragraph(
        "PPO (Schulman et al., 2017) risolve il problema limitando "
        "esplicitamente la distanza tra la nuova policy e quella precedente. "
        "Definito il rapporto di probabilità:",
        BODY),
    Paragraph("    r(theta) = pi_theta(a|s) / pi_theta_old(a|s)", MATH),
    Paragraph(
        "l'obiettivo PPO-CLIP è:",
        BODY),
    Paragraph(
        "    L_CLIP(theta) = E[ min( r(theta) * A_hat,  clip(r(theta), 1-eps, 1+eps) * A_hat ) ]",
        MATH),
    Paragraph(
        "con eps = 0.2 tipicamente. La funzione min garantisce un comportamento "
        "conservativo: se il vantaggio è positivo (azione buona), si impedisce "
        "a r(theta) di crescere oltre 1+eps; se il vantaggio è negativo (azione "
        "cattiva), si impedisce a r(theta) di scendere sotto 1-eps. "
        "Il risultato è un aggiornamento che migliora la policy "
        "senza stravolgerne il comportamento in un singolo passo.",
        BODY),

    Paragraph("3.3  Stima del Vantaggio tramite GAE", H2),
    Paragraph(
        "PPO utilizza la Generalized Advantage Estimation (GAE, Schulman et al., 2016) "
        "per calcolare il vantaggio A_hat con un compromesso controllato tra "
        "varianza e bias:",
        BODY),
    Paragraph(
        "    delta_t  =  r_t  +  gamma * V(s_{t+1})  -  V(s_t)",
        MATH),
    Paragraph(
        "    A_hat_t  =  sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}",
        MATH),
    Paragraph(
        "dove gamma = 0.99 è il fattore di sconto e lambda = 0.95 il parametro "
        "di mixing bias-varianza. V(s) è la rete <b>Critic</b>, che stima "
        "il valore atteso del return dallo stato s.",
        BODY),

    Paragraph("3.4  Architettura Actor-Critic", H2),
    Paragraph(
        "PPO adotta un'architettura Actor-Critic con due reti neurali distinte:",
        BODY),
]

ac_data = [
    ["Componente", "Ruolo", "Input", "Output"],
    ["Actor  pi(a|s, theta)",
     "Prende decisioni: seleziona l'azione da eseguire",
     "Osservazione locale s (o parziale o_i)",
     "Distribuzione di probabilita' P(a) sullo spazio azioni"],
    ["Critic  V(s, phi)",
     "Valuta gli stati: stima il return atteso",
     "Stato (locale o globale a seconda del paradigma)",
     "Scalare: valore V(s) usato per calcolare A_hat"],
]
story.append(Table(ac_data, colWidths=[4*cm, 4.5*cm, 4*cm, 4*cm],
    style=table_style()))
story.append(Paragraph("Tabella 3 — Ruoli delle reti Actor e Critic.", CAPTION))

story += [
    sp(6),
    Paragraph(
        "Una metafora utile per chiarire la distinzione: l'Actor è il "
        "calciatore in campo, che vede solo ciò che gli è intorno e deve "
        "decidere in tempo reale; il Critic è l'allenatore in panchina, che "
        "osserva l'intera partita e fornisce un giudizio retrospettivo sulla "
        "qualità di ogni decisione presa. Il Critic non tocca mai il pallone — "
        "non agisce — ma guida l'apprendimento dell'Actor.",
        BODY),
    PageBreak(),
]

# ── 4. MAPPO ───────────────────────────────────────────────────────────────────
story += [
    Paragraph("4. MAPPO: Estensione al Contesto Multi-Agente", H1),
    Paragraph("4.1  Il Paradigma CTDE", H2),
    Paragraph(
        "La sfida principale del MARL cooperativo è la <b>non-stazionarietà</b>: "
        "dal punto di vista di un singolo agente i, l'ambiente appare non-MDP "
        "perché le transizioni di stato dipendono anche dalle azioni degli agenti "
        "j ≠ i, le cui policy cambiano durante il training. "
        "MAPPO risolve questo problema adottando il paradigma "
        "<b>Centralized Training, Decentralized Execution (CTDE)</b>:",
        BODY),
    Paragraph(
        "durante il training, il Critic ha accesso allo stato globale "
        "s_global = [o_1, o_2, ..., o_N, stati_link, rate_flussi]; "
        "durante l'esecuzione, ogni agente i utilizza esclusivamente la "
        "propria policy locale pi_i(a_i | o_i, theta_i), senza alcuna "
        "comunicazione con gli altri agenti.",
        BODY),
    Paragraph(
        "Il Critic centralizzato rende l'ambiente stazionario dal proprio "
        "punto di vista: osservando tutti gli agenti, può calcolare vantaggi "
        "A_hat precisi che guidano gli Actor a coordinarsi implicitamente, "
        "anche senza comunicazione diretta.",
        BODY),

    Paragraph("4.2  Obiettivo di Training", H2),
    Paragraph(
        "In MAPPO, ogni agente i massimizza il proprio contributo "
        "all'obiettivo condiviso tramite il seguente loss clippato:",
        BODY),
    Paragraph(
        "    L_i(theta) = E_t[ min( r_i(theta) * A_hat_t,  "
        "clip(r_i(theta), 1-eps, 1+eps) * A_hat_t ) ]",
        MATH),
    Paragraph(
        "dove r_i(theta) = pi_{theta,i}(a_i|o_i) / pi_{theta_old,i}(a_i|o_i) "
        "e A_hat_t è calcolato usando il Critic centralizzato V(s_global, phi). "
        "Il Critic è aggiornato minimizzando l'errore quadratico medio:",
        BODY),
    Paragraph(
        "    L_critic(phi) = E_t[ (V_phi(s_t) - V_target_t)^2 ]",
        MATH),
    Paragraph(
        "con V_target calcolato tramite bootstrap: "
        "V_target_t = r_t + gamma * V(s_{t+1}).",
        BODY),

    Paragraph("4.3  Training Loop", H2),
    Paragraph(
        "Il training MAPPO segue un ciclo iterativo di raccolta esperienza "
        "e aggiornamento dei parametri:",
        BODY),
]

loop_data = [
    ["Step", "Operazione", "Dettaglio"],
    ["1", "Raccolta rollout",
     "T = 2048 passi nel simulatore EDS con policy corrente pi_theta. "
     "Ogni passo: osservazione o_i -> azione a_i -> reward r_i."],
    ["2", "Calcolo vantaggi",
     "Usa Critic V(s_global) per calcolare delta_t e poi A_hat_t "
     "tramite GAE con gamma=0.99, lambda=0.95."],
    ["3", "Update Actor",
     "K = 10 epoch di gradient ascent su L_CLIP(theta) con mini-batch "
     "da 256 step. Gradient clipping con max_norm=10."],
    ["4", "Update Critic",
     "K = 10 epoch di gradient descent su L_critic(phi). "
     "Learning rate critic = 1e-3 (piu' alto dell'Actor = 3e-4)."],
    ["5", "Iterazione",
     "Ripeti per M = 500 episodi. Episodio = 1 simulazione EDS "
     "su scenario random tra i 6 scenari canonici."],
]
story.append(Table(loop_data, colWidths=[1.2*cm, 3.5*cm, 11.8*cm],
    style=table_style()))
story.append(Paragraph("Tabella 4 — Ciclo di training MAPPO.", CAPTION))

story += [
    PageBreak(),
]

# ── 5. architettura ────────────────────────────────────────────────────────────
story += [
    Paragraph("5. Architettura delle Reti Neurali", H1),
    Paragraph("5.1  Actor Network", H2),
    Paragraph(
        "Ogni agente i dispone di un Actor con la seguente struttura:",
        BODY),
    Paragraph(
        "    Input(7) --> LayerNorm --> Linear(7->64) --> Tanh "
        "--> Linear(64->64) --> Tanh --> Linear(64->3) --> Softmax",
        MATH),
]

actor_data = [
    ["Layer", "Dimensione", "Attivazione", "Note"],
    ["Input", "7", "—",
     "ewma_occ, stato/4, hi_pri, lo_pri, drop_rate, link_util, t_stato/T"],
    ["LayerNorm", "7", "—",
     "Normalizza feature con scale eterogenee. Trick chiave paper MAPPO."],
    ["Hidden 1", "64", "Tanh",
     "W in R^{64x7}. Tanh preferita a ReLU: output bounded, gradienti stabili."],
    ["Hidden 2", "64", "Tanh", "W in R^{64x64}."],
    ["Output", "3", "Softmax",
     "P(ESCALATE), P(MAINTAIN), P(DE-ESCALATE). Campionamento stocastico."],
]
story.append(Table(actor_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 9*cm],
    style=table_style()))
story.append(Paragraph(
    "Tabella 5 — Architettura Actor. Parametri totali: 7x64+64+64x64+64+64x3+3 = 4.931.",
    CAPTION))

story += [
    sp(6),
    Paragraph("5.2  Critic Network", H2),
    Paragraph(
        "Il Critic è più ampio dell'Actor perché deve elaborare lo stato globale "
        "di tutti gli agenti:",
        BODY),
    Paragraph(
        "    Input(7N+4) --> LayerNorm --> Linear(->128) --> Tanh "
        "--> Linear(128->128) --> Tanh --> Linear(128->1)",
        MATH),
]

critic_data = [
    ["Layer", "Dimensione", "Attivazione", "Note"],
    ["Input", "7N+4", "—",
     "Concatenazione osservazioni di tutti gli N agenti + link_util + flow_rates. "
     "Single bottleneck (N=1): dim=11."],
    ["LayerNorm", "7N+4", "—", "Stesso motivo dell'Actor."],
    ["Hidden 1", "128", "Tanh", "Doppio rispetto all'Actor: elabora stato globale piu' ricco."],
    ["Hidden 2", "128", "Tanh", "W in R^{128x128}."],
    ["Output", "1", "Lineare",
     "Scalare V(s). Nessuna attivazione: il valore puo' essere negativo."],
]
story.append(Table(critic_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 9*cm],
    style=table_style()))
story.append(Paragraph(
    "Tabella 6 — Architettura Critic (N=1). Parametri totali: ~18.177.",
    CAPTION))

story += [
    sp(6),
    Paragraph("5.3  Spazio delle Osservazioni e delle Azioni", H2),
]

obs_data = [
    ["Feature", "Simbolo", "Range", "Motivazione"],
    ["EWMA queue occupancy", "ewma_occ", "[0, 1]",
     "Segnale primario di congestione, gia' disponibile in Phase 2"],
    ["Stato corrente normalizzato", "state / 4", "{0, 0.25, 0.5, 0.75, 1}",
     "Fornisce contesto sulla posizione attuale nella macchina a stati"],
    ["Ratio traffico alta priorita'", "hi_pri_ratio", "[0, 1]",
     "Quantita' di traffico CONTROL nella coda (da proteggere)"],
    ["Ratio traffico bassa priorita'", "lo_pri_ratio", "[0, 1]",
     "Quantita' di traffico VIDEO nella coda (sacrificabile)"],
    ["Drop rate recente", "drop_rate", "[0, 1]",
     "Segnale di congestione acuta nell'ultima finestra temporale"],
    ["Utilizzo del link", "link_util", "[0, 1]",
     "Carico attuale sul link bottleneck (byte/s / capacita')"],
    ["Tempo nello stato corrente", "t_stato / T_max", "[0, 1]",
     "Evita transizioni troppo rapide (complementa l'isteresi di Phase 2)"],
]
story.append(Table(obs_data, colWidths=[4.2*cm, 3*cm, 1.8*cm, 7.5*cm],
    style=table_style()))
story.append(Paragraph("Tabella 7 — Vettore di osservazione locale o_i (dim=7).", CAPTION))

story += [
    sp(6),
    Paragraph(
        "Lo spazio delle azioni è discreto con 3 opzioni, coerente con la "
        "filosofia eFRAC di transizioni graduali (Abate, Sacco, Fiore, Esposito):",
        BODY),
]

act_data = [
    ["Azione", "Valore", "Effetto"],
    ["ESCALATE",     "0", "Avanza di un livello: es. NORMAL -> HEADER_COMPRESSION"],
    ["MAINTAIN",     "1", "Rimane nello stato corrente"],
    ["DE-ESCALATE",  "2", "Retrocede di un livello: es. DELTA -> HEADER_COMPRESSION"],
]
story.append(Table(act_data, colWidths=[4*cm, 2*cm, 10.5*cm],
    style=table_style()))
story.append(Paragraph("Tabella 8 — Spazio delle azioni discreto (3 azioni).", CAPTION))

story += [
    PageBreak(),
]

# ── 6. reward ──────────────────────────────────────────────────────────────────
story += [
    Paragraph("6. Funzione di Reward", H1),
    Paragraph(
        "La funzione di reward è calcolata a ogni step di osservazione "
        "(ogni delta_t = 1 secondo di simulazione) e integra quattro "
        "componenti che riflettono gli obiettivi di qualità della rete:",
        BODY),
    Paragraph(
        "    r_t  =  PDR_t  -  lambda_d * drop_rate_t  "
        "-  lambda_l * (latency_t / latency_max)  +  lambda_f * J_t",
        MATH),
]

rew_data = [
    ["Termine", "Simbolo", "Peso", "Descrizione"],
    ["Packet Delivery Ratio", "PDR_t", "1.0",
     "Pacchetti consegnati / generati nella finestra. Obiettivo primario."],
    ["Drop rate", "lambda_d * drop_rate_t", "0.3",
     "Penalizza i drop attivi (coda piena + DROP_LOW_PRIORITY). "
     "Segnale diretto di congestione grave."],
    ["Latenza normalizzata", "lambda_l * lat_t/lat_max", "0.2",
     "Penalizza latenza end-to-end eccessiva. lat_max = 2 secondi "
     "(soglia accettabilita' per traffico di controllo)."],
    ["Jain Fairness Index", "lambda_f * J_t", "0.2",
     "Premia equita' tra flussi. J = (sum x_i)^2 / (N * sum x_i^2). "
     "Evita che l'agente sacrifichi sistematicamente flussi a bassa priorita'."],
]
story.append(Table(rew_data, colWidths=[3.5*cm, 4.5*cm, 1.5*cm, 7*cm],
    style=table_style()))
story.append(Paragraph("Tabella 9 — Componenti della funzione di reward.", CAPTION))

story += [
    sp(6),
    Paragraph(
        "I pesi lambda_d = 0.3, lambda_l = 0.2, lambda_f = 0.2 "
        "sono valori iniziali da sottoporre a hyperparameter tuning "
        "durante la fase di training. La somma dei pesi non deve superare "
        "1 per evitare che i termini negativi dominino il reward positivo "
        "del PDR in condizioni nominali.",
        BODY),
]

# ── 7. integrazione EDS ────────────────────────────────────────────────────────
story += [
    Paragraph("7. Integrazione con il Simulatore EDS", H1),
    Paragraph("7.1  Punto di Integrazione", H2),
    Paragraph(
        "Il simulatore EDS espone già il metodo <b>react(event, scheduler)</b> "
        "nella classe RuleBasedController (simulator/control/rule_based.py), "
        "invocato ad ogni evento STATE_UPDATE. Nella Fase 3, questo metodo "
        "viene sostituito da MARLController.react(), che:",
        BODY),
    Paragraph("— accumula le osservazioni o_i in un buffer di rollout;", BODY_SM),
    Paragraph("— ogni T = 2048 passi, invoca la policy Actor pi(a|o) e applica l'azione "
              "chiamando node.state_machine.transition(new_state);", BODY_SM),
    Paragraph("— calcola il reward dalla finestra di metriche MetricsEngine;", BODY_SM),
    Paragraph("— al termine del rollout, esegue l'update PPO (Actor + Critic).", BODY_SM),

    Paragraph("7.2  Pipeline Training e Validazione", H2),
    Paragraph(
        "La pipeline segue uno schema di curriculum learning sui 6 scenari "
        "canonici del simulatore:",
        BODY),
]

pipe_data = [
    ["Fase", "Operazione", "Dettaglio"],
    ["Training", "500 episodi nel Simulator EDS",
     "Scenario random tra i 6 ad ogni episodio. "
     "Ogni episodio dura end_time = 100s simulati."],
    ["Valutazione", "Ogni 50 episodi",
     "Esecuzione deterministica (argmax invece di campionamento) "
     "su tutti e 6 gli scenari. Metriche: PDR, latenza, Jain, compression_ratio."],
    ["Export", "Checkpoint migliore",
     "Salvataggio pesi theta in formato JSON / PyTorch .pt "
     "al miglioramento del reward medio di validazione."],
    ["Deploy emulatore", "ContainerLab",
     "Actor caricato come rete neurale standalone nel container router. "
     "Critic non necessario: sparisce in produzione."],
    ["Confronto", "Phase 2 vs Phase 3",
     "Stessi 6 scenari sull'emulatore hardware. "
     "KPI: PDR, latenza, Jain, compression_ratio, state_transitions."],
]
story.append(Table(pipe_data, colWidths=[2.5*cm, 3.5*cm, 10.5*cm],
    style=table_style()))
story.append(Paragraph("Tabella 10 — Pipeline training, validazione e deploy.", CAPTION))

story += [
    PageBreak(),
]

# ── 8. transizione fase 4 ──────────────────────────────────────────────────────
story += [
    Paragraph("8. Transizione verso la Fase 4: Agentic AI", H1),
    Paragraph(
        "MAPPO è stato scelto, oltre che per le sue qualità intrinseche, anche "
        "perché i suoi limiti strutturali costituiscono la motivazione accademica "
        "per la Fase 4 del progetto.",
        BODY),
    Paragraph(
        "I limiti di MAPPO che la Fase 4 intende superare sono:",
        BODY),
    Paragraph("— <b>Policy congelata</b>: dopo il training, MAPPO non si adatta a scenari "
              "significativamente fuori distribuzione senza retraining.", BODY_SM),
    Paragraph("— <b>Assenza di spiegabilità</b>: la rete neurale non può giustificare "
              "le proprie decisioni in linguaggio naturale.", BODY_SM),
    Paragraph("— <b>Mancanza di ragionamento</b>: non è in grado di pianificare "
              "su orizzonti lunghi o di usare conoscenza di dominio esplicita.", BODY_SM),
    Paragraph("— <b>Tool use limitato</b>: non può interrogare sistemi esterni, "
              "modificare la propria configurazione o coordinarsi via messaggi.", BODY_SM),
    sp(6),
    Paragraph(
        "La struttura multi-agente di MAPPO si mappa direttamente sulla Fase 4:",
        BODY),
]

trans_data = [
    ["Fase 3 — MAPPO", "Fase 4 — Agentic LLM"],
    ["Actor locale pi(a|o_i, theta)",    "LLM agent locale con tool use e memoria"],
    ["Critic centralizzato V(s_global)", "Orchestratore LLM globale / shared memory"],
    ["Reward r_t",                       "Feedback metrico + critica in linguaggio naturale"],
    ["Azione discreta {0,1,2}",          "Tool call: escalate() / maintain() / deescalate()"],
    ["Parameter sharing theta",          "Stesso LLM base su tutti gli agenti (prompt comune)"],
    ["Aggiornamento PPO",                "In-context learning / RAG su nuovi scenari"],
]
story.append(Table(trans_data, colWidths=[8.25*cm, 8.25*cm],
    style=table_style()))
story.append(Paragraph("Tabella 11 — Mappatura componenti Fase 3 -> Fase 4.", CAPTION))

story += [
    sp(6),
    Paragraph(
        "La progressione 2 -> 3 -> 4 costituisce una narrativa di tesi coerente "
        "e progressiva: da regole deterministiche (Phase 2, eFRAC) a policy appresa "
        "da dati (Phase 3, MAPPO) ad agente autonomo ragionante "
        "(Phase 4, LLM multi-agent). Ogni fase supera un limite preciso della "
        "precedente, giustificando l'avanzamento.",
        BODY),
    PageBreak(),
]

# ── riferimenti ────────────────────────────────────────────────────────────────
story += [
    Paragraph("Riferimenti Bibliografici", H1),
    hr(),
    Paragraph(
        "[1] Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). "
        "<i>The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.</i> "
        "Advances in Neural Information Processing Systems (NeurIPS), vol. 35. "
        "arXiv:2103.01955.",
        REF),
    Paragraph(
        "[2] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "
        "<i>Proximal Policy Optimization Algorithms.</i> arXiv:1707.06347.",
        REF),
    Paragraph(
        "[3] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). "
        "<i>High-Dimensional Continuous Control Using Generalized Advantage Estimation.</i> "
        "ICLR 2016. arXiv:1506.02438.",
        REF),
    Paragraph(
        "[4] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). "
        "<i>Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.</i> "
        "NeurIPS 2017. arXiv:1706.02275.",
        REF),
    Paragraph(
        "[5] Rashid, T., Samvelyan, M., de Witt, C. S., Farquhar, G., Foerster, J., & "
        "Whiteson, S. (2018). <i>QMIX: Monotonic Value Function Factorisation for Deep "
        "Multi-Agent Reinforcement Learning.</i> ICML 2018. arXiv:1803.11605.",
        REF),
    Paragraph(
        "[6] Abate, M., Sacco, A., Fiore, M., & Esposito, F. (2024). "
        "<i>eFRAC: Elastic Flow-Rate Adaptive Compression for Network Congestion Management.</i> "
        "(Paper di riferimento per Phase 2 e la logica di compressione semantica.)",
        REF),
    Paragraph(
        "[7] Bernstein, D. S., Givan, R., Immerman, N., & Zilberstein, S. (2002). "
        "<i>The Complexity of Decentralized Control of Markov Decision Processes.</i> "
        "Mathematics of Operations Research, 27(4), 819-840.",
        REF),
]

# ── build ──────────────────────────────────────────────────────────────────────

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2*cm, 1.2*cm,
        "MAPPO — Fase 3 EDS  |  Towards Agentic Networks: Autonomous Congestion Management")
    canvas.drawRightString(W - 2*cm, 1.2*cm, f"Pagina {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#dddddd"))
    canvas.line(2*cm, 1.6*cm, W - 2*cm, 1.6*cm)
    canvas.restoreState()

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm,  bottomMargin=2.5*cm,
    title="MAPPO — Fase 3 EDS",
    author="Flavio Bianco",
    subject="Multi-Agent PPO per controllo congestione di rete",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF generato: {OUTPUT}")
