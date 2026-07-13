# Towards Agentic Networks: Autonomous Congestion Management
## Report di ricerca — base documentale per la tesi di laurea magistrale

**Autore:** Flavio Bianco
**Contesto:** Tesi di Laurea Magistrale, A.A. 2025/2026
**Repository:** `Event-Driven_Simulator` (branch `phase4-llm-supervisor`) e `eds-containerlab`

---

## 1. Introduzione e Panoramica del Progetto

### 1.1 Contesto e motivazione

La gestione della congestione nelle reti di telecomunicazione costituisce un problema classico dell'ingegneria dei sistemi distribuiti, tradizionalmente affrontato mediante meccanismi reattivi a soglia (AQM, controllo di finestra TCP) che operano esclusivamente sul piano del *trasporto* dei dati. Il presente lavoro si colloca in un filone di ricerca più recente — quello delle *reti agentiche* (agentic networks) — che si interroga su come tecniche di intelligenza artificiale, dal reinforcement learning multi-agente ai modelli linguistici di grandi dimensioni, possano conferire alla rete capacità di gestione autonoma: non soltanto reagire alla congestione, ma comprenderla, spiegarla e adattare la propria strategia.

La leva di controllo adottata è la **compressione semantica adattiva del traffico**: anziché scartare pacchetti indiscriminatamente in condizioni di congestione, il sistema modula il livello di compressione applicato ai flussi in funzione dello stato della rete, secondo una macchina a stati a cinque livelli di aggressività crescente (nessuna compressione, compressione delle intestazioni, compressione differenziale, compressione incrementale, scarto selettivo del traffico a bassa priorità). Tale approccio, ispirato al framework eFRAC (Elastic Flow-Rate Adaptive Compression), consente di preservare l'informazione a maggior valore semantico — in particolare il traffico di controllo — degradando in modo controllato quella a minor priorità.

### 1.2 Obiettivi e articolazione in fasi

Il progetto è stato articolato in quattro fasi incrementali, ciascuna delle quali costruisce sulla precedente e ne costituisce la baseline di valutazione:

| Fase | Contenuto | Paradigma di controllo | Esito sintetico |
|------|-----------|------------------------|-----------------|
| 1 | Simulatore di rete a eventi discreti | — (infrastruttura) | Completata e validata (25 test; modello M/D/1) |
| 2 | Controllo a regole (eFRAC) | EWMA + isteresi su soglie | Baseline funzionante su 6 scenari canonici |
| 3 | Controllo appreso (MARL) | MAPPO, actor deterministico | Supera la baseline su 5/5 KPI globali; validata su emulatore hardware |
| 4 | Supervisione con modelli linguistici | SLM su percorso lento; svolta agentica | Spiegabilità dimostrata; ruolo decisionale ricollocato; agente investigativo |

L'obiettivo scientifico complessivo non era la mera dimostrazione che "l'IA migliora la rete", bensì la **caratterizzazione rigorosa di quale paradigma di intelligenza sia appropriato a quale livello temporale e funzionale del controllo**. Come si argomenterà nelle sezioni 4 e 5, il risultato più significativo del lavoro è proprio la delimitazione empirica dei confini di applicabilità di ciascun paradigma.

### 1.3 Architettura complessiva

L'architettura finale è organizzata su due livelli temporali, secondo un'analogia esplicita con la teoria dei processi cognitivi duali di Kahneman (System 1 / System 2):

- **Percorso veloce (System 1):** una politica MAPPO congelata, esportata come actor puramente NumPy (~24 KB), decide ogni secondo il livello di compressione. Deterministica, riproducibile, a latenza di calcolo sub-millisecondo (misurata: ≈0,35 ms per decisione).
- **Percorso lento (System 2):** un livello supervisorio interviene su cadenza di ~30 secondi o su anomalia. Nella sua forma finale esso comprende: (i) una guardia deterministica a soglia per la valutazione dello stato di salute; (ii) uno Small Language Model (Qwen2.5-3B, eseguito localmente via Ollama) per la spiegazione in linguaggio naturale; (iii) un **agente investigativo** dotato di strumenti (tool) per la diagnosi dei regimi anomali.

Un vincolo architetturale è stato mantenuto inderogabile per l'intera durata del progetto: *il modello linguistico non entra mai nel ciclo di controllo per-secondo*. Ne consegue, per costruzione e per verifica empirica (sezione 5.4), che nessuna scelta relativa al supervisore può degradare le prestazioni validate del percorso veloce.

> **Suggerimento grafico n. 1 — Architettura a due livelli.** Diagramma a blocchi con i due percorsi: in alto il ciclo veloce (osservazioni → actor MAPPO → azione di compressione, cadenza 1 s), in basso il ciclo lento (metriche di finestra → assess deterministico → agente/SLM → eventuale override vincolato dal guardrail, cadenza 30 s). Evidenziare con colori distinti la direzione dei flussi informativi e il punto di applicazione dell'override. In alternativa, riutilizzare la timeline già prodotta (`agent_timeline.png`), che riporta: barra continua del percorso veloce, punti di campionamento della guardia, risveglio dell'agente a t=60, intervento a t=90.

### 1.4 Valore scientifico e pratico

Sul piano pratico, il sistema dimostra un controllo di congestione che, rispetto alla baseline a regole, incrementa il rapporto di consegna dei pacchetti (PDR) dal 90,12% al 92,95%, riduce la latenza media da 771 ms a 224 ms e aumenta l'efficacia di compressione da 1,15× a 1,48×, con validazione su emulatore hardware (ContainerLab con compressione reale dei byte via iptables NFQUEUE) e uno scarto simulazione-reale trascurabile. Sul piano scientifico, il contributo risiede in tre risultati metodologici: la **separazione architetturale fra decisione ed spiegazione** nei sistemi supervisionati da modelli linguistici; l'identificazione di **due limiti di natura distinta** (floor di capacità e floor di osservabilità) che precludono l'uso decisionale diretto degli SLM; la dimostrazione che il **ciclo agentico** (osservare–indagare–agire) risolve il secondo limite laddove nessun decisore one-shot, per quanto capace, potrebbe farlo.

---

## 2. Processo Decisionale e Motivazioni

Questa sezione ricostruisce le decisioni strategiche e tecniche assunte, con le rispettive giustificazioni teoriche o logiche. Una sintesi tabellare è fornita in coda (Tabella 2.1).

### 2.1 Simulazione a eventi discreti come fondamento sperimentale

Si è scelto di costruire un simulatore a eventi discreti dedicato (14 tipi di evento, scheduler a coda di priorità, modello di servizio M/D/1 con arrivi poissoniani e servizio deterministico) anziché adottare simulatori generalisti (ns-3, OMNeT++). La motivazione è duplice: da un lato il pieno controllo del modello di compressione semantica, non nativamente supportato dagli strumenti esistenti; dall'altro la necessità di un ambiente *leggero e interrogabile programmaticamente* per l'addestramento per rinforzo (migliaia di episodi in tempi dell'ordine dei minuti — il training completo di Fase 3, 500 episodi, si esegue in ~25 s). La correttezza del nucleo è stata verificata mediante confronto con i risultati analitici della teoria delle code su casi limite noti.

### 2.2 Baseline a regole: EWMA con isteresi

La Fase 2 implementa il controllo a regole con media mobile esponenziale (EWMA) dell'occupazione di coda e isteresi sulle transizioni di stato. L'isteresi è motivata dalla necessità di evitare oscillazioni rapide fra stati di compressione adiacenti (fenomeno di *flapping*), che comportano costi di riconfigurazione. Questa baseline non è un mero termine di paragone: definisce le *soglie semantiche* degli stati che tutte le fasi successive riutilizzano.

### 2.3 Scelta del paradigma di apprendimento: MAPPO

Per la Fase 3 si è adottato MAPPO (Multi-Agent Proximal Policy Optimization) nella configurazione *centralized training, decentralized execution* (CTDE). Le ragioni: (i) PPO offre stabilità di addestramento documentata in ambienti cooperativi multi-agente (Yu et al., 2022) senza la fragilità degli approcci value-based rispetto alla non-stazionarietà; (ii) il CTDE consente un critic centralizzato in addestramento mantenendo l'esecuzione locale per nodo, coerente con il vincolo di deployment; (iii) lo spazio di azione relativo {ESCALATE, MAINTAIN, DE-ESCALATE} rispetta per costruzione la struttura a macchina a stati, evitando salti non fisici fra livelli di compressione.

Una decisione implementativa di rilievo è stata la **realizzazione in NumPy puro** (retropropagazione manuale, ottimizzatore Adam implementato ex novo, LayerNorm non affine), senza dipendenza da framework di deep learning. La motivazione è il deployment: l'actor esportato è un file JSON di ~500 KB caricabile da un interprete Python privo di dipendenze, requisito essenziale per l'esecuzione su nodi emulati a risorse contenute. La funzione di ricompensa segue la specifica documentale — r = PDR − 0,3·drop − 0,2·(latenza/2s) + 0,2·Jain — con l'unica aggiunta di una **penalità di stabilità** (−0,1 per transizione di stato per passo), introdotta per ridurre il flapping osservato nella politica vanilla; la sezione 4.6 discute la verifica multi-seed di questa scelta.

### 2.4 Validazione su emulatore hardware

Si è ritenuto insufficiente validare la politica appresa nel solo simulatore di addestramento (rischio di *self-evaluation bias*). La politica è stata pertanto trasferita su un emulatore ContainerLab in cui la compressione avviene realmente sui byte dei pacchetti (iptables NFQUEUE). Il confronto ha mostrato uno scarto simulazione-reale minimo (PDR 92,95% simulato contro 93,0% emulato; latenza 224 ms contro 264 ms), con la baseline identica nei due ambienti a fungere da controllo di sanità. Questo risultato conferisce credibilità esterna a tutte le valutazioni successive condotte in simulazione.

### 2.5 Fase 4: Small Language Model anziché modello frontier

Per il livello supervisorio si è scelto di impiegare uno **Small Language Model locale** (Qwen2.5-3B via Ollama) anziché un modello di frontiera via API. Le motivazioni, allineate alla letteratura recente sull'IA agentica (Belcak et al., 2024): il compito supervisorio è *stretto e strutturato*, profilo per cui i modelli compatti sono documentatamente sufficienti; l'esecuzione locale garantisce riproducibilità (temperatura 0, seed fissato), operatività offline e costo nullo per inferenza; la dimensione del modello diventa essa stessa una variabile sperimentale (ablation 0,5B–7B). Il **constrained decoding** (schema JSON imposto al decodificatore) garantisce la validità sintattica dell'output a qualunque dimensione di modello: il modello sceglie *quale* azione, mai il formato.

### 2.6 Sicurezza by design: guardrail e milestone incrementali

L'intera Fase 4 è stata progettata attorno a un sistema di guardrail antecedente a qualunque autorità di controllo: azioni ammissibili solo da uno spazio vincolato e rivalidate; override limitati nel tempo (max 120 s) e reversibili; revoca automatica se il PDR di finestra scende sotto una soglia critica; kill switch globale; fail-safe sul backend (errore del modello → nessun intervento). Le milestone sono state ordinate per rischio crescente: M1 (osservazione pura, kill switch attivo), M2 (autorità di override), M3 (valutazione fuori distribuzione). L'intero sviluppo è avvenuto su branch separato e dichiaratamente sacrificabile, a protezione della base validata delle fasi precedenti. Come si vedrà (sezione 4.4), i guardrail si sono rivelati non un ornamento prudenziale ma il componente che ha materialmente contenuto i danni di una regola errata.

**Tabella 2.1 — Sintesi delle decisioni chiave e relative motivazioni.**

| # | Decisione | Motivazione principale | Riferimento |
|---|-----------|------------------------|-------------|
| D1 | Simulatore a eventi discreti dedicato | Controllo del modello di compressione; velocità per RL | §2.1 |
| D2 | Baseline EWMA + isteresi | Anti-flapping; definizione soglie semantiche | §2.2 |
| D3 | MAPPO (CTDE), azioni relative | Stabilità PPO; esecuzione decentralizzata; rispetto della macchina a stati | §2.3 |
| D4 | Implementazione NumPy pura, export JSON | Deployment su nodi leggeri senza dipendenze | §2.3 |
| D5 | Penalità di stabilità nel reward | Riduzione flapping (−40% transizioni, verifica multi-seed) | §2.3, §4.6 |
| D6 | Validazione su emulatore hardware | Eliminazione del self-evaluation bias | §2.4 |
| D7 | SLM locale, non modello frontier | Compito stretto; riproducibilità; ablation sulla dimensione | §2.5 |
| D8 | Constrained decoding | Validità sintattica garantita a ogni scala | §2.5 |
| D9 | Guardrail prima dell'autorità; milestone a rischio crescente | Sicurezza by design; contenimento dei danni | §2.6 |
| D10 | LLM mai nel ciclo per-secondo | Impossibilità strutturale di degradare la Fase 3 | §1.3, §5.4 |

> **Suggerimento grafico n. 2 — Curva di addestramento.** Grafico a linee: asse x = episodi di addestramento (0–500), asse y = ricompensa media per passo; sovrapporre la ricompensa di valutazione deterministica campionata ogni 50 episodi. Dati disponibili nei log di `train_mappo.py`. Annotare il punto di convergenza (~episodio 250, reward di valutazione 1,0692) e riportare in didascalia l'esito della verifica di convergenza a 4000 episodi (KPI invariati).

---

## 3. Evoluzione del Progetto e Cambiamenti in Corsa

Il piano originale della Fase 4 prevedeva un supervisore LLM che *decidesse* gli interventi correttivi e li *spiegasse*. L'evidenza sperimentale ha imposto una serie di revisioni sostanziali, documentate cronologicamente di seguito. Si sottolinea che ciascun pivot è stato guidato da dati riproducibili, non da preferenze progettuali.

### 3.1 Primo pivot: dalla decisione LLM alla separazione decisione/spiegazione

Durante la milestone M1 si è rilevato che il modello da 3 miliardi di parametri **non è in grado di confrontare due numeri in modo affidabile**: interrogato su metriche reali, ha qualificato come «basso» un PDR pari a 0,997 e come «alto» un tasso di scarto pari a 0,000, disattendendo la regola di soglia esplicitamente fornita nel prompt. Tre successive riformulazioni del prompt non hanno risolto il problema (sezione 4.1). Ne è derivata la prima revisione architetturale: la **decisione** (endorse/override e stato bersaglio) è stata affidata a una regola deterministica (`assess`: soglie su PDR e drop rate), mentre al modello linguistico è stata riservata la sola **spiegazione** del verdetto già calcolato. La separazione è stata successivamente blindata da test automatici che verificano l'impossibilità, per l'output del modello, di determinare azione o stato di controllo.

### 3.2 Secondo pivot: ridimensionamento dell'ablation sulla dimensione

L'ablation prevista (0,5B / 1,5B / 3B / 7B) è stata eseguita, ma la metrica automatica progettata (errori di direzione nelle spiegazioni) si è rivelata fuorviante: il modello più piccolo otteneva zero errori *perché si limitava a ripetere la valutazione iniettata nel prompt* (fenomeno di parroting), senza alcuna interpretazione. La qualità interpretativa reale è risultata crescente con la dimensione (solo il 7B legge la traiettoria e propone un rimedio). L'ablation è stata pertanto ridimensionata da risultato quantitativo a evidenza qualitativa, con l'acquisizione di una lezione metodologica: le metriche di correttezza superficiale premiano l'eco, non il ragionamento.

### 3.3 Terzo pivot: la regola di escalation, provata e rimossa

Per il caso del collasso di capacità si è introdotta una regola deterministica di escalation allo stato 4 (scarto attivo). L'esperimento è fallito su entrambi i fronti (dettagli in §4.3): la regola è stata **rimossa** e si è adottato il principio opposto — *first, do no harm* — fissato in un test di regressione che vieta lo stato 4 da soglie statiche. Si è scelto deliberatamente di documentare l'esperimento fallito nel report di fase anziché ometterlo, in ossequio al valore metodologico dei risultati negativi.

### 3.4 Quarto pivot: la svolta agentica

L'analisi delle cause profonde dei fallimenti decisionali (i «due floor», §4.1–4.2) ha condotto alla revisione più significativa: abbandonare il paradigma del **decisore one-shot** in favore di un **agente** che opera in un ciclo percezione–ragionamento–uso di strumenti–osservazione. L'intuizione risolutiva: laddove l'informazione necessaria alla decisione non è presente nell'input (perché risiede nel futuro del sistema), un agente può *agire per procurarsela* — nel caso specifico, attendere e ri-osservare. Questa svolta realizza compiutamente il titolo del progetto e ricolloca il modello linguistico nel ruolo per cui è adatto: orchestrare un protocollo investigativo, non calcolare decisioni numeriche.

### 3.5 Quinto pivot: dal sintomo alla causa

Lo studio di robustezza dell'agente (§ 5.5) ha rivelato un confine temporale netto nella sua capacità di discriminazione. L'ultima revisione ha introdotto un **sensore della causa** (interrogazione della capacità corrente del collo di bottiglia), dimostrando che il confine è abbattibile quando i modi di guasto differiscono per causa osservabile, e caratterizzando onestamente il residuo irriducibile (§5.6).

**Tabella 3.1 — Cronologia dei pivot.**

| Pivot | Da → A | Evidenza scatenante |
|-------|--------|---------------------|
| P1 | LLM decide e spiega → regola decide, LLM spiega | 3B incapace di confronto numerico (0,997 «basso») |
| P2 | Ablation quantitativa → evidenza qualitativa | Parroting del modello 0,5B azzera la metrica |
| P3 | Escalation automatica a stato 4 → *first, do no harm* | Danno sul degrado transitorio (PDR −0,175, drop +189) |
| P4 | Decisore one-shot → agente investigativo | Floor di osservabilità: informazione nel futuro |
| P5 | Osservazione del sintomo → sensore della causa | Confine temporale ~60–80 s nello studio di robustezza |

---

## 4. Analisi dei Fallimenti e Problem Solving

Questa sezione analizza criticamente i fallimenti riscontrati, distinguendone le cause profonde. Si tratta della parte metodologicamente più densa del lavoro: i fallimenti non sono stati incidenti di percorso, bensì gli esperimenti che hanno prodotto la conoscenza più solida.

### 4.1 Il floor di capacità: l'aritmetica dei modelli piccoli

**Fenomeno.** Il modello da 3B, ricevuti i valori numerici grezzi e la regola di soglia, ha prodotto giudizi di direzione sistematicamente errati (PDR 0,997 dichiarato «basso»; drop 0,000 dichiarato «alto»).

**Analisi delle cause.** Un modello linguistico autoregressivo rappresenta i numeri come sequenze di token, prive di semantica di grandezza: il confronto numerico non è un'operazione primitiva ma una capacità *emergente* con la scala, documentatamente assente al di sotto di una certa dimensione. La regola esplicita nel prompt non può supplire: applicarla richiede esattamente il confronto che il modello non sa eseguire. Si è inoltre osservato che l'arricchimento del prompt con indicatori qualitativi per-metrica (flag «CRITICO») ha *peggiorato* il comportamento, inducendo override spuri: i modelli piccoli non ragionano sui contrassegni salienti, li imitano.

**Lezione.** Per le decisioni a soglia, il decisore corretto è una regola deterministica: esatta, riproducibile, verificabile, a costo nullo. Tre iterazioni di prompt engineering non hanno spostato il problema, confermandone la natura di limite di capacità e non di formulazione.

### 4.2 Il floor di osservabilità: l'informazione nel futuro

**Fenomeno.** Riformulato il compito sulla forza del modello (classificazione di pattern simbolici e scelta fra azioni pre-vagliate), il 3B ha prodotto sul collasso di capacità una diagnosi corretta con giustificazione fluente. Il test di discriminazione ha però rivelato che il modello produceva la **stessa identica risposta** anche sul degrado transitorio — dove essa è dannosa (PDR da 0,857 a 0,695). Il modello non ragionava: ripeteva.

**Analisi delle cause.** All'istante della decisione, collasso permanente e degrado transitorio presentano metriche di finestra pressoché identiche (stesso PDR, stesso drop, stessa traiettoria di stati). L'informazione discriminante — *se il sistema recupererà* — risiede nel futuro e non è presente nell'input. Nessun modello, di qualunque dimensione, può estrarre informazione assente: il collo di bottiglia non è il decisore ma l'**osservabilità**. Il primo apparente successo era un artefatto: il prompt conteneva già gli elementi della conclusione, e il modello li completava.

**Lezione metodologica di portata generale.** La fluenza linguistica maschera l'assenza di ragionamento. Un output ben argomentato e plausibile non è evidenza di analisi: la validazione richiede un *test di discriminazione*, ossia un caso in cui la risposta superficialmente ovvia è errata. In assenza di tale test, un classificatore costante era indistinguibile da un ragionatore.

### 4.3 L'escalation automatica: quando l'intervento danneggia

**Fenomeno.** La regola «se critico e compressione già attiva → forza lo stato 4» è fallita su entrambi i fronti: nel collasso non scattava mai (bloccata dal PDR floor del guardrail, che rifiuta override quando il PDR di finestra è già sotto soglia); sul degrado transitorio dello scenario canonico forzava scarti attivi devastanti (PDR 0,865 → 0,690; +189 pacchetti scartati).

**Analisi delle cause.** Il confine fra oscillazione-da-stabilizzare e collasso-che-richiede-lo-stato-4 non è separabile con le metriche di finestra disponibili: i due regimi presentano valori quasi identici di PDR e drop. Ogni soglia statica che tenti la separazione commette errori sistematici in una delle due direzioni. Si osservi inoltre la tensione progettuale interna: il PDR floor, concepito come protezione contro override speculativi in condizioni critiche, blocca l'escalation proprio quando essa servirebbe — evidenza che i vincoli di sicurezza e le regole di intervento aggressive sono strutturalmente in conflitto.

**Rilievo positivo.** Nello stesso esperimento, i guardrail (revoca su PDR sotto soglia, rifiuto di nuovi override) hanno contenuto il danno della soglia base sul collasso a −0,18 di consegna del traffico di controllo, trasformando un potenziale disastro in una degradazione limitata: la componente di sicurezza ha funzionato esattamente come progettata.

### 4.4 Il confine temporale dell'agente

**Fenomeno.** L'agente investigativo (che attende ~60 s per confermare la natura del regime) discrimina correttamente i transitori di durata ≤ 60 s (5/5, traffico intatto), ma scambia per collasso i transitori di durata ≥ 80 s, intervenendo con danno rilevante (PDR da 0,766 a 0,331).

**Analisi delle cause.** L'agente basato sull'attesa non elimina il floor di osservabilità: lo *sposta* alla scala della propria finestra di osservazione. Un transitorio più lungo dell'attesa non è ancora recuperato quando l'agente delibera, ed è pertanto indistinguibile da un collasso. Il confine misurato (~60–80 s) coincide con la finestra di attesa: il limite è strutturale, non accidentale. La componente irriducibile è di natura quasi-causale: distinguere «recupera all'istante T» da «non recupera mai» osservando il solo sintomo richiede un'osservazione di durata almeno T.

### 4.5 Difetto di orchestrazione dell'SLM agente

**Fenomeno.** Il 3B impiegato come agente eseguiva le azioni corrette (indagine, riconfigurazione) ma non emetteva mai l'azione di conclusione, entrando in cicli di attesa fino all'esaurimento dei passi.

**Analisi.** Si tratta di una debolezza di *controllo di flusso* (terminazione), documentata in letteratura per i modelli compatti in ruoli agentici, distinta sia dalla capacità aritmetica sia dall'osservabilità: il modello sapeva *cosa* fare ma non *quando fermarsi*. La soluzione (regole di terminazione esplicite per stato nel prompt e conclusione forzata a esaurimento passi) appartiene alla categoria dello *scaffolding deterministico del control-flow*.

### 4.6 Incidenti di gestione sperimentale

Due incidenti non algoritmici meritano documentazione per il loro valore metodologico:

1. **Sovrascrittura del checkpoint canonico.** Un riaddestramento lanciato senza specificare il seed ha sovrascritto il checkpoint validato (seed 0, reward 1,0692) con un modello addestrato a seed differente e sensibilmente peggiore (92,95%→91,11% PDR; 224→679 ms; 1,48×→1,13×). L'analisi ha rivelato una **elevata sensibilità al seme casuale** della politica appresa a parità di iperparametri — un dato sperimentale di per sé rilevante. Il ripristino è avvenuto dal versionamento; le contromisure adottate: seed predefinito allineato al canonico (un riaddestramento nudo ora *riproduce* il checkpoint, verificato byte-identico), flag `--tag` per isolare i run sperimentali, procedura documentata di verifica di integrità.
2. **Difetto latente del simulatore.** Durante la costruzione dello scenario di picco di domanda si è scoperto che l'evento di riavvio di un flusso non ripristina il flag di attività: un flusso fermato non riparte. Il difetto rendeva lo scenario pulsato un impulso singolo anziché un'onda quadra. Il difetto è stato documentato, aggirato nello scenario interessato e segnalato per correzione separata, con nota di ricontrollo dei risultati potenzialmente influenzati.

**Tabella 4.1 — Sintesi dei fallimenti, cause profonde e lezioni.**

| Fallimento | Sintomo | Causa profonda | Lezione |
|------------|---------|----------------|---------|
| Giudizi numerici errati (3B) | Output sbagliato in modo palese | Floor di capacità: numeri come token, confronto non emergente | La decisione a soglia spetta a una regola |
| Prompt arricchito → override spuri | Peggioramento con più contesto | Imitazione dei contrassegni salienti | Non sovra-ingegnerizzare i prompt dei modelli piccoli |
| Classificatore costante (escalation simbolica) | Output coerente ma identico ovunque | Floor di osservabilità: informazione nel futuro | Test di discriminazione obbligatorio; la fluenza inganna |
| Regola escalation stato 4 | Danno sul transitorio, inerzia sul collasso | Regimi non separabili con metriche statiche | *First, do no harm*; conflitto vincoli/aggressività |
| Confine temporale agente (~60–80 s) | Danno sui transitori lunghi | Floor spostato alla scala d'osservazione | Caratterizzare il confine con un numero |
| Mancata terminazione (3B agente) | Cicli di attesa senza conclusione | Debolezza di control-flow degli SLM | Scaffolding deterministico del flusso |
| Sovrascrittura checkpoint | KPI degradati inspiegabilmente | Sensibilità al seed + gestione artefatti | Canonico versionato, seed di default allineato |
| Flusso non riavviabile | Scenario pulsato degenere | Difetto latente del gestore eventi | Sanity check ad hoc per ogni scenario nuovo |

> **Suggerimento grafico n. 3 — Anatomia di un intervento dannoso.** Grafico a linee temporali sullo scenario di degrado transitorio: asse x = tempo di simulazione (0–200 s), asse y = PDR di finestra; tre serie: MAPPO solo, soglia con escalation errata (crollo dopo l'intervento a t≈60), agente con attesa corretta (sovrapposto a MAPPO). Dati ricavabili dai log JSON di `run_m3_escalation.py` e `run_agent.py`. Evidenziare con banda verticale l'intervallo del degrado e con marcatore l'istante dell'intervento errato.

---

## 5. Soluzioni Adottate e Risultati

### 5.1 Separazione decisione/spiegazione con blindatura

La soluzione al floor di capacità è architetturale: il verdetto di controllo è calcolato da una regola deterministica; il modello linguistico riceve il verdetto già formato, con istruzione esplicita di spiegarlo senza ricalcolarlo. La separazione è garantita da test di regressione dedicati: anche un backend malevolo che suggerisca sistematicamente override aggressivi non può influenzare né l'azione né lo stato applicato. Il risultato pratico: azioni sempre corrette e spiegazioni in linguaggio naturale ancorate ai fatti, con residui di qualità prosastica funzione della dimensione del modello (documentati nell'ablation).

### 5.2 Il ciclo agentico: risolvere l'osservabilità agendo

La soluzione al floor di osservabilità è paradigmatica: l'agente dispone di strumenti — interrogazione diagnostica, **attesa e ri-osservazione**, riconfigurazione protettiva vincolata, conclusione con diagnosi — e ragiona in un ciclo multi-passo. Lo strumento di attesa trasforma la decisione sotto-determinata in un processo di raccolta dell'informazione mancante: anziché indovinare il futuro, l'agente lo osserva. I risultati di discriminazione (medesimo SLM da 3B che falliva come decisore one-shot):

**Tabella 5.1 — Discriminazione dell'agente (3 semi per scenario, backend Qwen2.5-3B).**

| Scenario | Comportamento dell'agente | Metrica chiave | Correttezza |
|----------|---------------------------|----------------|-------------|
| Collasso permanente (link 10→2) | attende → ancora critico → riconfigura → conclude «permanente» | consegna controllo 0,968 | 3/3 |
| Degrado transitorio (scenario 3) | attende → recuperato → conclude «transitorio», nessun intervento | PDR 0,936 (invariato) | 3/3 |

Si sottolinea, per onestà espositiva, che l'SLM *esegue* un protocollo strutturato dal prompt, non lo *inventa*: il merito misurabile del modello risiede nella scelta di indagare anziché deliberare, nella mappatura corretta osservazione→strumento e nella terminazione appropriata (dopo scaffolding). L'affermazione difendibile è che *un SLM guida in modo affidabile un ciclo agentico su un compito stretto e ben definito* — coerente con la tesi di Belcak et al.

### 5.3 Prestazioni della politica appresa (Fase 3)

**Tabella 5.2 — Confronto Fase 2 (regole) vs Fase 3 (MAPPO), media su 5 semi, 6 scenari.**

| KPI | Fase 2 | Fase 3 | Variazione |
|-----|--------|--------|------------|
| PDR globale | 90,12% | 92,95% | +2,83 punti |
| Latenza media | 771,2 ms | 224,1 ms | −71% |
| Fairness (Jain) | 0,935 | 0,936 | ≈ |
| Compressione | 1,15× | 1,48× | +29% |
| Transizioni di stato | 11 | 9 | −18% |

La Fase 3 prevale su 5/5 KPI. La validazione su emulatore hardware conferma i valori (93,0% PDR; 264 ms; 1,49× su byte reali), con scarto simulazione-reale trascurabile. Un run di verifica a 4000 episodi (8× il budget standard) ha prodotto KPI identici alla terza cifra decimale, attestando la convergenza della politica già a ~250–500 episodi.

> **Suggerimento grafico n. 4 — Confronto per scenario.** Istogramma a barre appaiate: asse x = i sei scenari canonici (single bottleneck, flash crowd, degrado di banda, guasto/ripristino, sovraccarico persistente, traffico misto); asse y = PDR; due barre per scenario (Fase 2, Fase 3), con barre di errore da deviazione standard sui 5 semi. Ripetere in un secondo pannello per la latenza. Dati: output di `compare_phase2_phase3.py`.

### 5.4 Non-interferenza e costo del livello supervisorio

Tre misure verificano le affermazioni architetturali:

**Tabella 5.3 — Misure di non-interferenza e prestazioni del livello agentico.**

| Esperimento | Misura | Esito |
|-------------|--------|-------|
| Non-interferenza | KPI del traffico con agente presente ma non intervenuto vs MAPPO solo | Differenza = 0,00 su tutti i semi: agente trasparente |
| Tempo del percorso veloce | Wall-clock per decisione MAPPO | ≈0,35 ms; l'inferenza SLM (secondi) avviene una volta per finestra sul percorso lento |
| Prestazioni nel collasso | MAPPO solo vs +agente | Consegna controllo +0,051; latenza −1000 ms; PDR −0,021 (sacrificio deliberato del traffico a bassa priorità) |

Il dato di latenza merita nota: l'intervento protettivo *riduce* la latenza media di circa un secondo, poiché lo scarto selettivo a monte decongestiona la coda. Il costo (−2,1 punti di PDR globale) è il prezzo esplicito della protezione delle priorità alte: un compromesso dichiarato, non un pasto gratuito.

### 5.5 Robustezza e caratterizzazione del confine

**Tabella 5.4 — Robustezza alla severità del collasso (5 semi).**

| Capacità residua del collo | Diagnosi «permanente» | Guadagno di consegna controllo |
|----------------------------|------------------------|-------------------------------|
| 2 pkt/s | 5/5 | +0,061 |
| 3 pkt/s | 5/5 | +0,063 |
| 4 pkt/s | 5/5 | +0,043 |
| 5 pkt/s | 5/5 | +0,025 |

**Tabella 5.5 — Confine sulla durata del transitorio (5 semi).**

| Durata del degrado | Diagnosi | Intervento | Effetto sul PDR |
|--------------------|----------|------------|-----------------|
| ≤ 60 s | transitorio (corretta) | nessuno | invariato |
| ≥ 80 s | «collasso» (errata) | sì | 0,766 → 0,331 |

Il confine (~60–80 s) coincide con la finestra di attesa: sotto di essa l'agente è robusto, sopra il floor di osservabilità riemerge. Il compromesso è governabile (attese più lunghe spostano il confine al costo di reattività sui collassi reali) ed è riportato con il suo numero anziché occultato.

> **Suggerimento grafico n. 5 — Il confine del transitorio.** Grafico a linee: asse x = durata del degrado transitorio (20–120 s); asse y = PDR medio di episodio; due serie (MAPPO solo; MAPPO + agente). Le serie coincidono fino a 60 s e divergono bruscamente da 80 s. Banda verticale ombreggiata sull'intervallo 60–80 s etichettata «finestra di attesa dell'agente». Dati: output di `run_agent_robustness.py`, studio 2.

### 5.6 Abbattimento del confine mediante osservazione della causa

L'ultima soluzione distingue i *modi* di guasto per causa osservabile: un collasso è una perdita di **capacità** (collo di bottiglia a capacità ridotta), un transitorio realistico è spesso un eccesso di **domanda** (capacità nominale, carico anomalo). Il sensore `query_link_capacity` legge la causa all'istante del sintomo:

**Tabella 5.6 — Agente ad attesa vs agente a causa sul picco di domanda (5 semi).**

| Durata del picco | Agente ad attesa | Agente a causa |
|------------------|------------------|----------------|
| 40 s | transitorio (corretto) | capacità nominale → domanda (corretto) |
| 80 s | «collasso» (errato) | capacità nominale → domanda (corretto) |
| 120 s | «collasso» (errato) | capacità nominale → domanda (corretto) |

Per la coppia di guasti a causa distinta, il confine temporale è **abbattuto**: la diagnosi è istantanea e indipendente dalla durata. La controprova sul collasso reale conferma la correttezza (capacità letta ridotta → intervento; consegna controllo da 0,904 a 0,985). Rimane il **residuo irriducibile**: un calo di capacità *transitorio* presenta la stessa causa del collasso permanente; entro lo stesso modo di guasto, la permanenza è distinguibile solo col tempo. La soluzione completa combina i due strumenti: sensore della causa per il modo, attesa per la permanenza entro il modo.

### 5.7 Quadro conclusivo

Il percorso sperimentale consente di formulare la conclusione generale del lavoro con precisione insolita per l'ambito:

1. **La decisione di controllo per-tick non è un compito linguistico.** Essa è numerica (dominio della regola deterministica: esatta, gratuita, verificabile) oppure limitata dall'osservabilità (dominio della feature o del sensore giusto: nessun decisore, per quanto capace, supplisce all'informazione assente).
2. **I ruoli genuini del modello linguistico** nel sistema sono la *spiegazione* (dimostrata con un modello da 3B locale) e l'*orchestrazione agentica* di un protocollo investigativo (indagare, diagnosticare, agire tramite strumenti vagliati). In entrambi, la forza sfruttata è il linguaggio; in nessuno il modello calcola o decide alla cieca.
3. **Il ciclo agentico è il contributo risolutivo:** trasforma un problema di decisione sotto-determinata in un processo di acquisizione dell'informazione, risolvendo il floor di osservabilità che nessuna variante one-shot — a regole o a modello — poteva superare.
4. **La sicurezza è il componente critico, non accessorio:** guardrail, reversibilità e revoca hanno materialmente contenuto ogni errore di progetto durante lo sviluppo, permettendo la sperimentazione aggressiva senza compromettere il sistema validato.

Il titolo del progetto risulta così dimostrato nella sua accezione precisa: la rete agentica non è una rete in cui il modello linguistico comanda il piano di controllo, bensì una rete in cui esso opera come **operatore autonomo sul piano di gestione** — osserva, indaga, spiega e interviene con parsimonia — mentre il controllo in tempo reale resta affidato a meccanismi deterministici e appresi, validati e riproducibili.

---

## Appendice A — Riferimenti essenziali

1. Belcak, P., et al. (2024). *Small Language Models are the Future of Agentic AI.* NVIDIA Research, arXiv:2506.02153.
2. Yu, C., et al. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.* NeurIPS 35, arXiv:2103.01955.
3. Kahneman, D. (2011). *Thinking, Fast and Slow.* (Analogia System 1 / System 2.)
4. Abate, M., Sacco, A., Fiore, M., Esposito, F. *eFRAC: Elastic Flow-Rate Adaptive Compression for Network Congestion Management.*
5. Anon. (2026). *CoDi-NetLLM: Adapting Continuous Distributional Outputs for LLM-based Networking.* (Decoupling decisione/spiegazione; quantificazione dell'incertezza; sufficienza dei backbone compatti.)

## Appendice B — Riepilogo dei grafici suggeriti

| N. | Grafico | Dati | Sezione |
|----|---------|------|---------|
| 1 | Architettura a due livelli / timeline dei due orologi | `agent_timeline.png` o diagramma a blocchi | §1.3 |
| 2 | Curva di addestramento MAPPO | log di `train_mappo.py` (reward/episodio; eval ogni 50) | §2.6 |
| 3 | Anatomia di un intervento dannoso | log JSON `run_m3_escalation.py`, `run_agent.py` (PDR di finestra nel tempo, 3 serie) | §4 |
| 4 | Confronto per scenario F2 vs F3 (PDR e latenza, barre + errore) | output `compare_phase2_phase3.py` | §5.3 |
| 5 | Confine del transitorio (PDR vs durata, 2 serie + banda 60–80 s) | output `run_agent_robustness.py` studio 2 | §5.5 |

## Appendice C — Riproducibilità

Tutti i risultati citati sono riproducibili dal repository (133 test automatici; nessuna modifica alle fasi precedenti, sviluppo interamente additivo):

```bash
python3.12 examples/compare_phase2_phase3.py --ckpt checkpoints/mappo_best_stab.json
python3.12 examples/run_m3_ood.py --ood capacity_collapse --seeds 5
python3.12 examples/run_agent.py --backend ollama --model qwen2.5:3b --verbose
python3.12 examples/run_agent_perf.py
python3.12 examples/run_agent_robustness.py --seeds 5
python3.12 examples/run_agent_cause.py --seeds 5
```

Il checkpoint canonico della Fase 3 (seed 0) è versionato e rigenerabile in modo byte-identico (`train_mappo.py --seed 0 --stability-penalty 0.1`).
