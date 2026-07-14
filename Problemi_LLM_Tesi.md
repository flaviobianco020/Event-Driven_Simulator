# Problemi riscontrati con i modelli linguistici — catalogo per la tesi

**Progetto:** Towards Agentic Networks — Fase 4 (supervisore/agente LLM)
**Scopo del documento:** raccogliere in forma sistematica tutti i limiti, i fallimenti e
le criticità osservati nell'impiego di modelli linguistici (Small Language Model locali,
principalmente Qwen2.5-3B via Ollama) all'interno di un anello di controllo di congestione.
Ogni voce riporta: descrizione, causa profonda, evidenza sperimentale, tentativi di
mitigazione e lezione ai fini della tesi. Il documento è pensato come base per il capitolo
sui limiti e per la sezione «lessons learned».

---

## Quadro sintetico

| # | Problema | Categoria | Fixabile con modello più grande? | Esito |
|---|----------|-----------|:---:|-------|
| 1 | Confronto numerico inaffidabile | Capacità (aritmetica) | sì | Decisione affidata a regola deterministica |
| 2 | Ragionamento causale errato (dal vivo) | Capacità (ragionamento) | in parte | Verdetto iniettato, non richiesto |
| 3 | Output costante / nessuna discriminazione | Osservabilità | no | Ciclo agentico (acquisire informazione) |
| 4 | La fluenza maschera l'assenza di ragionamento | Metodologico | no | Test di discriminazione obbligatorio |
| 5 | Iper-ingegneria del prompt controproducente | Prompt | no | Prompt minimale + regola giusta |
| 6 | Il prompt suggerisce la risposta | Prompt | no | Separare percezione da giudizio |
| 7 | Parroting del verdetto iniettato | Capacità (scala) | sì | Metrica di valore interpretativo |
| 8 | Mancata terminazione (agente) | Orchestrazione | in parte | Scaffolding del control-flow |
| 9 | Vincoli di risorse (7B su 8 GB) | Operativo | — | Timeout esteso; 3B sufficiente |
| 10 | Latenza d'inferenza reale (3,6 s) | Operativo | peggiora | Fuori dal loop veloce (asincrono) |
| 11 | Tag del modello inesistente | Operativo | — | Correzione del nome (`qwen2.5:3b`) |
| 12 | Non-determinismo intrinseco | Architetturale | no | Constrained decoding (formato, non correttezza) |
| 13 | Nessun impatto sui KPI | Architetturale | no | LLM = spiegazione/interfaccia, non controllo |

---

## 1. Floor di capacità: confronto numerico inaffidabile

**Descrizione.** Fornendo al modello i valori grezzi delle metriche e una regola di soglia
esplicita, il modello da 3 miliardi di parametri ha prodotto giudizi di direzione
sistematicamente errati.

**Evidenza.** PDR pari a 0,997 qualificato come «basso»; tasso di scarto pari a 0,000
qualificato come «alto». La regola «se PDR < 0,85 allora critico», inclusa nel prompt, è
stata disattesa.

**Causa profonda.** Un modello linguistico autoregressivo rappresenta i numeri come
sequenze di token, prive di semantica di grandezza. Il confronto numerico affidabile è una
capacità *emergente* con la scala, assente al di sotto di una certa dimensione. Applicare la
regola richiede esattamente il confronto che il modello non sa eseguire.

**Tentativi di mitigazione.** Tre riformulazioni successive del prompt: (i) numeri nudi;
(ii) flag qualitativi per-metrica; (iii) prompt minimale con regola esplicita. Tutte fallite.

**Risoluzione.** La decisione a soglia è stata affidata a una regola deterministica
(`assess`), riservando al modello la sola spiegazione. Blindata da test di regressione.

**Lezione.** Le decisioni a soglia appartengono a una regola: esatta, riproducibile,
verificabile, a costo nullo. Non è un problema di prompt ma di capacità.

---

## 2. Floor di capacità: ragionamento causale errato (osservato in deployment)

**Descrizione.** Nel test di deployment con inferenza reale, richiesto di *determinare* se la
condizione critica fosse un guasto strutturale (perdita di capacità) o un eccesso di domanda,
il modello ha invertito la diagnosi.

**Evidenza.** Con «capacità del collo di bottiglia = 2 (nominale 10)» esplicitamente nel
prompt — quindi capacità palesemente ridotta — il modello ha concluso «eccesso di domanda che
ha superato le capacità del sistema». La causa reale era una perdita di capacità.

**Causa profonda.** La medesima incapacità di ragionamento numerico/causale del punto 1: pur
avendo i valori davanti, il modello non li ha comparati correttamente. Inoltre, gli era stato
chiesto di *determinare* la causa (ragionamento), non di *spiegare* una causa già stabilita.

**Risoluzione (principio).** Applicare la separazione della Fase M1 anche in deployment:
iniettare il verdetto del sensore-causa deterministico e chiedere al modello di spiegarlo,
non di ricavarlo.

**Lezione.** La conferma più netta e «dal vivo» della tesi: la decisione causale spetta al
sensore deterministico; al modello va lasciata la sola narrazione di un verdicto già formato.

---

## 3. Floor di osservabilità: output costante, nessuna discriminazione

**Descrizione.** Riformulato il compito sulla forza del modello (classificazione di pattern
simbolici, non aritmetica), il modello produceva una diagnosi corretta e fluente sul collasso,
ma la *stessa identica* risposta anche sul degrado transitorio, dove è dannosa.

**Evidenza.** Risposta parola-per-parola identica nei due regimi; sul transitorio l'intervento
conseguente degradava il PDR da 0,857 a 0,695. Il modello non ragionava: ripeteva.

**Causa profonda.** All'istante della decisione, collasso e transitorio presentano metriche
identiche; l'informazione discriminante (se il sistema recupererà) è nel futuro, non
nell'input. Nessun modello, di qualunque dimensione, estrae informazione assente. Il collo di
bottiglia è l'osservabilità, non il modello.

**Risoluzione.** Il ciclo agentico: l'agente *agisce per procurarsi l'informazione* (attende e
ri-osserva). Non un modello più capace, ma un processo di acquisizione dell'informazione.

**Lezione.** Un limite indipendente dal modello. Distingue nettamente «il modello non sa» da
«l'informazione non c'è».

---

## 4. La fluenza maschera l'assenza di ragionamento

**Descrizione.** L'output del punto 3 era coerente, plausibile e ben argomentato — eppure privo
di ragionamento reale (un classificatore costante travestito da ragionatore).

**Causa profonda.** I modelli linguistici producono prosa fluente indipendentemente dalla
correttezza del contenuto. Un output ben scritto non è evidenza di analisi.

**Risoluzione.** Il *test di discriminazione*: validare su un caso in cui la risposta
superficialmente ovvia è errata (il transitorio, dove ripetere «collasso» fa danno). Solo lì
una costante si tradisce.

**Lezione (metodologica, di portata generale).** Mai fidarsi della fluenza. La validazione dei
sistemi LLM richiede casi avversari costruiti apposta, non esempi favorevoli.

---

## 5. Iper-ingegneria del prompt controproducente

**Descrizione.** Nel tentativo di aiutare il modello, l'arricchimento del prompt con indicatori
qualitativi per-metrica (flag «CRITICO») ha *peggiorato* il comportamento.

**Evidenza.** Il modello seguiva ciecamente ogni flag «CRITICO», producendo override spuri —
inclusi casi in cui la saturazione del collo di bottiglia era normale.

**Causa profonda.** I modelli piccoli non ragionano sui contrassegni salienti: li imitano. Più
contesto suggestivo introduce più modi di sbagliare.

**Lezione.** Per i modelli compatti, prompt minimale e regola corretta a monte; non
sovraccaricare il contesto di segnali «guida».

---

## 6. Il prompt suggerisce la risposta

**Descrizione.** Il prompt simbolico costruito per l'escalation conteneva già gli elementi
della conclusione («compressione già massima + critico persistente»), che il modello si
limitava a completare come «collasso».

**Causa profonda.** Nel tentativo di guidare il modello, il prompt aveva codificato di fatto la
risposta; l'apparente ragionamento era completamento del pattern fornito.

**Lezione.** Separare la *percezione* (fornita) dal *giudizio* (richiesto). Se il prompt
contiene la conclusione, non si sta misurando il ragionamento del modello.

---

## 7. Parroting del verdetto iniettato

**Descrizione.** Nell'ablation sulla dimensione, il modello più piccolo (0,5B) otteneva zero
errori su una metrica automatica di correttezza — non perché ragionasse, ma perché *ripeteva
verbatim* la valutazione iniettata nel prompt.

**Causa profonda.** Un modello troppo piccolo per interpretare ricade sull'eco del contesto.
La metrica di correttezza superficiale premiava l'eco anziché il ragionamento.

**Risoluzione.** Ridimensionare l'ablation a evidenza qualitativa e adottare una metrica di
*valore interpretativo aggiunto* (il modello cita la traiettoria? propone un rimedio?).

**Lezione.** Le metriche di correttezza direzionale sono ingannevoli: un modello che ripete il
verdetto le supera senza comprendere.

---

## 8. Mancata terminazione dell'agente (debolezza di orchestrazione)

**Descrizione.** Impiegato come agente con strumenti, il modello eseguiva le azioni corrette
(indagine, riconfigurazione) ma non emetteva mai l'azione di conclusione, entrando in cicli di
attesa fino all'esaurimento dei passi.

**Causa profonda.** Debolezza di *controllo di flusso* (sapere *quando* fermarsi), documentata
per i modelli compatti in ruoli agentici, distinta dalla capacità di scelta del singolo passo.

**Risoluzione.** Scaffolding deterministico: regole di terminazione esplicite per stato nel
prompt e conclusione forzata a esaurimento passi (la diagnosi è inferita dallo stato).

**Lezione.** Un SLM può *eseguire* un protocollo agentico ma non *auto-governarne* il flusso;
il control-flow va imbrigliato deterministicamente.

---

## 9. Vincoli di risorse

**Descrizione.** Il modello da 7B, su una macchina con 8 GB di RAM, non completava l'inferenza
entro il timeout predefinito (30 s), a causa del caricamento dei pesi e dello swapping.

**Evidenza.** Ripetuti «backend errore (timed out)»; nessuna decisione LLM reale prodotta finché
il timeout non è stato esteso e il modello pre-caricato.

**Risoluzione.** Timeout esteso, pre-riscaldamento del modello, e conferma che il 3B è
sufficiente al compito (il 7B non era necessario).

**Lezione.** Il costo di deployment locale di un modello, anche «piccolo», è concreto e va
dimensionato sull'hardware target.

---

## 10. Latenza d'inferenza reale

**Descrizione.** Ogni chiamata reale al modello da 3B ha richiesto circa 3,6 secondi.

**Rilievo.** La latenza è ordini di grandezza superiore alla decisione del percorso veloce
(~0,35 ms). Il problema *non* impatta il traffico solo perché il modello è confinato al
percorso lento e invocato in modo asincrono (verificato: 300 tick veloci a ~156 ms mentre il
modello «pensa» per 3,6 s, senza blocco del loop veloce).

**Lezione.** Un modello linguistico non può stare in un anello di controllo a bassa latenza;
la sua collocazione corretta è un livello lento e asincrono.

---

## 11. Errore nel nome del modello

**Descrizione.** Il tag iniziale `qwen2.5:3b-instruct` non esiste nel registro Ollama; il
comando di download falliva.

**Risoluzione.** Correzione in `qwen2.5:3b`.

**Lezione.** Criticità operativa minore ma tipica dell'integrazione con runtime di modelli
locali; da verificare per la riproducibilità.

---

## 12. Non-determinismo intrinseco

**Descrizione.** Un modello linguistico è un generatore stocastico di testo; anche a
temperatura 0 non offre le garanzie formali richieste da un anello di controllo.

**Rilievo.** Il *constrained decoding* (schema JSON imposto) garantisce la validità
*sintattica* dell'output a qualunque scala, ma non la *correttezza* della decisione.

**Lezione.** «Probabilmente corretto» non è una garanzia di controllo; il constrained decoding
risolve il formato, non l'affidabilità decisionale.

---

## 13. Assenza di impatto sui KPI (verdetto architetturale)

**Descrizione.** Poiché ogni decisione effettivamente affidabile è stata resa deterministica,
la rimozione del modello linguistico dal sistema lascia le prestazioni invariate.

**Evidenza.** L'agente completamente deterministico (sensore-causa + monitoraggio-e-ritiro)
eguaglia o supera MAPPO-solo dove deve (collasso: consegna controllo 0,905 → 0,987;
transitorio corto 0,846 → 0,851), è neutro sul picco di domanda (0,737 = 0,737) e mostra un
costo minimo e limitato sul transitorio lungo (0,695 → 0,686). In deployment con inferenza
reale, il medesimo scenario dà PDR 0,695 — identico: il modello produceva solo spiegazioni.

**Lezione conclusiva.** In un anello di controllo a telemetria numerica strutturata e spazio
d'azione ridotto, il modello linguistico è un livello di *spiegazione e interfaccia*, non di
*controllo*. I *pattern* agentici che apportano valore (indagare prima di decidere, sensore
della causa, azione reversibile con ri-valutazione) sono realizzabili — e più affidabili — in
forma deterministica. Il valore decisionale di un modello linguistico emergerebbe solo con
input *non strutturato* (intento dell'operatore in linguaggio naturale, report testuali,
segnali cross-dominio), scenario non esplorato in questo lavoro e indicato come frontiera.

---

## Sintesi in tre affermazioni difendibili

1. **Capacità.** Un SLM da 3B non esegue in modo affidabile confronti numerici né ragionamento
   causale sui numeri; le decisioni a soglia e di causa spettano a regole o sensori
   deterministici.
2. **Osservabilità.** Alcuni limiti (distinguere transitorio da permanente all'istante della
   decisione) non dipendono dal modello ma dall'informazione disponibile; si superano
   acquisendo informazione (ciclo agentico, sensori di causa), non ingrandendo il modello.
3. **Ruolo.** Rimuovere il modello dal sistema non altera le prestazioni: il suo contributo
   robusto è la spiegazione in linguaggio naturale per l'operatore, non la decisione di
   controllo.
