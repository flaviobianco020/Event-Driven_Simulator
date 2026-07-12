# Perché lo SLM non decide: motivazione della separazione decisione/spiegazione

**Contesto.** Nel supervisore di Fase 4 la decisione di controllo (endorse / override
+ stato target) è calcolata da una regola deterministica (`SupervisorController.assess`),
non dallo Small Language Model. Lo SLM (Qwen2.5-3B via Ollama) fornisce **solo** la
spiegazione in linguaggio naturale. Questo documento motiva quella scelta: perché lo
SLM è stato rimosso dal livello decisionale.

## 1. È sotto il floor aritmetico

La decisione di congestione si riduce a un confronto numerico: *"il PDR è sotto la
soglia?"*. Empiricamente il modello da 3B non sa confrontare due numeri in modo
affidabile: nei test ha definito «basso» un PDR di 0,997 e «alto» un drop di 0,000,
ignorando la soglia esplicitata nel prompt. Se il modello non sa dire quale numero è
più grande, non può prendere la decisione che dipende proprio da quel confronto. Non è
un problema di prompt: tre riformulazioni successive non l'hanno risolto.

## 2. Il controllo richiede determinismo, l'LLM non lo garantisce

Un anello di controllo che tocca traffico reale ha bisogno di decisioni riproducibili e
verificabili. Un LLM, anche a temperatura 0, resta un generatore stocastico di testo:
il *constrained decoding* garantisce che l'output sia un JSON valido, non che la
decisione sia corretta. «Probabilmente giusto» non è una garanzia di controllo.

## 3. L'iper-ingegneria del prompt è controproducente sui modelli piccoli

Aggiungere indicazioni per-metrica per aiutare il 3B ha peggiorato le cose: il modello
seguiva ciecamente ogni flag «CRITICO» → override spuri. I modelli piccoli non ragionano
sui flag, li imitano. Più contesto = più modi di sbagliare.

## 4. La decisione è banale e non richiede un LLM

La scelta effettiva è `if pdr < 0.85 or drop > 0.15`. Usare una rete neurale da ~2 GB
per approssimare (male) un `if` è puro spreco: più lento, non riproducibile, e sbagliato.
La forza dell'LLM è il linguaggio; la decisione richiede aritmetica e determinismo — le
sue due debolezze.

## 5. Persino nel caso in cui servirebbe, il 3B non arriva

L'unico punto in cui un ragionatore potrebbe battere la soglia è il caso ambiguo
(collasso di capacità vs oscillazione: stessi due numeri, situazioni opposte — cfr.
risultati M3). Ma distinguerli richiede un modello sopra il floor aritmetico, non il 3B.
Quindi il 3B fallisce sulle decisioni facili (aritmetica) e non raggiunge quelle
difficili (ragionamento sul pattern). Verificare se un modello più capace (7B / Haiku)
decida bene sul solo caso ambiguo è lavoro futuro.

## Conclusione

La decisione di controllo è affidata alla regola deterministica; allo SLM resta la sola
spiegazione in linguaggio naturale — l'unico compito allineato alle sue capacità. Questa
non è una sconfitta dell'approccio SLM ma un suo posizionamento corretto: *un SLM da 3B
è sufficiente a spiegare un verdetto, non a calcolarlo*. Il codice blinda questa
separazione: né l'azione né lo stato target derivano mai dall'output del modello (vedi
`controller.py::tick`, test `test_llm_output_never_sets_control_action`).
