*Useful files:* 

- `train-16batch.py` è un file terribile ma funzionale. Per sbatcharlo, usare `train16.sh`. Questo è il file su cui ho fatto i test preliminari, gli argomenti di training vengono passati da dizionario (all'inizio del file). Salva checkpoints ogni `x` steps, cosa indicata tra questi parametri, e continua a trainare dal checkpoint indicato come path sempre lì. Nota rilevante: per pigrizia o skill issue il numero di steps già fatti nel checkpoints va anche passato, non lo estrae da solo. 

- `training.py` dovrebbe essere la versione più carina di `train-16batch.py`, quindi commentata e razionalizzata (mah). I parametri vanno fissati in `training_config.json` e il file sbatch correlato è `aspecific_training.sh`. Una cosa che ora sto controllando è che mi sembrava di avere forzato il salvataggio del checkpoint alla fine di ogni epoca. Ciò che probabilmente invece sto facendo è salvare OGNI STEP (cannato un'indentazione?), quindi grazie al cazz che è lento. 

*TO-DO:* 

- salvare ogni `x` steps (e.g. 500) e alla fine di ogni epoca così da poter fare i plottini bellini

- magari usare una GPU bene

- magari usare bene anche più GPU

- trainare il modello su almeno due misture di training data (formule più semplici - formule più difficili); valutare l'impatto di questo training sugli stessi files di validation e di test. 


*DATI:*

- `dummy_training.csv`: ho travasato le prime 1.000 formule in questo file così se si vuole fare un test rapido si può fare; 
- `dummy_validation.csv`: idem ma con le prime 200 del validation set.

*Struttura dei files:*
Il tokenizer, il modello e le configurazioni che lo inizializzano estendono classi di HuggingFace (anche se non uso necessariamente cose particolari di queste: PreTrainedTokenizer, PreTrainedModel, PretrainedConfig). 
La struttura interna del modello gioca sul fatto che l'embedding della sequenza che il decoder decodifica è già dato (`encoder.py` è il lavoro di Gaia Saveri). Per decodificare questo embedding, i file utili sono: 

- `modeling_stldec.py`, che definisce concretamente il modello causale come piace ad HuggingFace, ma di fatto è un wrapper di ciò che accade in `decoder.py`;

- `decoder.py`, che costruisce il decoder. Qui ci sono due meccanismi di attenzione. Quello di self-attention prende in input la sequenza di tokens generata "up to now" e guarda cosa è rilevante per cosa (matrice quadrata). Quello di cross-attention, che subentra poi, prende l'output di questa self-attention e lo mescola con l'informazione che proviene dall'embedding dell'encoder (matrice rettangolare). Qua c'è un casino di `past_key_values` etc, l'ho grossomodo copiato dal file di esempio di HuggingFace per evitare che il modello si debba calcolare ogni volta le proiezioni dell'input come query, key e value. Secondo me qua potrebbero succedere magie strane lato utilizzo risorse. 

- `utils.py`, che contiene effettivamente il meccanismo di attenzione (insieme ad altre mille robe). Di nuovo, qua potrei stare facendo qualcosa di non ottimale.   
