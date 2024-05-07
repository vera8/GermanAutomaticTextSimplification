## Automatische Textvereinfachung mit transformerbasierten Sprachmodellen

In diesem Repository befinden sich die für das Fine-Tuning der verschiedenen mT5-Varianten verwendeten Skripte und Notebooks.
Darunter versteht sich Folgendes: 
* Allgemein:
  * Jupyter Notebook in Kombination Python-Skripten zum Fine-Tunen und Evaluieren der Modelle: 
    * [finetuning_evaluation_pipeline.ipynb](finetuning_evaluation_pipeline.ipynb)
    * [scripts.py](scripts.py)
  * Funktion zum Zusammenstellen eines deutschen Datensatzes aus verschiednen News-Texten (für das Filtern des Vokabulars, Trainieren des Tokenizers, Span-MLM):
    * [make_german_ds.py](make_german_ds.py)
* Zum Filtern des Vokabulars:
  * Jupyter-Notebook zum Verkleinern des Vokabulars des mT5-Tokenizers:
    * [filter_tokenizer_vocabulary.ipynb](filter_tokenizer_vocabulary.ipynb)
* Für Vocabulary Transfer:
  * Jupyter Notebook zum Training eines SentencePiece Tokenizers: 
    * [train_german_tokenizer.py](train_german_tokenizer.ipynb)
  * Funktion zum Initialisieren der Embeddings des neuen Vokabulars: 
    * [vocabulary_transfer.py](vocabulary_transfer.py)
  * Jupyter Notebook und Python-Skripte zum Preprocessing von Trainingsdaten für Span-MLM:
    * [smlm_data_preprocessing.ipynb](smlm_data_preprocessing.ipynb)
    * [mt5_smlm_scripts.py](mt5_smlm_scripts.py) (abgewandelt von [Hugging Face Beispiel](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py))
  * Jupyter Notebook zum Durchführen von Span-MLM:
    * [mt5_smlm_train.ipynb](mt5_smlm_train.ipynb)
  * Notwendige Python-Libraries sind im [requirements.txt](requirements.txt) zu finden.

Die zum Fine-Tuning verwendeten Datensätze sind auf Anfrage über https://zenodo.org/records/7674560 erhältlich. 

### Die finegetuneten Modelle sind unter folgenden Links zu finden: 
* mT5-Small: https://huggingface.co/vera-8/mT5-small-trimmed_deplain-apa
* mT5-Base: https://huggingface.co/vera-8/mT5-base-trimmed_deplain-apa
* mT5-Large: https://huggingface.co/vera-8/mT5-large-trimmed_deplain-apa
* mT5-XL: https://huggingface.co/vera-8/mT5-xl-trimmed_deplain-apa
* mT5-Small mit VT: https://huggingface.co/vera-8/mT5-small-VT_deplain-apa
* mT5-Small mit VT und Span-MLM: https://huggingface.co/vera-8/mT5-small-VT-span-mlm_deplain-apa

Bei allen Modellen befindet sich ein passender Tokenizer mit im Hub.

### Referenzen
* Finetuning- und Test-Datensatz von Stodden et al. (2023): https://huggingface.co/datasets/DEplain/DEplain-APA-sent (nur auf Anfrage erhältlich)
* EASSE von Alva-Manchego et al. (2019): https://github.com/feralvam/easse (es befindet sich eine Kopie von dem originalen Repository in dem Ordner [easse](easse))
* Code zur Datenvorverarbeitung für Span-MLM: https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py (Teile davon wurden leicht abgewandelt übernommen und sind entsprechend gekennzeichnet)
* Vocabulary Transfer nach Mosin et al. (2021): https://doi.org/10.48550/arXiv.2112.14569
* mT5-Basismodelle von Xue et al. (2020), bereitgestell über Hugging Face: 
  * https://huggingface.co/google/mt5-small
  * https://huggingface.co/google/mt5-base
  * https://huggingface.co/google/mt5-large
  * https://huggingface.co/google/mt5-xl