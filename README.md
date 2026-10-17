# LEDGAR Contract Review — NLP core

Cœur NLP d'un assistant de revue de contrats :

1. **Classification** des clauses en 100 types (fine-tuning de BERT sur LEDGAR) ;
2. **Recherche** de clauses similaires dans une banque de clauses (sentence-transformers + FAISS) ;
3. **Audit** de fiabilité et de biais : où le système fonctionne, où il échoue, et ce pour quoi il ne doit pas être utilisé.

Données : [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) (benchmark LexGLUE), clauses de contrats déposés à la SEC, en anglais.

## Installation

Python ≥ 3.10. Référence : Python 3.13 (version de Colab utilisée pour l'entraînement).

```bash
git clone <url-du-repo>
cd ledgar-contract-review
python -m venv .venv && source .venv/bin/activate
pip install torch            # inutile sur Colab, déjà installé
pip install -r requirements.txt
pip install -e .
nbstripout --install         # une fois par clone : nettoie les notebooks à chaque commit
```

Sur Colab :

```python
!git clone <url-du-repo>
%cd ledgar-contract-review
!pip install -q -r requirements.txt && pip install -q -e .
```

## Structure

```
configs/        hyperparamètres (YAML)
notebooks/      01_eda → 05_audit : rapports qui appellent src/
src/ledgar/     tout le code (data, baseline, train, evaluate, retrieval, audit)
reports/        figures et model card
```

## Reproduire les résultats

_À compléter au fil du projet._

## Équipe

_À compléter._
