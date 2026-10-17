# LEDGAR Contract Review — NLP core

Cœur NLP d'un assistant de revue de contrats :

1. **Classification** des clauses en 100 types : baseline TF-IDF, LSTM, puis BERT fine-tuné (complet et LoRA) ;
2. **Recherche** de clauses similaires dans une banque de clauses (sentence-transformers) ;
3. **Audit** de fiabilité et de biais : où le système fonctionne, où il échoue, et ce pour quoi il ne doit pas être utilisé.

Données : [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) (benchmark LexGLUE), clauses de contrats déposés à la SEC, en anglais. Splits officiels : 60 000 train / 10 000 validation / 10 000 test, 100 classes.

## Installation

Python ≥ 3.10. Référence : Python 3.13 (version de Colab utilisée pour l'entraînement).

```bash
git clone https://github.com/saouss-07/ledgar-contract-review.git
cd ledgar-contract-review
python -m venv .venv && source .venv/bin/activate
pip install torch            # inutile sur Colab, déjà installé
pip install -r requirements.txt
pip install -e .
nbstripout --install         # une fois par clone : nettoie les notebooks à chaque commit
```

Sur Colab :

```python
!git clone https://github.com/saouss-07/ledgar-contract-review.git
%cd ledgar-contract-review
!pip install -q -r requirements.txt && pip install -q -e .
```

Mode rapide pour vérifier que tout tourne (sous-échantillon des données) : `FAST_RUN=1`.

## Structure

```
configs/        hyperparamètres (YAML)
notebooks/      exploration et rapports : ils appellent src/, sans logique propre
src/ledgar/     data, metadata, metrics, baseline, lstm, finetune, retrieval, audit
results/        tableaux (CSV) et figures produits par les expériences
reports/        model card
```

## Jalons (tags Git)

| Tag | Contenu |
| --- | --- |
| `m1` | Données chargées, distribution des classes, baseline TF-IDF + régression logistique (macro-F1) |
| `m2` | Index de recherche par sentence-embeddings, vérifié sur des requêtes |
| `m3` | Classifieur LSTM et analyse d'erreurs (F1 par classe, classes les plus confondues) |
| `m4` | BERT fine-tuné, runs LoRA et tableau de compromis, test apparié |
| `v1.0` | Version rendue : audit, model card, résultats finaux |

## Hypothèse (écrite avant les expériences du jalon m4)

_À compléter avant de lancer le fine-tuning._

## Reproduire les résultats

_Une commande par expérience, à compléter au fil du projet._

## Résultats

_À compléter._

## Équipe

_À compléter._
