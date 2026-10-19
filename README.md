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

Une commande par expérience ; `FAST_RUN=1` devant une commande la lance sur un sous-échantillon pour vérifier qu'elle tourne.

| Expérience | Commande | Sorties |
| --- | --- | --- |
| Tests unitaires | `python -m pytest tests` | — |
| Exploration des données | `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb` | `results/figures/`, `results/tables/class_counts.csv` |
| Couverture des métadonnées | `python -m ledgar.metadata` | `results/tables/metadata_coverage.csv` |
| Baseline TF-IDF (m1) | `python -m ledgar.baseline` (≈ 25 min sur CPU) | `results/tables/baseline_*.csv`, `results/predictions/baseline_*.csv` |
| Recherche de clauses (m2) | `python -m ledgar.retrieval` (≈ 5 min sur T4, ≈ 1 h 30 sur CPU) | `results/tables/retrieval_*.csv` |
| LSTM + GloVe (m3) | `python -m ledgar.lstm` (GPU conseillé) | `results/tables/lstm_*.csv`, `results/predictions/lstm_*.csv` |
| Volet B sur Colab | `notebooks/B_retrieval_lstm.ipynb` | les deux lignes ci-dessus + analyse d'erreurs |

## Résultats

Métrique principale : **macro-F1** (100 classes très déséquilibrées). Tous les choix sont faits sur la validation ; le test n'est évalué qu'avec le modèle retenu.

| Modèle | Réglage (choisi sur la validation) | Macro-F1 validation | Macro-F1 test | Accuracy test | Jalon |
| --- | --- | --- | --- | --- | --- |
| TF-IDF (mots + bigrammes) + régression logistique | C = 30, `class_weight="balanced"` | 0,819 | 0,828 | 0,873 | m1 |

Sans les 2,8 % de clauses du test déjà présentes dans le train, la macro-F1 de la baseline est de 0,825 : les doublons ne gonflent pas le score de manière notable. La grille complète est dans `results/tables/baseline_grid.csv` ; avec ou sans pondération des classes, l'écart est de 0,001 au plus. Les confusions les plus fréquentes (`results/tables/baseline_top_confusions.csv`) sont les paires quasi synonymes repérées pendant l'exploration (*Governing Laws* / *Applicable Laws*, *Tax Withholdings* / *Withholdings*…).

## Règles de travail

- On ne modifie jamais `main` directement : une branche par tâche, puis une Pull Request (modèle pré-rempli).
- Phase 0 (setup, jusqu'au tag `m1`) : PR fusionnées par leur auteur. Ensuite : relecture obligatoire par un autre membre, puis squash merge.
- Jamais de données, de modèles ni de sorties de notebook dans Git. Les prédictions (`results/predictions/`) ne contiennent que `idx, label, pred, confidence`.

## Équipe

_À compléter._
