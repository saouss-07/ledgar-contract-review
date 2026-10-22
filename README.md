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

Modèle : `nlpaueb/legal-bert-base-uncased`, max_length 256, lr 2e-5, 3 epochs, batch 16, warmup 10 %, weight decay 0,01.
LoRA : W_Q et W_V, α = 2r, lr 5e-4, r ∈ {4, 8, 16}. Tous les choix sont faits sur la validation.

- **H1** : Legal-BERT fine-tuné complètement dépasse la baseline TF-IDF en macro-F1 sur le test ; l'intervalle à 95 % du bootstrap apparié (1 000 tirages) de la différence exclut 0.
- **H2** : LoRA r = 8 reste à moins de 2 points de macro-F1 du fine-tuning complet en entraînant moins de 2 % des paramètres ; la différence d'accuracy est testée par McNemar.
- **H3** : passer de r = 4 à r = 16 apporte moins de 1 point de macro-F1.

## Reproduire les résultats

Une commande par expérience ; `FAST_RUN=1` devant une commande la lance sur un sous-échantillon pour vérifier qu'elle tourne.

| Expérience | Commande | Sorties |
| --- | --- | --- |
| Tests unitaires | `python -m pytest tests` | — |
| Exploration des données | `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb` | `results/figures/`, `results/tables/class_counts.csv` |
| Couverture des métadonnées | `python -m ledgar.metadata` | `results/tables/metadata_coverage.csv` |
| Baseline TF-IDF (m1) | `python -m ledgar.baseline` (≈ 25 min sur CPU) | `results/tables/baseline_*.csv`, `results/predictions/baseline_*.csv` |
| Fine-tuning Legal-BERT et LoRA (m4) | `notebooks/partie_A_notebook.ipynb` sur Colab, GPU T4 (≈ 42 min par run complet de 3 epochs, ≈ 34 min par run LoRA) | `results/predictions/legalbert_*.csv`, `results/tables/finetune_metrics_*.json` |
| Tableaux du volet A contre la baseline commune (m4) | `python -m ledgar.finetune_report` (quelques secondes sur CPU) | `results/tables/finetune_tradeoff.csv`, `finetune_paired_tests.csv`, `finetune_f1_by_freq_group.csv`, `finetune_f1_per_class.csv` |
| Recherche de clauses (m2) | `python -m ledgar.retrieval` (≈ 5 min sur T4, ≈ 1 h 30 sur CPU) | `results/tables/retrieval_*.csv` |
| LSTM + GloVe (m3) | `python -m ledgar.lstm` (GPU conseillé) | `results/tables/lstm_*.csv`, `results/predictions/lstm_*.csv`, `results/figures/lstm_learning_curve.png` |
| Volet B sur Colab | `notebooks/B_retrieval_lstm.ipynb` | les deux lignes ci-dessus + analyse d'erreurs |
| Comparaison de 7 encodeurs (volet B) | `python -m ledgar.encoders` (GPU, ≈ 45 min sur T4) ou `notebooks/B2_retrieval_encoders.ipynb` | `results/tables/retrieval_encoders.csv`, `results/figures/retrieval_encoders.png` |
| Audit de fiabilité et de biais (volet C) | `python -m ledgar.audit` (≈ 10 min sur CPU) | `results/tables/audit_*.csv`, `results/figures/audit_*.png`, résumé dans `reports/model_card.md` |

## Résultats

Métrique principale : **macro-F1** (100 classes très déséquilibrées). Tous les choix sont faits sur la validation ; le test n'est évalué qu'avec le modèle retenu.

| Modèle | Réglage (choisi sur la validation) | Macro-F1 validation | Macro-F1 test | Accuracy test | Jalon |
| --- | --- | --- | --- | --- | --- |
| TF-IDF (mots + bigrammes) + régression logistique | C = 30, `class_weight="balanced"` | 0,819 | 0,828 | 0,873 | m1 |
| Legal-BERT, fine-tuning complet (pré-enregistré) | 3 epochs, lr 2e-5, max_length 256 | 0,792 | 0,779 | 0,869 | m4 |
| Legal-BERT + LoRA r = 8 (W_Q, W_V) | 3 epochs, lr 5e-4 ; r = 8 meilleur rang sur la validation | 0,768 | 0,768 | 0,853 | m4 |
| Legal-BERT, fine-tuning complet jusqu'à convergence (protocole révisé) | arrêt anticipé, meilleure epoch 10 | 0,825 | 0,825 | 0,883 | m4 |
| BiLSTM + GloVe 6B 300d, max-pooling | 20 epochs max, early stopping (patience 2), meilleure epoch = 11 | 0,788 | 0,787 | 0,860 | m3 |

Sans les 2,8 % de clauses du test déjà présentes dans le train, la macro-F1 de la baseline est de 0,825 : les doublons ne gonflent pas le score de manière notable. La grille complète est dans `results/tables/baseline_grid.csv` ; avec ou sans pondération des classes, l'écart est de 0,001 au plus. Les confusions les plus fréquentes (`results/tables/baseline_top_confusions.csv`) sont les paires quasi synonymes repérées pendant l'exploration (*Governing Laws* / *Applicable Laws*, *Tax Withholdings* / *Withholdings*…).

### Volet A · Fine-tuning de Legal-BERT et LoRA (m4)

Correctement entraîné, Legal-BERT égale la baseline TF-IDF en macro-F1 et la dépasse en accuracy ; le run pré-enregistré (3 epochs) était sous-entraîné. LoRA r = 8 entraîne 0,34 % des paramètres pour 1,1 point de macro-F1 de moins que le fine-tuning complet à protocole égal. Toutes les comparaisons utilisent la baseline commune (C = 30) et les fonctions de `ledgar.metrics`.

**Tableau de compromis** (test ; `results/tables/finetune_tradeoff.csv`)

| Modèle | Paramètres entraînés | Temps GPU (T4) | Mémoire GPU | Macro-F1 val. | Accuracy test | Macro-F1 test | Macro-F1 test sans doublons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline TF-IDF (commune) | – | ≈ 25 min CPU | – | 0,819 | 0,873 | **0,828** | 0,825 |
| Legal-BERT complet, 3 epochs | 109,6 M (100 %) | 42 min | 3,3 Go | 0,792 | 0,869 | 0,779 | 0,777 |
| LoRA r = 4, 3 epochs | 224 k (0,20 %) | 35 min | 2,0 Go | 0,749 | 0,849 | 0,760 | 0,759 |
| LoRA r = 8, 3 epochs | 372 k (0,34 %) | 34 min | 2,0 Go | 0,768 | 0,853 | 0,768 | 0,767 |
| LoRA r = 16, 3 epochs | 667 k (0,61 %) | 34 min | 2,0 Go | 0,768 | 0,858 | 0,774 | 0,772 |
| Legal-BERT complet, arrêt anticipé (protocole révisé) | 109,6 M (100 %) | 152 min (12 epochs) | 3,3 Go | 0,825 | **0,883** | 0,825 | 0,822 |

**Tests appariés** sur les mêmes 10 000 clauses : McNemar pour l'accuracy (seuls les désaccords comptent), bootstrap apparié de 1 000 tirages pour la macro-F1 (`results/tables/finetune_paired_tests.csv`).

| A vs B | Seul A juste | Seul B juste | McNemar p | Δ macro-F1 (A − B) [IC 95 %] |
| --- | --- | --- | --- | --- |
| BERT complet 3 epochs vs baseline | 422 | 468 | 0,13 | −0,049 [−0,060 ; −0,032] |
| BERT complet arrêt anticipé vs baseline | 427 | 331 | 0,0006 | −0,003 [−0,014 ; +0,008] |
| BERT arrêt anticipé vs BERT 3 epochs | 393 | 251 | < 10⁻⁷ | +0,046 [+0,026 ; +0,056] |
| BERT complet 3 epochs vs LoRA r = 8 | 338 | 180 | < 10⁻¹⁰ | +0,011 [+0,002 ; +0,027] |
| LoRA r = 8 vs baseline | 416 | 620 | < 10⁻⁹ | −0,060 [−0,074 ; −0,047] |
| LoRA r = 16 vs LoRA r = 4 | 273 | 181 | < 10⁻⁴ | +0,014 [+0,004 ; +0,022] |

**F1 moyen par groupe de fréquence** (groupes de `class_counts.csv` ; `results/tables/finetune_f1_by_freq_group.csv`)

| Classes | Baseline | BERT 3 epochs | LoRA r = 8 | BERT arrêt anticipé |
| --- | --- | --- | --- | --- |
| Fréquentes | 0,892 | 0,895 | 0,879 | **0,902** |
| Moyennes | 0,835 | 0,807 | 0,785 | **0,840** |
| Rares | **0,756** | 0,632 | 0,638 | 0,731 |

À 3 epochs, BERT ne prédit jamais *Books*, *Qualifications*, *Assigns* ni *Applicable Laws* (F1 = 0), et range les classes quasi synonymes repérées pendant l'exploration dans la classe dominante. Entraîné jusqu'à convergence, il n'a plus aucune classe à 0 mais reste derrière la baseline sur plusieurs classes rares (*Books* 0,67 contre 1,00, *Qualifications* 0,55 contre 0,75 ; 5 exemples de test chacune ou moins).

**Hypothèses (écrites avant les runs)**

- **H1 rejetée.** À 3 epochs, Legal-BERT a une macro-F1 inférieure de 4,9 points à la baseline (IC 95 % excluant 0) pour une accuracy équivalente (p = 0,13). Run protocole révisé jusqu'à convergence : macro-F1 équivalente (−0,3 point, IC contenant 0) et accuracy supérieure de 1,0 point (p = 0,0006).
- **H2 validée sur l'estimation, sans garantie.** LoRA r = 8 entraîne 0,34 % des paramètres et utilise 40 % de mémoire GPU en moins, pour 1,1 point de macro-F1 de moins que le fine-tuning complet ; la borne haute de l'intervalle (2,7 points) dépasse le seuil de 2 points.
- **H3 rejetée.** De r = 4 à r = 16, la macro-F1 gagne 1,4 point en test (significatif) et 1,8 point en validation ; en validation, le gain se fait entre r = 4 et r = 8.

**Référence publiée.** LexGLUE ([Chalkidis et al., 2022](https://arxiv.org/abs/2110.00976), [résultats](https://github.com/coastalcph/lex-glue)) rapporte sur LEDGAR 88,2 de µ-F1 (= accuracy ici) et 83,0 de macro-F1 pour Legal-BERT, 87,0 / 81,4 pour TF-IDF + SVM. Notre Legal-BERT jusqu'à convergence obtient 88,3 / 82,5 avec des clauses tronquées à 256 tokens. LexGLUE n'évalue pas LoRA.

**Limites**

- Le run jusqu'à convergence a été ajouté après avoir vu les résultats du protocole pré-enregistré : il est présenté comme protocole révisé.
- `max_length = 256` : 13,5 % des clauses sont tronquées (sous-groupe de l'audit).
- La validation loss du run long remonte dès l'epoch 4 alors que la macro-F1 progresse : le modèle devient trop confiant, à vérifier par la calibration (audit).
- Le run LoRA r = 8 jusqu'à convergence a été interrompu par la limite de GPU de Colab à l'epoch 10 (macro-F1 de validation 0,784) : la comparaison LoRA / complet se fait à 3 epochs pour les deux.
- Loss non pondérée par classe pour BERT, contrairement à la baseline (`class_weight="balanced"`).

### Volet B · Recherche de clauses (m2) et LSTM (m3)

**LSTM (m3).** Entraîné sur un GPU T4 de Colab (19 s par epoch) ; l'early stopping s'arrête à l'epoch 13 et garde l'epoch 11, la meilleure sur la validation (`results/figures/lstm_learning_curve.png`). Les 10 premières epochs sont identiques, au dixième de millième près, à celles d'un premier run plafonné à 10 epochs : le seed rend l'entraînement reproductible. Le LSTM reste sous la baseline : −4,1 points de macro-F1 (bootstrap apparié, IC 95 % [−5,5 ; −2,8]), McNemar p < 10⁻⁵ (384 clauses bien classées par le seul LSTM, 521 par la seule baseline). L'écart vient surtout des classes rares (F1 moyenne 0,70 contre 0,77 pour le tiers le moins fréquent, 0,87 contre 0,88 pour le plus fréquent). Les confusions principales sont les mêmes paires quasi synonymes que la baseline (*Applicable Laws* → *Governing Laws*, *Waivers* → *No Waivers*, *Integration* → *Entire Agreements*).

#### Recherche de clauses (m2)

Banque = les 60 000 clauses du train ; une clause retrouvée est pertinente si elle a le type de la clause requête. Les deux réglages (poids de la fusion, seuil de confiance) sont choisis sur la validation (`results/tables/retrieval_choices.csv`). Test, jumeaux exacts du train retirés du classement :

| Méthode | P@1 | P@5 | P@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| TF-IDF, cosinus | 0,832 | 0,779 | 0,748 | 0,872 |
| `all-MiniLM-L6-v2`, cosinus | 0,807 | 0,748 | 0,719 | 0,853 |
| Fusion 0,5 × MiniLM + 0,5 × TF-IDF | 0,842 | 0,789 | 0,760 | 0,880 |
| Fusion + filtre sur la classe prédite par la baseline | 0,873 | 0,873 | 0,873 | 0,873 |
| Fusion + filtre seulement si la confiance du classifieur ≥ 0,2 (version retenue) | 0,873 | 0,873 | 0,872 | 0,875 |

- Le vocabulaire des contrats est très standardisé : la recherche lexicale bat les embeddings génériques, et la fusion des deux bat chacune (+1 point de P@5 sur TF-IDF, +4 sur MiniLM).
- Avec le filtre, la P@k est bornée par l'accuracy du classifieur (0,873) : un meilleur classifieur améliore directement la recherche.
- Seuil de confiance : sur la validation, toutes les clauses appartiennent aux 100 types, donc filtrer aide toujours et le meilleur seuil serait 0, ce qui ne protège pas les questions hors des types. On garde le seuil le plus haut qui coûte au plus 0,5 point de P@5 en validation : 0,2. Sur le test, il ne coûte rien.
- Questions d'avocat (`results/tables/retrieval_examples.csv`, vérifiées à la main) : pour « termination clauses with a 30-day notice period », les 5 résultats hybrides sont des résiliations à 30 jours. Les questions hors des 100 types (non-concurrence, limitation de responsabilité du fournisseur) ont une confiance de 0,16 et 0,05 : elles ne sont plus forcées dans une mauvaise classe (avant : *Benefits*). Les résultats restent peu pertinents, car LEDGAR n'a pas de type pour ces clauses : c'est une limite du jeu de données, pas un réglage.

#### Quel encodeur pour la banque de clauses ?

7 encodeurs pré-entraînés, sans entraînement, même protocole que ci-dessus (`python -m ledgar.encoders`, Colab T4, textes coupés à 256 sous-mots pour tous). Chacun est mesuré seul, puis fusionné avec TF-IDF (alpha choisi sur la validation). Test, jumeaux retirés (`results/tables/retrieval_encoders.csv`, `results/figures/retrieval_encoders.png`) :

| Encodeur | Paramètres | P@5 seul | P@5 fusion (alpha) | MRR@10 fusion |
| --- | --- | --- | --- | --- |
| TF-IDF seul (référence) | — | 0,779 | — | 0,872 |
| `all-MiniLM-L6-v2` (jalon m2) | 23 M | 0,748 | 0,790 (0,5) | 0,879 |
| `all-mpnet-base-v2` | 110 M | 0,745 | 0,792 (0,5) | 0,882 |
| `bge-small-en-v1.5` | 33 M | 0,778 | 0,797 (0,75) | 0,882 |
| `gte-base` | 110 M | 0,777 | 0,797 (0,75) | 0,885 |
| Legal-BERT (mean pooling) | 110 M | 0,783 | 0,797 (0,75) | 0,883 |
| `e5-base-v2` | 110 M | 0,785 | 0,799 (0,75) | 0,885 |
| `bge-base-en-v1.5` | 110 M | 0,783 | 0,799 (0,75) | 0,884 |

- Les encodeurs récents entraînés pour la recherche (e5, bge, gte) rattrapent TF-IDF seuls ; MiniLM et mpnet restent 3 points dessous.
- La fusion avec TF-IDF aide **tous** les encodeurs : lexical et sémantique se complètent.
- Les 5 meilleures fusions tiennent en 0,2 point de P@5 : l'écart n'est pas significatif à cette échelle. `bge-small` fait aussi bien que les modèles 3 fois plus gros pour 2 fois moins de temps d'encodage : c'est le meilleur compromis pour un déploiement.
- Legal-BERT, pré-entraîné sur des contrats mais pas pour la similarité, égale les encodeurs généralistes : le domaine compense l'absence d'entraînement à la recherche.
- La fusion MiniLM retrouve 0,790 comme `ledgar.retrieval` (0,789) : les deux implémentations concordent.

### Volet C · Audit : où lui faire confiance

Détail, usages interdits et limites : **[`reports/model_card.md`](reports/model_card.md)**. Les quatre constats principaux (test) :

- **Formules types contre formulations nouvelles.** 98,8 % d'accuracy sur les clauses quasi copiées du train (15 % du test), 81,5 % sur les formulations nouvelles (63 %) pour la baseline ; Legal-BERT y gagne 1,5 point (83,0 %). Les types rares perdent ~15 points.
- **Confiance.** Celle de la baseline est honnête (ECE 0,023) ; Legal-BERT est très surconfiant (confiance moyenne 0,98 pour 0,88 d'accuracy, 74 % de ses erreurs à plus de 0,9).
- **Relecture humaine.** Un seuil choisi sur la validation pour 95 % d'accuracy laisse les deux modèles étiqueter seuls ~80 % des clauses (94,8 % d'accuracy au test) et envoie les ~20 % restantes à un avocat.
- **Robustesse.** Genre, forme de société et chiffres changent moins de 3,5 % des prédictions de la baseline ; remplacer la juridiction par la France en change 11,8 % (accuracy 81 % → 72 %). Les clauses en français et hors des 100 types sont toutes envoyées à un avocat par le seuil.

## Règles de travail

- On ne modifie jamais `main` directement : une branche par tâche, puis une Pull Request (modèle pré-rempli).
- Phase 0 (setup, jusqu'au tag `m1`) : PR fusionnées par leur auteur. Ensuite : relecture obligatoire par un autre membre, puis squash merge.
- Jamais de données, de modèles ni de sorties de notebook dans Git. Les prédictions (`results/predictions/`) ne contiennent que `idx, label, pred, confidence`.

## Équipe

_À compléter._
