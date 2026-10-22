# Model card · Classifieur de clauses LEDGAR

Cœur NLP d'un assistant de revue de contrats : il attribue à une clause l'un des 100 types de LEDGAR et retrouve des
clauses similaires dans une banque. Cette fiche dit **où le système est fiable, où il ne l'est pas, et ce pour quoi il ne
doit pas être utilisé**. Chiffres issus de `python -m ledgar.audit` (tableaux `results/tables/audit_*.csv`).

## Usage prévu

- **Aide au tri** pendant une due diligence : proposer un type pour chaque clause d'un contrat et des précédents
  similaires, **toujours relus par un juriste**.
- Contrats **américains, en anglais**, du type de ceux déposés à la SEC (prêts, emploi, acquisitions, baux…).
- Avec un **seuil de confiance** : sous le seuil, la clause est envoyée à un avocat au lieu d'être étiquetée seule.

## Usages interdits

- **Aucune décision juridique sans relecture humaine.** Le système classe, il ne juge pas de la validité, du caractère
  abusif ou des risques d'une clause.
- **Pas de contrats en français ni dans une autre langue** : les 3 clauses françaises testées sont toutes mal classées
  (voir « Hors distribution »).
- **Pas de droit non américain sans validation préalable** : changer « State of New York » en « France » fait changer
  la prédiction de 12 % des clauses concernées.
- **Pas de types hors des 100 de LEDGAR** (non-concurrence, limitation de responsabilité…) : le système n'a pas de
  réponse « autre » ; il propose un type existant, avec une confiance faible.
- **Pas d'usage de la confiance brute de Legal-BERT comme probabilité** : elle est très surestimée (voir « Calibration »).

## Données

LEDGAR, version LexGLUE ([Chalkidis et al., 2022](https://arxiv.org/abs/2110.00976)) : clauses de contrats déposés à
la SEC, 100 types, découpage chronologique (train 2016-2017 : 60 000 ; validation 2018 : 10 000 ; test 2019 : 10 000).
Très déséquilibré (×138 entre *Governing Laws* et *Books*). Les attributs d'audit (juridiction, partie, type de contrat,
genre…) sont déduits du texte par expressions régulières (`ledgar.metadata`) : ce sont des approximations.

## Performances globales (test, 10 000 clauses)

| Modèle | Macro-F1 | Accuracy | Coût |
| --- | --- | --- | --- |
| TF-IDF + régression logistique (baseline) | 0,828 | 0,873 | ≈ 4 min, CPU |
| Legal-BERT, fine-tuning complet, arrêt anticipé | 0,825 | 0,883 | ≈ 2 h 30, GPU T4 |
| BiLSTM + GloVe | 0,788 | 0,860 | GPU, 19 s / epoch |
| Legal-BERT + LoRA r = 8 (3 epochs) | 0,768 | 0,853 | ≈ 35 min, GPU |

Legal-BERT égale la baseline en macro-F1 (écart non significatif, bootstrap apparié) et la dépasse d'un point en
accuracy (McNemar, p = 0,0006). Référence publiée : 83,0 de macro-F1 pour Legal-BERT dans LexGLUE.

## Où lui faire confiance, où non

Accuracy au test par sous-groupe (`results/figures/audit_subgroups.png`) :

| Sous-groupe | Part du test | Baseline | Legal-BERT | Lecture |
| --- | --- | --- | --- | --- |
| Quasi-copie d'une clause du train (similarité ≥ 0,95) | 15 % | 98,8 % | 98,9 % | Formules types : fiable |
| Formulation nouvelle (similarité < 0,80) | 63 % | 81,5 % | 83,0 % | **Le vrai niveau du système** ; BERT y gagne 1,5 point |
| Type fréquent / rare | 64 % / 12 % | 90,8 % / 76,3 % | 92,3 % / 77,1 % | Les types rares perdent ~15 points |
| Baux / services (type de contrat déduit) | 3 % / 2 % | 74,8 % / 79,1 % | 74,1 % / 81,6 % | Contrats les moins bien couverts |
| Clauses tronquées (> 256 tokens) | 11 % | 91,1 % | 89,8 % | Pas pénalisées : le type se lit dès le début |

Écarts entre juridictions (New York, Delaware, Californie…) : groupes de 40 à 280 clauses, intervalles larges ; ils
reflètent surtout les types de clauses qui citent un État. Les groupes de moins de 50 clauses (féminin, hors US, Texas)
ne permettent pas de conclure.

## Calibration et relecture humaine

| | Baseline | Legal-BERT |
| --- | --- | --- |
| Confiance moyenne / accuracy | 0,867 / 0,873 | **0,980 / 0,883** |
| Erreur de calibration (ECE, 0 = honnête) | 0,023 | **0,097** |
| Part des erreurs commises avec une confiance ≥ 0,9 | 17 % | **74 %** |

La confiance de la baseline est honnête ; celle de Legal-BERT ne l'est pas : trois quarts de ses erreurs sont faites
avec plus de 90 % de confiance. Un seuil choisi **sur la validation** pour atteindre 95 % d'accuracy sur les clauses
traitées seules donne au test (`results/figures/audit_abstention.png`) :

| | Seuil | Clauses étiquetées seules | Leur accuracy | Envoyées à un avocat |
| --- | --- | --- | --- | --- |
| Baseline | 0,74 | 81 % | 94,8 % | 19 % |
| Legal-BERT | 0,996 | 80 % | 94,8 % | 20 % |

Avec une relecture humaine calibrée, les deux modèles se valent ; la baseline le fait sur CPU, avec une confiance
lisible et des décisions explicables (mots pondérés).

## Robustesse : tests contrefactuels (baseline)

On modifie un élément qui ne change pas le type de la clause, et on compte les prédictions qui changent.

| Modification | Clauses | Prédictions changées | Accuracy avant → après |
| --- | --- | --- | --- |
| Genre des pronoms (he ↔ she) | 479 | 1,5 % | 82,0 % → 82,0 % |
| Forme de société (Inc. → SAS, LLC → GmbH…) | 218 | 3,2 % | 80,3 % → 81,2 % |
| Chiffres (délais + 15, $ → €) | 878 | 0,5 % | 90,3 % → 90,4 % |
| Juridiction → England and Wales | 796 | 8,9 % | 81,0 % → 85,6 % |
| **Juridiction → France** | 797 | **11,8 %** | **81,1 % → 71,6 %** |

Pas de sensibilité notable au genre, à la forme de société ou aux chiffres. En revanche, le modèle est lié au
vocabulaire américain : une juridiction française dégrade nettement les prédictions, ce qui confirme qu'un déploiement
européen exige une validation sur des contrats européens.

## Hors distribution (9 clauses rédigées à la main, baseline)

| Sonde | Correctes | Envoyées à un avocat par le seuil |
| --- | --- | --- |
| Anglais, style européen (4) | 2 | les 2 erreurs |
| Français (3) | 0 | les 3 |
| Type absent des 100 (2) | — | les 2 |

Les 7 erreurs ont toutes une confiance sous le seuil : la relecture humaine les aurait rattrapées. Détail :
`results/tables/audit_ood.csv`. Ces sondes illustrent, elles ne mesurent pas.

## Limites connues

- Les contrefactuels et les sondes hors distribution sont faits sur la baseline : les poids de Legal-BERT ne sont pas
  dans le repo. `ledgar.audit.counterfactual_report` et `ood_report` acceptent n'importe quelle fonction de prédiction
  pour les relancer sur BERT.
- Attributs d'audit déduits par regex, sur une seule clause ; « emploi » inclut les attributions d'actions.
- Un seul run (seed) par modèle ; clauses tronquées à 256 tokens pour BERT.
- Évaluation sur des contrats de 2019 : le vocabulaire juridique évolue (dérive à surveiller).
- La recherche de clauses juge la pertinence par « même type », un critère trop strict pour les types quasi synonymes
  (*Governing Laws* / *Applicable Laws*).

## Reproduire

```bash
python -m ledgar.baseline   # baseline et ses prédictions
python -m ledgar.audit      # cette fiche : tableaux et figures audit_*
```
