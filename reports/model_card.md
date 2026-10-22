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
  la prédiction de 12 % des clauses concernées pour la baseline (4 % pour Legal-BERT).
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

## Robustesse : tests contrefactuels

On modifie un élément qui ne change pas le type de la clause, et on compte les prédictions qui changent
(`audit_counterfactuals.csv` pour la baseline, `audit_counterfactuals_legalbert.csv` pour Legal-BERT, lancé sur Colab
avec le modèle final par le volet A).

| Modification | Clauses | Baseline : prédictions changées (accuracy avant → après) | Legal-BERT |
| --- | --- | --- | --- |
| Genre des pronoms (he ↔ she) | 479 | 1,5 % (82,0 % → 82,0 %) | 1,5 % (82,5 % → 82,0 %) |
| Forme de société (Inc. → SAS, LLC → GmbH…) | 218 | 3,2 % (80,3 % → 81,2 %) | 1,8 % (80,3 % → 79,4 %) |
| Chiffres (délais + 15, $ → €) | 878 | 0,5 % (90,3 % → 90,4 %) | 0,7 % (91,0 % → 90,9 %) |
| Juridiction → England and Wales | 796 | 8,9 % (81,0 % → 85,6 %) | 3,8 % (86,2 % → 85,7 %) |
| **Juridiction → France** | 797 | **11,8 % (81,1 % → 71,6 %)** | **4,4 % (86,2 % → 84,7 %)** |

Aucun des deux modèles n'est sensible au genre, à la forme de société ou aux chiffres. La baseline est liée au
vocabulaire américain (une juridiction française lui coûte 10 points) ; Legal-BERT, qui lit le sens de la clause, l'est
trois fois moins. Un déploiement européen exige tout de même une validation sur des contrats européens.

## Hors distribution (9 clauses rédigées à la main)

| Sonde | Baseline : correctes | Legal-BERT : correctes | Erreurs qui échappent à l'avocat (baseline / BERT seul / combiné) |
| --- | --- | --- | --- |
| Anglais, style européen (4) | 2 | 3 | 0 / 0 / 0 |
| Français (3) | 0 | 1 | 0 / 0 / 0 |
| Type absent des 100 (2) | — | — | 0 / **1** / 0 |

Legal-BERT se trompe moins, mais **une de ses erreurs passe le seuil** : la clause de plafond de responsabilité (type
absent des 100) est classée *Fees* avec 99,9 % de confiance. Il classe aussi la clause française sur le droit
applicable faux avec 98,6 % de confiance (rattrapée de justesse par son seuil de 0,996). Avec la règle combinée
ci-dessous, aucune erreur n'échappe à la relecture. Détail : `audit_ood.csv`, `audit_ood_legalbert.csv`,
`audit_ood_combined.csv`. Ces sondes illustrent, elles ne mesurent pas.

## Système recommandé : Legal-BERT ET la baseline

Chaque modèle a une force : Legal-BERT est plus précis et plus robuste, la baseline a une confiance honnête. On les
combine : **Legal-BERT propose le type, et la clause n'est étiquetée sans relecture que si la baseline est d'accord et
confiante** (seuil 0,74, choisi sur la validation). Sinon, elle part chez un avocat.

| Règle (test) | Clauses étiquetées seules | Leur accuracy | Erreurs non relues pour 1 000 clauses |
| --- | --- | --- | --- |
| Legal-BERT seul (seuil 0,996) | 80,0 % | 94,8 % | 41 |
| Baseline seule (seuil 0,74) | 81,0 % | 94,8 % | 42 |
| **Legal-BERT ET baseline** | **78,3 %** | **96,1 %** | **31** |

Un quart d'erreurs non relues en moins, pour 2 points de clauses en plus envoyées à un avocat ; même résultat sur la
validation (30 contre 40). Quand les deux modèles sont en désaccord (10 % des clauses), Legal-BERT n'a raison que dans
43 % des cas : le désaccord est un excellent signal d'alerte. Figure : `results/figures/audit_combined.png`.

## Limites connues

- Les poids de Legal-BERT ne sont pas dans le repo : ses contrefactuels et sondes ont été lancés sur Colab
  (`ledgar.audit.counterfactual_report` et `ood_report` acceptent n'importe quelle fonction de prédiction).
- La règle combinée a été conçue après avoir vu les sondes hors distribution ; ses seuils viennent de la validation et
  son gain est le même sur la validation et sur le test.
- La confiance de Legal-BERT pourrait être recalibrée (*temperature scaling* sur la validation) : piste non testée.
- Attributs d'audit déduits par regex, sur une seule clause ; « emploi » inclut les attributions d'actions.
- Un seul run (seed) par modèle ; clauses tronquées à 256 tokens pour BERT.
- Évaluation sur des contrats de 2019 : le vocabulaire juridique évolue (dérive à surveiller).
- La recherche de clauses juge la pertinence par « même type », un critère trop strict pour les types quasi synonymes
  (*Governing Laws* / *Applicable Laws*).

## Reproduire

```bash
python -m ledgar.baseline   # baseline et ses prédictions
python -m ledgar.audit      # cette fiche : tableaux et figures audit_*
python -m ledgar.audit --combined   # la règle Legal-BERT ET baseline seule, en quelques secondes
```
