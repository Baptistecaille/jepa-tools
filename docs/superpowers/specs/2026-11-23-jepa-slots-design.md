# Design : cœur JEPA par emplacements (jepa-tools)

Statut : brouillon en attente d'approbation. Date : 2026-11-23.
Emplacement prévu dans le repo : `docs/superpowers/specs/2026-11-23-jepa-slots-design.md`.

## 1. Intention

Pouvoir créer un JEPA en choisissant une pièce par emplacement (échantillonnage, encodeur, cible, prédicteur, objectif), depuis un fichier de config ou en Python, sans recoder le modèle ni la boucle d'entraînement. Le périmètre initial est l'apprentissage de représentations. Les modèles du monde conditionnés par des actions et la planification sont hors périmètre pour l'instant.

Pourquoi : la littérature récente montre qu'un JEPA se décompose en quelques pièces et que chaque papier n'en change qu'une ou deux. I-JEPA : masquage multi-blocs, cible par EMA, prédicteur étroit. LeJEPA : perte prédictive plus SIGReg, sans stop-gradient ni EMA. V-JEPA 2 : même schéma sur vidéo, puis post-entraînement avec actions. Le fork actuel code un seul schéma (prédiction d'un état suivant sur vecteurs poolés) et le fige dans `JEPA.forward` et dans le trainer.

## 2. Objectifs, non-objectifs, anti-objectifs

Objectifs
1. Un preset YAML par variante connue tourne sans écrire de code : `temporal` (comportement actuel du fork), `ijepa`, `lejepa`.
2. Ajouter une pièce (un nouveau régularisateur, un nouveau masque) = un fichier et un décorateur d'enregistrement, zéro modification du cœur.
3. Une spec de run (YAML ou dataclass) détermine entièrement le run. Elle est sauvegardée avec chaque checkpoint.
4. L'évaluation est intégrée : sonde linéaire sur encodeur gelé et diagnostics de collapse (écart-type par dimension, rang effectif). La perte seule ne suffit pas.
5. Réutiliser du fork : loggers (console, W&B, TensorBoard), code DDP, save/from_pretrained, VICReg.

Non-objectifs
- Planification, MPC, prédicteur conditionné par des actions ou du langage (l'interface laisse la place, rien n'est implémenté).
- Têtes probabilistes ou génératives (Var-JEPA, Branch-JEPA, flow matching).
- Reproduire les chiffres publiés à grande échelle (ImageNet complet).
- Réécrire un ViT concurrent de timm, recherche d'hyperparamètres, serving.
- Supporter tout modèle HuggingFace.

Anti-objectifs (échec même si ça tourne)
- Un cadre dont l'usage demande de comprendre tous les crochets.
- Un défaut qui s'effondre en silence (voir garde-fous, section 5).
- Un nouveau god-object de type `JEPATrainer` actuel (31 Ko qui mélangent DDP, HF, logging et formats de batch).

## 3. Contraintes

- PyTorch. Python 3.10 ou plus. Dépendances obligatoires minimales (torch, pyyaml, numpy). timm, transformers, wandb, tensorboard, matplotlib sont optionnels et jamais importés à l'import du paquet.
- Licence MIT de l'amont conservée, avec attribution (le repo est un fork de dipsivenkatesh/jepa).
- Tous les tests passent sur CPU en moins de quelques minutes. Un GPU n'est requis que pour les runs réels.
- DDP via `torchrun`. Un seul GPU est le cas de référence.

## 4. Approches envisagées

- A. Refactorer en place : peu coûteux au départ, mais `forward` est câblé sur `(state_t, state_t1)` et le trainer dépend du format des tuples. Rejetée : chaque nouveau mode toucherait tout.
- B. Nouveau cœur par emplacements avec registre : plus de travail initial, seul chemin vers "un JEPA par config". Choisie par Baptiste.
- C. Couche de recettes sur stable_pretraining (wrapper Lightning utilisé par le repo LeJEPA) : le plus rapide, mais dépendance à l'API d'une autre bibliothèque et perte de contrôle sur la boucle. Non retenue ; un adaptateur Lightning reste possible plus tard car `JEPAModule` expose un `training_step`.

## 5. Architecture

### 5.1 Contrat de données (le cœur du design)

Tous les schémas passent par la même structure. Les indices désignent des positions de tokens dans la grille complète de l'entrée.

```python
@dataclass
class Plan:                      # produit par le Sampler
    ctx_inputs: Tensor           # ce que voit l'encodeur contexte
    tgt_inputs: Tensor           # ce que voit la cible (peut être identique à ctx_inputs)
    ctx_idx: LongTensor | None   # (B, Nc) tokens visibles du contexte, None = tous
    tgt_idx: LongTensor | None   # (B, Nt) tokens à prédire, None = tous

@dataclass
class StepOutputs:               # transmis à l'objectif
    pred: Tensor                 # (B, Nt, D)
    target: Tensor               # (B, Nt, D)
    emb: Tensor                  # (B, Nc, D) embeddings en ligne du contexte
    progress: float              # 0 à 1, pour les schedules
```

Interfaces des pièces

- `Sampler.__call__(batch) -> Plan`
- `Encoder.forward(x, keep_idx=None) -> (B, N, D)`. Un modèle poolé renvoie N=1.
- `Target.encode(plan, online_encoder) -> (B, Nt, D)`. Modes : `ema` (copie, mise à jour par `update(progress)`), `shared` (même encodeur, option `detach`), `frozen` (encodeur préentraîné). Expose aussi `eval_encoder()`.
- `Predictor.forward(ctx_tokens, ctx_idx, tgt_idx, cond=None) -> (B, Nt, D)`. `cond` est réservé (None partout pour l'instant). Implémentations : `narrow_vit`, `mlp`, `identity`.
- `Objective.forward(out: StepOutputs) -> (loss, logs)`. Composition : une distance (`l2`, `smooth_l1`, `cosine`) plus un régularisateur optionnel (`vicreg`, `sigreg`, `none`), chacun avec un poids.

Vérification que les trois schémas tiennent

| Schéma | Sampler | Cible | Prédicteur | Objectif |
|---|---|---|---|---|
| I-JEPA | multi-blocs, ctx_inputs = tgt_inputs, idx renseignés | ema | narrow_vit | l2 sur tgt_idx |
| LeJEPA | multi-vues, deux vues augmentées, idx None | shared, detach=False | identity ou mlp | alignement + sigreg |
| temporal (fork actuel) | paires (x_t, x_t+1), N=1 | shared, detach=True | mlp | mse + régularisateur |

V-JEPA tient aussi (masques en tubes sur une grille spatio-temporelle) mais n'est pas dans le premier jalon.

### 5.2 Assemblage

`JEPAModule` (nn.Module) contient les pièces et expose :
- `training_step(batch, progress) -> (loss, logs)`
- `after_step(progress)` (mise à jour EMA)
- `eval_encoder()` (pour I-JEPA, l'encodeur cible moyenné, comme dans l'annexe du papier)
- `save_pretrained` / `from_pretrained` (repris de `BaseModel`, avec la spec dans `config.json`)

Registre : `@register("encoder", "vit")`. Résolution : `build(spec) -> JEPAModule` avec messages d'erreur clairs (liste des noms disponibles, comme `get_loss` aujourd'hui).

Spec de run (illustration) :
```yaml
preset: ijepa
data:      {name: cifar10, root: ./data, batch_size: 256}
sampler:   {name: multiblock, n_targets: 4}
encoder:   {name: vit, dim: 384, depth: 12}
target:    {name: ema, momentum: [0.996, 1.0]}
predictor: {name: narrow_vit, dim: 192, depth: 6}
objective: {distance: l2, regularizer: none}
trainer:   {epochs: 100, lr: 1.0e-3, amp: bf16}
```
Surcharges en ligne de commande : `jepa-train spec.yaml encoder.depth=6`.

### 5.3 Entraînement

Boucle mince, qui ne connaît que l'API de `JEPAModule` : optimisation, AMP, clipping, schedules (lr cosinus avec warmup, weight decay), DDP, checkpoint (spec, états, RNG), reprise, logging via les loggers conservés. Pas de dispatch sur la forme des batchs : un `DataModule` renvoie toujours ce que le `Sampler` attend.

### 5.4 Évaluation et garde-fous

- Sonde linéaire sur `eval_encoder()` gelé, avec pooling moyen.
- Diagnostics loggés tous les N pas : écart-type par dimension, rang effectif, cosinus moyen entre échantillons. Alerte si l'écart-type moyen passe sous un seuil.
- Validation à la construction : un montage `target=shared, detach=true` avec un objectif sans régularisateur échoue avec un message explicite, sauf option `allow_collapse_risk: true`. Les formes (Sampler, Encoder, Predictor) sont vérifiées sur un mini-batch fictif avant de lancer le run.

## 6. Migration du fork

Structure cible (indicative) :
```
jepa_tools/
  core/        registre, spec, types, module
  slots/       sampler/ encoder/ target/ predictor/ objective/
  data/        datasets utiles, DataModule
  train/       boucle, schedules, checkpoint, DDP
  eval/        probe, diagnostics
  loggers/     repris du fork
  presets/     temporal.yaml ijepa.yaml lejepa.yaml
  cli.py
```
Reprises : loggers, DDP, save/from_pretrained, VICReg, idée des dataclasses de config. Supprimés : `old_structure/`, `JEPA`, `JEPAAction`, `JEPAInterleaved`, l'ancien trainer (le comportement `temporal` survit comme preset ; l'historique git garde l'ancien code). Corrections intégrées : `__all__` erroné, import de matplotlib au chargement, désaccord `wandb:` / `logging.wandb:` dans les YAML, `vcreg_loss` incompatible avec le trainer.

## 7. Jalons proposés

1. Hygiène et tests de base (paquet importable, imports légers).
2. Cœur : types, registre, spec, `JEPAModule`, boucle.
3. Preset `temporal` avec test d'équivalence numérique contre l'ancien `JEPA` sur un jouet (à faire avant la suppression).
4. Preset `ijepa` sur un petit jeu d'images, avec sonde.
5. Preset `lejepa` avec SIGReg.
6. Diagnostics, garde-fous, documentation, exemples.

Critères de réussite
- Les trois presets tournent depuis un YAML sans écrire de code.
- Ajouter un régularisateur exige un seul fichier et aucune modification du cœur (démontré en ajoutant `sigreg` après `vicreg`).
- Un run `shared + detach + mse seul` est refusé à la construction, ou signalé par les diagnostics s'il est forcé.
- Sur le petit jeu d'images, la sonde linéaire de l'encodeur entraîné bat celle d'un encodeur à poids aléatoires.
- Suite de conformité paramétrée sur le registre : chaque pièce enregistrée passe les tests de formes.

## 8. Choix faits par l'auteur du design (non validés par Baptiste)

- Nom du paquet `jepa_tools` (le nom `jepa` est celui du paquet PyPI amont, risque de collision).
- Python 3.10 ou plus, pas de rétrocompatibilité avec l'ancienne API (un seul utilisateur, historique git conservé).
- Sonde linéaire sur l'encodeur cible pour I-JEPA, comme dans le papier.
- Registre par décorateur seulement (pas de plugins externes pour l'instant).
- Tests CPU, petits.

## 9. Laissé au constructeur

- Hyperparamètres exacts de chaque preset.
- ViT maison minimal ou enveloppe timm.
- Implémentation de SIGReg : à reprendre du papier LeJEPA (arXiv 2511.08544) et de son dépôt de référence, que je n'ai pas lus en détail.
- Format exact des DataModules, détails de la CLI, documentation.

## 10. Risques

- SIGReg mal implémenté faute de lecture fine du papier : à valider contre le dépôt de référence.
- Le contrat `Plan` avec indices est le point le plus délicat : trop souple pour les modèles poolés, trop rigide pour V-JEPA. À éprouver sur le jalon 4.
- Dérive du registre vers un framework de plus : garder les interfaces à cinq méthodes au plus.

## 11. Question ouverte pour Baptiste

Quel jeu de données ou quelle modalité sert de premier banc d'essai réel (images, séries temporelles, autre) ? La réponse change les presets et tests livrés en premier, pas l'architecture.

Réponse (2026-11-23) : images d'abord.
