# Structure des fichiers — JEPA par emplacements

Spec : `docs/superpowers/specs/2026-11-23-jepa-slots-design.md`. Plan complet (interfaces, tests, étapes) : `docs/superpowers/plans/2026-11-23-jepa-slots.md`. Ce document ne contient que l'arborescence et le rôle de chaque fichier. Le code est à écrire par Baptiste.

`T#` renvoie à la tâche du plan qui crée ou modifie le fichier.

## Arborescence cible

```
jepa-tools/
├── NOTICE.md                         # attribution amont (dipsivenkatesh/jepa) + MIT                 T1
├── pyproject.toml                    # nom jepa-tools, python>=3.10, deps minimales, extras, script jepa-train   T1, T13, T17
├── setup.py, MANIFEST.in             # ajout du paquet jepa_tools puis retrait de jepa               T1, T17
├── README.md, CHANGELOG.md           # quick start par preset, rupture d'API                          T17
│
├── jepa_tools/
│   ├── __init__.py                   # __version__, exporte build / load_spec / JEPAModule            T1, T9
│   ├── cli.py                        # main(argv): `jepa-train spec.yaml a.b=c`, sonde finale         T13
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                  # Plan, StepOutputs                                              T2
│   │   ├── registry.py               # register / get / names / RegistryError                         T2
│   │   ├── spec.py                   # RunSpec, load_spec (presets, fusion, surcharges)               T3
│   │   ├── module.py                 # JEPAModule (training_step, after_step, eval_encoder, save/load) T9
│   │   ├── build.py                  # build(spec), garde-fous collapse et formes                     T9
│   │   └── persistence.py            # logique save/from_pretrained reprise de jepa/models/base.py   T17
│   │
│   ├── slots/
│   │   ├── __init__.py               # discover(): importe tous les sous-modules (enregistrement auto) T2
│   │   ├── sampler/
│   │   │   ├── __init__.py
│   │   │   ├── multiblock.py         # I-JEPA                                                         T8
│   │   │   ├── temporal_pair.py      # (x_t, x_t+1)                                                   T8
│   │   │   └── multiview.py          # LeJEPA                                                         T15
│   │   ├── encoder/
│   │   │   ├── __init__.py
│   │   │   ├── mlp.py                # N=1, expose .net                                               T5
│   │   │   └── vit.py                # ViT minimal, keep_idx                                          T5
│   │   ├── predictor/
│   │   │   ├── __init__.py
│   │   │   ├── simple.py             # identity, mlp (expose .net)                                    T6
│   │   │   └── narrow_vit.py         # prédicteur étroit I-JEPA                                       T6
│   │   ├── target/
│   │   │   ├── __init__.py
│   │   │   └── target.py             # ema, shared, frozen                                            T7
│   │   └── objective/
│   │       ├── __init__.py
│   │       ├── distances.py          # l2, smooth_l1, cosine                                          T4
│   │       ├── regularizers.py       # none, vicreg                                                   T4
│   │       ├── objective.py          # Objective(distance + regularizer)                              T4
│   │       └── sigreg.py             # un seul fichier pour ajouter SIGReg                            T15
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── images.py                 # make_data, fake_images, cifar10 (torchvision paresseux)        T11
│   │   └── temporal.py               # jeu jouet toy_temporal                                         T14
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── schedules.py              # momentum_at, cosine_with_warmup                                T7
│   │   ├── loop.py                   # fit()                                                          T12
│   │   ├── checkpoint.py             # save/load (spec, états, RNG)                                   T12
│   │   └── dist.py                   # setup_dist / cleanup_dist (torchrun)                           T12
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── diagnostics.py            # collapse_stats, check_collapse                                 T10
│   │   └── probe.py                  # extract_features, linear_probe                                 T10
│   │
│   ├── loggers/                      # déplacé depuis jepa/loggers (git mv)                           T12
│   │   └── (base, console, multi, tensorboard, wandb)
│   │
│   └── presets/
│       ├── __init__.py
│       ├── ijepa.yaml                                                                                  T13
│       ├── temporal.yaml                                                                               T14
│       └── lejepa.yaml                                                                                 T15
│
├── tests/
│   ├── test_imports.py                                                                                 T1
│   ├── test_ijepa_e2e.py                                                                               T13
│   ├── test_temporal_equivalence.py  # à passer AVANT toute suppression de jepa/                       T14
│   ├── test_lejepa_e2e.py                                                                              T15
│   ├── test_no_legacy.py                                                                               T17
│   ├── core/    test_registry.py, test_spec.py, test_module.py                                         T2, T3, T9
│   ├── slots/   test_objective.py, test_encoder.py, test_predictor.py, test_target.py,
│   │            test_sampler.py, test_sigreg.py                                                        T4–T8, T15
│   ├── data/    test_images.py                                                                         T11
│   ├── train/   test_schedules.py, test_loop.py, test_checkpoint.py                                    T7, T12
│   ├── eval/    test_diagnostics.py, test_probe.py                                                     T10
│   └── conformance/ test_registry_conformance.py                                                       T16
│
└── examples/                         # un exemple par preset (remplace les anciens)                   T17
```

## Fichiers existants touchés

| Fichier | Changement | Tâche |
|---|---|---|
| `jepa/trainer/utils.py:8`, `jepa/cli/evaluate.py:16`, `jepa/data/utils.py:12` | `import matplotlib` déplacé dans les fonctions | T1 |
| `jepa/__init__.py` | `__all__` corrigé si le test d'import échoue | T1 |
| `jepa/loggers/` | `git mv` vers `jepa_tools/loggers/`, imports et `tests/test_logging.py` mis à jour | T12 |

## À supprimer en fin de parcours (T17, après le test d'équivalence)

`old_structure/`, `jepa/` en entier (dont `JEPA`, `JEPAAction`, `JEPAInterleaved`, `trainer.py`), les anciens tests devenus sans objet, `LOGGING_STATUS.md` et `TESTING.md` s'ils décrivent l'ancienne API. Les YAML de `jepa/config/` utiles sont repris dans `jepa_tools/presets/` avec la clé `logging.wandb:` unifiée.

## Règles de structure

- Une pièce = un fichier dans `slots/<emplacement>/`, avec `@register(...)`. Aucun autre fichier à modifier : `slots/__init__.py` découvre les modules.
- `core/` ne dépend d'aucune pièce concrète. `train/` ne connaît que l'API de `JEPAModule`.
- `eval/` et `data/` ne dépendent que de `core/types.py` et des interfaces.
- Dépendances optionnelles (torchvision, timm, wandb, tensorboard, matplotlib, transformers) : importées dans les fonctions, jamais au niveau du module.
