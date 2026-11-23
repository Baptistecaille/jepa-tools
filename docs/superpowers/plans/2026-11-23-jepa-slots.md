# JEPA par emplacements — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un nouveau paquet `jepa_tools` où un JEPA se construit en choisissant une pièce par emplacement depuis un YAML, avec trois presets (`ijepa`, `temporal`, `lejepa`), évaluation et garde-fous intégrés.

**Architecture:** Contrat de données `Plan` / `StepOutputs`, cinq types de pièces enregistrées par décorateur, un `JEPAModule` qui les assemble, une boucle d'entraînement mince (`fit`) qui ne connaît que l'API du module. L'ancien paquet `jepa` reste en place jusqu'au test d'équivalence `temporal`, puis est supprimé (Task 17).

**Tech Stack:** PyTorch, pyyaml, numpy, pytest (CPU). torchvision, timm, wandb, tensorboard, matplotlib, transformers : optionnels, jamais importés à l'import.

**Spec:** `docs/superpowers/specs/2026-11-23-jepa-slots-design.md`

**Ordre des jalons (écart au spec §7, sur demande « images d'abord »)** : hygiène → cœur → `ijepa` sur images avec sonde → `temporal` + équivalence + suppression → `lejepa` → conformité → nettoyage. Le test d'équivalence reste obligatoire avant toute suppression.

## Global Constraints

- PyTorch. Python 3.10 ou plus. Dépendances obligatoires minimales : torch, pyyaml, numpy.
- timm, transformers, wandb, tensorboard, matplotlib sont optionnels et jamais importés à l'import du paquet.
- Licence MIT de l'amont conservée, avec attribution (fork de dipsivenkatesh/jepa).
- Tous les tests passent sur CPU en moins de quelques minutes. GPU requis seulement pour les runs réels.
- DDP via `torchrun`. Un seul GPU est le cas de référence.
- Nom du paquet : `jepa_tools`. Pas de rétrocompatibilité avec l'ancienne API.
- Interfaces de pièces : cinq méthodes au plus. Registre par décorateur seulement.
- Une spec de run (YAML ou dataclass) détermine entièrement le run et est sauvegardée avec chaque checkpoint.
- Environnement de départ : `torch` n'est pas installé dans le conteneur (Task 1 l'installe en version CPU).

## Review Focus

Entrées que le spec n'exerce pas mais qu'un utilisateur rencontrera, de la plus probable à la moins probable. Chaque ligne a son test dans la tâche indiquée.

1. Clé inconnue dans le YAML ou dans une surcharge CLI (`encoder.dpeth=6`) : erreur qui nomme la clé fautive, jamais ignorée en silence (Task 3).
2. Nom de pièce inconnu (`encoder: {name: vitt}`) : erreur qui liste les noms disponibles (Task 2).
3. Image dont la taille n'est pas multiple de `patch_size`, ou batch de taille 1 : erreur claire ou exécution correcte, jamais un crash de forme obscur (Tasks 5, 15).
4. Masque multi-blocs : le contexte ne doit jamais contenir un token cible, et `ctx_idx`/`tgt_idx` ne sont jamais vides (Task 8).
5. Reprise depuis un checkpoint : la perte du pas suivant est identique à celle d'un run non interrompu (RNG, EMA, optimiseur restaurés) (Task 12).

---

### Task 1: Hygiène, environnement, squelette du paquet

**Files:**
- Modify: `jepa/trainer/utils.py:8`, `jepa/cli/evaluate.py:16`, `jepa/data/utils.py:12` (déplacer `import matplotlib.pyplot` dans les fonctions qui l'utilisent)
- Modify: `pyproject.toml`, `setup.py`, `MANIFEST.in` (paquet `jepa_tools` ajouté à côté de `jepa`, `requires-python >=3.10`, dépendances réduites à torch/numpy/pyyaml, extras `vision`, `logging`, `hf`, `plot`, `dev`)
- Create: `jepa_tools/__init__.py` (`__version__ = "0.2.0.dev0"`), `NOTICE.md` (attribution amont + licence MIT)
- Test: `tests/test_imports.py`

**Interfaces:**
- Produces: paquet importable `jepa_tools`; dépendances optionnelles chargées paresseusement.

- [ ] **Step 1: Installer l'environnement** : `pip install torch --index-url https://download.pytorch.org/whl/cpu`, puis `pip install -e ".[dev]"`. Si l'index est bloqué par le proxy, lire `/root/.ccr/README.md`.
- [ ] **Step 2: Test** `test_import_is_light` : lance un sous-processus `python -c "import jepa_tools, jepa; import sys; print([m for m in ('matplotlib','wandb','tensorboard','transformers','timm') if m in sys.modules])"` et vérifie que la liste imprimée est `[]`. Test `test_all_names_exist` : pour chaque nom de `jepa.__all__`, `hasattr(jepa, nom)`.
- [ ] **Step 3: Lancer** `pytest tests/test_imports.py -v` ; attendu : FAIL (matplotlib importé, `jepa_tools` absent).
- [ ] **Step 4: Implémenter** les imports paresseux et le squelette ; corriger `__all__` si `test_all_names_exist` échoue (le désaccord `wandb:` / `logging.wandb:` des YAML est traité en Task 17).
- [ ] **Step 5: Lancer** `pytest tests -q` ; attendu : PASS, sans régression sur les tests existants.
- [ ] **Step 6: Commit** `chore: lazy optional imports, jepa_tools skeleton, python>=3.10`.

### Task 2: Types et registre

**Files:**
- Create: `jepa_tools/core/__init__.py`, `jepa_tools/core/types.py`, `jepa_tools/core/registry.py`, `jepa_tools/slots/__init__.py`
- Test: `tests/core/test_registry.py`

**Interfaces:**
- Produces (`types.py`) : `Plan(ctx_inputs: Tensor, tgt_inputs: Tensor, ctx_idx: Tensor | None, tgt_idx: Tensor | None)` et `StepOutputs(pred: Tensor, target: Tensor, emb: Tensor, progress: float)`, tous deux `@dataclass` exactement comme au spec §5.1.
- Produces (`registry.py`) : `KINDS = ("sampler","encoder","target","predictor","distance","regularizer")` ; `register(kind: str, name: str) -> Callable[[T], T]` ; `get(kind: str, name: str) -> type` ; `names(kind: str) -> list[str]` ; `class RegistryError(ValueError)`.
- Produces (`slots/__init__.py`) : `discover() -> None`, appelée à l'import, qui importe tous les sous-modules de `jepa_tools.slots` via `pkgutil.walk_packages`, de sorte qu'un nouveau fichier de pièce s'enregistre sans toucher au cœur.

- [ ] **Step 1: Tests** : `test_register_and_get` (enregistre une classe factice, `get` la renvoie, `names` la liste) ; `test_unknown_name_lists_available` (`pytest.raises(RegistryError, match="vit")` quand on demande `vitt` avec `vit` enregistré) ; `test_duplicate_name_rejected` ; `test_unknown_kind_rejected`.
- [ ] **Step 2: Lancer** `pytest tests/core/test_registry.py -v` ; attendu : FAIL.
- [ ] **Step 3: Implémenter** types, registre (dict de dicts) et `discover`.
- [ ] **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(core): data contract and slot registry`.

### Task 3: Spec de run, presets, surcharges

**Files:**
- Create: `jepa_tools/core/spec.py`, `jepa_tools/presets/__init__.py` (vide, pour le packaging ; les YAML sont ajoutés par Tasks 13, 14, 15)
- Test: `tests/core/test_spec.py`

**Interfaces:**
- Consumes: rien.
- Produces: `@dataclass RunSpec` avec `preset: str | None`, `data: dict`, `sampler: dict`, `encoder: dict`, `target: dict`, `predictor: dict`, `objective: dict`, `trainer: dict`, `allow_collapse_risk: bool = False`, méthodes `to_dict() -> dict` et classmethod `from_dict(d: dict) -> RunSpec`.
- Produces: `load_spec(source: str | os.PathLike | dict, overrides: Sequence[str] = ()) -> RunSpec`. Le preset nommé est chargé depuis `jepa_tools/presets/<preset>.yaml`, la spec utilisateur est fusionnée par-dessus (fusion récursive, la spec gagne), puis les surcharges `a.b.c=valeur` (valeur parsée par `yaml.safe_load`).

- [ ] **Step 1: Tests** : `test_roundtrip` (`from_dict(spec.to_dict()) == spec`) ; `test_override_nested` (`encoder.depth=6` change `encoder["depth"]` en `int` 6) ; `test_unknown_top_level_key_names_it` (`match="sampelr"`) ; `test_unknown_override_path_names_it` (`encoder.dpeth=6` n'est accepté que si la clé existe déjà dans la spec fusionnée ou si `encoder` est vide ; sinon `ValueError` citant `dpeth`) ; `test_user_spec_beats_preset` avec un preset écrit dans `tmp_path` et résolu via un paramètre `preset_dir` de `load_spec`.
- [ ] **Step 2: Lancer** ; attendu : FAIL.
- [ ] **Step 3: Implémenter** `RunSpec` et `load_spec`.
- [ ] **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(core): run spec with presets and overrides`.

### Task 4: Objectif (distances, régularisateurs)

**Files:**
- Create: `jepa_tools/slots/objective/__init__.py`, `distances.py`, `regularizers.py`, `objective.py`
- Test: `tests/slots/test_objective.py`

**Interfaces:**
- Consumes: `StepOutputs`, `register`.
- Produces: distances enregistrées (`kind="distance"`) `l2`, `smooth_l1`, `cosine`, chacune une classe appelable `(pred: Tensor, target: Tensor) -> Tensor` scalaire sur des tenseurs `(B, Nt, D)`.
- Produces: régularisateurs (`kind="regularizer"`) `none` (renvoie 0) et `vicreg` (appelable `(out: StepOutputs) -> Tensor`, aplatit `out.emb` en `(B*N, D)` puis réutilise la fonction de variance et covariance de `jepa/loss_functions/loss_functions.py:23` — copier la logique, ne pas importer l'ancien paquet).
- Produces: `Objective(distance: str = "l2", regularizer: str = "none", distance_weight: float = 1.0, reg_weight: float = 1.0, **reg_kwargs)` avec `forward(out: StepOutputs) -> tuple[Tensor, dict[str, float]]` ; `logs` contient `loss`, `distance`, `reg`. Propriété `has_regularizer: bool`.

- [ ] **Step 1: Tests** : `l2` sur `pred == target` vaut 0 ; `cosine` vaut 0 sur des vecteurs colinéaires et 2 sur des vecteurs opposés ; `vicreg` est plus grand sur des embeddings constants que sur du bruit gaussien ; `Objective("l2","vicreg").forward` renvoie `loss == distance*wd + reg*wr` (à `1e-6`) ; `Objective(...).has_regularizer` est `False` avec `none`.
- [ ] **Step 2: Lancer** ; attendu : FAIL.
- [ ] **Step 3: Implémenter** ; `Objective` résout les noms via `registry.get`.
- [ ] **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(slots): objective with distances and vicreg`.

### Task 5: Encodeurs `mlp` et `vit`

**Files:**
- Create: `jepa_tools/slots/encoder/__init__.py`, `mlp.py`, `vit.py`
- Test: `tests/slots/test_encoder.py`

**Interfaces:**
- Produces: `Encoder` = `nn.Module` avec `forward(x: Tensor, keep_idx: LongTensor | None = None) -> Tensor (B, N, D)` et attribut `dim: int`.
- Produces: `@register("encoder","mlp") class MLPEncoder(in_dim: int, dim: int, hidden: int = 256, depth: int = 2)` ; aplatit l'entrée, renvoie `(B, 1, dim)` ; `keep_idx` ignoré ; expose `self.net: nn.Sequential` (Linear/GELU) — le test d'équivalence de la Task 14 en dépend.
- Produces: `@register("encoder","vit") class ViTEncoder(image_size: int, patch_size: int, dim: int, depth: int, heads: int = 6, in_chans: int = 3)` ; embedding de patchs + positions apprises (sans CLS) ; `keep_idx (B, Nc)` sélectionne les tokens **après** ajout de la position, avant les blocs ; attributs `num_tokens` et `grid: tuple[int, int]` ; `ValueError` si `image_size % patch_size != 0` (message citant les deux valeurs).

- [ ] **Step 1: Tests** : sortie `(2, 16, 32)` pour `image_size=32, patch_size=8, dim=32` ; avec `keep_idx` de longueur 5, sortie `(2, 5, 32)` ; avec `keep_idx`, vérifier seulement la forme et l'absence de NaN (les valeurs diffèrent d'un passage complet, l'attention ne voit que les tokens gardés) ; `test_bad_patch_size_message` ; batch de taille 1 fonctionne ; `MLPEncoder` renvoie `(B, 1, dim)`.
- [ ] **Step 2: Lancer** ; attendu : FAIL.
- [ ] **Step 3: Implémenter** ViT minimal maison (blocs pré-norm `nn.TransformerEncoderLayer(norm_first=True, batch_first=True)` suffisent).
- [ ] **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(slots): mlp and vit encoders`.

### Task 6: Prédicteurs `identity`, `mlp`, `narrow_vit`

**Files:**
- Create: `jepa_tools/slots/predictor/__init__.py`, `simple.py`, `narrow_vit.py`
- Test: `tests/slots/test_predictor.py`

**Interfaces:**
- Produces: `Predictor.forward(ctx_tokens: Tensor (B,Nc,D), ctx_idx: LongTensor | None, tgt_idx: LongTensor | None, cond: Tensor | None = None) -> Tensor (B,Nt,D)`. Si `cond` n'est pas `None`, lever `NotImplementedError("cond reserved")`.
- Produces: `@register("predictor","identity") IdentityPredictor()` (renvoie `ctx_tokens`) ; `@register("predictor","mlp") MLPPredictor(dim: int, hidden: int = 256, depth: int = 2)` appliqué token par token, expose `self.net: nn.Sequential` ; `@register("predictor","narrow_vit") NarrowViTPredictor(in_dim: int, dim: int, depth: int, num_tokens: int, heads: int = 6)` : projette `in_dim → dim`, ajoute les positions de `ctx_idx`, ajoute des tokens masque avec les positions de `tgt_idx`, applique les blocs, ne garde que les positions cibles, reprojette vers `in_dim`.

- [ ] **Step 1: Tests** : `narrow_vit` avec `Nc=6, Nt=4` renvoie `(B,4,in_dim)` ; changer `tgt_idx` change la sortie (les tokens masque dépendent de la position) ; `cond=torch.zeros(1)` lève `NotImplementedError` ; `identity` renvoie l'entrée ; `mlp` conserve la forme.
- [ ] **Step 2: Lancer** ; attendu : FAIL. **Step 3: Implémenter.** **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(slots): identity, mlp and narrow_vit predictors`.

### Task 7: Cibles `ema`, `shared`, `frozen`

**Files:**
- Create: `jepa_tools/slots/target/__init__.py`, `target.py`
- Create: `jepa_tools/train/__init__.py`, `jepa_tools/train/schedules.py`
- Test: `tests/slots/test_target.py`, `tests/train/test_schedules.py`

**Interfaces:**
- Produces (`schedules.py`) : `momentum_at(progress: float, start: float, end: float) -> float` (interpolation linéaire) ; `cosine_with_warmup(step: int, total: int, warmup: int, base: float, final: float = 0.0) -> float`.
- Produces: `Target` avec `encode(plan: Plan, online_encoder: nn.Module) -> Tensor (B,Nt,D)`, `update(progress: float) -> None`, `eval_encoder() -> nn.Module`.
- `@register("target","ema") EMATarget(online_encoder: nn.Module, momentum: tuple[float, float] = (0.996, 1.0), layernorm: bool = True)` : copie profonde gelée de l'encodeur en ligne ; `encode` passe `plan.tgt_inputs` en entier sans `keep_idx`, applique `F.layer_norm` si `layernorm`, puis sélectionne `plan.tgt_idx` avec `gather` ; `update` fait l'EMA avec `momentum_at(progress, *momentum)` ; `eval_encoder()` renvoie la copie EMA.
- `@register("target","shared") SharedTarget(online_encoder, detach: bool = True)` : mêmes passes que ci-dessus via l'encodeur en ligne ; `.detach()` si `detach` ; `update` ne fait rien ; `eval_encoder()` renvoie l'encodeur en ligne.
- `@register("target","frozen") FrozenTarget(online_encoder, path: str | None = None)` : copie gelée, `requires_grad=False`, charge `path` si fourni ; `update` ne fait rien.

- [ ] **Step 1: Tests** : après `update(progress=1.0)` avec momentum `(0.5, 1.0)`, les poids EMA sont inchangés (momentum 1.0) ; avec `update(0.0)` les poids EMA valent `0.5*ema + 0.5*online` ; `EMATarget.encode` ne produit aucun gradient pour l'encodeur en ligne ; `SharedTarget(detach=False)` propage le gradient, `detach=True` non ; `tgt_idx=None` renvoie tous les tokens ; `momentum_at(0.5, 0.996, 1.0) == pytest.approx(0.998)` ; `cosine_with_warmup(0, 100, 10, 1.0) == 0.0` et `cosine_with_warmup(10, 100, 10, 1.0) == pytest.approx(1.0)`.
- [ ] **Step 2: Lancer** ; attendu : FAIL. **Step 3: Implémenter.** **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(slots): ema, shared and frozen targets with schedules`.

### Task 8: Échantillonneurs `multiblock` et `temporal_pair`

**Files:**
- Create: `jepa_tools/slots/sampler/__init__.py`, `multiblock.py`, `temporal_pair.py`
- Test: `tests/slots/test_sampler.py`

**Interfaces:**
- Produces: `Sampler.__call__(batch: dict[str, Tensor]) -> Plan`. Les batchs sont des `dict` ; chaque sampler documente les clés qu'il lit.
- `@register("sampler","multiblock") MultiBlockSampler(grid: tuple[int,int], n_targets: int = 4, target_scale: tuple[float,float] = (0.15, 0.2), target_aspect: tuple[float,float] = (0.75, 1.5), context_scale: tuple[float,float] = (0.85, 1.0), generator: torch.Generator | None = None)` lit `batch["x"]` ; `ctx_inputs is tgt_inputs` ; `tgt_idx` = union des `n_targets` blocs, `ctx_idx` = un bloc de contexte **moins** tous les tokens de `tgt_idx` ; tous les échantillons d'un batch ont la même longueur (tronquer au minimum du batch).
- `@register("sampler","temporal_pair") TemporalPairSampler()` lit `batch["x_t"]`, `batch["x_t1"]` ; `Plan(ctx_inputs=x_t, tgt_inputs=x_t1, ctx_idx=None, tgt_idx=None)`.
- Les paramètres `grid` et `generator` sont fournis par `build` (Task 9), pas par le YAML.

- [ ] **Step 1: Tests** : sur 200 tirages avec `grid=(8,8)`, `set(ctx_idx) ∩ set(tgt_idx) == ∅` pour chaque échantillon ; `ctx_idx.shape[1] > 0` et `tgt_idx.shape[1] > 0` toujours, y compris sur `grid=(2,2)` (le sampler doit dégrader proprement, jamais produire un index vide) ; deux appels avec le même `generator` seedé donnent les mêmes indices ; `temporal_pair` renvoie `ctx_idx is None`.
- [ ] **Step 2: Lancer** ; attendu : FAIL. **Step 3: Implémenter.** **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(slots): multiblock and temporal_pair samplers`.

### Task 9: `JEPAModule`, `build`, garde-fous

**Files:**
- Create: `jepa_tools/core/module.py`, `jepa_tools/core/build.py`
- Modify: `jepa_tools/__init__.py` (exporter `build`, `load_spec`, `JEPAModule`)
- Test: `tests/core/test_module.py`

**Interfaces:**
- Consumes: Tasks 2 à 8.
- Produces: `class JEPAModule(nn.Module)` avec `__init__(sampler, encoder, target, predictor, objective, spec: RunSpec)` ; `training_step(batch: dict, progress: float) -> tuple[Tensor, dict[str, float]]` (sampler → `encoder(plan.ctx_inputs, plan.ctx_idx)` → `target.encode` sous `torch.no_grad()` sauf si la cible laisse passer le gradient → `predictor` → `objective(StepOutputs)`) ; `after_step(progress: float) -> None` (appelle `target.update`) ; `eval_encoder() -> nn.Module` ; `save_pretrained(directory: str) -> None` (poids + `config.json` contenant `spec.to_dict()`) ; classmethod `from_pretrained(directory: str) -> JEPAModule` qui reconstruit via `build`.
- Produces: `build(spec: RunSpec) -> JEPAModule` : résout chaque pièce par le registre (`spec.encoder["name"]`, etc.), injecte `grid` (issu de l'encodeur) dans le sampler et `num_tokens` dans `narrow_vit`, puis lance les deux validations ci-dessous.
- Garde-fou collapse : si la cible est `shared` avec `detach=True` et `objective.has_regularizer` est `False`, `build` lève `ValueError` dont le message contient `allow_collapse_risk` ; avec `spec.allow_collapse_risk=True`, il émet un `warnings.warn` et continue.
- Garde-fou formes : `build` exécute un `training_step` sur un mini-batch fictif fabriqué par `JEPAModule.dummy_batch(batch_size=2) -> dict` (délégué à l'attribut `example_batch` que chaque spec `data` produit via `data.example_batch`, voir Task 11) ; toute exception de forme est relevée en `ValueError("shape check failed: ...")` avec la pièce fautive nommée.

- [ ] **Step 1: Tests** : `build` d'une spec `ijepa` miniature (écrite dans le test, `vit` dim 32 depth 2, `narrow_vit`, `ema`, `multiblock`, `l2`) puis `training_step` renvoie une perte finie et `loss.backward()` donne un gradient non nul à l'encodeur ; `shared + detach + regularizer none` lève `ValueError` avec `allow_collapse_risk` dans le message ; la même spec avec `allow_collapse_risk=True` passe et émet un warning ; `save_pretrained` puis `from_pretrained` redonne des poids identiques (`torch.equal`) et la même `spec` ; `after_step` change les poids EMA ; une spec où `narrow_vit.in_dim != encoder.dim` échoue avec « shape check failed » et le mot `predictor`.
- [ ] **Step 2: Lancer** ; attendu : FAIL.
- [ ] **Step 3: Implémenter.** Le test peut fournir `example_batch` directement à `build` via un argument `example_batch: dict | None = None` tant que la Task 11 n'existe pas ; Task 11 branche ensuite le `DataModule`.
- [ ] **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(core): JEPAModule, build and collapse guard`.

### Task 10: Diagnostics et sonde linéaire

**Files:**
- Create: `jepa_tools/eval/__init__.py`, `diagnostics.py`, `probe.py`
- Test: `tests/eval/test_diagnostics.py`, `tests/eval/test_probe.py`

**Interfaces:**
- Produces: `collapse_stats(z: Tensor) -> dict[str, float]` sur `(M, D)` avec les clés `std_mean`, `effective_rank` (exp de l'entropie des valeurs singulières normalisées), `mean_cos` (cosinus moyen hors diagonale) ; `check_collapse(stats: dict[str, float], min_std: float = 0.01) -> str | None` renvoie un message d'alerte quand `std_mean < min_std`.
- Produces: `extract_features(encoder: nn.Module, loader: Iterable[dict], device: str = "cpu") -> tuple[Tensor, Tensor]` (pooling moyen sur N, encodeur en `eval` et `no_grad`) ; `linear_probe(encoder, train_loader, val_loader, num_classes: int, epochs: int = 100, lr: float = 1e-2, device: str = "cpu") -> float` renvoie l'exactitude de validation (régression logistique entraînée sur features gelées).
- Les batchs lus par la sonde sont `{"x": Tensor, "y": LongTensor}`.

- [ ] **Step 1: Tests** : bruit gaussien `(512, 32)` donne `effective_rank > 25` ; matrice constante donne `std_mean < 1e-6` et `check_collapse` renvoie un message ; matrice de rang 1 donne `effective_rank < 1.5` et `mean_cos > 0.9` ; `linear_probe` sur un encodeur identité et des classes linéairement séparables atteint `> 0.95` ; la sonde ne modifie pas les poids de l'encodeur (`state_dict` identique avant et après).
- [ ] **Step 2: Lancer** ; attendu : FAIL. **Step 3: Implémenter.** **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(eval): collapse diagnostics and linear probe`.

### Task 11: Données images

**Files:**
- Create: `jepa_tools/data/__init__.py`, `images.py`
- Test: `tests/data/test_images.py`

**Interfaces:**
- Produces: `DataModule` avec `train_loader() -> Iterable[dict]`, `val_loader() -> Iterable[dict]`, `num_classes: int`, `example_batch(batch_size: int = 2) -> dict` ; fabrique `make_data(data_spec: dict) -> DataModule`.
- `name: fake_images` : jeu synthétique déterministe (`n_train`, `n_val`, `image_size`, `num_classes`, `seed`) où la classe est déterminée par la position et la couleur de grands blobs, de sorte qu'un encodeur appris puisse faire mieux que des poids aléatoires ; batchs `{"x": (B,3,H,W) float, "y": (B,) long}`.
- `name: cifar10` : `torchvision.datasets.CIFAR10` importé paresseusement ; `ImportError` explicite (« pip install jepa-tools[vision] ») si torchvision est absent ; ne télécharge que si `download: true`.

- [ ] **Step 1: Tests** : `fake_images` est reproductible (deux instanciations avec la même `seed` donnent des tenseurs égaux) ; formes et dtypes des batchs ; `example_batch` renvoie `x` de forme `(2,3,H,W)` ; `make_data({"name":"cifar10",...})` sans torchvision lève `ImportError` citant `jepa-tools[vision]` (test sauté si torchvision est installé).
- [ ] **Step 2: Lancer** ; attendu : FAIL. **Step 3: Implémenter.** **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5:** brancher `make_data(spec.data).example_batch` dans `build` (Task 9) et ajouter un test de `build` sans argument `example_batch`.
- [ ] **Step 6: Commit** `feat(data): image datamodules (fake_images, cifar10)`.

### Task 12: Boucle d'entraînement, checkpoint, reprise, loggers, DDP

**Files:**
- Create: `jepa_tools/train/loop.py`, `checkpoint.py`, `dist.py`
- Move: `git mv jepa/loggers jepa_tools/loggers`, puis remplacer les imports de `jepa/` par `jepa_tools.loggers` (chercher avec grep `loggers`) et mettre à jour `tests/test_logging.py`
- Test: `tests/train/test_loop.py`, `tests/train/test_checkpoint.py`

**Interfaces:**
- Produces (`checkpoint.py`) : `save_checkpoint(path: str, module: JEPAModule, optimizer, step: int, scaler=None) -> None` (spec via `module.spec.to_dict()`, états du module et de l'optimiseur, `step`, RNG torch CPU/CUDA et numpy/python) ; `load_checkpoint(path: str, module: JEPAModule, optimizer, scaler=None) -> int` restaure tout et renvoie `step`.
- Produces (`dist.py`) : `setup_dist() -> tuple[int, int]` (rang, taille ; `(0, 1)` hors `torchrun`) ; `cleanup_dist() -> None`.
- Produces (`loop.py`) : `fit(module: JEPAModule, data: DataModule, trainer_spec: dict, loggers: BaseLogger | None = None, resume: str | None = None, out_dir: str = "runs/default") -> dict[str, float]` ; clés lues de `trainer_spec` : `epochs`, `lr`, `weight_decay`, `warmup_steps`, `clip_grad`, `amp` (`"none"|"bf16"`), `diag_every`, `ckpt_every`. `fit` : lr via `cosine_with_warmup`, AMP bf16 par `torch.autocast`, clipping, `module.after_step(progress)` après chaque pas, `collapse_stats` sur les embeddings loggés tous les `diag_every` pas avec `check_collapse` (alerte via `warnings.warn` et log), enveloppe DDP quand la taille est supérieure à 1. `fit` ne dispatche jamais sur la forme du batch.

- [ ] **Step 1: Tests** : `fit` sur une spec `ijepa` miniature et `fake_images` pendant 2 époques réduit la perte moyenne de l'époque 2 sous celle de l'époque 1 ; **reprise** : entraîner 6 pas en s'arrêtant au pas 3 avec checkpoint, reprendre, et comparer la perte du pas 4 à celle d'un run ininterrompu (`torch.testing.assert_close`, mêmes seeds) ; le checkpoint contient la clé `spec` ; `fit` avec `amp: bf16` tourne sur CPU ; `setup_dist()` hors `torchrun` renvoie `(0, 1)`.
- [ ] **Step 2: Lancer** ; attendu : FAIL.
- [ ] **Step 3: Implémenter.** Le seed des données doit dépendre de `(seed, step)` pour que la reprise soit exacte.
- [ ] **Step 4: Lancer** `pytest tests -q` (inclut les tests de loggers déplacés) ; attendu : PASS.
- [ ] **Step 5: Commit** `feat(train): thin fit loop, checkpoint/resume, ddp, loggers moved`.

### Task 13: CLI et preset `ijepa` sur images, avec sonde

**Files:**
- Create: `jepa_tools/cli.py`, `jepa_tools/presets/ijepa.yaml`
- Modify: `pyproject.toml` (`[project.scripts] jepa-train = "jepa_tools.cli:main"`)
- Test: `tests/test_ijepa_e2e.py`

**Interfaces:**
- Consumes: Tasks 3, 9, 10, 11, 12.
- Produces: `main(argv: Sequence[str] | None = None) -> int` : `jepa-train spec.yaml key.sub=value ...` ; charge la spec, `build`, `fit`, puis lance la sonde linéaire sur `module.eval_encoder()` et écrit `probe_acc` dans `out_dir/metrics.json`. Option `--out-dir`.
- `ijepa.yaml` : valeurs du spec §5.2 réduites pour CPU par défaut (le YAML de référence garde `cifar10`, `vit dim 384 depth 12`, etc. ; les tests surchargent en `fake_images`, `dim=64 depth=2`, 20 époques).

- [ ] **Step 1: Test** `test_ijepa_probe_beats_random` : spec `preset: ijepa` avec surcharges miniatures sur `fake_images` ; entraîner via `main`, comparer `probe_acc` de l'encodeur entraîné à celle d'un `ViTEncoder` initialisé au hasard (mêmes dimensions, seed différent) ; assert `trained > random + 0.05`. Si le test est instable, rendre `fake_images` plus discriminant ou allonger l'entraînement — jamais baisser la marge. Test `test_cli_unknown_key` : `main(["spec.yaml","encoder.dpeth=6"])` lève `ValueError` citant `dpeth`.
- [ ] **Step 2: Lancer** ; attendu : FAIL.
- [ ] **Step 3: Implémenter** CLI et preset ; fixer les hyperparamètres des tests dans le test, pas dans le preset.
- [ ] **Step 4: Lancer** ; attendu : PASS (critère de réussite du spec §7 sur la sonde).
- [ ] **Step 5: Commit** `feat: ijepa preset and jepa-train CLI with probe`.

### Task 14: Preset `temporal` et équivalence numérique

**Files:**
- Create: `jepa_tools/presets/temporal.yaml`, `jepa_tools/data/temporal.py` (jeu jouet `toy_temporal` produisant `{"x_t","x_t1"}`)
- Test: `tests/test_temporal_equivalence.py`

**Interfaces:**
- `temporal.yaml` : `sampler: temporal_pair`, `encoder: mlp`, `target: {name: shared, detach: true}`, `predictor: mlp`, `objective: {distance: l2, regularizer: vicreg}`.

- [ ] **Step 1: Test** `test_matches_legacy_jepa` : construire `torch.manual_seed(0)`, deux `MLPEncoder`/`MLPPredictor` ; construire l'ancien `jepa.models.JEPA(encoder.net, predictor.net)` avec des `deepcopy` des `net` (les deux `nn.Sequential`) ; sur un même batch `x_t, x_t1`, comparer `pred` et `target` de l'ancien `forward` à ceux du nouveau module (spec `allow_collapse_risk=True`, régularisateur `none`) et la perte au `jepa.loss_functions.mse_loss(pred, target)` ; `torch.testing.assert_close` avec `rtol=1e-6`. Second test : avec `vicreg`, la perte vaut `mse + reg_weight * vicreg_loss` calculé à la main sur `emb` aplati.
- [ ] **Step 2: Lancer** ; attendu : FAIL. **Step 3: Implémenter** preset et jeu jouet ; ne toucher ni au cœur ni à l'ancien code. **Step 4: Lancer** ; attendu : PASS. Si l'écart persiste, comparer la convention de pooling avant de modifier un slot (l'ancien `JEPA` aplatit les sorties >2D).
- [ ] **Step 5: Commit** `feat: temporal preset with legacy equivalence test`.

### Task 15: SIGReg, multi-vues et preset `lejepa`

**Files:**
- Create: `jepa_tools/slots/objective/sigreg.py` (**seul fichier de régularisateur modifié ou ajouté**), `jepa_tools/slots/sampler/multiview.py`, `jepa_tools/presets/lejepa.yaml`
- Test: `tests/slots/test_sigreg.py`, `tests/test_lejepa_e2e.py`

**Interfaces:**
- `@register("regularizer","sigreg") SIGReg(num_slices: int = 256, t_max: float = 5.0, n_points: int = 17)` appelable `(out: StepOutputs) -> Tensor` ; test d'Epps-Pulley sur projections aléatoires unitaires de `out.emb` aplati `(B*N, D)` : fonction caractéristique empirique comparée à `exp(-t²/2)` sur une grille de `t`, intégrée avec poids gaussien, multipliée par le nombre d'échantillons, moyennée sur les tranches. **Valider contre le dépôt de référence de LeJEPA (arXiv 2511.08544) avant de figer** : l'outil `mcp__alphaXiv__read_files_from_github_repository` peut lire son code ; noter dans le docstring les écarts éventuels.
- `@register("sampler","multiview") MultiViewSampler(n_views: int = 2, ...)` lit `batch["views"]` (liste de `(B,C,H,W)` produite par le DataModule via augmentations) ; `Plan(ctx_inputs=view0, tgt_inputs=view1, ctx_idx=None, tgt_idx=None)`.
- `lejepa.yaml` : `sampler: multiview`, `target: {name: shared, detach: false}`, `predictor: identity`, `objective: {distance: l2, regularizer: sigreg}`. `fake_images` accepte `views: 2` et renvoie `batch["views"]` avec des perturbations aléatoires (bruit, retournement).

- [ ] **Step 1: Tests** : SIGReg sur `torch.randn(2048, 64)` est plus petit que sur la même matrice multipliée par 0.01, que sur une matrice constante, et que sur une matrice de rang 1 ; la valeur est finie et sans NaN pour `B=2` (batch minimal) ; le gradient existe ; `test_sigreg_is_a_single_file` vérifie via `registry.names("regularizer")` que `sigreg` apparaît après avoir importé `jepa_tools.slots` **sans** import explicite du fichier (découverte automatique) ; e2e `lejepa` : le preset `lejepa` avec `allow_collapse_risk` absent passe `build` (le régularisateur est présent), `fit` tourne et `collapse_stats(...)["effective_rank"] > 2` sur les embeddings après entraînement ; et le preset `lejepa` avec `regularizer: none` est refusé par `build` (garde-fou de la Task 9 ne s'applique que pour `detach=true` : ajouter ici un test documentant que `shared + detach=false + none` est autorisé mais loggue un warning de risque — choix de l'auteur du plan, à relire).
- [ ] **Step 2: Lancer** ; attendu : FAIL. **Step 3: Implémenter.** **Step 4: Lancer** ; attendu : PASS.
- [ ] **Step 5: Commit** `feat: sigreg, multiview sampler and lejepa preset`.

### Task 16: Suite de conformité sur le registre

**Files:**
- Create: `tests/conformance/test_registry_conformance.py`

**Interfaces:**
- Consumes: `names(kind)` et les constructeurs de toutes les tâches précédentes.

- [ ] **Step 1: Test paramétré** par `pytest.mark.parametrize` sur `(kind, name)` pour chaque pièce de `registry.names(kind)` : construit la pièce avec des paramètres par défaut de test (table `DEFAULTS[(kind, name)]` dans le fichier de test ; une pièce enregistrée sans entrée dans la table fait échouer un test `test_every_piece_has_conformance_defaults`) ; vérifie la forme de sortie sur un mini-batch `(B=2, N=16, D=32)` et l'absence de NaN ; pour les encodeurs, vérifie la sortie `(B,N,D)` avec et sans `keep_idx` ; pour les samplers, vérifie que `ctx_idx`/`tgt_idx` sont `None` ou de forme `(B, n)` avec `n > 0`.
- [ ] **Step 2: Lancer** `pytest tests/conformance -v` ; attendu : PASS.
- [ ] **Step 3: Commit** `test: registry conformance suite`.

### Task 17: Nettoyage, suppression de l'ancien code, documentation

**Files:**
- Delete: `old_structure/`, `jepa/models/jepa.py` (`JEPA`, `JEPAAction`, `JEPAInterleaved`), `jepa/trainer/trainer.py`, puis le reste de `jepa/` une fois les reprises faites ; les tests de l'ancien paquet qui ne couvrent plus rien
- Move: `jepa/models/base.py` → `jepa_tools/core/persistence.py` (logique `save_pretrained` / `from_pretrained` réutilisée par `JEPAModule` si elle ne l'est pas déjà)
- Modify: les YAML (`jepa/config/*.yaml` reportés dans `jepa_tools/presets/` si utiles) pour uniformiser `wandb:` vs `logging.wandb:` en `logging.wandb:` ; `README.md`, `CHANGELOG.md`, `pyproject.toml` (nom du projet `jepa-tools`, retrait du paquet `jepa`), `examples/` (un exemple par preset), `NOTICE.md`
- Test: `tests/test_no_legacy.py`

- [ ] **Step 1: Test** `test_legacy_package_gone` : `importlib.util.find_spec("jepa") is None` ; `test_no_wandb_key_conflict` : tous les YAML sous `jepa_tools/presets/` n'ont aucune clé `wandb:` au niveau racine.
- [ ] **Step 2: Lancer** ; attendu : FAIL.
- [ ] **Step 3: Vérifier avant de supprimer** que `pytest tests/test_temporal_equivalence.py` passe, puis supprimer par `git rm` (l'historique git conserve l'ancien code). Mettre à jour README (quick start : `jepa-train` avec chacun des trois presets), CHANGELOG (rupture d'API assumée), attribution MIT.
- [ ] **Step 4: Lancer** `pytest tests -q` ; attendu : PASS, tout sur CPU en quelques minutes. **Step 5: Commit** `refactor!: remove legacy jepa package, docs and examples for jepa_tools`.

---

## Auto-revue

- **Couverture du spec** : §5.1 Tasks 2, 8 ; §5.2 Tasks 3, 9 ; §5.3 Task 12 ; §5.4 Tasks 9, 10, 12 ; §6 Tasks 1, 12, 17 ; critères §7 : trois presets (13, 14, 15), régularisateur en un fichier (15), `shared + detach + mse` refusé (9), sonde bat aléatoire (13), conformité (16). V-JEPA et actions : hors périmètre, `cond` réservé (Task 6).
- **Cohérence** : `build` (Task 9) injecte `grid` et `num_tokens` ; `Objective.has_regularizer` (Task 4) alimente le garde-fou (Task 9) ; `example_batch` (Tasks 9, 11) ; `.net` des `mlp` (Tasks 5, 6) alimente l'équivalence (Task 14).
- **Risques du spec traités** : SIGReg validé contre le dépôt de référence (Task 15) ; contrat `Plan` éprouvé sur images à la Task 13 avant d'aller plus loin.
- **Point à relire par Baptiste** : la décision en fin de Task 15 sur `shared + detach=false + none` (autorisé avec warning, car le spec ne couvre que `detach=true`).
