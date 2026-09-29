# Plan d'implémentation — fine-tune LoRA `fr-CA` du T3 Chatterbox Multilingual v3

> Document de handoff : chaque phase est exécutable indépendamment par un agent/modèle
> qui n'a pas lu la discussion d'origine. Les chemins, numéros de ligne et commandes ont été
> vérifiés contre le commit Chatterbox épinglé dans `requirements-engine.txt`
> (`5de7a54aa4e5e2baadb0182dde554908b48b85c2`) et contre les dépôts externes cités.

## 0. Contexte et décision

**Problème.** Une voix québécoise clonée avec Chatterbox Multilingual (`language_id="fr"`)
ressort avec un accent de France. Cause : l'accent est porté par le *prior* du token de langue
`fr`, pas par le clip de référence.

**Faits vérifiés qui dictent la solution**

| Fait | Source |
|---|---|
| Seules les **6 premières secondes** de la référence alimentent le prompt in-context du T3 (`ENC_COND_LEN = 6 * S3_SR`) ; 10 s pour le timbre côté décodeur (`DEC_COND_LEN`). | `src/chatterbox/mtl_tts.py:156-157, 259, 266` |
| `cfg_weight` contrôle le transfert d'accent de la référence (« set `cfg_weight` to `0` » pour *éviter* l'accent de la référence → le monter pour le *garder*). | README Chatterbox, section « Original Chatterbox Tips » |
| Resemble traite les accents régionaux par **fine-tune du T3** : *Single Language Pack* `es-es` / `es-mx-latam`, `pt-pt` / `pt-br`. Un pack = un seul `t3_<locale>.safetensors` (292 tenseurs, float32, ~2,14 Go) + décodeur partagé, même `language_id`. **Aucun pack français n'existe.** | HF `ResembleAI/Chatterbox-Multilingual-pt-br` (model card) ; listing HF `author=ResembleAI` |
| `from_local(ckpt_dir, device, t3_model=...)` accepte **n'importe quel nom `.safetensors`**, y compris un **chemin absolu** (`Path(ckpt_dir) / "/abs/x.safetensors"` → `/abs/x.safetensors`). Chargement `load_state_dict` **strict**. | `mtl_tts.py:58-67, 183-207` |
| Un corpus QC existe, déjà préparé et **CC0** : ~10 000 clips, 1,77 Go, 24 kHz mono, silences coupés, 1–15 s, 281 locuteurs (~12 h), extrait de Common Voice fr filtré sur l'accent « Québécois / Canadien ». | HF `tontate/f5-tts-quebec-french-finetune` → `dataset/processed/` + `scripts/prep_common_voice_qc.py` |
| Un fine-tune F5-TTS sur ce corpus a fait passer P(accent QC) de 0,17 → 0,40 et la similarité locuteur de 0,48 → 0,68, WER 0,067 → 0,077. Preuve que 12 h suffisent pour déplacer l'accent. | même model card, section « Evaluation » |
| Toolkit LoRA pour le T3 multilingue : `lora.py` (LoRA maison sur `q/k/v/o/gate/up/down` de `t3.tfmr`, fusion propre dans `weight.data`) + `fix_merged_model.py` (→ safetensors). **Par défaut il charge v2 et hardcode `language_id='ar'`.** | GitHub `Ahmed-Ezzat20/chatterbox-finetuning-multilingual` — `lora.py:393, 739, 915, 941, 1079` |

**Décision.** Produire `t3_fr_ca.safetensors`, un T3 v3 fusionné LoRA entraîné sur le corpus QC,
chargé par l'app via une simple variable d'env. Licences : Chatterbox MIT, corpus CC0 →
usage commercial possible. Le moteur VC (`app/engines/vc.py`) reste un squelette hors périmètre.

## 1. Vue d'ensemble

| Phase | Livrable | Dépend de | Effort estimé |
|---|---|---|---|
| 0 — App : checkpoint T3 custom + UX accent | `CHATTERBOX_T3_MODEL`, presets UI, tests | — | 2–3 h |
| 1 — Données | `data/finetune/qc/audio_data/{metadata.csv,audio/}` | — | 1 h (option A) à ½ j (option B) |
| 2 — Entraînement LoRA + fusion | `t3_fr_ca.safetensors` validé strictement | 1 | 2 h setup + 3–6 h GPU |
| 3 — Évaluation | rapport base vs fr-ca (WER, sim, P(QC), ABX) | 0, 2 | ½ j dev + 1 h run |
| 4 — Ta propre voix (optionnel) | `t3_fr_ca_<slug>.safetensors` | 2, 3 | 1–2 h enregistrement + 2–3 h correction + 1–2 h GPU |
| 5 — Déploiement | checkpoint dans le volume `/data`, `.env`, README | 0, 3 | 1 h |

Les phases 0 et 1 sont indépendantes et parallélisables. Matériel : 1 GPU ≥ 16 Go VRAM
(24 Go recommandé), ~10 Go disque libre.

Convention : tout nouveau script vit sous `scripts/finetune/` (paquet Python, `__init__.py`),
respecte `ruff` (`make lint`, line-length 120) et n'est **pas** importé par l'app.

---

## 2. Phase 0 — App : charger un T3 custom et exposer les bons réglages

### 2.1 Config — `app/config.py`

- Ajouter au dataclass `Settings` : `chatterbox_t3_model: str = "v3"`.
- `from_env` : `chatterbox_t3_model=os.environ.get("CHATTERBOX_T3_MODEL", "v3")`.
- `__post_init__` : `.strip()` seulement (ne pas lowercaser : c'est potentiellement un chemin).
- Sémantique : `"v2"` / `"v3"` = checkpoints officiels ; toute valeur finissant par `.safetensors`
  = chemin de fichier, **absolu ou relatif à `data_dir`**. Ajouter une propriété :

```python
@property
def chatterbox_t3_path(self) -> Path | None:
    """Chemin résolu d'un T3 custom, ou None pour un checkpoint officiel (v2/v3)."""
    value = self.chatterbox_t3_model
    if not value.endswith(".safetensors"):
        return None
    path = Path(value)
    return path if path.is_absolute() else self.data_dir / path
```

### 2.2 Registre — `app/engines/__init__.py`

Passer le réglage au moteur :
`ChatterboxEngine(variant=..., device=..., t3_model=settings.chatterbox_t3_model, t3_path=settings.chatterbox_t3_path)`.

### 2.3 Moteur — `app/engines/chatterbox.py`

- `__init__` : stocker `self.t3_model` et `self.t3_path`. Si `t3_path` est fourni et que
  `variant != "multilingual"`, lever `ValueError` (un T3 custom n'a de sens qu'en multilingue).
- `load()`, branche `multilingual`, remplacer le bloc `try/except TypeError` actuel (lignes 66-73) par :

```python
from chatterbox.mtl_tts import REPO_ID, ChatterboxMultilingualTTS

if self.t3_path is not None:
    if not self.t3_path.is_file():
        raise FileNotFoundError(f"CHATTERBOX_T3_MODEL points to a missing file: {self.t3_path}")
    from huggingface_hub import snapshot_download

    # Assets partagés (encodeur de voix, décodeur, tokenizer, voix intégrée) : mêmes fichiers que
    # from_pretrained(), sans le T3 officiel.
    ckpt_dir = snapshot_download(
        repo_id=REPO_ID, repo_type="model", revision="main",
        allow_patterns=["ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
        token=os.getenv("HF_TOKEN"),
    )
    # from_local() accepte un chemin absolu pour t3_model (Path / "/abs" == "/abs").
    self.model = ChatterboxMultilingualTTS.from_local(ckpt_dir, self.device, t3_model=str(self.t3_path.resolve()))
    self.model_id = f"ResembleAI/chatterbox (multilingual, custom T3 {self.t3_path.name})"
else:
    try:
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device, t3_model=self.t3_model)
        self.model_id = f"ResembleAI/chatterbox (multilingual {self.t3_model})"
    except TypeError:  # chatterbox-tts PyPI 0.1.7 : pas de sélecteur t3_model
        log.warning("Installed chatterbox-tts has no t3_model selector; using its default checkpoint")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self.model_id = "ResembleAI/chatterbox (multilingual, default)"
```

- `info()` : ajouter `"t3_model": self.t3_model` (et `"t3_path": str(self.t3_path)` si défini).
  `/api/config` expose déjà `engine.info()` → l'UI peut afficher le checkpoint actif.
- `scripts/download_models.py` fonctionne tel quel (il appelle `engine.load()`).

### 2.4 UI — presets et indice « 6 secondes »

`app/static/index.html`

- Lignes 88-103 (sliders `#exaggeration`, `#cfg`, `#temp`) : ajouter au-dessus une rangée de
  boutons `button.preset[data-exag][data-cfg][data-temp]` :
  - **Neutre** — 0.5 / 0.5 / 0.8 (défauts actuels)
  - **Accent fidèle** — 0.4 / 0.8 / 0.6 (colle à la référence : c'est le réglage à utiliser
    pour un accent régional)
  - **Expressif** — 0.7 / 0.3 / 0.8 (déjà documenté dans le texte d'aide)
- Texte d'aide (l. 102-103) : ajouter une phrase : *« Seules les 6 premières secondes de la
  référence pilotent la prononciation et l'accent ; placez-y les traits les plus marqués. »*
- Section d'enregistrement de la référence : même indice à côté de `#max-ref-seconds`
  (`app.js:101` l'alimente).

`app/static/app.js`

- Près de la boucle des sliders (l. ~597) : handler `click` sur `.preset` qui écrit les trois
  valeurs et leurs `<output>` (réutiliser la logique des l. 514-516).
- Afficher `engine.t3_model` dans le bandeau statut si présent (l'objet `engine` de
  `/api/config` est déjà dans `state.config`).

Ne pas changer `MAX_REFERENCE_SECONDS` (les 10 s du décodeur et l'encodeur de voix profitent
du reste du clip).

### 2.5 Documentation — `.env.example`, `README.md`

Bloc `# --- Engine ---` : ajouter

```
# v2, v3, ou chemin d'un T3 fine-tuné (.safetensors), absolu ou relatif à DATA_DIR,
# ex. models/t3_fr_ca.safetensors  (voir docs/finetune-fr-ca-plan.md)
CHATTERBOX_T3_MODEL=v3
```

README : sous-section « Checkpoint T3 personnalisé (accent régional) » renvoyant à ce document.

### 2.6 Tests (`tests/`)

Nouveau `tests/test_config.py` :
- `CHATTERBOX_T3_MODEL` absent → `chatterbox_t3_model == "v3"`, `chatterbox_t3_path is None`.
- `"v2"` → path `None`.
- `"models/t3_x.safetensors"` → `chatterbox_t3_path == data_dir / "models/t3_x.safetensors"`.
- `"/abs/t3_x.safetensors"` → path absolu inchangé.

Dans `tests/test_api.py` ou nouveau `tests/test_engines.py` : construire
`ChatterboxEngine(variant="multilingual", t3_model="x.safetensors", t3_path=Path("/nope/x.safetensors"))`
sans `load()` → `info()["t3_model"]` correct ; `ChatterboxEngine(variant="turbo", t3_path=...)` → `ValueError`.
(Aucun test ne doit importer `chatterbox` : les imports restent lazy dans `load()`.)

### 2.7 Critères d'acceptation

- `make test` et `make lint` verts.
- Sur GPU : sans variable → comportement identique à aujourd'hui (`model_id` contient `v3`).
- Sur GPU : copier le v3 officiel (`hf_hub_download("ResembleAI/chatterbox", "t3_mtl23ls_v3.safetensors")`)
  vers `data/models/t3_test.safetensors`, `CHATTERBOX_T3_MODEL=models/t3_test.safetensors` →
  l'app charge, `/api/config` montre `custom T3 t3_test.safetensors`, une génération FR réussit.
- Fichier manquant → erreur de chargement lisible dans `/api/status.load_error`.

---

## 3. Phase 1 — Données

Cible : `data/finetune/qc/audio_data/` au format attendu par `lora.py` :

```
audio_data/
├── metadata.csv          # file_name,transcription,duration_seconds[,client_id]
└── audio/xxx.wav         # WAV mono (le toolkit rééchantillonne lui-même à 24 kHz / 16 kHz)
```

### 3.1 Option A (recommandée) — corpus QC déjà préparé (CC0)

> **Implémenté** : `scripts/finetune/prepare_qc_dataset.py` (sous-commandes `download`,
> `from-processed`, `from-common-voice`). Commandes réelles :

```bash
pip install -U huggingface_hub soundfile
python -m scripts.finetune.prepare_qc_dataset download --dest data/finetune/qc_src   # révision HF épinglée
python -m scripts.finetune.prepare_qc_dataset from-processed \
  --src data/finetune/qc_src/dataset/processed --out data/finetune/qc/audio_data
```

La spécification d'origine suit.

Format source : `dataset/processed/metadata.csv`, séparateur `|`, en-tête `audio_file|text`,
chemins absolus `/workspace/data/processed/wavs/qc/common_voice_fr_<id>.wav` → ne garder que
le **basename**. ~9 992 wav / 1,77 Go.

Script à écrire : `scripts/finetune/prepare_qc_dataset.py --src data/finetune/qc_src/dataset/processed --out data/finetune/qc/audio_data`
1. Lire le CSV (`csv` avec `delimiter="|"`, `quoting=csv.QUOTE_NONE`).
2. Pour chaque ligne : résoudre `wavs/qc/<basename>` ; ignorer si absent ; lire la durée
   (`soundfile.info`) ; garder `1.0 ≤ durée ≤ 15.0` ; ignorer texte `< 3` caractères.
3. Normalisation texte **minimale** : espaces, apostrophes typographiques `’` → `'`,
   guillemets, `…` → `...`. **Ne pas** « corriger » les formes québécoises (`pis`, `tsé`, `icitte`) :
   c'est l'orthographe que le modèle doit associer à la prononciation.
4. Lien symbolique ou copie vers `audio/<basename>`, ligne
   `audio/<basename>,<transcription>,<durée>` (échapper avec `csv.writer`, `QUOTE_MINIMAL`).
5. Réserver 40 clips (hash stable du basename) dans `holdout.csv` — **jamais** utilisés à
   l'entraînement, utilisés comme références/réel QC en phase 3.
6. Afficher : nb clips, heures totales, histogramme de durées.

Limite de l'option A : pas de `client_id` → pas d'équilibrage locuteur possible.

### 3.2 Option B — Common Voice complet + script de filtrage existant

Si l'on veut `client_id`, `up_votes/down_votes`, ou une version CV plus récente.
Télécharger « Common Voice Scripted Speech » fr (Mozilla Data Collective, compte + accord
CC0 ; ~1 210 h, plusieurs dizaines de Go), puis :

> **Implémenté** sans dépendre du script externe (ffmpeg au lieu de librosa) :
> `python -m scripts.finetune.prepare_qc_dataset from-common-voice --cv-dir <cv-corpus-fr> --out data/finetune/qc/audio_data`
> (filtres `--min-up-votes 2 --max-down-votes 0 --max-per-speaker 300` par défaut ; holdout par locuteur).

Le script filtre le champ `accents` (regex `qu[ée]b[ée]cois|canadien|canada|\bqc\b|montr[ée]al`,
insensible aux accents/casse), rééchantillonne à 24 kHz, coupe les silences, borne 1–15 s,
déduplique. Ajouter en amont un filtre `up_votes ≥ 2 and down_votes == 0` et un plafond
**≤ 300 clips par `client_id`** (évite qu'un locuteur domine l'accent appris). Puis `prepare_qc_dataset.py` comme en A.

### 3.3 Option C — flux HF sans téléchargement complet

`mozilla-foundation/common_voice_17_0` (dataset *gated* : accepter les conditions, `HF_TOKEN`),
config `fr`, `streaming=True`, filtrer `example["accent"]` avec la même regex, n'écrire que
les clips retenus. Même sortie qu'en B.

### 3.4 Critères d'acceptation

- `metadata.csv` valide (`pandas.read_csv` sans erreur, colonnes exactes), tous les fichiers
  existent, aucune durée hors bornes.
- ≥ 9 000 clips / ≥ 10 h après filtrage ; `holdout.csv` de 40 clips disjoint.
- Écoute aléatoire de 10 clips : accent QC net, transcription fidèle.

---

## 4. Phase 2 — Entraînement LoRA et fusion

> **Implémenté** — commandes réelles (environnement : `requirements-finetune.txt`) :
>
> ```bash
> python -m scripts.finetune.setup_toolkit --run-dir data/finetune/runs/fr_ca_r16 \
>     --data-dir data/finetune/qc/audio_data            # télécharge, vérifie SHA-256, patche
> python data/finetune/runs/fr_ca_r16/toolkit/lora.py   # entraînement (GPU)
> python data/finetune/runs/fr_ca_r16/toolkit/fix_merged_model.py
> python -m scripts.finetune.validate_t3_checkpoint \
>     data/finetune/runs/fr_ca_r16/merged_model/t3_fr_ca.safetensors --strict-load \
>     --smoke-reference data/finetune/qc/audio_data/audio/<clip du holdout>.wav
> python -m scripts.finetune.merge_adapter --adapter data/finetune/runs/fr_ca_r16/checkpoint_epoch1_stepN.pt \
>     --out data/finetune/runs/fr_ca_r16/t3_fr_ca_e1.safetensors   # export d'une époque intermédiaire
> ```
>
> **Deux écarts découverts en exécutant réellement le toolkit** (mini-entraînement CPU) :
>
> 1. **Cibles de parole sans BOS/EOS.** Le toolkit entraîne sur les tokens S3 bruts, sans
>    `start_speech_token` (6561) ni `stop_speech_token` (6562), alors que l'inférence démarre
>    sur l'un et s'arrête sur l'autre. `setup_toolkit` encadre désormais les cibles comme à
>    l'inférence (correct par construction, **non prouvé expérimentalement** ;
>    `--upstream-speech-targets` rétablit le cadrage amont pour comparer sur GPU).
>    Un mini-entraînement CPU volontairement agressif (LR 1e-3) produisait 40 s d'audio pour
>    une phrase de 7 mots, avec **et** sans ce correctif : c'est le LR qui casse le modèle.
>    D'où le contrôle ajouté à `validate_t3_checkpoint --smoke-reference`, qui échoue si le
>    débit tombe sous 3 caractères/s.
> 2. **`WARMUP_STEPS` n'est jamais utilisé** (le planificateur est un cosinus simple) : le
>    réglage `WARMUP_STEPS = 200` ci-dessous est sans effet et n'est pas exposé.
>
> Les patches effectivement appliqués sont documentés en tête de `scripts/finetune/setup_toolkit.py`.
> La spécification d'origine suit.

### 4.1 Environnement (séparé de l'app)

```bash
python3.11 -m venv .venv-ft && . .venv-ft/bin/activate
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2"
pip install pandas matplotlib tqdm soundfile
```

**Important** : installer Chatterbox depuis le **même commit que l'app** (v3 disponible,
mêmes `T3Config.multilingual()` et vocabulaires 2454 / 8194). **Ne pas** `pip install` le fork
du toolkit (il embarque un `chatterbox-tts` 0.1.4 antérieur à v3) : ne récupérer que ses deux scripts.

```bash
mkdir -p scripts/finetune/vendor && cd scripts/finetune/vendor
R=https://raw.githubusercontent.com/Ahmed-Ezzat20/chatterbox-finetuning-multilingual/<COMMIT>
curl -sSLO $R/lora.py && curl -sSLO $R/fix_merged_model.py
```

Épingler `<COMMIT>` (noter le hash dans `scripts/finetune/vendor/VERSION`). Les seuls imports
Chatterbox de `lora.py` (`ChatterboxMultilingualTTS`, `punc_norm`, `S3Gen`, `S3GEN_SR`, `S3_SR`,
`VoiceEncoder`, `MTLTokenizer`, `T3Cond`) existent tous au commit épinglé (vérifié dans `mtl_tts.py`).

### 4.2 Patches obligatoires de `lora.py`

Implémenter dans `scripts/finetune/patch_lora.py` (idempotent, échoue si un motif est introuvable)
ou appliquer à la main. Numéros de ligne au commit consulté — **rechercher par motif**, pas par numéro.

| # | Motif (ligne) | Remplacement | Pourquoi |
|---|---|---|---|
| 1 | `model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)` (l. 739) | `..., t3_model="v3")` | sinon le LoRA part de **v2** |
| 2 | `merged_model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)` (l. 915) | `..., t3_model="v3")` | la fusion doit repartir de la même base |
| 3 | `language_id: str = "ar"` (l. 393) et `language_id='ar'  # Arabic language ID` (l. 1079) | `"fr"` | token de langue des données |
| 4 | `merged_dir / "t3_mtl23ls_v2.pt"` (l. 941) | `merged_dir / "t3_fr_ca.pt"` | nom cohérent |
| 5 | Bloc de config (l. 38-53) | voir 4.3 | hyperparamètres |

`fix_merged_model.py` : remplacer les deux occurrences `t3_mtl23ls_v2` par `t3_fr_ca`
(entrée `.pt`, sortie `.safetensors`).

### 4.3 Hyperparamètres (bloc de config de `lora.py`)

```python
AUDIO_DATA_DIR = "data/finetune/qc/audio_data"
CHECKPOINT_DIR = "data/finetune/runs/fr_ca_r16"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8     # batch effectif 8
EPOCHS = 4                          # PAS 50 : la perte plafonne vers l'époque 2 sur ce corpus
LEARNING_RATE = 2e-5
WARMUP_STEPS = 200
LORA_RANK = 16                      # 32 si sous-apprentissage visible (P(QC) ne bouge pas)
LORA_ALPHA = 32                     # = 2 × rank
LORA_DROPOUT = 0.05
MAX_AUDIO_LENGTH = 15.0
MIN_AUDIO_LENGTH = 1.0
MAX_TEXT_LENGTH = 300
SAVE_EVERY_N_STEPS = 500
VALIDATION_SPLIT = 0.05
```

Cibles LoRA inchangées (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` du backbone
`t3.tfmr`) — les embeddings texte/parole ne sont pas entraînés, ce qui est voulu : le vocabulaire
graphémique couvre déjà le français.

### 4.4 Lancement

```bash
. .venv-ft/bin/activate
cd scripts/finetune/vendor && python lora.py 2>&1 | tee ../../../data/finetune/runs/fr_ca_r16/train.log
```

Suivre `training_metrics.png` (mis à jour en continu) : perte d'entraînement et de validation
décroissantes puis plateau ; pas de NaN. Ordre de grandeur : ~10 000 clips × 4 époques, batch 1
→ ~40 000 pas ; sur une 24 Go attendre **3–6 h**. OOM → `MAX_AUDIO_LENGTH = 12`, `LORA_RANK = 8`.

### 4.5 Fusion et conversion

À la fin, `lora.py` produit `merged_model/` (`ve.pt`, `t3_fr_ca.pt`, `s3gen.pt`, tokenizer,
`conds.pt`) et `final_lora_adapter.pt`. Puis :

```bash
python fix_merged_model.py          # → merged_model/t3_fr_ca.safetensors
```

Pour fusionner un checkpoint **intermédiaire** (choisi en phase 3), écrire
`scripts/finetune/merge_adapter.py --adapter <checkpoint_epochX_stepY.pt> --out t3_fr_ca_eX.safetensors` :
`from_pretrained(t3_model="v3")` → `inject_lora_layers(model.t3.tfmr, target_modules, rank, alpha, 0.0)`
→ `load_lora_adapter` → `merge_lora_weights` → `save_file(model.t3.state_dict(), out)`
(fonctions importées depuis `vendor/lora.py`). Les couches LoRA ne sont pas des sous-modules
enregistrés (monkeypatch de `forward`) : `t3.state_dict()` ne contient **que** les clés de base.

### 4.6 Validation stricte du checkpoint — `scripts/finetune/validate_t3_checkpoint.py`

```
python -m scripts.finetune.validate_t3_checkpoint data/finetune/runs/fr_ca_r16/merged_model/t3_fr_ca.safetensors
```

1. `base = load_file(hf_hub_download("ResembleAI/chatterbox", "t3_mtl23ls_v3.safetensors"))`.
2. Clés **identiques** (ensemble et shapes) ; aucune clé contenant `lora` ; dtype float32.
3. `T3(T3Config.multilingual()).load_state_dict(state, strict=True)` réussit.
4. Lister les tenseurs qui diffèrent de la base : doivent être exactement les `weight` des
   projections ciblées (`*.q_proj.weight`, …) — rien d'autre. Afficher `max |Δ|` et le nombre.
5. Taille ≈ 2,14 Go, 292 tenseurs (référence : pack `pt-br`).
6. Fumée GPU : `from_local(ckpt_dir_officiel, "cuda", t3_model=<abs>)` puis 3 phrases QC avec
   `language_id="fr"` et une référence du `holdout.csv` ; écrire les wav dans `runs/.../smoke/`.

### 4.7 Critères d'acceptation

- Validation 4.6 entièrement verte ; le fichier se charge dans l'app via la phase 0.
- Perte de validation finale < perte initiale ; pas de divergence.
- Fumée : intelligible, voix reconnaissable, accent QC audible sur au moins 2/3 phrases.

---

## 5. Phase 3 — Évaluation base v3 vs `fr-ca`

> **Implémenté** — `scripts/finetune/eval/` (dépendances : section « Phase 3 » de `requirements-finetune.txt`) :
>
> ```bash
> E=data/finetune/eval/fr_ca_r16
> # négatifs de la sonde : clips européens (Common Voice complet requis)
> python -m scripts.finetune.prepare_qc_dataset from-common-voice --accent europe \
>     --cv-dir <cv-corpus-fr> --out data/finetune/eu/audio_data
> python -m scripts.finetune.eval.accent_probe train --out data/finetune/eval/accent_probe.npz \
>     --positive data/finetune/qc/audio_data --negative data/finetune/eu/audio_data   # exige ≥ 0,8 d'exactitude
> python -m scripts.finetune.eval.battery --out $E --checkpoint base=v3 \
>     --checkpoint fr_ca=data/finetune/runs/fr_ca_r16/merged_model/t3_fr_ca.safetensors \
>     --voice me=<ta référence>.wav --voice qc1=<holdout> --voice qc2=<holdout>   # 600 clips, reprise possible
> python -m scripts.finetune.eval.score --battery $E --probe data/finetune/eval/accent_probe.npz
> python -m scripts.finetune.eval.abx make --battery $E --candidate fr_ca --voice me   # puis remplir abx/pairs.csv
> python -m scripts.finetune.eval.report --battery $E    # report.md + checkpoint recommandé (exit 1 si aucun)
> ```
>
> Plusieurs époques : ajouter un `--checkpoint fr_ca_e1=...` par export de `merge_adapter`, le rapport
> les compare tous à la base, CFG par CFG. Relancer `score` ne calcule que les colonnes manquantes.
> La batterie (`sentences_qc.tsv`) écrit les nombres en toutes lettres ; la normalisation WER
> (`eval/text.py`) épelle les chiffres de l'ASR et ramène le lexique QC à sa forme standard des deux côtés.

### 5.1 Matériel de test — `scripts/finetune/eval/`

- `sentences_qc.txt` : **50 phrases** ciblant les traits québécois. Exemples à compléter :
  affrication (« Tu dis que la petite est partie. »), voyelles relâchées (« Vite, toute la
  ville dort. »), /a/ vs /ɑ/ (« Il a mis la pâte sur la patte du chat. »), diphtongues
  (« Mon père fête ça au bord de la mer. »), lexique/élisions (« Faque j'ai pris mon char pis
  chu allé au dépanneur, tsé. »), nombres (« Ça coûte quatre-vingt-dix-sept dollars et
  trente-cinq. »), particules (« Tu viens-tu souper icitte à soir ? »), longues voyelles
  (« Une belle grosse fête plate. »).
- 3 voix de référence : la tienne + 2 locuteurs du `holdout.csv` (≥ 6 s, accent marqué au début).
- 4 conditions : {base v3, fr-ca} × {cfg 0.5, cfg 0.8}, exag 0.4, temp 0.6, seed fixe.
  → 4 × 3 × 50 = 600 clips (`generate_battery.py`, réutiliser `ChatterboxEngine` de l'app).

### 5.2 Métriques — `scripts/finetune/eval/score.py`

| Métrique | Outil | Détail |
|---|---|---|
| WER / CER | `faster-whisper` large-v3, `language="fr"` | normaliser les deux côtés : minuscules, ponctuation retirée, nombres → mots (`num2words`, `lang="fr"`) ; **exclure** du WER un petit lexique QC que l'ASR standardise (`pis→puis`, `tsé→tu sais`, `chu→je suis`, `faque→ça fait que`) via table de mapping appliquée aux deux côtés |
| Similarité locuteur | ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) | cosinus entre clip généré et référence, moyenne par condition |
| Sonde d'accent | `facebook/wav2vec2-xls-r-300m`, moyenne temporelle de la couche ~12 → `sklearn.LogisticRegression` | entraîner sur clips CV **réels** : QC (holdout + 1 000 clips d'entraînement) vs européens (1 000 clips fr taggés France/Belgique/Suisse, via option B/C ou flux HF) ; exiger une exactitude tenue-à-part ≥ 0,8 avant de l'utiliser ; rapporter la **moyenne de P(QC)** par condition |
| ABX humain | 10 paires aléatoires base vs fr-ca, même phrase/voix, ordre masqué | le locuteur cible juge « lequel a l'accent le plus proche du mien » |

Sortie : `report.md` avec un tableau conditions × métriques, et les 600 clips accessibles.

### 5.3 Go / no-go et sélection de checkpoint

- WER(fr-ca) ≤ WER(base) + 0,02 absolu ; CER idem.
- Similarité locuteur(fr-ca) ≥ base − 0,02.
- P(QC)(fr-ca, cfg 0.8) nettement > base (référence F5 : 0,17 → 0,40 ; viser ≥ 2× la base).
- ABX : ≥ 7/10 en faveur de fr-ca.
- Évaluer chaque checkpoint d'époque fusionné (4.5) ; retenir celui qui maximise P(QC) sous la
  contrainte WER. Si P(QC) stagne : rank 32, 6 époques, ou LR 3e-5. Si WER se dégrade :
  époque antérieure, ou LR 1e-5.

---

## 6. Phase 4 — Ta propre voix (identité + accent)

> **Implémenté** — commandes réelles :
>
> ```bash
> # 1. enregistrements longs (m4a, wav, webm...) → clips 2-12 s transcrits, 5 clips de holdout
> python -m scripts.finetune.segment_recording --out data/finetune/me/audio_data --speaker me recordings/*.m4a
> #    → RELIRE metadata.csv / holdout.csv : remettre pis, chu, faque, tsé... là où Whisper a standardisé
> # 2. tes clips ×3 + 30 % du corpus QC
> python -m scripts.finetune.build_personal_dataset --own data/finetune/me/audio_data \
>     --qc data/finetune/qc/audio_data --out data/finetune/me_mix/audio_data
> # 3. seconde étape, repartie du fr-ca retenu en phase 3
> python -m scripts.finetune.setup_toolkit --run-dir data/finetune/runs/fr_ca_me \
>     --data-dir data/finetune/me_mix/audio_data --output-name t3_fr_ca_me \
>     --base-t3 data/finetune/runs/fr_ca_r16/merged_model/t3_fr_ca.safetensors --lr 1e-5 --epochs 2
> python data/finetune/runs/fr_ca_me/toolkit/lora.py && python data/finetune/runs/fr_ca_me/toolkit/fix_merged_model.py
> python -m scripts.finetune.validate_t3_checkpoint data/finetune/runs/fr_ca_me/merged_model/t3_fr_ca_me.safetensors \
>     --base data/finetune/runs/fr_ca_r16/merged_model/t3_fr_ca.safetensors --strict-load \
>     --smoke-reference data/finetune/me/audio_data/audio/<un de tes clips de holdout>.wav
> ```
>
> `--base-t3` sert aussi à l'entraînement **et** à la fusion : le résultat est `fr-ca + LoRA perso`.
> Le holdout de tes clips se fait clip par clip (un seul locuteur). La validation interne du toolkit
> tire au hasard parmi des doublons suréchantillonnés : sa perte de validation est optimiste ; juger
> sur l'évaluation de la phase 3 (voix `me` = un clip du holdout).

Optionnelle mais c'est le meilleur résultat possible : le modèle apprend **ta** prononciation.

1. **Enregistrer 30–60 min** : pièce calme, même micro, 24 kHz+ mono, registre naturel
   québécois. Contenu varié : narration, dialogue, nombres, questions, et **les formes que tu
   emploies vraiment** (`pis`, `faque`, `tsé`) — lues comme tu les dis.
2. **Transcrire** : `faster-whisper` large-v3 (`language="fr"`, `word_timestamps=True`) ;
   segmenter aux pauses en clips de 2–12 s (`scripts/finetune/segment_recording.py`,
   VAD Silero ou frontières de segments Whisper). **Relecture manuelle obligatoire** des
   transcriptions : l'ASR normalise le lexique QC (écrit « puis » quand tu dis « pis ») — remettre
   la forme prononcée. Coût : ~2–3 h pour 45 min d'audio.
3. **Jeu de données étape 2** : `audio_data_me/` = tes clips **suréchantillonnés ×3** + un
   sous-ensemble de 30 % du corpus QC (régularisation contre l'oubli). Colonne `client_id=me`.
4. **Entraîner depuis `fr-ca`** : dans `lora.py`, remplacer le `from_pretrained(... "v3")` de
   l'étape 1 par `from_local(<merged_model fr-ca>, device=DEVICE, t3_model="t3_fr_ca.safetensors")`
   (les deux occurrences), `LEARNING_RATE = 1e-5`, `EPOCHS = 2`, `LORA_RANK = 16`. Sortie :
   `t3_fr_ca_<slug>.safetensors` (validation 4.6, même base de comparaison = v3).
5. **Évaluer** (phase 3) sur ta voix : similarité locuteur attendue en forte hausse ; vérifier
   que les 2 voix de holdout ne se dégradent pas trop. Si elles se dégradent, garder **deux**
   checkpoints : `t3_fr_ca` (générique) et `t3_fr_ca_<slug>` (personnel) et choisir via
   `CHATTERBOX_T3_MODEL`. (Sélection par voix dans l'app : hors périmètre, voir §9.)
6. **Confidentialité** : le checkpoint personnel encode ta voix → hors git (`data/` est déjà
   ignoré), jamais publié.

---

## 7. Phase 5 — Déploiement

> **Implémenté** : procédure de déploiement / retour arrière dans le README (section *Regional
> accents*) ; les voix `fr` démarrent sur le preset *Faithful accent* tant que l'utilisateur n'a
> touché à aucun réglage (vérifié dans Chromium) ;
> `python -m scripts.finetune.model_card --checkpoint … --report … --out README.md` génère la fiche
> Hugging Face (métadonnées, SHA-256, rapport d'évaluation) et **refuse** tout checkpoint de seconde
> étape ou entraîné sur des clips `audio/own_*` (ta voix).

1. Copier le checkpoint retenu dans le volume : `data/models/t3_fr_ca.safetensors`
   (`docker-compose` monte `./data` sur `/data`, `DATA_DIR=/data`).
2. `.env` : `CHATTERBOX_T3_MODEL=models/t3_fr_ca.safetensors`.
3. `make models` (précharge) puis `make up` ; vérifier `/api/config → engine.model_id` et
   `/api/status.load_error == null`.
4. UI : preset **Accent fidèle** par défaut pour les voix `fr` (option : petite règle dans
   `app.js` quand la voix sélectionnée a `language == "fr"`).
5. Retour arrière : retirer la variable → v3 officiel.
6. Publication (facultatif, générique seulement) : dépôt HF `…/Chatterbox-Multilingual-fr-ca`,
   licence MIT, model card sur le modèle de `pt-br` (fichiers, locale `fr-CA`, `language_id: fr`,
   métadonnées du checkpoint : nb tenseurs, shapes d'embeddings 2454/8194, SHA256), données CC0
   créditées (Common Voice, script de préparation de `tontate`).

---

## 8. Risques et parades

| Risque | Parade |
|---|---|
| Le toolkit part de **v2** (défaut) → checkpoint incompatible/inférieur | patches #1 et #2 (§4.2) ; validation 4.6 compare aux clés/shapes de **v3** |
| Fork du toolkit installé comme paquet (`chatterbox-tts` 0.1.4, pré-v3) | n'installer que Chatterbox au commit de l'app ; copier uniquement `lora.py` + `fix_merged_model.py` |
| `load_state_dict` strict échoue | validation 4.6 avant tout déploiement ; jamais de clé `lora_*` |
| Le modèle ne s'arrête plus de parler (LR trop fort, ou fin de parole mal apprise) | contrôle de débit du fumage (§4.6) ; LR 2e-5 ; cadrage BOS/EOS (comparable via `--upstream-speech-targets`) |
| Parole lue Common Voice = prosodie plate → sortie moins expressive | peu d'époques, rank 16, LoRA sur l'attention/MLP seulement ; phase 4 ré-injecte de la parole naturelle |
| Un locuteur domine le corpus | option B : plafond par `client_id` ; sinon écoute d'échantillons |
| WER gonflé par la normalisation ASR du lexique QC | table de mapping des deux côtés (§5.2) ; juger surtout ΔWER, pas la valeur absolue |
| Dataset HF *gated* / téléchargement CV volumineux | option A (1,8 Go, sans compte) suffit pour la v1 |
| OOM 16 Go | `MAX_AUDIO_LENGTH=12`, `LORA_RANK=8`, accumulation 16 |
| Oubli des autres langues/accents | attendu et acceptable pour ce cas ; le v3 officiel reste sélectionnable |
| Filigrane PerTh | inchangé : appliqué par le décodeur, pas par le T3 |

## 9. Hors périmètre (idées de suite)

- Choix du checkpoint T3 **par voix** dans l'app (colonne `t3_model` sur `voices`, un moteur par
  checkpoint chargé à la demande — coûteux en VRAM : 2 T3 ≈ +2 Go chacun).
- Fine-tune du décodeur `s3gen` : inutile pour l'accent (il ne porte que le timbre/vocodage).
- Moteur VC (`app/engines/vc.py`) : conservé comme squelette, utile seulement si voix cible ≠ locuteur.

## 10. Checklist de fin

- [ ] Phase 0 fusionnée : tests + lint verts, `CHATTERBOX_T3_MODEL` documenté
- [ ] `data/finetune/qc/audio_data` ≥ 10 h, `holdout.csv` disjoint
- [ ] `t3_fr_ca.safetensors` validé strictement (§4.6) et fumée OK
- [ ] `report.md` : WER ≤ base + 0,02, sim ≥ base − 0,02, P(QC) ≥ 2× base, ABX ≥ 7/10
- [ ] Déployé via `.env`, retour arrière testé
- [ ] (optionnel) `t3_fr_ca_<slug>` personnel, hors git
