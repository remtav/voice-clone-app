# Cloner sa voix québécoise avec Qwen3-TTS : guide pas à pas

Ce guide reproduit la démarche qui a donné une voix clonée jugée juste à l'écoute (accent,
timbre et intonations) pour un balado d'information en français québécois, en octobre 2026.
Aucun entraînement : le modèle Qwen3-TTS reste tel quel, et la voix tient dans **un extrait
de référence et sa transcription**.

Compter une heure, enregistrement compris, plus le temps de construire l'image Docker.

## Ce qui a marché, et ce qui n'a pas marché

| Essai | Verdict à l'écoute |
|---|---|
| Voix « dessinée » par une description (Qwen3-TTS VoiceDesign) | Échec : ni l'accent ni les intonations |
| Clonage, extrait de 25 s lu posément | Accent et timbre justes, **intonations trop plates** |
| **Clonage, extrait de 55 s lu de façon vivante, avec sa transcription, température 0,9** | **Retenu** |
| Même chose, température 1,1 | Plus variée à la mesure, mais la version à 0,9 a été préférée à l'écoute |

Ce qu'il faut en retenir :

- **La transcription de l'extrait est indispensable.** Avec elle, Qwen3-TTS continue l'extrait
  comme si la même personne poursuivait : il copie l'accent et la mélodie des phrases. Sans
  elle, il ne copie que le timbre.
- **L'extrait compte plus que les réglages.** Le modèle imite les intonations qu'on lui donne :
  un extrait long et vivant (questions, insistance, vraies pauses) donne une voix vivante.
- **Les mesures automatiques ne remplacent pas l'oreille.** Un détecteur d'accent et une note
  de naturel prédite jugeaient excellente la voix dessinée, rejetée à l'écoute. Décidez à l'oreille.

## 1. Enregistrer l'extrait de référence

- **Où** : la pièce la plus calme et la plus meublée possible (salon, chambre; pas cuisine ni
  salle de bain), fenêtres fermées, appareils bruyants éteints.
- **Avec quoi** : un téléphone suffit, à environ 20 cm de la bouche, un peu de côté.
- **Combien** : 45 à 60 secondes. Pas plus de 60 (voir l'étape 3).
- **Comment** : comme si vous expliquiez la nouvelle à un ami au café. Montez sur les
  questions, appuyez sur les mots importants, prenez de vraies pauses. Mieux vaut un peu plus
  animé que d'habitude que trop posé : le modèle a tendance à aplatir. Gardez votre accent
  naturel, sans le forcer ni le gommer. Faites deux ou trois prises, gardez celle où vous
  sonnez le plus vous-même.

Texte utilisé (environ 55 s; remplacez les tournures qui ne vous ressemblent pas, ou écrivez
un texte du même genre sur votre propre sujet) :

> Bon. Vous avez sûrement entendu parler du projet de centre de données à Sherbrooke.
> Peut-être que vous avez signé la pétition. Peut-être que vous vous dites : « Voyons donc,
> c'est quoi le problème? » Ben, on va regarder ça ensemble.
>
> D'abord, une question simple : c'est quoi, un centre de données? C'est un gros bâtiment
> rempli d'ordinateurs. Pis ces ordinateurs-là, ils chauffent. Beaucoup.
>
> Maintenant, le chiffre qui fait jaser : quatre-vingt-seize mégawatts. Est-ce que c'est
> beaucoup? Oui. C'est à peu près le quart de toute l'électricité vendue par Hydro-Sherbrooke
> dans une année. Le quart!
>
> Mais attention. Rien n'est décidé. La Ville n'a pas reçu de projet officiel, pis Québec non
> plus. Alors avant de vous faire une idée, prenez deux minutes. Regardez les sources. Vous
> allez voir, c'est pas si simple que ça.

## 2. Préparer l'extrait et sa transcription

L'application n'enlève pas les silences et ne règle pas le volume à l'importation, et le
modèle copie tout ce que contient l'extrait. Préparez-le d'abord :

```bash
python3 -m venv .venv-prep && . .venv-prep/bin/activate
pip install numpy soundfile faster-whisper      # ffmpeg doit aussi être installé
python -m scripts.prepare_reference mon-enregistrement.m4a voix/reference.wav --transcribe
```

Le script écrit `voix/reference.wav` (mono 24 kHz, silences retirés, volume normalisé) et
`voix/reference.txt`, transcrit par un Whisper ajusté au français québécois. Il affiche aussi
le **plancher de bruit** : visez moins de −55 dBFS (l'extrait retenu était à −54; plus bas
est mieux).

**Relisez `reference.txt` en réécoutant l'extrait, et corrigez-le mot pour mot.** Whisper
résume parfois (« pis » devient « puis », un mot saute) ou se trompe (« sources à l'appui »
était devenu « c'est une source à l'appui »). Une transcription inexacte fait lire de travers.
Écrivez les nombres comme vous les dites : « quatre-vingt-seize », pas « 96 ».

## 3. Lancer l'application avec le moteur Qwen3-TTS

Dans `.env` :

```bash
TTS_ENGINE=qwen
ENGINE_REQUIREMENTS=requirements-engine-qwen.txt
MAX_REFERENCE_SECONDS=60
```

Puis :

```bash
docker compose build app      # image distincte de celle de Chatterbox
docker compose up -d app
docker compose logs -f app    # premier démarrage : téléchargement du modèle (~4,3 Go)
```

La pastille d'état affiche `Ready · Qwen/Qwen3-TTS-12Hz-1.7B-Base`. Un extrait plus long que
`MAX_REFERENCE_SECONDS` serait coupé, et sa transcription ne correspondrait plus : l'app le
refuse donc quand une transcription l'accompagne.

## 4. Ajouter la voix

Dans l'interface, carte **Add a voice** → onglet **Upload** → `reference.wav`, langue
**French**, et collez le contenu de `reference.txt` dans **Transcript**. Ou en ligne de
commande :

```bash
curl -c jar -X POST localhost:8000/api/login -H 'content-type: application/json' -d '{"password":"…"}'
curl -b jar -F file=@voix/reference.wav -F name="Ma voix" -F language=fr \
     -F transcript="$(cat voix/reference.txt)" localhost:8000/api/voices
```

Pour corriger la transcription d'une voix existante :
`curl -b jar -X PATCH localhost:8000/api/voices/<id> -H 'content-type: application/json' -d '{"transcript":"…"}'`.

## 5. Générer

- Langue **French**, **temperature 0.9** (dans *Advanced*). `exaggeration` et `cfg_weight`
  ne s'appliquent pas à ce moteur.
- Fixez un **seed** pour pouvoir reproduire une prise; changez-le pour en obtenir une autre.
- Commencez par **deux ou trois paragraphes** : c'est assez pour juger l'accent et les
  intonations, sans attendre la production d'un texte entier.
- Un texte long est découpé en morceaux d'au plus `MAX_CHUNK_CHARS` caractères (300 par
  défaut), aux fins de phrase; chaque paragraphe commence un nouveau morceau. Séparez vos
  paragraphes par une ligne vide.

### Écrire pour l'oreille

Ces écueils, relevés en produisant un épisode de 5 minutes, ne dépendent pas de la voix :

- **Nombres et unités en toutes lettres** : « quatre-vingt-seize mégawatts », pas « 96 MW ».
- **Sigles** : écrivez-les comme ils se disent (« le Bape », « I-A »). En capitales, « BAPE »
  était épelé de travers.
- **Noms étrangers** : écrivez-les selon la prononciation voulue (« Kil » pour Keel), puis
  vérifiez qu'ils ne forment pas un homophone. « Kil a deux options » s'entendait « qu'il a deux
  options » : reformulez (« Keel Infrastructure a deux options », « L'entreprise voudrait… »).
- **Écoutez tout** avant de publier. Une prise ratée se régénère avec un autre seed.

## 6. Conserver la voix

La voix, c'est `reference.wav`, `reference.txt` et les réglages (modèle
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, température 0,9). Gardez-les ensemble, avec l'enregistrement
original, **en privé** : ils suffisent à faire dire n'importe quoi à votre voix. Ne les
versionnez pas dans un dépôt public. Le dossier `data/` de l'app (ignoré par git) contient
aussi la voix importée.

## 7. Si ce n'est pas encore assez bon

Dans l'ordre :

1. **Réenregistrer** : pièce plus calme, lecture plus vivante, 45 à 60 s. C'est le levier qui
   a fait la plus grande différence.
2. **Ajuster la température** entre 0,8 et 1,1 (plus haut : plus varié, mais moins sûr).
3. **Ajustement fin (fine-tuning)** de Qwen3-TTS sur 30 à 60 minutes de votre voix, avec la
   procédure officielle de Qwen pour une voix unique, sur un ordinateur avec carte graphique.
   Gardez dès maintenant vos enregistrements dans les mêmes conditions : ce seront les données
   d'entraînement. (Le fine-tuning de l'interface web entraîne des checkpoints Chatterbox; il
   ne s'applique pas à ce moteur.)
