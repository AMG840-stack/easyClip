# VIBRA CLIP ENGINE

Upload une vidéo → clips verticaux 9:16 finis (transcription Whisper, détection des moments viraux par Claude,
recadrage centré sur le visage, sous-titres animés mot par mot, son normalisé, légendes et hashtags).

## Déploiement sur Render (≈10 min)
1. Pousse ce dossier sur un dépôt GitHub (privé).
2. Render → New → Blueprint → choisis le dépôt (`render.yaml` est détecté).
3. Renseigne `ANTHROPIC_API_KEY` et `APP_PASSWORD` (protège l’accès : identifiant libre, mot de passe = APP_PASSWORD).
4. Ouvre l’URL du service, glisse ta vidéo, clique sur « Générer ».

Plan `standard` (2 Go de RAM) recommandé pour Whisper `small`. Plan plus petit : `WHISPER_MODEL=base`.
Plus de précision : `WHISPER_MODEL=medium` (4 Go+).

## En local
    pip install -r requirements.txt   # + ffmpeg installé sur la machine
    ANTHROPIC_API_KEY=... uvicorn app:app --port 8000

## Réglages (variables d’environnement)
WHISPER_MODEL (small), CLAUDE_MODEL (claude-sonnet-5-5), APP_PASSWORD, DATA_DIR (/tmp/jobs).

## Limites actuelles
Un seul montage à la fois (file d’attente), stockage éphémère (télécharge le .zip), recadrage sur le visage
principal (un plan fixe par clip).
