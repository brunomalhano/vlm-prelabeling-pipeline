# Public Repository Preparation Guide

This guide turns `platform/vlm-pipeline` into a reusable standalone public repository.

## Target repository

Recommended name:
- `vlm-prelabeling-pipeline`

Recommended URL pattern:
- `https://github.com/<org-or-user>/vlm-prelabeling-pipeline`

## Why this is needed

The manuscript `Declarations` section should point to a public and stable code/data location, not a local path or a private workspace structure.

## Step-by-step

1. Prepare a clean public bundle:

```bash
cd platform/vlm-pipeline
bash scripts/prepare_public_release.sh
```

2. Create a new GitHub repository (public):
- Name: `vlm-prelabeling-pipeline`
- Visibility: Public
- Do not initialize with README/license if you will push the bundle directly.

3. Publish the bundle:

```bash
cd .tmp/public-release/vlm-prelabeling-pipeline
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/<org-or-user>/vlm-prelabeling-pipeline.git
git push -u origin main
```

4. Optional but recommended:
- Create a release tag (e.g., `v1.0.0`) for reproducibility.
- Archive the release to Zenodo and mint a DOI.

## Manuscript update rule

After publishing, set the paper `Declarations` URLs to this public repository URL (and Zenodo DOI if available).
