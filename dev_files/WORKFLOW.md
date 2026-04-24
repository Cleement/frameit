# FrameIt - Development workflow

## Overview

FrameIt uses two separate Git repositories hosted on two GitLab instances:

- **gitlab.meteo.fr/souffletc/frameit-dev** (internal network only): development repository with all working files.
- **git.meteo.fr/souffletc/frameit** (public): stable releases only, clean codebase shared externally.

## Branches

The project uses two permanent branches:

- **main**: stable branch, pushed to both instances. Contains only the production code.
- **dev**: working branch, pushed only to `gitlab.meteo.fr`. Contains everything in `main` plus development-only files in `dev_files/`.

All work happens on short-lived feature branches created from `dev`. Code reaches `dev` through merge requests, then `main` through the merge script.

## Repository structure

```
frameit/
    src/frameit/         # source code
    tests/               # (future) unit tests
    README.md
    pyproject.toml
    .gitignore
    dev_files/           # dev-only (excluded from main)
        DRAFT/
        configs_test/
        merge_to_main.sh
        WORKFLOW.md
        ...
```

The `dev_files/` directory exists only on the `dev` branch. It must never appear on `main`.

## Remotes

Two remotes are configured locally:

```bash
git remote -v
# origin    https://gitlab.meteo.fr/souffletc/frameit-dev.git  (internal)
# public    https://git.meteo.fr/souffletc/frameit.git          (public)
```

If starting from a fresh clone of `gitlab.meteo.fr`, add the public remote:

```bash
git remote add public https://git.meteo.fr/souffletc/frameit.git
```

## Branch protection (gitlab.meteo.fr)

Both permanent branches are protected:

- **dev**: merge allowed for Maintainers, push allowed for no one (merge requests only).
- **main**: merge and push allowed for Maintainers (push needed for the merge script).

## Daily workflow

### 1. Create a feature branch from dev

```bash
git checkout dev
git pull origin dev
git checkout -b feature/my-feature
```

### 2. Work, commit, push

```bash
git add .
git commit -m "Description of changes"
git push origin feature/my-feature
```

### 3. Open a merge request

On `gitlab.meteo.fr`, open a merge request from `feature/my-feature` into `dev`. Review and merge via the interface.

### 4. Update dev locally

```bash
git checkout dev
git pull origin dev
```

### 5. Publish to main

When `dev` is stable and ready to be shared publicly:

```bash
./dev_files/merge_to_main.sh "Description of release"
```

The script does the following:

1. Checks that the working tree is clean.
2. Switches to `main`.
3. Merges `dev` without committing.
4. Removes `dev_files/` from the merge (handles conflicts automatically).
5. If there are conflicts outside `dev_files/`, the script stops and asks for manual resolution.
6. If only `dev_files/` content was modified, nothing is committed and no push happens.
7. Commits the merge.
8. Pushes `main` to `origin` (gitlab.meteo.fr).
9. Pushes `main` to `public` (git.meteo.fr).
10. Switches back to `dev`.

## Important notes

- If a feature branch only modifies files inside `dev_files/`, running the merge script will have no effect on `main`. This is expected behavior.
- The merge script must be run from a machine that has access to both `gitlab.meteo.fr` and `git.meteo.fr`.
- SSH key or stored credentials are needed for both remotes.

### Manual merge (if needed)

```bash
git checkout main
git merge dev --no-commit --no-ff
git rm -rf dev_files/
rm -rf dev_files/
git add -A
git commit -m "Merge dev into main"
git push origin main
git push public main
git checkout dev
```

## Rules

- Never push directly to `dev` or `main`. Always use merge requests or the merge script.
- Never push `dev` to the public remote.
- All development-only files go in `dev_files/`.
- Keep `main` in a clean, functional state at all times.