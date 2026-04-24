# FrameIt - Development workflow

## Overview

FrameIt uses two separate Git repositories hosted on two GitLab instances:

- **gitlab.meteo.fr/souffletc/frameit-dev** (internal network only): development repository with all working files.
- **git.meteo.fr/souffletc/frameit** (public): stable releases only, clean codebase shared externally.

## Branches

The project uses two branches:

- **main**: stable branch, published on both instances. Contains only the production code.
- **dev**: working branch, pushed only to `gitlab.meteo.fr`. Contains everything in `main` plus development-only files.

All daily work happens on `dev`. Code is merged into `main` only when it is tested and ready to be shared publicly.

## Repository structure

```
frameit/
    frameit/            # source code
    tests/              # (future) unit tests
    README.md
    requirements.txt
    .gitignore
    dev_files/           # dev-only (excluded from main)
        DRAFT/
        configs_test/
        merge_to_main.sh
        ...
```

The `dev_files/` directory exists only on the `dev` branch. It is automatically excluded when merging into `main`.

## Remotes

After cloning, configure the two remotes:

```bash
git remote -v
# origin    https://gitlab.meteo.fr/souffletc/frameit-dev.git  (internal)
# public    https://git.meteo.fr/souffletc/frameit.git          (public)
```

If you cloned from `gitlab.meteo.fr`, add the public remote:

```bash
git remote add public https://git.meteo.fr/souffletc/frameit.git
```

## Daily workflow

### Working on dev

```bash
git checkout dev
# work, edit, test...
git add .
git commit -m "Description of changes"
git push origin dev
```

### Publishing to main

Use the merge script in `dev_files/`:

```bash
./dev_files/merge_to_main.sh                    # default commit message
./dev_files/merge_to_main.sh "Add new feature"  # custom commit message
```

The script does the following:

1. Checks that the working tree is clean.
2. Switches to `main`.
3. Merges `dev` without committing.
4. Removes `dev_files/` from the merge.
5. Commits and pushes `main` to the public remote.
6. Switches back to `dev`.

### Manual merge (if needed)

```bash
git checkout main
git merge dev --no-commit --no-ff
git reset HEAD dev_files/
rm -rf dev_files/
git commit -m "Merge dev into main"
git push public main
git checkout dev
```

## Rules

- Never push directly to `main`. Always merge from `dev`.
- Never push `dev` to the public remote.
- All development-only files go in `dev_files/`.
- Keep `main` in a clean, functional state at all times.
