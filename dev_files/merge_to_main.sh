#!/bin/bash
# --------------------------------------------------------------------
# Merge dev into main, excluding dev-only files, then push to both
# remotes (origin = gitlab.meteo.fr, public = git.meteo.fr).
#
# Usage:
#     ./dev_files/merge_to_main.sh              # default commit message
#     ./dev_files/merge_to_main.sh "my message" # custom commit message
# --------------------------------------------------------------------

set -e

DEV_DIR="dev_files"
MSG="${1:-Merge dev into main}"

# Check that we start from a clean working tree
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working tree is not clean. Commit or stash your changes first."
    exit 1
fi

# Switch to main
git checkout main

# Merge dev without committing
if git merge dev --no-commit --no-ff 2>/dev/null; then
    # Remove dev-only files from the merge
    if [ -d "$DEV_DIR" ]; then
        git reset HEAD "$DEV_DIR" > /dev/null 2>&1
        rm -rf "$DEV_DIR"
        echo "Excluded $DEV_DIR from merge."
    fi

    # Check if there is anything to commit
    if git diff --cached --quiet; then
        echo "Nothing new to merge."
        git merge --abort 2>/dev/null || true
    else
        git commit -m "$MSG"
        echo "Merge committed."
    fi
else
    echo "Already up to date."
fi

# Push main to both remotes
git push origin main
echo "Pushed main to origin (gitlab.meteo.fr)."
git push public main
echo "Pushed main to public (git.meteo.fr)."

echo "Done."

# Switch back to dev
git checkout dev