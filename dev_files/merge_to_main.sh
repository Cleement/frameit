#!/bin/bash
# --------------------------------------------------------------------
# Merge dev into main, excluding dev-only files, then push to both
# remotes (origin = gitlab.meteo.fr, public = git.meteo.fr).
#
# Usage:
#     ./dev_files/merge_to_main.sh              # default commit message
#     ./dev_files/merge_to_main.sh "my message" # custom commit message
# --------------------------------------------------------------------
#

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

# Attempt merge (may fail due to conflicts on dev_files)
MERGE_OK=true
git merge dev --no-commit --no-ff 2>/dev/null || MERGE_OK=false

# Remove dev-only files whether merge succeeded or conflicted
if [ -d "$DEV_DIR" ]; then
    git rm -rf "$DEV_DIR" > /dev/null 2>&1 || true
    rm -rf "$DEV_DIR"
    echo "Excluded $DEV_DIR from merge."
fi

# Check for remaining conflicts (outside dev_files)
if git diff --name-only --diff-filter=U | grep -qv "^${DEV_DIR}/"; then
    echo "Error: conflicts found outside $DEV_DIR. Resolve them manually."
    exit 1
fi

# Stage everything
git add -A

# Check if there is anything to commit
if git diff --cached --quiet; then
    echo "Nothing new to merge."
    git merge --abort 2>/dev/null || true
else
    git commit -m "$MSG"
    echo "Merge committed."
fi

# Push main to both remotes
git push origin main
echo "Pushed main to origin (gitlab.meteo.fr)."
git push public main
echo "Pushed main to public (git.meteo.fr)."

echo "Done."

# Switch back to dev
git checkout dev