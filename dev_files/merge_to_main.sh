#!/bin/bash
# --------------------------------------------------------------------
# Merge dev into main, excluding dev-only files, then push to public.
#
# Usage:
#     ./merge_to_main.sh              # default commit message
#     ./merge_to_main.sh "my message" # custom commit message
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
git merge dev --no-commit --no-ff

# Remove dev-only files from the merge
if [ -d "$DEV_DIR" ]; then
    git reset HEAD "$DEV_DIR" > /dev/null 2>&1
    rm -rf "$DEV_DIR"
    echo "Excluded $DEV_DIR from merge."
fi

# Check if there is anything to commit
if git diff --cached --quiet; then
    echo "Nothing to merge. main is already up to date."
    git merge --abort 2>/dev/null || true
    git checkout dev
    exit 0
fi

# Commit and push
git commit -m "$MSG"
git push public main
echo "Done. main pushed to public."

# Switch back to dev
git checkout dev

