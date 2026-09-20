#!/usr/bin/env bash
# Copy exactly the files that a fresh `git add .` would pick up (i.e. respecting .gitignore) into a
# NEW directory, so you can start a clean-history repository there.
#
#   scripts/export_public_tree.sh ../demand-forecasting-platform
#
# It does NOT run git init / add / commit / push. Owner-only working documents (the publication
# manifest/checklist and the personal interview / resume notes) are left out unless --all-docs is given.
set -euo pipefail

ALL_DOCS=0
TARGET=""
for arg in "$@"; do
  case "$arg" in
    --all-docs) ALL_DOCS=1 ;;
    *) TARGET="$arg" ;;
  esac
done
[ -n "$TARGET" ] || { echo "usage: $0 [--all-docs] <new-empty-directory>"; exit 2; }
if [ -e "$TARGET" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then
  echo "refusing: $TARGET exists and is not empty"; exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$TARGET"

OWNER_ONLY='^docs/(PUBLIC_REPOSITORY_MANIFEST|PUBLICATION_CHECKLIST|INTERVIEW_GUIDE|RESUME_ENTRY|ENSEMBLE_MIGRATION_AUDIT)\.md$'

# tracked + untracked-but-not-ignored files that exist on disk
git ls-files -co --exclude-standard | sort -u | while IFS= read -r f; do
  [ -f "$f" ] || continue
  if [ "$ALL_DOCS" -eq 0 ] && printf '%s\n' "$f" | grep -Eq "$OWNER_ONLY"; then continue; fi
  printf '%s\n' "$f"
done > "$TARGET/.export_list.tmp"

rsync -a --files-from="$TARGET/.export_list.tmp" ./ "$TARGET/"
COUNT=$(wc -l < "$TARGET/.export_list.tmp" | tr -d ' ')
rm "$TARGET/.export_list.tmp"
echo "Exported $COUNT files to $TARGET ($(du -sh "$TARGET" | cut -f1))."
echo "Next (yourself): cd \"$TARGET\" && git init && git add . && review 'git status' && git commit"
