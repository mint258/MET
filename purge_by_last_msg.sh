#!/usr/bin/env bash
# ɾ�������һ���ύ��Ϣ=������Ϣ���������ļ����Լ���˱�յ�Ŀ¼��
# ���� Git 1.8.3.1
set -u

MESSAGE="${1:-first commit}"

# --- ��ȫ��� ---
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Not inside a git repository." >&2
  exit 1
}

# --- 1) �ҵ����� message == $MESSAGE ���ύ��ϣ ---
# ���Ʊ���ָ���������Ϣ���пո��������
TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t gitpurge)"
COMMITS_FILE="$TMPDIR/commits.txt"
FILES_RAW_Z="$TMPDIR/files_raw.z"
FILES_UNIQ_Z="$TMPDIR/files_uniq.z"
CANDIDATES_TXT="$TMPDIR/candidates.txt"
TOREMOVE_TXT="$TMPDIR/to_remove.txt"

trap 'rm -rf "$TMPDIR"' EXIT

# �г������ύ����ϣ<TAB>���⣩���ϸ�������� MESSAGE ƥ��
git log --all --pretty=format:'%H	%s' \
| awk -F '	' -v m="$MESSAGE" '$2 == m {print $1}' > "$COMMITS_FILE"

if [ ! -s "$COMMITS_FILE" ]; then
  echo "No commits found with message exactly: \"$MESSAGE\""
  exit 0
fi

# --- 2) ������Щ�ύ�г��ֹ��������ļ���·������NUL �ָ�������ո� ---
: > "$FILES_RAW_Z"
while read -r commit; do
  # �����ļ�·����-z �� NUL ��β�������Ƚ�����
  git ls-tree -r -z --name-only "$commit" >> "$FILES_RAW_Z"
done < "$COMMITS_FILE"

# --- 3) ȥ���ļ��б�����ʹ�� sort -z��������������˷����� ---
if sort -z -u "$FILES_RAW_Z" -o "$FILES_UNIQ_Z" 2>/dev/null; then
  :
else
  # ���ˣ��û������ NUL ��ȥ�أ����ܶ԰������е��ļ�������Ϊ���������ټ���
  tr '\0' '\n' < "$FILES_RAW_Z" | sort -u | tr '\n' '\0' > "$FILES_UNIQ_Z"
fi

# --- 4) ����ļ���飺�ڵ�ǰ HEAD �ϣ����ļ����һ���ύ�ġ����⡱�Ƿ���� MESSAGE ---
# �����߼���ɾ���嵥
: > "$TOREMOVE_TXT"
while IFS= read -r -d '' f; do
  # �����ѱ�ɾ����·����git log ���������ʷ·���������һ���޸�
  subj="$(git log -n 1 --pretty=format:%s -- "$f" 2>/dev/null || true)"
  if [ "$subj" = "$MESSAGE" ]; then
    printf '%s\n' "$f" >> "$TOREMOVE_TXT"
  fi
done < "$FILES_UNIQ_Z"

if [ ! -s "$TOREMOVE_TXT" ]; then
  echo "No files whose last commit message equals \"$MESSAGE\" at HEAD."
  exit 0
fi

echo "Matched files (last commit subject == \"$MESSAGE\"):"
cat "$TOREMOVE_TXT"

# --- 5) ����ģʽ������ӡ��ִ�� ---
if [ "${DRY_RUN:-}" = "1" ]; then
  echo "[DRY RUN] Nothing removed. Set DRY_RUN= to actually delete."
  exit 0
fi

# --- 6) ִ��ɾ�����ύ ---
# ʹ�� git rm ɾ�������빤�����е��ļ���ʣ���Ŀ¼ Git �����٣�����Ȼ��ʧ
# ���������ʷ���Ƴ����������������ɸ��� --cached
while IFS= read -r path; do
  # ��Щ·�������ѱ�ɾ�����Դ���
  git rm -r -- "$path" >/dev/null 2>&1 || true
done < "$TOREMOVE_TXT"

# ��û��ʵ�ʱ����ֱ���˳�
if git diff --cached --quiet; then
  echo "No staged changes to commit (perhaps they were already removed)."
  exit 0
fi

git commit -m "Remove files whose last commit message is \"$MESSAGE\""

echo "Done. Review the commit, then push as needed, e.g.:"
echo "  git push origin HEAD -f"
