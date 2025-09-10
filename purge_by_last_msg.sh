#!/usr/bin/env bash
# 删除“最后一次提交信息=给定消息”的所有文件（以及因此变空的目录）
# 兼容 Git 1.8.3.1
set -u

MESSAGE="${1:-first commit}"

# --- 安全检查 ---
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Not inside a git repository." >&2
  exit 1
}

# --- 1) 找到所有 message == $MESSAGE 的提交哈希 ---
# 用制表符分隔，避免消息里有空格造成误判
TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t gitpurge)"
COMMITS_FILE="$TMPDIR/commits.txt"
FILES_RAW_Z="$TMPDIR/files_raw.z"
FILES_UNIQ_Z="$TMPDIR/files_uniq.z"
CANDIDATES_TXT="$TMPDIR/candidates.txt"
TOREMOVE_TXT="$TMPDIR/to_remove.txt"

trap 'rm -rf "$TMPDIR"' EXIT

# 列出所有提交（哈希<TAB>主题），严格按主题等于 MESSAGE 匹配
git log --all --pretty=format:'%H	%s' \
| awk -F '	' -v m="$MESSAGE" '$2 == m {print $1}' > "$COMMITS_FILE"

if [ ! -s "$COMMITS_FILE" ]; then
  echo "No commits found with message exactly: \"$MESSAGE\""
  exit 0
fi

# --- 2) 汇总这些提交中出现过的所有文件（路径），NUL 分隔以适配空格 ---
: > "$FILES_RAW_Z"
while read -r commit; do
  # 仅列文件路径；-z 以 NUL 结尾，便于稳健处理
  git ls-tree -r -z --name-only "$commit" >> "$FILES_RAW_Z"
done < "$COMMITS_FILE"

# --- 3) 去重文件列表（优先使用 sort -z；若不可用则回退方案） ---
if sort -z -u "$FILES_RAW_Z" -o "$FILES_UNIQ_Z" 2>/dev/null; then
  :
else
  # 回退：用换行替代 NUL 再去重（可能对包含换行的文件名无能为力，但极少见）
  tr '\0' '\n' < "$FILES_RAW_Z" | sort -u | tr '\n' '\0' > "$FILES_UNIQ_Z"
fi

# --- 4) 逐个文件检查：在当前 HEAD 上，该文件最后一次提交的“主题”是否等于 MESSAGE ---
# 符合者加入删除清单
: > "$TOREMOVE_TXT"
while IFS= read -r -d '' f; do
  # 对于已被删除的路径，git log 仍能针对历史路径返回最后一次修改
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

# --- 5) 干跑模式：仅打印不执行 ---
if [ "${DRY_RUN:-}" = "1" ]; then
  echo "[DRY RUN] Nothing removed. Set DRY_RUN= to actually delete."
  exit 0
fi

# --- 6) 执行删除并提交 ---
# 使用 git rm 删除索引与工作区中的文件；剩余空目录 Git 不跟踪，会自然消失
# 若想仅从历史中移除但保留工作区，可改用 --cached
while IFS= read -r path; do
  # 有些路径可能已被删，忽略错误
  git rm -r -- "$path" >/dev/null 2>&1 || true
done < "$TOREMOVE_TXT"

# 若没有实际变更，直接退出
if git diff --cached --quiet; then
  echo "No staged changes to commit (perhaps they were already removed)."
  exit 0
fi

git commit -m "Remove files whose last commit message is \"$MESSAGE\""

echo "Done. Review the commit, then push as needed, e.g.:"
echo "  git push origin HEAD -f"
