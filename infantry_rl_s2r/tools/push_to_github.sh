#!/usr/bin/env bash
# 只更新 GitHub 上的 infantry_rl_locomotion/，不碰仓库里的其他目录。
# 不需要重新 clone：基于现有 .git 的对象库 + 临时 index 构造一个 commit。
#
# 用法:
#   bash tools/push_to_github.sh ["commit message"]
#
# 位置:
#   D:/IsaacLab-main/.git                      现有仓库
#   D:/IsaacLab-main/infantry_rl_locomotion    要上传的目录
#
# 说明:
#   - 只提交 content 和 origin/main 不同的文件；
#   - 不删除远端已有的其他文件（包括别人的目录、旧日志、旧资产）；
#   - 不执行 git checkout / git clone，不会动本地工作区。

set -euo pipefail

REPO_ROOT="D:/IsaacLab-main"
PROJECT_DIR="$REPO_ROOT/infantry_rl_locomotion"
REMOTE="origin"
BRANCH="main"
TMP_INDEX="D:/push_irl_index"
TMP_COMMIT="D:/push_irl_commit.txt"
MSG="${1:-Update infantry_rl_locomotion}"

cd "$REPO_ROOT"

echo "[1/5] fetch $REMOTE/$BRANCH"
git fetch "$REMOTE" "$BRANCH"
git rev-parse --verify "refs/remotes/$REMOTE/$BRANCH" >/dev/null

echo "[2/5] build temporary index from $REMOTE/$BRANCH"
rm -f "$TMP_INDEX" "$TMP_COMMIT"
export GIT_INDEX_FILE="$TMP_INDEX"
git read-tree "$REMOTE/$BRANCH"

echo "[3/5] hash changed/new files under infantry_rl_locomotion"
set +e
python - "$PROJECT_DIR" <<'PY'
import os
import subprocess
import sys

proj = sys.argv[1].replace('\\', '/')
env = dict(os.environ)
env['GIT_INDEX_FILE'] = 'D:/push_irl_index'

remote = {}
out = subprocess.check_output(
    ['git', 'ls-tree', '-r', 'origin/main', '--', 'infantry_rl_locomotion'],
    text=True, env=env,
)
for line in out.splitlines():
    meta, path = line.split('\t', 1)
    mode, typ, sha = meta.split()
    remote[path[len('infantry_rl_locomotion/'):]] = sha

skip_dirs = {'logs', '__pycache__', '.vscode', '.git'}
updates = []
for dp, dn, fn in os.walk(proj):
    dn[:] = [d for d in dn if d not in skip_dirs]
    for f in fn:
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, proj).replace('\\', '/')
        sha = subprocess.check_output(
            ['git', 'hash-object', '-w', p], text=True, env=env
        ).strip()
        if remote.get(rel) != sha:
            updates.append((rel, sha, os.path.getsize(p)))

if not updates:
    print('no changes; nothing to push')
    sys.exit(2)

for rel, sha, size in updates:
    path = 'infantry_rl_locomotion/' + rel
    subprocess.check_call(
        ['git', 'update-index', '--add', '--replace', '--cacheinfo',
         f'100644,{sha},{path}'],
        env=env,
    )

print(f'{len(updates)} file(s) changed:')
for rel, sha, size in sorted(updates):
    print(f'  {size:>10d}  {rel}')
PY
RC=$?
set -e
if [ "$RC" -eq 2 ]; then
    echo "[3/5] no changes, exit"
    rm -f "$TMP_INDEX"
    exit 0
elif [ "$RC" -ne 0 ]; then
    echo "ERROR: hash/update-index step failed (rc=$RC)"
    exit "$RC"
fi

TREE=$(git write-tree)
COMMIT=$(git commit-tree "$TREE" -p "$REMOTE/$BRANCH" -m "$MSG")
echo "$COMMIT" > "$TMP_COMMIT"

echo "[4/5] verify diff contains only infantry_rl_locomotion/"
BAD=$(git diff --name-only "$REMOTE/$BRANCH" "$COMMIT" | grep -v '^infantry_rl_locomotion/' || true)
if [ -n "$BAD" ]; then
    echo "ERROR: diff outside infantry_rl_locomotion/:"
    echo "$BAD"
    exit 1
fi
git diff --stat "$REMOTE/$BRANCH" "$COMMIT" | tail -40

echo "[5/5] push $COMMIT -> $REMOTE/$BRANCH"
# GitHub 偶尔返回 500；HTTP/1.1 + --no-thin 更稳。失败时可重试本脚本。
GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 push --no-thin \
    "$REMOTE" "$COMMIT:$BRANCH"

echo "done: $COMMIT"
rm -f "$TMP_INDEX" "$TMP_COMMIT"
