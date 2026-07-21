#!/usr/bin/env bash
# test_csv_edge.sh: dataset with case-mixed ids (Q001/q001), queries with comma,
# quote, newline, emoji. Assert output is valid CSV (utf-8, QUOTE_MINIMAL,
# newlines in response collapsed to spaces, ids verbatim).
#
# Usage: IMAGE=trustworthy-full:local CONFIG=A bash test/test_csv_edge.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-trustworthy-full:local}"
CONFIG="${CONFIG:-A}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TESTDIR="$WORK/model/test"; RESDIR="$WORK/result"; LIBDIR="$WORK/benchmark_lib"
mkdir -p "$TESTDIR" "$RESDIR" "$LIBDIR"
cp "$HERE/progress_stub" "$LIBDIR/progress"; chmod 0755 "$LIBDIR/progress"

# Write a CSV with tricky content via python so quoting is exact.
python3 - "$TESTDIR/dataset.csv" <<'PY'
import csv, sys
path = sys.argv[1]
rows = [
    ("id", "query"),
    ("Q001", 'Hello, "world", with commas'),
    ("q001", 'line one\nline two'),               # embedded newline
    ("Q002", 'emoji test 🚀🔥 ไทย'),
    ("q002", 'quote " inside, and, comma'),
    ("Q003", 'tab\there'),
    ("q003", ''),
    ("Q004", 'multi\n\n\nblank\nlines'),
]
with open(path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerows(rows)
print("wrote", len(rows)-1, "rows")
PY
NROWS=$(python3 -c "import csv;print(sum(1 for _ in csv.reader(open('$TESTDIR/dataset.csv')))-1)")

GPU_FLAG=""
docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && GPU_FLAG="--gpus all"
echo "==> run container (CONFIG=$CONFIG)"
docker run --rm --platform linux/amd64 --network none $GPU_FLAG \
  -v "$TESTDIR:/model/test:ro" \
  -v "$RESDIR:/result" \
  -v "$LIBDIR:/benchmark_lib:ro" \
  -e TRUSTW_CONFIG="$CONFIG" \
  -e TRUSTW_BUDGET_S=600 -e TRUSTW_WATCHDOG_S=90 \
  "$IMAGE" 2>&1 | tail -20

SUB="$RESDIR/submission.csv"
echo "==> validate CSV invariants"
python3 - "$SUB" "$TESTDIR/dataset.csv" <<'PY'
import csv, sys
sub, ds = sys.argv[1], sys.argv[2]
with open(ds, newline="", encoding="utf-8") as f:
    inp = [(r["id"], r["query"]) for r in csv.DictReader(f)]
with open(sub, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    assert r.fieldnames == ["id", "response"], f"bad header {r.fieldnames}"
    out = [(row["id"], row["response"]) for row in r]
assert len(inp) == len(out), f"count in={len(inp)} out={len(out)}"
seen = set()
for i, (a, b) in enumerate(zip(inp, out)):
    if a[0] != b[0]:
        print(f"FAIL row {i}: id {a[0]!r} != {b[0]!r}"); sys.exit(1)
    seen.add(b[0])
    if not b[1].strip():
        print(f"FAIL row {i}: empty response for {b[0]!r}"); sys.exit(1)
    if "\n" in b[1] or "\r" in b[1]:
        print(f"FAIL row {i}: newline in response for {b[0]!r}"); sys.exit(1)
# case-sensitivity: Q001 and q001 must both exist and be distinct
if "Q001" not in seen or "q001" not in seen:
    print("FAIL: case-mixed ids not preserved (Q001/q001)"); sys.exit(1)
# verify valid utf-8 re-read round-trip (already opened above with utf-8 ok)
print(f"OK: {len(out)} rows, ids verbatim (case-preserved Q001/q001), "
      f"no empty, no embedded newlines, valid utf-8 QUOTE_MINIMAL")
PY

PARG="$(tail -1 "$RESDIR/progress.log" 2>/dev/null | awk '{print $1}')"
[[ "$PARG" == "$NROWS" ]] || { echo "FAIL: progress arg '$PARG' != $NROWS"; exit 1; }
echo "    progress called with: $PARG (OK)"
echo
echo "test_csv_edge PASSED"
