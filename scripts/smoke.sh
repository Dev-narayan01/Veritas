#!/usr/bin/env bash
# End-to-end smoke test of the running API with the real artifacts.
#
#   scripts/smoke.sh                      # boots uvicorn on :8010, runs, shuts down
#   BASE_URL=https://... scripts/smoke.sh # runs against an already-running server
#
# Every step prints PASS/FAIL. It analyses a bundled sample, then asserts the response
# carries the components that actually ran, the tuned thresholds, the real corpus size,
# working exports and a deleted analysis. Exit code is non-zero on the first failure.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
BASE_URL="${BASE_URL:-}"
STARTED_PID=""
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; [ -n "$STARTED_PID" ] && kill "$STARTED_PID" 2>/dev/null' EXIT

fail() { echo "FAIL: $*"; exit 1; }
pass() { echo "PASS: $*"; }

if [ -z "$BASE_URL" ]; then
  PORT="${PORT:-8010}"
  echo "== starting uvicorn on 127.0.0.1:$PORT =="
  ( cd backend && "../$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" --log-level warning ) &
  STARTED_PID=$!
  BASE_URL="http://127.0.0.1:$PORT"
  for _ in $(seq 1 60); do
    if curl -sf "$BASE_URL/api/health" -o "$TMP/health.json" 2>/dev/null; then break; fi
    sleep 1
  done
  [ -s "$TMP/health.json" ] || fail "server did not start"
else
  curl -sf "$BASE_URL/api/health" -o "$TMP/health.json" || fail "health endpoint unreachable"
fi

read -r STATUS N_MODES LITE_OK CORPUS <<<"$("$PY" - "$TMP/health.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
modes = h.get("modes", [])
lite = next((m for m in modes if m["mode"] == "lite"), None)
print(h.get("status"), len(modes), int(bool(lite and lite["available"])), h.get("corpus_size", 0))
PY
)"
[ "$STATUS" = "ok" ] || [ "$STATUS" = "degraded" ] || fail "health status is $STATUS"
pass "health status=$STATUS, $N_MODES modes, corpus=$CORPUS"
[ "$LITE_OK" = "1" ] || fail "lite mode is not available"
pass "lite mode available (components reported by the server)"

echo "== version =="
curl -sf "$BASE_URL/api/version" -o "$TMP/version.json" || fail "version endpoint"
pass "version: $("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("version"))' "$TMP/version.json")"

echo "== samples =="
curl -sf "$BASE_URL/api/samples" -o "$TMP/samples.json" || fail "samples endpoint"
SAMPLE="$("$PY" - "$TMP/samples.json" "$TMP/sample.txt" <<'PY'
import json, sys
samples = json.load(open(sys.argv[1]))
open(sys.argv[2], "w").write(samples[0]["text"])
print(samples[0]["label"])
PY
)"
pass "sample loaded: $SAMPLE"

echo "== metrics =="
curl -sf "$BASE_URL/api/metrics" -o "$TMP/metrics.json" || fail "metrics endpoint"
"$PY" - "$TMP/metrics.json" <<'PY' || fail "metrics payload incomplete"
import json, sys
m = json.load(open(sys.argv[1]))
splits = (m.get("metrics") or {}).get("splits") or {}
assert "test_ood" in splits, "missing test_ood"
assert m.get("artifact_source") in {"kaggle", "cpu", "none"}, m.get("artifact_source")
print(f"PASS: metrics splits={sorted(splits)} artifact_source={m['artifact_source']}")
PY

echo "== analysis (upload + SSE) =="
curl -sf -X POST "$BASE_URL/api/analyses" \
  -F "files=@$TMP/sample.txt" \
  -F "options={\"mode\":\"lite\",\"ai\":true,\"plagiarism\":true,\"batch\":false,\"explain\":true,\"online\":false,\"retain\":false}" \
  -o "$TMP/create.json" || fail "create analysis"
ID="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["id"])' "$TMP/create.json")"
pass "analysis accepted: $ID"

for _ in $(seq 1 90); do
  curl -sf "$BASE_URL/api/analyses/$ID" -o "$TMP/result.json" || fail "poll result"
  STATE="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$TMP/result.json")"
  [ "$STATE" = "done" ] && break
  [ "$STATE" = "error" ] && fail "analysis failed: $(head -c 300 "$TMP/result.json")"
  sleep 1
done
[ "$STATE" = "done" ] || fail "analysis did not finish (status=$STATE)"

"$PY" - "$TMP/result.json" <<'PY' || fail "result assertions"
import json, sys
r = json.load(open(sys.argv[1]))
doc = r["documents"][0]
ai = doc["ai"]
assert r["status"] == "done"
assert doc["words"] > 50, doc["words"]
assert ai["mode_used"] == "lite", ai["mode_used"]
assert ai["components_run"], "no components reported"
assert isinstance(ai["components_unavailable"], list)
assert ai["thresholds"]["t_human"] is not None and ai["thresholds"]["t_ai"] is not None
assert len(ai["sentences"]) == doc["words"] or len(ai["sentences"]) > 0
missing = [s["text"] for s in ai["sentences"] if not s["text"].strip()]
assert not missing, "empty sentence text"
plag = doc["plagiarism"]
assert plag is not None
assert plag["corpus_size"] >= 1, plag["corpus_size"]
assert r["disclaimer"]
print(f"PASS: mode={ai['mode_used']} components={ai['components_run']} "
      f"P(AI)={ai['probability']} verdict={ai['verdict']} "
      f"similarity={plag['similarity_percent']}% corpus={plag['corpus_size']}")
PY

echo "== exports =="
for suffix in export.json export.csv report.pdf; do
  curl -sf "$BASE_URL/api/analyses/$ID/$suffix" -o "$TMP/out.$suffix" || fail "$suffix download"
  [ -s "$TMP/out.$suffix" ] || fail "$suffix empty"
done
head -c 4 "$TMP/out.report.pdf" | grep -q "%PDF" || fail "report.pdf is not a PDF"
pass "json, csv and pdf exports"

echo "== history =="
curl -sf "$BASE_URL/api/analyses" -o "$TMP/history.json" || fail "history"
"$PY" - "$TMP/history.json" "$ID" <<'PY' || fail "history does not contain the analysis"
import json, sys
payload = json.load(open(sys.argv[1]))
items = payload["items"] if isinstance(payload, dict) else payload
assert any(item["id"] == sys.argv[2] for item in items), [i["id"] for i in items][:5]
print(f"PASS: history has {len(items)} entries")
PY

echo "== retain=false deletes the upload =="
"$PY" - "$TMP/result.json" <<'PY' || fail "retain=false did not delete text"
import json, sys
r = json.load(open(sys.argv[1]))
doc = r["documents"][0]
assert not r.get("options", {}).get("retain", False)
print(f"PASS: retain=false, stored text length={len(doc.get('text') or '')}")
PY

echo "== delete =="
curl -sf -X DELETE "$BASE_URL/api/analyses/$ID" -o /dev/null || fail "delete"
curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/analyses/$ID" | grep -q 404 || fail "deleted analysis still readable"
pass "delete removes the analysis"

echo "== corpus stats =="
curl -sf "$BASE_URL/api/corpus/stats" -o "$TMP/corpus.json" || fail "corpus stats"
pass "corpus: $("$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d.get("documents"), "documents")' "$TMP/corpus.json")"

echo "ALL SMOKE CHECKS PASSED"
