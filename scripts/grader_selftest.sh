#!/usr/bin/env bash
# Grader check on the GPU before training: the dataset's reference answers for N train tasks must grade correct,
# and the same answers made wrong (output + 1) or without an own kernel must not. ~5 min on one MI300X.
# Usage: scripts/grader_selftest.sh WORK_DIR [N]    (N default 24, seed 0)
set -euo pipefail
. "$(dirname "$0")/env.sh"
W=$(readlink -f "${1:?work dir}"); N=${2:-24}; mkdir -p "$W"
cd "$(dirname "$0")/.."
python3 -I scripts/make_ref_rollouts.py --data "$AIG_DATA" --n "$N" --seed 0 --out "$W/refs.jsonl" >/dev/null
python3 - "$W" <<'EOF'
import json, re, sys
W = sys.argv[1]; rows = [json.loads(l) for l in open(f"{W}/refs.jsonl")]
out = []
for r in rows:
    out.append({**r, "id": r["id"] + "_ref"})
    # wrong: add 1 to the returned tensor (the last `return <expr>;` of forward)
    m = list(re.finditer(r"return\s+([^;]+);", r["response"]))
    if m:
        x = m[-1]; resp = r["response"][:x.start()] + f"return ({x.group(1)}) + 1;" + r["response"][x.end():]
        out.append({**r, "id": r["id"] + "_wrong", "response": resp})
    # no own kernel: strip __global__ (the scan must reject it)
    out.append({**r, "id": r["id"] + "_nokernel", "response": r["response"].replace("__global__", "")})
with open(f"{W}/selftest.jsonl", "w") as f:
    f.writelines(json.dumps(o) + "\n" for o in out)
EOF
python3 -I scripts/hip_grade.py --tasks "$W/selftest.jsonl" --work "$W/work" --out "$W/graded.jsonl" \
  --jobs "$(nproc)" --gpu --no-timing --workers 6 > "$W/grade.log" 2>&1
python3 - "$W" <<'EOF'
import json, sys
g = [json.loads(l) for l in open(f"{sys.argv[1]}/graded.jsonl")]
c = lambda kind: [o for o in g if o["id"].endswith(kind)]
ok = lambda xs: sum(bool(o.get("correct")) for o in xs)
ref, wrong, nok = c("_ref"), c("_wrong"), c("_nokernel")
print(f"references correct {ok(ref)}/{len(ref)} (expect all or all but 1-2: a few dataset refs are wrong); "
      f"output+1 correct {ok(wrong)}/{len(wrong)} (expect 0); no own kernel rewarded "
      f"{sum(o.get('reward', 0) > 0 for o in nok)}/{len(nok)} (expect 0)")
EOF
