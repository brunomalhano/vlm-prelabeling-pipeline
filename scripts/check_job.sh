#!/usr/bin/env bash
# Quick diagnostic for ACA job execution
set -euo pipefail

RG="rg-trainning-models"
JOB="vlm-pipeline-job"
EXEC="${1:-vlm-pipeline-job-qe8wd39}"
WS="550bbb75-3cfc-4c3f-b587-e1bcc361a5b7"

echo "=== Job Execution Status ==="
az containerapp job execution show -g "$RG" -n "$JOB" --job-execution-name "$EXEC" -o table 2>/dev/null

echo ""
echo "=== Streaming Logs ==="
az containerapp job logs show -g "$RG" -n "$JOB" --execution "$EXEC" --container vlm-pipeline 2>&1 | grep -v "^WARNING" | python3 -c "
import sys,json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        print(d.get('Log',''))
    except:
        print(line)
"

echo ""
echo "=== Log Analytics: Console Logs (last 20) ==="
az rest --method post \
  --url "https://api.loganalytics.io/v1/workspaces/$WS/query" \
  --headers "Content-Type=application/json" \
  --body "{\"query\":\"ContainerAppConsoleLogs_CL | where ContainerAppName_s == 'vlm-pipeline-job' | order by TimeGenerated desc | take 20 | project TimeGenerated, Log_s\"}" \
  -o json 2>/dev/null | python3 -c "
import sys,json
d = json.load(sys.stdin)
rows = d['tables'][0]['rows']
cols = [c['name'] for c in d['tables'][0]['columns']]
print(f'  Rows found: {len(rows)}')
for row in rows:
    r = dict(zip(cols,row))
    print(f'  {r.get(\"TimeGenerated\",\"\")[:19]} | {r.get(\"Log_s\",\"\")[:150]}')
" 2>/dev/null

echo ""
echo "=== Log Analytics: System Logs (last 20) ==="
az rest --method post \
  --url "https://api.loganalytics.io/v1/workspaces/$WS/query" \
  --headers "Content-Type=application/json" \
  --body "{\"query\":\"ContainerAppSystemLogs_CL | where ContainerAppName_s == 'vlm-pipeline-job' | order by TimeGenerated desc | take 20 | project TimeGenerated, EventSource_s, Log_s, Level\"}" \
  -o json 2>/dev/null | python3 -c "
import sys,json
d = json.load(sys.stdin)
rows = d['tables'][0]['rows']
cols = [c['name'] for c in d['tables'][0]['columns']]
print(f'  Rows found: {len(rows)}')
for row in rows:
    r = dict(zip(cols,row))
    print(f'  {r.get(\"TimeGenerated\",\"\")[:19]} [{r.get(\"EventSource_s\",\"\")}] [{r.get(\"Level\",\"\")}] {r.get(\"Log_s\",\"\")[:120]}')
" 2>/dev/null

echo ""
echo "=== Done ==="
