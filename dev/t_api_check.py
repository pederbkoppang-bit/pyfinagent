"""Quick API smoke test — verifies backtest endpoints return data."""
import http.client
import json
import sys

def check(path: str) -> tuple[int, dict | str]:
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read().decode()
    try:
        return r.status, json.loads(body)
    except json.JSONDecodeError:
        return r.status, body[:200]

print("=== /api/health ===")
status, data = check("/api/health")
print(f"  {status} {data}")

print("\n=== /api/backtest/status ===")
status, data = check("/api/backtest/status")
print(f"  {status}")
if isinstance(data, dict):
    print(f"  status={data.get('status')}")
    print(f"  has_result={data.get('has_result')}")
    print(f"  run_id={data.get('run_id')}")
    print(f"  engine_source={data.get('engine_source')}")
else:
    print(f"  {data}")

print("\n=== /api/backtest/results ===")
status, data = check("/api/backtest/results")
print(f"  {status}")
if isinstance(data, dict):
    print(f"  keys={list(data.keys())[:10]}")
    a = data.get("analytics", {})
    print(f"  sharpe={a.get('sharpe')}")
    print(f"  total_return={a.get('total_return_pct')}")
else:
    print(f"  {data}")

print("\n=== /api/backtest/runs ===")
status, data = check("/api/backtest/runs")
print(f"  {status}")
if isinstance(data, dict):
    runs = data.get("runs", [])
    print(f"  {len(runs)} runs found")
    for r in runs[:3]:
        print(f"    {r.get('run_id')} sharpe={r.get('sharpe')}")
