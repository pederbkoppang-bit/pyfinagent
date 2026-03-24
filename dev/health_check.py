"""Quick health check — writes results to _health_result.txt"""
import urllib.request, json, pathlib

out = []
try:
    r = urllib.request.urlopen("http://localhost:8000/api/health", timeout=5)
    out.append(f"HEALTH: {r.read().decode()}")
except Exception as e:
    out.append(f"HEALTH: FAILED - {e}")

try:
    r = urllib.request.urlopen("http://localhost:8000/api/backtest/status", timeout=5)
    out.append(f"BACKTEST_STATUS: {r.read().decode()}")
except Exception as e:
    out.append(f"BACKTEST_STATUS: FAILED - {e}")

try:
    r = urllib.request.urlopen("http://localhost:3000", timeout=5)
    out.append(f"FRONTEND: UP (status {r.status})")
except Exception as e:
    out.append(f"FRONTEND: FAILED - {e}")

pathlib.Path("_health_result.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
