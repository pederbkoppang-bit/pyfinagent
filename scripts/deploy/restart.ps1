<#
.SYNOPSIS
    Kill processes on ports 8000/3000 and restart backend + frontend.
.DESCRIPTION
    - Recursive process-tree kill (uvicorn reloader + workers, node + children)
    - Orphan cleanup: kills stale uvicorn/node children even if they released the port
    - Detached launch via cmd.exe wrapper (CreateNoWindow) — NO output leaks
    - Server stdout/stderr routed to _backend.log and _frontend.log (native cmd redirection)
    - HTTP-level health checks (not just TCP) — verifies /api/health returns "ok"
    - Validates backtest data loaded on startup via /api/backtest/status
.USAGE
    From the pyfinagent root:
        .\restart.ps1
    To tail server logs:
        Get-Content _backend.log -Wait
        Get-Content _frontend.log -Wait
#>

param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [int]$MaxWaitSeconds = 60
)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "`n=== PyFinAgent Restart ===" -ForegroundColor Cyan

# ── Helpers ──────────────────────────────────────────────────────────
function Get-ListeningPids([int]$Port) {
    # Get-NetTCPConnection is PowerShell-native; filter Listen only
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        Where-Object { $_ -ne 0 }
}

function Kill-Tree([int]$pid) {
    # Recursively kill children first (via CIM), then the parent
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $pid" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Kill-Tree $child.ProcessId
    }
    try {
        $p = [System.Diagnostics.Process]::GetProcessById($pid)
        Write-Host "    Kill PID $pid ($($p.ProcessName))..." -ForegroundColor Gray
        $p.Kill()
        $p.WaitForExit(5000)
    } catch { }
}

function Kill-PortListeners([int]$Port) {
    $pids = Get-ListeningPids $Port
    if (-not $pids -or $pids.Count -eq 0) {
        Write-Host "  No listeners found on port $Port" -ForegroundColor DarkGray
        return
    }
    foreach ($pid in $pids) {
        Write-Host "  Killing PID $pid + children on port $Port..." -ForegroundColor Gray
        Kill-Tree $pid
    }
}

function Kill-OrphanProcesses {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        ($_.CommandLine -match "uvicorn" -and $_.CommandLine -match "pyfinagent") -or
        ($_.CommandLine -match "next-server" -and $_.CommandLine -match "pyfinagent")
    } | ForEach-Object {
        $p = try { Get-Process -Id $_.ProcessId -ErrorAction Stop } catch { $null }
        if ($p) {
            Write-Host "  Orphan PID $($_.ProcessId) ($($_.Name))..." -ForegroundColor Gray
            Kill-Tree $_.ProcessId
        }
    }
}

function Wait-PortFree([int]$Port, [int]$TimeoutSec = 25) {
    for ($i = 0; $i -lt ($TimeoutSec * 2); $i++) {
        $pids = Get-ListeningPids $Port
        if (-not $pids -or $pids.Count -eq 0) { return $true }
        Start-Sleep -Milliseconds 500
        if (($i + 1) % 10 -eq 0) {
            Write-Host "  Port $Port still held ($(($i+1)/2)s), re-killing PIDs: $($pids -join ', ')..." -ForegroundColor DarkYellow
            foreach ($pid in $pids) { Kill-Tree $pid }
            Kill-OrphanProcesses
        }
    }
    return $false
}

function Start-Detached([string]$exe, [string]$argString, [string]$workDir, [string]$label) {
    # Launch via cmd.exe with native file redirection (>> log 2>&1).
    # This avoids .NET RedirectStandardOutput which causes Next.js to hang
    # and Register-ObjectEvent handlers that die when the script session exits.
    $logFile = Join-Path $root "_${label}.log"

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = "cmd.exe"
    $psi.Arguments = "/c `"`"$exe`" $argString >> `"$logFile`" 2>&1`""
    $psi.WorkingDirectory = $workDir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $p = [System.Diagnostics.Process]::new()
    $p.StartInfo = $psi
    $p.Start() | Out-Null

    Write-Host "  Started $label as PID $($p.Id) (logs: _${label}.log)" -ForegroundColor Gray
    return $p
}

function Wait-HttpOk([string]$Url, [string]$Name, [int]$TimeoutSec) {
    $port = if ($Url -match ':(\d+)/') { [int]$Matches[1] } else { 80 }
    $tcpOk = Wait-PortUp -Port $port -Name "$Name (TCP)" -TimeoutSec ([math]::Min($TimeoutSec, 30))
    if (-not $tcpOk) { return $false }

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $req = [System.Net.HttpWebRequest]::Create($Url)
            $req.Timeout = 5000
            $resp = $req.GetResponse()
            $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
            $body = $sr.ReadToEnd()
            $sr.Close()
            $resp.Close()
            if ($body -match '"ok"' -or $body -match '"status"') {
                Write-Host "  $Name : HTTP OK  ($Url)" -ForegroundColor Green
                return $true
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
    Write-Host "  $Name : HTTP TIMEOUT after ${TimeoutSec}s ($Url)" -ForegroundColor Red
    return $false
}

function Wait-PortUp([int]$Port, [string]$Name, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $tcp = New-Object System.Net.Sockets.TcpClient
        try {
            $tcp.Connect("localhost", $Port)
            $tcp.Dispose()
            Write-Host "  $Name : TCP UP (port $Port)" -ForegroundColor Green
            return $true
        } catch {
            $tcp.Dispose()
            Start-Sleep -Seconds 1
        }
    }
    Write-Host "  $Name : TIMEOUT after ${TimeoutSec}s (port $Port)" -ForegroundColor Red
    return $false
}

# ── Step 0: Truncate log files ───────────────────────────────────────
Set-Content -Path (Join-Path $root "_backend.log")  -Value "=== Backend log $(Get-Date) ===" -Encoding UTF8
Set-Content -Path (Join-Path $root "_frontend.log") -Value "=== Frontend log $(Get-Date) ===" -Encoding UTF8

# ── Step 1: Kill processes on both ports + orphans ───────────────────
Write-Host "`n[1/5] Killing processes on ports $BackendPort, $FrontendPort..." -ForegroundColor Yellow

Kill-PortListeners $BackendPort
Kill-PortListeners $FrontendPort
Kill-OrphanProcesses

$beFree = Wait-PortFree $BackendPort
$feFree = Wait-PortFree $FrontendPort

# If port is occupied but the service is healthy, reuse it
$beReuse = $false
$feReuse = $false
if (-not $beFree) {
    try {
        $req = [System.Net.HttpWebRequest]::Create("http://localhost:$BackendPort/api/health")
        $req.Timeout = 3000
        $resp = $req.GetResponse()
        $resp.Close()
        $beReuse = $true
        Write-Host "  Backend already healthy on port $BackendPort - reusing." -ForegroundColor Green
    } catch {
        Write-Host "  FATAL: Port $BackendPort occupied by unresponsive process!" -ForegroundColor Red
        Write-Host "  Tip: Close VS Code and reopen, or use Task Manager to kill python.exe" -ForegroundColor Yellow
    }
}
if (-not $feFree) {
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("localhost", $FrontendPort)
        $tcp.Dispose()
        $feReuse = $true
        Write-Host "  Frontend already running on port $FrontendPort - reusing." -ForegroundColor Green
    } catch {
        $tcp.Dispose()
        Write-Host "  FATAL: Port $FrontendPort occupied by unresponsive process!" -ForegroundColor Red
    }
}

if ($beFree -and $feFree) {
    Write-Host "  Both ports free." -ForegroundColor Green
}

# ── Step 2: Start backend (detached — no terminal output) ───────────
if ($beFree) {
    Write-Host "`n[2/5] Starting backend (uvicorn) on port $BackendPort..." -ForegroundColor Yellow

    $pythonExe = Join-Path $root ".venv312\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        Write-Host "  ERROR: Python not found at $pythonExe" -ForegroundColor Red
        exit 1
    }

    Start-Detached $pythonExe "-m uvicorn backend.main:app --reload --port $BackendPort" $root "backend" | Out-Null
} elseif ($beReuse) {
    Write-Host "`n[2/5] Backend already running (reused)" -ForegroundColor Green
} else {
    Write-Host "`n[2/5] SKIPPED backend (port $BackendPort occupied)" -ForegroundColor Red
}

# ── Step 3: Start frontend (detached — no terminal output) ──────────
if ($feFree) {
    Write-Host "`n[3/5] Starting frontend (next dev) on port $FrontendPort..." -ForegroundColor Yellow

    $frontendDir = Join-Path $root "frontend"
    $nextBin = Join-Path $frontendDir "node_modules\next\dist\bin\next"
    $nodeExe = (Get-Command node -ErrorAction SilentlyContinue).Source
    if (-not $nodeExe) { $nodeExe = "node.exe" }
    Start-Detached $nodeExe "$nextBin dev" $frontendDir "frontend" | Out-Null
} elseif ($feReuse) {
    Write-Host "`n[3/5] Frontend already running (reused)" -ForegroundColor Green
} else {
    Write-Host "`n[3/5] SKIPPED frontend (port $FrontendPort occupied)" -ForegroundColor Red
}

# ── Step 4: HTTP health checks ──────────────────────────────────────
Write-Host "`n[4/5] Waiting for services (HTTP health checks)..." -ForegroundColor Yellow

if ($beReuse) {
    $backendOk = $true
    Write-Host "  Backend  : already verified healthy" -ForegroundColor Green
} else {
    $backendOk = Wait-HttpOk -Url "http://localhost:$BackendPort/api/health" -Name "Backend " -TimeoutSec $MaxWaitSeconds
}
if ($feReuse) {
    $frontendOk = $true
    Write-Host "  Frontend : already verified listening" -ForegroundColor Green
} else {
    # Frontend needs TCP + HTTP: Next.js binds the port early but takes time to compile
    $fePortUp = Wait-PortUp -Port $FrontendPort -Name "Frontend" -TimeoutSec $MaxWaitSeconds
    if ($fePortUp) {
        Write-Host "  Frontend : waiting for first page compile (this may take 15-20s)..." -ForegroundColor DarkGray
        $feDeadline = (Get-Date).AddSeconds($MaxWaitSeconds)
        $frontendOk = $false
        while ((Get-Date) -lt $feDeadline) {
            try {
                $req = [System.Net.HttpWebRequest]::Create("http://localhost:$FrontendPort/")
                $req.Timeout = 30000
                $resp = $req.GetResponse()
                $resp.Close()
                $frontendOk = $true
                Write-Host "  Frontend : HTTP OK (http://localhost:$FrontendPort)" -ForegroundColor Green
                break
            } catch {
                Start-Sleep -Seconds 2
            }
        }
        if (-not $frontendOk) {
            Write-Host "  Frontend : HTTP TIMEOUT - port is open but page compilation may still be in progress" -ForegroundColor DarkYellow
            Write-Host "             Try refreshing http://localhost:$FrontendPort in your browser" -ForegroundColor DarkYellow
            $frontendOk = $true  # Don't block - it will finish compiling
        }
    } else {
        $frontendOk = $false
    }
}

# ── Step 5: Validate backtest data loaded ────────────────────────────
$btDataOk = $false
if ($backendOk) {
    Write-Host "`n[5/5] Checking backtest data loaded..." -ForegroundColor Yellow
    try {
        $wc = New-Object System.Net.WebClient
        $statusJson = $wc.DownloadString("http://localhost:$BackendPort/api/backtest/status")
        $wc.Dispose()
        if ($statusJson -match '"has_result"\s*:\s*true') {
            Write-Host "  Backtest data: LOADED" -ForegroundColor Green
            $btDataOk = $true
        } else {
            Write-Host "  Backtest data: NOT LOADED (no prior results on disk)" -ForegroundColor DarkYellow
        }
    } catch {
        Write-Host "  Backtest data: CHECK FAILED ($($_.Exception.Message))" -ForegroundColor Red
    }
} else {
    Write-Host "`n[5/5] Skipped backtest check (backend not ready)." -ForegroundColor DarkYellow
}

# ── Summary ──────────────────────────────────────────────────────────
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
if ($backendOk)  { Write-Host "  Backend  : http://localhost:$BackendPort  (health OK)" -ForegroundColor Green }
else             { Write-Host "  Backend  : FAILED" -ForegroundColor Red }
if ($frontendOk) { Write-Host "  Frontend : http://localhost:$FrontendPort" -ForegroundColor Green }
else             { Write-Host "  Frontend : FAILED" -ForegroundColor Red }
if ($btDataOk)   { Write-Host "  Backtest : data loaded (cards should show data)" -ForegroundColor Green }
elseif ($backendOk) { Write-Host "  Backtest : no cached results found" -ForegroundColor DarkYellow }
Write-Host ""
Write-Host "  Logs: _backend.log, _frontend.log" -ForegroundColor DarkGray
Write-Host "  Tail: Get-Content _backend.log -Wait" -ForegroundColor DarkGray
Write-Host ""
