# Run this AFTER fully quitting Codex (check the tray icon / task manager).
# It points this thread's persisted paths at the new D: location and removes
# the leftover empty C: project directory.

$ErrorActionPreference = 'Stop'

$threadId = '01a14db0-7492-7ca0-af87-67e0f0b73733'
$oldCwd = 'C:\Users\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper'
$newCwd = 'D:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper'
$newOutputs = Join-Path $newCwd 'outputs'
$newRoot = 'D:\ASUS\Documents\Codex'
$statePath = 'C:\Users\ASUS\.codex\.codex-global-state.json'

if (-not (Test-Path -LiteralPath $newCwd)) {
    throw "New project dir not found: $newCwd"
}
if (-not (Test-Path -LiteralPath $statePath)) {
    throw "State file not found: $statePath"
}

$raw = [System.IO.File]::ReadAllText($statePath, [System.Text.Encoding]::UTF8)
$state = $raw | ConvertFrom-Json

# 1. Back up the original state file.
$backup = "$statePath.bak-migrate"
[System.IO.File]::WriteAllText($backup, $raw, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "backup written: $backup"

# 2. Update the functional keys for this thread.
$state.'thread-projectless-output-directories'.$threadId = $newOutputs
$state.'thread-workspace-root-hints'.$threadId = $newRoot

$atom = $state.'electron-persisted-atom-state'
$routeKey = "thread-tab-routes-v1:$threadId"
if ($atom.PSObject.Properties.Name -contains $routeKey) {
    foreach ($r in $atom.$routeKey.routes) {
        if ($r.params.cwd) { $r.params.cwd = $newCwd }
    }
}

# 3. Best-effort: rewrite recent-file paths for this thread's client thread id.
$clientThreadId = 'client-new-thread:40550398-710f-4bd5-b292-eab69d24e04b'
$recent = $atom.'recent-workspace-files-v3'
if ($recent) {
    foreach ($prop in $recent.PSObject.Properties) {
        $entry = $prop.Value
        if ($entry.threadId -eq $clientThreadId) {
            if ($entry.path) {
                $entry.path = $entry.path.Replace('/C:/Users/ASUS/Documents/Codex/2026-10-18/idea-x7167-0-1-x7684-paper', '/D:/ASUS/Documents/Codex/2026-10-18/idea-x7167-0-1-x7684-paper')
                $entry.path = $entry.path.Replace('C:\Users\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper', 'D:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper')
            }
        }
    }
}

# 4. Write back as UTF-8 without BOM.
$json = $state | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText($statePath, $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "updated: $statePath"

# 5. Remove the leftover C: project directory (it should be empty after the move).
if (Test-Path -LiteralPath $oldCwd) {
    Remove-Item -LiteralPath $oldCwd -Recurse -Force
    Write-Host "removed old dir: $oldCwd"
} else {
    Write-Host "old dir already gone: $oldCwd"
}

Write-Host 'Done. Restart Codex and reopen this thread.'
