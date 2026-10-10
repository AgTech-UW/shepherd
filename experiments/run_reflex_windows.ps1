# Reflex follow-up study on Windows: all four AI conditions (zero-shot, what-if,
# lessons, memory, with their practice days), then the comparators, then the report.
# Needs Ollama running with qwen2.5:7b and nomic-embed-text pulled.
# From the repository folder, in a normal (not administrator) PowerShell:
#   powershell -ExecutionPolicy Bypass -File experiments\run_reflex_windows.ps1
# Writes reflex_results.zip to Downloads when everything is done (trajectories left out).
# Resumable: if it stops, run the same command again.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }
& $py experiments/farm_study.py --followers reflex 2>&1 | Tee-Object -FilePath reflex_run.log -Append
& $py experiments/farm_comparators.py --followers reflex --workers 4 2>&1 | Tee-Object -FilePath reflex_run.log -Append
& $py experiments/farm_study.py --followers reflex --report-only 2>&1 | Tee-Object -FilePath reflex_run.log -Append
$tmp = Join-Path $env:TEMP "reflex_results"
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
robocopy results\farm_study_reflex "$tmp\farm_study_reflex" /E /XD traj | Out-Null
Copy-Item reflex_run.log $tmp
Compress-Archive -Path "$tmp\*" -DestinationPath "$HOME\Downloads\reflex_results.zip" -Force
Write-Host "done: $HOME\Downloads\reflex_results.zip"
