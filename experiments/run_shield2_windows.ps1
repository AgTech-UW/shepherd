# Safety-filter study, fit check version 2 (rocks and poles left to the reflex), on Windows:
# all four AI conditions (zero-shot, what-if, lessons, memory, with their practice days) with
# the reflex and the safety filter on, then the comparators with the same filter, then the report.
# Needs Ollama running with qwen2.5:7b and nomic-embed-text pulled.
# From the repository folder, in a normal (not administrator) PowerShell:
#   powershell -ExecutionPolicy Bypass -File experiments\run_shield2_windows.ps1
# Writes shield2_results_windows.zip to Downloads when everything is done (trajectories left out).
# Resumable: if it stops, run the same command again.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }
& $py experiments/farm_study.py --followers reflex --shield --fit-version 2 2>&1 | Tee-Object -FilePath shield2_run.log -Append
& $py experiments/farm_comparators.py --followers reflex --shield --fit-version 2 --workers 4 2>&1 | Tee-Object -FilePath shield2_run.log -Append
& $py experiments/farm_study.py --followers reflex --shield --fit-version 2 --report-only 2>&1 | Tee-Object -FilePath shield2_run.log -Append
$tmp = Join-Path $env:TEMP "shield2_results"
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
robocopy results\farm_study_reflex_shield_fit2 "$tmp\farm_study_reflex_shield_fit2" /E /XD traj | Out-Null
Copy-Item shield2_run.log $tmp
Compress-Archive -Path "$tmp\*" -DestinationPath "$HOME\Downloads\shield2_results_windows.zip" -Force
Write-Host "done: $HOME\Downloads\shield2_results_windows.zip"
