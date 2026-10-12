# Final configuration on Windows: safety filter version 3 (fit check 2: rocks and poles left to the
# reflex; single file and narrowing never held back; single file when nothing fits) and a stronger
# reflex (reacts 15 m out, keeps 2.5 m clear; was 10 m and 1.5 m):
# all four AI conditions (zero-shot, what-if, lessons, memory, with their practice days) with
# the reflex and the safety filter on, then the comparators with the same filter, then the report.
# Needs Ollama running with qwen2.5:7b and nomic-embed-text pulled.
# From the repository folder, in a normal (not administrator) PowerShell:
#   powershell -ExecutionPolicy Bypass -File experiments\run_final_windows.ps1
# Writes final_results_windows.zip to Downloads when everything is done (trajectories left out).
# Resumable: if it stops, run the same command again.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }
& $py experiments/farm_study.py --followers reflex --shield --fit-version 3 --reflex-look 15 --reflex-margin 2.5 2>&1 | Tee-Object -FilePath final_run.log -Append
& $py experiments/farm_comparators.py --followers reflex --shield --fit-version 3 --reflex-look 15 --reflex-margin 2.5 --workers 4 2>&1 | Tee-Object -FilePath final_run.log -Append
& $py experiments/farm_study.py --followers reflex --shield --fit-version 3 --reflex-look 15 --reflex-margin 2.5 --report-only 2>&1 | Tee-Object -FilePath final_run.log -Append
$tmp = Join-Path $env:TEMP "final_results"
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
robocopy results\farm_study_reflex_shield_fit3_rL15M2.5 "$tmp\farm_study_reflex_shield_fit3_rL15M2.5" /E /XD traj | Out-Null
Copy-Item final_run.log $tmp
Compress-Archive -Path "$tmp\*" -DestinationPath "$HOME\Downloads\final_results_windows.zip" -Force
Write-Host "done: $HOME\Downloads\final_results_windows.zip"
