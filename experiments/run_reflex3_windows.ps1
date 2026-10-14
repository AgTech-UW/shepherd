# Reflex version 3 on Windows: the final configuration (safety filter version 4, reflex reacting
# 15 m out and keeping 2.5 m clear) with each machine committing to the side of least resistance
# when it passes a rock or pole (version 2), and the reflex deciding the turn while it passes: the
# flock rules (keeping apart from neighbours) may slow it or push it away from the obstacle, but not
# turn it toward it.
# All four AI conditions with their practice days, then the comparators with the same settings,
# then the report. Needs Ollama running with qwen2.5:7b and nomic-embed-text pulled.
# From the repository folder, in a normal (not administrator) PowerShell:
#   powershell -ExecutionPolicy Bypass -File experiments\run_reflex3_windows.ps1
# Writes reflex3_results_windows.zip to Downloads when everything is done (trajectories left out).
# Resumable: if it stops, run the same command again.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }
$cfg = @("--followers", "reflex", "--shield", "--fit-version", "4", "--reflex-look", "15", "--reflex-margin", "2.5", "--reflex-version", "3")
& $py experiments/farm_study.py @cfg 2>&1 | Tee-Object -FilePath reflex3_run.log -Append
& $py experiments/farm_comparators.py @cfg --workers 4 2>&1 | Tee-Object -FilePath reflex3_run.log -Append
& $py experiments/farm_study.py @cfg --report-only 2>&1 | Tee-Object -FilePath reflex3_run.log -Append
$name = "farm_study_reflex_shield_fit4_rL15M2.5_rv3"
$tmp = Join-Path $env:TEMP "reflex3_results"
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
robocopy "results\$name" "$tmp\$name" /E /XD traj | Out-Null
Copy-Item reflex3_run.log $tmp
Compress-Archive -Path "$tmp\*" -DestinationPath "$HOME\Downloads\reflex3_results_windows.zip" -Force
Write-Host "done: $HOME\Downloads\reflex3_results_windows.zip"
