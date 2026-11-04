# Freeze runs (Nov 3): the locked test sets, once, with the code as it stands. Logs to ..\data\runs\final_runs.log.
# Run from second-ear\:  powershell -ExecutionPolicy Bypass -File scripts\final_runs.ps1
# About 3 hours and $60 of API credit. Each step stops at its own spend cap; a failed step does not stop the rest.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$log = "..\data\runs\final_runs.log"
function Run([string]$cmd) {
    "`n=== $(Get-Date -Format s)  $cmd ===" | Tee-Object -FilePath $log -Append
    Invoke-Expression "$cmd 2>&1" | Tee-Object -FilePath $log -Append
}
Run "uv sync"
Run "uv run pytest -q"
# T1: 832 human-labeled text pairs
Run "uv run python scripts\run_text_eval.py --set t1-test --locked-test-run --verify --cap 30 --yes"
# T3: 2 h of clean interpreted speech (real voices): false alarms per clean hour
Run "uv run python scripts\run_asr.py --set t3"
Run "uv run python scripts\run_e2e.py --set t3 --locked-test-run --adjudicate --dual-asr --cap 15 --yes"
# T4: the T1 test pairs as 4 long voiced sessions (6.7 h), pipeline vs one-prompt audio baselines
Run "uv run python scripts\run_asr.py --set t4-test"
foreach ($l in @("zh", "hi", "ta", "es")) {
    Run "uv run python scripts\run_e2e.py --set t4-test --locked-test-run --langs $l --cap 20 --yes"
}
Run "uv run python scripts\run_audio_baseline.py --set t4-test --locked-test-run --model muse-wonder-1.0 --prompt a2 --reasoning medium --yes"
Run "uv run python scripts\run_audio_baseline.py --set t4-test --locked-test-run --prompt a2 --reasoning medium --yes"
# Live demo recording for the video (no sound needed; events are replayed later)
Run "uv run python scripts\live.py --file ..\data\dev\sessions\es-S02.wav --lang es --consent --yes"
"`n=== done $(Get-Date -Format s) ===" | Tee-Object -FilePath $log -Append
