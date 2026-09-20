param(
  [string]$Port="9001",
  [string]$Seeds=""
)
$env:PYTHONPATH = "$PWD"
if ($Seeds) { $env:SEEDS = $Seeds }
python -m wavelock.network.server --port $Port
exit $LASTEXITCODE
