param(
  [string]$Port="9001",
  [string]$Seeds=""
)
$env:PYTHONPATH = "$PWD"
if ($Seeds) { $env:SEEDS = $Seeds }
python -m network.server --port $Port
