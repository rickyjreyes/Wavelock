param(
  [string]$Peer="127.0.0.1:9001",
  [string]$User="ricky",
  [string]$Message=$null
)
$env:PYTHONPATH = "$PWD"
if (-not $Message) { python -m chain.cli mine-daemon --peer $Peer --user $User }
else { python -m chain.cli mine-daemon --peer $Peer --user $User --message $Message }
