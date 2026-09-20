param(
  [string]$SignedPath="signed_message.json"
)
$env:PYTHONPATH = "$PWD"
python -m wavelock.chain.cli mine --signed-path $SignedPath
exit $LASTEXITCODE
