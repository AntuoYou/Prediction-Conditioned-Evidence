param(
    [Parameter(Mandatory = $true)]
    [string]$RepoPath
)

$resolvedRepo = (Resolve-Path -LiteralPath $RepoPath).Path
$gitMarker = Join-Path $resolvedRepo '.git'
if (-not (Test-Path -LiteralPath $gitMarker)) {
    throw "RepoPath is not a Git working tree: $resolvedRepo"
}

$targetDir = Join-Path $resolvedRepo 'seadronesse_supplement_queue'
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Get-ChildItem -LiteralPath $PSScriptRoot -File |
    Where-Object { $_.Name -ne 'install_into_repo.ps1' } |
    Copy-Item -Destination $targetDir -Force

Write-Host "Copied experiment tools to $targetDir"
git -C $resolvedRepo status --short
