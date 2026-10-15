param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$upstreamRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot 'work/four-paper-audit-2026-10-14'))
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'pilot-upstreams.json') -Raw | ConvertFrom-Json

Get-Command git -ErrorAction Stop | Out-Null
Get-Command gh -ErrorAction Stop | Out-Null
gh auth status --hostname github.com 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Authenticate GitHub CLI with gh auth login before running this bootstrap.'
}

foreach ($upstream in $manifest.repositories) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $upstreamRoot $upstream.directory))
    $allowedPrefix = $upstreamRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $target.StartsWith($allowedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Upstream target is outside the declared work directory: $target"
    }
    if (-not (Test-Path -LiteralPath $target)) {
        if ($CheckOnly) {
            throw "Missing upstream checkout: $target. Run without -CheckOnly to clone it."
        }
        New-Item -ItemType Directory -Path $upstreamRoot -Force | Out-Null
        # Clone only a previously absent destination; never reset an existing tree.
        gh repo clone $upstream.remote $target -- --no-checkout --config core.autocrlf=false
        if ($LASTEXITCODE -ne 0) { throw "Clone failed for $($upstream.name); no directory was deleted." }
        git -C $target checkout --detach $upstream.commit
        if ($LASTEXITCODE -ne 0) { throw "Pinned checkout failed for $($upstream.name); preserve the directory for inspection." }
    }

    $actualRemote = git -C $target remote get-url origin
    if ($LASTEXITCODE -ne 0) { throw "Cannot read origin for existing path: $target" }
    if ($actualRemote.Trim() -cne $upstream.remote) {
        throw "Origin differs from the verified manifest for $($upstream.name). Existing files were not changed."
    }
    $actualCommit = git -C $target rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actualCommit.Trim() -ne $upstream.commit) {
        throw "Commit differs from the pinned manifest for $($upstream.name). Existing files were not changed."
    }
    $changes = git -C $target status --porcelain
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect the existing checkout for $($upstream.name)." }
    if ($changes) { throw "The $($upstream.name) checkout has local changes; bootstrap will not reset it." }
    [PSCustomObject]@{ Name=$upstream.name; Commit=$actualCommit.Trim(); Origin=$actualRemote.Trim(); Status='verified' }
}

Write-Output 'Pinned source checkouts are ready. No upstream setup script or training job was executed.'
