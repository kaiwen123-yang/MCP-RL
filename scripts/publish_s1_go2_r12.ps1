param(
    [ValidateSet('upload','finish')][string]$Action = 'upload',
    [string[]]$Attempt = @(),
    [string[]]$Asset = @(),
    [switch]$Dependencies,
    [Parameter(Mandatory=$true)][double]$DeadlineUnix,
    [string]$SshHost = 'fy-5080',
    [string]$RemoteRoot = '/home/fanyuan/projects/kaiwen-learning/terrain-memory-r12'
)
# Only API metadata crosses this machine. Artifact bytes stay on the execution host.
$ErrorActionPreference = 'Stop'
$repo = 'kaiwen123-yang/MCP-RL'
$tag = 'tm-s1-go2-local-readiness-r12'
$release = gh api "repos/$repo/releases/tags/$tag" | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Release metadata request failed.' }
$pages = gh api "repos/$repo/releases/$($release.id)/assets?per_page=100" --paginate --slurp | ConvertFrom-Json -NoEnumerate
if ($LASTEXITCODE -ne 0) { throw 'Asset metadata request failed.' }
$remoteAssets = [System.Collections.Generic.List[object]]::new()
foreach ($page in $pages) { foreach ($item in $page) { $remoteAssets.Add($item) } }
$credentials = @{ token=(gh auth token); release_metadata=$release; remote_assets=@($remoteAssets.ToArray()) } | ConvertTo-Json -Depth 30 -Compress
$remaining = [int][math]::Floor($DeadlineUnix - [DateTimeOffset]::UtcNow.ToUnixTimeSeconds())
if ($remaining -le 0) { throw 'The original absolute window deadline has passed.' }
$arguments = @($Action, '--deadline-unix', $DeadlineUnix.ToString([cultureinfo]::InvariantCulture))
foreach ($name in $Attempt) { $arguments += @('--attempt', $name) }
foreach ($name in $Asset) { $arguments += @('--asset', $name) }
if ($Dependencies) { $arguments += '--dependencies' }
$quoted = $arguments | ForEach-Object { "'" + $_.Replace("'", "'\''") + "'" }
$remote = "cd '$RemoteRoot' && timeout $remaining /home/fanyuan/miniconda3/envs/kaiwen-learn/bin/python -u scripts/publish_s1_go2_r12.py " + ($quoted -join ' ')
$credentials | ssh $SshHost $remote
if ($LASTEXITCODE -ne 0) { throw "Publication command exited with code $LASTEXITCODE." }
