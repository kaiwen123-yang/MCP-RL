param([Parameter(Mandatory=$true)][int]$LocalProxyPort)
# Start from the user's Windows computer. The SSH alias is kept in ~/.ssh/config.
$taskTunnelArgs = @('-N', '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
    '-R', "127.0.0.1:17897:127.0.0.1:$LocalProxyPort", 'autodl-d57')
$taskTunnel = Start-Process -FilePath 'ssh.exe' -ArgumentList $taskTunnelArgs -WindowStyle Hidden -PassThru
Write-Output ("D57 VPN tunnel PID: " + $taskTunnel.Id)
