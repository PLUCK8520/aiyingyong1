# 取本机"对外网卡"的 IPv4 地址（供 scripts\start.bat --lan 打印局域网访问地址）。
#
# 为什么不能简单取第一个非 127 的地址——实测本机（2026-09-13）有 8 个 IPv4：
#     169.254.x.x × 6   APIPA（网线没通/无 DHCP 时的自分配地址，别人访问不到）
#     192.168.56.1      VirtualBox 虚拟网卡（只在本机虚拟网络里存在）
#     192.168.1.11      WLAN ← **只有这个是真的**
# 随便挑一个的后果是：脚本打印出一个看着像地址、但同事怎么都打不开的 IP，
# 排查方向会被彻底带偏。
#
# 判据（三条同时满足才是"真网卡"）：
#   ① 网卡处于 Up；
#   ② 不是 Loopback / Tunnel；
#   ③ **有默认网关**——APIPA 与虚拟网卡都没有默认网关，这一条是分水岭。
#
# 输出：一行 IPv4 地址；识别不到则不输出任何内容（调用方按"未识别"处理）。

$ErrorActionPreference = 'SilentlyContinue'

# ---- 首选：NetworkInterface（.NET）----
# 不用 Get-NetIPConfiguration 打底：它在不同 Windows 版本/虚拟化环境下返回结构不一致，
# 实测本机直接返回空（而 ipconfig 明明有值）。.NET 这条路最稳。
$ip = [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
    Where-Object {
        $_.OperationalStatus -eq [System.Net.NetworkInformation.OperationalStatus]::Up
    } |
    Where-Object {
        $_.NetworkInterfaceType -notin @(
            [System.Net.NetworkInformation.NetworkInterfaceType]::Loopback,
            [System.Net.NetworkInformation.NetworkInterfaceType]::Tunnel
        )
    } |
    ForEach-Object { $_.GetIPProperties() } |
    Where-Object { @($_.GatewayAddresses).Count -gt 0 } |
    ForEach-Object { $_.UnicastAddresses } |
    Where-Object {
        $_.Address.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork
    } |
    ForEach-Object { $_.Address.IPAddressToString } |
    Select-Object -First 1

if ($ip) {
    Write-Output $ip
    exit 0
}

# ---- 回退：Get-NetIPConfiguration ----
$cfg = Get-NetIPConfiguration |
    Where-Object { $_.IPv4DefaultGateway -and $_.IPv4Address } |
    Select-Object -First 1
if ($cfg) {
    Write-Output (@($cfg.IPv4Address)[0].IPAddress)
}
