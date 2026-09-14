# 放行 Attest 局域网演示所需端口（5173 前端 / 8000 后端）。
#
# ⚠️ 需要**管理员权限**运行：
#     右键「以管理员身份运行 PowerShell」，然后执行
#       powershell -ExecutionPolicy Bypass -File scripts\lan_firewall.ps1
#     或直接在管理员 PowerShell 里跑本文件。
#
# 卸载（演示完把口子关掉）：
#     Remove-NetFirewallRule -DisplayName "Attest LAN Demo"
#
# 为什么只放行 -Profile Private：
#     Private = 家庭/专用网络，Public = 咖啡厅/机场这类公共网络。
#     只在专用网络上开口子，能避免"在公共 WiFi 里把调研页面暴露给陌生人"。
#     代价是：如果你当前把网络标记成了「公用」，规则不会生效——
#     那时应该改网络类型，而不是给 Public 也开口子。

$ErrorActionPreference = 'Stop'

$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host ""
    Write-Host "  需要管理员权限。" -ForegroundColor Red
    Write-Host "  请右键「以管理员身份运行 PowerShell」后重新执行本脚本。" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$ruleName = 'Attest LAN Demo'
$ports = '5173,8000'

# 先删旧规则：避免重复执行时堆出多条同名规则（排查时很难看出哪条在生效）
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Description 'Attest 质证 · 局域网演示（前端 5173 / 后端 8000）' `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort $ports `
    -Profile Private | Out-Null

Write-Host ""
Write-Host "  已放行 TCP $ports（仅专用网络配置）。" -ForegroundColor Green
Write-Host "  演示结束后建议关闭：" -ForegroundColor Gray
Write-Host "    Remove-NetFirewallRule -DisplayName `"$ruleName`"" -ForegroundColor Gray
Write-Host ""
