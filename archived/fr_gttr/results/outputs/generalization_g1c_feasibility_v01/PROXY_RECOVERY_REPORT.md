# FR_G1C_PROXY_RECOVERY_CHECK — Proxy Recovery Report

> 执行时间：2026-09-17 ~20:05 UTC
> 目的：寻找当前节点恢复 HTTPS 的最短路径（不修改实验、不下载 checkpoint）
> 原始日志：`network_diag/recovery_*.log`

## 执行摘要

| 检查项 | 结果 |
|---|---|
| 1. WSL → gateway:7897 | ❌ TCP SYN 无响应（超时，exit=124） |
| 1b. WSL → gateway:7890 | ❌ 同上 |
| 2. env proxy | ❌ 无 |
| 3. ss -lntp (WSL) | 仅 VSCode/SSH/DNS，无代理 |
| 4a. Windows netstat | 127.0.0.1:7897 LISTENING (PID 29564, mihomo) |
| 4b. Windows 防火墙 | BlockInbound（默认拦截入站） |
| 4c. Windows WSL 适配器 | 172.30.144.1，仅 139 (SMB) 监听 |
| 5. 其他 Windows 端口 | 3001=HTTP 404，22=SSH，其余拒绝 |
| 6. SSH WSL→Windows | ❌ Permission denied（无密钥/密码） |
| 7. SSH 密钥 | WSL 有 authorized_keys（公钥），无私钥 |
| 8. mihomo 绑定 | 仅 127.0.0.1:7897，未绑 172.30.144.1:7897 |

## 1. WSL → Windows 宿主代理可达性

```
gateway: 172.30.144.1

curl -x http://172.30.144.1:7897 https://olmo-checkpoints.org
→ Trying 172.30.144.1:7897... (hang, 8s timeout)

curl -x http://172.30.144.1:7890 https://olmo-checkpoints.org
→ Trying 172.30.144.1:7890... (hang, 8s timeout)

bash /dev/tcp/172.30.144.1/7897 → exit=124 (timeout)
bash /dev/tcp/172.30.144.1/7890 → exit=124 (timeout)
```

**结论**：Windows 防火墙（BlockInbound）拦截了来自 WSL2 NAT 网络的所有入站 TCP 连接。Clash mixed-port 仅绑定 `127.0.0.1`，且 `allow-lan: false` 阻止了 LAN/WSL 访问。

## 2. 环境变量

```
env | grep -i proxy → (无)
```

## 3. WSL 监听端口 (ss -lntp)

```
127.0.0.1:51872  VSCode
127.0.0.1:47431  VSCode
127.0.0.1:44929  VSCode
...
0.0.0.0:22       SSH (sshd)
127.0.0.53:53    systemd-resolved
```

无代理软件在 WSL 内运行。

## 4. Windows 监听端口 (netstat -an)

```
127.0.0.1:7897   LISTENING  PID 29564  verge-mihomo.exe  ← Clash 核心
127.0.0.1:3001   LISTENING  PID 9868                     ← HTTP (404)
0.0.0.0:22       LISTENING             OpenSSH_for_Windows
0.0.0.0:8000     LISTENING  PID 6664
0.0.0.0:3389     LISTENING             RDP
...
172.30.144.1:139 LISTENING             SMB (唯一绑 WSL 适配器)
```

**关键**：没有任何代理或 HTTP 服务绑定到 `172.30.144.1`（WSL 适配器）。Clash 仅监听 `127.0.0.1:7897`。

## 5. Windows 防火墙状态

```
netsh advfirewall show currentprofile
状态: 开启
防火墙策略: BlockInbound, AllowOutbound
```

默认拦截所有入站连接。WSL2 NAT 网络（172.30.144.0/20）发出的连接被防火墙拦截。

## 6. SSH 隧道可行性

```
ssh win10@172.30.144.1
→ Permission denied (publickey,password,keyboard-interactive)
```

- WSL 有 `~/.ssh/authorized_keys`（含 2 个公钥），但**无对应私钥**
- Windows 有 `C:\Users\win10\.ssh\authorized_keys.txt`
- 尝试将 WSL 公钥追加到 Windows authorized_keys.txt 被权限系统阻止（未授权持久化）
- 无密码less sudo，无法读取 `/root/.ssh` 或修改 `sshd_config`

## 7. 其他潜在路径

| 端口 | 服务 | 可复用为代理？ |
|---|---|---|
| 3001 | HTTP (404) | ❌ 非代理 |
| 8000 | 未知 | ❌ 连接拒绝 |
| 10000 | 未知 | ❌ 仅 127.0.0.1 |
| 12638 | 未知 | ❌ 仅 127.0.0.1 |
| 15149 | 未知 | ❌ 仅 127.0.0.1 |
| 33331 | Clash Verge UI | ❌ Tauri 桌面应用 |
| 35038 | 未知 | ❌ 仅 127.0.0.1 |
| 36709 | 未知 | ❌ 仅 127.0.0.1 |

---

## 方案分类

### A. 当前用户无需 Windows 权限可执行

| # | 方案 | 可行性 | 操作步骤 | 预计时间 |
|---|---|---|---|---|
| A1 | 等待 Clash/网络自愈 | ✅ 当前正在执行 | 下载循环持续重试，无额外操作 | 未知（取决于上游） |
| A2 | WSL 内安装轻量代理客户端 | ⚠️ 理论可行 | 在 WSL 内运行 `docker run -d -p 7897:7897 ...` 或安装 `v2ray`/`sing-box`，手动配置代理服务器 | 15-30 分钟 |
| A3 | 使用 Python/pip 安装 `requests[socks]` + 免费 SOCKS5 | ❌ 不推荐 | 免费代理不可靠，可能泄露数据 | - |
| A4 | 修改 WSL /etc/hosts 指向其他 CDN | ❌ 不可行 | 目标站无公开 mirror |

**A2 详细步骤**（如用户选择）：
```bash
# 在 WSL 内安装 sing-box（无需 Windows 权限）
curl -fsSL https://sing-box.sagernet.org/install.sh | bash
# 配置 SOCKS5 代理服务器（需用户自备节点信息）
# 或运行 docker 容器（若 WSL2 已启用 docker）
```

### B. 需要 Windows 操作（用户手动执行）

| # | 方案 | 可行性 | 操作步骤 | 预计时间 |
|---|---|---|---|---|
| B1 | **Clash Verge 开启 allow-lan** | ✅ **最短路径** | 1. 打开 Clash Verge → Settings → 允许局域网连接 (allow-lan: true)<br>2. 确保防火墙允许 7897 入站（或临时关闭防火墙）<br>3. WSL 内 `export https_proxy=http://172.30.144.1:7897` | 2 分钟 |
| B2 | Windows 防火墙添加入站规则 | ✅ | 以管理员运行：`netsh advfirewall firewall add rule name="Allow WSL to Clash" dir=in action=allow protocol=TCP localport=7897 remoteip=172.30.144.0/20` | 1 分钟 |
| B3 | 配置 WSL 为 mirrored 模式 | ⚠️ 需重启 WSL | 在 `C:\Users\win10\.wslconfig` 添加 `[wsl2]\nnetworkingMode=mirrored`，重启 WSL | 5 分钟 |
| B4 | 重启 Clash Verge / 切换节点 | ✅ | 关闭并重新打开 Clash Verge，或切换代理节点 | 1 分钟 |
| B5 | 设置 Windows 系统代理 | ⚠️ | 设置 → 网络 → 代理 → 手动设置 `127.0.0.1:7897`，但 WSL NAT 模式下不生效 | - |

**B1 详细步骤**（推荐）：
```
1. Windows 任务栏右键 Clash Verge 图标 → 打开主界面
2. 左侧 Settings → 找到 "Allow LAN" / "允许局域网连接" → 开启
3. Windows 设置 → 网络 → Windows 防火墙 → 高级设置 → 入站规则
   → 新建规则 → 端口 → TCP 7897 → 允许连接 → 名称 "WSL to Clash"
   （或临时关闭防火墙验证）
4. WSL 终端执行:
   export https_proxy=http://172.30.144.1:7897
   export http_proxy=http://172.30.144.1:7897
   curl https://olmo-checkpoints.org
```

### C. 不可行项

| # | 方案 | 原因 |
|---|---|---|
| C1 | WSL 直接访问 127.0.0.1:7897 | WSL2 NAT 模式下 127.0.0.1 隔离 |
| C2 | WSL 访问 172.30.144.1:7897 | 防火墙 BlockInbound + allow-lan: false |
| C3 | SSH 隧道 WSL→Windows→Clash | 无 SSH 密钥/密码，无法认证 |
| C4 | 修改 Clash 配置文件 | 需 Windows 权限写入 C:\Users\win10\AppData\... |
| C5 | 使用 Windows 端口 3001/8000 等 | 非代理服务，无法转发 |
| C6 | 绕过 Clash 直接出口 | TUN auto-route 已劫持所有流量，无法绕过 |

---

## 推荐最短路径

**B1（Clash allow-lan + 防火墙规则）** 是当前最短路径，预计 2-3 分钟恢复 HTTPS：

1. 用户在 Windows 上开启 Clash Verge 的 "Allow LAN"
2. 用户添加 Windows 防火墙入站规则（或临时关闭防火墙）
3. WSL 内 `export https_proxy=http://172.30.144.1:7897`

若用户无法操作 Windows，则 **A1（等待自愈）** 为唯一选择，下载循环持续运行。
