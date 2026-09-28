# FR_G1C_NETWORK_DIAGNOSTIC — Network Status Report

> 执行时间：2026-09-17 ~19:58 UTC
> 目的：判断当前 HTTPS 出口故障是否存在替代路径（不下载 checkpoint）
> 原始日志：`network_diag/checks_*.log`
> 本文档不含任何科学实验数据，不修改任何 A0/G1/G2/G3 冻结制品。

## 执行摘要

| 检查项 | 结果 |
|---|---|
| 1. curl -4 HTTPS | ❌ TLS `unexpected eof` |
| 1b. curl -6 HTTPS | ❌ 无 IPv6 出口 |
| 2. HTTP (port 80) | ❌ Empty reply from server |
| 3. DNS 解析 | ⚠️ 返回 Clash fake-ip (198.18.x.x) |
| 4. 默认路由 | ✅ eth0 → 172.30.144.1 |
| 5. 环境 proxy 变量 | ❌ 无 |
| 6. 节点拓扑 | WSL2 (NAT 模式) ↔ Windows 11 + Clash Verge TUN |
| 7. TLS 握手详情 | 客户端写 1569B → 服务器读 0B → RST |
| 8. 代理可达性 | ❌ 127.0.0.1:7897 refused；172.30.144.1:7897 超时 |
| 9. Clash 状态 | ✅ 运行中，mixed-port 7897 LISTENING |
| 10. REST API | ❌ external-controller 为空，Unix pipe 不可达 |
| 11. Verge 配置 | mixed-port 7897, allow-lan: false, tun: gvisor |
| 12. DNS hijack | ❌ TCP 53 无响应 |
| 13. 直接 IP 443 | ❌ TLS 同样被 reset |
| 14. WSL 网络模式 | NAT（非 mirrored） |
| 15. Windows netstat | TUN 接口 198.18.0.1 有活跃连接 |

## 1. curl -4 / -6 HTTPS

```
curl -4 https://olmo-checkpoints.org
→ curl: (35) TLS connect error: error:0A000126:SSL routines::unexpected eof while reading
→ remote_ip=(空)  time=5.00s

curl -6 https://olmo-checkpoints.org
→ curl: (6) Could not resolve host: olmo-checkpoints.org
→ 无 IPv6 默认路由（ip -6 route show default 为空）
```

## 2. HTTP (port 80)

```
curl http://olmo-checkpoints.org
→ curl: (52) Empty reply from server
→ remote_ip=198.18.0.17

curl http://www.google.com
→ curl: (52) Empty reply from server
```

## 3. DNS 解析

```
getent hosts olmo-checkpoints.org  → 198.18.0.17
getent hosts www.google.com        → 198.18.0.6
getent hosts huggingface.co        → 198.18.0.7

/etc/resolv.conf nameserver: 172.30.144.1
```

**关键发现**：DNS 返回 `198.18.0.0/16` 网段——这是 Clash TUN 的 **fake-ip 池**。DNS 解析本身正常，但返回的是 TUN 设备的虚拟 IP。

## 4. 默认路由

```
default via 172.30.144.1 dev eth0 proto kernel
eth0: 172.30.159.154/20
```

WSL2 默认 NAT 模式，默认网关 = Windows 主机 vEthernet (WSL) 适配器。

## 5. 环境 proxy 变量

```
(无 proxy 环境变量)
git config http.proxy  → (未设置)
git config https.proxy → (未设置)
```

## 6. 节点拓扑

```
hostname: DESKTOP-I00SAUH
kernel:   Linux 6.18.33.2-microsoft-standard-WSL2
hypervisor flag: 是（WSL2 运行在 Windows Hyper-V 之上）

Windows 注册表:
  ProxyEnable = 0x1
  ProxyServer = 127.0.0.1:7897

Clash Verge 安装路径:
  C:\Program Files\Clash Verge
  C:\Users\win10\AppData\Roaming\io.github.clash-verge-rev.clash-verge-rev
```

## 7. TLS 握手详情

```
openssl s_client -connect olmo-checkpoints.org:443 -servername olmo-checkpoints.org

CONNECTED(00000003)
SSL handshake has read 0 bytes and written 1569 bytes
Verification: OK
New, (NONE), Cipher is (NONE)
Protocol: TLSv1.3
```

**解读**：TCP 三次握手成功，客户端发出 TLS ClientHello（1569 字节），但服务器端立即关闭连接（RST/FIN），未返回任何 TLS 数据。这是典型的 **TUN 代理路径故障** 特征，而非目标服务器问题。

## 8. 代理可达性

```
# 从 WSL2 内部
bash /dev/tcp/127.0.0.1/7897  → Connection refused
curl -x http://127.0.0.1:7897 → Failed to connect (0 ms)

# 通过 Windows 主机 IP
curl -x http://172.30.144.1:7897 → (超时/无响应)
```

## 9. Clash Verge 状态

```
Windows netstat:
  TCP 127.0.0.1:7897  0.0.0.0:0  LISTENING  ← mixed-port 正常监听
  TCP 127.0.0.1:7897  127.0.0.1:59968  TIME_WAIT  ← 有历史连接
  TCP 198.18.0.1:56256  198.18.0.5:443  ESTABLISHED  ← TUN 有活跃转发
```

Clash 核心（mihomo）**正在运行**，TUN 设备已建立连接。

## 10. REST API 探测

```
GET http://127.0.0.1:9090/version → (空)
GET http://127.0.0.1:9097/version → (空)
GET http://127.0.0.1:7890/version → (空)
GET http://127.0.0.1:7891/version → (空)
```

Verge 配置中 `external-controller: ''`（空字符串）——REST API 未暴露，仅通过 `\\.\pipe\verge-mihomo` Unix pipe 通信，WSL2 无法访问。

## 11. Verge 配置

```yaml
mixed-port: 7897
allow-lan: false
external-controller: ''
tun:
  enable: true
  stack: gvisor
  auto-route: true
  dns-hijack:
    - any:53
dns:
  enhanced-mode: fake-ip
  fake-ip-range: 198.18.0.1/16
```

- `allow-lan: false`：mixed-port 仅绑定 `127.0.0.1`，WSL2 无法通过 `172.30.144.1:7897` 访问
- `external-controller: ''`：REST API 未启用，无法远程管理

## 12. DNS hijack 测试

```
TCP 172.30.144.1:53 → (无响应)
dig @172.30.144.1 olmo-checkpoints.org → (无响应)
```

Clash TUN 的 `dns-hijack: any:53` 理论上应拦截所有 DNS 查询，但从 WSL2 发起的 TCP 53 连接未得到响应。

## 13. 直接 IP 连接（绕过 DNS）

```
bash /dev/tcp/1.1.1.1/443 → TCP connect OK
curl --resolve olmo-checkpoints.org:443:1.1.1.1 https://olmo-checkpoints.org
→ TLS unexpected eof（同样失败）
```

即使绕过 DNS fake-ip，直接连接真实 IP，TLS 仍被 reset——**进一步确认故障在 TUN 代理路径，而非 DNS 或目标服务器**。

## 14. WSL 网络模式

```
cat /mnt/c/Users/win10/.wslconfig → (文件不存在)
```

未配置 mirrored 模式，WSL2 使用默认 NAT。在此模式下：
- WSL2 的 `127.0.0.1` ≠ Windows 的 `127.0.0.1`
- WSL2 需要通过 `172.30.144.1`（默认网关）访问 Windows 服务
- 但 Clash `allow-lan: false` 使 mixed-port 仅绑定 Windows 的 `127.0.0.1`

## 15. Windows netstat TUN 接口

```
TCP 198.18.0.1:139  0.0.0.0:0  LISTENING  ← Clash TUN SMB
TCP 198.18.0.1:56256  198.18.0.5:443  ESTABLISHED  ← 活跃代理连接
TCP 198.18.0.1:61465  8.162.1.65:8888  ESTABLISHED  ← 活跃代理连接
```

TUN 设备 `198.18.0.1` 有活跃连接，说明 Clash 核心本身**仍在正常工作**，但新连接建立失败。

---

## 根因分析

```
WSL2 (NAT)
  │
  ▼
Windows 11 Clash Verge (mihomo) TUN 模式
  │  mixed-port: 7897 (仅 127.0.0.1)
  │  allow-lan: false
  │  external-controller: '' (REST API 关闭)
  │  tun: gvisor stack
  │
  ▼
出口网络 (代理服务器 → 目标)
```

**故障点**：从 WSL2 发出的 TCP 连接到达 Windows 后，经 Clash TUN 转发时失败。TCP 握手能完成（说明 TUN 路由正常），但 TLS ClientHello 发出后立即被 RST（`read 0 bytes`）。可能原因：

1. **Clash TUN 代理上游故障**：代理服务器（Clash config 中的 proxy-server）连接失败或证书问题
2. **gvisor stack 兼容性问题**：WSL2 NAT + gvisor TUN 在某些 Windows 版本上有已知兼容性问题
3. **Clash 配置变更**：`external-controller: ''` 可能是近期修改，导致管理 API 关闭

## 替代路径评估

| 路径 | 可行性 | 说明 |
|---|---|---|
| WSL2 → `127.0.0.1:7897` | ❌ | WSL2 NAT 模式下 127.0.0.1 隔离 |
| WSL2 → `172.30.144.1:7897` | ❌ | `allow-lan: false`，仅绑 127.0.0.1 |
| WSL2 → 直接出口（无代理） | ❌ | TUN `auto-route: true` 已劫持所有流量 |
| WSL2 → Clash DNS 53 | ❌ | TCP 53 无响应 |
| 修改 WSL 为 mirrored 模式 | ⚠️ | 需重启 WSL + 管理员权限，可能影响现有环境 |
| 开启 Clash `allow-lan: true` | ⚠️ | 需 Windows 侧修改配置 + 重启 Clash |
| 本地下载 + scp 上传 | ✅ | 用户已拒绝（"我就不本地下载了"） |
| **等待 Clash/网络自愈** | ✅ | 当前策略：下载循环持续重试 |

## 结论

**当前无可用替代路径。** 故障位于 WSL2 ↔ Windows Clash TUN 代理之间，非目标服务器问题。恢复依赖：

1. Clash Verge 代理服务器自愈（上游恢复）
2. 用户手动重启 Clash Verge 或切换节点
3. 用户修改 Clash 配置（`allow-lan: true` + mirrored WSL 模式）

后台下载循环（`g1c_phase2_downloads.sh`）继续运行，一旦网络恢复将自动续传 checkpoint。
