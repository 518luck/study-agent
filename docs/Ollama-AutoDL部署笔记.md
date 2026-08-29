# Ollama 在 AutoDL 上的部署笔记

> **环境**: AutoDL 容器(Ubuntu 22.04, x86_64), GPU: RTX 3080 Ti, Ollama 版本 0.33.2
> **目录**:
>
> - 系统盘 `/`(30G): 存代码和系统文件,模型会默认放在 `/root/.ollama`(系统盘)
> - 数据盘 `/root/autodl-tmp`(50G): 读写快,关机不丢数据,但**重置系统会丢**
>
> **本次实测结论**:
>
> - 官方 `install.sh` 在开启 AutoDL 学术加速后**可以直接装成功**,不需要手动下载安装包
> - `ollama run` 官方仓库拉模型会卡在 Cloudflare R2(TLS 超时),**用本地 SS 代理绕过去**
> - AutoDL 的公网端口映射(6006/6008)个人账号访问会被网关拒绝(403),**本机调用走 SSH 隧道**

---

## 一、安装 Ollama

### ① 安装 zstd

**作用**: 刷新软件包索引并安装 `zstd`——install.sh 下载的新版安装包是 `.tar.zst` 格式,需要它解压。

```bash
apt-get update && apt-get install -y zstd
```

**命令要点**:

- `apt-get`: Ubuntu 软件包管理器;`update` 同步索引;`install -y` 自动确认安装
- `zstd`: zstandard 压缩工具
- 不开学术加速直接走国内镜像源(华为云),很快;`&&` 保证索引更新成功后再装

### ② 开启学术加速

**作用**: 让后面的 install.sh 能下载到 Ollama 的安装包(文件在 GitHub / ollama.com 上,直连很慢甚至卡死)。

```bash
source /etc/network_turbo
```

**命令要点**:

- `source`: 在当前 shell 里执行脚本,否则脚本里的代理环境变量不会保留
- 只对 GitHub / HuggingFace 等学术域名生效;注意**开着加速时访问国内资源会更慢**,所以 ① 先做

### ③ 安装 Ollama

**作用**: 官方一条命令安装,装到 `/usr/local`,并创建 ollama 用户。

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

实测输出:下载 100% → 创建 ollama user → 完成。出现以下警告均为正常,可忽略:

- `WARNING: systemd is not running` — 容器里没有 systemd,服务不会开机自启,后面用 nohup 自己起
- `WARNING: Unable to detect NVIDIA/AMD GPU` — 没装 lspci 的误报,实际能正常用 GPU

验证安装(此时服务还没跑,出现 `could not connect` 警告是正常的):

```bash
ollama --version
# Warning: client version is 0.33.2
```

---

## 二、下载模型(挂本地代理)

> 为什么: `ollama pull` / `ollama run` 从官方仓库拉模型,大文件重定向到 Cloudflare R2(国内直连不稳定,TLS 握手超时),且不在学术加速白名单里。方案:在服务器上起一个本地 SS 隧道代理,让 Ollama 走代理下载。**(如果你有自己现成的代理链路可按同样思路替换下面 ss-local 一条)**

### ① 安装 shadowsocks 客户端

```bash
apt-get install -y shadowsocks-libev
```

### ② 起本地 SOCKS5 代理

**作用**: 在本机 1080 端口开一个 SOCKS5 代理,流量经中转服务器(节点地址/端口/密码/加密方式来自你的订阅,换成你自己的即可)。

```bash
nohup ss-local -s 207.56.230.16 -p 55412 \
  -k "6d62c483-c85e-463f-a9bd-c32cbc58a829" \
  -m chacha20-ietf-poly1305 \
  -b 127.0.0.1 -l 1080 > /tmp/sslocal.log 2>&1 &
```

**命令要点**:

- `nohup ... &`: 后台运行,SSH 断开不退出;`> log 2>&1`: 日志进文件
- `-s/-p`: 中转服务器地址和端口;`-k`: 密钥;`-m`: 加密方式;`-l 1080`: 本地监听端口
- `-b 127.0.0.1`: 只监听本机,不暴露公网

### ③ 验证代理连通

**作用**: 返回 HTTP 状态码即说明隧道通(404 是正常现象,连上了 registry 但根路径本来就没有内容)。

```bash
curl -sI --max-time 15 --socks5 127.0.0.1:1080 https://registry.ollama.ai | head -1
# HTTP/2 404
```

### ④ 重启服务并注入代理

**作用**: 模型下载是 `ollama serve` 进程干的,必须把代理放进它的环境变量里(`env -u` 是为了清掉学术加速设的其他代理变量,防止干扰)。

```bash
pkill -f "ollama serve"; sleep 1
env -u http_proxy -u https_proxy -u HTTP_PROXY -u all_proxy -u ALL_PROXY \
  HTTPS_PROXY=socks5://127.0.0.1:1080 \
  nohup ollama serve > /tmp/ollama.log 2>&1 &
```

> 如果这个 shell 从未 `source /etc/network_turbo` 过,**可以直接省略 `env -u ...` 那一段**。

### ⑤ 拉取模型

```bash
ollama run qwen:4b
```

下载 2.3GB 完成后自动进入 `>>>` 对话(输入 `/bye` 或 `Ctrl+D` 退出)。`qwen:4b` 约 40 亿参数,3080 Ti 跑起来很快。

---

## 三、日常使用:起服务、聊天

> 容器没有 systemd,Ollama 不会开机自启。**SSH 重新连接或实例重启后,服务不会在跑**,先执行 ①。

① 启动服务(不带代理,日常直连):

```bash
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

② 验证 + 进入对话:

```bash
ollama list              # 能看到 qwen:4b 列表
ollama run qwen:4b       # 直接进对话,不会再下载
```

其他常用:`ollama ps`(当前运行中的模型)、`nvidia-smi`(看显存占用,确认跑在 GPU 上)、`tail -f /tmp/ollama.log`(看服务日志)。

---

## 四、本机(Mac)调用:SSH 隧道

> AutoDL 实例没有公网 IP,公网端口映射(6006/6008)个人账号会被网关 403 拒绝;官方对个人用户推荐的就是 SSH 隧道。

### ① 本机开隧道(保持终端开着)

```bash
ssh -N -C -g -o ServerAliveInterval=60 \
  -L 11434:127.0.0.1:11434 \
  root@connect.westb.seetacloud.com -p 31563
```

**前提**: 服务器上 `ollama serve` 正在跑(第三节 ①)。

**要点**:

- `-L 11434:127.0.0.1:11434`: 把服务器的 11434 映射成本机的 11434
- `-N`: 纯转发、不登录 shell;缺了它 SSH 会给你一个服务器终端,容易误操作
- 看到 Welcome 后**一直安静挂着就是正常**;`ServerAliveInterval=60` 保活防断

### ② 本机调用

```bash
curl http://localhost:11434/api/version
# {"version":"0.33.2"}
```

之后本机所有程序(Cherry Studio / Chatbox / Python 代码)**统一填 `http://localhost:11434`**(Ollama 类型,模型 `qwen:4b`)。

---

## 五、清理

| 场景                               | 命令                             |
| ---------------------------------- | -------------------------------- |
| 停服务                             | `pkill -f "ollama serve"`        |
| 停本地代理                         | `pkill ss-local`                 |
| 删除指定模型                       | `ollama rm qwen:4b`              |
| 清理整块模型残留(之前放在数据盘的) | `rm -rf /root/autodl-tmp/ollama` |

---

## 附录:排障速查

### 1. `Error: max retries exceeded ... TLS handshake timeout`(拉模型超时)

原因: 大文件下载落点在 `*.r2.cloudflarestorage.com`(Cloudflare R2),国内直连不稳定,学术加速白名单也不含它。解决: 走第二节的本地代理(或换任何可用代理),让整个 serve 进程带 `HTTPS_PROXY` 运行。

### 2. 公网映射 6006/6008 访问 403

AutoDL 控制台显示的公网地址(`https://u1147255-...:8443`)对个人账号实际是网关直接拒绝(403)。官方文档:**开放公网端口需要企业认证**;个人用户推荐 SSH 隧道。给朋友访问需要一台有公网 IP 的机器做中转(如 frp / `ssh -R`,或给朋友建隧道)。

### 3. `screen: Cannot find terminfo entry for 'xterm-ghostty'`

本机 Ghostty 的终端类型在服务器 terminfo 里不存在。修法(二选一): 本机 Ghostty 配置加 `shell-integration-features = ...ssh-terminfo...`(自动同步 terminfo);或干脆不用 screen,用 `nohup … &` 起服务(本文全程用 nohup);临时应急 `TERM=xterm screen -S ollama ollama serve`。

### 4. 模型存放位置

默认 `/root/.ollama/models`(系统盘 30G)。qwen:4b 约 2.4G,系统盘足够;若以后下大模型,可 `export OLLAMA_MODELS=/root/autodl-tmp/ollama/models`(写进 `~/.bashrc` 永久生效),再重启 serve。
