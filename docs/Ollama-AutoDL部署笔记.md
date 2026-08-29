# Ollama 在 AutoDL 上的部署笔记

> **环境**: AutoDL 容器(Ubuntu 22.04, x86_64 架构), GPU: RTX 3080 Ti, Ollama 版本 0.33.2
> **目录布局**:
>
> - 系统盘 `/`(30G): 存放代码和系统文件,速度一般
> - 数据盘 `/root/autodl-tmp`(50G): 读写快,适合放大文件(如模型),关机不丢数据
>
> **安装方式**: 手动从 GitHub 下载安装包。官方 `install.sh` 在 AutoDL 的加速代理下会报 404,原因见文末附录。
> **模型获取**: `ollama pull` 直连官方仓库拉不动(网络不通),改用魔搭 ModelScope 下载 GGUF 文件后本地创建,详见第四节。

---

## 一、安装 Ollama

### ① 关闭学术加速

**作用**: 删掉代理环境变量,让 `apt` 直连国内镜像(华为云),下载软件包更快。

```bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

**命令要点**:

- `unset`: 删除 shell 变量的命令,删掉后程序就不知道有代理了,走直连
- `http_proxy` / `https_proxy` / `all_proxy`: 小写的一组代理变量,`curl`、`wget`、`apt` 等工具靠它们走代理
- `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY`: 大写的同一组变量,有些程序只认大写,所以要大小写一起清
- 这 6 个变量就是 AutoDL 学术加速的"开关",`source /etc/network_turbo` 开加速、`unset` 关加速

### ② 安装 zstd

**作用**: 先刷新软件包索引,再安装 `zstd`——新版 Ollama 的安装包是 `.tar.zst` 格式,需要这个工具解压。

```bash
apt-get update && apt-get install -y zstd
```

**命令要点**:

- `apt-get`: Ubuntu/Debian 的软件包管理器
- `update`: 从软件源拉取最新的软件包列表(不装东西,只是同步索引)
- `&&`: 连接符,前一条命令成功才执行后一条,保证索引是最新的再安装
- `install`: 安装软件包
- `-y`: 对所有确认提示自动回答 yes(不手动敲确认)
- `zstd`: zstandard 压缩工具,处理 `.zst` 格式的压缩包

### ③ 开启学术加速

**作用**: 启用 AutoDL 的学术加速,让请求通过代理转发,快速访问 GitHub / HuggingFace——接下来要下载 Ollama 的安装包,文件在 GitHub 上。

```bash
source /etc/network_turbo
```

**命令要点**:

- `source`: 在当前 shell 里直接执行脚本文件。必须用它而不是 `sh` / `./` 执行,否则脚本里设置的代理变量不会保留到当前终端
- `/etc/network_turbo`: AutoDL 预装的加速脚本,执行后自动设置好 ① 里那些代理环境变量
- 注意: 加速只对 GitHub / HuggingFace 生效,访问其他资源反而更慢

### ④ 下载并解压安装包(替代官方 install.sh)

**作用**: 三连命令,把 Ollama 安装包从 GitHub 下载下来、解两层压缩、放到系统目录里。

```bash
curl -fL -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
```

**作用**: 从 GitHub Releases 下载 Ollama 的 Linux 安装包(第一层压缩: `.zst`)。

**命令要点**:

- `curl`: 命令行下载工具,`-o` 指定保存到哪个文件
- `-f`: 服务器返回错误(如 404)时直接失败退出,不会把错误页面当成文件存下来
- `-L`: 跟随重定向。GitHub 的下载链接会先跳到真实的 CDN 地址,不跟就跟丢
- `-o /tmp/ollama.tar.zst`: 把下载内容保存为 `/tmp` 下的 `ollama.tar.zst`
- URL 拆解:
  - `github.com/ollama/ollama`: Ollama 官方仓库
  - `releases/latest/download/`: GitHub 固定写法,自动指向"最新版本"的下载区
  - `ollama-linux-amd64.tar.zst`: 包名 = 系统(`linux`) + 架构(`amd64`) + 压缩格式(`tar.zst`)
- 为什么是 `amd64`: 本机是 x86_64 架构,对应 amd64;如果是 ARM 机器要用 `arm64`

```bash
zstd -d -f /tmp/ollama.tar.zst -o /tmp/ollama.tar
```

**作用**: 解掉第一层 `.zst` 压缩,得到一个 `.tar` 归档文件(里面才是安装内容)。

**命令要点**:

- `zstd`: 就是 ② 里装的压缩工具
- `-d`: 解压模式(decompress)
- `-f`: 强制覆盖,输出文件已存在时不询问直接覆盖(force)
- `-o /tmp/ollama.tar`: 指定解压结果保存为 `ollama.tar`

```bash
tar -xf /tmp/ollama.tar -C /usr/local
```

**作用**: 解掉第二层,把归档内容放进 `/usr/local`(里面含 `bin/ollama` 可执行文件),装完即可直接使用 `ollama` 命令。

**命令要点**:

- `tar`: 打包/解包工具,`-x` 解包(extract)
- `-f`: 指定要处理的归档文件
- `-C /usr/local`: 先切到该目录再解包,也就是"解压到哪里"
- 为什么是 `/usr/local`: 系统装第三方软件的惯例目录,里面的 `/usr/local/bin` 在 PATH 中,所以解压后直接敲 `ollama` 就能找到命令

### ⑤ 验证安装

**作用**: 打印版本号,确认安装成功。

```bash
ollama --version
```

**命令要点**:

- `ollama`: 主程序命令
- `--version`: 参数,要求显示版本号
- 输出里出现 `Warning: could not connect to a running Ollama instance` 是正常的,只是说服务还没启动,不影响安装成功

### ⑥ 清理临时文件(可选)

**作用**: 删掉下载和解压过程中的两个临时文件,释放系统盘空间。

```bash
rm -f /tmp/ollama.tar.zst /tmp/ollama.tar
```

**命令要点**:

- `rm`: 删除文件
- `-f`: 不询问直接删,文件不存在也不报错(force)
- 两个文件加起来约 3.5G,删掉后系统盘会宽裕很多

---

## 二、把模型目录指向数据盘

> 为什么: Ollama 默认把模型存在 `/root/.ollama/models`(系统盘),而系统盘只有 30G,一个模型动辄几 GB 到几十 GB,很快会满。数据盘 50G 且读写更快,所以把模型路径指过去。

### 1. 创建模型目录

**作用**: 在数据盘上建好存放模型的文件夹。

```bash
mkdir -p /root/autodl-tmp/ollama/models
```

**命令要点**:

- `mkdir`: 创建目录
- `-p`: 父目录不存在时自动一并创建,目录已存在也不会报错(parents)
- `/root/autodl-tmp`: AutoDL 数据盘的挂载点,后面的 `/ollama/models` 是我们自己定的层级

### 2. 把环境变量写进 ~/.bashrc

**作用**: 把"模型存放目录"这行配置追加进 `~/.bashrc`,以后每次登录终端自动生效,不用每次手动设置。

```bash
echo 'export OLLAMA_MODELS=/root/autodl-tmp/ollama/models' >> ~/.bashrc
```

**命令要点**:

- `echo '...'`: 把引号里的字符串原样打印出来
- `>>`: 追加写入,把打印的内容加到文件末尾(注意: 单个 `>` 是覆盖整个文件,会清掉原有内容)
- `export`: 把变量导出成"环境变量",这样从当前 shell 启动的所有程序都能读到它
- `OLLAMA_MODELS`: Ollama 专用的环境变量,指定模型存储位置
- `~/.bashrc`: bash 的配置文件,每次打开新终端都会自动执行一遍,是"开机自启动配置"最常用的落点

### 3. 让配置立即生效

**作用**: 马上在当前终端执行一遍 `~/.bashrc`,刚加的变量立刻可用,不用重开终端。

```bash
source ~/.bashrc
```

**命令要点**:

- `source` 的机制和 ③ 一样: 在当前 shell 里执行脚本,变量当场生效

---

## 三、后台启动服务

### 1. 启动服务(挂后台)

**作用**: 后台启动 Ollama 服务,日志写进 `/tmp/ollama.log`,SSH 断开连接后进程也不会被终止。

```bash
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

**命令要点**:

- `nohup`: no hang up 的缩写,忽略终端挂断信号,终端退出后进程继续跑
- `ollama serve`: 前台启动 Ollama 服务(默认监听本机 11434 端口)。注意: 它自己不会后台化,所以要靠外面的手段(这里用 `nohup ... &`,也可以用 screen)
- `>`: 输出重定向,把原本打印到屏幕的内容写进文件
- `/tmp/ollama.log`: 日志文件路径,启动过程、报错都记在里面
- `2>&1`: 把标准错误(编号 2)也重定向到标准输出(编号 1)所指的位置,即错误日志也进同一个文件
- `&`: 放到后台执行,命令立即返回,终端不会被占住
- 重要: `OLLAMA_MODELS`(第二步设置的那个变量)必须在启动 serve **之前**设置好,因为服务启动那一刻就读取它

### 2. 确认服务已启动

**作用**: 向本机的 Ollama 接口发一个请求,返回类似 `{"version":"0.33.2"}` 就说明服务正常在跑。

```bash
curl -s http://127.0.0.1:11434/api/version
```

**命令要点**:

- `curl`: 这里用它发 HTTP 请求(不写 `-o`,默认把响应打印到屏幕)
- `-s`: 静默模式,不显示下载进度条(silent)
- `http://127.0.0.1:11434`: 本机回环地址(`127.0.0.1`) + Ollama 默认端口(11434)
- `/api/version`: Ollama 提供的 REST API 接口之一,返回版本信息
- 刚启动可能慢一两秒,没输出就再执行一次

---

## 四、获取模型(魔搭下载 GGUF)

> 为什么不用 `ollama pull`: 官方模型仓库 `registry.ollama.ai` 的下载走 Cloudflare,国内直连很不稳定,而且它也不在 AutoDL 学术加速的白名单里,所以 `ollama pull qwen2.5:7b` 拉不下来。
>
> 替代方案: 魔搭 ModelScope 是阿里的平台,AutoDL 也在阿里云上,下载走国内链路非常快。魔搭上有和官方同源的 GGUF 格式模型文件,而 GGUF 正是 Ollama 能直接加载的格式,下载后让 Ollama"认领"它即可,效果和 `ollama pull` 等价。

### 1. 安装 modelscope 工具

**作用**: 安装魔搭的命令行工具,装完就有 `modelscope` 命令用于下载模型。

```bash
pip install modelscope
```

**命令要点**:

- `pip`: Python 的包管理器
- `install`: 安装软件包
- `modelscope`: 阿里开源的模型下载/使用工具(魔搭)。AutoDL 的 pip 已配置国内镜像,这一步装起来很快,不需要开加速

### 2. 下载 GGUF 模型文件

**作用**: 从魔搭下载 Qwen2.5-7B 的 GGUF 文件,只取 `q4_k_m` 这一个量化档位,存到当前目录下的 `qwen2.5-7b-gguf` 文件夹里。

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include 'qwen2.5-7b-instruct-q4_k_m-*' \
  --local_dir ./qwen2.5-7b-gguf
```

**命令要点**:

- `modelscope download`: 魔搭的"下载模型"子命令
- `--model Qwen/Qwen2.5-7B-Instruct-GGUF`: 模型 ID,格式是 `组织/模型名`,指魔搭上的这个仓库(可以按需换成其他模型,如 `Qwen/Qwen2.5-14B-Instruct-GGUF`)
- `\`: 行尾的反斜杠是**续行符**,表示"命令还没写完,下一行是它的继续"——纯粹为了把长命令拆成多行更好看,执行效果和写在一行完全相同
- `--include 'qwen2.5-7b-instruct-q4_k_m-*'`: 只下载文件名匹配这个模式的文件
  - `*` 是通配符,代表任意字符;结尾的 `-*` 能匹配到该量化档位拆出来的**所有分片文件**(大模型一个文件放不下,会拆成多个,名字形如 `...-00001-of-00002.gguf`)
  - 不加 `--include` 会把仓库里所有量化版本(`q4_k_m`、`q8_0` 等)全部下载,文件又多又占空间——之前"下了太多文件"就是这个原因
  - 模式必须用**单引号**包起来,否则 `*` 会被 shell 提前当成文件名展开,匹配就失效了
- `--local_dir ./qwen2.5-7b-gguf`: 下载到哪个本地目录(不写的话会存到默认缓存目录)
- GGUF: 一种把模型权重量化压缩后的文件格式,Ollama 官方仓库里的模型本质上就是 GGUF 文件
- 下载完成后 `ls ./qwen2.5-7b-gguf` 看一眼实际文件名(现在只会是 `q4_k_m` 档,通常是按序号拆开的多个分片),下一步 `FROM` 要指到第一个分片

### 3. 整理分片文件

**作用**: 把下载到的 GGUF 分片单独放进一个干净的目录,方便下一步让 `ollama create` 一次识别全部文件。

```bash
mkdir -p ~/model-import
mv ./qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m-0000*.gguf ~/model-import/
cd ~/model-import
```

**命令要点**:

- `mkdir -p ~/model-import`: 创建新目录,`-p` 表示父目录不存在时自动一并创建;`~/model-import` 是家目录下的 `model-import` 文件夹(名字可以随意起)
- `mv 源文件 目标目录`: 移动文件。`qwen2.5-7b-instruct-q4_k_m-0000*.gguf` 里的 `*` 是通配符,会一次匹配 00001、00002 等全部分片
- `cd ~/model-import`: 把当前工作目录切到模型目录,下一步要在这个目录里执行
- 为什么要单独放: `ollama create` 会自动把**当前目录里**的 `.gguf` 文件当作模型源,单独放可以保证它只看到分片、不混进无关文件
- 关键点: 保持分片原样即可,**不要手动用 `cat` 把分片拼成一个文件,也不要让 Modelfile 只指向单个分片**——分片内部带有分割信息,只有让 Ollama 自己按序号合并才是对的

### 4. 创建 Ollama 模型

**作用**: 用当前目录里的 GGUF 分片构建模型,命名为 `qwen2.5:7b`。这一步不需要写 Modelfile,`ollama create` 会自动识别目录里的 `.gguf` 并合并分片。

```bash
ollama create qwen2.5:7b
```

**命令要点**:

- `ollama create`: 用本地文件构建模型的子命令
- `qwen2.5:7b`: 给模型起的名字(格式 `模型名:标签`),建议和官方命名保持一致,方便记忆和后续换命令
- 为什么没有 `-f Modelfile`: 当当前目录存在 `.gguf` 文件时,`ollama create` 会直接以它们为模型源;多个分片会被自动按 `-00001-of-00002` 的序号合并成一个完整模型
- 执行前提: 服务必须已在运行(第三节),否则会报连接失败,先跑第三节的 `nohup ollama serve ...`
- 成功后 `ollama list` 能看到 `qwen2.5:7b`(约 4.7GB);效果与 `ollama pull qwen2.5:7b` 等价

---

## 五、运行模型

**作用**: 启动模型,进入交互式对话(输入 `/bye` 退出)。

```bash
ollama run qwen2.5:7b
```

**命令要点**:

- `ollama run`: 运行模型
- 对话过程中可以在另一个终端跑 `nvidia-smi`,看到 ollama 进程占用显存就说明模型跑在 GPU 上而不是纯 CPU

---

## 附录

### 为什么不用官方安装脚本

官方 `curl -fsSL https://ollama.com/install.sh | sh` 在 AutoDL 上会报 `curl: (22) 404`,原因是两个问题叠加:

1. **加速代理搞挂了 `ollama.com`**: 安装脚本内部要从 `ollama.com` 下载二进制包,而这个域名不在 AutoDL 加速的白名单里,代理处理不了,请求直接 404。
2. **新版不再发布 `.tgz` 包**: 脚本先探测 `.tar.zst` 是否存在(存在就下载它),探测失败才退回 `.tgz`。代理导致探测失败,而最新版 Ollama 只发 `.tar.zst`、没有 `.tgz`,兜底路径也必然 404。

所以改为绕过脚本、直接从 GitHub Releases(在加速白名单内)下载 `.tar.zst` 手动安装,即第一步 ④ 的做法。

### 为什么 `ollama pull` 拉不动模型

`ollama pull` 从官方仓库 `registry.ollama.ai` 下载,该服务的 CDN 走 Cloudflare,国内直连不稳定;同时它不在 AutoDL 学术加速的白名单里,加速帮不上忙甚至更慢。所以改用第四节"魔搭下载 GGUF + `ollama create`"的方式,绕开这条网络链路。

### 常用命令备忘

| 命令                              | 作用                                 |
| --------------------------------- | ------------------------------------ |
| `ollama list`                     | 查看本地已下载的模型                 |
| `ollama ps`                       | 查看当前正在运行的模型(占用情况)     |
| `ollama create 名字`          | 用当前目录的 GGUF 文件创建模型(自动合并分片) |
| `ollama rm 模型名`                | 删除本地模型(释放磁盘)               |
| `ollama stop 模型名`              | 停止正在运行的模型                   |
| `pkill ollama`                    | 停止整个 ollama 服务(需重新 serve)   |
| `tail -f /tmp/ollama.log`         | 实时查看服务日志                     |
| `nvidia-smi`                      | 查看 GPU 占用,验证模型是否跑在显卡上 |
