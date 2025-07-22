
# clash 安装
参考 https://doc.6bc.net/article/35/

## 一些链接
- clash 各个版本: https://clashcn.com/clash-releases-official 

## clash安装流程
```bash
cd && mkdir clash

# 本站下载文件 https://clashcn.com/clash-releases-official
# amd 架构
wget https://doc.6bc.net/kehuduan/clash/clash-linux-amd64-v1.14.0.gz
# arm 架构 
wget https://github.com/netboy1024/clash/releases/download/v1.18.0/clash-linux-arm64-v1.18.0.gz

# 根据下载的clash 文件名 修改下面的命令

# 解压文件 

gzip -d clash-linux-amd64-v1.14.0.gz 

# 给予权限 

chmod +x clash-linux-amd64-v1.14.0 

# 改名移动 

mv clash-linux-amd64-v1.14.0 /usr/local/bin/clash 

# 查看版本 

clash -v

# 启动 

clash 

# 进入目录 

cd $HOME/.config/clash/ 

# 导入订阅 

wget -O config.yaml http://47.243.226.51/link/3qMT153i5IneLPJN?clash=2

# 启用代理
export http_proxy=http://127.0.0.1:7890\nexport https_proxy=http://127.0.0.1:7890

# 启动clash 并使用外部控制
clash -f $HOME/.config/clash/config.yaml -ext-ctl 127.0.0.1:9091
# 进入 https://clash.razord.top/#/connections 配置代理
# 代理模式使用 全局
# 代理页面选择节点

```
# 环境配置
```bash
conda create -n sam2 python=3.10
conda activate sam2
```

# pytorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# sam1 安装
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git 
```

# sam2 安装
```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```


# Download Checkpoints
```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
