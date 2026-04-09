# 语义增强级联 Web 入侵检测系统
基于 LightGBM + 大语言模型的两级 Web 入侵检测，低误报、可解释，支持 PCAP/CSV 流量分析。

## 功能
- 一级快速筛查：LightGBM
- 二级语义判定：大模型
- 路径白名单，降低误报
- 支持 CLI / HTTP 服务

## 快速运行
pip install -r requirements.txt

# 命令行检测
python realtime_ids_demo.py cli --input test.csv

# HTTP 服务
python realtime_ids_demo.py http --port 8000

## 说明
仅用于安全学习与防御研究
