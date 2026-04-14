# 语义增强级联 Web 入侵检测系统
基于 LightGBM + 大语言模型的两级联动 Web 入侵检测，兼顾速度、准确率与可解释性，专注解决常见 Web 攻击检测。

## 当前支持检测类型
本项目**目前仅针对以下 4 类典型 Web 攻击**进行检测与识别：
- XSS 跨站脚本攻击
- SQL 注入攻击
- 目录枚举 / 路径扫描
- 目录暴力破解

## 核心特性
- 两级级联检测：LightGBM 快速筛查 + 大模型语义精判
- 内置正常路径白名单，大幅降低正常爬虫误报
- 支持 PCAP 流量文件、CSV 格式数据输入
- 命令行批量检测 + HTTP API 服务双模式
- 自带评估工具：准确率、混淆矩阵、误报样本定位

## 技术栈
Python • LightGBM • DeepSeek • Flask • Scapy • Pandas

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

命令行模式

```
运行
python realtime_ids_demo.py cli --input ./datasets/test.csv
```

HTTP 服务模式

```bash
运行
python realtime_ids_demo.py http --host 0.0.0.0 --port 8000
模型评估

```bash
运行
python evaluate_labeled_flows_csv.py --input ./datasets/labeled.csv --dump-errors

```适用场景
Web 入侵检测与异常流量分析
XSS、SQL 注入、目录扫描、暴力破解识别
网络安全研究、教学实验、原型验证```

声明
本项目仅用于网络安全学习与防御研究，请勿用于未经授权的渗透测试或非法行为。
