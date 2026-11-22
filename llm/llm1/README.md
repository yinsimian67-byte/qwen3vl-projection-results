# QVHighlights 多模态视频定位大作业

每组 4 人，最低配置为一张显存 24 GB 的 NVIDIA GPU。给定一句查询，任务 A 从视频中找出三个候选时间区间，任务 C 用 3B 多模态模型给课程提供的三个候选重新排序。每组在 A 和 C 各实现一项改进。

1. 先看必做实验、提交材料和评分规则。[学生手册](docs/学生手册.md)
2. 在 Linux 实验机解压，进入含本文件的目录，按顺序安装环境、检查数据并运行基线。[实现与实验指南](docs/学生作业说明.md)
3. 选定两个自主实现方向，在 `student_code/` 中开发。[核心模块方法参考](docs/核心模块方法参考.md)
4. 训练与评测时保存逐种子日志、逐查询预测和资源开销，用于提交和课堂汇报。[实验记录与 PPT 模板](docs/实验报告模板.md)

**A 的预提取特征、C 的正式训练清单、固定候选和所需四帧缓存均在包内。** C 可直接用缓存读取 1,248 对训练片段及所有 dev、final 候选，无需原视频。

train 有 1,500 条查询，dev 有 200 条，两者附标注。final 提供 300 条无标签查询。用 dev 选定方案后生成 final 预测，随实验记录提交，由教师统一评分。

## 下载 Qwen2.5-VL-3B-Instruct 基础权重

任务 C 使用 Qwen 官方仓库的固定版本 `6628554`。两个权重分片共 7.51 GB，逐候选和成对微调共用一份。[官方版本文件列表](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3)

在作业包根目录运行以下命令，即可下载两个分片到 `models/Qwen2.5-VL-3B-Instruct/` 并逐个校验 SHA-256。

```bash
python3 scripts/download_model.py --weights-only
```

从国内网络下载时，也可以使用魔搭镜像，同一脚本会按课程固定版本校验文件。

```bash
python3 scripts/download_model.py --weights-only --weights-source modelscope
```

手动下载时，请将下列两个文件按原文件名放入 `models/Qwen2.5-VL-3B-Instruct/`，与 `model.safetensors.index.json` 放在一起。随后运行上面的第一条命令即可校验已有文件。[模型文件校验清单](downloads/qwen_model_manifest.json)

| 下载直链 | 保存文件名 | 大小 |
| --- | --- | ---: |
| [下载第 1 片](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/66285546d2b821cf421d4f5eb2576359d3770cd3/model-00001-of-00002.safetensors?download=true) | `model-00001-of-00002.safetensors` | 3.98 GB |
| [下载第 2 片](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/66285546d2b821cf421d4f5eb2576359d3770cd3/model-00002-of-00002.safetensors?download=true) | `model-00002-of-00002.safetensors` | 3.53 GB |

数据来源为 QVHighlights，视频和文本特征由 Lighthouse 镜像提供。资源按各自许可使用。[QVHighlights / Moment-DETR](https://github.com/jayleicn/moment_detr) · [Lighthouse](https://github.com/line/lighthouse) · [数据许可](upstream/moment_detr/data/LICENSE) · [定位器代码许可](upstream/moment_detr/LICENSE) · [Qwen 许可](models/Qwen2.5-VL-3B-Instruct/LICENSE)
