# XTTSv2 Finetuning Guide for New Languages

This guide provides instructions for finetuning XTTSv2 on a new language, using Vietnamese (`vi`) as an example.

[UPDATE] A finetuned model for Vietnamese is now available at [anhnh2002/vnTTS](https://huggingface.co/anhnh2002/vnTTS) on Hugging Face


## Table of Contents
- [XTTSv2 Finetuning Guide for New Languages](#xttsv2-finetuning-guide-for-new-languages)
  - [Table of Contents](#table-of-contents)
  - [1. Installation](#1-installation)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Pretrained Model Download](#3-pretrained-model-download)
  - [4. Vocabulary Extension and Configuration Adjustment](#4-vocabulary-extension-and-configuration-adjustment)
  - [5. DVAE Finetuning (Optional)](#5-dvae-finetuning-optional)
  - [6. GPT Finetuning](#6-gpt-finetuning)
  - [7. 合成音频](#7-合成音频)
    - [通过脚本](#通过脚本)
    - [通过 Web UI](#通过-web-ui)

## 1. Installation

本 repo 已经包含了 [Coqui TTS](https://github.com/coqui-ai/TTS) 的所有必要代码，直接下载本 repo 即可，无需下载 [Coqui TTS](https://github.com/coqui-ai/TTS) 到本地，也无需安装 Coqui TTS

```
git clone https://github.com/EL-CL/XTTSv2-Finetuning-for-New-Languages.git
（现在连接 GitHub 常失败，可下载本 repo 的 ZIP，将其上传至服务器再解压）

cd XTTSv2-Finetuning-for-New-Languages
conda create -n coqui python=3.10 -y
conda activate coqui
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## 2. Data Preparation

数据集的格式：所有音频放在一个文件夹内，文件夹可以放在任意地方。采样率无需调整，训练时会自动转换。每条音频最好不要超过 12 s，否则训练时会遇到数组越界报错

转写文件的格式——`bel_tts_formatter`（用于单发音人或者不区分发音人）：

```
001.wav|How do you do?
002.wav|Nice to meet you.
003.wav|Good to see you.
...
```

转写文件的格式——`coqui`（用于多发音人）：

```
audio_file|text|speaker_name
001.wav|How do you do?|@X
002.wav|Nice to meet you.|@Y
003.wav|Good to see you.|@Z
...
```

（格式出处：[TTS/tts/datasets/formatters.py](TTS/tts/datasets/formatters.py)）

转写中的标点不是必要的。[train_gpt_xtts.py](train_gpt_xtts.py) 第 70 行 `formatter=` 需改为对应的格式名

转写文件可以保存在任意地方（如 `models/metadata.txt`），无需和数据集放在一个文件夹下

如果新增多个语言，需要每个语言准备一个转写文件

## 3. Pretrained Model Download

Hugging Face 墙内无法访问，可自己下载预训练模型后，上传至服务器（如存放在本目录下的 `original_model` 中）

预训练模型在 https://huggingface.co/coqui/XTTS-v2/tree/main，需要下载的文件见 [download_checkpoint.py](download_checkpoint.py) 中的链接

## 4. Vocabulary Extension and Configuration Adjustment

词典扩充：运行 [extend_vocab.py](extend_vocab.py) 以根据新语言的标注扩充词典（详见文件末尾）

配置文件扩充：复制一份预训练模型的 config.json（如复制到 `models/config.json`），在第 130 行 `"languages"` 列表中添加要训练的新语言的语言代码/名称（与 [extend_vocab.py](extend_vocab.py) 中的一致）

## 5. DVAE Finetuning (Optional)

（本项跳过）

To finetune the DVAE, run:

```bash
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=datasets/metadata_train.csv \
--eval_csv_path=datasets/metadata_eval.csv \
--language="vi" \
--num_epochs=5 \
--batch_size=512 \
--lr=5e-6
```

Update: If you have enough short texts in your datasets (about 20 hours), you do not need to finetune DVAE.

## 6. GPT Finetuning

训练参数的设置详见 `train_gpt_xtts.py` 开头的各项赋值

单卡训练：

```bash
TRAINER_TELEMETRY=0 CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py
```

多卡训练：

```bash
TRAINER_TELEMETRY=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m trainer.distribute --script train_gpt_xtts.py
```

其中 `CUDA_VISIBLE_DEVICES` 是需要用到的显卡号。`TRAINER_TELEMETRY` 是[在训练途中向 Coqui 反馈匿名数据](https://github.com/coqui-ai/Trainer#anonymized-telemetry)，不需要开启，否则会因为无法连接外网而训练失败

训练过程中，best model 只会保存一个，可在途中手动复制出来一些保存

Note: Finetuning the HiFiGAN decoder was attempted but resulted in worse performance. DVAE and GPT finetuning are sufficient for optimal results.

## 7. 合成音频

### 通过脚本

运行 `run_tts.py`，相关输入和参数见文件

### 通过 Web UI

安装 `pip install streamlit`，然后执行 `streamlit run run_tts_webui.py`

（安装 streamlit 可能导致 numpy 被更新到 2.x 版本，进而导致运行时报错。此时重新执行 `pip install numpy==1.22.0` 即可）
