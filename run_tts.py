import os
import datetime
from run_tts_lib import *

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model paths
xtts_checkpoint = "models/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/best_model_99875.pth"
xtts_config = "models/config.json"
xtts_vocab = "models/vocab.json"

speakers = {
    # 合成所参照的音色
    "M": "male.wav",
    "F": "female.wav",
}

text_inputs = {
    # 文件标题: (语言名, 文本)
    # 文本可以是多个句子连在一起，也可以是列表
    # 多个句子连在一起时，后面会由 sent_tokenize 根据标点（必须是西文标点）切分成列表
    "文本1": ("en", "How do you do?"),
    "文本2": ("en", "Nice to meet you. Good to see you."),
    "文本3": ("en", [
        "Good to see you.",
        "Nice to meet you.",
    ]),
}

time = datetime.datetime.now().strftime("%m-%d_%H-%M")
model = load_model(xtts_checkpoint, xtts_config, xtts_vocab, device)
os.makedirs("outputs", exist_ok=True)
for speaker_tag, speaker in speakers.items():
    gpt_cond_latent, speaker_embedding = \
        initialize_speaker(model, f"targets/{speaker}")
    for text_tag, (lang, texts) in text_inputs.items():
        inference(
            split_sentences(texts), lang,
            f"outputs/{time}_{text_tag}_{speaker_tag}.wav",
            model, gpt_cond_latent, speaker_embedding,
            temperature=0.1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=10,
            top_p=0.3,
            speed=1.0,
        )
