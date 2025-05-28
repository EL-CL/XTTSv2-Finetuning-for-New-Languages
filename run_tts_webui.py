import os
import re
import datetime
import uuid
import streamlit as st
from run_tts_lib import *


@st.cache_resource(show_spinner="正在加载模型，请稍候……")
def initialize():
    # Device configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Model paths
    xtts_checkpoint = "models/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/best_model_99875.pth"
    xtts_config = "models/config.json"
    xtts_vocab = "models/vocab.json"

    model = load_model(xtts_checkpoint, xtts_config, xtts_vocab, device)
    os.makedirs("outputs", exist_ok=True)
    return model


# 合成所参照的音色
speakers = os.listdir("targets")
speaker_default = "Rogger.wav"
speaker_default_index = 0
if speaker_default in speakers:
    speaker_default_index = speakers.index(speaker_default)

# 显示的语言名: 语言代码
langs = {
    "英语": "en",
    "汉语": "zh-cn",
}

model = initialize()

st.title("语音合成")
lang_name = st.radio("要合成的语言：", langs, 0, horizontal=True)
texts = st.text_area("要合成的文本：", "Nice to meet you.\nGood to see you.")

with st.sidebar:
    speaker = st.selectbox("合成参照的音色", speakers, speaker_default_index)
    speed = st.slider("语速", 0.1, 1.99, 1.0, 0.01)
    temperature = st.slider("温度", 0.1, 2.0, 0.1, 0.01)
    length_penalty = st.slider("长度控制", -2.0, 4.0, 1.0, 0.1)
    repetition_penalty = st.slider("避免重复", 1.0, 20.0, 10.0, 0.1)
    top_k = st.slider("Top K", 0, 100, 10, 1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.3, 0.01)

if st.button("合成", disabled=not texts):
    texts = re.sub(r"\.? ?\n", ". ", texts)
    texts = split_sentences(texts)
    print(f"[XTTS] 合成文本：{texts}")

    time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    short_id = str(uuid.uuid4())[:6]
    lang = langs[lang_name]
    filename = f"outputs/{time}_{short_id}_{lang}_{speaker.split('.')[0]}.wav"
    with st.spinner("正在合成，请稍候……"):
        gpt_cond_latent, speaker_embedding = \
            initialize_speaker(model, f"targets/{speaker}")
        inference(
            texts, lang, filename,
            model, gpt_cond_latent, speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            speed=speed,
        )
    st.audio(filename, format="audio/wav")
