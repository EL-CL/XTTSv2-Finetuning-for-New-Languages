import os
import re
import time
import datetime
import uuid
import streamlit as st
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from run_tts_lib import *


def initialize():
    os.makedirs("outputs", exist_ok=True)
    st.session_state.filename = ""
    st.session_state.is_generating = False
    st.session_state.last_speaker = ""
    st.session_state.gpt_cond_latent = None
    st.session_state.speaker_embedding = None


def do_load_model():
    # Device configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Model paths
    xtts_checkpoint = "models/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/best_model_99875.pth"
    xtts_config = "models/config.json"
    xtts_vocab = "models/vocab.json"

    st.session_state.model = \
        load_model(xtts_checkpoint, xtts_config, xtts_vocab, device)


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

if "is_generating" not in st.session_state:
    initialize()
if "model" not in st.session_state:
    with st.spinner("正在加载模型，请稍候……"):
        do_load_model()

st.title("语音合成")
lang_name = st.radio("要合成的语言：", langs, 0, horizontal=True)
input_texts = st.text_area(
    "要合成的文本：",
    "Nice to meet you.\nGood to see you.",
    height=200,
)

with st.sidebar:
    speaker = st.selectbox("合成参照的音色", speakers, speaker_default_index)
    speed = st.slider("语速", 0.1, 1.99, 1.0, 0.01)
    temperature = st.slider("温度", 0.1, 2.0, 0.1, 0.01)
    length_penalty = st.slider("长度控制", -2.0, 4.0, 1.0, 0.1)
    repetition_penalty = st.slider("避免重复", 1.0, 20.0, 10.0, 0.1)
    top_k = st.slider("Top K", 0, 100, 10, 1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.3, 0.01)


def generate():
    st.session_state.is_generating = True
    texts = re.sub(r"\.? *\n", ". ", input_texts)
    texts = split_sentences(texts)
    print(f"[XTTS] 合成文本（处理前）：{input_texts}")
    print(f"[XTTS] 合成文本（处理后）：{texts}")

    time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    short_id = str(uuid.uuid4())[:6]
    lang = langs[lang_name]
    filename = f"outputs/{time}_{short_id}_{lang}_{speaker.split('.')[0]}.wav"

    if speaker != st.session_state.last_speaker:
        # 仅在发音人改变时初始化发音人，节省时间
        st.session_state.last_speaker = speaker
        st.session_state.gpt_cond_latent, st.session_state.speaker_embedding = \
            initialize_speaker(st.session_state.model, f"targets/{speaker}")
    inference(
        texts, lang, filename,
        st.session_state.model, st.session_state.gpt_cond_latent, st.session_state.speaker_embedding,
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        speed=speed,
    )
    st.session_state.filename = filename
    st.session_state.is_generating = False


if st.button("合成", disabled=st.session_state.is_generating):
    if st.session_state.is_generating:
        st.error("请先等待当前合成结束！")
    elif not input_texts:
        st.error("请输入要合成的文本！")
    else:
        thread = Thread(target=generate, daemon=True)
        add_script_run_ctx(thread, get_script_run_ctx())
        thread.start()

if st.session_state.filename and os.path.exists(st.session_state.filename):
    st.audio(st.session_state.filename, format="audio/wav")
    if st.session_state.is_generating:
        st.info("正在合成新音频，请稍候……")
    else:
        st.success("合成完成！")
elif st.session_state.is_generating:
    st.info("正在合成，请稍候……")

while st.session_state.is_generating:
    # 随时刷新页面以等待合成完成
    time.sleep(0.1)
    st.rerun()
