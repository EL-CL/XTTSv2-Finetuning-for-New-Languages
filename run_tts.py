import datetime
import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model paths
xtts_checkpoint = "models/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/best_model_99875.pth"
xtts_config = "models/config.json"
xtts_vocab = "models/vocab.json"

speaker_audio_files = {
    # 合成所参照的音色
    "M": "male.wav",
    "F": "female.wav",
}

text_inputs = {
    # 文件标题: (语言名, 文本)
    # 文本可以是多个句子连在一起，也可以是列表
    # 多个句子连在一起时，后面会由 sent_tokenize 根据标点（必须是西文标点）切分成列表
    "文本1": ("sample", "How do you do?"),
    "文本2": ("sample", "Nice to meet you. Good to see you."),
    "文本3": ("sample", [
        "Good to see you.",
        "Nice to meet you.",
    ]),
}

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Model loaded successfully!")

# Inference
time = datetime.datetime.now().strftime("%m-%d_%H-%M")
for speaker_name, speaker_audio_file in speaker_audio_files.items():
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )
    for filename, (lang, text) in text_inputs.items():
        filename = f"outputs/{time} {filename} {speaker_name}.wav"
        print(filename)

        texts = sent_tokenize(text) if isinstance(text, str) else text

        wav_chunks = []
        for text in tqdm(texts):
            wav_chunk = XTTS_MODEL.inference(
                text=text,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.1,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=10,
                top_p=0.3,
            )
            wav_chunks.append(torch.tensor(wav_chunk["wav"]))

        out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

        torchaudio.save(filename, out_wav, 24000, encoding="PCM_S", bits_per_sample=16)
