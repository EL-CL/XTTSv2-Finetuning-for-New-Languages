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

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Model loaded successfully!")

# Inference
text_lang_pairs = {
    # 文件标题: (文本, 语言名)
    "句子1": ("How do you do?", "sample"),
    "句子2": ("Nice to meet you.", "sample"),
    "句子3": ("Good to see you.", "sample"),
}
speaker_audio_files = {
    # 合成所参照的音色
    "M": "male.wav",
    "F": "female.wav",
}

time = datetime.datetime.now().strftime("%m-%d_%H-%M")
for speaker_name, speaker_audio_file in speaker_audio_files.items():
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )
    for filename, (text, lang) in text_lang_pairs.items():
        filename = f"outputs/{time} {filename} {speaker_name}.wav"
        print(filename)

        texts = sent_tokenize(text)

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
