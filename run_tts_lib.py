import torch
import torchaudio
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def split_sentences(texts: str | list[str]):
    if isinstance(texts, str):
        texts = [texts]
    sentences = [sentence for text in texts
                 for sentence in sent_tokenize(text)]
    return sentences


def load_model(xtts_checkpoint, xtts_config, xtts_vocab, device):
    print("Loading model...")
    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config,
                          checkpoint_path=xtts_checkpoint,
                          vocab_path=xtts_vocab,
                          use_deepspeed=False)
    model.to(device)
    print("Model loaded successfully!")
    return model


def initialize_speaker(xtts_model, speaker_audio_file):
    gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=xtts_model.config.gpt_cond_len,
        max_ref_length=xtts_model.config.max_ref_len,
        sound_norm_refs=xtts_model.config.sound_norm_refs,
    )
    return gpt_cond_latent, speaker_embedding


def inference(
    sentences, lang, filename,
    xtts_model, gpt_cond_latent, speaker_embedding,
    **hf_generate_kwargs,
):
    print(filename, "is generating...")
    wav_chunks = []
    for sentence in sentences:
        wav_chunk = xtts_model.inference(
            text=sentence,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **hf_generate_kwargs,
        )
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))
    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()
    torchaudio.save(filename, out_wav, 24000,
                    encoding="PCM_S", bits_per_sample=16)
    print(filename, "saved successfully!")
