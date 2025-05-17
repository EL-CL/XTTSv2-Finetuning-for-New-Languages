import os
import gc

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

# Logging parameters
RUN_NAME = "GPT_XTTS_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Training Parameters
OUT_PATH = "models"
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # 单卡训练设 True，多卡训练设 False
START_WITH_EVAL = False  # 对于增加新语言的微调，初次训练设 False，继续训练设 True
NUM_EPOCHS = 200
BATCH_SIZE = 3
GRAD_ACUMM_STEPS = 84
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
DATASETS_CONFIG_LIST = []
for language, metadata in [
    # 这里根据实际情况修改
    ("sample", "models/metadata.txt"),
]:
    config_dataset = BaseDatasetConfig(
        formatter="bel_tts_formatter",
        dataset_name=language + "_dataset",
        meta_file_train=metadata,
        language=language,
        path="数据集的路径，即存放音频文件的文件夹",
    )
    DATASETS_CONFIG_LIST.append(config_dataset)

MAX_WAV_LENGTH = 265000  # ~12.02 seconds
MAX_TEXT_LENGTH = 350
MEL_NORM_FILE = "original_model/mel_stats.pth"
DVAE_CHECKPOINT = "original_model/mel_stats.pth"
XTTS_CHECKPOINT = "original_model/model.pth"
TOKENIZER_FILE = "models/vocab.json"
# 继续训练时，将 XTTS_CHECKPOINT 改为上次训练的 best model 路径，如 "models/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/best_model_99875.pth"

# 测试顺利后，可将两个数值改大
PRINT_STEP = 50
SAVE_STEP = 2000

WEIGHT_DECAY = 1e-2
LR = 5e-06


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=11025,  # 0.5 secs
        debug_loading_failures=False,
        max_wav_length=MAX_WAV_LENGTH,
        max_text_length=MAX_TEXT_LENGTH,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    # training parameters config
    config = GPTTrainerConfig()
    config.load_json("models/config.json")
    config.epochs = NUM_EPOCHS
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = "GPT XTTS training"
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    config.batch_group_size = 48  # not in anhnh2002"s code
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = PRINT_STEP
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = SAVE_STEP
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": WEIGHT_DECAY}
    config.lr = LR
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()


if __name__ == "__main__":
    main()
