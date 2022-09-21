"""

Train a diffusion model on images.
"""

import argparse
from random import sample
from re import I
import librosa

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch
import torchaudio
from improved_diffusion.train_util import TrainLoop

def trim_audio_tensor(audio, sr, seconds, start=0, samples=None):
    """Takes a 1d tensor as input"""
    if samples:
        return(audio[start * sr: (start * sr) + samples])
    return(audio[start * sr:(start * sr) + sr * seconds])

def resample(audio, oldsr, newsr):
    audio = torchaudio.transforms.Resample(oldsr, newsr)(audio[:1, :])
    return(audio, newsr)

import os
files = [f for f in os.listdir('./scripts')]
print(files)
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    # )

    audio, sr = librosa.load("./scripts/test.mp3", duration=2, offset=50, sr=22050) 
    audio = trim_audio_tensor(audio, sr, seconds=None, samples=32768/2)
    audio = torch.from_numpy(audio)
    torchaudio.save("./scripts/sample.mp3", audio.unsqueeze(0), sr)
    audio = audio.unsqueeze(0).unsqueeze(0)
    data = (audio, None)
    torchaudio.save(f"./x_{0}.mp3", audio.squeeze(0), sr)
    for i in range(5):
        noise = torch.randn_like(audio)
        x = diffusion.q_sample(audio, torch.tensor(5), noise=noise)
        torchaudio.save(f"./x_{i+1}.mp3", x.squeeze(0), sr)
    x = diffusion.q_sample(audio, torch.tensor(400))
    torchaudio.save(f"./x_{400}.mp3", x.squeeze(0), sr)
    # data = (torch.ones(128).unsqueeze(0).unsqueeze(0), None)
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()



def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        noise_schedule="linear",
        diffusion_steps=4000,
        num_res_blocks=3,
        num_channels=128,
        image_size=64

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
