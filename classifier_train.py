
import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from imblearn.metrics import specificity_score, sensitivity_score
from sklearn.metrics import f1_score
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from utils import dist_util, logger
from utils.fp16_util import MixedPrecisionTrainer
from utils.image_datasets import load_data
from utils.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_defaults,
    create_classifier,
    create_diffusion
)
from utils.train_util import parse_resume_step_from_filename, log_loss_dict


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model = create_classifier(
        **args_to_dict(args, classifier_defaults().keys())
    )
    model.to(dist_util.dev())

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    diffusion = create_diffusion()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, diffusion, prefix="train"):
        batch, extra, img_paths = next(data_loader)
        labels = extra["y"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())

        with th.no_grad():
            encoder_posterior = diffusion.encode_first_stage(batch)
            batch = diffusion.get_first_stage_encoding(encoder_posterior).detach()


        t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):

            logits = model(sub_batch, timesteps=sub_t)
            # _, logits = model(sub_batch)

            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            _, pred = logits.topk(1, 1, True, True)
            pred = pred.t()
            preds = pred.squeeze(0).data.cpu().numpy()
            targets = sub_labels.reshape(1, -1).expand_as(pred).squeeze(0).data.cpu().numpy()

            f1 = f1_score(targets, preds, average='weighted')
            specificity = specificity_score(targets, preds, average='weighted')
            sensitivity = sensitivity_score(targets, preds, average='weighted')

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_f1"] = f1
            losses[f"{prefix}_specificity"] = specificity
            losses[f"{prefix}_sensitivity"] = sensitivity


            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)

        forward_backward_log(data, diffusion)

        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, diffusion, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()

        if step % args.save_interval == 0 and step > 0:
            logger.log("saving model...")
            save_model(mp_trainer, args, step + resume_step)

    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, args, step):
    save_path = 'saved_classifier_' + args.data_dir.split('/')[-1]
    # save_path = 'saved_classifier_crc5%'

    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(save_path, f"model_{step:06d}.pt"),
        )
        print(f'model saved to {os.path.join(save_path, f"model_{step:06d}.pt")}')
        # th.save(opt.state_dict(), os.path.join(save_path), f"opt{step:06d}.pt")


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=False,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=100,
    )
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()






























