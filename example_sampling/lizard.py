import torch
import os
import sys
import numpy as np
from PIL import Image
import torch.nn.functional as F
from functorch import einops
from omegaconf import OmegaConf
import cv2
import time
from torchvision.utils import save_image
from pytorch_lightning import seed_everything


# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

seed = 9

seed_everything(seed)

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.image_degradation.utils_image import tensor2img
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config
from ldm.data.histo import VirtualStainingDataset
from ldm.modules.diffusionmodules.openaimodel import EncoderUNetModel


def custom_to_np(x):
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample
def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, c, steps, shape, cond_fn, model_kwargs, eta=1.0, **kwargs):
    print('model, steps, shape, cond_fn, model_kwargs: ',
          steps, shape, cond_fn, model_kwargs)
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, conditioning=c, batch_size=bs, shape=shape, cond_fn=cond_fn, model_kwargs=model_kwargs, eta=eta, verbose=False, **kwargs)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, c, batch_size, cond_fn=None, model_kwargs=None, vanilla=False, custom_steps=None, eta=1.0, **kwargs):
    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model, c=c, steps=custom_steps, shape=shape,
                                                    cond_fn=cond_fn,
                                                    model_kwargs=model_kwargs,
                                                    eta=eta, **kwargs)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def cond_fn(x, t, l=None):

    assert l is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        classifier.to(x_in.device)

        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), l.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0]


if __name__ == '__main__':

    val_dataset = VirtualStainingDataset(dataset='lizard',
                                         mode='test')
    # val_dataset = VirtualStainingDataset(dataset='shift')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True)

    # lizard
    ckpt = 'logs/2024-04-03_05-39-46_histo-ldm-kl-8-512-lizard/checkpoints/last.ckpt'


    config = OmegaConf.load('configs/latent-diffusion/histo-ldm-kl-8-512-lizard.yaml')
    device = 'cuda:4'

    model = instantiate_from_config(config.model).to(device)
    model.eval()

    classifier = EncoderUNetModel(
            image_size=64,
            in_channels=3,
            model_channels=128,
            out_channels=5,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            channel_mult=(1, 1, 2, 2, 4, 4),
            use_fp16=False,
            num_head_channels=64,
            use_scale_shift_norm=True,
            resblock_updown=True,
            pool='attention',
        )
    # change path to the saved classifier model
    # pl_sd = torch.load('/data/karenyyy/HistoDiffAug5/saved_classifier_virtual_staining/model_000100.pt', map_location="cpu")
    # classifier.load_state_dict(pl_sd)
    # classifier.eval()


    all_images = []

    with torch.no_grad():

        for i, batch in enumerate(val_dataloader):
            print(f'Processing {i}th batch')
            #if i > 100:
            #    break
            # z, dict(c_crossattn=[c], c_concat=[control])
            x, z, c = model.get_input(batch, 'jpg')
            # save the original image
            ori_img = custom_to_pil(x[0])
            # encoder_posterior = model.encode_first_stage(x)
            # z = model.get_first_stage_encoding(encoder_posterior).detach()
            control = c['c_concat'][0]
            # # save the mask
            # mask = control[0].detach().cpu().numpy()
            # mask = np.transpose(mask, (1, 2, 0))
            # mask = (mask * 255).astype(np.uint8)
            # mask_img = Image.fromarray(mask)
            #
            # features_adapter = model.model_ad(control)

            control = control.to(z.device)
            mask = custom_to_pil(control[0])
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()

            N = z.shape[0]

            c_cat = c["c_concat"][0][:N]

            recon = model.decode_first_stage(z)
            # save the reconstructed image
            recon_img = custom_to_pil(recon[0])

            # kwargs = {'features_adapter': features_adapter,
            #           'class_label': torch.tensor([1]),
            #           'grad_scale': 500.0}

            with model.ema_scope("Plotting"):
                logs = make_convolutional_sample(model, c=c, batch_size=1,
                                                 cond_fn=cond_fn,
                                                 vanilla=False,
                                                 custom_steps=200,
                                                 eta=1.0)
                # directly save the sample
                gen_img = custom_to_pil(logs["sample"][0])

            # combine mask, original, recon, and fake in a grid of 1 row 4 columns with text
            combined = Image.new('RGB', (4 * 512, 512+30))
            combined.paste(mask, (0, 30))
            combined.paste(ori_img, (512, 30))
            combined.paste(recon_img, (512 * 2, 30))
            combined.paste(gen_img, (512 * 3, 30))
            # add label to the top of each image
            combined = np.array(combined)
            combined = cv2.putText(combined, 'Control', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            combined = cv2.putText(combined, 'Original', (512 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            combined = cv2.putText(combined, 'Reconstructed', (512 * 2 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            combined = cv2.putText(combined, 'Generated', (512 * 3 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # make the edge of each image white, so that it is easier to differentiate
            combined[:, 512, :] = 255
            combined[:, 512 * 2, :] = 255
            combined[:, 512 * 3, :] = 255
            # thicker
            combined[:, 512 - 1, :] = 255
            combined[:, 512 * 2 - 1, :] = 255
            combined[:, 512 * 3 - 1, :] = 255
            # even thicker
            combined[:, 512 + 1, :] = 255
            combined[:, 512 * 2 + 1, :] = 255
            combined[:, 512 * 3 + 1, :] = 255

            combined = Image.fromarray(combined)

            combined.save(f'imgs/lizard_0422/combined_{i}_{seed}.png')
            print(f'imgs/lizard_0422/combined_{i}_{seed}.png saved!')





        #         all_images.extend([custom_to_np(logs["sample"])])
        # all_img = np.concatenate(all_images, axis=0)
        # np.savez('imgs/tmp.npz', all_img)
        # images = np.load('imgs/tmp.npz', allow_pickle=True)
        # images = images["arr_0"]
        # for img_idx in range(images.shape[0]):
        #     img = Image.fromarray(np.asarray(images[img_idx, :, :, :]).astype(np.uint8))
        #     img.save(f'imgs/fake_{img_idx}{img_idx}.png')
        #     print(f'imgs/fake_{img_idx}{img_idx}.png saved!')

