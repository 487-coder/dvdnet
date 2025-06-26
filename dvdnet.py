import torch
import torch.nn.functional as F
from motion import align_image, prepare_image
import numpy as np
from utils import numpy_to_tensor
def spatial_denoise(model, image, noise_map):
    size = image.size()
    expand_h = size[-2] % 2
    expand_w = size[-1] % 2
    pad = (0, expand_w, 0, expand_h)
    image = F.pad(input=image, pad=pad, mode='reflect')
    noise_map = F.pad(input=noise_map, pad=pad, mode='reflect')
    denoised_image = torch.clamp(model(image, noise_map), 0., 1.)
    if expand_h != 0:
        denoised_image = denoised_image[:, :, :-1, :]
    if expand_w != 0:
        denoised_image = denoised_image[:, :, :, :-1]
    return denoised_image


def temporal_denoise(model, images, noise_map):
    size = images.size()
    expand_h = size[-2] % 2
    expand_w = size[-1] % 2
    pad = (0, expand_w, 0, expand_h)
    images = F.pad(input=images, pad=pad, mode='reflect')
    noise_map = F.pad(input=noise_map, pad=pad, mode='reflect')
    denoised_images = torch.clamp(model(images, noise_map), 0., 1.)
    if expand_h != 0:
        denoised_images =denoised_images[:, :, :-1, :]
    if expand_w != 0:
        denoised_images =denoised_images[:, :, :, :-1]
    return denoised_images


def denoise_seq_dvdnet(seq, noise_std, temporal_patch, spatial_model, temporal_model):
    num_frames, _, C, H, W = seq.size()
    noise_map = torch.full((1, C, H, W), noise_std)
    inframes_wrpd = np.empty((temporal_patch, H, W, C))
    denoise_window = list()
    denframes = torch.empty((num_frames, C, H, W)).to(seq.device)
    for central_frame in range(num_frames):
        if central_frame == 0:
            for i in range(-(temporal_patch//2), (temporal_patch//2)+1):
                index = min(max(central_frame + i, 0), num_frames-1)
                denoise_window.append(spatial_denoise(spatial_model,seq[index],noise_map))
        else:
            del denoise_window[0]
            index = min(max(central_frame+ (temporal_patch//2),0), num_frames-1)
            denoise_window.append(spatial_denoise(spatial_model,seq[index],noise_map))
            # (B,C,H,W)
        for i in [x for x in range(0, temporal_patch) if x != temporal_patch//2]:
            inframes_wrpd[i] = align_image(denoise_window[i],denoise_window[temporal_patch//2])
        inframes_wrpd[temporal_patch//2] = prepare_image(denoise_window[temporal_patch//2])
        temporal_seq = numpy_to_tensor(inframes_wrpd, seq.device)
        denframes[central_frame] = temporal_denoise(temporal_model, temporal_seq, noise_map)
    del denoise_window
    del inframes_wrpd
    del temporal_seq
    torch.cuda.empty_cache()
    return denframes





