import cv2
import numpy as np
import torch


def prepare_image(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input must be a tensor')
    image = image.detach().cpu()
    if len(image.shape) == 4:
        if image.shape[0] == 1:
            # (1,C,H,W)
            image = image.squeeze(0)
        else:
            raise ValueError('expect batch size of 1 image')
    if len(image.shape) == 3:
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
            if image.max() <= 1.0:
                image = torch.clamp(image, 0, 1) * 255.0
            img_np = image.numpy().astype(np.uint8)
            return img_np
        elif image.shape[0] == 1:
            image = image.squeeze(0)
            if image.max() <= 1.0:
                image = torch.clamp(image, 0, 1) * 255.0
            img_np = image.numpy().astype(np.uint8)
            return img_np
        else:
            raise ValueError('channel number error')
            # (C,H,W)
            #r, g, b = image[0], image[1], image[2]
            #image = 0.299 * r + 0.587 * g + 0.114 * b  # 转灰度
    elif len(image.shape) == 2:
        if image.max() <= 1.0:
            image = torch.clamp(image, 0, 1) * 255.0
        img_np = image.numpy().astype(np.uint8)
        return img_np
    else:
        raise ValueError('Image shape error')






def Flow_estimation(image1, central_image):
    # 输入： np.ndarray, (H,W)/ (H,W,C)
    if image1.dtype != np.uint8 or central_image.dtype != np.uint8:
        raise ValueError("Both images must be uint8")
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(central_image.shape) == 3:
        central_image = cv2.cvtColor(central_image, cv2.COLOR_RGB2GRAY)
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    flow = deepflow.calc(central_image, image1, None)
    # flow (H,W,2)
    return flow

def warp(image1, flow):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    # 对每个通道单独 remap
    if image1.ndim == 3 and image1.shape[2] == 3:
        warped_channels = [
            cv2.remap(image1[..., c], map_x, map_y, interpolation=cv2.INTER_LINEAR)
            for c in range(3)
        ]
        warped_image = np.stack(warped_channels, axis=-1)
    else:
        warped_image = cv2.remap(image1, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return warped_image


def align_image(image1, central_image):

    image1_np = prepare_image(image1)           # RGB, uint8
    central_image_np = prepare_image(central_image)

    # 光流估计
    flow = Flow_estimation(image1_np, central_image_np)

    # 对 image1 彩色图进行warp
    result = warp(image1_np, flow)

    return result  # np.ndarray (H, W, 3)