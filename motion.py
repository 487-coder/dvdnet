import cv2
import numpy as np
import torch


def prepare_image(image):
    # 输入形状(C,H,W)或(H,W) tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input must be a tensor')
    image = image.detach().cpu()
    if len(image.shape) == 3:
        if image.shape[0] == 3:
            r, g, b = image[0], image[1], image[2]
            image = 0.299 * r + 0.587 * g + 0.114 * b  # 转灰度
        elif image.shape[0] == 1:
                image = image.squeeze(0)
    elif len(image.shape) == 2:
        image = image
    else:
        raise ValueError('Image shape error')
    if image.max() <= 1.0:
        image = torch.clamp(image, 0, 1) * 255.0
    img_np = image.numpy()
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    # 输出 np.ndarray， (H,W)
    return img_np





def Flow_estimation(image1, central_image):
    # 输入： np.ndarray, (H,W)
    if image1.dtype != np.uint8 and central_image.dtype != np.uint8:
        raise ValueError("Both images must be uint8")
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    flow = deepflow.calc(central_image, image1, None)
    # flow (H,W,2)
    return flow

def warp(image1, flow):
    # input: np.ndarray, (H,W)
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped_image = cv2.remap(image1, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    # output np.ndarray,(H,W)
    return warped_image

def align_image(image1, central_image):
    # input: tensor
    image1 = prepare_image(image1)
    central_image = prepare_image(central_image)
    flow = Flow_estimation(image1, central_image)
    result = warp(image1, flow)
    # output: (H, W), np.ndarray
    return result
