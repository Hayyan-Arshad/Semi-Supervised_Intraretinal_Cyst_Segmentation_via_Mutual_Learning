import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        return metric.binary.dc(pred, gt), metric.binary.hd95(pred, gt)
    return 0, 0


def test_single_volume(image, label, net, patch_size=(512, 512), device="cuda"):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    net.eval()

    for ind in range(image.shape[0]):
        image_slice = image[ind, :, :]
        x, y = image_slice.shape
        resized = zoom(image_slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = net(input_tensor)
            pred = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        prediction[ind] = zoom(pred, (x / patch_size[0], y / patch_size[1]), order=0)

    return [calculate_metric_percase(prediction == 1, label == 1)]

