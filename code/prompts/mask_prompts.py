import torch


def masks_to_boxes_and_points(mask_probs, threshold=0.5, margin=4):
    boxes = []
    points = []
    labels = []
    _, _, height, width = mask_probs.shape
    binary_masks = mask_probs[:, 0] > threshold

    for mask in binary_masks:
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            boxes.append([0, 0, width - 1, height - 1])
            points.append([width // 2, height // 2])
            labels.append(0)
            continue

        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values
        x_min = torch.clamp(x_min - margin, 0, width - 1)
        y_min = torch.clamp(y_min - margin, 0, height - 1)
        x_max = torch.clamp(x_max + margin, 0, width - 1)
        y_max = torch.clamp(y_max + margin, 0, height - 1)
        positive_idx = coords[torch.randint(coords.shape[0], (1,), device=coords.device)[0]]

        boxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
        points.append([positive_idx[1].item(), positive_idx[0].item()])
        labels.append(1)

    return (
        torch.as_tensor(boxes, dtype=torch.float32, device=mask_probs.device),
        torch.as_tensor(points, dtype=torch.float32, device=mask_probs.device),
        torch.as_tensor(labels, dtype=torch.int64, device=mask_probs.device),
    )

