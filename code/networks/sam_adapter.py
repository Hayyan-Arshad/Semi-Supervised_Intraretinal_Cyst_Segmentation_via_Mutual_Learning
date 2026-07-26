import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMPromptSegmenter(nn.Module):
    def __init__(self, checkpoint, model_type="vit_b", freeze_image_encoder=False):
        super().__init__()
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "Install segment-anything and pass --sam_checkpoint with SAM or MedSAM ViT-B weights."
            ) from exc
        if checkpoint is None:
            raise ValueError("Pass --sam_checkpoint with SAM or MedSAM ViT-B weights.")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.train()
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, images, boxes, points, point_labels):
        target_size = self.sam.image_encoder.img_size
        height, width = images.shape[-2:]
        sam_images = images.repeat(1, 3, 1, 1)
        sam_images = F.interpolate(sam_images, size=(target_size, target_size), mode="bilinear", align_corners=False)
        sam_images = self.sam.preprocess(sam_images)

        box_scale = torch.tensor(
            [target_size / width, target_size / height, target_size / width, target_size / height],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        point_scale = torch.tensor([target_size / width, target_size / height], dtype=points.dtype, device=points.device)

        image_embeddings = self.sam.image_encoder(sam_images)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(points[:, None, :] * point_scale, point_labels[:, None]),
            boxes=boxes * box_scale,
            masks=None,
        )
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(low_res_masks, size=(height, width), mode="bilinear", align_corners=False)

