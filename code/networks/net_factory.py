from .efficient_unet import build_efficient_unet
from .sam_adapter import SAMPromptSegmenter


def build_cnn_segmenter(args, device):
    if args.cnn_model != "efficient_unet":
        raise ValueError(f"Unsupported cnn_model '{args.cnn_model}'. Available: efficient_unet")
    return build_efficient_unet(args, device)


def build_prompt_segmenter(args, device):
    if args.prompt_model != "sam":
        raise ValueError(f"Unsupported prompt_model '{args.prompt_model}'. Available: sam")
    return SAMPromptSegmenter(
        checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        freeze_image_encoder=args.sam_freeze_image_encoder,
    ).to(device)

