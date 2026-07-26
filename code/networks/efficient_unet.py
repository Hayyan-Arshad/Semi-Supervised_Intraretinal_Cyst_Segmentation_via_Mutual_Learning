import segmentation_models_pytorch as smp


def build_efficient_unet(args, device):
    encoder_weights = args.encoder_weights
    if isinstance(encoder_weights, str) and encoder_weights.lower() in {"none", "null"}:
        encoder_weights = None
    model = smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=args.in_channels,
        classes=args.out_channels,
    )
    return model.to(device)
