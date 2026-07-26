from functools import partial

from .mask_prompts import masks_to_boxes_and_points


def build_prompt_generator(args):
    if args.prompt_generator != "mask_box_point":
        raise ValueError(f"Unsupported prompt_generator '{args.prompt_generator}'. Available: mask_box_point")
    return partial(masks_to_boxes_and_points, threshold=args.prompt_threshold, margin=args.prompt_margin)

