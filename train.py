import argparse
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from semisam.dataset import OCTH5Dataset, RandomGenerator, TwoStreamBatchSampler
from semisam.losses import DiceBCELoss, sigmoid_dice_loss
from semisam.prompts import masks_to_boxes_and_points
from semisam.ramps import sigmoid_rampup
from semisam.sam_adapter import SAMPromptSegmenter
from semisam.validation import test_single_volume


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="data/OCT_IRF")
    parser.add_argument("--exp", type=str, default="SemiSAM_EffiB2")
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument("--labeled_bs", type=int, default=6)
    parser.add_argument("--labeled_num", type=int, default=85)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--sam_lr", type=float, default=1e-5)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--consistency", type=float, default=1.0)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--warmup_iterations", type=int, default=1000)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_b")
    parser.add_argument("--sam_freeze_image_encoder", action="store_true")
    parser.add_argument("--prompt_threshold", type=float, default=0.5)
    parser.add_argument("--prompt_margin", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=3000)
    parser.add_argument("--snapshot_path", type=str, default="model")
    return parser.parse_args()


def create_efficientnet_b2_unet(device):
    model = smp.Unet(encoder_name="efficientnet-b2", encoder_weights="imagenet", in_channels=1, classes=1)
    return model.to(device)


def get_current_consistency_weight(iter_num, args):
    if iter_num < args.warmup_iterations:
        return 0.0
    ramp_iter = iter_num - args.warmup_iterations
    return args.consistency * sigmoid_rampup(ramp_iter, args.consistency_rampup)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_path = Path(args.snapshot_path) / args.exp
    snapshot_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=snapshot_path / "log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    model_eff = create_efficientnet_b2_unet(device)
    model_sam = SAMPromptSegmenter(
        checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        freeze_image_encoder=args.sam_freeze_image_encoder,
    ).to(device)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = OCTH5Dataset(args.root_path, split="train", transform=RandomGenerator(args.patch_size))
    db_val = OCTH5Dataset(args.root_path, split="val")
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.Adam(
        [
            {"params": model_eff.parameters(), "lr": args.base_lr},
            {"params": [p for p in model_sam.parameters() if p.requires_grad], "lr": args.sam_lr},
        ]
    )
    supervised_criterion = DiceBCELoss()
    writer = SummaryWriter(str(snapshot_path / "log"))
    iter_num = 0
    best_performance = 0.0
    max_epoch = args.max_iterations // len(trainloader) + 1

    for _ in tqdm(range(max_epoch), ncols=80):
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].to(device)
            label_batch = (sampled_batch["label"] > 0).to(device)

            eff_logits = model_eff(volume_batch)
            boxes, points, point_labels = masks_to_boxes_and_points(
                torch.sigmoid(eff_logits.detach()), args.prompt_threshold, args.prompt_margin
            )
            sam_logits = model_sam(volume_batch, boxes, points, point_labels)

            eff_sup = supervised_criterion(eff_logits[: args.labeled_bs], label_batch[: args.labeled_bs])
            sam_sup = supervised_criterion(sam_logits[: args.labeled_bs], label_batch[: args.labeled_bs])
            supervised_loss = 0.5 * (eff_sup + sam_sup)
            mutual_loss = 0.5 * (
                sigmoid_dice_loss(eff_logits, sam_logits.detach()) + sigmoid_dice_loss(sam_logits, eff_logits.detach())
            )
            consistency_weight = get_current_consistency_weight(iter_num, args)
            loss = supervised_loss + consistency_weight * mutual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            sam_lr = args.sam_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = sam_lr
            iter_num += 1

            writer.add_scalar("info/lr", lr, iter_num)
            writer.add_scalar("info/sam_lr", sam_lr, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/supervised_loss", supervised_loss, iter_num)
            writer.add_scalar("info/mutual_loss", mutual_loss, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)

            if iter_num % args.save_every == 0:
                torch.save(model_eff.state_dict(), snapshot_path / f"eff_iter_{iter_num}.pth")
                torch.save(model_sam.state_dict(), snapshot_path / f"sam_iter_{iter_num}.pth")

            if iter_num > 0 and iter_num % 200 == 0:
                metric_list = 0.0
                for val_batch in valloader:
                    metric_i = test_single_volume(
                        val_batch["image"], val_batch["label"], model_eff, patch_size=args.patch_size, device=device
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    best_path = snapshot_path / f"eff_best_dice_{round(best_performance, 4)}.pth"
                    torch.save(model_eff.state_dict(), best_path)
                    shutil.copy(str(best_path), str(snapshot_path / "eff_best.pth"))
                logging.info("iter %d: val mean dice %.4f, best %.4f", iter_num, performance, best_performance)
                model_eff.train()
                model_sam.train()

            if iter_num >= args.max_iterations:
                writer.close()
                return


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    train(args)

