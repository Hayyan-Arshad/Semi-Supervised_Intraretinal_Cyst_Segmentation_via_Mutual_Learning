import functools
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.dataloaders import TwoStreamBatchSampler, build_datasets
from code.networks import build_cnn_segmenter, build_prompt_segmenter
from code.prompts import build_prompt_generator
from code.trainers.common import seed_worker
from code.utils.losses import DiceBCELoss, sigmoid_dice_loss
from code.utils.ramps import sigmoid_rampup
from code.utils.validation import test_single_volume


class SemiSAMTrainer:
    def __init__(
        self,
        args,
        cnn_builder=build_cnn_segmenter,
        prompt_model_builder=build_prompt_segmenter,
        dataset_builder=build_datasets,
        prompt_generator_builder=build_prompt_generator,
    ):
        self.args = args
        self.cnn_builder = cnn_builder
        self.prompt_model_builder = prompt_model_builder
        self.dataset_builder = dataset_builder
        self.prompt_generator_builder = prompt_generator_builder
        self.device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    def consistency_weight(self, iter_num):
        if iter_num < self.args.warmup_iterations:
            return 0.0
        ramp_iter = iter_num - self.args.warmup_iterations
        return self.args.consistency * sigmoid_rampup(ramp_iter, self.args.consistency_rampup)

    def run(self):
        args = self.args
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

        model_cnn = self.cnn_builder(args, self.device)
        model_prompt = self.prompt_model_builder(args, self.device)
        prompt_generator = self.prompt_generator_builder(args)
        db_train, db_val = self.dataset_builder(args)

        labeled_idxs = list(range(0, args.labeled_num))
        unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
        )
        trainloader = DataLoader(
            db_train,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=functools.partial(seed_worker, base_seed=args.seed),
        )
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=max(0, min(1, args.num_workers)))

        optimizer = optim.Adam(
            [
                {"params": model_cnn.parameters(), "lr": args.base_lr},
                {"params": [p for p in model_prompt.parameters() if p.requires_grad], "lr": args.prompt_lr},
            ]
        )
        supervised_criterion = DiceBCELoss()
        writer = SummaryWriter(str(snapshot_path / "log"))
        iter_num = 0
        best_performance = 0.0
        max_epoch = args.max_iterations // len(trainloader) + 1

        for _ in tqdm(range(max_epoch), ncols=80):
            for sampled_batch in trainloader:
                volume_batch = sampled_batch["image"].to(self.device)
                label_batch = (sampled_batch["label"] > 0).to(self.device)

                cnn_logits = model_cnn(volume_batch)
                boxes, points, point_labels = prompt_generator(torch.sigmoid(cnn_logits.detach()))
                prompt_logits = model_prompt(volume_batch, boxes, points, point_labels)

                cnn_sup = supervised_criterion(cnn_logits[: args.labeled_bs], label_batch[: args.labeled_bs])
                prompt_sup = supervised_criterion(prompt_logits[: args.labeled_bs], label_batch[: args.labeled_bs])
                supervised_loss = 0.5 * (cnn_sup + prompt_sup)
                mutual_loss = 0.5 * (
                    sigmoid_dice_loss(cnn_logits, prompt_logits.detach())
                    + sigmoid_dice_loss(prompt_logits, cnn_logits.detach())
                )
                consistency_weight = self.consistency_weight(iter_num)
                loss = supervised_loss + consistency_weight * mutual_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
                prompt_lr = args.prompt_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = prompt_lr
                iter_num += 1

                writer.add_scalar("info/lr", lr, iter_num)
                writer.add_scalar("info/prompt_lr", prompt_lr, iter_num)
                writer.add_scalar("info/total_loss", loss, iter_num)
                writer.add_scalar("info/supervised_loss", supervised_loss, iter_num)
                writer.add_scalar("info/mutual_loss", mutual_loss, iter_num)
                writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)

                if iter_num % args.save_every == 0:
                    torch.save(model_cnn.state_dict(), snapshot_path / f"cnn_iter_{iter_num}.pth")
                    torch.save(model_prompt.state_dict(), snapshot_path / f"prompt_iter_{iter_num}.pth")

                if iter_num > 0 and iter_num % args.val_every == 0:
                    metric_list = 0.0
                    for val_batch in valloader:
                        metric_i = test_single_volume(
                            val_batch["image"],
                            val_batch["label"],
                            model_cnn,
                            patch_size=args.patch_size,
                            device=self.device,
                        )
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                    performance = np.mean(metric_list, axis=0)[0]
                    writer.add_scalar("info/val_mean_dice", performance, iter_num)
                    if performance > best_performance:
                        best_performance = performance
                        best_path = snapshot_path / f"cnn_best_dice_{round(best_performance, 4)}.pth"
                        torch.save(model_cnn.state_dict(), best_path)
                        shutil.copy(str(best_path), str(snapshot_path / "cnn_best.pth"))
                    logging.info("iter %d: val mean dice %.4f, best %.4f", iter_num, performance, best_performance)
                    model_cnn.train()
                    model_prompt.train()

                if iter_num >= args.max_iterations:
                    writer.close()
                    return

