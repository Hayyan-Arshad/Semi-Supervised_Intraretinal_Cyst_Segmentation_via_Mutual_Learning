from .semisam_trainer import SemiSAMTrainer


def build_trainer(args):
    if args.trainer != "semisam":
        raise ValueError(f"Unsupported trainer '{args.trainer}'. Available: semisam")
    return SemiSAMTrainer(args)

