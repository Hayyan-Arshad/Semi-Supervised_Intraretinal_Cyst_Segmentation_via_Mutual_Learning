from code.configs import parse_args, set_reproducibility
from code.trainers import build_trainer


def main(argv=None):
    args = parse_args(argv)
    set_reproducibility(args.seed)
    trainer = build_trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()

