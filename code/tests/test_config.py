from pathlib import Path
import contextlib
import io
import tempfile

import train

from code.configs import parse_args


def write_config(contents):
    root = Path(tempfile.mkdtemp(prefix="semisam_config_", dir="/private/tmp"))
    path = root / "experiment.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def main():
    config_path = write_config(
        """
sam_checkpoint: from_config.pth
batch_size: 8
labeled_bs: 3
patch_size: [256, 320]
intensity_norm: clip_zscore
sam_freeze_image_encoder: true
""".strip()
    )
    args = parse_args(
        [
            "--config",
            str(config_path),
            "--batch_size",
            "6",
            "--no_sam_freeze_image_encoder",
        ]
    )

    assert args.config == config_path
    assert args.sam_checkpoint == "from_config.pth"
    assert args.batch_size == 6
    assert args.labeled_bs == 3
    assert args.patch_size == [256, 320]
    assert args.intensity_norm == "clip_zscore"
    assert args.sam_freeze_image_encoder is False

    invalid_path = write_config("sam_checkpoint: test.pth\npatch_size: [256]\n")
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parse_args(["--config", str(invalid_path)])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("invalid patch_size was accepted")

    captured = {}

    class CapturingTrainer:
        def __init__(self, args):
            captured["args"] = args

        def run(self):
            captured["ran"] = True

    original_builder = train.build_trainer
    train.build_trainer = CapturingTrainer
    try:
        train.main(["--config", str(config_path), "--batch_size", "5"])
    finally:
        train.build_trainer = original_builder

    assert captured["args"].sam_checkpoint == "from_config.pth"
    assert captured["args"].batch_size == 5
    assert captured["ran"] is True

    print("CONFIG_OK")


if __name__ == "__main__":
    main()
