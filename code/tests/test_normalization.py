import numpy as np

from code.dataloaders.oct_h5 import normalize_image


def main():
    image = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    unchanged = normalize_image(image, "none")
    assert unchanged.dtype == np.float32
    assert np.allclose(unchanged, image.astype(np.float32))

    minmax = normalize_image(image, "minmax")
    assert np.isclose(minmax.min(), 0.0)
    assert np.isclose(minmax.max(), 1.0)

    zscore = normalize_image(image, "zscore")
    assert abs(float(zscore.mean())) < 1e-6
    assert abs(float(zscore.std()) - 1.0) < 1e-6

    clipped = normalize_image(np.array([[-1000.0, 0.0], [0.0, 1000.0]], dtype=np.float32), "clip_zscore")
    assert clipped.min() >= -1.0
    assert clipped.max() <= 1.0

    print("NORMALIZATION_OK")


if __name__ == "__main__":
    main()

