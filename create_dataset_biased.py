import numpy as np
from pathlib import Path

from simple_shapes_dataset.cli.utils import generate_dataset, generate_dataset_biased, generate_image,generate_color
from simple_shapes_dataset.cli import create_dataset_biased
from tqdm import tqdm
from simple_shapes_dataset.version import __version__


class_configs = {
    0: {'fixed_color_hls': np.array([60, 128, 255])},   # vert  = cercle
    1: {'fixed_color_hls': np.array([0, 128, 255])},    # rouge = triangle
    2: {'fixed_color_hls': np.array([120, 128, 255])},  # bleu  = carré
}

color_given_class_0 = {
    #  vert   rouge  bleu
    0: [1/3,  1/3,  1/3],  # diamant
    1: [1/3,  1/3, 1/3],  # oeuf
    2: [1/3,  1/3,  1/3],  # triangle
}


color_given_class_1 = {
    #  vert   rouge  bleu
    0: [0.3,  0.3,  0.4],  # diamant
    1: [0.8,  0.1,  0.1],  # oeuf
    2: [0.2,  0.7,  0.1],  # triangle
}

color_given_class_2 = {
    #  vert   rouge  bleu
    0: [0.3,  0.3,  0.4],  # diamant
    1: [0.1,  0.1,  0.8],  # oeuf
    2: [0.2,  0.7,  0.1],  # triangle
}


color_given_class_3 = {
    #  vert   rouge  bleu
    0: [0.2,  0.2,  0.6],  # diamant
    1: [0.1,  0.1,  0.8],  # oeuf
    2: [0.2,  0.7,  0.1],  # triangle
}


color_given_class_full = {
    #  vert   rouge  bleu
    0: [0,  0,  1],  # diamant
    1: [1,  0, 0],  # oeuf
    2: [0,  1,  0],  # triangle
}


create_dataset_biased(
    seed= 0,
    max_train_size = None,
    domain_alignment= [],
    img_size=32,
    output_path=Path('/home/lucas/gwsyn/simple_shapes_dataset_biased10'),
    num_train_examples=500000,
    num_val_examples=1000,
    num_test_examples=1000,
    min_scale=7,
    max_scale=14,
    min_lightness=46,
    max_lightness=256,
    biased=True,
    fixed_color_rate=0.1,
    color_given_class = color_given_class_full,
    class_configs = class_configs,
)
