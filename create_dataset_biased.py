import numpy as np
from pathlib import Path

from simple_shapes_dataset.cli.utils import generate_dataset, generate_dataset_biased, generate_image,generate_color
from simple_shapes_dataset.cli import create_dataset_biased
from tqdm import tqdm
from simple_shapes_dataset.version import __version__



create_dataset_biased(
    seed= 0,
    max_train_size = None,
    domain_alignment= [],
    img_size=32,
    output_path=Path('/home/lucas/gwsyn/simple_shapes_dataset_biased_05'),
    num_train_examples=500000,
    num_val_examples=1000,
    num_test_examples=1000,
    min_scale=7,
    max_scale=14,
    min_lightness=46,
    max_lightness=256,
    biased=True,
    class_configs = {0: {'bias_rate': 0.05, 'fixed_color_hls': np.array([0,128,255])}, 
                     1: {'bias_rate': 0.05, 'fixed_color_hls': np.array([60,128,255])},
                     2: {'bias_rate': 0.05, 'fixed_color_hls': np.array([120,128,255])}},


)