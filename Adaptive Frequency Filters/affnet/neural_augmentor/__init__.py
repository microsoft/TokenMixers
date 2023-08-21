# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# Written by Zhipeng Huang
# --------------------------------------------------------

import argparse

from .neural_aug import build_neural_augmentor, BaseNeuralAugmentor


def arguments_neural_augmentor(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    return BaseNeuralAugmentor.add_arguments(parser=parser)
