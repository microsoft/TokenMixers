# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import argparse


from options.utils import extend_selected_args_with_prefix
from affnet.misc.common import parameter_list
from affnet.anchor_generator import arguments_anchor_gen
from affnet.image_projection_layers import arguments_image_projection_head
from affnet.layers import arguments_nn_layers
from affnet.matcher_det import arguments_box_matcher
from affnet.misc.averaging_utils import arguments_ema, EMA
from affnet.misc.profiler import module_profile
from affnet.models import arguments_model, get_model
from affnet.models.detection.base_detection import DetectionPredTuple
from affnet.neural_augmentor import arguments_neural_augmentor


def modeling_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # model arguments
    parser = arguments_model(parser)
    # neural network layer argumetns
    parser = arguments_nn_layers(parser)
    # EMA arguments
    parser = arguments_ema(parser)
    # anchor generator arguments (for object detection)
    parser = arguments_anchor_gen(parser)
    # box matcher arguments (for object detection)
    parser = arguments_box_matcher(parser)
    # image projection head arguments (usually for multi-modal tasks)
    parser = arguments_image_projection_head(parser)
    # neural aug arguments
    parser = arguments_neural_augmentor(parser)

    # Add teacher as a prefix to enable distillation tasks
    # keep it as the last entry
    parser = extend_selected_args_with_prefix(
        parser, check_string="--model", add_prefix="--teacher."
    )

    return parser
