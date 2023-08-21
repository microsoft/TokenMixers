# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# Written by Zhipeng Huang
# --------------------------------------------------------

from .download_utils_base import get_basic_local_path

try:
    from internal.utils.blobby_utils import get_local_path_blobby

    get_local_path = get_local_path_blobby

except ModuleNotFoundError as mnfe:
    get_local_path = get_basic_local_path
