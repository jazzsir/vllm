"""Utilities for selecting and loading models."""
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import *  # pylint: disable=wildcard-import
from vllm.model_executor.weight_utils import initialize_dummy_weights

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "BloomForCausalLM": BloomForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
}


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


# HBSEO 실제 모델 로딩
def get_model(model_config: ModelConfig) -> nn.Module:

    # HBSEO config.json의 architecture 필드의 값으로 모델 클래스가 결정된다.
    model_class = _get_model_architecture(model_config.hf_config)
    torch.set_default_dtype(model_config.dtype)

    # Create a model instance.
    # The weights will be initialized as empty tensors.
    model = model_class(model_config.hf_config)
    if model_config.use_dummy_weights:
        model = model.cuda()
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        initialize_dummy_weights(model)
    else:
        # Load the weights from the cached or downloaded files.
        model.load_weights(model_config.model, model_config.download_dir,
                           model_config.use_np_weights)
        model = model.cuda()

    # HBSEO Pytorch는 Training과 Evaluation 두가지 모드가 있다.
    #  - model.train() 모드: 모델 학습 모드이다. dropout 레이어가 활성화됨, batch norm이 현재 배치의 통계를 사용
    #  - model.eval() 모드: 모델 추론 모드이다. dropout 레이어가 비활성화됨, batch norm이 학습된 계산된 고정된 통계를 사용
    # 모델 로딩 후 모드를 변경하는 이유는 모델 로딩 시 모델의 파라미터들이 초기화되는데, 이 때 모델은 Training 모드로 설정된다.
    # 모델을 로딩한 후에는 모델을 Evaluation 모드로 변경하여 추론 모드로 사용해야 한다.
    return model.eval()
