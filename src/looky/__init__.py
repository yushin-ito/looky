__version__ = "1.0.0"

from transformers import Mask2FormerForUniversalSegmentation
from transformers.pipelines import PIPELINE_REGISTRY

from .pipelines.pipeline_agnostic_mask_generation import AgnosticMaskGenerationPipeline


PIPELINE_REGISTRY.register_pipeline(
    "agnostic-mask-generation",
    pipeline_class=AgnosticMaskGenerationPipeline,
    pt_model=Mask2FormerForUniversalSegmentation,
)
