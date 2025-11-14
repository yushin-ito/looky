from typing import Any, Tuple, Union, overload

import numpy as np
from skimage.morphology import remove_small_objects
from transformers import Pipeline
from transformers.utils import (
    add_end_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from transformers.pipelines.base import build_pipeline_init_args


if is_vision_available():
    from PIL import Image

    from transformers.image_utils import load_image

if is_torch_available():
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES,
    )


logger = logging.get_logger(__name__)


# Copied from transformers.models.detr.image_processing_detr.masks_to_boxes
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    Args:
        masks: masks in format `[number_masks, height, width]` where N is the number of masks

    Returns:
        boxes: bounding boxes in format `[number_masks, 4]` in xyxy format
    """
    if masks.size == 0:
        return np.zeros((0, 4))

    h, w = masks.shape[-2:]
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    # see https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    return np.stack([x_min, y_min, x_max, y_max], 1)


def get_upper_body_mask(
    segmentation: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    target_size: Tuple[int, int],
):
    height, width = target_size

    keypoints = keypoints.copy()
    keypoints[..., 0] *= width
    keypoints[..., 1] *= height

    face_keypoints = keypoints[24:92]
    face_scores = scores[24:92]

    right_hand_keypoints = keypoints[92:113]
    right_hand_scores = scores[92:113]

    left_hand_keypoints = keypoints[113:]
    left_hand_scores = scores[113:]

    body_keypoints = keypoints[:18]
    body_scores = scores[:18]

    mask = np.isin(segmentation, [5, 6, 7])
    mask = remove_small_objects(mask, min_size=300)
    boxes = masks_to_boxes(mask[None, ...])
    box1 = boxes[0]

    x_min = float("inf")
    y_min = float("inf")
    x_max = float("-inf")
    y_max = float("-inf")

    for i in [2, 3, 4]:
        if body_scores[i] > 0.3:
            x = body_keypoints[i, 0]
            x_min = min(x_min, x)

    for i in [2, 5]:
        if body_scores[i] > 0.3:
            y = body_keypoints[i, 1]
            y_min = min(y_min, y)

    for i in [5, 6, 7]:
        if body_scores[i] > 0.3:
            x = body_keypoints[i, 0]
            x_max = max(x_max, x)

    for i in [8, 11]:
        if body_scores[i] > 0.3:
            y = body_keypoints[i, 1]
            y_max = max(y_max, y)

    for i in [5, 9, 13]:
        if left_hand_scores[i] > 0.3:
            x = left_hand_keypoints[i, 0]
            y = left_hand_keypoints[i, 1]

            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        if right_hand_scores[i] > 0.3:
            x = right_hand_keypoints[i, 0]
            y = right_hand_keypoints[i, 1]

            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

    for i in [5, 11]:
        if face_scores[i] > 0.3:
            y = face_keypoints[i, 1]
            y_min = min(y_min, y)

    box2 = [x_min, y_min, x_max, y_max]

    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])

    x_min = max(0, x_min - 20)
    y_min = max(0, y_min - 10)
    x_max = min(width, x_max + 25)
    if box2[3] > box1[3]:
        y_max = min(height, y_max)
    else:
        y_max = min(height, y_max + 25)

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    agnostic_mask = np.zeros((height, width), dtype=np.uint8)
    agnostic_mask[y_min:y_max, x_min:x_max] = 255

    mask = np.isin(segmentation, [1, 2, 4, 13])
    agnostic_mask[mask] = 0

    agnostic_mask = Image.fromarray(agnostic_mask)

    return agnostic_mask


def get_lower_body_mask(
    segmentation,
    keypoints,
    scores,
    target_size,
):
    height, width = target_size

    keypoints = keypoints.copy()
    keypoints[..., 0] *= width
    keypoints[..., 1] *= height

    body_keypoints = keypoints[:18]
    body_scores = scores[:18]

    right_foot_keypoints = keypoints[18:21]
    right_foot_scores = scores[18:21]

    left_foot_keypoints = keypoints[21:24]
    left_foot_scores = scores[21:24]

    mask = np.isin(segmentation, [6, 9, 12])
    mask = remove_small_objects(mask, min_size=300)
    boxes = masks_to_boxes(mask[None, ...])
    box1 = boxes[0]

    x_min = float("inf")
    y_min = float("inf")
    x_max = float("-inf")
    y_max = float("-inf")

    for i in [8, 9, 10, 11, 12, 13]:
        if body_scores[i] > 0.3:
            x = body_keypoints[i, 0]
            x_min = min(x_min, x)
            x_max = max(x_max, x)

    for i in [8, 11]:
        if body_scores[i] > 0.3:
            y = body_keypoints[i, 1]
            y_min = min(y_min, y)

    for i in [10, 13]:
        if body_scores[i] > 0.3:
            y = body_keypoints[i, 1]
            y_max = max(y_max, y)

    for i in [0, 1, 2]:
        if right_foot_scores[i] > 0.3:
            x = right_foot_keypoints[i, 0]

            x_min = min(x_min, x)
            x_max = max(x_max, x)

        if left_foot_scores[i] > 0.3:
            x = left_foot_keypoints[i, 0]

            x_min = min(x_min, x)
            x_max = max(x_max, x)

    for i in [2]:
        if right_foot_scores[i] > 0.3:
            y = right_foot_keypoints[i, 1]
            y_max = max(y_max, y)

        if left_foot_scores[i] > 0.3:
            y = left_foot_keypoints[i, 1]
            y_max = max(y_max, y)

    box2 = [x_min, y_min, x_max, y_max]

    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])

    x_min = max(0, x_min - 30)
    if box2[1] > box1[1]:
        y_min = max(0, y_min - 50)
    else:
        y_min = max(0, y_min - 5)
    x_max = min(width, x_max + 30)
    y_max = min(height, y_max + 10)

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    agnostic_mask = np.zeros((height, width), dtype=np.uint8)
    agnostic_mask[y_min:y_max, x_min:x_max] = 255

    mask = np.isin(segmentation, [18, 19])
    agnostic_mask[mask] = 0

    agnostic_mask = Image.fromarray(agnostic_mask)

    return agnostic_mask


def get_full_body_mask(
    segmentation: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    target_size: Tuple[int, int],
):
    height, width = target_size

    keypoints = keypoints.copy()
    keypoints[..., 0] *= width
    keypoints[..., 1] *= height

    face_keypoints = keypoints[24:92]
    face_scores = scores[24:92]

    right_hand_keypoints = keypoints[92:113]
    right_hand_scores = scores[92:113]

    left_hand_keypoints = keypoints[113:]
    left_hand_scores = scores[113:]

    body_keypoints = keypoints[:18]
    body_scores = scores[:18]

    right_foot_keypoints = keypoints[18:21]
    right_foot_scores = scores[18:21]

    left_foot_keypoints = keypoints[21:24]
    left_foot_scores = scores[21:24]

    mask = np.isin(segmentation, [5, 6, 7, 9, 12])
    mask = remove_small_objects(mask, min_size=300)
    boxes = masks_to_boxes(mask[None, ...])
    box1 = boxes[0]

    x_min = float("inf")
    y_min = float("inf")
    x_max = float("-inf")
    y_max = float("-inf")

    for i in [2, 3, 4, 8, 9, 10]:
        if body_scores[i] > 0.3:
            x = body_keypoints[i, 0]
            x_min = min(x_min, x)

    for i in [5, 6, 7, 11, 12, 13]:
        if body_scores[i] > 0.3:
            x = body_keypoints[i, 0]
            x_max = max(x_max, x)

    for i in [2, 5]:
        if body_scores[i] > 0.3:
            y = body_keypoints[i, 1]
            y_min = min(y_min, y)

    for i in [10, 13]:
        if body_scores[i] > 0.3:
            y = body_keypoints[i, 1]
            y_max = max(y_max, y)

    for i in [5, 9, 13, 17]:
        if right_hand_scores[i] > 0.3:
            x = right_hand_keypoints[i, 0]
            y = right_hand_keypoints[i, 1]

            x_min = min(x_min, x)
            y_min = min(y_min, y)

        if left_hand_scores[i] > 0.3:
            x = left_hand_keypoints[i, 0]
            y = left_hand_keypoints[i, 1]

            x_max = max(x_max, x)
            y_min = min(y_min, y)

    for i in [0, 1, 2]:
        if right_foot_scores[i] > 0.3:
            x = right_foot_keypoints[i, 0]
            y = right_foot_keypoints[i, 1]

            x_max = max(x_max, x)
            y_max = max(y_max, y)

        if left_foot_scores[i] > 0.3:
            x = left_foot_keypoints[i, 0]
            y = left_foot_keypoints[i, 1]

            x_min = min(x_min, x)
            y_max = max(y_max, y)

    for i in [5, 11]:
        if face_scores[i] > 0.3:
            y = face_keypoints[i, 1]
            y_min = min(y_min, y)

    box2 = [x_min, y_min, x_max, y_max]

    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])

    x_min = max(0, x_min - 50)
    if box2[1] > box1[1]:
        y_min = max(0, y_min - 20)
    else:
        y_min = max(0, y_min)
    x_max = min(width, x_max + 50)
    y_max = min(height, y_max + 10)

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    agnostic_mask = np.zeros((height, width), dtype=np.uint8)
    agnostic_mask[y_min:y_max, x_min:x_max] = 255

    mask = np.isin(segmentation, [1, 2, 4, 13, 18, 19])
    agnostic_mask[mask] = 0

    agnostic_mask = Image.fromarray(agnostic_mask)

    return agnostic_mask


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class AgnosticMaskGenerationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
    >>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    >>> len(segments)
    2

    >>> segments[0]["label"]
    'bird'

    >>> segments[1]["label"]
    'bird'

    >>> type(segments[0]["mask"])  # This is a black and white mask showing where is the bird on the original image.
    <class 'PIL.Image.Image'>

    >>> segments[0]["mask"].size
    (768, 512)
    ```


    This image segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).
    """

    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = None  # Oneformer uses it but no-one else does

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES)

    def _sanitize_parameters(
        self, timeout=None, category=None, keypoints=None, scores=None, **kwargs
    ):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout

        postprocess_params = {}
        if category is not None:
            postprocess_params["category"] = category
        if keypoints is not None:
            postprocess_params["keypoints"] = keypoints
        if scores is not None:
            postprocess_params["scores"] = scores

        return preprocess_params, {}, postprocess_params

    @overload
    def __call__(
        self, inputs: Union[str, "Image.Image"], **kwargs: Any
    ) -> list[dict[str, Any]]: ...

    @overload
    def __call__(
        self, inputs: list[str] | list["Image.Image"], **kwargs: Any
    ) -> list[list[dict[str, Any]]]: ...

    def __call__(
        self,
        inputs: Union[str, "Image.Image", list[str], list["Image.Image"]],
        **kwargs: Any,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

        Args:
            inputs (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            subtask (`str`, *optional*):
                Segmentation task to be performed, choose [`semantic`, `instance` and `panoptic`] depending on model
                capabilities. If not set, the pipeline will attempt tp resolve in the following order:
                  `panoptic`, `instance`, `semantic`.
            threshold (`float`, *optional*, defaults to 0.9):
                Probability threshold to filter out predicted masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.5):
                Mask overlap threshold to eliminate small, disconnected segments.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            If the input is a single image, will return a list of dictionaries, if the input is a list of several images,
            will return a list of list of dictionaries corresponding to each image.

            The dictionaries contain the mask, label and score (where applicable) of each detected object and contains
            the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **mask** (`PIL.Image`) -- A binary mask of the detected object as a Pil Image of shape (width, height) of
              the original image. Returns a mask filled with zeros if no object is found.
            - **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of the
              "object" described by the label and the mask.
        """
        # After deprecation of this is completed, remove the default `None` value for `images`
        if "images" in kwargs:
            inputs = kwargs.pop("images")
        if inputs is None:
            raise ValueError(
                "Cannot call the image-classification pipeline without an inputs argument!"
            )
        return super().__call__(inputs, **kwargs)

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)
        target_size = (image.height, image.width)

        inputs = self.image_processor(images=[image], return_tensors="pt")
        inputs = inputs.to(self.dtype)

        inputs["target_size"] = target_size

        return inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        model_outputs = self.model(**model_inputs)
        model_outputs["target_size"] = target_size

        return model_outputs

    def postprocess(
        self,
        model_outputs,
        category,
        keypoints,
        scores,
    ):
        target_size = model_outputs["target_size"]

        outputs = self.image_processor.post_process_semantic_segmentation(
            model_outputs, target_sizes=[target_size]
        )[0]

        segmentation = outputs.numpy()

        if category == "upper-body":
            mask = get_upper_body_mask(segmentation, keypoints, scores, target_size)
            return {"mask": mask}

        if category == "lower-body":
            mask = get_lower_body_mask(segmentation, keypoints, scores, target_size)
            return {"mask": mask}

        if category == "full-body":
            mask = get_full_body_mask(segmentation, keypoints, scores, target_size)
            return {"mask": mask}

        return {}
