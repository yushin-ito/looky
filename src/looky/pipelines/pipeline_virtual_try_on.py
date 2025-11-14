import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    SD3LoraLoaderMixin,
)
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from .pipeline_output import VirtualTryOnPipelineOutput
from ..models.transformer_garment import GarmentTransformer2DModel
from ..models.transformer_vton import VtonTransformer2DModel


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # type: ignore

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", 
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class VirtualTryOnPipeline(
    DiffusionPipeline,
    SD3LoraLoaderMixin,
    FromSingleFileMixin,
):
    r"""
    Args:
        transformer_vton ([`VtonTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded person image latents.
        transformer_garment ([`GarmentTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded garment image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        image_encoder (`CLIPVisionModelWithProjection`):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPImageModelWithProjection),
            specifically the
            [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
            variant.
        image_encoder_2 (`CLIPVisionModelWithProjection`):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPImageModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
        feature_extractor (`CLIPImageProcessor`):
            Image processor of class.
    """

    model_cpu_offload_seq = (
        "image_encoder->image_encoder_2->transformer_garment->transformer_vton->vae",
    )
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "image_embeds"]

    def __init__(
        self,
        transformer_vton: VtonTransformer2DModel,
        transformer_garment: GarmentTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        image_encoder_2: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer_vton=transformer_vton,
            transformer_garment=transformer_garment,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_encoder_2=image_encoder_2,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        latent_channels = (
            self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        )

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=latent_channels,
            do_normalize=False,
            # do_binarize=True,
            do_convert_grayscale=True,
        )
        self.default_sample_size = (
            self.transformer_vton.config.sample_size
            if hasattr(self, "transformer") and self.transformer_vton is not None
            else 128
        )
        self.patch_size = (
            self.transformer_vton.config.patch_size
            if hasattr(self, "transformer") and self.transformer_vton is not None
            else 2
        )

    def _get_clip_image_embeds(
        self,
        image: Optional[PipelineImageInput] = None,
        device: Optional[torch.device] = None,
        clip_model_index: int = 0,
    ) -> torch.Tensor:
        device = device or self._execution_device

        image_encoders = [self.image_encoder, self.image_encoder_2]

        feature_extractor = self.feature_extractor
        image_encoder = image_encoders[clip_model_index]

        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        image_embeds = image_encoder(pixel_values.to(device=device)).image_embeds

        image_embeds = image_embeds.to(dtype=self.image_encoder.dtype, device=device)

        return image_embeds

    def encode_image(
        self,
        image: Optional[PipelineImageInput] = None,
        device: Optional[torch.device] = None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device

        if image_embeds is None:
            image_embed = self._get_clip_image_embeds(
                image=image, device=device, clip_model_index=0
            )
            image_2_embed = self._get_clip_image_embeds(
                image=image, device=device, clip_model_index=1
            )
            image_embeds = torch.cat([image_embed, image_2_embed], dim=1)

        return image_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.check_inputs
    def check_inputs(
        self,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
    ):
        if (
            height % (self.vae_scale_factor * self.patch_size) != 0
            or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * self.patch_size} but are {height} and {width}."
                f"You can use height {height - height % (self.vae_scale_factor * self.patch_size)} and width {width - width % (self.vae_scale_factor * self.patch_size)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if return_image_latents:
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 16:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(
                batch_size // image_latents.shape[0], 1, 1, 1
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        outputs = (latents,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]), generator=generator[i]
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator
            )

        image_latents = (
            image_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor

        return image_latents

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask,
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 16:
            masked_image_latents = masked_image
        else:
            masked_image_latents = retrieve_latents(
                self.vae.encode(masked_image), generator=generator
            )

        masked_image_latents = (
            masked_image_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_pose_image(
        self,
        image,
        width,
        height,
        batch_size,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        image = self.image_processor.preprocess(image, height=height, width=width).to(
            dtype=torch.float32
        )

        image = image.repeat_interleave(batch_size, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: PipelineImageInput = None,
        garment_image: PipelineImageInput = None,
        pose_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,  # 1.0
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        garment_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        mu: Optional[float] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            mask_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
                H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
                1)`, or `(H, W)`.
            mask_image_latent (`torch.Tensor`, `List[torch.Tensor]`):
                `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask
                latents tensor will be generated by `mask_image`.
            height (`int`, *optional*, defaults to self.transformer_vton.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer_vton.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            padding_mask_crop (`int`, *optional*, defaults to `None`):
                The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to
                image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region
                with the same aspect ration of the image and contains all masked area, and then expand that area based
                on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before
                resizing to the original image size for inpainting. This is useful when the masked area is small while
                the image is large and contain information irrelevant for inpainting, such as background.
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of torch generator(s) to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents to be used as inputs for image generation. If not provided, a latents
                tensor will be randomly generated.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `VirtualTryOnPipelineOutput` instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising step during the inference.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            mu (`float`, *optional*):
                `mu` value used for `dynamic_shifting`.

        Examples:

        Returns:
            `VirtualTryOnPipelineOutput` or `tuple`:
            `VirtualTryOnPipelineOutput` if `return_dict` is True, otherwise a tuple. When returning a tuple,
            the first element is a list with the generated images.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height,
            width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1

        device = self._execution_device
        dtype = self.transformer_vton.dtype

        garment_image_embeds = self.encode_image(
            image=garment_image,
            image_embeds=garment_image_embeds,
            device=device,
        )

        if self.do_classifier_free_guidance:
            garment_image_embeds = torch.cat(
                [torch.zeros_like(garment_image_embeds), garment_image_embeds], dim=0
            )

        # 3. Prepare pose image
        pose_image = self.prepare_pose_image(
            image=pose_image,
            width=width,
            height=height,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        height, width = pose_image.shape[-2:]

        # 4. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, h, w = latents.shape
            image_seq_len = (h // self.transformer_vton.config.patch_size) * (
                w // self.transformer_vton.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Preprocess mask and image
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(
                mask_image, width, height, pad=padding_mask_crop
            )
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        original_image = image
        init_image = self.image_processor.preprocess(
            image,
            height=height,
            width=width,
            crops_coords=crops_coords,
            resize_mode=resize_mode,
        )
        init_image = init_image.to(dtype=torch.float32)

        init_garment_image = self.image_processor.preprocess(garment_image)
        init_garment_image = init_garment_image.to(dtype=torch.float32)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels

        _, garment_image_latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            garment_image_embeds.dtype,
            device,
            generator,
            image=init_garment_image,
            return_image_latents=True,
        )

        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer_vton.config.in_channels

        (latents,) = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            garment_image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image,
            height=height,
            width=width,
            crops_coords=crops_coords,
            resize_mode=resize_mode,
        )

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size,
            height,
            width,
            garment_image_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        # match the inpainting pipeline and will be updated with input + mask inpainting model later
        if num_channels_transformer == 33:
            # default case for stable-diffusion-v1-5/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if (
                num_channels_latents + num_channels_mask + num_channels_masked_image
                != self.transformer_vton.config.in_channels
            ):
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.transformer`: {self.transformer_vton.config} expects"
                    f" {self.transformer_vton.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                    " `pipeline.transformer` or your `mask_image` or `image` input."
                )
        elif num_channels_transformer != 16:
            raise ValueError(
                f"The transformer {self.transformer_vton.__class__} should have 16 input channels or 33 input channels, not {self.transformer_vton.config.in_channels}."
            )

        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if i == 0:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat(
                            [
                                torch.zeros_like(garment_image_latents),
                                garment_image_latents,
                            ]
                        )
                        if self.do_classifier_free_guidance
                        else garment_image_latents
                    )
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0]) * 0

                    cache = self.transformer_garment(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=None,
                        pooled_projections=garment_image_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[1]

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                latent_model_input = torch.cat(
                    [latent_model_input, masked_image_latents, mask], dim=1
                )

                noise_pred = self.transformer_vton(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=None,
                    pooled_projections=garment_image_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    cache=cache,
                    pose_cond=pose_image,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

            if padding_mask_crop is not None:
                image = [
                    self.image_processor.apply_overlay(
                        mask_image, original_image, i, crops_coords
                    )
                    for i in image
                ]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return VirtualTryOnPipelineOutput(images=image)
