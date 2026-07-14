"""
Copy from DiffusionSat pipeline, modify the code to implement the harmonidiff pipeline.

Self-contained DiffusionSat text-to-image pipeline that can be loaded directly
from the checkpoint folder without importing the project package.
"""

from __future__ import annotations
import inspect
import numpy as np
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch
from torchvision import transforms
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMInverseScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
)

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline as DiffusersStableDiffusionPipeline,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def clamp_bbox_open(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x2 = max(x2, x1+1)
    y2 = max(y2, y1+1)
    return x1, y1, x2, y2

def shrink_bbox_open(x1, y1, x2, y2, ew, W, H):
    return clamp_bbox_open(x1+ew, y1+ew, x2-ew, y2-ew, W, H)

def expand_bbox_open(x1, y1, x2, y2, ew, W, H):
    return clamp_bbox_open(x1-ew, y1-ew, x2+ew, y2+ew, W, H)

def rect_mask_open(H, W, x1, y1, x2, y2, device=None):
    ys = torch.arange(H, device=device).view(H,1)
    xs = torch.arange(W, device=device).view(1,W)
    return (ys>=y1) & (ys<y2) & (xs>=x1) & (xs<x2)

def build_edge_mask(height, width, bbox, inner_edge_width=2, outer_edge_width=2, device='cuda'):
    x1, y1, x2, y2 = bbox
    ox1, oy1, ox2, oy2 = expand_bbox_open(x1, y1, x2, y2, outer_edge_width, width, height)
    ix1, iy1, ix2, iy2 = shrink_bbox_open(x1, y1, x2, y2, inner_edge_width, width, height)
    
    outer_mask = rect_mask_open(height, width, ox1, oy1, ox2, oy2, device=device)
    inner_mask = rect_mask_open(height, width, ix1, iy1, ix2, iy2, device=device)

    edge = (outer_mask ^ inner_mask).float() 
    return edge.view(1, 1, height, width)

def create_mask(W, H, box):
    x0, y0, x1, y1 = box 
    mask_np = np.zeros((H, W), dtype=np.float32)
    mask_np[y0:y1, x0:x1] = 1.0 
    m = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
        (W, H), Image.NEAREST
    )
    m = torch.from_numpy((np.array(m) > 127).astype(np.float32)).unsqueeze(0)
    return m

def metadata_normalize(metadata, base_lon=180, base_lat=90, base_year=1980, max_gsd=1., scale=1000):
    lon, lat, gsd, cloud_cover, year, month, day = metadata
    lon = lon / (180 + base_lon) * scale
    lat = lat / (90 + base_lat) * scale
    gsd = gsd / max_gsd * scale
    cloud_cover = cloud_cover * scale
    year = year / (2100 - base_year) * scale
    month = month / 12 * scale
    day = day / 31 * scale
    return torch.tensor([lon, lat, gsd, cloud_cover, year, month, day])




class HarmoniDiffPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using the DiffusionSat UNet with optional metadata.
    """

    _optional_components = ["safety_checker", "feature_extractor", "inverse_scheduler"]
 
    @classmethod
    def _get_signature_types(cls) -> Dict[str, tuple]:
        """Return init param names so diffusers type validation does not KeyError on custom pipeline."""
        sig = inspect.signature(cls.__init__)
        empty = (inspect.Signature.empty,)
        return {name: empty for name in sig.parameters}
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: Any,
        scheduler: KarrasDiffusionSchedulers,
        inverse_scheduler: DDIMInverseScheduler,
        feature_extractor: CLIPFeatureExtractor,
        image_encoder=None,
        safety_checker = None,
        requires_safety_checker: bool = True,
        discriminator = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            inverse_scheduler=inverse_scheduler,
            feature_extractor=feature_extractor,
            discriminator=discriminator,
            image_encoder=image_encoder,
            safety_checker=safety_checker
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    @classmethod
    def from_pretrained_custom(cls, model_id, discriminator=None, **kwargs):
        orig_pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
        components = orig_pipe.components
        
        components["inverse_scheduler"] = DDIMInverseScheduler.from_config(
            components["scheduler"].config
        )
        components["discriminator"] = discriminator
        return cls(**components)

    # Borrow helper implementations from diffusers' StableDiffusionPipeline for convenience.
    enable_vae_slicing = DiffusersStableDiffusionPipeline.enable_vae_slicing
    disable_vae_slicing = DiffusersStableDiffusionPipeline.disable_vae_slicing
    enable_sequential_cpu_offload = DiffusersStableDiffusionPipeline.enable_sequential_cpu_offload
    _execution_device = DiffusersStableDiffusionPipeline._execution_device
    encode_prompt = DiffusersStableDiffusionPipeline.encode_prompt
    run_safety_checker = DiffusersStableDiffusionPipeline.run_safety_checker
    prepare_extra_step_kwargs = DiffusersStableDiffusionPipeline.prepare_extra_step_kwargs
    check_inputs = DiffusersStableDiffusionPipeline.check_inputs
    prepare_latents = DiffusersStableDiffusionPipeline.prepare_latents

    def prepare_metadata(
        self, batch_size, metadata, do_classifier_free_guidance, device, dtype,
    ):
        has_metadata = getattr(self.unet.config, "use_metadata", False)
        num_metadata = getattr(self.unet.config, "num_metadata", 0)

        if metadata is None and has_metadata and num_metadata > 0:
            metadata = torch.zeros((batch_size, num_metadata), device=device, dtype=dtype)

        if metadata is None:
            return None

        metadata = metadata_normalize(metadata).tolist()
        md = torch.tensor(metadata) if not torch.is_tensor(metadata) else metadata
        if len(md.shape) == 1:
            md = md.unsqueeze(0).expand(batch_size, -1)
        md = md.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            md = torch.cat([torch.zeros_like(md), md])

        return md
    
    def channel_shift(self, shift_latents, ref_latents):

        channel_num = ref_latents.shape[1]                    
        for cha in range(channel_num):
            mean_ch = shift_latents[:, cha:cha+1, :, :].mean() 
            mean_ch_ref = ref_latents[:, cha:cha+1, :, :].mean() 
            shift = mean_ch_ref - mean_ch
            shift_latents[:, cha, :, :] = shift_latents[:, cha, :, :] + shift 
        _, _, fg_h, fg_w = shift_latents.shape
        ref_latents[:, :, self.y:self.y+fg_h, self.x:self.x+fg_w] = shift_latents

        return ref_latents


    def harmonization_score(self, model, image, box):
        harm_size = 256
        box = [b//2 for b in box]

        tf = transforms.Compose([
        transforms.Resize((harm_size, harm_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        x = tf(image).to(self.device)
        m = create_mask(harm_size, harm_size, box).to(self.device)
        input = torch.cat([x, m], dim=0)
        input = input.unsqueeze(0).to(self.device)
        logit = model(input)
        score = torch.sigmoid(logit).item()
        return score

    def _get_inverted_latents(self, image, prompt, metadata, device, generator):
        tensor = self.image_processor.preprocess(image)
        latents = self.prepare_image_latents(
            tensor, 1, self.vae.dtype, device, generator
        )
        self.inverse_scheduler.set_timesteps(self.num_inference_steps, device=device)
        return latents, self.invert(metadata=metadata, prompt=prompt, latents=latents)

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)
        init_latents = self.vae.encode(image).latent_dist.sample(generator)
        init_latents = self.vae.config.scaling_factor * init_latents
        if batch_size > init_latents.shape[0]:
            init_latents = init_latents.repeat(batch_size // init_latents.shape[0], 1, 1, 1)
            
        return init_latents

    @torch.no_grad()
    def __call__(
        self,
        bg_prompt: str,
        fg_prompt: str,
        bg_image: Image.Image = None,
        fg_image: Image.Image = None,
        xy: List[float] = [0.3,0.3],
        start_ratio: float=0.3,
        end_ratio: float=0.7,   
        edge_width_ratio: float=0.1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        metadata: Optional[List[float]] = None,
        viz_checkpoints: str = None,
    ):


        # 0. Default height and width to unet
        self.height = height or self.unet.config.sample_size * self.vae_scale_factor
        self.width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.generator = generator
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        batch_size = 1
        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=fg_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        self.prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.timesteps = self.scheduler.timesteps

        # 5. Prepare metadata
        self.metadata = self.prepare_metadata(batch_size, metadata, True, device, self.unet.dtype)

        # 6. Get the inverted latents of the background and foreground images
        fg_latent, self.fg_latents = self._get_inverted_latents(
            fg_image, fg_prompt, metadata, device, generator
        )
        
        bg_latent, self.bg_latents = self._get_inverted_latents(
            bg_image, bg_prompt, metadata, device, generator
        )
        
        _, _, fg_h, fg_w = fg_latent.shape
        edge_width = int(min(fg_h, fg_w) * edge_width_ratio)

        # 7. Prepare the edge mask
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        self.x, self.y = round(xy[0]*latent_h), round(xy[1]*latent_w)
        self.box = [self.x, self.y, self.x+fg_w, self.y+fg_h]

        self.edge_mask = build_edge_mask(latent_h, latent_w, self.box, device=self.device, inner_edge_width=edge_width, outer_edge_width=edge_width).to(dtype=self.unet.dtype)
        
        if viz_checkpoints is not None:
            mask = self.edge_mask.squeeze().cpu().numpy()
            if mask.dtype != np.uint8:
                mask = (mask * 255).clip(0, 255).astype(np.uint8)     
            mask = Image.fromarray(mask)
            mask.save(f'{viz_checkpoints}/edge_mask.png')

            shift_latents = self.channel_shift(fg_latent, bg_latent)
            shift_image = self.vae.decode(shift_latents / self.vae.config.scaling_factor, return_dict=False)[0]
            shift_image = self.image_processor.postprocess(shift_image, output_type=output_type)[0] 
            shift_image.save(f'{viz_checkpoints}/shift_img.png')       

        # 8. Denoising loop with harmonization
        # the timestep order is denoising order, so start step is greater than end step.
        start_step = int(start_ratio * num_inference_steps)
        end_step = int(end_ratio * num_inference_steps)

        self.replace_step = self.timesteps[int((end_ratio+0.1) * num_inference_steps)].item()

        best_score = -np.inf
        best_image = None

        mid_timesteps = self.timesteps[start_step:end_step]
        # print(f'from {mid_timesteps[0]} to {mid_timesteps[-1]}')
        for ts in mid_timesteps.flip(0):

            bg_latents_ts = self.bg_latents[ts.item()]
            fg_latents_ts = self.fg_latents[ts.item()]
                    
            latents_ts = self.channel_shift(fg_latents_ts, bg_latents_ts)
            latents = self.sample(latents_ts, start_step=ts, replace_step=self.replace_step)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)[0]
            if viz_checkpoints is not None:
                image.save(f'{viz_checkpoints}/{ts.item()}.png')

            fgx1, fgy1, fgx2, fgy2 = round(xy[0]*self.width), round(xy[1]*self.height), round(xy[0]*self.width)+fg_image.width, round(xy[1]*self.height)+fg_image.height
            fg_box = [fgx1, fgy1, fgx2, fgy2]
            ew = edge_width*8
            inner_box = clamp_bbox_open(fgx1+ew, fgy1+ew, fgx2-ew, fgy2-ew, self.width, self.height)
            outer_box = clamp_bbox_open(fgx1-ew, fgy1-ew, fgx2+ew, fgy2+ew, self.width, self.height)

            score1 = self.harmonization_score(self.discriminator, image, fg_box)
            score2 = self.harmonization_score(self.discriminator, image, inner_box)
            score3 = self.harmonization_score(self.discriminator, image, outer_box)

            score = (score1  + score2 + score3)/3
            if viz_checkpoints is not None:
                print(f'{ts}, {score:.4f}, {score1:.4f},  {score2:.4f}, {score3:.4f}')
            if score > best_score:
                best_image = image
                best_score = score

        return best_image
    


    @torch.no_grad()
    def invert(
            self,
            prompt,
            metadata,
            latents: Optional[torch.FloatTensor] = None,
    ):

        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            device = self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        metadata = self.prepare_metadata(1, metadata, False, self.device, self.unet.dtype)

        inverted_latents = {}

        # Prepare timesteps
        timesteps = self.inverse_scheduler.timesteps

        with self.progress_bar(total=self.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                inverted_latents[t.item()] = latents

                latent_model_input = self.inverse_scheduler.scale_model_input(latents, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    metadata=metadata,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample
                
                progress_bar.update()

        return inverted_latents


    def sample(self, latents, start_step, replace_step):

        with self.progress_bar(total=self.num_inference_steps) as progress_bar:
            
            for i, t in enumerate(self.timesteps):
                if t >= start_step+1:
                    continue
                
                if t <= replace_step:
                    bg_latents_id = self.bg_latents[t.item()]
                    fg_latents_id = self.fg_latents[t.item()]
                    latents_id = self.channel_shift(fg_latents_id, bg_latents_id)
                    latents = self.edge_mask*latents + (1-self.edge_mask)*latents_id

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    metadata=self.metadata,
                    encoder_hidden_states=self.prompt_embeds,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample


                progress_bar.update()
                
        return latents


