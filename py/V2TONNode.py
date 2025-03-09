import os
import torch
import numpy as np
from PIL import Image, ImageFilter
import torch.nn.functional as F
from huggingface_hub import snapshot_download

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class V2TONNode:
    """
    ComfyUI custom node for V2TON virtual try-on functionality.
    This allows clothing items to be virtually fitted onto person images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "base_model_path": ("STRING", {"default": "alibaba-pai/EasyAnimateV4-XL-2-InP"}),
                "v2ton_model_path": ("STRING", {"default": "zhengchong/CatV2TON"}),
                "catvton_path": ("STRING", {"default": "zhengchong/CatVTON"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0}),
            },
            "optional": {
                "seed": ("INT", {"default": 555}),
                "repaint": ("BOOLEAN", {"default": True}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "width": ("INT", {"default": 384, "min": 192, "max": 768}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_virtual_tryon"
    CATEGORY = "image/processing"
    
    def __init__(self):
        self.pipeline = None
        self.automasker = None
        self.densepose = None
        self.vae_processor = None
        
    def load_models(self, base_model_path, v2ton_model_path, catvton_path):
        """Load all required models for inference."""
        import sys
        
        # Add the modules directory to the path if not already there
        modules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
        if modules_path not in sys.path:
            sys.path.append(modules_path)
            
        # Now import the necessary modules
        from .modules.pipeline import V2TONPipeline
        from .modules.cloth_masker import AutoMasker
        from .modules.densepose import DensePose
        from diffusers.image_processor import VaeImageProcessor
        
        # Download models from HuggingFace if not available locally
        base_model_path = snapshot_download(base_model_path) if not os.path.exists(base_model_path) else base_model_path
        v2ton_model_path = snapshot_download(v2ton_model_path) if not os.path.exists(v2ton_model_path) else v2ton_model_path
        catvton_path = snapshot_download(catvton_path) if not os.path.exists(catvton_path) else catvton_path
        
        # Path to the finetuned model
        finetuned_model_path = os.path.join(v2ton_model_path, "512-64K")
        
        # Load V2TON pipeline
        if self.pipeline is None:
            print(f"Loading V2TON pipeline from {base_model_path} and {finetuned_model_path}")
            self.pipeline = V2TONPipeline(
                base_model_path=base_model_path,
                finetuned_model_path=finetuned_model_path,
                torch_dtype=torch.float16,
                device="cuda",
                load_pose=True
            )
            
        # Load AutoMasker for generating cloth masks
        if self.automasker is None:
            print(f"Loading AutoMasker from {catvton_path}")
            self.automasker = AutoMasker(
                densepose_ckpt=os.path.join(catvton_path, "DensePose"),
                schp_ckpt=os.path.join(catvton_path, "SCHP"),
                device='cuda'
            )
            
        # Load DensePose for pose estimation
        if self.densepose is None:
            print(f"Loading DensePose from {catvton_path}")
            self.densepose = DensePose(
                model_path=os.path.join(catvton_path, "DensePose"),
                device='cuda'
            )
            
        # Initialize VAE image processor
        if self.vae_processor is None:
            self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
            self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        
    def preprocess_images(self, person_img, garment_img, height, width):
        """
        Preprocess input images for the model.
        - Convert ComfyUI images to PIL format
        - Generate pose and mask
        - Prepare for model input
        """
        # Convert ComfyUI image format to PIL
        person_np = (person_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        person_pil = Image.fromarray(person_np)
        
        garment_np = (garment_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        garment_pil = Image.fromarray(garment_np)
        
        # Generate pose and mask using the automasker
        conditions = self.automasker(person_pil, mask_type='upper')
        pose_pil = conditions['densepose']
        mask_pil = conditions['mask']
        
        # Process images with VAE processor
        person_tensor = self.vae_processor.preprocess(person_pil, height, width)[0].unsqueeze(1)
        garment_tensor = self.vae_processor.preprocess(garment_pil, height, width)[0].unsqueeze(1)
        pose_tensor = self.vae_processor.preprocess(pose_pil, height, width)[0].unsqueeze(1)
        mask_tensor = self.vae_processor.preprocess(mask_pil, height, width)[0].unsqueeze(1)
        
        return {
            'person_pil': person_pil,
            'garment_pil': garment_pil,
            'pose_pil': pose_pil,
            'mask_pil': mask_pil,
            'person_tensor': person_tensor,
            'garment_tensor': garment_tensor,
            'pose_tensor': pose_tensor,
            'mask_tensor': mask_tensor
        }
    
    def to_pil_image(self, images):
        """Convert tensor to PIL images"""
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images
    
    def repaint(self, person, mask, result, use_gaussian_blur=True):
        """Repaint the result with the original background using the mask"""
        w, h = result.size
        kernal_size = h // 50
        if kernal_size % 2 == 0:
            kernal_size += 1
        if use_gaussian_blur:
            mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
        person_np = np.array(person)
        result_np = np.array(result)
        mask_np = np.array(mask) / 255
        repaint_result = person_np * (1 - mask_np) + result_np * mask_np
        repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
        return repaint_result
    
    def apply_virtual_tryon(self, person_image, garment_image, base_model_path, v2ton_model_path, catvton_path, 
                           num_inference_steps=30, guidance_scale=3.0, seed=555, repaint=True, height=512, width=384):
        """
        Apply V2TON virtual try-on to fit garment onto person.
        
        Args:
            person_image: Image of the person
            garment_image: Image of the garment to try on
            base_model_path: Path to the base model
            v2ton_model_path: Path to the V2TON model
            catvton_path: Path to the CatVTON model
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for reproducibility
            repaint: Whether to repaint the result with the original background
            height: Height of the processed images
            width: Width of the processed images
            
        Returns:
            Rendered image with garment fitted on person
        """
        # Load all required models
        self.load_models(base_model_path, v2ton_model_path, catvton_path)
        
        # Preprocess images
        processed = self.preprocess_images(person_image, garment_image, height, width)
        
        # Setup generator for reproducibility
        generator = torch.Generator(device='cuda').manual_seed(seed)
        
        # Run inference
        results = self.pipeline.image_try_on(
            source_image=processed['person_tensor'],
            source_mask=processed['mask_tensor'],
            conditioned_image=processed['garment_tensor'],
            pose_image=processed['pose_tensor'],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Postprocess the result
        result = results[0]  # Get the first (only) result
        
        # Apply repainting if requested
        if repaint:
            person_pil = processed['person_pil'].resize(result.size, Image.LANCZOS)
            mask_pil = processed['mask_pil'].resize(result.size, Image.NEAREST).convert('RGB')
            result = self.repaint(person_pil, mask_pil, result)
        
        # Convert back to ComfyUI format [B, H, W, C] in range [0, 1]
        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        
        return (result_tensor,)

# This is what ComfyUI will use to identify and register your node
NODE_CLASS_MAPPINGS = {
    "V2TONNode": V2TONNode
}

# Optional: provides categories for organization in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "V2TONNode": "CatV2TON Wrapper"
}
