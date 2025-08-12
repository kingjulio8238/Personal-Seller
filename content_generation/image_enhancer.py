"""
Image Enhancement System using OpenAI Image Edit API
Transforms non-professional product photos into high-quality social media content
"""

import os
import requests
import json
import base64
import time
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import openai
from datetime import datetime
import logging

class ImageEnhancer:
    """
    OpenAI Image Edit API integration for product photo enhancement
    Transforms amateur photos into professional-quality images optimized for social platforms
    """
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Platform-specific image requirements from config
        self.platform_specs = {
            'x': {
                'aspect_ratio': '16:9',
                'max_size': (1600, 900),
                'format': 'JPEG',
                'quality': 90,
                'style': 'clean and professional'
            },
            'tiktok': {
                'aspect_ratio': '9:16',
                'max_size': (1080, 1920),
                'format': 'JPEG',
                'quality': 95,
                'style': 'vibrant and trendy'
            },
            'instagram': {
                'aspect_ratio': '1:1',
                'max_size': (1080, 1080),
                'format': 'JPEG',
                'quality': 95,
                'style': 'aesthetic and polished'
            }
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_image(self, image_url: str) -> bytes:
        """Download image from URL"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            self.logger.error(f"Failed to download image from {image_url}: {e}")
            raise

    def create_mask_for_enhancement(self, image: Image.Image) -> Image.Image:
        """Create a mask for selective enhancement areas"""
        # Create a simple mask - in production this would be more sophisticated
        # For now, create a mask that targets the center area of the image
        width, height = image.size
        mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Create a centered rectangular mask
        left = width // 4
        top = height // 4
        right = 3 * width // 4
        bottom = 3 * height // 4
        
        # Fill the center area with white (areas to edit)
        for x in range(left, right):
            for y in range(top, bottom):
                mask.putpixel((x, y), (255, 255, 255, 255))
        
        return mask

    def generate_enhancement_prompt(self, platform: str, product_data: Dict[str, Any]) -> str:
        """Generate platform-specific enhancement prompt"""
        platform_spec = self.platform_specs.get(platform, self.platform_specs['x'])
        product_name = product_data.get('name', 'product')
        category = product_data.get('category', 'item')
        brand_voice = product_data.get('brand_voice', 'professional')
        
        base_prompt = f"Transform this {category} photo into a professional, high-quality product image suitable for {platform.upper()} marketing."
        
        platform_prompts = {
            'x': f"{base_prompt} Create a clean, professional background with optimal lighting. Focus on clarity and professionalism. Style: {platform_spec['style']}. Make the {product_name} stand out clearly against a neutral background.",
            
            'tiktok': f"{base_prompt} Create a vibrant, eye-catching background that would appeal to TikTok's young audience. Use trendy colors and modern styling. Style: {platform_spec['style']}. Make the {product_name} pop with dynamic lighting and contemporary aesthetics.",
            
            'instagram': f"{base_prompt} Create an aesthetically pleasing, Instagram-worthy background with perfect lighting and styling. Focus on visual appeal and shareability. Style: {platform_spec['style']}. Make the {product_name} look premium and photogenic with soft, appealing lighting."
        }
        
        prompt = platform_prompts.get(platform, platform_prompts['x'])
        
        # Add brand voice considerations
        if 'luxury' in brand_voice.lower():
            prompt += " Emphasize premium quality and elegance."
        elif 'casual' in brand_voice.lower():
            prompt += " Keep it approachable and friendly."
        elif 'tech' in brand_voice.lower():
            prompt += " Use modern, sleek styling with clean lines."
        
        return prompt

    def resize_image_for_platform(self, image: Image.Image, platform: str) -> Image.Image:
        """Resize and optimize image for specific platform requirements"""
        platform_spec = self.platform_specs.get(platform, self.platform_specs['x'])
        target_size = platform_spec['max_size']
        aspect_ratio = platform_spec['aspect_ratio']
        
        # Calculate target dimensions based on aspect ratio
        if aspect_ratio == '16:9':
            if image.width / image.height > 16/9:
                # Image is wider than 16:9, crop height
                new_height = int(image.width * 9 / 16)
                top = (image.height - new_height) // 2
                image = image.crop((0, top, image.width, top + new_height))
            else:
                # Image is taller than 16:9, crop width
                new_width = int(image.height * 16 / 9)
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, image.height))
        
        elif aspect_ratio == '9:16':
            if image.width / image.height > 9/16:
                # Image is wider than 9:16, crop width
                new_width = int(image.height * 9 / 16)
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, image.height))
            else:
                # Image is taller than 9:16, crop height
                new_height = int(image.width * 16 / 9)
                top = (image.height - new_height) // 2
                image = image.crop((0, top, image.width, top + new_height))
        
        elif aspect_ratio == '1:1':
            # Square crop - crop to smallest dimension
            min_dim = min(image.width, image.height)
            left = (image.width - min_dim) // 2
            top = (image.height - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))
        
        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image

    def enhance_image_with_openai(self, image_bytes: bytes, platform: str, product_data: Dict[str, Any]) -> bytes:
        """Use OpenAI Image Edit API to enhance the product image"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_bytes))
            
            # Ensure image is in RGBA format for editing
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Create mask for selective editing
            mask = self.create_mask_for_enhancement(image)
            
            # Save temporary files for API call
            temp_image_path = os.path.join(self.temp_dir, f'temp_image_{int(time.time())}.png')
            temp_mask_path = os.path.join(self.temp_dir, f'temp_mask_{int(time.time())}.png')
            
            image.save(temp_image_path, 'PNG')
            mask.save(temp_mask_path, 'PNG')
            
            # Generate enhancement prompt
            prompt = self.generate_enhancement_prompt(platform, product_data)
            
            self.logger.info(f"Enhancing image for {platform} with prompt: {prompt[:100]}...")
            
            # Call OpenAI Image Edit API
            with open(temp_image_path, 'rb') as image_file, open(temp_mask_path, 'rb') as mask_file:
                response = self.client.images.edit(
                    image=image_file,
                    mask=mask_file,
                    prompt=prompt,
                    n=1,
                    size="1024x1024",  # OpenAI's supported size
                    response_format="url"
                )
            
            # Download the enhanced image
            enhanced_image_url = response.data[0].url
            enhanced_image_response = requests.get(enhanced_image_url, timeout=30)
            enhanced_image_response.raise_for_status()
            
            # Clean up temporary files
            os.remove(temp_image_path)
            os.remove(temp_mask_path)
            
            self.logger.info("Image enhancement completed successfully")
            return enhanced_image_response.content
            
        except Exception as e:
            self.logger.error(f"OpenAI image enhancement failed: {e}")
            # Return original image if enhancement fails
            return image_bytes

    def apply_final_optimizations(self, image_bytes: bytes, platform: str) -> bytes:
        """Apply final optimizations for the specific platform"""
        try:
            image = Image.open(BytesIO(image_bytes))
            platform_spec = self.platform_specs.get(platform, self.platform_specs['x'])
            
            # Resize for platform
            image = self.resize_image_for_platform(image, platform)
            
            # Apply platform-specific enhancements
            if platform == 'tiktok':
                # Increase saturation and contrast for TikTok
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.2)  # 20% more saturation
                
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)  # 10% more contrast
            
            elif platform == 'instagram':
                # Apply subtle sharpening for Instagram
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            elif platform == 'x':
                # Optimize for clarity and readability
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)  # Slight sharpening
            
            # Convert to RGB if needed and save with quality settings
            if image.mode in ('RGBA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(image)
                image = background
            
            # Save optimized image
            output_buffer = BytesIO()
            image.save(
                output_buffer,
                format=platform_spec['format'],
                quality=platform_spec['quality'],
                optimize=True
            )
            
            return output_buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Final optimization failed: {e}")
            return image_bytes

    def enhance_for_platform(self, base_image_url: str, platform: str, product_data: Dict[str, Any]) -> str:
        """
        Main method: Enhance product image for specific platform
        Returns URL or path to enhanced image
        """
        try:
            self.logger.info(f"Starting image enhancement for {platform}")
            
            # Step 1: Download original image
            original_image_bytes = self.download_image(base_image_url)
            
            # Step 2: Enhance using OpenAI Image Edit API
            enhanced_image_bytes = self.enhance_image_with_openai(
                original_image_bytes, platform, product_data
            )
            
            # Step 3: Apply platform-specific optimizations
            final_image_bytes = self.apply_final_optimizations(enhanced_image_bytes, platform)
            
            # Step 4: Save enhanced image
            timestamp = int(time.time())
            product_name = product_data.get('name', 'product').replace(' ', '_')
            filename = f"{product_name}_{platform}_{timestamp}.jpg"
            output_path = os.path.join(self.temp_dir, filename)
            
            with open(output_path, 'wb') as f:
                f.write(final_image_bytes)
            
            self.logger.info(f"Image enhancement completed: {output_path}")
            
            # In production, this would upload to cloud storage and return URL
            # For demo, return local file path
            return output_path
            
        except Exception as e:
            self.logger.error(f"Image enhancement failed completely: {e}")
            # Return original URL if all enhancement fails
            return base_image_url

    def create_image_variants(self, base_image_url: str, product_data: Dict[str, Any], 
                             platforms: List[str] = None) -> Dict[str, str]:
        """Create enhanced image variants for multiple platforms"""
        if platforms is None:
            platforms = ['x', 'tiktok', 'instagram']
        
        variants = {}
        
        for platform in platforms:
            try:
                enhanced_url = self.enhance_for_platform(base_image_url, platform, product_data)
                variants[platform] = enhanced_url
                self.logger.info(f"Created {platform} variant: {enhanced_url}")
            except Exception as e:
                self.logger.error(f"Failed to create {platform} variant: {e}")
                variants[platform] = base_image_url  # Fallback to original
        
        return variants

    def get_enhancement_metadata(self, enhanced_image_path: str) -> Dict[str, Any]:
        """Get metadata about the enhanced image"""
        try:
            with Image.open(enhanced_image_path) as img:
                return {
                    'dimensions': img.size,
                    'format': img.format,
                    'mode': img.mode,
                    'file_size': os.path.getsize(enhanced_image_path),
                    'created_at': datetime.utcnow().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Failed to get image metadata: {e}")
            return {}

if __name__ == "__main__":
    # Test the image enhancer
    enhancer = ImageEnhancer()
    
    # Sample product data
    sample_product = {
        'name': 'Wireless Headphones',
        'category': 'electronics',
        'brand_voice': 'modern and tech-savvy'
    }
    
    # Test image enhancement (would need actual image URL)
    # enhanced_path = enhancer.enhance_for_platform(
    #     "https://example.com/product-image.jpg",
    #     "instagram", 
    #     sample_product
    # )
    # print(f"Enhanced image saved to: {enhanced_path}")
    
    print("Image enhancer initialized successfully")