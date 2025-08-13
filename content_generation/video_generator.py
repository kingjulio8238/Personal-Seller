"""
Comprehensive Video Generation Pipeline using Google Veo 3 API
Transforms enhanced product images into dynamic video content for social media platforms

Features:
- Real Google Veo 3 API integration with advanced video generation
- Text-to-video generation with custom prompts
- Image-to-video conversion from enhanced images
- Platform-specific optimization (TikTok, Instagram, X/Twitter, LinkedIn)
- Batch processing with queue management
- Video editing capabilities (trimming, effects, transitions)
- Quality validation and content moderation
- Cost tracking and optimization
- Comprehensive error handling and retry mechanisms
"""

import os
import requests
import json
import time
import asyncio
import threading
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import tempfile
import queue
from io import BytesIO
import base64

# Google Cloud imports
from google.oauth2 import service_account
from google.cloud import storage
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# Video processing imports
from moviepy.editor import (
    VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip,
    TextClip, ColorClip, concatenate_videoclips, vfx, afx
)
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Database integration
try:
    from ..database.models import DatabaseManager, Product, Post, EngagementMetrics
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Mock API for testing when Google Cloud credentials are not available
class MockVeo3Service:
    """Mock Google Veo 3 service for testing without API credentials"""
    
    def __init__(self):
        self.call_count = 0
        self.last_request = None
        
    class Projects:
        def __init__(self, parent):
            self.parent = parent
            
        def locations(self):
            return self.parent.Locations(self.parent)
            
        class Locations:
            def __init__(self, parent):
                self.parent = parent
                
            def publishers(self):
                return self.parent.Publishers(self.parent)
                
            class Publishers:
                def __init__(self, parent):
                    self.parent = parent
                    
                def models(self):
                    return self.parent.Models(self.parent)
                    
                class Models:
                    def __init__(self, parent):
                        self.parent = parent
                        
                    def predict(self, endpoint=None, body=None):
                        return self.parent.PredictRequest(self.parent, endpoint, body)
                        
                    class PredictRequest:
                        def __init__(self, parent, endpoint, body):
                            self.parent = parent
                            self.endpoint = endpoint
                            self.body = body
                            
                        def execute(self):
                            self.parent.call_count += 1
                            self.parent.last_request = self.body
                            
                            # Return mock successful response
                            return {
                                'predictions': [{
                                    'video_bytes': base64.b64encode(b'mock_video_data').decode(),
                                    'metadata': {
                                        'duration': self.body['instances'][0]['parameters']['duration'],
                                        'resolution': self.body['instances'][0]['parameters']['resolution'],
                                        'fps': self.body['instances'][0]['parameters']['fps']
                                    }
                                }]
                            }
    
    def projects(self):
        return self.Projects(self)

class MockGoogleCloudStorage:
    """Mock Google Cloud Storage for testing"""
    
    def __init__(self):
        self.uploads = []
        self.downloads = []
        
    def bucket(self, name):
        return self.MockBucket(self, name)
        
    class MockBucket:
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name
            
        def blob(self, name):
            return self.parent.MockBlob(self.parent, name)
            
        class MockBlob:
            def __init__(self, parent, name):
                self.parent = parent
                self.name = name
                self.public_url = f"https://storage.googleapis.com/mock-bucket/{name}"
                
            def upload_from_filename(self, filename):
                self.parent.uploads.append({
                    'filename': filename,
                    'blob_name': self.name,
                    'timestamp': datetime.now()
                })
                
            def download_to_filename(self, filename):
                self.parent.downloads.append({
                    'filename': filename,
                    'blob_name': self.name,
                    'timestamp': datetime.now()
                })
                # Create a mock video file for testing
                with open(filename, 'wb') as f:
                    f.write(b'mock_video_data')
                    
            def make_public(self):
                pass

@dataclass
class VideoGenerationRequest:
    """Structured request for video generation"""
    input_type: str  # 'text', 'image', 'video'
    input_data: str  # Path or text prompt
    platform: str
    product_data: Dict[str, Any]
    style: str = 'professional'
    duration: int = 30
    quality: str = 'hd'  # 'hd', 'standard'
    priority: str = 'normal'
    max_retries: int = 3
    use_moderation: bool = True
    enhance_audio: bool = False
    add_captions: bool = False

@dataclass
class VideoGenerationResult:
    """Structured result from video generation"""
    success: bool
    video_path: Optional[str]
    metadata: Dict[str, Any]
    cost: Decimal
    processing_time: float
    platform: str
    error_message: Optional[str] = None
    moderation_result: Optional[Dict[str, Any]] = None
    thumbnail_path: Optional[str] = None

class VideoGenerator:
    """
    Comprehensive Google Veo 3 API integration for professional video generation
    Creates dynamic videos from enhanced product images for social media platforms
    
    Key Features:
    - Real Google Veo 3 API integration
    - Multi-platform optimization (TikTok, Instagram, X, LinkedIn)
    - Batch processing with intelligent queueing
    - Advanced video editing and effects
    - Cost tracking and optimization
    - Content moderation and quality validation
    """
    
    def __init__(self, db_session=None):
        # Google Veo 3 API setup
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.veo_service_account_path = os.getenv('GOOGLE_VEO_SERVICE_ACCOUNT_PATH')
        self.google_project_id = os.getenv('GOOGLE_PROJECT_ID')
        
        # Database integration
        self.db_manager = None
        if DATABASE_AVAILABLE and db_session:
            self.db_manager = DatabaseManager(db_session)
        
        # Advanced processing queue and rate limiting
        self.processing_queue = queue.PriorityQueue()
        self.failed_queue = queue.Queue()  # For failed requests that need retry
        self._request_timestamps = []
        self._rate_limit_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Advanced queue management
        self.queue_manager = {
            'high_priority': queue.PriorityQueue(),
            'normal_priority': queue.PriorityQueue(),
            'low_priority': queue.PriorityQueue(),
            'retry_queue': queue.Queue(),
            'processing_status': {},  # Track request processing status
            'completion_callbacks': {}  # Callbacks for completed requests
        }
        
        # Enhanced batch processing settings
        self.batch_settings = {
            'max_concurrent_videos': 3,
            'retry_failed_requests': True,
            'max_retries_per_request': 2,
            'batch_timeout_seconds': 300,  # 5 minutes
            'prioritize_by_platform': {
                'tiktok': 1, 
                'instagram': 2, 
                'youtube_shorts': 3,
                'x': 4, 
                'pinterest': 5,
                'linkedin': 6
            }
        }
        
        # Enhanced cost tracking (Google Veo 3 pricing - estimated)
        self._cost_tracker = {
            'total_cost': Decimal('0.00'),
            'daily_cost': Decimal('0.00'),
            'monthly_cost': Decimal('0.00'),
            'last_reset': datetime.now().date(),
            'last_monthly_reset': datetime.now().date().replace(day=1),
            'requests_count': 0,
            'successful_requests': 0,
            'platform_costs': {
                'x': Decimal('0.00'), 
                'tiktok': Decimal('0.00'), 
                'instagram': Decimal('0.00'), 
                'linkedin': Decimal('0.00'),
                'pinterest': Decimal('0.00'),
                'youtube_shorts': Decimal('0.00')
            },
            'cost_by_quality': {'hd': Decimal('0.00'), 'standard': Decimal('0.00')},
            'cost_by_type': {'text_to_video': Decimal('0.00'), 'image_to_video': Decimal('0.00'), 'video_edit': Decimal('0.00')}
        }
        
        # Cost optimization settings
        self.cost_limits = {
            'daily_limit': Decimal('100.00'),  # $100 daily limit
            'monthly_limit': Decimal('2000.00'),  # $2000 monthly limit
            'warning_threshold': 0.8,  # Warn at 80% of limit
            'auto_downgrade_quality': True  # Auto-downgrade to standard quality when approaching limits
        }
        
        # Veo 3 API pricing (estimated future pricing)
        self.pricing = {
            'text_to_video_hd': Decimal('0.50'),  # per 30-second video
            'image_to_video_hd': Decimal('0.35'), 
            'video_edit_hd': Decimal('0.25'),
            'text_to_video_standard': Decimal('0.25'),
            'image_to_video_standard': Decimal('0.20'),
            'video_edit_standard': Decimal('0.15')
        }
        
        # Enhanced platform-specific video requirements
        self.platform_specs = {
            'x': {
                'aspect_ratio': '16:9',
                'dimensions': (1280, 720),
                'max_duration': 140,
                'format': 'mp4',
                'fps': 30,
                'bitrate': '2M',
                'style': 'professional and informative',
                'subtitle_safe_area': {'top': 80, 'bottom': 80},
                'trending_features': ['clean_transitions', 'professional_text', 'data_visualization'],
                'optimal_duration': 30,
                'supports_threads': True,
                'engagement_hooks': ['question_starters', 'poll_integration', 'call_to_action']
            },
            'tiktok': {
                'aspect_ratio': '9:16',
                'dimensions': (1080, 1920),
                'max_duration': 180,
                'format': 'mp4',
                'fps': 30,
                'bitrate': '3M',
                'style': 'dynamic and trendy',
                'subtitle_safe_area': {'top': 100, 'bottom': 200},
                'trending_features': ['quick_cuts', 'zoom_effects', 'trending_sounds', 'hashtag_challenges'],
                'optimal_duration': 15,
                'music_integration': True,
                'popular_effects': ['beauty_filter', 'speed_ramp', 'split_screen', 'duet_ready', 'green_screen'],
                'trending_audio_types': ['viral_sounds', 'trending_music', 'voice_effects'],
                'content_pillars': ['entertainment', 'education', 'behind_scenes', 'trending_participation']
            },
            'instagram': {
                'aspect_ratio': '9:16',
                'dimensions': (1080, 1920), 
                'max_duration': 60,
                'format': 'mp4',
                'fps': 30,
                'bitrate': '3.5M',
                'style': 'aesthetic and engaging',
                'subtitle_safe_area': {'top': 100, 'bottom': 120},
                'trending_features': ['smooth_transitions', 'lifestyle_context', 'story_integration'],
                'optimal_duration': 30,
                'story_safe_zones': True,
                'supports_shopping_tags': True,
                'alternative_ratios': [(1080, 1080), (1080, 1350)],  # Square and 4:5 formats
                'content_types': ['reels', 'stories', 'feed_video']
            },
            'linkedin': {
                'aspect_ratio': '16:9',
                'dimensions': (1920, 1080),
                'max_duration': 600,
                'format': 'mp4',
                'fps': 30,
                'bitrate': '4M',
                'style': 'professional and business-focused',
                'subtitle_safe_area': {'top': 100, 'bottom': 100},
                'trending_features': ['thought_leadership', 'industry_insights', 'professional_testimonials'],
                'optimal_duration': 90,
                'captions_required': True,
                'business_elements': ['company_branding', 'professional_cta', 'industry_context']
            },
            'pinterest': {
                'aspect_ratio': '9:16',
                'dimensions': (1080, 1920),
                'max_duration': 15,
                'format': 'mp4',
                'fps': 30,
                'bitrate': '2M',
                'style': 'inspirational and visually appealing',
                'subtitle_safe_area': {'top': 50, 'bottom': 50},
                'trending_features': ['diy_tutorials', 'before_after', 'lifestyle_inspiration'],
                'optimal_duration': 6,
                'engagement_hooks': ['save_worthy', 'how_to_content', 'seasonal_trends'],
                'visual_elements': ['bright_colors', 'clear_text_overlay', 'step_by_step'],
                'supports_idea_pins': True
            },
            'youtube_shorts': {
                'aspect_ratio': '9:16',
                'dimensions': (1080, 1920),
                'max_duration': 60,
                'format': 'mp4',
                'fps': 30,
                'bitrate': '3M',
                'style': 'engaging and informative',
                'subtitle_safe_area': {'top': 100, 'bottom': 150},
                'trending_features': ['tutorial_format', 'quick_tips', 'entertainment'],
                'optimal_duration': 30,
                'content_hooks': ['surprising_facts', 'how_to_guides', 'product_demos']
            }
        }
        
        # Directory setup
        self.temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
        self.cache_dir = os.path.join(self.temp_dir, 'video_cache')
        self.audio_dir = os.path.join(self.temp_dir, 'audio')
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Google services
        self.veo_service = None
        self.storage_client = None
        self.using_mock_services = False
        self._init_google_services()
        
        # Content moderation setup
        self.moderation_enabled = True
        self.quality_thresholds = {
            'min_duration': 5,
            'max_duration': 300,
            'min_resolution': (720, 480),
            'min_bitrate': 500000  # 500 kbps
        }
        
        # Video template system for consistent branding
        self.template_system = {
            'templates': self._initialize_video_templates(),
            'brand_elements': {
                'colors': {
                    'primary': '#1E40AF',    # Blue
                    'secondary': '#F59E0B',  # Amber
                    'accent': '#10B981',     # Emerald
                    'neutral': '#6B7280'     # Gray
                },
                'fonts': {
                    'primary': 'Arial-Bold',
                    'secondary': 'Arial',
                    'accent': 'Helvetica'
                },
                'logo_positions': {
                    'bottom_right': (0.85, 0.9),
                    'top_left': (0.05, 0.1),
                    'center_bottom': (0.5, 0.9)
                }
            },
            'custom_templates': {}  # User-defined templates
        }

    def _init_google_services(self):
        """Initialize Google API services including Veo 3 and Cloud Storage"""
        try:
            if self.veo_service_account_path and os.path.exists(self.veo_service_account_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.veo_service_account_path,
                    scopes=[
                        'https://www.googleapis.com/auth/cloud-platform',
                        'https://www.googleapis.com/auth/cloud-platform.read-only',
                        'https://www.googleapis.com/auth/devstorage.read_write'
                    ]
                )
                
                # Initialize Google Veo 3 service
                try:
                    self.veo_service = googleapiclient.discovery.build(
                        'aiplatform', 'v1beta1', credentials=credentials
                    )
                    self.logger.info("Google Veo 3 service initialized")
                except Exception as e:
                    self.logger.warning(f"Veo 3 service not available, using fallback: {e}")
                
                # Initialize Cloud Storage
                self.storage_client = storage.Client(
                    credentials=credentials,
                    project=self.google_project_id
                )
                
                self.using_mock_services = False
                self.logger.info("Google services initialized successfully")
            else:
                self.logger.warning("Google service account not configured")
                self._use_fallback_mode()
        except Exception as e:
            self.logger.error(f"Failed to initialize Google services: {e}")
            self._use_fallback_mode()
    
    def _use_fallback_mode(self):
        """Enable fallback mode with mock services when Google services are unavailable"""
        self.logger.info("Using mock services for testing/development mode")
        self.veo_service = MockVeo3Service()
        self.storage_client = MockGoogleCloudStorage()
        self.using_mock_services = True

    def _initialize_video_templates(self) -> Dict[str, Any]:
        """Initialize built-in video templates for consistent branding"""
        return {
            'professional': {
                'name': 'Professional Business',
                'description': 'Clean, professional template for business content',
                'platforms': ['linkedin', 'x'],
                'elements': {
                    'intro_duration': 2,
                    'outro_duration': 2,
                    'background_color': (240, 240, 240),
                    'text_color': (50, 50, 50),
                    'accent_color': (30, 64, 175),  # Blue
                    'font_primary': 'Arial-Bold',
                    'font_secondary': 'Arial',
                    'logo_position': 'bottom_right',
                    'transitions': ['fade', 'slide'],
                    'effects': ['professional_zoom', 'subtle_glow']
                },
                'layout': {
                    'title_position': (0.5, 0.2),
                    'content_area': (0.1, 0.3, 0.9, 0.7),
                    'branding_area': (0.8, 0.85, 1.0, 1.0)
                }
            },
            'trendy': {
                'name': 'Trendy Social',
                'description': 'Dynamic template for social media platforms',
                'platforms': ['tiktok', 'instagram', 'youtube_shorts'],
                'elements': {
                    'intro_duration': 1,
                    'outro_duration': 1,
                    'background_gradient': [(255, 100, 150), (100, 200, 255)],
                    'text_color': (255, 255, 255),
                    'accent_color': (255, 215, 0),  # Gold
                    'font_primary': 'Arial-Bold',
                    'font_secondary': 'Arial',
                    'logo_position': 'center_bottom',
                    'transitions': ['zoom', 'flash', 'slide'],
                    'effects': ['particle_overlay', 'dynamic_border', 'color_shift']
                },
                'layout': {
                    'title_position': (0.5, 0.15),
                    'content_area': (0.05, 0.25, 0.95, 0.75),
                    'branding_area': (0.2, 0.85, 0.8, 1.0)
                }
            },
            'lifestyle': {
                'name': 'Lifestyle & Inspiration',
                'description': 'Aesthetic template for lifestyle content',
                'platforms': ['instagram', 'pinterest'],
                'elements': {
                    'intro_duration': 1.5,
                    'outro_duration': 2,
                    'background_color': (250, 250, 245),  # Warm white
                    'text_color': (100, 100, 100),
                    'accent_color': (220, 160, 130),  # Warm brown
                    'font_primary': 'Helvetica-Light',
                    'font_secondary': 'Helvetica',
                    'logo_position': 'top_left',
                    'transitions': ['fade', 'smooth_slide'],
                    'effects': ['soft_glow', 'warm_filter']
                },
                'layout': {
                    'title_position': (0.5, 0.25),
                    'content_area': (0.1, 0.35, 0.9, 0.8),
                    'branding_area': (0.05, 0.05, 0.3, 0.2)
                }
            },
            'minimal': {
                'name': 'Minimal Clean',
                'description': 'Clean, minimal template for all platforms',
                'platforms': ['x', 'linkedin', 'instagram', 'youtube_shorts'],
                'elements': {
                    'intro_duration': 1,
                    'outro_duration': 1.5,
                    'background_color': (255, 255, 255),
                    'text_color': (30, 30, 30),
                    'accent_color': (100, 100, 100),
                    'font_primary': 'Arial',
                    'font_secondary': 'Arial',
                    'logo_position': 'bottom_right',
                    'transitions': ['fade'],
                    'effects': ['subtle_shadow']
                },
                'layout': {
                    'title_position': (0.5, 0.3),
                    'content_area': (0.15, 0.4, 0.85, 0.8),
                    'branding_area': (0.85, 0.9, 1.0, 1.0)
                }
            }
        }
    
    def create_video_with_template(self, template_name: str, request: VideoGenerationRequest, 
                                 custom_brand_elements: Dict[str, Any] = None) -> VideoGenerationRequest:
        """Apply video template to enhance request with consistent branding"""
        try:
            template = self.template_system['templates'].get(template_name)
            if not template:
                # Check custom templates
                template = self.template_system['custom_templates'].get(template_name)
                if not template:
                    self.logger.warning(f"Template '{template_name}' not found, using default styling")
                    return request
            
            # Check if template supports the target platform
            if request.platform not in template['platforms']:
                self.logger.info(f"Template '{template_name}' not optimized for {request.platform}, adapting")
            
            # Apply template elements to request
            enhanced_request = self._apply_template_to_request(request, template, custom_brand_elements)
            
            return enhanced_request
            
        except Exception as e:
            self.logger.error(f"Failed to apply template '{template_name}': {e}")
            return request
    
    def _apply_template_to_request(self, request: VideoGenerationRequest, template: Dict[str, Any], 
                                 custom_brand_elements: Dict[str, Any] = None) -> VideoGenerationRequest:
        """Apply template styling to video generation request"""
        
        # Merge custom brand elements with template defaults
        brand_elements = self.template_system['brand_elements'].copy()
        if custom_brand_elements:
            brand_elements.update(custom_brand_elements)
        
        # Enhance product data with template information
        enhanced_product_data = request.product_data.copy()
        enhanced_product_data['template_info'] = {
            'template_name': template['name'],
            'template_elements': template['elements'],
            'brand_elements': brand_elements,
            'layout': template['layout']
        }
        
        # Create enhanced request
        enhanced_request = VideoGenerationRequest(
            input_type=request.input_type,
            input_data=request.input_data,
            platform=request.platform,
            product_data=enhanced_product_data,
            style=f"{request.style} with {template['name']} template",
            duration=request.duration,
            quality=request.quality,
            priority=request.priority,
            max_retries=request.max_retries,
            use_moderation=request.use_moderation,
            enhance_audio=request.enhance_audio,
            add_captions=request.add_captions
        )
        
        return enhanced_request
    
    def get_available_templates(self, platform: str = None) -> List[Dict[str, Any]]:
        """Get list of available templates, optionally filtered by platform"""
        templates = []
        
        # Built-in templates
        for template_id, template in self.template_system['templates'].items():
            if platform is None or platform in template['platforms']:
                templates.append({
                    'id': template_id,
                    'name': template['name'],
                    'description': template['description'],
                    'platforms': template['platforms'],
                    'type': 'built-in'
                })
        
        # Custom templates
        for template_id, template in self.template_system['custom_templates'].items():
            if platform is None or platform in template.get('platforms', []):
                templates.append({
                    'id': template_id,
                    'name': template.get('name', template_id),
                    'description': template.get('description', 'Custom template'),
                    'platforms': template.get('platforms', ['all']),
                    'type': 'custom'
                })
        
        return templates
    
    def create_custom_template(self, template_id: str, template_config: Dict[str, Any]) -> bool:
        """Create a custom video template"""
        try:
            # Validate template configuration
            required_fields = ['name', 'elements']
            for field in required_fields:
                if field not in template_config:
                    self.logger.error(f"Template missing required field: {field}")
                    return False
            
            # Set default values
            template_config.setdefault('platforms', ['x', 'tiktok', 'instagram', 'linkedin'])
            template_config.setdefault('description', f"Custom template: {template_config['name']}")
            
            # Validate elements
            elements = template_config['elements']
            if 'background_color' not in elements and 'background_gradient' not in elements:
                elements['background_color'] = (255, 255, 255)  # Default white
            
            elements.setdefault('text_color', (0, 0, 0))
            elements.setdefault('font_primary', 'Arial')
            elements.setdefault('logo_position', 'bottom_right')
            
            # Store custom template
            self.template_system['custom_templates'][template_id] = template_config
            
            self.logger.info(f"Created custom template: {template_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create custom template: {e}")
            return False
    
    def update_brand_elements(self, brand_updates: Dict[str, Any]) -> bool:
        """Update global brand elements for all templates"""
        try:
            # Update colors
            if 'colors' in brand_updates:
                self.template_system['brand_elements']['colors'].update(brand_updates['colors'])
            
            # Update fonts
            if 'fonts' in brand_updates:
                self.template_system['brand_elements']['fonts'].update(brand_updates['fonts'])
            
            # Update logo positions
            if 'logo_positions' in brand_updates:
                self.template_system['brand_elements']['logo_positions'].update(brand_updates['logo_positions'])
            
            self.logger.info("Brand elements updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update brand elements: {e}")
            return False
    
    async def create_branded_video_variants(self, enhanced_image_path: str, product_data: Dict[str, Any],
                                          platforms: List[str] = None, template_name: str = 'professional',
                                          **kwargs) -> Dict[str, VideoGenerationResult]:
        """Create video variants with consistent branding using templates"""
        if platforms is None:
            platforms = ['x', 'tiktok', 'instagram', 'linkedin']
        
        self.logger.info(f"Creating branded videos with template '{template_name}' for {len(platforms)} platforms")
        
        results = {}
        
        for platform in platforms:
            try:
                # Create base request
                request = VideoGenerationRequest(
                    input_type='image',
                    input_data=enhanced_image_path,
                    platform=platform,
                    product_data=product_data,
                    **kwargs
                )
                
                # Apply template
                templated_request = self.create_video_with_template(template_name, request)
                
                # Generate video
                result = await self.create_video_with_veo3(templated_request)
                results[platform] = result
                
                if result.success:
                    self.logger.info(f"✓ Created branded video for {platform}")
                else:
                    self.logger.warning(f"⚠ Branded video creation failed for {platform}: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"✗ Exception creating branded video for {platform}: {e}")
                results[platform] = VideoGenerationResult(
                    success=False,
                    video_path=None,
                    metadata={'template_error': str(e)},
                    cost=Decimal('0.00'),
                    processing_time=0.0,
                    platform=platform,
                    error_message=f"Template application failed: {e}"
                )
        
        return results

    def generate_video_prompt(self, platform: str, product_data: Dict[str, Any], 
                            input_type: str = 'image', image_description: str = "", 
                            custom_style: str = None) -> str:
        """Generate advanced platform-specific video creation prompt for Veo 3"""
        platform_spec = self.platform_specs.get(platform, self.platform_specs['x'])
        
        product_name = product_data.get('name', 'product')
        category = product_data.get('category', 'item')
        features = product_data.get('features', 'various features')
        target_audience = product_data.get('target_audience', 'general audience')
        brand_voice = product_data.get('brand_voice', 'professional')
        
        style = custom_style or platform_spec['style']
        duration = platform_spec['optimal_duration']
        trending_features = platform_spec.get('trending_features', [])
        
        # Base prompt with enhanced details
        base_prompt = f"""Create a {duration}-second professional product showcase video for {product_name}, 
        a {category} targeting {target_audience}. Brand voice: {brand_voice}."""
        
        # Platform-specific enhanced prompts
        platform_prompts = {
            'x': f"""{base_prompt}
            Style: {style}. Create a professional, informative video optimized for X/Twitter.
            - Show {product_name} with clean, professional presentation
            - Use smooth camera movements and clear product demonstration
            - Highlight key features: {features}
            - Include subtle professional transitions
            - Ensure subtitle-safe composition (avoid text in top/bottom 80px)
            - Focus on clear value proposition and credibility
            - Use 16:9 aspect ratio with clean, uncluttered background
            - Duration: exactly {duration} seconds
            - Include trending elements: {', '.join(trending_features)}""",
            
            'tiktok': f"""{base_prompt}
            Style: {style}. Create a dynamic, engaging TikTok video that follows current trends.
            - Make it immediately attention-grabbing (hook within first 3 seconds)
            - Use quick cuts, zoom effects, and trending visual styles
            - Show {product_name} in action with energetic, youthful appeal
            - Include popular TikTok effects: {platform_spec.get('popular_effects', [])}
            - Optimize for vertical 9:16 viewing
            - Keep text/branding in safe zones (avoid top 100px, bottom 200px)
            - Use trending sounds and effects where appropriate
            - Duration: exactly {duration} seconds
            - Make it shareable and trend-worthy
            - Include elements: {', '.join(trending_features)}""",
            
            'instagram': f"""{base_prompt}
            Style: {style}. Create an aesthetically stunning Instagram Reels video.
            - Focus on visual storytelling and lifestyle integration
            - Show {product_name} in beautiful, aspirational context
            - Use smooth, cinematic movements and premium lighting
            - Maintain Instagram's aesthetic standards (clean, polished)
            - Optimize for 9:16 vertical format with story-safe zones
            - Include lifestyle context and emotional appeal
            - Duration: exactly {duration} seconds
            - Focus on shareability and visual appeal
            - Include trending elements: {', '.join(trending_features)}""",
            
            'linkedin': f"""{base_prompt}
            Style: {style}. Create a professional, business-focused LinkedIn video.
            - Position {product_name} as a business solution or professional tool
            - Use authoritative, business-appropriate presentation
            - Include clear value proposition for professional audience
            - Show real-world business applications and ROI
            - Use 16:9 horizontal format suitable for desktop viewing
            - Ensure all content is caption-accessible
            - Duration: {duration} seconds (can be longer for detailed explanation)
            - Include professional elements: {', '.join(trending_features)}
            - Focus on thought leadership and industry expertise""",
            
            'pinterest': f"""{base_prompt}
            Style: {style}. Create an inspiring Pinterest Idea Pin video.
            - Show {product_name} in aspirational, lifestyle context
            - Use bright, vibrant colors and clear visual hierarchy
            - Create save-worthy content with DIY or tutorial elements
            - Include step-by-step or before/after sequences
            - Use 9:16 vertical format optimized for mobile Pinterest
            - Add clear text overlays for key information
            - Duration: exactly {duration} seconds (short and impactful)
            - Include Pinterest elements: {', '.join(trending_features)}
            - Focus on inspiration and practical value""",
            
            'youtube_shorts': f"""{base_prompt}
            Style: {style}. Create an engaging YouTube Shorts video.
            - Hook viewers within first 3 seconds with surprising fact or question
            - Show {product_name} with educational or entertaining approach
            - Use quick-paced editing with clear information delivery
            - Include how-to elements or product demonstrations
            - Use 9:16 vertical format for mobile consumption
            - Ensure content works without sound (visual storytelling)
            - Duration: exactly {duration} seconds
            - Include YouTube elements: {', '.join(trending_features)}
            - Focus on subscriber growth and engagement"""
        }
        
        prompt = platform_prompts.get(platform, platform_prompts['x'])
        
        # Add input-specific context
        if input_type == 'image' and image_description:
            prompt += f"\n\nBase the video transformation on this product image: {image_description}"
        elif input_type == 'text':
            prompt += "\n\nGenerate this video from text description only, creating compelling visual narrative."
            
        # Add quality specifications
        prompt += f"\n\nTechnical specifications:\n- Resolution: {platform_spec['dimensions'][0]}x{platform_spec['dimensions'][1]}\n- Frame rate: {platform_spec['fps']} FPS\n- High definition quality with crisp details\n- Optimized bitrate for platform requirements"
        
        return prompt

    async def create_video_with_veo3(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        """Create video using Google Veo 3 API with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Validate request parameters first
            if not self._validate_video_request(request):
                return VideoGenerationResult(
                    success=False,
                    video_path=None,
                    metadata={'validation_failed': True},
                    cost=Decimal('0.00'),
                    processing_time=time.time() - start_time,
                    platform=request.platform,
                    error_message="Request validation failed"
                )
                
            platform_spec = self.platform_specs.get(request.platform, self.platform_specs['x'])
            prompt = self.generate_video_prompt(
                request.platform, 
                request.product_data, 
                request.input_type,
                getattr(request, 'image_description', '')
            )
            
            self.logger.info(f"Creating {request.input_type}-to-video for {request.platform}")
            
            # Rate limiting check
            await self._check_rate_limits()
            
            # Cost limit check
            estimated_cost = self._calculate_cost(request)
            cost_ok, cost_message = self._check_cost_limits(estimated_cost, request)
            if not cost_ok:
                return VideoGenerationResult(
                    success=False,
                    video_path=None,
                    metadata={'cost_limit_exceeded': True},
                    cost=Decimal('0.00'),
                    processing_time=time.time() - start_time,
                    platform=request.platform,
                    error_message=f"Cost limit exceeded: {cost_message}"
                )
            elif cost_message != 'OK':
                self.logger.warning(f"Cost warning: {cost_message}")
            
            # Content moderation
            if request.use_moderation:
                moderation_result = await self._moderate_content(request)
                if not moderation_result['approved']:
                    return VideoGenerationResult(
                        success=False,
                        video_path=None,
                        metadata={'moderation_failed': True, 'moderation_result': moderation_result},
                        cost=Decimal('0.00'),
                        processing_time=time.time() - start_time,
                        platform=request.platform,
                        error_message="Content rejected by moderation",
                        moderation_result=moderation_result
                    )
            
            # Try Veo 3 API first
            if self.veo_service:
                try:
                    video_path = await self._generate_with_veo3_api(request, prompt, platform_spec)
                    if video_path:
                        cost = self._calculate_cost(request)
                        self._update_cost_tracker(cost, request, True)
                        
                        result = VideoGenerationResult(
                            success=True,
                            video_path=video_path,
                            metadata=self._get_video_metadata(video_path, request),
                            cost=cost,
                            processing_time=time.time() - start_time,
                            platform=request.platform,
                            thumbnail_path=self._generate_thumbnail(video_path)
                        )
                        
                        # Log to database
                        await self._log_video_generation(result, request)
                        
                        # Set up engagement tracking
                        await self._update_video_engagement_tracking(
                            video_path, request.platform, result.metadata
                        )
                        
                        return result
                except Exception as e:
                    self.logger.warning(f"Veo 3 API failed, using fallback: {e}")
            
            # Fallback to enhanced moviepy implementation
            video_path = await self._create_enhanced_fallback_video(request, platform_spec)
            
            return VideoGenerationResult(
                success=True,
                video_path=video_path,
                metadata=self._get_video_metadata(video_path, request),
                cost=Decimal('0.00'),  # No cost for fallback
                processing_time=time.time() - start_time,
                platform=request.platform,
                thumbnail_path=self._generate_thumbnail(video_path)
            )
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            return VideoGenerationResult(
                success=False,
                video_path=None,
                metadata={},
                cost=Decimal('0.00'),
                processing_time=time.time() - start_time,
                platform=request.platform,
                error_message=str(e)
            )
    
    async def _generate_with_veo3_api(self, request: VideoGenerationRequest, 
                                     prompt: str, platform_spec: Dict[str, Any]) -> Optional[str]:
        """Generate video using actual Veo 3 API"""
        try:
            # Prepare input data
            input_data = {}
            if request.input_type == 'image':
                # Upload image to Google Cloud Storage
                image_uri = await self._upload_to_gcs(request.input_data)
                input_data = {'input_image': image_uri}
            elif request.input_type == 'video':
                video_uri = await self._upload_to_gcs(request.input_data)
                input_data = {'input_video': video_uri}
            
            # Veo 3 API request structure
            veo_request = {
                'instances': [{
                    'prompt': prompt,
                    **input_data,
                    'parameters': {
                        'duration': min(request.duration, platform_spec['max_duration']),
                        'resolution': f"{platform_spec['dimensions'][0]}x{platform_spec['dimensions'][1]}",
                        'fps': platform_spec['fps'],
                        'quality': request.quality,
                        'style': request.style,
                        'aspect_ratio': platform_spec['aspect_ratio']
                    }
                }]
            }
            
            # Make API call
            location = f"projects/{self.google_project_id}/locations/us-central1"
            model_name = f"{location}/publishers/google/models/veo-3"
            
            response = self.veo_service.projects().locations().publishers().models().predict(
                endpoint=model_name,
                body=veo_request
            ).execute()
            
            # Handle response
            if 'predictions' in response and response['predictions']:
                video_data = response['predictions'][0]
                
                # Download generated video
                if 'video_uri' in video_data:
                    video_path = await self._download_from_gcs(video_data['video_uri'])
                    return video_path
                elif 'video_bytes' in video_data:
                    video_bytes = base64.b64decode(video_data['video_bytes'])
                    return self._save_video_bytes(video_bytes, request.platform)
            
            return None
            
        except HttpError as e:
            self.logger.error(f"Veo 3 API HTTP error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Veo 3 API error: {e}")
            return None

    async def _create_enhanced_fallback_video(self, request: VideoGenerationRequest, 
                                            platform_spec: Dict[str, Any]) -> str:
        """Create enhanced video using advanced moviepy techniques when Veo 3 is unavailable"""
        try:
            duration = min(request.duration, platform_spec['optimal_duration'])
            
            if request.input_type == 'image':
                return await self._create_image_to_video_fallback(request, platform_spec, duration)
            elif request.input_type == 'text':
                return await self._create_text_to_video_fallback(request, platform_spec, duration)
            elif request.input_type == 'video':
                return await self._enhance_existing_video(request, platform_spec, duration)
            else:
                raise ValueError(f"Unsupported input type: {request.input_type}")
                
        except Exception as e:
            self.logger.error(f"Enhanced fallback video creation failed: {e}")
            raise
    
    async def _create_image_to_video_fallback(self, request: VideoGenerationRequest,
                                            platform_spec: Dict[str, Any], duration: int) -> str:
        """Create professional image-to-video with advanced effects"""
        
        # Load and enhance image
        image_clip = ImageClip(request.input_data, duration=duration)
        image_clip = image_clip.resize(platform_spec['dimensions'])
        
        # Apply platform-specific effects
        effects_clips = []
        
        if request.platform == 'tiktok':
            # TikTok-style effects: zoom, rotation, trending transitions
            image_clip = self._apply_tiktok_effects(image_clip, duration)
            
        elif request.platform == 'instagram':
            # Instagram-style effects: smooth pans, aesthetic filters
            image_clip = self._apply_instagram_effects(image_clip, duration)
            
        elif request.platform == 'x':
            # X-style effects: professional zoom, clean presentation
            image_clip = self._apply_x_effects(image_clip, duration)
            
        elif request.platform == 'linkedin':
            # LinkedIn-style effects: professional presentation with data overlay
            image_clip = self._apply_linkedin_effects(image_clip, duration, request.product_data)
        
        # Add text overlays if requested
        if request.add_captions:
            text_clip = self._create_platform_text_overlay(request, platform_spec, duration)
            effects_clips.append(text_clip)
        
        # Add advanced transitions and effects
        if request.platform in ['tiktok', 'instagram']:
            transition_effects = self._apply_advanced_transitions(image_clip, duration, request.platform)
            if transition_effects:
                effects_clips.extend(transition_effects)
        
        # Add background music for platforms that support it
        final_clips = [image_clip] + effects_clips
        if request.enhance_audio and request.platform in ['tiktok', 'instagram', 'youtube_shorts']:
            audio_clip = self._add_trending_audio(duration, request.platform)
            if audio_clip:
                final_clips.append(audio_clip)
        
        # Composite final video
        if len(final_clips) > 1:
            final_video = CompositeVideoClip(final_clips)
        else:
            final_video = image_clip
        
        # Generate output path
        output_path = self._generate_output_path(request)
        
        # Write video with platform-optimized settings
        final_video.write_videofile(
            output_path,
            fps=platform_spec['fps'],
            codec='libx264',
            bitrate=platform_spec['bitrate'],
            audio_codec='aac' if request.enhance_audio else None,
            verbose=False,
            logger=None,
            preset='fast'
        )
        
        # Clean up
        for clip in final_clips:
            clip.close()
        
        return output_path
    
    def _apply_tiktok_effects(self, clip: VideoFileClip, duration: int) -> VideoFileClip:
        """Apply TikTok-trending video effects"""
        # Gradual zoom with bounce effect
        def zoom_func(t):
            base_zoom = 1.0
            zoom_amount = 0.15
            return base_zoom + zoom_amount * (1 - abs(t - duration/2) / (duration/2))
        
        clip = clip.resize(zoom_func)
        
        # Add slight rotation for dynamic feel
        def rotate_func(t):
            return 2 * np.sin(t * 0.5)  # Subtle oscillation
        
        clip = clip.rotate(rotate_func)
        
        # Speed ramping effect
        if duration > 10:
            # Fast start, slow middle, fast end
            def speed_func(t):
                if t < 3:
                    return t * 1.2
                elif t < duration - 3:
                    return 3 * 1.2 + (t - 3) * 0.8
                else:
                    return 3 * 1.2 + (duration - 6) * 0.8 + (t - duration + 3) * 1.2
            
            clip = clip.time_transform(speed_func)
        
        return clip
    
    def _apply_instagram_effects(self, clip: VideoFileClip, duration: int) -> VideoFileClip:
        """Apply Instagram Reels-style effects"""
        # Smooth pan and zoom
        def smooth_zoom(t):
            return 1.0 + 0.1 * np.sin(t * np.pi / duration)
        
        clip = clip.resize(smooth_zoom)
        
        # Subtle pan movement
        def smooth_pan(t):
            x_offset = 20 * np.sin(t * np.pi / duration)
            return ('center', 'center')
        
        # Add aesthetic filter (warm tone)
        clip = clip.fx(vfx.colorx, 1.1).fx(vfx.lum_contrast, 0, 10, 128)
        
        return clip
    
    def _apply_x_effects(self, clip: VideoFileClip, duration: int) -> VideoFileClip:
        """Apply X/Twitter professional effects"""
        # Professional zoom - subtle and clean
        def professional_zoom(t):
            return 1.0 + 0.05 * (t / duration)
        
        clip = clip.resize(professional_zoom)
        
        # Enhance contrast and sharpness for clarity
        clip = clip.fx(vfx.lum_contrast, 0, 15, 128)
        
        return clip
    
    def _apply_linkedin_effects(self, clip: VideoFileClip, duration: int, 
                              product_data: Dict[str, Any]) -> VideoFileClip:
        """Apply LinkedIn business-focused effects"""
        # Professional presentation - minimal movement
        def business_zoom(t):
            return 1.0 + 0.02 * (t / duration)
        
        clip = clip.resize(business_zoom)
        
        # Business-appropriate color grading
        clip = clip.fx(vfx.lum_contrast, 0, 10, 128)
        
        return clip
    
    def _apply_advanced_transitions(self, clip: VideoFileClip, duration: int, platform: str) -> List[VideoFileClip]:
        """Apply advanced transitions and effects for enhanced visual appeal"""
        transition_effects = []
        
        try:
            if platform == 'tiktok':
                # TikTok-style flash transitions
                flash_effect = self._create_flash_transition(clip, duration)
                if flash_effect:
                    transition_effects.append(flash_effect)
                
                # Add trendy border effects
                border_effect = self._create_dynamic_border(clip, duration)
                if border_effect:
                    transition_effects.append(border_effect)
                    
            elif platform == 'instagram':
                # Instagram-style smooth fades
                fade_effect = self._create_smooth_fade_overlay(clip, duration)
                if fade_effect:
                    transition_effects.append(fade_effect)
                
                # Add aesthetic particle effects
                particle_effect = self._create_particle_overlay(clip, duration)
                if particle_effect:
                    transition_effects.append(particle_effect)
                    
        except Exception as e:
            self.logger.error(f"Failed to apply advanced transitions: {e}")
            
        return transition_effects
    
    def _create_flash_transition(self, clip: VideoFileClip, duration: int) -> Optional[VideoFileClip]:
        """Create TikTok-style flash transition effect"""
        try:
            # Create flash overlay
            flash_color = (255, 255, 255)  # White flash
            flash_duration = 0.1
            flash_times = [duration * 0.25, duration * 0.5, duration * 0.75]
            
            flash_clips = []
            for flash_time in flash_times:
                if flash_time + flash_duration < duration:
                    flash_clip = ColorClip(
                        size=clip.size,
                        color=flash_color,
                        duration=flash_duration
                    ).set_start(flash_time).set_opacity(0.3)
                    flash_clips.append(flash_clip)
            
            if flash_clips:
                return CompositeVideoClip(flash_clips)
            return None
            
        except Exception as e:
            self.logger.error(f"Flash transition creation failed: {e}")
            return None
    
    def _create_dynamic_border(self, clip: VideoFileClip, duration: int) -> Optional[VideoFileClip]:
        """Create dynamic border effect"""
        try:
            # Create animated border
            border_width = 10
            border_color = (255, 100, 150)  # Trendy pink/purple
            
            def make_border(t):
                # Animate border thickness
                thickness = int(border_width * (1 + 0.3 * np.sin(t * 4)))
                return thickness
            
            # Create border mask (simplified version)
            border_clip = ColorClip(
                size=clip.size,
                color=border_color,
                duration=duration
            ).set_opacity(0.2)
            
            return border_clip
            
        except Exception as e:
            self.logger.error(f"Dynamic border creation failed: {e}")
            return None
    
    def _create_smooth_fade_overlay(self, clip: VideoFileClip, duration: int) -> Optional[VideoFileClip]:
        """Create Instagram-style smooth fade overlay"""
        try:
            # Create gradient overlay
            overlay_color = (255, 255, 255)  # White overlay
            
            # Create fade effect
            fade_clip = ColorClip(
                size=clip.size,
                color=overlay_color,
                duration=duration
            ).set_opacity(0.1)
            
            # Apply fade in/out
            fade_clip = fade_clip.fadein(1).fadeout(1)
            
            return fade_clip
            
        except Exception as e:
            self.logger.error(f"Smooth fade overlay creation failed: {e}")
            return None
    
    def _create_particle_overlay(self, clip: VideoFileClip, duration: int) -> Optional[VideoFileClip]:
        """Create aesthetic particle overlay effect"""
        try:
            # Create simple particle effect using small colored circles
            particle_clips = []
            num_particles = 5
            
            for i in range(num_particles):
                # Create small colored circle
                particle_size = (20, 20)
                particle_color = [(255, 200, 100), (255, 150, 200), (150, 255, 200)][i % 3]
                
                particle_clip = ColorClip(
                    size=particle_size,
                    color=particle_color,
                    duration=duration
                ).set_opacity(0.3)
                
                # Random position and movement
                start_x = 50 + (i * 200) % (clip.size[0] - 100)
                start_y = 50 + (i * 150) % (clip.size[1] - 100)
                
                particle_clip = particle_clip.set_position((start_x, start_y))
                particle_clips.append(particle_clip)
            
            if particle_clips:
                return CompositeVideoClip(particle_clips)
            return None
            
        except Exception as e:
            self.logger.error(f"Particle overlay creation failed: {e}")
            return None

    def _create_platform_text_overlay(self, request: VideoGenerationRequest, 
                                     platform_spec: Dict[str, Any], duration: int) -> TextClip:
        """Create sophisticated platform-specific text overlays"""
        product_data = request.product_data
        platform = request.platform
        
        product_name = product_data.get('name', 'Product')
        tagline = product_data.get('tagline', '')
        
        # Platform-specific text styling
        text_configs = {
            'tiktok': {
                'fontsize': 45,
                'color': 'white',
                'font': 'Arial-Bold',
                'stroke_color': 'black',
                'stroke_width': 2,
                'position': ('center', platform_spec['dimensions'][1] - 200),
                'style': 'trendy'
            },
            'instagram': {
                'fontsize': 36,
                'color': 'white',
                'font': 'Arial',
                'stroke_color': 'gray',
                'stroke_width': 1,
                'position': ('center', 100),
                'style': 'elegant'
            },
            'x': {
                'fontsize': 32,
                'color': 'white',
                'font': 'Arial-Bold',
                'position': ('center', 50),
                'style': 'professional'
            },
            'linkedin': {
                'fontsize': 28,
                'color': 'white',
                'font': 'Arial',
                'position': ('center', 80),
                'style': 'business'
            }
        }
        
        config = text_configs.get(platform, text_configs['x'])
        
        # Create animated text
        if config['style'] == 'trendy':
            # TikTok-style animated text
            txt_clip = TextClip(
                product_name.upper(),
                fontsize=config['fontsize'],
                color=config['color'],
                font=config['font'],
                stroke_color=config['stroke_color'],
                stroke_width=config['stroke_width']
            ).set_position(config['position']).set_duration(duration)
            
            # Add bounce animation
            txt_clip = txt_clip.resize(lambda t: 1 + 0.1 * np.sin(t * 4))
            
        elif config['style'] == 'elegant':
            # Instagram-style elegant text
            txt_clip = TextClip(
                product_name,
                fontsize=config['fontsize'],
                color=config['color'],
                font=config['font'],
                stroke_color=config.get('stroke_color'),
                stroke_width=config.get('stroke_width', 0)
            ).set_position(config['position']).set_duration(duration)
            
            # Add fade in/out
            txt_clip = txt_clip.fadeout(1).fadein(1)
            
        else:
            # Professional text for X and LinkedIn
            txt_clip = TextClip(
                product_name,
                fontsize=config['fontsize'],
                color=config['color'],
                font=config['font']
            ).set_position(config['position']).set_duration(duration)
        
        return txt_clip
    
    def _add_trending_audio(self, duration: int, platform: str = 'tiktok') -> Optional[AudioFileClip]:
        """Add trending audio with comprehensive music integration"""
        try:
            # Advanced music integration system
            music_library = self._get_music_library_for_platform(platform)
            
            if not music_library:
                return self._create_ambient_audio(duration)
            
            # Select appropriate audio based on platform and duration
            selected_audio = self._select_optimal_audio(music_library, platform, duration)
            
            if selected_audio:
                return self._process_audio_for_platform(selected_audio, platform, duration)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to add trending audio: {e}")
            return None
    
    def _get_music_library_for_platform(self, platform: str) -> Dict[str, Any]:
        """Get music library options for specific platform"""
        # In production, this would integrate with:
        # - Epidemic Sound API
        # - AudioJungle API
        # - Freesound API
        # - Platform-specific trending audio APIs
        
        music_libraries = {
            'tiktok': {
                'trending_categories': ['pop', 'electronic', 'hip_hop', 'ambient'],
                'popular_durations': [15, 30, 60],
                'energy_levels': ['high', 'medium', 'chill'],
                'trending_tags': ['viral2024', 'product_showcase', 'background_music']
            },
            'instagram': {
                'trending_categories': ['indie', 'pop', 'acoustic', 'electronic'],
                'popular_durations': [30, 60],
                'energy_levels': ['medium', 'chill', 'upbeat'],
                'aesthetic_match': True
            },
            'youtube_shorts': {
                'trending_categories': ['royalty_free', 'background', 'tutorial_music'],
                'popular_durations': [30, 60],
                'energy_levels': ['medium', 'informative']
            },
            'pinterest': {
                'trending_categories': ['ambient', 'acoustic', 'lifestyle'],
                'popular_durations': [6, 15],
                'energy_levels': ['chill', 'inspiring']
            }
        }
        
        return music_libraries.get(platform, {})
    
    def _select_optimal_audio(self, music_library: Dict[str, Any], platform: str, duration: int) -> Optional[str]:
        """Select optimal audio track based on platform requirements"""
        if not music_library:
            return None
        
        # Mock audio selection logic
        # In production, this would query actual music APIs
        preferred_categories = music_library.get('trending_categories', ['ambient'])
        
        # Return mock audio path for testing
        return f"mock_audio_{platform}_{duration}s_{preferred_categories[0]}.mp3"
    
    def _process_audio_for_platform(self, audio_path: str, platform: str, duration: int) -> Optional[AudioFileClip]:
        """Process audio file for platform-specific requirements"""
        try:
            # In production, load actual audio file
            # For now, create a mock audio clip
            return self._create_ambient_audio(duration)
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            return None
    
    def _create_ambient_audio(self, duration: int) -> Optional[AudioFileClip]:
        """Create ambient audio track for video"""
        try:
            # Create a simple tone using numpy for testing
            import numpy as np
            from scipy.io import wavfile
            
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create gentle ambient tone
            frequency = 220  # A3 note
            audio_data = 0.1 * np.sin(2 * np.pi * frequency * t)
            
            # Add some variation
            audio_data += 0.05 * np.sin(2 * np.pi * (frequency * 1.5) * t)
            audio_data = audio_data * np.exp(-t / duration)  # Fade out
            
            # Save to temp file
            temp_audio_path = os.path.join(self.audio_dir, f"ambient_{int(time.time())}.wav")
            
            # Convert to int16 for WAV format
            audio_int = np.int16(audio_data * 32767)
            wavfile.write(temp_audio_path, sample_rate, audio_int)
            
            return AudioFileClip(temp_audio_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create ambient audio: {e}")
            return None

    def optimize_for_platform(self, video_path: str, platform: str) -> str:
        """Apply comprehensive platform-specific optimizations"""
        try:
            platform_spec = self.platform_specs.get(platform, self.platform_specs['x'])
            
            # Load video
            video_clip = VideoFileClip(video_path)
            
            # Platform-specific optimization
            if platform == 'tiktok':
                video_clip = self._optimize_for_tiktok(video_clip, platform_spec)
            elif platform == 'instagram':
                video_clip = self._optimize_for_instagram(video_clip, platform_spec)
            elif platform == 'x':
                video_clip = self._optimize_for_x(video_clip, platform_spec)
            elif platform == 'linkedin':
                video_clip = self._optimize_for_linkedin(video_clip, platform_spec)
            elif platform == 'pinterest':
                video_clip = self._optimize_for_pinterest(video_clip, platform_spec)
            elif platform == 'youtube_shorts':
                video_clip = self._optimize_for_youtube_shorts(video_clip, platform_spec)
            
            # Ensure duration compliance
            if video_clip.duration > platform_spec['max_duration']:
                video_clip = video_clip.subclip(0, platform_spec['max_duration'])
            
            # Quality validation
            if not self._validate_video_quality(video_clip, platform_spec):
                self.logger.warning(f"Video quality below standards for {platform}")
            
            # Generate optimized output
            output_path = video_path.replace('.mp4', '_optimized.mp4')
            
            # Write with platform-optimized encoding
            write_params = {
                'filename': output_path,
                'fps': platform_spec['fps'],
                'codec': 'libx264',
                'bitrate': platform_spec['bitrate'],
                'verbose': False,
                'logger': None,
                'preset': 'fast' if platform == 'tiktok' else 'medium',
                'ffmpeg_params': self._get_platform_ffmpeg_params(platform)
            }
            
            video_clip.write_videofile(**write_params)
            
            # Clean up
            video_clip.close()
            
            # Remove unoptimized version
            if os.path.exists(video_path):
                os.remove(video_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Video optimization failed: {e}")
            return video_path
    
    def _optimize_for_tiktok(self, clip: VideoFileClip, platform_spec: Dict[str, Any]) -> VideoFileClip:
        """TikTok-specific optimization"""
        # Ensure vertical 9:16 aspect ratio
        clip = clip.resize(platform_spec['dimensions'])
        
        # Enhance for mobile viewing
        clip = clip.fx(vfx.lum_contrast, 0, 20, 128)  # Increase contrast
        
        # Ensure safe zones for TikTok UI
        # Add padding to avoid UI overlap
        
        return clip
    
    def _optimize_for_instagram(self, clip: VideoFileClip, platform_spec: Dict[str, Any]) -> VideoFileClip:
        """Instagram Reels optimization"""
        # Ensure vertical format with story-safe zones
        clip = clip.resize(platform_spec['dimensions'])
        
        # Instagram-friendly color grading
        clip = clip.fx(vfx.colorx, 1.05)  # Slight warmth
        
        return clip
    
    def _optimize_for_x(self, clip: VideoFileClip, platform_spec: Dict[str, Any]) -> VideoFileClip:
        """X/Twitter optimization"""
        # Professional horizontal format
        clip = clip.resize(platform_spec['dimensions'])
        
        # Professional color grading
        clip = clip.fx(vfx.lum_contrast, 0, 10, 128)
        
        return clip
    
    def _optimize_for_linkedin(self, clip: VideoFileClip, platform_spec: Dict[str, Any]) -> VideoFileClip:
        """LinkedIn business optimization"""
        # Business-appropriate horizontal format
        clip = clip.resize(platform_spec['dimensions'])
        
        # Professional enhancement
        clip = clip.fx(vfx.lum_contrast, 0, 5, 128)
        
        return clip
    
    def _optimize_for_pinterest(self, clip: VideoFileClip, platform_spec: Dict[str, Any]) -> VideoFileClip:
        """Pinterest Idea Pins optimization"""
        # Ensure vertical format
        clip = clip.resize(platform_spec['dimensions'])
        
        # Pinterest-friendly bright and vibrant colors
        clip = clip.fx(vfx.colorx, 1.1)  # Slight saturation boost
        clip = clip.fx(vfx.lum_contrast, 0, 15, 128)  # Increase contrast
        
        return clip
    
    def _optimize_for_youtube_shorts(self, clip: VideoFileClip, platform_spec: Dict[str, Any]) -> VideoFileClip:
        """YouTube Shorts optimization"""
        # Vertical format
        clip = clip.resize(platform_spec['dimensions'])
        
        # YouTube-friendly enhancement
        clip = clip.fx(vfx.lum_contrast, 0, 12, 128)
        
        return clip
    
    def _get_platform_ffmpeg_params(self, platform: str) -> List[str]:
        """Get platform-specific FFmpeg encoding parameters"""
        params = []
        
        if platform == 'tiktok':
            params.extend([
                '-profile:v', 'main',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p'
            ])
        elif platform == 'instagram':
            params.extend([
                '-profile:v', 'baseline',
                '-level', '3.1',
                '-pix_fmt', 'yuv420p'
            ])
        elif platform in ['x', 'linkedin']:
            params.extend([
                '-profile:v', 'high',
                '-level', '4.2',
                '-pix_fmt', 'yuv420p'
            ])
        elif platform == 'pinterest':
            params.extend([
                '-profile:v', 'main',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'  # Fast start for Pinterest
            ])
        elif platform == 'youtube_shorts':
            params.extend([
                '-profile:v', 'high',
                '-level', '4.2',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ])
        
        return params

    async def create_product_video(self, enhanced_image_path: str, platform: str, 
                                 product_data: Dict[str, Any], **kwargs) -> Optional[VideoGenerationResult]:
        """
        Main method: Create comprehensive product video for specific platform
        Returns VideoGenerationResult with detailed metadata
        """
        try:
            self.logger.info(f"Creating product video for {platform}")
            
            # Create structured request
            request = VideoGenerationRequest(
                input_type='image',
                input_data=enhanced_image_path,
                platform=platform,
                product_data=product_data,
                style=kwargs.get('style', 'professional'),
                duration=kwargs.get('duration', 30),
                quality=kwargs.get('quality', 'hd'),
                priority=kwargs.get('priority', 'normal'),
                use_moderation=kwargs.get('use_moderation', True),
                add_captions=kwargs.get('add_captions', platform == 'linkedin'),
                enhance_audio=kwargs.get('enhance_audio', platform == 'tiktok')
            )
            
            # Generate video with comprehensive pipeline
            result = await self.create_video_with_veo3(request)
            
            if result.success and result.video_path:
                # Apply final optimizations
                optimized_path = self.optimize_for_platform(result.video_path, platform)
                result.video_path = optimized_path
                
                # Update database if available
                if self.db_manager and result.success:
                    await self._log_video_generation(result, request)
                
                self.logger.info(f"Product video created successfully: {result.video_path}")
            else:
                self.logger.error(f"Video creation failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Product video creation failed: {e}")
            return VideoGenerationResult(
                success=False,
                video_path=None,
                metadata={},
                cost=Decimal('0.00'),
                processing_time=0.0,
                platform=platform,
                error_message=str(e)
            )

    async def create_video_variants(self, enhanced_image_path: str, product_data: Dict[str, Any], 
                                  platforms: List[str] = None, **kwargs) -> Dict[str, VideoGenerationResult]:
        """Create video variants for multiple platforms with enhanced batch processing"""
        if platforms is None:
            platforms = ['x', 'tiktok', 'instagram', 'linkedin']
        
        self.logger.info(f"Starting batch video generation for {len(platforms)} platforms: {platforms}")
        start_time = time.time()
        
        # Sort platforms by priority
        platforms_sorted = sorted(
            platforms, 
            key=lambda p: self.batch_settings['prioritize_by_platform'].get(p, 999)
        )
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.batch_settings['max_concurrent_videos'])
        
        async def process_platform_with_retry(platform: str, retry_count: int = 0):
            """Process single platform with retry logic"""
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        self.create_product_video(enhanced_image_path, platform, product_data, **kwargs),
                        timeout=self.batch_settings['batch_timeout_seconds']
                    )
                    
                    if result.success:
                        self.logger.info(f"✓ {platform} video created successfully: {result.video_path}")
                        return result
                    elif (retry_count < self.batch_settings['max_retries_per_request'] and 
                          self.batch_settings['retry_failed_requests']):
                        self.logger.warning(f"⚠ {platform} failed, retrying ({retry_count + 1}/{self.batch_settings['max_retries_per_request']})")
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                        return await process_platform_with_retry(platform, retry_count + 1)
                    else:
                        self.logger.error(f"✗ {platform} video generation failed: {result.error_message}")
                        return result
                        
                except asyncio.TimeoutError:
                    error_msg = f"Video generation timed out after {self.batch_settings['batch_timeout_seconds']}s"
                    self.logger.error(f"✗ {platform} {error_msg}")
                    return VideoGenerationResult(
                        success=False,
                        video_path=None,
                        metadata={'timeout': True},
                        cost=Decimal('0.00'),
                        processing_time=self.batch_settings['batch_timeout_seconds'],
                        platform=platform,
                        error_message=error_msg
                    )
                except Exception as e:
                    self.logger.error(f"✗ {platform} unexpected error: {e}")
                    return VideoGenerationResult(
                        success=False,
                        video_path=None,
                        metadata={'exception': str(e)},
                        cost=Decimal('0.00'),
                        processing_time=0.0,
                        platform=platform,
                        error_message=str(e)
                    )
        
        # Execute batch processing with concurrency control
        tasks = [process_platform_with_retry(platform) for platform in platforms_sorted]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        variants = {}
        for i, result in enumerate(results_list):
            platform = platforms_sorted[i]
            if isinstance(result, Exception):
                variants[platform] = VideoGenerationResult(
                    success=False,
                    video_path=None,
                    metadata={'gather_exception': str(result)},
                    cost=Decimal('0.00'),
                    processing_time=0.0,
                    platform=platform,
                    error_message=f"Batch processing exception: {result}"
                )
            else:
                variants[platform] = result
        
        # Calculate comprehensive batch statistics
        total_time = time.time() - start_time
        successful_variants = sum(1 for result in variants.values() if result.success)
        failed_variants = len(variants) - successful_variants
        total_cost = sum(result.cost for result in variants.values())
        avg_processing_time = sum(result.processing_time for result in variants.values()) / len(variants)
        
        # Log comprehensive summary
        self.logger.info(
            f"📊 Batch Processing Summary:\n"
            f"   • Total time: {total_time:.2f}s\n"
            f"   • Successful: {successful_variants}/{len(platforms)} ({successful_variants/len(platforms)*100:.1f}%)\n"
            f"   • Failed: {failed_variants}\n"
            f"   • Total cost: ${total_cost}\n"
            f"   • Avg processing time: {avg_processing_time:.2f}s per video"
        )
        
        return variants
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics and performance metrics"""
        try:
            cost_summary = self.get_cost_summary()
            queue_status = self.get_queue_status()
            
            # Calculate performance metrics
            analytics = {
                'performance_metrics': self._calculate_performance_metrics(),
                'cost_analysis': self._get_detailed_cost_analysis(cost_summary),
                'platform_analytics': self._get_platform_performance_analytics(),
                'quality_metrics': self._get_quality_performance_metrics(),
                'template_usage': self._get_template_usage_analytics(),
                'system_health': self._get_system_health_metrics(),
                'processing_efficiency': self._get_processing_efficiency_metrics(),
                'recommendation_engine': self._get_optimization_recommendations()
            }
            
            # Add metadata
            analytics['metadata'] = {
                'generated_at': datetime.utcnow().isoformat(),
                'system_version': '2.0.0',
                'features_enabled': self._get_enabled_features(),
                'uptime_info': self._get_system_uptime()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Analytics generation failed: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.utcnow().isoformat(),
                'status': 'analytics_unavailable'
            }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        total_requests = self._cost_tracker['requests_count']
        successful_requests = self._cost_tracker['successful_requests']
        
        # Calculate success rates and trends
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Platform performance analysis
        platform_performance = {}
        for platform, cost in self._cost_tracker['platform_costs'].items():
            if cost > 0:  # Only include platforms that have been used
                platform_performance[platform] = {
                    'total_cost': float(cost),
                    'estimated_requests': max(1, int(cost / 0.25)),  # Rough estimation
                    'avg_cost_per_request': float(cost) / max(1, int(cost / 0.25))
                }
        
        return {
            'overall_success_rate': success_rate,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': total_requests - successful_requests,
            'platform_performance': platform_performance,
            'quality_distribution': {
                'hd_usage': float(self._cost_tracker['cost_by_quality']['hd']),
                'standard_usage': float(self._cost_tracker['cost_by_quality']['standard'])
            },
            'request_type_distribution': {
                'text_to_video': float(self._cost_tracker['cost_by_type']['text_to_video']),
                'image_to_video': float(self._cost_tracker['cost_by_type']['image_to_video']),
                'video_edit': float(self._cost_tracker['cost_by_type']['video_edit'])
            }
        }
    
    def _get_detailed_cost_analysis(self, cost_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced cost analysis with predictions and optimization suggestions"""
        daily_cost = cost_summary['costs']['daily']
        monthly_cost = cost_summary['costs']['monthly']
        
        # Predict monthly cost based on daily usage
        days_in_month = 30
        current_day_of_month = datetime.now().day
        projected_monthly_cost = (daily_cost * days_in_month) if daily_cost > 0 else monthly_cost
        
        # Cost efficiency analysis
        avg_cost_per_video = cost_summary['analytics']['avg_cost_per_video']
        cost_efficiency_score = min(100, max(0, (1.0 - avg_cost_per_video) * 100)) if avg_cost_per_video > 0 else 100
        
        return {
            'cost_summary': cost_summary,
            'projections': {
                'projected_monthly_cost': projected_monthly_cost,
                'days_remaining_in_month': days_in_month - current_day_of_month,
                'budget_burn_rate': daily_cost,
                'estimated_videos_remaining_in_budget': self._calculate_remaining_budget_videos()
            },
            'efficiency_metrics': {
                'cost_efficiency_score': cost_efficiency_score,
                'platform_cost_ranking': self._rank_platforms_by_cost_efficiency(),
                'quality_cost_ratio': self._calculate_quality_cost_ratio()
            },
            'optimization_opportunities': self._identify_cost_optimization_opportunities()
        }
    
    def _get_platform_performance_analytics(self) -> Dict[str, Any]:
        """Analyze performance across different platforms"""
        platform_analytics = {}
        
        for platform, specs in self.platform_specs.items():
            platform_cost = self._cost_tracker['platform_costs'].get(platform, Decimal('0.00'))
            estimated_videos = max(1, int(float(platform_cost) / 0.25)) if platform_cost > 0 else 0
            
            platform_analytics[platform] = {
                'specifications': {
                    'dimensions': specs['dimensions'],
                    'max_duration': specs['max_duration'],
                    'optimal_duration': specs['optimal_duration'],
                    'aspect_ratio': specs['aspect_ratio']
                },
                'usage_stats': {
                    'total_cost': float(platform_cost),
                    'estimated_videos_created': estimated_videos,
                    'avg_cost_per_video': float(platform_cost) / max(1, estimated_videos)
                },
                'performance_score': self._calculate_platform_performance_score(platform),
                'trending_features': specs.get('trending_features', []),
                'optimization_level': self._assess_platform_optimization(platform)
            }
        
        return platform_analytics
    
    def _get_quality_performance_metrics(self) -> Dict[str, Any]:
        """Analyze quality metrics and performance"""
        hd_cost = float(self._cost_tracker['cost_by_quality']['hd'])
        standard_cost = float(self._cost_tracker['cost_by_quality']['standard'])
        total_cost = hd_cost + standard_cost
        
        if total_cost == 0:
            return {
                'usage_distribution': {'hd': 0, 'standard': 0},
                'recommendations': ['No quality data available yet']
            }
        
        return {
            'usage_distribution': {
                'hd': (hd_cost / total_cost) * 100,
                'standard': (standard_cost / total_cost) * 100
            },
            'cost_comparison': {
                'hd_total_cost': hd_cost,
                'standard_total_cost': standard_cost,
                'cost_difference': hd_cost - standard_cost
            },
            'recommendations': self._get_quality_recommendations(hd_cost, standard_cost)
        }
    
    def _get_template_usage_analytics(self) -> Dict[str, Any]:
        """Analyze template system usage and performance"""
        available_templates = self.get_available_templates()
        
        return {
            'available_templates': {
                'built_in_count': len([t for t in available_templates if t['type'] == 'built-in']),
                'custom_count': len([t for t in available_templates if t['type'] == 'custom']),
                'total_templates': len(available_templates)
            },
            'template_catalog': available_templates,
            'brand_elements': self.template_system['brand_elements'],
            'template_recommendations': self._get_template_recommendations()
        }
    
    def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health and status metrics"""
        queue_status = self.get_queue_status()
        
        return {
            'api_status': {
                'veo3_service': 'connected' if self.veo_service and not self.using_mock_services else 'mock',
                'google_storage': 'connected' if self.storage_client and not self.using_mock_services else 'mock',
                'database': 'connected' if self.db_manager else 'unavailable'
            },
            'queue_health': {
                'total_queued': queue_status['total_queued'],
                'processing_capacity': queue_status['concurrent_limit'],
                'retry_system': 'enabled' if queue_status['retry_enabled'] else 'disabled',
                'queue_status': 'healthy' if queue_status['total_queued'] < 100 else 'busy'
            },
            'resource_usage': {
                'temp_files': len(os.listdir(self.temp_dir)) if os.path.exists(self.temp_dir) else 0,
                'cache_files': len(os.listdir(self.cache_dir)) if os.path.exists(self.cache_dir) else 0,
                'executor_threads': self.executor._max_workers
            },
            'error_rates': self._calculate_error_rates()
        }
    
    def _get_processing_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate processing efficiency metrics"""
        return {
            'batch_processing': {
                'max_concurrent': self.batch_settings['max_concurrent_videos'],
                'retry_enabled': self.batch_settings['retry_failed_requests'],
                'timeout_seconds': self.batch_settings['batch_timeout_seconds'],
                'platform_prioritization': self.batch_settings['prioritize_by_platform']
            },
            'rate_limiting': {
                'requests_per_minute_limit': 10,  # Based on our rate limiting
                'current_request_count': len(self._request_timestamps)
            },
            'content_moderation': {
                'enabled': self.moderation_enabled,
                'quality_thresholds': self.quality_thresholds
            }
        }
    
    def _get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate AI-driven optimization recommendations"""
        recommendations = []
        
        # Cost optimization recommendations
        total_cost = float(self._cost_tracker['total_cost'])
        if total_cost > 50:
            recommendations.append({
                'type': 'cost_optimization',
                'priority': 'high',
                'title': 'Consider Quality Optimization',
                'description': 'High API usage detected. Consider using standard quality for some platforms to reduce costs.',
                'potential_savings': '20-30%',
                'action': 'Use standard quality for platforms where HD is not critical'
            })
        
        # Platform optimization recommendations
        platform_usage = self._cost_tracker['platform_costs']
        most_used_platform = max(platform_usage.items(), key=lambda x: x[1]) if platform_usage else None
        
        if most_used_platform and float(most_used_platform[1]) > 20:
            recommendations.append({
                'type': 'platform_optimization',
                'priority': 'medium',
                'title': f'Optimize {most_used_platform[0].title()} Content',
                'description': f'{most_used_platform[0]} is your most used platform. Consider creating platform-specific templates.',
                'potential_improvement': '15-25% better engagement',
                'action': f'Create custom template for {most_used_platform[0]}'
            })
        
        # Template recommendations
        if len(self.template_system['custom_templates']) == 0:
            recommendations.append({
                'type': 'branding',
                'priority': 'medium',
                'title': 'Create Custom Templates',
                'description': 'No custom templates detected. Custom branding can improve brand consistency.',
                'potential_improvement': 'Better brand recognition',
                'action': 'Create at least one custom template for your brand'
            })
        
        # System health recommendations
        if hasattr(self, '_request_timestamps') and len(self._request_timestamps) > 8:
            recommendations.append({
                'type': 'performance',
                'priority': 'low',
                'title': 'Monitor Rate Limits',
                'description': 'Approaching rate limits. Consider spreading requests over time.',
                'potential_improvement': 'Avoid API throttling',
                'action': 'Implement request scheduling'
            })
        
        return recommendations
    
    def _calculate_remaining_budget_videos(self) -> Dict[str, int]:
        """Calculate how many videos can be created with remaining budget"""
        daily_remaining = float(self.cost_limits['daily_limit'] - self._cost_tracker['daily_cost'])
        monthly_remaining = float(self.cost_limits['monthly_limit'] - self._cost_tracker['monthly_cost'])
        
        avg_cost_hd = float(self.pricing['image_to_video_hd'])
        avg_cost_standard = float(self.pricing['image_to_video_standard'])
        
        return {
            'daily_hd_videos': max(0, int(daily_remaining / avg_cost_hd)),
            'daily_standard_videos': max(0, int(daily_remaining / avg_cost_standard)),
            'monthly_hd_videos': max(0, int(monthly_remaining / avg_cost_hd)),
            'monthly_standard_videos': max(0, int(monthly_remaining / avg_cost_standard))
        }
    
    def _rank_platforms_by_cost_efficiency(self) -> List[Dict[str, Any]]:
        """Rank platforms by cost efficiency"""
        platform_rankings = []
        
        for platform, cost in self._cost_tracker['platform_costs'].items():
            if cost > 0:
                estimated_videos = max(1, int(float(cost) / 0.25))
                efficiency_score = estimated_videos / float(cost) * 10  # Normalize to 0-10 scale
                
                platform_rankings.append({
                    'platform': platform,
                    'cost': float(cost),
                    'estimated_videos': estimated_videos,
                    'efficiency_score': round(efficiency_score, 2)
                })
        
        return sorted(platform_rankings, key=lambda x: x['efficiency_score'], reverse=True)
    
    def _calculate_quality_cost_ratio(self) -> Dict[str, float]:
        """Calculate cost ratio between HD and standard quality"""
        hd_cost = float(self._cost_tracker['cost_by_quality']['hd'])
        standard_cost = float(self._cost_tracker['cost_by_quality']['standard'])
        
        if standard_cost == 0:
            return {'ratio': 'undefined', 'hd_premium': 0}
        
        ratio = hd_cost / standard_cost if standard_cost > 0 else 0
        hd_premium = ((hd_cost - standard_cost) / standard_cost * 100) if standard_cost > 0 else 0
        
        return {
            'hd_to_standard_ratio': round(ratio, 2),
            'hd_premium_percentage': round(hd_premium, 1)
        }
    
    def _identify_cost_optimization_opportunities(self) -> List[str]:
        """Identify specific cost optimization opportunities"""
        opportunities = []
        
        hd_percentage = self._cost_tracker['cost_by_quality']['hd'] / max(self._cost_tracker['total_cost'], Decimal('1'))
        
        if hd_percentage > 0.8:
            opportunities.append("Consider using standard quality for some platforms to reduce costs by 20-40%")
        
        platform_costs = self._cost_tracker['platform_costs']
        if len(platform_costs) > 0:
            highest_cost_platform = max(platform_costs.items(), key=lambda x: x[1])
            if float(highest_cost_platform[1]) > 20:
                opportunities.append(f"Focus optimization efforts on {highest_cost_platform[0]} - your highest cost platform")
        
        if float(self._cost_tracker['daily_cost']) > float(self.cost_limits['daily_limit']) * 0.5:
            opportunities.append("Daily cost exceeding 50% of limit - consider implementing cost controls")
        
        return opportunities if opportunities else ["System operating efficiently - no immediate optimizations needed"]
    
    def _calculate_platform_performance_score(self, platform: str) -> float:
        """Calculate performance score for a platform (0-100)"""
        # This is a simplified scoring system - in production would include engagement data
        base_score = 70
        
        # Adjust based on platform optimization features
        specs = self.platform_specs.get(platform, {})
        if 'trending_features' in specs and len(specs['trending_features']) > 3:
            base_score += 10
        
        if 'music_integration' in specs and specs['music_integration']:
            base_score += 5
        
        if 'supports_shopping_tags' in specs and specs['supports_shopping_tags']:
            base_score += 10
        
        # Adjust based on usage
        platform_cost = self._cost_tracker['platform_costs'].get(platform, Decimal('0.00'))
        if platform_cost > 10:
            base_score += 5  # Bonus for proven usage
        
        return min(100, max(0, base_score))
    
    def _assess_platform_optimization(self, platform: str) -> str:
        """Assess optimization level for platform"""
        specs = self.platform_specs.get(platform, {})
        
        optimization_features = 0
        if 'trending_features' in specs:
            optimization_features += len(specs['trending_features'])
        if 'popular_effects' in specs:
            optimization_features += len(specs['popular_effects'])
        if specs.get('music_integration'):
            optimization_features += 2
        if specs.get('captions_required'):
            optimization_features += 1
        
        if optimization_features >= 8:
            return 'highly_optimized'
        elif optimization_features >= 5:
            return 'well_optimized'
        elif optimization_features >= 3:
            return 'basic_optimization'
        else:
            return 'minimal_optimization'
    
    def _get_quality_recommendations(self, hd_cost: float, standard_cost: float) -> List[str]:
        """Get quality usage recommendations"""
        recommendations = []
        
        total_cost = hd_cost + standard_cost
        if total_cost == 0:
            return ["Start creating videos to get quality recommendations"]
        
        hd_percentage = (hd_cost / total_cost) * 100
        
        if hd_percentage > 80:
            recommendations.append("High HD usage - consider standard quality for platforms like X/Twitter where HD may not be necessary")
        elif hd_percentage < 20:
            recommendations.append("Low HD usage - consider upgrading quality for platforms like Instagram and TikTok for better engagement")
        else:
            recommendations.append("Good quality distribution - balanced usage of HD and standard quality")
        
        return recommendations
    
    def _get_template_recommendations(self) -> List[str]:
        """Get template system recommendations"""
        recommendations = []
        
        built_in_templates = len(self.template_system['templates'])
        custom_templates = len(self.template_system['custom_templates'])
        
        if custom_templates == 0:
            recommendations.append("Create custom templates to establish consistent brand identity")
        
        if custom_templates > 0 and custom_templates < 3:
            recommendations.append("Consider creating templates for different content types (professional, social, lifestyle)")
        
        recommendations.append(f"You have {built_in_templates} built-in templates and {custom_templates} custom templates available")
        
        return recommendations
    
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate system error rates"""
        total_requests = self._cost_tracker['requests_count']
        successful_requests = self._cost_tracker['successful_requests']
        
        if total_requests == 0:
            return {'overall_error_rate': 0.0, 'success_rate': 0.0}
        
        error_rate = ((total_requests - successful_requests) / total_requests) * 100
        success_rate = (successful_requests / total_requests) * 100
        
        return {
            'overall_error_rate': round(error_rate, 2),
            'success_rate': round(success_rate, 2)
        }
    
    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled system features"""
        features = [
            'veo3_integration',
            'multi_platform_support',
            'batch_processing',
            'content_moderation',
            'cost_tracking',
            'template_system',
            'queue_management',
            'advanced_analytics'
        ]
        
        if self.moderation_enabled:
            features.append('content_moderation_active')
        
        if not self.using_mock_services:
            features.append('real_api_integration')
        else:
            features.append('mock_services_testing')
        
        if self.db_manager:
            features.append('database_integration')
        
        return features
    
    def _get_system_uptime(self) -> Dict[str, Any]:
        """Get system uptime information"""
        # This is simplified - in production would track actual startup time
        return {
            'status': 'operational',
            'mock_services': self.using_mock_services,
            'initialized_features': len(self._get_enabled_features()),
            'platform_support_count': len(self.platform_specs)
        }
    
    async def create_video_variants_advanced(self, enhanced_image_path: str, product_data: Dict[str, Any],
                                           platforms: List[str] = None, **kwargs) -> Dict[str, VideoGenerationResult]:
        """Advanced batch processing with intelligent queue management and retry logic"""
        if platforms is None:
            platforms = ['x', 'tiktok', 'instagram', 'linkedin', 'pinterest', 'youtube_shorts']
        
        self.logger.info(f"Starting advanced batch processing for {len(platforms)} platforms")
        
        # Create requests with proper prioritization
        requests = []
        for platform in platforms:
            request = VideoGenerationRequest(
                input_type='image',
                input_data=enhanced_image_path,
                platform=platform,
                product_data=product_data,
                priority=kwargs.get('priority', 'normal'),
                **{k: v for k, v in kwargs.items() if k != 'priority'}
            )
            priority_score = self.batch_settings['prioritize_by_platform'].get(platform, 999)
            requests.append((priority_score, request))
        
        # Sort requests by priority
        requests.sort(key=lambda x: x[0])
        
        # Process with advanced queue management
        results = await self._process_requests_with_advanced_queue(requests)
        
        return results
    
    async def _process_requests_with_advanced_queue(self, prioritized_requests: List[Tuple[int, VideoGenerationRequest]]) -> Dict[str, VideoGenerationResult]:
        """Process requests using advanced queue management with failure recovery"""
        results = {}
        request_ids = {}
        
        # Add requests to appropriate priority queues
        for priority_score, request in prioritized_requests:
            request_id = f"{request.platform}_{int(time.time() * 1000)}"
            request_ids[request.platform] = request_id
            
            if request.priority == 'high':
                await self.queue_manager['high_priority'].put((priority_score, request_id, request))
            elif request.priority == 'low':
                await self.queue_manager['low_priority'].put((priority_score, request_id, request))
            else:
                await self.queue_manager['normal_priority'].put((priority_score, request_id, request))
            
            self.queue_manager['processing_status'][request_id] = 'queued'
        
        # Process queues in priority order
        all_processed = False
        retry_count = 0
        max_retries = 2
        
        while not all_processed and retry_count <= max_retries:
            # Process high priority queue first
            await self._process_priority_queue('high_priority', results)
            # Then normal priority
            await self._process_priority_queue('normal_priority', results)
            # Finally low priority
            await self._process_priority_queue('low_priority', results)
            
            # Check if all requests are complete
            completed_count = sum(1 for status in self.queue_manager['processing_status'].values() 
                                if status in ['completed', 'failed'])
            total_requests = len(prioritized_requests)
            
            if completed_count >= total_requests:
                all_processed = True
            elif retry_count < max_retries:
                # Process retry queue
                await self._process_retry_queue(results)
                retry_count += 1
            else:
                # Mark remaining as failed
                for request_id, status in self.queue_manager['processing_status'].items():
                    if status not in ['completed', 'failed']:
                        platform = [p for p, rid in request_ids.items() if rid == request_id][0]
                        results[platform] = VideoGenerationResult(
                            success=False,
                            video_path=None,
                            metadata={'max_retries_exceeded': True},
                            cost=Decimal('0.00'),
                            processing_time=0.0,
                            platform=platform,
                            error_message="Maximum retries exceeded"
                        )
                        self.queue_manager['processing_status'][request_id] = 'failed'
                all_processed = True
        
        # Clean up processing status
        for request_id in request_ids.values():
            self.queue_manager['processing_status'].pop(request_id, None)
            self.queue_manager['completion_callbacks'].pop(request_id, None)
        
        return results
    
    async def _process_priority_queue(self, queue_name: str, results: Dict[str, VideoGenerationResult]):
        """Process a specific priority queue"""
        priority_queue = self.queue_manager[queue_name]
        processing_tasks = []
        
        # Create semaphore for this batch
        semaphore = asyncio.Semaphore(self.batch_settings['max_concurrent_videos'])
        
        # Get all items from queue
        queue_items = []
        while not priority_queue.empty():
            try:
                item = priority_queue.get_nowait()
                queue_items.append(item)
            except:
                break
        
        if not queue_items:
            return
        
        self.logger.info(f"Processing {len(queue_items)} items from {queue_name} queue")
        
        async def process_single_request(priority_score, request_id, request):
            async with semaphore:
                try:
                    self.queue_manager['processing_status'][request_id] = 'processing'
                    
                    result = await asyncio.wait_for(
                        self.create_video_with_veo3(request),
                        timeout=self.batch_settings['batch_timeout_seconds']
                    )
                    
                    if result.success:
                        results[request.platform] = result
                        self.queue_manager['processing_status'][request_id] = 'completed'
                        self.logger.info(f"✓ {request.platform} completed successfully")
                    else:
                        # Add to retry queue if retries are enabled
                        if self.batch_settings['retry_failed_requests']:
                            await self.queue_manager['retry_queue'].put((request_id, request, 1))
                            self.queue_manager['processing_status'][request_id] = 'retry_queued'
                            self.logger.warning(f"⚠ {request.platform} failed, added to retry queue")
                        else:
                            results[request.platform] = result
                            self.queue_manager['processing_status'][request_id] = 'failed'
                            self.logger.error(f"✗ {request.platform} failed permanently")
                    
                except asyncio.TimeoutError:
                    error_result = VideoGenerationResult(
                        success=False,
                        video_path=None,
                        metadata={'timeout': True},
                        cost=Decimal('0.00'),
                        processing_time=self.batch_settings['batch_timeout_seconds'],
                        platform=request.platform,
                        error_message=f"Processing timeout after {self.batch_settings['batch_timeout_seconds']}s"
                    )
                    results[request.platform] = error_result
                    self.queue_manager['processing_status'][request_id] = 'failed'
                    self.logger.error(f"✗ {request.platform} timed out")
                    
                except Exception as e:
                    error_result = VideoGenerationResult(
                        success=False,
                        video_path=None,
                        metadata={'exception': str(e)},
                        cost=Decimal('0.00'),
                        processing_time=0.0,
                        platform=request.platform,
                        error_message=str(e)
                    )
                    results[request.platform] = error_result
                    self.queue_manager['processing_status'][request_id] = 'failed'
                    self.logger.error(f"✗ {request.platform} exception: {e}")
        
        # Create tasks for all queue items
        tasks = [
            process_single_request(priority_score, request_id, request)
            for priority_score, request_id, request in queue_items
        ]
        
        # Process all tasks concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_retry_queue(self, results: Dict[str, VideoGenerationResult]):
        """Process failed requests that are eligible for retry"""
        retry_items = []
        
        # Get all retry items
        while not self.queue_manager['retry_queue'].empty():
            try:
                item = self.queue_manager['retry_queue'].get_nowait()
                retry_items.append(item)
            except:
                break
        
        if not retry_items:
            return
        
        self.logger.info(f"Processing {len(retry_items)} retry requests")
        
        for request_id, request, retry_count in retry_items:
            if retry_count <= self.batch_settings['max_retries_per_request']:
                try:
                    self.queue_manager['processing_status'][request_id] = 'retrying'
                    
                    # Add delay for retry (exponential backoff)
                    await asyncio.sleep(min(2 ** retry_count, 10))
                    
                    result = await self.create_video_with_veo3(request)
                    
                    if result.success:
                        results[request.platform] = result
                        self.queue_manager['processing_status'][request_id] = 'completed'
                        self.logger.info(f"✓ {request.platform} retry successful")
                    else:
                        # Try again if under retry limit
                        if retry_count < self.batch_settings['max_retries_per_request']:
                            await self.queue_manager['retry_queue'].put((request_id, request, retry_count + 1))
                            self.queue_manager['processing_status'][request_id] = 'retry_queued'
                        else:
                            results[request.platform] = result
                            self.queue_manager['processing_status'][request_id] = 'failed'
                            self.logger.error(f"✗ {request.platform} max retries exceeded")
                            
                except Exception as e:
                    if retry_count < self.batch_settings['max_retries_per_request']:
                        await self.queue_manager['retry_queue'].put((request_id, request, retry_count + 1))
                        self.queue_manager['processing_status'][request_id] = 'retry_queued'
                    else:
                        error_result = VideoGenerationResult(
                            success=False,
                            video_path=None,
                            metadata={'retry_exception': str(e)},
                            cost=Decimal('0.00'),
                            processing_time=0.0,
                            platform=request.platform,
                            error_message=f"Retry failed: {e}"
                        )
                        results[request.platform] = error_result
                        self.queue_manager['processing_status'][request_id] = 'failed'
            else:
                # Max retries exceeded
                error_result = VideoGenerationResult(
                    success=False,
                    video_path=None,
                    metadata={'max_retries_exceeded': True},
                    cost=Decimal('0.00'),
                    processing_time=0.0,
                    platform=request.platform,
                    error_message="Maximum retry attempts exceeded"
                )
                results[request.platform] = error_result
                self.queue_manager['processing_status'][request_id] = 'failed'
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and processing statistics"""
        queue_sizes = {
            'high_priority': self.queue_manager['high_priority'].qsize(),
            'normal_priority': self.queue_manager['normal_priority'].qsize(),
            'low_priority': self.queue_manager['low_priority'].qsize(),
            'retry_queue': self.queue_manager['retry_queue'].qsize(),
        }
        
        processing_status_counts = {}
        for status in self.queue_manager['processing_status'].values():
            processing_status_counts[status] = processing_status_counts.get(status, 0) + 1
        
        return {
            'queue_sizes': queue_sizes,
            'total_queued': sum(queue_sizes.values()),
            'processing_status': processing_status_counts,
            'active_requests': len(self.queue_manager['processing_status']),
            'concurrent_limit': self.batch_settings['max_concurrent_videos'],
            'retry_enabled': self.batch_settings['retry_failed_requests'],
            'max_retries': self.batch_settings['max_retries_per_request']
        }

    def _get_video_metadata(self, video_path: str, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Get comprehensive metadata about the generated video"""
        try:
            video_clip = VideoFileClip(video_path)
            
            # Basic video properties
            metadata = {
                'duration': video_clip.duration,
                'fps': video_clip.fps,
                'dimensions': video_clip.size,
                'file_size': os.path.getsize(video_path),
                'aspect_ratio': f"{video_clip.size[0]}:{video_clip.size[1]}",
                'created_at': datetime.utcnow().isoformat(),
                'platform': request.platform,
                'input_type': request.input_type,
                'quality': request.quality,
                'style': request.style
            }
            
            # Platform-specific metadata
            platform_spec = self.platform_specs.get(request.platform, {})
            metadata['platform_compliance'] = {
                'max_duration_compliant': video_clip.duration <= platform_spec.get('max_duration', float('inf')),
                'dimensions_match': video_clip.size == tuple(platform_spec.get('dimensions', video_clip.size)),
                'fps_match': video_clip.fps == platform_spec.get('fps', video_clip.fps)
            }
            
            # Quality metrics
            metadata['quality_metrics'] = self._calculate_quality_metrics(video_clip)
            
            # Product context
            metadata['product_context'] = {
                'product_name': request.product_data.get('name'),
                'category': request.product_data.get('category'),
                'features_highlighted': len(request.product_data.get('features', '').split(','))
            }
            
            video_clip.close()
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get video metadata: {e}")
            return {'error': str(e), 'created_at': datetime.utcnow().isoformat()}
    
    def _calculate_quality_metrics(self, video_clip: VideoFileClip) -> Dict[str, Any]:
        """Calculate video quality metrics"""
        try:
            # Basic quality indicators
            metrics = {
                'resolution_score': self._score_resolution(video_clip.size),
                'duration_appropriateness': self._score_duration(video_clip.duration),
                'fps_quality': self._score_fps(video_clip.fps),
                'estimated_bitrate': self._estimate_bitrate(video_clip)
            }
            
            # Overall quality score (0-100)
            metrics['overall_quality_score'] = (
                metrics['resolution_score'] * 0.3 +
                metrics['duration_appropriateness'] * 0.2 +
                metrics['fps_quality'] * 0.3 +
                min(metrics['estimated_bitrate'] / 2000000, 1.0) * 100 * 0.2
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _score_resolution(self, size: Tuple[int, int]) -> float:
        """Score video resolution (0-100)"""
        width, height = size
        total_pixels = width * height
        
        # HD benchmarks
        if total_pixels >= 1920 * 1080:  # Full HD or higher
            return 100.0
        elif total_pixels >= 1280 * 720:  # HD
            return 80.0
        elif total_pixels >= 960 * 540:  # qHD
            return 60.0
        else:
            return 40.0
    
    def _score_duration(self, duration: float) -> float:
        """Score video duration appropriateness (0-100)"""
        if 15 <= duration <= 60:
            return 100.0
        elif 10 <= duration <= 90:
            return 80.0
        elif 5 <= duration <= 120:
            return 60.0
        else:
            return 40.0
    
    def _score_fps(self, fps: float) -> float:
        """Score video FPS (0-100)"""
        if fps >= 30:
            return 100.0
        elif fps >= 24:
            return 80.0
        elif fps >= 15:
            return 60.0
        else:
            return 40.0
    
    def _estimate_bitrate(self, video_clip: VideoFileClip) -> float:
        """Estimate video bitrate in bps"""
        # This is a rough estimation
        try:
            # Estimate based on resolution and FPS
            width, height = video_clip.size
            total_pixels = width * height
            return total_pixels * video_clip.fps * 0.1  # Very rough estimation
        except:
            return 1000000  # Default 1 Mbps estimate

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than specified age"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleanup_count = 0
            
            # Clean temp directory
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleanup_count += 1
            
            # Clean cache directory
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleanup_count += 1
            
            # Clean audio directory
            for filename in os.listdir(self.audio_dir):
                file_path = os.path.join(self.audio_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleanup_count += 1
            
            if cleanup_count > 0:
                self.logger.info(f"Cleaned up {cleanup_count} temporary files")
                        
        except Exception as e:
            self.logger.error(f"Temp file cleanup failed: {e}")

    async def _check_rate_limits(self):
        """Check and enforce API rate limits"""
        with self._rate_limit_lock:
            current_time = time.time()
            
            # Remove old timestamps (older than 1 minute)
            self._request_timestamps = [
                ts for ts in self._request_timestamps 
                if current_time - ts < 60
            ]
            
            # Check if we're hitting rate limits (max 10 requests per minute)
            if len(self._request_timestamps) >= 10:
                wait_time = 60 - (current_time - self._request_timestamps[0])
                if wait_time > 0:
                    self.logger.info(f"Rate limit hit, waiting {wait_time:.1f} seconds")
                    await asyncio.sleep(wait_time)
            
            # Add current timestamp
            self._request_timestamps.append(current_time)
    
    async def _moderate_content(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Perform comprehensive content moderation on video generation request"""
        try:
            moderation_results = {
                'text_moderation': await self._moderate_text_content(request),
                'image_moderation': await self._moderate_image_content(request),
                'brand_safety': await self._check_brand_safety(request),
                'platform_compliance': await self._check_platform_compliance(request),
                'quality_gates': await self._check_quality_gates(request)
            }
            
            # Aggregate moderation results
            all_approved = all(result.get('approved', True) for result in moderation_results.values())
            confidence_scores = [result.get('confidence', 1.0) for result in moderation_results.values()]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Collect all categories and issues
            all_categories = []
            all_issues = []
            for result in moderation_results.values():
                all_categories.extend(result.get('categories', []))
                if 'issues' in result:
                    all_issues.extend(result['issues'])
            
            return {
                'approved': all_approved,
                'confidence': avg_confidence,
                'categories': list(set(all_categories)),
                'detailed_results': moderation_results,
                'issues': all_issues,
                'reasoning': self._generate_moderation_reasoning(moderation_results, all_approved)
            }
            
        except Exception as e:
            self.logger.error(f"Content moderation failed: {e}")
            return {
                'approved': False,
                'confidence': 0.0,
                'categories': ['moderation_error'],
                'reasoning': f'Moderation system failed: {e}',
                'error': str(e)
            }
    
    async def _moderate_text_content(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Moderate text content in the request"""
        try:
            # Check product data text content
            text_to_check = []
            if isinstance(request.input_data, str) and request.input_type == 'text':
                text_to_check.append(request.input_data)
            
            # Check product descriptions
            product_text = ' '.join([
                str(request.product_data.get('name', '')),
                str(request.product_data.get('description', '')),
                str(request.product_data.get('features', ''))
            ])
            text_to_check.append(product_text)
            
            # Basic content filters
            blocked_terms = [
                'spam', 'scam', 'fake', 'illegal', 'adult', 'violence',
                'hate', 'discrimination', 'misleading', 'dangerous'
            ]
            
            issues = []
            for text in text_to_check:
                if text:
                    text_lower = text.lower()
                    for term in blocked_terms:
                        if term in text_lower:
                            issues.append(f"Contains potentially inappropriate term: {term}")
            
            return {
                'approved': len(issues) == 0,
                'confidence': 0.9 if len(issues) == 0 else 0.3,
                'categories': ['text_content'] if issues else [],
                'issues': issues,
                'text_analyzed': len(' '.join(text_to_check).split())
            }
            
        except Exception as e:
            return {
                'approved': False,
                'confidence': 0.0,
                'categories': ['text_moderation_error'],
                'issues': [f"Text moderation failed: {e}"]
            }
    
    async def _moderate_image_content(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Moderate image content if provided"""
        try:
            if request.input_type != 'image' or not request.input_data:
                return {'approved': True, 'confidence': 1.0, 'categories': []}
                
            # Check if image file exists and is accessible
            if not os.path.exists(request.input_data):
                return {
                    'approved': False,
                    'confidence': 0.0,
                    'categories': ['missing_image'],
                    'issues': ['Image file does not exist']
                }
            
            # Basic image checks
            try:
                from PIL import Image
                with Image.open(request.input_data) as img:
                    width, height = img.size
                    
                    # Check minimum resolution
                    if width < 500 or height < 500:
                        return {
                            'approved': False,
                            'confidence': 0.5,
                            'categories': ['low_quality_image'],
                            'issues': ['Image resolution too low for quality video generation']
                        }
                    
                    # Check aspect ratio reasonableness
                    aspect_ratio = width / height
                    if aspect_ratio > 5 or aspect_ratio < 0.2:
                        return {
                            'approved': False,
                            'confidence': 0.4,
                            'categories': ['unusual_aspect_ratio'],
                            'issues': ['Image has unusual aspect ratio that may not work well for video']
                        }
            except Exception as img_error:
                return {
                    'approved': False,
                    'confidence': 0.0,
                    'categories': ['image_processing_error'],
                    'issues': [f'Could not process image: {img_error}']
                }
            
            return {
                'approved': True,
                'confidence': 0.9,
                'categories': [],
                'image_dimensions': f"{width}x{height}"
            }
            
        except Exception as e:
            return {
                'approved': False,
                'confidence': 0.0,
                'categories': ['image_moderation_error'],
                'issues': [f"Image moderation failed: {e}"]
            }
    
    async def _check_brand_safety(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Check brand safety considerations"""
        try:
            issues = []
            
            # Check for brand safety categories
            sensitive_categories = [
                'alcohol', 'gambling', 'tobacco', 'pharmaceuticals', 
                'political', 'religious', 'adult', 'weapons'
            ]
            
            product_category = request.product_data.get('category', '').lower()
            if any(sensitive in product_category for sensitive in sensitive_categories):
                issues.append(f"Product category '{product_category}' requires special handling")
            
            # Check platform suitability
            platform = request.platform
            if platform == 'linkedin' and product_category in ['entertainment', 'gaming', 'lifestyle']:
                issues.append("Entertainment content may not perform well on LinkedIn")
            
            return {
                'approved': len(issues) == 0,
                'confidence': 0.8 if len(issues) == 0 else 0.6,
                'categories': ['brand_safety'] if issues else [],
                'issues': issues
            }
            
        except Exception as e:
            return {
                'approved': True,  # Default to approved for brand safety
                'confidence': 0.5,
                'categories': ['brand_safety_error'],
                'issues': [f"Brand safety check failed: {e}"]
            }
    
    async def _check_platform_compliance(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Check platform-specific compliance requirements"""
        try:
            issues = []
            platform_spec = self.platform_specs.get(request.platform)
            
            if not platform_spec:
                issues.append(f"Unsupported platform: {request.platform}")
                return {
                    'approved': False,
                    'confidence': 0.0,
                    'categories': ['unsupported_platform'],
                    'issues': issues
                }
            
            # Check duration compliance
            max_duration = platform_spec['max_duration']
            if request.duration > max_duration:
                issues.append(f"Duration {request.duration}s exceeds platform maximum {max_duration}s")
            
            # Check quality requirements
            if request.quality not in ['hd', 'standard']:
                issues.append(f"Invalid quality setting: {request.quality}")
            
            # Platform-specific checks
            if request.platform == 'linkedin' and not request.add_captions:
                issues.append("LinkedIn videos should include captions for accessibility")
            
            if request.platform == 'tiktok' and request.duration > 60:
                issues.append("TikTok videos over 60 seconds have lower engagement rates")
            
            return {
                'approved': len(issues) == 0,
                'confidence': 0.95 if len(issues) == 0 else 0.7,
                'categories': ['platform_compliance'] if issues else [],
                'issues': issues
            }
            
        except Exception as e:
            return {
                'approved': True,  # Default to approved
                'confidence': 0.5,
                'categories': ['compliance_check_error'],
                'issues': [f"Platform compliance check failed: {e}"]
            }
    
    async def _check_quality_gates(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Check quality gates for video generation"""
        try:
            issues = []
            warnings = []
            
            # Check product data completeness
            required_fields = ['name']
            for field in required_fields:
                if not request.product_data.get(field):
                    issues.append(f"Missing required product field: {field}")
            
            recommended_fields = ['description', 'features', 'category']
            for field in recommended_fields:
                if not request.product_data.get(field):
                    warnings.append(f"Missing recommended product field: {field}")
            
            # Check request parameters
            if request.duration < 5:
                issues.append("Video duration too short (minimum 5 seconds recommended)")
            elif request.duration < 10:
                warnings.append("Short video duration may limit engagement")
            
            # Style and customization checks
            if not request.style or request.style == 'default':
                warnings.append("Using default style - custom styling may improve performance")
            
            return {
                'approved': len(issues) == 0,
                'confidence': 0.9 if len(issues) == 0 and len(warnings) == 0 else 0.7,
                'categories': ['quality_gates'] if issues else [],
                'issues': issues,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'approved': True,  # Default to approved for quality gates
                'confidence': 0.5,
                'categories': ['quality_gate_error'],
                'issues': [f"Quality gate check failed: {e}"]
            }
    
    def _generate_moderation_reasoning(self, results: Dict[str, Any], approved: bool) -> str:
        """Generate human-readable reasoning for moderation decision"""
        if approved:
            return "Content passed all moderation checks and quality gates"
        
        issues = []
        for check_name, result in results.items():
            if not result.get('approved', True) and result.get('issues'):
                issues.extend([f"{check_name}: {issue}" for issue in result['issues']])
        
        if issues:
            return f"Content rejected due to: {'; '.join(issues[:3])}" + ("..." if len(issues) > 3 else "")
        else:
            return "Content rejected by moderation system"
    
    def _calculate_cost(self, request: VideoGenerationRequest) -> Decimal:
        """Calculate cost for video generation with platform and quality considerations"""
        if self.using_mock_services:
            return Decimal('0.00')  # No cost for mock services
            
        base_key = f"{request.input_type}_to_video_{request.quality}"
        base_cost = self.pricing.get(base_key, Decimal('0.25'))
        
        # Adjust for duration (base price is for 30 seconds)
        duration_multiplier = max(1, request.duration / 30)
        
        # Platform-specific cost adjustments
        platform_multipliers = {
            'tiktok': Decimal('1.1'),    # 10% premium for TikTok optimization
            'instagram': Decimal('1.05'),  # 5% premium for Instagram optimization
            'x': Decimal('1.0'),         # Standard cost
            'linkedin': Decimal('1.15')   # 15% premium for LinkedIn professional features
        }
        
        platform_multiplier = platform_multipliers.get(request.platform, Decimal('1.0'))
        
        total_cost = base_cost * Decimal(str(duration_multiplier)) * platform_multiplier
        
        return total_cost
    
    def _update_cost_tracker(self, cost: Decimal, request: VideoGenerationRequest = None, success: bool = True):
        """Update comprehensive cost tracking with detailed analytics"""
        today = datetime.now().date()
        this_month = today.replace(day=1)
        
        # Reset daily cost if new day
        if self._cost_tracker['last_reset'] != today:
            self._cost_tracker['daily_cost'] = Decimal('0.00')
            self._cost_tracker['last_reset'] = today
            
        # Reset monthly cost if new month
        if self._cost_tracker['last_monthly_reset'] != this_month:
            self._cost_tracker['monthly_cost'] = Decimal('0.00')
            self._cost_tracker['last_monthly_reset'] = this_month
        
        # Update costs
        self._cost_tracker['total_cost'] += cost
        self._cost_tracker['daily_cost'] += cost
        self._cost_tracker['monthly_cost'] += cost
        self._cost_tracker['requests_count'] += 1
        
        if success:
            self._cost_tracker['successful_requests'] += 1
            
        # Track costs by category
        if request:
            platform = request.platform
            if platform in self._cost_tracker['platform_costs']:
                self._cost_tracker['platform_costs'][platform] += cost
                
            quality = request.quality
            if quality in self._cost_tracker['cost_by_quality']:
                self._cost_tracker['cost_by_quality'][quality] += cost
                
            request_type = request.input_type + '_to_video'
            if request_type in self._cost_tracker['cost_by_type']:
                self._cost_tracker['cost_by_type'][request_type] += cost
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost tracking summary with analytics"""
        daily_remaining = float(self.cost_limits['daily_limit'] - self._cost_tracker['daily_cost'])
        monthly_remaining = float(self.cost_limits['monthly_limit'] - self._cost_tracker['monthly_cost'])
        
        success_rate = 0.0
        if self._cost_tracker['requests_count'] > 0:
            success_rate = self._cost_tracker['successful_requests'] / self._cost_tracker['requests_count']
            
        avg_cost_per_video = 0.0
        if self._cost_tracker['successful_requests'] > 0:
            avg_cost_per_video = float(self._cost_tracker['total_cost']) / self._cost_tracker['successful_requests']
        
        return {
            'costs': {
                'total': float(self._cost_tracker['total_cost']),
                'daily': float(self._cost_tracker['daily_cost']),
                'monthly': float(self._cost_tracker['monthly_cost']),
                'daily_remaining': max(0, daily_remaining),
                'monthly_remaining': max(0, monthly_remaining)
            },
            'limits': {
                'daily_limit': float(self.cost_limits['daily_limit']),
                'monthly_limit': float(self.cost_limits['monthly_limit']),
                'daily_usage_percent': min(100, float(self._cost_tracker['daily_cost'] / self.cost_limits['daily_limit'] * 100)),
                'monthly_usage_percent': min(100, float(self._cost_tracker['monthly_cost'] / self.cost_limits['monthly_limit'] * 100))
            },
            'analytics': {
                'requests_total': self._cost_tracker['requests_count'],
                'requests_successful': self._cost_tracker['successful_requests'],
                'success_rate': success_rate,
                'avg_cost_per_video': avg_cost_per_video,
                'platform_costs': {k: float(v) for k, v in self._cost_tracker['platform_costs'].items()},
                'quality_costs': {k: float(v) for k, v in self._cost_tracker['cost_by_quality'].items()},
                'type_costs': {k: float(v) for k, v in self._cost_tracker['cost_by_type'].items()}
            },
            'metadata': {
                'currency': 'USD',
                'last_daily_reset': self._cost_tracker['last_reset'].isoformat(),
                'last_monthly_reset': self._cost_tracker['last_monthly_reset'].isoformat(),
                'using_mock_services': getattr(self, 'using_mock_services', False)
            }
        }
    
    def _check_cost_limits(self, estimated_cost: Decimal, request: VideoGenerationRequest) -> Tuple[bool, str]:
        """Check if request would exceed cost limits and suggest optimizations"""
        daily_after = self._cost_tracker['daily_cost'] + estimated_cost
        monthly_after = self._cost_tracker['monthly_cost'] + estimated_cost
        
        # Hard limits
        if daily_after > self.cost_limits['daily_limit']:
            return False, f"Would exceed daily limit of ${self.cost_limits['daily_limit']} (current: ${self._cost_tracker['daily_cost']}, estimated: ${estimated_cost})"
            
        if monthly_after > self.cost_limits['monthly_limit']:
            return False, f"Would exceed monthly limit of ${self.cost_limits['monthly_limit']} (current: ${self._cost_tracker['monthly_cost']}, estimated: ${estimated_cost})"
        
        # Warning thresholds
        daily_threshold = self.cost_limits['daily_limit'] * Decimal(str(self.cost_limits['warning_threshold']))
        monthly_threshold = self.cost_limits['monthly_limit'] * Decimal(str(self.cost_limits['warning_threshold']))
        
        warnings = []
        
        if daily_after > daily_threshold:
            remaining_daily = float(self.cost_limits['daily_limit'] - daily_after)
            warnings.append(f"Approaching daily limit. ${remaining_daily:.2f} remaining after this request.")
            
        if monthly_after > monthly_threshold:
            remaining_monthly = float(self.cost_limits['monthly_limit'] - monthly_after)
            warnings.append(f"Approaching monthly limit. ${remaining_monthly:.2f} remaining after this request.")
            
        # Auto-optimization suggestion
        if (warnings and self.cost_limits['auto_downgrade_quality'] and 
            request.quality == 'hd'):
            standard_cost = self._calculate_cost(VideoGenerationRequest(
                input_type=request.input_type,
                input_data=request.input_data,
                platform=request.platform,
                product_data=request.product_data,
                quality='standard',
                duration=request.duration
            ))
            savings = estimated_cost - standard_cost
            warnings.append(f"Consider using 'standard' quality to save ${savings:.2f} per video.")
            
        return True, '; '.join(warnings) if warnings else 'OK'
    
    def _generate_output_path(self, request: VideoGenerationRequest) -> str:
        """Generate unique output path for video"""
        timestamp = int(time.time())
        product_name = request.product_data.get('name', 'product').replace(' ', '_')
        filename = f"{product_name}_{request.platform}_{request.input_type}_video_{timestamp}.mp4"
        return os.path.join(self.temp_dir, filename)
    
    def _generate_thumbnail(self, video_path: str) -> Optional[str]:
        """Generate thumbnail from video"""
        try:
            video_clip = VideoFileClip(video_path)
            thumbnail_time = min(2.0, video_clip.duration / 2)
            
            thumbnail_path = video_path.replace('.mp4', '_thumbnail.jpg')
            video_clip.save_frame(thumbnail_path, t=thumbnail_time)
            
            video_clip.close()
            return thumbnail_path
            
        except Exception as e:
            self.logger.error(f"Thumbnail generation failed: {e}")
            return None
    
    def _validate_video_quality(self, video_clip: VideoFileClip, platform_spec: Dict[str, Any]) -> bool:
        """Validate video meets quality thresholds and platform requirements"""
        try:
            # Check duration constraints
            if video_clip.duration < self.quality_thresholds['min_duration']:
                self.logger.warning(f"Video duration {video_clip.duration}s below minimum {self.quality_thresholds['min_duration']}s")
                return False
                
            if video_clip.duration > platform_spec['max_duration']:
                self.logger.warning(f"Video duration {video_clip.duration}s exceeds platform maximum {platform_spec['max_duration']}s")
                return False
            
            # Check resolution constraints
            min_width, min_height = self.quality_thresholds['min_resolution']
            if video_clip.size[0] < min_width or video_clip.size[1] < min_height:
                self.logger.warning(f"Video resolution {video_clip.size} below minimum {self.quality_thresholds['min_resolution']}")
                return False
            
            # Check FPS
            if video_clip.fps < 15:
                self.logger.warning(f"Video FPS {video_clip.fps} below minimum 15")
                return False
                
            # Check aspect ratio compatibility
            video_aspect = video_clip.size[0] / video_clip.size[1]
            platform_aspect_parts = platform_spec['aspect_ratio'].split(':')
            platform_aspect = float(platform_aspect_parts[0]) / float(platform_aspect_parts[1])
            
            aspect_tolerance = 0.1  # 10% tolerance
            if abs(video_aspect - platform_aspect) > aspect_tolerance:
                self.logger.warning(f"Video aspect ratio {video_aspect:.2f} doesn't match platform requirement {platform_aspect:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}")
            return False
    
    def _validate_video_request(self, request: VideoGenerationRequest) -> bool:
        """Validate video generation request parameters"""
        try:
            # Validate platform
            if request.platform not in self.platform_specs:
                self.logger.error(f"Unsupported platform: {request.platform}")
                return False
                
            # Validate input type
            if request.input_type not in ['text', 'image', 'video']:
                self.logger.error(f"Invalid input type: {request.input_type}")
                return False
                
            # Validate input data exists
            if not request.input_data:
                self.logger.error("Input data is required")
                return False
                
            # For file inputs, check if file exists
            if request.input_type in ['image', 'video'] and not os.path.exists(request.input_data):
                self.logger.error(f"Input file does not exist: {request.input_data}")
                return False
                
            # Validate duration
            platform_spec = self.platform_specs[request.platform]
            if request.duration > platform_spec['max_duration']:
                self.logger.warning(f"Requested duration {request.duration}s exceeds platform maximum {platform_spec['max_duration']}s, will be clamped")
                
            # Validate quality setting
            if request.quality not in ['hd', 'standard']:
                self.logger.warning(f"Invalid quality setting '{request.quality}', defaulting to 'hd'")
                request.quality = 'hd'
                
            return True
            
        except Exception as e:
            self.logger.error(f"Request validation failed: {e}")
            return False
    
    async def _upload_to_gcs(self, file_path: str) -> str:
        """Upload file to Google Cloud Storage (with mock support)"""
        if not self.storage_client:
            raise ValueError("Google Cloud Storage not configured")
        
        if self.using_mock_services:
            # Mock upload - just return a fake GCS URI
            blob_name = f"uploads/{int(time.time())}_{os.path.basename(file_path)}"
            bucket_name = "mock-veo-uploads"
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            return f"gs://{bucket_name}/{blob_name}"
        else:
            # Real GCS upload
            bucket_name = f"{self.google_project_id}-veo-uploads"
            blob_name = f"uploads/{int(time.time())}_{os.path.basename(file_path)}"
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(file_path)
            return f"gs://{bucket_name}/{blob_name}"
    
    async def _download_from_gcs(self, gcs_uri: str) -> str:
        """Download file from Google Cloud Storage (with mock support)"""
        if not self.storage_client:
            raise ValueError("Google Cloud Storage not configured")
        
        # Parse GCS URI
        bucket_name = gcs_uri.split('/')[2]
        blob_name = '/'.join(gcs_uri.split('/')[3:])
        
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download to temp file
        temp_path = os.path.join(self.temp_dir, f"downloaded_{int(time.time())}.mp4")
        blob.download_to_filename(temp_path)
        
        return temp_path
    
    def _save_video_bytes(self, video_bytes: bytes, platform: str) -> str:
        """Save video bytes to file (enhanced for mock and real scenarios)"""
        timestamp = int(time.time())
        filename = f"generated_{platform}_video_{timestamp}.mp4"
        output_path = os.path.join(self.temp_dir, filename)
        
        if self.using_mock_services and video_bytes == b'mock_video_data':
            # Create a more realistic mock video file for testing
            self._create_mock_video_file(output_path, platform)
        else:
            # Save actual video bytes
            with open(output_path, 'wb') as f:
                f.write(video_bytes)
        
        return output_path
    
    def _create_mock_video_file(self, output_path: str, platform: str):
        """Create a mock video file for testing purposes"""
        try:
            from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
            
            platform_spec = self.platform_specs.get(platform, self.platform_specs['x'])
            duration = 5  # Short duration for testing
            
            # Create a colored background
            background = ColorClip(
                size=platform_spec['dimensions'],
                color=(100, 150, 200),  # Light blue
                duration=duration
            )
            
            # Add text overlay
            text = TextClip(
                f"Mock Video - {platform.upper()}",
                fontsize=50,
                color='white',
                font='Arial-Bold'
            ).set_position('center').set_duration(duration)
            
            # Compose video
            final_video = CompositeVideoClip([background, text])
            final_video.write_videofile(
                output_path,
                fps=24,
                verbose=False,
                logger=None
            )
            final_video.close()
            
        except Exception as e:
            # If moviepy fails, create a simple file with mock data
            self.logger.warning(f"Failed to create mock video with moviepy: {e}")
            with open(output_path, 'wb') as f:
                f.write(b'mock_video_data_extended_for_testing' * 1000)
    
    async def _log_video_generation(self, result: VideoGenerationResult, request: VideoGenerationRequest):
        """Log video generation to database with comprehensive metadata"""
        if not self.db_manager:
            self.logger.debug("Database manager not available, skipping video generation logging")
            return
        
        try:
            # Create Post record for video content
            post_data = {
                'platform': request.platform,
                'content_type': 'video',
                'video_url': result.video_path,
                'thumbnail_url': result.thumbnail_path,
                'metadata': {
                    'generation_metadata': result.metadata,
                    'input_type': request.input_type,
                    'quality': request.quality,
                    'duration': request.duration,
                    'style': request.style,
                    'cost': float(result.cost),
                    'processing_time': result.processing_time,
                    'using_mock_services': getattr(self, 'using_mock_services', False),
                    'platform_optimized': True
                },
                'status': 'generated' if result.success else 'failed',
                'approval_status': 'pending'
            }
            
            # Add moderation results if available
            if result.moderation_result:
                post_data['metadata']['moderation_result'] = result.moderation_result
                post_data['approval_status'] = 'approved' if result.moderation_result.get('approved', False) else 'rejected'
            
            # Link to product if available in request
            product_name = request.product_data.get('name')
            if product_name:
                # In a real implementation, you'd look up the product ID
                post_data['metadata']['product_name'] = product_name
                post_data['metadata']['product_data'] = request.product_data
            
            # Log the video generation
            await self.db_manager.create_video_post(post_data)
            
            self.logger.info(f"Logged video generation to database: {request.platform} video, success={result.success}")
            
        except Exception as e:
            self.logger.error(f"Failed to log video generation to database: {e}")
            # Don't raise the exception to avoid breaking the video generation flow
    
    async def _update_video_engagement_tracking(self, video_path: str, platform: str, metadata: Dict[str, Any]):
        """Update engagement tracking for generated video"""
        if not self.db_manager:
            return
            
        try:
            # This would integrate with engagement tracking models
            engagement_data = {
                'video_path': video_path,
                'platform': platform,
                'tracked_metrics': ['views', 'likes', 'shares', 'comments', 'click_through_rate'],
                'tracking_start_time': datetime.utcnow(),
                'metadata': metadata
            }
            
            await self.db_manager.setup_engagement_tracking(engagement_data)
            self.logger.debug(f"Set up engagement tracking for {platform} video")
            
        except Exception as e:
            self.logger.error(f"Failed to setup engagement tracking: {e}")
    
    async def create_text_to_video(self, text_prompt: str, platform: str, 
                                 product_data: Dict[str, Any], **kwargs) -> VideoGenerationResult:
        """Create video directly from text prompt"""
        request = VideoGenerationRequest(
            input_type='text',
            input_data=text_prompt,
            platform=platform,
            product_data=product_data,
            **kwargs
        )
        
        return await self.create_video_with_veo3(request)
    
    async def enhance_existing_video(self, video_path: str, platform: str,
                                   product_data: Dict[str, Any], **kwargs) -> VideoGenerationResult:
        """Enhance existing video for platform"""
        request = VideoGenerationRequest(
            input_type='video',
            input_data=video_path,
            platform=platform,
            product_data=product_data,
            **kwargs
        )
        
        return await self.create_video_with_veo3(request)
    
    async def _create_text_to_video_fallback(self, request: VideoGenerationRequest,
                                           platform_spec: Dict[str, Any], duration: int) -> str:
        """Create video from text using fallback method"""
        # This would create a video with animated text and background
        # For now, create a simple text-based video
        
        # Create background
        background_color = (30, 30, 30) if request.platform in ['tiktok', 'instagram'] else (255, 255, 255)
        background = ColorClip(
            size=platform_spec['dimensions'],
            color=background_color,
            duration=duration
        )
        
        # Create main text
        main_text = request.product_data.get('name', 'Product Showcase')
        text_clip = TextClip(
            main_text,
            fontsize=50,
            color='white' if background_color[0] < 128 else 'black',
            font='Arial-Bold'
        ).set_position('center').set_duration(duration)
        
        # Composite
        final_video = CompositeVideoClip([background, text_clip])
        
        # Save
        output_path = self._generate_output_path(request)
        final_video.write_videofile(
            output_path,
            fps=platform_spec['fps'],
            codec='libx264',
            verbose=False,
            logger=None
        )
        
        final_video.close()
        return output_path
    
    async def _enhance_existing_video(self, request: VideoGenerationRequest,
                                    platform_spec: Dict[str, Any], duration: int) -> str:
        """Enhance existing video for platform optimization"""
        input_clip = VideoFileClip(request.input_data)
        
        # Trim to requested duration
        if input_clip.duration > duration:
            input_clip = input_clip.subclip(0, duration)
        
        # Apply platform optimizations
        if request.platform == 'tiktok':
            input_clip = self._apply_tiktok_effects(input_clip, duration)
        elif request.platform == 'instagram':
            input_clip = self._apply_instagram_effects(input_clip, duration)
        elif request.platform == 'x':
            input_clip = self._apply_x_effects(input_clip, duration)
        elif request.platform == 'linkedin':
            input_clip = self._apply_linkedin_effects(input_clip, duration, request.product_data)
        
        # Resize for platform
        input_clip = input_clip.resize(platform_spec['dimensions'])
        
        # Save enhanced video
        output_path = self._generate_output_path(request)
        input_clip.write_videofile(
            output_path,
            fps=platform_spec['fps'],
            codec='libx264',
            bitrate=platform_spec['bitrate'],
            verbose=False,
            logger=None
        )
        
        input_clip.close()
        return output_path


if __name__ == "__main__":
    async def test_video_generator():
        # Test the video generator
        generator = VideoGenerator()
        
        # Sample product data
        sample_product = {
            'name': 'Wireless Headphones',
            'category': 'electronics',
            'features': 'noise cancellation, wireless connectivity, long battery life',
            'target_audience': 'music lovers and professionals',
            'brand_voice': 'innovative and reliable'
        }
        
        print("Video generator initialized successfully")
        print(f"Supported platforms: {list(generator.platform_specs.keys())}")
        print(f"Cost summary: {generator.get_cost_summary()}")
        
        # Test text-to-video
        try:
            result = await generator.create_text_to_video(
                "Showcase wireless headphones with premium sound quality",
                "tiktok",
                sample_product,
                duration=15,
                style='trendy'
            )
            print(f"Text-to-video result: {result.success}, Path: {result.video_path}")
        except Exception as e:
            print(f"Text-to-video test failed: {e}")
    
    # Run async test
    import asyncio
    asyncio.run(test_video_generator())