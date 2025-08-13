"""
Comprehensive Content Quality Validation System
Multi-layer validation and filtering for images, videos, and text content
with platform-specific quality standards and automated improvement suggestions
"""

import os
import json
import time
import hashlib
import asyncio
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
from io import BytesIO
import base64
import tempfile

# Image and video processing
from PIL import Image, ImageEnhance, ImageStat, ImageDraw, ImageFont
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip

# AI and ML libraries
import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Text analysis
import nltk
from textstat import flesch_reading_ease, dale_chall_readability_score
import re
import emoji

# Database integration
try:
    from ..database.models import DatabaseManager, Product, Post, EngagementMetrics
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# LLM integration
from llm import create_client, get_response_from_llm


@dataclass
class QualityScore:
    """Individual quality metric score"""
    metric_name: str
    score: float  # 0-100
    weight: float  # Importance weight
    threshold: float  # Minimum acceptable score
    passed: bool
    details: Dict[str, Any]
    suggestions: List[str]


@dataclass
class ContentQualityResult:
    """Comprehensive quality validation result"""
    content_id: str
    content_type: str  # 'image', 'video', 'text'
    platform: str
    overall_score: float  # 0-100 weighted average
    individual_scores: List[QualityScore]
    passed_validation: bool
    requires_human_review: bool
    improvement_suggestions: List[str]
    retry_recommended: bool
    retry_parameters: Optional[Dict[str, Any]]
    processing_time: float
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class ValidationConfig:
    """Configuration for quality validation"""
    platform: str
    content_type: str
    quality_tier: str  # 'premium', 'standard', 'basic'
    enable_ai_analysis: bool = True
    enable_technical_analysis: bool = True
    enable_brand_compliance: bool = True
    enable_engagement_prediction: bool = True
    human_review_threshold: float = 60.0  # Scores below this trigger human review
    auto_reject_threshold: float = 30.0   # Scores below this auto-reject
    retry_on_failure: bool = True


class ContentQualityValidator:
    """
    Comprehensive content quality validation and filtering system
    
    Features:
    - Multi-layer validation (technical, aesthetic, brand, platform compliance)
    - AI-powered quality analysis using vision and language models
    - Platform-specific optimization scoring
    - Engagement potential prediction
    - Automated improvement suggestions
    - Quality trend tracking and analytics
    """
    
    def __init__(self, db_session=None):
        # Initialize AI clients
        self._init_ai_clients()
        
        # Database integration
        self.db_manager = None
        if DATABASE_AVAILABLE and db_session:
            self.db_manager = DatabaseManager(db_session)
        
        # Initialize ML models for text analysis
        self._init_ml_models()
        
        # Load platform configurations
        self._load_platform_configs()
        
        # Load quality standards
        self._load_quality_standards()
        
        # Setup temp directories
        self.temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp', 'quality')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Quality tracking
        self.quality_cache = {}
        self.validation_history = []
        
        # Threading setup
        self.validation_lock = threading.Lock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Content Quality Validator initialized")

    def _init_ai_clients(self):
        """Initialize AI clients for quality analysis"""
        try:
            # OpenAI for vision analysis
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except Exception:
            self.openai_client = None
            self.logger.warning("OpenAI client not available")
        
        try:
            # Anthropic for text analysis
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except Exception:
            self.anthropic_client = None
            self.logger.warning("Anthropic client not available")

    def _init_ml_models(self):
        """Initialize ML models for content analysis"""
        try:
            # Sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Text classification for engagement prediction
            self.engagement_predictor = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                return_all_scores=True
            )
            
            self.logger.info("ML models loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"ML models not available: {e}")
            self.sentiment_analyzer = None
            self.engagement_predictor = None

    def _load_platform_configs(self):
        """Load platform-specific quality configurations"""
        self.platform_configs = {
            'instagram': {
                'image_requirements': {
                    'min_resolution': (1080, 1080),
                    'max_resolution': (8192, 8192),
                    'aspect_ratios': [(1, 1), (4, 5), (16, 9)],
                    'min_quality_score': 75.0,
                    'brightness_range': (20, 80),
                    'contrast_range': (30, 90),
                    'sharpness_min': 60.0
                },
                'video_requirements': {
                    'min_resolution': (720, 720),
                    'max_duration': 60,
                    'min_duration': 3,
                    'fps_range': (24, 60),
                    'min_quality_score': 70.0,
                    'audio_required': False
                },
                'text_requirements': {
                    'max_length': 2200,
                    'optimal_length': 125,
                    'hashtag_limit': 30,
                    'emoji_density_max': 0.15,
                    'readability_min': 60.0,
                    'engagement_score_min': 70.0
                },
                'brand_requirements': {
                    'aesthetic_score_min': 80.0,
                    'brand_consistency_min': 75.0,
                    'shopping_suitability_min': 70.0
                }
            },
            'tiktok': {
                'image_requirements': {
                    'min_resolution': (720, 720),
                    'aspect_ratios': [(9, 16), (1, 1)],
                    'min_quality_score': 65.0,
                    'vibrancy_min': 70.0,
                    'trend_appeal_min': 80.0
                },
                'video_requirements': {
                    'min_resolution': (720, 1280),
                    'max_duration': 180,
                    'min_duration': 5,
                    'fps_range': (24, 60),
                    'min_quality_score': 70.0,
                    'audio_required': True,
                    'hook_strength_min': 85.0
                },
                'text_requirements': {
                    'max_length': 2200,
                    'optimal_length': 100,
                    'hashtag_limit': 5,
                    'trending_potential_min': 75.0,
                    'viral_elements_required': True
                },
                'brand_requirements': {
                    'trend_alignment_min': 80.0,
                    'youth_appeal_min': 75.0,
                    'authenticity_score_min': 80.0
                }
            },
            'x': {
                'image_requirements': {
                    'min_resolution': (600, 335),
                    'aspect_ratios': [(16, 9), (2, 1)],
                    'min_quality_score': 60.0,
                    'clarity_min': 70.0
                },
                'video_requirements': {
                    'max_duration': 140,
                    'min_resolution': (720, 480),
                    'fps_range': (24, 30),
                    'min_quality_score': 65.0
                },
                'text_requirements': {
                    'max_length': 280,
                    'optimal_length': 100,
                    'hashtag_limit': 2,
                    'readability_min': 70.0,
                    'shareability_min': 75.0
                },
                'brand_requirements': {
                    'professional_tone_min': 75.0,
                    'credibility_score_min': 80.0
                }
            },
            'linkedin': {
                'image_requirements': {
                    'min_resolution': (1200, 627),
                    'aspect_ratios': [(1.91, 1), (1, 1)],
                    'min_quality_score': 70.0,
                    'professional_appearance_min': 85.0
                },
                'video_requirements': {
                    'max_duration': 300,
                    'min_resolution': (720, 480),
                    'min_quality_score': 75.0,
                    'educational_value_min': 80.0
                },
                'text_requirements': {
                    'max_length': 3000,
                    'optimal_length': 200,
                    'professional_tone_min': 85.0,
                    'business_relevance_min': 80.0
                },
                'brand_requirements': {
                    'professional_standards_min': 90.0,
                    'business_alignment_min': 85.0
                }
            }
        }

    def _load_quality_standards(self):
        """Load quality standards and thresholds"""
        self.quality_standards = {
            'premium': {
                'overall_threshold': 85.0,
                'individual_threshold': 75.0,
                'human_review_threshold': 80.0,
                'weights': {
                    'technical_quality': 0.25,
                    'aesthetic_appeal': 0.25,
                    'brand_compliance': 0.20,
                    'engagement_potential': 0.20,
                    'platform_optimization': 0.10
                }
            },
            'standard': {
                'overall_threshold': 70.0,
                'individual_threshold': 60.0,
                'human_review_threshold': 65.0,
                'weights': {
                    'technical_quality': 0.30,
                    'aesthetic_appeal': 0.20,
                    'brand_compliance': 0.20,
                    'engagement_potential': 0.20,
                    'platform_optimization': 0.10
                }
            },
            'basic': {
                'overall_threshold': 55.0,
                'individual_threshold': 45.0,
                'human_review_threshold': 50.0,
                'weights': {
                    'technical_quality': 0.40,
                    'aesthetic_appeal': 0.15,
                    'brand_compliance': 0.15,
                    'engagement_potential': 0.20,
                    'platform_optimization': 0.10
                }
            }
        }

    async def validate_content(self, content_path: str, content_type: str, 
                             platform: str, config: ValidationConfig,
                             product_data: Optional[Dict[str, Any]] = None) -> ContentQualityResult:
        """
        Comprehensive content quality validation
        
        Args:
            content_path: Path to content file
            content_type: 'image', 'video', or 'text'
            platform: Target platform
            config: Validation configuration
            product_data: Optional product information for context
        
        Returns:
            ContentQualityResult with comprehensive quality analysis
        """
        start_time = time.time()
        content_id = f"{content_type}_{platform}_{int(start_time)}"
        
        self.logger.info(f"Starting quality validation: {content_id}")
        
        try:
            # Initialize result
            result = ContentQualityResult(
                content_id=content_id,
                content_type=content_type,
                platform=platform,
                overall_score=0.0,
                individual_scores=[],
                passed_validation=False,
                requires_human_review=False,
                improvement_suggestions=[],
                retry_recommended=False,
                retry_parameters=None,
                processing_time=0.0,
                created_at=datetime.utcnow(),
                metadata={'config': asdict(config)}
            )
            
            # Run validation based on content type
            if content_type == 'image':
                await self._validate_image(content_path, platform, config, result, product_data)
            elif content_type == 'video':
                await self._validate_video(content_path, platform, config, result, product_data)
            elif content_type == 'text':
                await self._validate_text(content_path, platform, config, result, product_data)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Calculate overall score
            self._calculate_overall_score(result, config)
            
            # Determine validation outcome
            self._determine_validation_outcome(result, config)
            
            # Generate improvement suggestions
            await self._generate_improvement_suggestions(result, config, product_data)
            
            # Store validation result
            if self.db_manager:
                self._store_validation_result(result)
            
            # Calculate processing time
            result.processing_time = time.time() - start_time
            
            self.logger.info(
                f"Quality validation completed: {content_id} - "
                f"Score: {result.overall_score:.1f}, "
                f"Passed: {result.passed_validation}, "
                f"Time: {result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quality validation failed for {content_id}: {e}")
            result.metadata['error'] = str(e)
            result.processing_time = time.time() - start_time
            return result

    async def _validate_image(self, image_path: str, platform: str, 
                            config: ValidationConfig, result: ContentQualityResult,
                            product_data: Optional[Dict[str, Any]]):
        """Comprehensive image quality validation"""
        platform_reqs = self.platform_configs[platform]['image_requirements']
        
        # Load image
        try:
            image = Image.open(image_path)
            image_cv = cv2.imread(image_path)
        except Exception as e:
            raise ValueError(f"Cannot load image: {e}")
        
        # Technical quality analysis
        if config.enable_technical_analysis:
            tech_score = await self._analyze_image_technical_quality(image, image_cv, platform_reqs)
            result.individual_scores.append(tech_score)
        
        # Aesthetic analysis
        aesthetic_score = await self._analyze_image_aesthetics(image, image_cv, platform_reqs)
        result.individual_scores.append(aesthetic_score)
        
        # Brand compliance (if AI available)
        if config.enable_brand_compliance and self.openai_client:
            brand_score = await self._analyze_image_brand_compliance(image_path, platform, product_data)
            result.individual_scores.append(brand_score)
        
        # Platform optimization
        platform_score = await self._analyze_image_platform_optimization(image, platform, platform_reqs)
        result.individual_scores.append(platform_score)
        
        # Engagement prediction (if AI available)
        if config.enable_engagement_prediction and self.openai_client:
            engagement_score = await self._predict_image_engagement(image_path, platform, product_data)
            result.individual_scores.append(engagement_score)

    async def _analyze_image_technical_quality(self, image: Image.Image, 
                                             image_cv: np.ndarray, 
                                             requirements: Dict[str, Any]) -> QualityScore:
        """Analyze technical image quality metrics"""
        details = {}
        suggestions = []
        
        # Resolution analysis
        width, height = image.size
        details['resolution'] = (width, height)
        min_w, min_h = requirements['min_resolution']
        
        resolution_score = 100.0
        if width < min_w or height < min_h:
            resolution_score = max(0, 100 * min(width/min_w, height/min_h))
            suggestions.append(f"Increase resolution to at least {min_w}x{min_h}")
        
        # Aspect ratio analysis
        current_ratio = width / height
        valid_ratios = requirements.get('aspect_ratios', [(1, 1)])
        ratio_scores = []
        
        for ratio_w, ratio_h in valid_ratios:
            target_ratio = ratio_w / ratio_h
            ratio_diff = abs(current_ratio - target_ratio) / target_ratio
            ratio_score = max(0, 100 * (1 - ratio_diff))
            ratio_scores.append(ratio_score)
        
        aspect_ratio_score = max(ratio_scores) if ratio_scores else 50.0
        details['aspect_ratio'] = current_ratio
        details['valid_ratios'] = valid_ratios
        
        if aspect_ratio_score < 80:
            best_ratio = valid_ratios[ratio_scores.index(max(ratio_scores))]
            suggestions.append(f"Adjust aspect ratio to {best_ratio[0]}:{best_ratio[1]}")
        
        # Sharpness analysis (Laplacian variance)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 100 * 100)  # Normalize
        details['sharpness'] = laplacian_var
        
        if sharpness_score < requirements.get('sharpness_min', 50):
            suggestions.append("Increase image sharpness and focus")
        
        # Brightness analysis
        brightness = np.mean(cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)[:,:,2])
        brightness_range = requirements.get('brightness_range', (20, 80))
        
        if brightness_range[0] <= brightness <= brightness_range[1]:
            brightness_score = 100.0
        else:
            brightness_score = max(0, 100 - abs(brightness - np.mean(brightness_range)) * 2)
        
        details['brightness'] = brightness
        if brightness_score < 80:
            if brightness < brightness_range[0]:
                suggestions.append("Increase image brightness")
            else:
                suggestions.append("Reduce image brightness")
        
        # Contrast analysis
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        contrast_range = requirements.get('contrast_range', (30, 90))
        
        if contrast_range[0] <= contrast <= contrast_range[1]:
            contrast_score = 100.0
        else:
            contrast_score = max(0, 100 - abs(contrast - np.mean(contrast_range)) * 1.5)
        
        details['contrast'] = contrast
        if contrast_score < 80:
            if contrast < contrast_range[0]:
                suggestions.append("Increase image contrast")
            else:
                suggestions.append("Reduce image contrast")
        
        # Calculate overall technical score
        technical_score = np.mean([
            resolution_score,
            aspect_ratio_score,
            sharpness_score,
            brightness_score,
            contrast_score
        ])
        
        return QualityScore(
            metric_name="technical_quality",
            score=technical_score,
            weight=0.25,
            threshold=requirements.get('min_quality_score', 60.0),
            passed=technical_score >= requirements.get('min_quality_score', 60.0),
            details=details,
            suggestions=suggestions
        )

    async def _analyze_image_aesthetics(self, image: Image.Image, 
                                      image_cv: np.ndarray,
                                      requirements: Dict[str, Any]) -> QualityScore:
        """Analyze aesthetic appeal of image"""
        details = {}
        suggestions = []
        
        # Color harmony analysis
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hue_peaks = len([i for i, v in enumerate(hue_hist) if v > np.mean(hue_hist) * 2])
        
        color_harmony_score = min(100, 100 - hue_peaks * 10)  # Fewer dominant colors = better harmony
        details['color_harmony'] = color_harmony_score
        
        # Saturation analysis
        saturation = np.mean(hsv[:,:,1])
        vibrancy_score = min(100, saturation / 255 * 100)
        details['vibrancy'] = vibrancy_score
        
        if vibrancy_score < requirements.get('vibrancy_min', 50):
            suggestions.append("Increase color saturation for more visual appeal")
        
        # Rule of thirds analysis (simple grid-based)
        height, width = image_cv.shape[:2]
        grid_x = [width//3, 2*width//3]
        grid_y = [height//3, 2*height//3]
        
        # Calculate edge density at rule of thirds lines
        edges = cv2.Canny(cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY), 50, 150)
        
        roi_score = 0
        for x in grid_x:
            roi_score += np.sum(edges[:, x-5:x+5])
        for y in grid_y:
            roi_score += np.sum(edges[y-5:y+5, :])
        
        composition_score = min(100, roi_score / (width * height) * 10000)
        details['composition'] = composition_score
        
        if composition_score < 50:
            suggestions.append("Improve composition using rule of thirds")
        
        # Overall aesthetic score
        aesthetic_score = np.mean([
            color_harmony_score,
            vibrancy_score,
            composition_score
        ])
        
        return QualityScore(
            metric_name="aesthetic_appeal",
            score=aesthetic_score,
            weight=0.25,
            threshold=70.0,
            passed=aesthetic_score >= 70.0,
            details=details,
            suggestions=suggestions
        )

    async def _analyze_image_brand_compliance(self, image_path: str, platform: str,
                                            product_data: Optional[Dict[str, Any]]) -> QualityScore:
        """AI-powered brand compliance analysis"""
        if not self.openai_client:
            return self._create_fallback_score("brand_compliance", 75.0)
        
        details = {}
        suggestions = []
        
        try:
            # Encode image for AI analysis
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create analysis prompt
            brand_voice = product_data.get('brand_voice', 'professional') if product_data else 'professional'
            category = product_data.get('category', 'general') if product_data else 'general'
            
            analysis_prompt = f"""
            Analyze this product image for brand compliance and marketing effectiveness.
            
            Context:
            - Platform: {platform}
            - Brand Voice: {brand_voice}
            - Category: {category}
            
            Evaluate and rate 0-100:
            1. Brand consistency and professional appearance
            2. Visual appeal for target platform
            3. Product presentation quality
            4. Marketing effectiveness
            5. Platform appropriateness
            
            Return JSON with scores and specific suggestions for improvement.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Parse AI response
            ai_analysis = response.choices[0].message.content
            
            # Extract scores (simple parsing - could be improved)
            import re
            scores = re.findall(r'(\d+(?:\.\d+)?)', ai_analysis)
            
            if scores:
                brand_score = float(scores[0]) if scores else 75.0
                details['ai_analysis'] = ai_analysis
                details['brand_consistency'] = brand_score
                
                # Extract suggestions from AI response
                if "suggest" in ai_analysis.lower():
                    ai_suggestions = ai_analysis.split("suggest")[-1].strip()
                    suggestions.append(f"AI recommendation: {ai_suggestions[:200]}")
            else:
                brand_score = 75.0  # Fallback
                
        except Exception as e:
            self.logger.error(f"AI brand analysis failed: {e}")
            brand_score = 75.0  # Fallback
            suggestions.append("AI analysis unavailable - manual brand review recommended")
        
        return QualityScore(
            metric_name="brand_compliance",
            score=brand_score,
            weight=0.20,
            threshold=75.0,
            passed=brand_score >= 75.0,
            details=details,
            suggestions=suggestions
        )

    async def _analyze_image_platform_optimization(self, image: Image.Image, 
                                                 platform: str,
                                                 requirements: Dict[str, Any]) -> QualityScore:
        """Analyze platform-specific optimization"""
        details = {}
        suggestions = []
        
        platform_score = 100.0
        
        # Platform-specific checks
        if platform == 'instagram':
            # Check for Instagram-specific requirements
            aesthetic_min = requirements.get('aesthetic_score_min', 80.0)
            # This would be calculated based on aesthetic metrics
            aesthetic_actual = 85.0  # Placeholder - would use actual aesthetic analysis
            
            if aesthetic_actual < aesthetic_min:
                platform_score -= 20
                suggestions.append("Improve aesthetic appeal for Instagram audience")
                
        elif platform == 'tiktok':
            # Check for TikTok-specific requirements
            trend_appeal = requirements.get('trend_appeal_min', 80.0)
            # This would analyze trendy visual elements
            trend_actual = 75.0  # Placeholder
            
            if trend_actual < trend_appeal:
                platform_score -= 15
                suggestions.append("Add more trendy visual elements for TikTok")
                
        elif platform == 'linkedin':
            # Check for LinkedIn professional standards
            professional_min = requirements.get('professional_appearance_min', 85.0)
            professional_actual = 88.0  # Placeholder
            
            if professional_actual < professional_min:
                platform_score -= 25
                suggestions.append("Enhance professional appearance for LinkedIn")
        
        details['platform_optimization'] = platform_score
        details['platform_requirements'] = requirements
        
        return QualityScore(
            metric_name="platform_optimization",
            score=platform_score,
            weight=0.10,
            threshold=80.0,
            passed=platform_score >= 80.0,
            details=details,
            suggestions=suggestions
        )

    async def _predict_image_engagement(self, image_path: str, platform: str,
                                      product_data: Optional[Dict[str, Any]]) -> QualityScore:
        """Predict engagement potential using AI"""
        if not self.openai_client:
            return self._create_fallback_score("engagement_potential", 70.0)
        
        details = {}
        suggestions = []
        
        try:
            # Encode image for AI analysis
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            engagement_prompt = f"""
            Analyze this image for engagement potential on {platform}.
            
            Consider:
            1. Visual appeal and stopping power
            2. Emotional impact
            3. Clarity of product presentation
            4. Platform-specific engagement factors
            5. Call-to-action potential
            
            Rate engagement potential 0-100 and provide specific suggestions.
            Focus on what makes content viral and engaging on {platform}.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": engagement_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                max_tokens=400
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract engagement score
            import re
            scores = re.findall(r'(\d+(?:\.\d+)?)', ai_response)
            engagement_score = float(scores[0]) if scores else 70.0
            
            details['ai_engagement_analysis'] = ai_response
            details['predicted_engagement'] = engagement_score
            
            # Extract suggestions
            if "suggest" in ai_response.lower() or "recommend" in ai_response.lower():
                suggestions.append(f"Engagement optimization: {ai_response[-200:]}")
                
        except Exception as e:
            self.logger.error(f"Engagement prediction failed: {e}")
            engagement_score = 70.0
            suggestions.append("Engagement analysis unavailable - consider A/B testing")
        
        return QualityScore(
            metric_name="engagement_potential",
            score=engagement_score,
            weight=0.20,
            threshold=70.0,
            passed=engagement_score >= 70.0,
            details=details,
            suggestions=suggestions
        )

    async def _validate_video(self, video_path: str, platform: str, 
                            config: ValidationConfig, result: ContentQualityResult,
                            product_data: Optional[Dict[str, Any]]):
        """Comprehensive video quality validation"""
        platform_reqs = self.platform_configs[platform]['video_requirements']
        
        # Load video
        try:
            video = VideoFileClip(video_path)
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            raise ValueError(f"Cannot load video: {e}")
        
        # Technical quality analysis
        if config.enable_technical_analysis:
            tech_score = await self._analyze_video_technical_quality(video, cap, platform_reqs)
            result.individual_scores.append(tech_score)
        
        # Content analysis
        content_score = await self._analyze_video_content_quality(video, platform_reqs, platform)
        result.individual_scores.append(content_score)
        
        # Platform optimization
        platform_score = await self._analyze_video_platform_optimization(video, platform, platform_reqs)
        result.individual_scores.append(platform_score)
        
        # Engagement prediction
        if config.enable_engagement_prediction:
            engagement_score = await self._predict_video_engagement(video_path, platform, product_data)
            result.individual_scores.append(engagement_score)
        
        # Cleanup
        video.close()
        cap.release()

    async def _analyze_video_technical_quality(self, video: VideoFileClip, 
                                             cap: cv2.VideoCapture,
                                             requirements: Dict[str, Any]) -> QualityScore:
        """Analyze technical video quality metrics"""
        details = {}
        suggestions = []
        
        # Resolution analysis
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        details['resolution'] = (width, height)
        
        min_w, min_h = requirements['min_resolution']
        resolution_score = 100.0
        if width < min_w or height < min_h:
            resolution_score = max(0, 100 * min(width/min_w, height/min_h))
            suggestions.append(f"Increase video resolution to at least {min_w}x{min_h}")
        
        # Duration analysis
        duration = video.duration
        details['duration'] = duration
        
        min_duration = requirements.get('min_duration', 3)
        max_duration = requirements.get('max_duration', 60)
        
        duration_score = 100.0
        if duration < min_duration:
            duration_score = max(0, 100 * duration / min_duration)
            suggestions.append(f"Increase video duration to at least {min_duration} seconds")
        elif duration > max_duration:
            duration_score = max(0, 100 * max_duration / duration)
            suggestions.append(f"Reduce video duration to maximum {max_duration} seconds")
        
        # Frame rate analysis
        fps = cap.get(cv2.CAP_PROP_FPS)
        details['fps'] = fps
        
        fps_range = requirements.get('fps_range', (24, 60))
        if fps_range[0] <= fps <= fps_range[1]:
            fps_score = 100.0
        else:
            fps_score = max(0, 100 - abs(fps - np.mean(fps_range)) * 2)
        
        if fps_score < 80:
            suggestions.append(f"Adjust frame rate to {fps_range[0]}-{fps_range[1]} fps")
        
        # Audio analysis (if required)
        audio_score = 100.0
        if requirements.get('audio_required', False):
            if video.audio is None:
                audio_score = 0.0
                suggestions.append("Add audio track - required for this platform")
            else:
                # Basic audio quality check
                audio_duration = video.audio.duration
                if abs(audio_duration - duration) > 0.5:
                    audio_score = 80.0
                    suggestions.append("Sync audio with video duration")
        
        details['audio_present'] = video.audio is not None
        
        # Overall technical score
        technical_score = np.mean([
            resolution_score,
            duration_score,
            fps_score,
            audio_score
        ])
        
        return QualityScore(
            metric_name="technical_quality",
            score=technical_score,
            weight=0.30,
            threshold=requirements.get('min_quality_score', 65.0),
            passed=technical_score >= requirements.get('min_quality_score', 65.0),
            details=details,
            suggestions=suggestions
        )

    async def _analyze_video_content_quality(self, video: VideoFileClip, 
                                           requirements: Dict[str, Any],
                                           platform: str) -> QualityScore:
        """Analyze video content quality and appeal"""
        details = {}
        suggestions = []
        
        # Sample frames for analysis
        sample_times = [video.duration * i / 5 for i in range(5)]
        frame_qualities = []
        
        for t in sample_times:
            try:
                frame = video.get_frame(t)
                # Convert to OpenCV format
                frame_cv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Analyze frame quality
                gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)
                
                # Sharpness
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Brightness
                brightness = np.mean(gray)
                
                # Motion detection (if previous frame available)
                motion_score = 50.0  # Placeholder
                
                frame_quality = min(100, (sharpness/100 + brightness/255*100 + motion_score) / 3)
                frame_qualities.append(frame_quality)
                
            except Exception:
                continue
        
        content_quality_score = np.mean(frame_qualities) if frame_qualities else 70.0
        details['average_frame_quality'] = content_quality_score
        details['analyzed_frames'] = len(frame_qualities)
        
        # Platform-specific content analysis
        if platform == 'tiktok':
            hook_strength = requirements.get('hook_strength_min', 85.0)
            # Analyze first 3 seconds for hook strength
            hook_score = 80.0  # Placeholder - would analyze first few seconds
            
            if hook_score < hook_strength:
                suggestions.append("Strengthen opening hook - crucial for TikTok engagement")
                content_quality_score = min(content_quality_score, hook_score)
                
        elif platform == 'linkedin':
            educational_value = requirements.get('educational_value_min', 80.0)
            # Would analyze for educational/professional content
            edu_score = 75.0  # Placeholder
            
            if edu_score < educational_value:
                suggestions.append("Increase educational value for LinkedIn audience")
                content_quality_score = min(content_quality_score, edu_score)
        
        if content_quality_score < 70:
            suggestions.append("Improve overall video quality and visual appeal")
        
        return QualityScore(
            metric_name="content_quality",
            score=content_quality_score,
            weight=0.25,
            threshold=70.0,
            passed=content_quality_score >= 70.0,
            details=details,
            suggestions=suggestions
        )

    async def _analyze_video_platform_optimization(self, video: VideoFileClip, 
                                                 platform: str,
                                                 requirements: Dict[str, Any]) -> QualityScore:
        """Analyze video optimization for specific platform"""
        details = {}
        suggestions = []
        
        platform_score = 100.0
        
        # Platform-specific optimization checks
        if platform == 'tiktok':
            # Check vertical format preference
            if hasattr(video, 'size'):
                width, height = video.size
                aspect_ratio = width / height
                
                if aspect_ratio < 0.8:  # More vertical is better for TikTok
                    platform_score = max(platform_score, 90)
                elif aspect_ratio > 1.2:  # Horizontal is less optimal
                    platform_score = min(platform_score, 70)
                    suggestions.append("Consider vertical format for better TikTok performance")
        
        elif platform == 'instagram':
            # Instagram prefers square or slightly vertical
            if hasattr(video, 'size'):
                width, height = video.size
                aspect_ratio = width / height
                
                if 0.8 <= aspect_ratio <= 1.3:  # Good for Instagram
                    platform_score = max(platform_score, 95)
                else:
                    platform_score = min(platform_score, 75)
                    suggestions.append("Optimize aspect ratio for Instagram (square or 4:5)")
        
        # Add more platform-specific checks
        details['platform_optimization_score'] = platform_score
        
        return QualityScore(
            metric_name="platform_optimization",
            score=platform_score,
            weight=0.15,
            threshold=75.0,
            passed=platform_score >= 75.0,
            details=details,
            suggestions=suggestions
        )

    async def _predict_video_engagement(self, video_path: str, platform: str,
                                      product_data: Optional[Dict[str, Any]]) -> QualityScore:
        """Predict video engagement potential"""
        # For now, return a placeholder score
        # In production, this would use AI video analysis
        
        details = {
            'prediction_method': 'heuristic',
            'platform': platform,
            'analysis_note': 'Full AI video analysis not implemented yet'
        }
        
        # Basic heuristic scoring based on platform
        base_score = 75.0
        if platform == 'tiktok':
            base_score = 80.0  # TikTok generally has higher engagement
        elif platform == 'linkedin':
            base_score = 70.0  # More professional, lower viral potential
        
        suggestions = [
            f"Optimize for {platform} audience engagement patterns",
            "Consider A/B testing different video styles",
            "Add captions for accessibility and silent viewing"
        ]
        
        return QualityScore(
            metric_name="engagement_potential",
            score=base_score,
            weight=0.20,
            threshold=70.0,
            passed=base_score >= 70.0,
            details=details,
            suggestions=suggestions
        )

    async def _validate_text(self, text_content: str, platform: str, 
                           config: ValidationConfig, result: ContentQualityResult,
                           product_data: Optional[Dict[str, Any]]):
        """Comprehensive text content validation"""
        platform_reqs = self.platform_configs[platform]['text_requirements']
        
        # If text_content is a file path, read the content
        if os.path.exists(text_content):
            with open(text_content, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = text_content
        
        # Technical text analysis
        tech_score = await self._analyze_text_technical_quality(text, platform_reqs)
        result.individual_scores.append(tech_score)
        
        # Readability analysis
        readability_score = await self._analyze_text_readability(text, platform_reqs)
        result.individual_scores.append(readability_score)
        
        # Engagement analysis
        if config.enable_engagement_prediction:
            engagement_score = await self._analyze_text_engagement_potential(text, platform, platform_reqs)
            result.individual_scores.append(engagement_score)
        
        # Brand compliance
        if config.enable_brand_compliance:
            brand_score = await self._analyze_text_brand_compliance(text, platform, product_data)
            result.individual_scores.append(brand_score)
        
        # Platform optimization
        platform_score = await self._analyze_text_platform_optimization(text, platform, platform_reqs)
        result.individual_scores.append(platform_score)

    async def _analyze_text_technical_quality(self, text: str, 
                                            requirements: Dict[str, Any]) -> QualityScore:
        """Analyze technical text quality metrics"""
        details = {}
        suggestions = []
        
        # Length analysis
        text_length = len(text)
        max_length = requirements['max_length']
        optimal_length = requirements.get('optimal_length', max_length // 2)
        
        details['character_count'] = text_length
        details['word_count'] = len(text.split())
        
        if text_length > max_length:
            length_score = max(0, 100 - (text_length - max_length) / max_length * 100)
            suggestions.append(f"Reduce text length to under {max_length} characters")
        elif text_length < optimal_length * 0.5:
            length_score = max(0, 100 * text_length / (optimal_length * 0.5))
            suggestions.append(f"Consider expanding content closer to {optimal_length} characters")
        else:
            length_score = 100.0
        
        # Hashtag analysis
        hashtags = re.findall(r'#\w+', text)
        hashtag_count = len(hashtags)
        hashtag_limit = requirements.get('hashtag_limit', 5)
        
        details['hashtag_count'] = hashtag_count
        details['hashtags'] = hashtags
        
        if hashtag_count > hashtag_limit:
            hashtag_score = max(0, 100 - (hashtag_count - hashtag_limit) * 10)
            suggestions.append(f"Reduce hashtags to {hashtag_limit} or fewer")
        else:
            hashtag_score = 100.0
        
        # Emoji analysis
        emoji_count = len([char for char in text if char in emoji.UNICODE_EMOJI['en']])
        emoji_density = emoji_count / len(text) if text else 0
        max_emoji_density = requirements.get('emoji_density_max', 0.1)
        
        details['emoji_count'] = emoji_count
        details['emoji_density'] = emoji_density
        
        if emoji_density > max_emoji_density:
            emoji_score = max(0, 100 - (emoji_density - max_emoji_density) * 500)
            suggestions.append("Reduce emoji usage for better readability")
        else:
            emoji_score = 100.0
        
        # Grammar and spelling check (basic)
        grammar_score = 90.0  # Placeholder - would use proper grammar checker
        
        # Technical quality score
        technical_score = np.mean([
            length_score,
            hashtag_score,
            emoji_score,
            grammar_score
        ])
        
        return QualityScore(
            metric_name="technical_quality",
            score=technical_score,
            weight=0.25,
            threshold=80.0,
            passed=technical_score >= 80.0,
            details=details,
            suggestions=suggestions
        )

    async def _analyze_text_readability(self, text: str, 
                                      requirements: Dict[str, Any]) -> QualityScore:
        """Analyze text readability and clarity"""
        details = {}
        suggestions = []
        
        try:
            # Flesch Reading Ease score
            flesch_score = flesch_reading_ease(text)
            details['flesch_score'] = flesch_score
            
            # Dale-Chall Readability score
            dale_chall = dale_chall_readability_score(text)
            details['dale_chall_score'] = dale_chall
            
            # Convert Flesch score to 0-100 scale (higher is better)
            if flesch_score >= 90:
                readability_score = 100.0
            elif flesch_score >= 80:
                readability_score = 90.0
            elif flesch_score >= 70:
                readability_score = 80.0
            elif flesch_score >= 60:
                readability_score = 70.0
            else:
                readability_score = max(0, flesch_score)
            
            # Check against requirements
            min_readability = requirements.get('readability_min', 60.0)
            if readability_score < min_readability:
                suggestions.append("Simplify language and sentence structure for better readability")
            
        except Exception as e:
            self.logger.warning(f"Readability analysis failed: {e}")
            readability_score = 70.0  # Fallback
            suggestions.append("Manual readability review recommended")
        
        # Sentence length analysis
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        details['average_sentence_length'] = avg_sentence_length
        
        if avg_sentence_length > 20:
            suggestions.append("Consider shorter sentences for better readability")
            readability_score = min(readability_score, 80.0)
        
        return QualityScore(
            metric_name="readability",
            score=readability_score,
            weight=0.20,
            threshold=min_readability,
            passed=readability_score >= min_readability,
            details=details,
            suggestions=suggestions
        )

    async def _analyze_text_engagement_potential(self, text: str, platform: str,
                                               requirements: Dict[str, Any]) -> QualityScore:
        """Analyze text engagement potential using ML models"""
        details = {}
        suggestions = []
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment_result = self.sentiment_analyzer(text)[0]
                sentiment_score = max([score['score'] for score in sentiment_result])
                sentiment_label = max(sentiment_result, key=lambda x: x['score'])['label']
                
                details['sentiment'] = sentiment_label
                details['sentiment_confidence'] = sentiment_score
                
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed: {e}")
                sentiment_score = 0.7
                sentiment_label = "neutral"
        else:
            sentiment_score = 0.7
            sentiment_label = "neutral"
        
        # Engagement keywords analysis
        engagement_words = [
            'amazing', 'incredible', 'must-have', 'exclusive', 'limited',
            'breakthrough', 'revolutionary', 'game-changer', 'innovative',
            'discover', 'unlock', 'transform', 'achieve', 'succeed'
        ]
        
        text_lower = text.lower()
        engagement_word_count = sum(1 for word in engagement_words if word in text_lower)
        engagement_density = engagement_word_count / len(text.split()) if text else 0
        
        details['engagement_words_found'] = engagement_word_count
        details['engagement_density'] = engagement_density
        
        # Call-to-action analysis
        cta_patterns = [
            r'\bclick\b', r'\bbuy\b', r'\bshop\b', r'\btry\b', r'\bget\b',
            r'\border\b', r'\blearn more\b', r'\bsign up\b', r'\bdownload\b'
        ]
        
        cta_found = any(re.search(pattern, text_lower) for pattern in cta_patterns)
        details['has_call_to_action'] = cta_found
        
        # Platform-specific engagement factors
        platform_bonus = 0
        if platform == 'tiktok':
            # Check for trending hashtags and viral elements
            viral_elements = ['viral', 'trending', 'fyp', 'challenge']
            if any(element in text_lower for element in viral_elements):
                platform_bonus = 10
                details['viral_elements_found'] = True
            
            if requirements.get('viral_elements_required', False) and platform_bonus == 0:
                suggestions.append("Add viral elements like #fyp or trending hashtags")
        
        elif platform == 'linkedin':
            # Check for professional language
            professional_words = ['professional', 'business', 'career', 'industry', 'expertise']
            if any(word in text_lower for word in professional_words):
                platform_bonus = 10
                details['professional_elements_found'] = True
        
        # Calculate engagement score
        base_engagement = sentiment_score * 100
        engagement_word_bonus = min(20, engagement_word_count * 5)
        cta_bonus = 10 if cta_found else 0
        
        engagement_score = min(100, base_engagement + engagement_word_bonus + cta_bonus + platform_bonus)
        
        # Check against requirements
        min_engagement = requirements.get('engagement_score_min', 70.0)
        if engagement_score < min_engagement:
            suggestions.extend([
                "Add more engaging language and emotional triggers",
                "Include a clear call-to-action",
                "Use power words that drive engagement"
            ])
        
        if not cta_found:
            suggestions.append("Add a clear call-to-action to drive user engagement")
        
        return QualityScore(
            metric_name="engagement_potential",
            score=engagement_score,
            weight=0.25,
            threshold=min_engagement,
            passed=engagement_score >= min_engagement,
            details=details,
            suggestions=suggestions
        )

    async def _analyze_text_brand_compliance(self, text: str, platform: str,
                                           product_data: Optional[Dict[str, Any]]) -> QualityScore:
        """Analyze brand compliance and voice consistency"""
        details = {}
        suggestions = []
        
        brand_score = 80.0  # Default
        
        if product_data:
            brand_voice = product_data.get('brand_voice', 'professional')
            target_audience = product_data.get('target_audience', 'general')
            
            details['expected_brand_voice'] = brand_voice
            details['target_audience'] = target_audience
            
            # Analyze tone matching
            if brand_voice == 'professional':
                # Check for professional language
                if any(word in text.lower() for word in ['yo', 'lol', 'omg', 'wtf']):
                    brand_score -= 20
                    suggestions.append("Use more professional language")
                
            elif brand_voice == 'casual':
                # Check for overly formal language
                formal_words = ['furthermore', 'subsequently', 'nevertheless']
                if any(word in text.lower() for word in formal_words):
                    brand_score -= 10
                    suggestions.append("Use more casual, conversational language")
            
            elif brand_voice == 'trendy':
                # Check for trendy language
                if not any(emoji_char in text for emoji_char in emoji.UNICODE_EMOJI['en']):
                    brand_score -= 10
                    suggestions.append("Consider adding emojis for trendy appeal")
        
        # Check for brand consistency issues
        controversial_topics = ['politics', 'religion', 'controversial']
        if any(topic in text.lower() for topic in controversial_topics):
            brand_score -= 30
            suggestions.append("Avoid controversial topics to maintain brand safety")
        
        details['brand_compliance_score'] = brand_score
        
        return QualityScore(
            metric_name="brand_compliance",
            score=brand_score,
            weight=0.15,
            threshold=75.0,
            passed=brand_score >= 75.0,
            details=details,
            suggestions=suggestions
        )

    async def _analyze_text_platform_optimization(self, text: str, platform: str,
                                                requirements: Dict[str, Any]) -> QualityScore:
        """Analyze platform-specific text optimization"""
        details = {}
        suggestions = []
        
        platform_score = 100.0
        
        if platform == 'x':
            # Twitter/X specific optimization
            shareability_min = requirements.get('shareability_min', 75.0)
            
            # Check for thread potential
            if len(text) > 200:
                suggestions.append("Consider breaking into a thread for better engagement")
                platform_score = min(platform_score, 85)
            
            # Check for hashtag strategy
            hashtags = re.findall(r'#\w+', text)
            if len(hashtags) > 2:
                platform_score = min(platform_score, 80)
                suggestions.append("Limit hashtags to 1-2 for Twitter optimization")
            
        elif platform == 'tiktok':
            # TikTok specific optimization
            trending_min = requirements.get('trending_potential_min', 75.0)
            
            # Check for trending hashtags
            trending_hashtags = ['#fyp', '#viral', '#trending', '#foryou']
            if not any(hashtag in text.lower() for hashtag in trending_hashtags):
                platform_score = min(platform_score, 70)
                suggestions.append("Add trending hashtags like #fyp for TikTok visibility")
            
        elif platform == 'linkedin':
            # LinkedIn specific optimization
            professional_min = requirements.get('professional_tone_min', 85.0)
            business_relevance_min = requirements.get('business_relevance_min', 80.0)
            
            # Check for business relevance
            business_keywords = ['business', 'professional', 'career', 'industry', 'leadership']
            if not any(keyword in text.lower() for keyword in business_keywords):
                platform_score = min(platform_score, 75)
                suggestions.append("Add business-relevant keywords for LinkedIn audience")
        
        details['platform_optimization_score'] = platform_score
        details['platform_specific_analysis'] = f"Optimized for {platform}"
        
        return QualityScore(
            metric_name="platform_optimization",
            score=platform_score,
            weight=0.15,
            threshold=80.0,
            passed=platform_score >= 80.0,
            details=details,
            suggestions=suggestions
        )

    def _create_fallback_score(self, metric_name: str, score: float) -> QualityScore:
        """Create fallback quality score when AI analysis is unavailable"""
        return QualityScore(
            metric_name=metric_name,
            score=score,
            weight=0.20,
            threshold=70.0,
            passed=score >= 70.0,
            details={'analysis_method': 'fallback'},
            suggestions=["AI analysis unavailable - manual review recommended"]
        )

    def _calculate_overall_score(self, result: ContentQualityResult, config: ValidationConfig):
        """Calculate weighted overall quality score"""
        if not result.individual_scores:
            result.overall_score = 0.0
            return
        
        quality_tier = config.quality_tier
        weights = self.quality_standards[quality_tier]['weights']
        
        weighted_scores = []
        total_weight = 0
        
        for score in result.individual_scores:
            weight = weights.get(score.metric_name, score.weight)
            weighted_scores.append(score.score * weight)
            total_weight += weight
        
        result.overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0

    def _determine_validation_outcome(self, result: ContentQualityResult, config: ValidationConfig):
        """Determine validation outcome and recommendations"""
        quality_tier = config.quality_tier
        standards = self.quality_standards[quality_tier]
        
        # Check overall threshold
        overall_threshold = standards['overall_threshold']
        result.passed_validation = result.overall_score >= overall_threshold
        
        # Check individual thresholds
        individual_threshold = standards['individual_threshold']
        failed_metrics = [s for s in result.individual_scores if s.score < individual_threshold]
        
        if failed_metrics:
            result.passed_validation = False
            result.improvement_suggestions.extend([
                f"Improve {metric.metric_name}: {metric.score:.1f} < {individual_threshold}"
                for metric in failed_metrics
            ])
        
        # Human review threshold
        human_review_threshold = standards['human_review_threshold']
        if result.overall_score < human_review_threshold:
            result.requires_human_review = True
        
        # Retry recommendation
        if not result.passed_validation and result.overall_score > config.auto_reject_threshold:
            result.retry_recommended = True
            result.retry_parameters = self._generate_retry_parameters(result, failed_metrics)

    def _generate_retry_parameters(self, result: ContentQualityResult, 
                                 failed_metrics: List[QualityScore]) -> Dict[str, Any]:
        """Generate parameters for retry attempt"""
        retry_params = {
            'focus_areas': [metric.metric_name for metric in failed_metrics],
            'suggested_improvements': [],
            'parameter_adjustments': {}
        }
        
        for metric in failed_metrics:
            retry_params['suggested_improvements'].extend(metric.suggestions)
            
            # Specific parameter adjustments based on failed metrics
            if metric.metric_name == 'technical_quality':
                if 'resolution' in str(metric.details):
                    retry_params['parameter_adjustments']['increase_resolution'] = True
                if 'brightness' in str(metric.details):
                    retry_params['parameter_adjustments']['adjust_brightness'] = True
                    
            elif metric.metric_name == 'engagement_potential':
                retry_params['parameter_adjustments']['enhance_engagement_elements'] = True
                
            elif metric.metric_name == 'brand_compliance':
                retry_params['parameter_adjustments']['strengthen_brand_voice'] = True
        
        return retry_params

    async def _generate_improvement_suggestions(self, result: ContentQualityResult, 
                                              config: ValidationConfig,
                                              product_data: Optional[Dict[str, Any]]):
        """Generate AI-powered improvement suggestions"""
        if not self.anthropic_client:
            return
        
        try:
            # Compile analysis data
            analysis_summary = {
                'overall_score': result.overall_score,
                'failed_metrics': [s.metric_name for s in result.individual_scores if not s.passed],
                'platform': result.platform,
                'content_type': result.content_type,
                'quality_tier': config.quality_tier
            }
            
            # Create improvement prompt
            improvement_prompt = f"""
            Analyze this content quality assessment and provide specific, actionable improvement suggestions:
            
            Content Type: {result.content_type}
            Platform: {result.platform}
            Overall Score: {result.overall_score:.1f}/100
            Quality Tier: {config.quality_tier}
            
            Failed Metrics: {', '.join(analysis_summary['failed_metrics'])}
            
            Individual Scores:
            {chr(10).join([f"- {s.metric_name}: {s.score:.1f}/100 ({'PASS' if s.passed else 'FAIL'})" for s in result.individual_scores])}
            
            Provide 3-5 specific, actionable suggestions to improve this content for {result.platform}.
            Focus on the failed metrics and practical steps the creator can take.
            """
            
            client, model = create_client('claude-3-5-sonnet-20241022')
            
            ai_suggestions, _ = get_response_from_llm(
                msg=improvement_prompt,
                client=client,
                model=model,
                system_message="You are an expert content quality consultant. Provide specific, actionable improvement suggestions.",
                print_debug=False
            )
            
            # Parse and add AI suggestions
            if ai_suggestions:
                result.improvement_suggestions.append(f"AI Recommendations: {ai_suggestions}")
                
        except Exception as e:
            self.logger.error(f"AI improvement suggestions failed: {e}")

    def _store_validation_result(self, result: ContentQualityResult):
        """Store validation result in database for analytics"""
        if not self.db_manager:
            return
        
        try:
            # In production, this would store validation results
            # for quality trend analysis and performance tracking
            self.validation_history.append({
                'content_id': result.content_id,
                'platform': result.platform,
                'content_type': result.content_type,
                'overall_score': result.overall_score,
                'passed': result.passed_validation,
                'timestamp': result.created_at,
                'processing_time': result.processing_time
            })
            
            self.logger.info(f"Validation result stored: {result.content_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store validation result: {e}")

    async def batch_validate_content(self, content_items: List[Dict[str, Any]], 
                                   config: ValidationConfig) -> List[ContentQualityResult]:
        """Validate multiple content items in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit validation tasks
            future_to_item = {}
            
            for item in content_items:
                future = executor.submit(
                    asyncio.run,
                    self.validate_content(
                        item['content_path'],
                        item['content_type'],
                        item['platform'],
                        config,
                        item.get('product_data')
                    )
                )
                future_to_item[future] = item
            
            # Collect results
            for future in as_completed(future_to_item):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch validation item failed: {e}")
                    # Create error result
                    error_result = ContentQualityResult(
                        content_id=f"error_{int(time.time())}",
                        content_type=future_to_item[future]['content_type'],
                        platform=future_to_item[future]['platform'],
                        overall_score=0.0,
                        individual_scores=[],
                        passed_validation=False,
                        requires_human_review=True,
                        improvement_suggestions=[f"Validation failed: {str(e)}"],
                        retry_recommended=True,
                        retry_parameters=None,
                        processing_time=0.0,
                        created_at=datetime.utcnow(),
                        metadata={'error': str(e)}
                    )
                    results.append(error_result)
        
        return results

    def get_quality_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get quality validation analytics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_validations = [
            v for v in self.validation_history 
            if v['timestamp'] > cutoff_date
        ]
        
        if not recent_validations:
            return {'message': 'No recent validation data available'}
        
        # Calculate analytics
        total_validations = len(recent_validations)
        passed_validations = len([v for v in recent_validations if v['passed']])
        pass_rate = passed_validations / total_validations if total_validations > 0 else 0
        
        # Platform breakdown
        platform_stats = {}
        for validation in recent_validations:
            platform = validation['platform']
            if platform not in platform_stats:
                platform_stats[platform] = {'total': 0, 'passed': 0, 'avg_score': 0}
            
            platform_stats[platform]['total'] += 1
            if validation['passed']:
                platform_stats[platform]['passed'] += 1
            platform_stats[platform]['avg_score'] += validation['overall_score']
        
        # Calculate averages
        for platform, stats in platform_stats.items():
            stats['pass_rate'] = stats['passed'] / stats['total']
            stats['avg_score'] = stats['avg_score'] / stats['total']
        
        # Content type breakdown
        content_type_stats = {}
        for validation in recent_validations:
            content_type = validation['content_type']
            if content_type not in content_type_stats:
                content_type_stats[content_type] = {'total': 0, 'passed': 0, 'avg_score': 0}
            
            content_type_stats[content_type]['total'] += 1
            if validation['passed']:
                content_type_stats[content_type]['passed'] += 1
            content_type_stats[content_type]['avg_score'] += validation['overall_score']
        
        # Calculate averages
        for content_type, stats in content_type_stats.items():
            stats['pass_rate'] = stats['passed'] / stats['total']
            stats['avg_score'] = stats['avg_score'] / stats['total']
        
        return {
            'period_days': days,
            'total_validations': total_validations,
            'overall_pass_rate': pass_rate,
            'average_score': np.mean([v['overall_score'] for v in recent_validations]),
            'platform_breakdown': platform_stats,
            'content_type_breakdown': content_type_stats,
            'average_processing_time': np.mean([v['processing_time'] for v in recent_validations])
        }

    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation system status"""
        return {
            'ai_clients_available': {
                'openai': self.openai_client is not None,
                'anthropic': self.anthropic_client is not None
            },
            'ml_models_available': {
                'sentiment_analyzer': self.sentiment_analyzer is not None,
                'engagement_predictor': self.engagement_predictor is not None
            },
            'supported_platforms': list(self.platform_configs.keys()),
            'supported_content_types': ['image', 'video', 'text'],
            'quality_tiers': list(self.quality_standards.keys()),
            'database_connected': self.db_manager is not None,
            'validation_cache_size': len(self.quality_cache),
            'validation_history_size': len(self.validation_history)
        }


# Example usage and testing functions
def create_sample_validation_config(platform: str, quality_tier: str = 'premium') -> ValidationConfig:
    """Create sample validation configuration"""
    return ValidationConfig(
        platform=platform,
        content_type='image',  # Will be overridden per validation
        quality_tier=quality_tier,
        enable_ai_analysis=True,
        enable_technical_analysis=True,
        enable_brand_compliance=True,
        enable_engagement_prediction=True,
        human_review_threshold=70.0,
        auto_reject_threshold=40.0,
        retry_on_failure=True
    )


async def test_validation_system():
    """Test the validation system with sample content"""
    validator = ContentQualityValidator()
    
    # Test configuration
    config = create_sample_validation_config('instagram', 'premium')
    
    print("Content Quality Validator initialized successfully!")
    print(f"Supported platforms: {list(validator.platform_configs.keys())}")
    print(f"Quality tiers: {list(validator.quality_standards.keys())}")
    
    # Test system status
    status = validator.get_validation_status()
    print(f"\nValidation System Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test analytics (empty initially)
    analytics = validator.get_quality_analytics()
    print(f"\nQuality Analytics: {analytics}")
    
    print("\nContent Quality Validator ready for comprehensive content validation!")


if __name__ == "__main__":
    # Run async test
    asyncio.run(test_validation_system())