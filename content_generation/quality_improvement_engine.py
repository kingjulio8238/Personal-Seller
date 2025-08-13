"""
Automated Quality Improvement Engine
Provides intelligent suggestions and automated retry mechanisms for content optimization
"""

import os
import json
import time
import hashlib
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from enum import Enum
import random

# Image and video processing
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import cv2
import numpy as np
import moviepy.editor as mp

# AI and ML libraries
import openai
import anthropic

# Database integration
try:
    from ..database.models import DatabaseManager, Product, Post, EngagementMetrics
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# LLM integration
from llm import create_client, get_response_from_llm

# Import quality validation and filtering
from .content_quality_validator import ContentQualityValidator, ValidationConfig, ContentQualityResult, QualityScore
from .content_filter import ContentFilter, FilterLevel, FilteringResult, FilterResult


class ImprovementType(Enum):
    """Types of content improvements"""
    TECHNICAL_ENHANCEMENT = "technical_enhancement"
    AESTHETIC_IMPROVEMENT = "aesthetic_improvement"
    ENGAGEMENT_OPTIMIZATION = "engagement_optimization"
    BRAND_COMPLIANCE = "brand_compliance"
    PLATFORM_OPTIMIZATION = "platform_optimization"
    TEXT_REFINEMENT = "text_refinement"


@dataclass
class ImprovementSuggestion:
    """Individual improvement suggestion with actionable steps"""
    suggestion_id: str
    improvement_type: ImprovementType
    title: str
    description: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    estimated_impact: float  # 0-100 score improvement estimate
    implementation_difficulty: str  # 'easy', 'medium', 'hard'
    automated_fix_available: bool
    manual_steps: List[str]
    parameter_adjustments: Dict[str, Any]
    success_probability: float  # 0-1 likelihood of improvement
    cost_estimate: Optional[Decimal]
    processing_time_estimate: float  # seconds


@dataclass
class ImprovementPlan:
    """Comprehensive improvement plan for content"""
    content_id: str
    content_type: str
    platform: str
    current_scores: Dict[str, float]
    target_scores: Dict[str, float]
    suggestions: List[ImprovementSuggestion]
    automated_fixes: List[ImprovementSuggestion]
    manual_actions: List[ImprovementSuggestion]
    total_estimated_impact: float
    implementation_order: List[str]  # Suggestion IDs in recommended order
    estimated_cost: Decimal
    estimated_time: float
    success_probability: float
    created_at: datetime


@dataclass
class RetryResult:
    """Result of automated retry attempt"""
    retry_id: str
    original_content_id: str
    retry_attempt: int
    improvements_applied: List[str]
    new_quality_result: Optional[ContentQualityResult]
    new_filter_result: Optional[FilteringResult]
    success: bool
    score_improvement: float
    processing_time: float
    cost: Decimal
    next_retry_recommended: bool
    next_retry_plan: Optional[ImprovementPlan]
    created_at: datetime


class QualityImprovementEngine:
    """
    Intelligent content quality improvement engine with automated optimization
    
    Features:
    - AI-powered improvement analysis and suggestions
    - Automated image/video enhancement algorithms
    - Text optimization and refinement
    - Platform-specific optimization recommendations
    - Automated retry mechanisms with learning
    - A/B testing for improvement effectiveness
    - Cost-benefit analysis for improvements
    - Performance tracking and analytics
    """
    
    def __init__(self, db_session=None):
        # Initialize quality validator and content filter
        self.quality_validator = ContentQualityValidator(db_session)
        self.content_filter = ContentFilter(db_session, FilterLevel.MODERATE)
        
        # Database integration
        self.db_manager = None
        if DATABASE_AVAILABLE and db_session:
            self.db_manager = DatabaseManager(db_session)
        
        # Initialize AI clients
        self._init_ai_clients()
        
        # Load improvement templates and strategies
        self._load_improvement_templates()
        self._load_optimization_strategies()
        
        # Improvement tracking and analytics
        self.improvement_history = []
        self.retry_history = []
        self.effectiveness_metrics = {}
        
        # Setup temp directories
        self.temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp', 'improvements')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Threading setup
        self.improvement_lock = threading.Lock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Quality Improvement Engine initialized")

    def _init_ai_clients(self):
        """Initialize AI clients for improvement analysis"""
        try:
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except Exception:
            self.openai_client = None
            self.logger.warning("OpenAI client not available")
        
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except Exception:
            self.anthropic_client = None
            self.logger.warning("Anthropic client not available")

    def _load_improvement_templates(self):
        """Load improvement templates for different content types and issues"""
        self.improvement_templates = {
            'image_technical': {
                'low_resolution': {
                    'title': 'Increase Image Resolution',
                    'description': 'Image resolution is below platform requirements',
                    'automated_fix': True,
                    'parameters': {'upscale_factor': 2.0, 'method': 'lanczos'},
                    'estimated_impact': 25.0,
                    'difficulty': 'easy'
                },
                'poor_brightness': {
                    'title': 'Adjust Image Brightness',
                    'description': 'Image brightness is outside optimal range',
                    'automated_fix': True,
                    'parameters': {'brightness_adjustment': 1.2},
                    'estimated_impact': 15.0,
                    'difficulty': 'easy'
                },
                'low_contrast': {
                    'title': 'Enhance Image Contrast',
                    'description': 'Image contrast is too low for visual impact',
                    'automated_fix': True,
                    'parameters': {'contrast_factor': 1.3},
                    'estimated_impact': 20.0,
                    'difficulty': 'easy'
                },
                'poor_sharpness': {
                    'title': 'Sharpen Image Details',
                    'description': 'Image appears blurry or lacks detail',
                    'automated_fix': True,
                    'parameters': {'sharpness_factor': 1.5, 'unsharp_mask': True},
                    'estimated_impact': 18.0,
                    'difficulty': 'medium'
                }
            },
            'image_aesthetic': {
                'poor_composition': {
                    'title': 'Improve Image Composition',
                    'description': 'Apply rule of thirds and better framing',
                    'automated_fix': True,
                    'parameters': {'crop_to_thirds': True, 'auto_frame': True},
                    'estimated_impact': 22.0,
                    'difficulty': 'medium'
                },
                'color_balance': {
                    'title': 'Optimize Color Balance',
                    'description': 'Adjust colors for better visual appeal',
                    'automated_fix': True,
                    'parameters': {'auto_color_balance': True, 'saturation_boost': 1.1},
                    'estimated_impact': 15.0,
                    'difficulty': 'easy'
                },
                'lighting_enhancement': {
                    'title': 'Enhance Lighting',
                    'description': 'Improve lighting and shadows',
                    'automated_fix': True,
                    'parameters': {'shadow_lift': 0.2, 'highlight_recovery': 0.1},
                    'estimated_impact': 20.0,
                    'difficulty': 'medium'
                }
            },
            'video_technical': {
                'poor_resolution': {
                    'title': 'Upscale Video Resolution',
                    'description': 'Video resolution below platform standards',
                    'automated_fix': True,
                    'parameters': {'target_resolution': (1920, 1080), 'upscale_method': 'lanczos'},
                    'estimated_impact': 30.0,
                    'difficulty': 'hard'
                },
                'poor_framerate': {
                    'title': 'Optimize Frame Rate',
                    'description': 'Adjust frame rate for platform requirements',
                    'automated_fix': True,
                    'parameters': {'target_fps': 30, 'interpolation': True},
                    'estimated_impact': 15.0,
                    'difficulty': 'medium'
                },
                'audio_issues': {
                    'title': 'Fix Audio Problems',
                    'description': 'Improve audio quality and sync',
                    'automated_fix': True,
                    'parameters': {'noise_reduction': True, 'normalize_audio': True},
                    'estimated_impact': 25.0,
                    'difficulty': 'medium'
                }
            },
            'text_optimization': {
                'poor_readability': {
                    'title': 'Improve Text Readability',
                    'description': 'Simplify language and sentence structure',
                    'automated_fix': True,
                    'parameters': {'target_reading_level': 8, 'simplify_language': True},
                    'estimated_impact': 20.0,
                    'difficulty': 'medium'
                },
                'weak_engagement': {
                    'title': 'Enhance Engagement Elements',
                    'description': 'Add compelling hooks and calls-to-action',
                    'automated_fix': True,
                    'parameters': {'add_engagement_words': True, 'strengthen_cta': True},
                    'estimated_impact': 30.0,
                    'difficulty': 'easy'
                },
                'platform_mismatch': {
                    'title': 'Optimize for Platform',
                    'description': 'Adapt content style for specific platform',
                    'automated_fix': True,
                    'parameters': {'platform_optimization': True},
                    'estimated_impact': 25.0,
                    'difficulty': 'medium'
                }
            }
        }

    def _load_optimization_strategies(self):
        """Load platform-specific optimization strategies"""
        self.optimization_strategies = {
            'instagram': {
                'image_focus': 'aesthetic_appeal',
                'preferred_ratios': [(1, 1), (4, 5)],
                'color_preferences': 'vibrant_warm',
                'engagement_tactics': ['story_worthy', 'user_generated_content'],
                'hashtag_strategy': 'high_volume_targeted',
                'text_style': 'casual_storytelling'
            },
            'tiktok': {
                'video_focus': 'viral_potential',
                'preferred_ratios': [(9, 16)],
                'color_preferences': 'high_contrast_trendy',
                'engagement_tactics': ['trending_audio', 'challenges', 'duets'],
                'hashtag_strategy': 'trending_fyp',
                'text_style': 'energetic_youth'
            },
            'x': {
                'text_focus': 'concise_informative',
                'preferred_ratios': [(16, 9), (2, 1)],
                'color_preferences': 'clean_professional',
                'engagement_tactics': ['threads', 'replies', 'retweets'],
                'hashtag_strategy': 'minimal_targeted',
                'text_style': 'conversational_professional'
            },
            'linkedin': {
                'content_focus': 'professional_value',
                'preferred_ratios': [(1.91, 1)],
                'color_preferences': 'professional_clean',
                'engagement_tactics': ['thought_leadership', 'industry_insights'],
                'hashtag_strategy': 'professional_networking',
                'text_style': 'authoritative_educational'
            }
        }

    async def analyze_improvement_opportunities(self, quality_result: ContentQualityResult,
                                              filter_result: Optional[FilteringResult] = None,
                                              content_path: Optional[str] = None,
                                              product_data: Optional[Dict[str, Any]] = None) -> ImprovementPlan:
        """
        Analyze content and generate comprehensive improvement plan
        
        Args:
            quality_result: Quality validation result
            filter_result: Optional filtering result
            content_path: Path to content file for analysis
            product_data: Optional product information
        
        Returns:
            ImprovementPlan with actionable suggestions
        """
        start_time = time.time()
        
        self.logger.info(f"Analyzing improvement opportunities for {quality_result.content_id}")
        
        # Initialize improvement plan
        plan = ImprovementPlan(
            content_id=quality_result.content_id,
            content_type=quality_result.content_type,
            platform=quality_result.platform,
            current_scores={},
            target_scores={},
            suggestions=[],
            automated_fixes=[],
            manual_actions=[],
            total_estimated_impact=0.0,
            implementation_order=[],
            estimated_cost=Decimal('0.00'),
            estimated_time=0.0,
            success_probability=0.0,
            created_at=datetime.utcnow()
        )
        
        # Extract current scores
        plan.current_scores = {
            score.metric_name: score.score 
            for score in quality_result.individual_scores
        }
        plan.current_scores['overall'] = quality_result.overall_score
        
        # Set target scores (aim for 85+ overall, 80+ individual)
        plan.target_scores = {
            metric: max(85.0, current + 20) 
            for metric, current in plan.current_scores.items()
        }
        
        # Generate suggestions based on failed quality metrics
        for score in quality_result.individual_scores:
            if not score.passed or score.score < 75.0:
                suggestions = await self._generate_metric_suggestions(
                    score, quality_result.content_type, quality_result.platform, content_path
                )
                plan.suggestions.extend(suggestions)
        
        # Generate suggestions based on filter issues
        if filter_result and filter_result.final_decision != FilterResult.APPROVED:
            filter_suggestions = await self._generate_filter_suggestions(
                filter_result, quality_result.content_type, content_path
            )
            plan.suggestions.extend(filter_suggestions)
        
        # AI-powered improvement analysis
        if content_path and (self.openai_client or self.anthropic_client):
            ai_suggestions = await self._generate_ai_improvements(
                content_path, quality_result, plan.current_scores, product_data
            )
            plan.suggestions.extend(ai_suggestions)
        
        # Platform-specific optimizations
        platform_suggestions = await self._generate_platform_optimizations(
            quality_result.platform, quality_result.content_type, plan.current_scores
        )
        plan.suggestions.extend(platform_suggestions)
        
        # Categorize suggestions
        plan.automated_fixes = [s for s in plan.suggestions if s.automated_fix_available]
        plan.manual_actions = [s for s in plan.suggestions if not s.automated_fix_available]
        
        # Calculate total estimated impact and costs
        plan.total_estimated_impact = sum(s.estimated_impact for s in plan.suggestions)
        plan.estimated_cost = sum(s.cost_estimate or Decimal('0.00') for s in plan.suggestions)
        plan.estimated_time = sum(s.processing_time_estimate for s in plan.suggestions)
        
        # Calculate success probability (based on historical data and suggestion difficulty)
        plan.success_probability = self._calculate_success_probability(plan.suggestions)
        
        # Determine implementation order
        plan.implementation_order = self._optimize_implementation_order(plan.suggestions)
        
        self.logger.info(
            f"Improvement analysis completed for {quality_result.content_id}: "
            f"{len(plan.suggestions)} suggestions, "
            f"{plan.total_estimated_impact:.1f} total impact, "
            f"${plan.estimated_cost:.2f} estimated cost"
        )
        
        return plan

    async def _generate_metric_suggestions(self, quality_score: QualityScore, 
                                         content_type: str, platform: str,
                                         content_path: Optional[str]) -> List[ImprovementSuggestion]:
        """Generate suggestions based on specific quality metric failures"""
        suggestions = []
        metric_name = quality_score.metric_name
        current_score = quality_score.score
        
        # Map quality metrics to improvement templates
        template_mapping = {
            'technical_quality': f'{content_type}_technical',
            'aesthetic_appeal': f'{content_type}_aesthetic', 
            'engagement_potential': 'text_optimization',
            'brand_compliance': 'brand_optimization',
            'platform_optimization': 'platform_optimization'
        }
        
        template_category = template_mapping.get(metric_name)
        if not template_category or template_category not in self.improvement_templates:
            return suggestions
        
        templates = self.improvement_templates[template_category]
        
        # Determine specific issues based on quality score details
        issues_detected = self._detect_specific_issues(quality_score, content_path)
        
        for issue in issues_detected:
            if issue in templates:
                template = templates[issue]
                
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"{metric_name}_{issue}_{int(time.time())}",
                    improvement_type=ImprovementType.TECHNICAL_ENHANCEMENT,
                    title=template['title'],
                    description=template['description'],
                    priority=self._determine_priority(current_score, quality_score.threshold),
                    estimated_impact=template['estimated_impact'],
                    implementation_difficulty=template['difficulty'],
                    automated_fix_available=template['automated_fix'],
                    manual_steps=self._generate_manual_steps(template, issue),
                    parameter_adjustments=template.get('parameters', {}),
                    success_probability=self._estimate_success_probability(template, current_score),
                    cost_estimate=self._estimate_cost(template, content_type),
                    processing_time_estimate=self._estimate_processing_time(template, content_type)
                )
                
                suggestions.append(suggestion)
        
        return suggestions

    def _detect_specific_issues(self, quality_score: QualityScore, 
                              content_path: Optional[str]) -> List[str]:
        """Detect specific issues from quality score details"""
        issues = []
        details = quality_score.details
        
        # Technical quality issues
        if quality_score.metric_name == 'technical_quality':
            if details.get('resolution') and details['resolution'][0] < 1080:
                issues.append('low_resolution')
            if details.get('brightness', 50) < 30 or details.get('brightness', 50) > 80:
                issues.append('poor_brightness')
            if details.get('contrast', 50) < 40:
                issues.append('low_contrast')
            if details.get('sharpness', 100) < 60:
                issues.append('poor_sharpness')
        
        # Aesthetic issues
        elif quality_score.metric_name == 'aesthetic_appeal':
            if details.get('composition', 50) < 60:
                issues.append('poor_composition')
            if details.get('color_harmony', 50) < 60:
                issues.append('color_balance')
            if details.get('vibrancy', 50) < 70:
                issues.append('lighting_enhancement')
        
        # Text issues
        elif quality_score.metric_name == 'engagement_potential':
            if details.get('readability', 70) < 60:
                issues.append('poor_readability')
            if details.get('engagement_words_found', 3) < 2:
                issues.append('weak_engagement')
            if details.get('platform_optimization_score', 80) < 70:
                issues.append('platform_mismatch')
        
        return issues if issues else ['general_improvement']

    async def _generate_filter_suggestions(self, filter_result: FilteringResult,
                                         content_type: str, content_path: Optional[str]) -> List[ImprovementSuggestion]:
        """Generate suggestions based on filter violations"""
        suggestions = []
        
        for rule in filter_result.triggered_rules:
            suggestion = ImprovementSuggestion(
                suggestion_id=f"filter_{rule.rule_id}_{int(time.time())}",
                improvement_type=ImprovementType.BRAND_COMPLIANCE,
                title=f"Fix {rule.name} Violation",
                description=rule.description,
                priority='high' if rule.severity in ['critical', 'high'] else 'medium',
                estimated_impact=30.0 if rule.severity == 'critical' else 20.0,
                implementation_difficulty='easy' if rule.filter_type == 'text' else 'medium',
                automated_fix_available=rule.filter_type == 'text',
                manual_steps=self._generate_filter_fix_steps(rule),
                parameter_adjustments={'filter_rule': rule.rule_id},
                success_probability=0.8,
                cost_estimate=Decimal('0.00'),
                processing_time_estimate=30.0
            )
            suggestions.append(suggestion)
        
        return suggestions

    async def _generate_ai_improvements(self, content_path: str, quality_result: ContentQualityResult,
                                      current_scores: Dict[str, float], 
                                      product_data: Optional[Dict[str, Any]]) -> List[ImprovementSuggestion]:
        """Generate AI-powered improvement suggestions"""
        if not (self.openai_client or self.anthropic_client):
            return []
        
        suggestions = []
        
        try:
            # Create comprehensive analysis prompt
            analysis_prompt = f"""
            Analyze this {quality_result.content_type} content for {quality_result.platform} and provide specific improvement suggestions.
            
            Current Quality Scores:
            {json.dumps(current_scores, indent=2)}
            
            Content Type: {quality_result.content_type}
            Platform: {quality_result.platform}
            Overall Score: {quality_result.overall_score:.1f}/100
            
            Provide 3-5 specific, actionable improvement suggestions that would most effectively increase the quality scores.
            
            For each suggestion, include:
            1. Specific improvement action
            2. Expected impact on quality score
            3. Difficulty level (easy/medium/hard)
            4. Whether it can be automated
            5. Estimated implementation time
            
            Focus on the lowest scoring metrics for maximum impact.
            """
            
            # Use Claude for analysis
            client, model = create_client('claude-3-5-sonnet-20241022')
            
            ai_response, _ = get_response_from_llm(
                msg=analysis_prompt,
                client=client,
                model=model,
                system_message="You are an expert content optimization consultant. Provide specific, actionable improvement suggestions.",
                print_debug=False
            )
            
            # Parse AI response into structured suggestions
            ai_suggestions = self._parse_ai_suggestions(ai_response, quality_result.content_id)
            suggestions.extend(ai_suggestions)
            
        except Exception as e:
            self.logger.error(f"AI improvement analysis failed: {e}")
        
        return suggestions

    def _parse_ai_suggestions(self, ai_response: str, content_id: str) -> List[ImprovementSuggestion]:
        """Parse AI response into structured improvement suggestions"""
        suggestions = []
        
        # Simple parsing - could be enhanced with more sophisticated NLP
        lines = ai_response.split('\n')
        current_suggestion = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for suggestion markers
            if any(marker in line.lower() for marker in ['suggestion', 'improvement', 'recommendation']):
                if current_suggestion:
                    # Process previous suggestion
                    suggestion = self._create_suggestion_from_ai(current_suggestion, content_id)
                    if suggestion:
                        suggestions.append(suggestion)
                
                # Start new suggestion
                current_suggestion = {'title': line}
            
            elif 'impact' in line.lower():
                current_suggestion['impact'] = line
            elif 'difficulty' in line.lower():
                current_suggestion['difficulty'] = line
            elif 'automated' in line.lower() or 'automation' in line.lower():
                current_suggestion['automation'] = line
            else:
                # Add to description
                if 'description' not in current_suggestion:
                    current_suggestion['description'] = line
                else:
                    current_suggestion['description'] += ' ' + line
        
        # Process final suggestion
        if current_suggestion:
            suggestion = self._create_suggestion_from_ai(current_suggestion, content_id)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions[:5]  # Limit to 5 suggestions

    def _create_suggestion_from_ai(self, ai_data: Dict[str, str], content_id: str) -> Optional[ImprovementSuggestion]:
        """Create structured suggestion from AI data"""
        try:
            # Extract impact score
            impact_text = ai_data.get('impact', '15')
            impact_score = 15.0
            import re
            impact_numbers = re.findall(r'(\d+(?:\.\d+)?)', impact_text)
            if impact_numbers:
                impact_score = float(impact_numbers[0])
            
            # Extract difficulty
            difficulty_text = ai_data.get('difficulty', 'medium').lower()
            if 'easy' in difficulty_text:
                difficulty = 'easy'
            elif 'hard' in difficulty_text:
                difficulty = 'hard'
            else:
                difficulty = 'medium'
            
            # Extract automation capability
            automation_text = ai_data.get('automation', '').lower()
            automated = any(word in automation_text for word in ['yes', 'can', 'automated', 'automatic'])
            
            suggestion = ImprovementSuggestion(
                suggestion_id=f"ai_{content_id}_{int(time.time())}_{random.randint(1000,9999)}",
                improvement_type=ImprovementType.ENGAGEMENT_OPTIMIZATION,
                title=ai_data.get('title', 'AI-suggested improvement'),
                description=ai_data.get('description', 'AI-recommended optimization'),
                priority='medium',
                estimated_impact=impact_score,
                implementation_difficulty=difficulty,
                automated_fix_available=automated,
                manual_steps=[ai_data.get('description', 'Follow AI recommendation')],
                parameter_adjustments={'ai_suggestion': True},
                success_probability=0.7,
                cost_estimate=Decimal('2.00') if not automated else Decimal('0.50'),
                processing_time_estimate=120.0 if difficulty == 'hard' else 60.0
            )
            
            return suggestion
            
        except Exception as e:
            self.logger.error(f"Failed to create AI suggestion: {e}")
            return None

    async def _generate_platform_optimizations(self, platform: str, content_type: str,
                                             current_scores: Dict[str, float]) -> List[ImprovementSuggestion]:
        """Generate platform-specific optimization suggestions"""
        suggestions = []
        
        if platform not in self.optimization_strategies:
            return suggestions
        
        strategy = self.optimization_strategies[platform]
        
        # Platform-specific optimizations based on focus areas
        if content_type == 'image' and strategy.get('image_focus'):
            suggestion = ImprovementSuggestion(
                suggestion_id=f"platform_{platform}_image_{int(time.time())}",
                improvement_type=ImprovementType.PLATFORM_OPTIMIZATION,
                title=f"Optimize Image for {platform.title()}",
                description=f"Apply {platform}-specific image optimizations focusing on {strategy['image_focus']}",
                priority='medium',
                estimated_impact=20.0,
                implementation_difficulty='medium',
                automated_fix_available=True,
                manual_steps=[f"Apply {platform} optimization strategy"],
                parameter_adjustments={
                    'platform_strategy': strategy,
                    'color_preferences': strategy.get('color_preferences'),
                    'preferred_ratios': strategy.get('preferred_ratios')
                },
                success_probability=0.75,
                cost_estimate=Decimal('1.00'),
                processing_time_estimate=45.0
            )
            suggestions.append(suggestion)
        
        elif content_type == 'video' and strategy.get('video_focus'):
            suggestion = ImprovementSuggestion(
                suggestion_id=f"platform_{platform}_video_{int(time.time())}",
                improvement_type=ImprovementType.PLATFORM_OPTIMIZATION,
                title=f"Optimize Video for {platform.title()}",
                description=f"Apply {platform}-specific video optimizations focusing on {strategy['video_focus']}",
                priority='high',
                estimated_impact=25.0,
                implementation_difficulty='medium',
                automated_fix_available=True,
                manual_steps=[f"Apply {platform} video optimization strategy"],
                parameter_adjustments={
                    'platform_strategy': strategy,
                    'preferred_ratios': strategy.get('preferred_ratios'),
                    'engagement_tactics': strategy.get('engagement_tactics')
                },
                success_probability=0.8,
                cost_estimate=Decimal('2.00'),
                processing_time_estimate=90.0
            )
            suggestions.append(suggestion)
        
        elif content_type == 'text' and strategy.get('text_style'):
            suggestion = ImprovementSuggestion(
                suggestion_id=f"platform_{platform}_text_{int(time.time())}",
                improvement_type=ImprovementType.TEXT_REFINEMENT,
                title=f"Optimize Text for {platform.title()}",
                description=f"Adapt text style to {strategy['text_style']} for {platform}",
                priority='medium',
                estimated_impact=18.0,
                implementation_difficulty='easy',
                automated_fix_available=True,
                manual_steps=[f"Apply {platform} text style optimization"],
                parameter_adjustments={
                    'text_style': strategy['text_style'],
                    'hashtag_strategy': strategy.get('hashtag_strategy'),
                    'engagement_tactics': strategy.get('engagement_tactics')
                },
                success_probability=0.85,
                cost_estimate=Decimal('0.50'),
                processing_time_estimate=30.0
            )
            suggestions.append(suggestion)
        
        return suggestions

    def _determine_priority(self, current_score: float, threshold: float) -> str:
        """Determine improvement priority based on score gap"""
        gap = threshold - current_score
        
        if gap > 30:
            return 'critical'
        elif gap > 20:
            return 'high'
        elif gap > 10:
            return 'medium'
        else:
            return 'low'

    def _generate_manual_steps(self, template: Dict[str, Any], issue: str) -> List[str]:
        """Generate manual implementation steps"""
        base_steps = {
            'low_resolution': [
                "Use image editing software to increase resolution",
                "Apply high-quality upscaling algorithm",
                "Ensure final resolution meets platform requirements"
            ],
            'poor_brightness': [
                "Adjust brightness levels in image editor",
                "Balance highlights and shadows",
                "Test on different devices for consistency"
            ],
            'poor_composition': [
                "Crop image using rule of thirds",
                "Reframe to improve focal points",
                "Consider different aspect ratios"
            ],
            'weak_engagement': [
                "Add emotional trigger words",
                "Include clear call-to-action",
                "Test different hooks and openings"
            ]
        }
        
        return base_steps.get(issue, ["Review and manually optimize content"])

    def _estimate_success_probability(self, template: Dict[str, Any], current_score: float) -> float:
        """Estimate probability of successful improvement"""
        base_probability = 0.7
        
        # Adjust based on difficulty
        difficulty_adjustments = {
            'easy': 0.15,
            'medium': 0.0,
            'hard': -0.2
        }
        
        difficulty = template.get('difficulty', 'medium')
        base_probability += difficulty_adjustments.get(difficulty, 0.0)
        
        # Adjust based on current score (lower scores easier to improve)
        if current_score < 40:
            base_probability += 0.1
        elif current_score > 80:
            base_probability -= 0.1
        
        return max(0.1, min(0.95, base_probability))

    def _estimate_cost(self, template: Dict[str, Any], content_type: str) -> Decimal:
        """Estimate implementation cost"""
        base_costs = {
            'easy': Decimal('0.25'),
            'medium': Decimal('1.00'),
            'hard': Decimal('3.00')
        }
        
        difficulty = template.get('difficulty', 'medium')
        base_cost = base_costs.get(difficulty, Decimal('1.00'))
        
        # Adjust based on content type
        if content_type == 'video':
            base_cost *= Decimal('2.0')
        elif content_type == 'image':
            base_cost *= Decimal('1.5')
        
        return base_cost

    def _estimate_processing_time(self, template: Dict[str, Any], content_type: str) -> float:
        """Estimate processing time in seconds"""
        base_times = {
            'easy': 30.0,
            'medium': 60.0,
            'hard': 180.0
        }
        
        difficulty = template.get('difficulty', 'medium')
        base_time = base_times.get(difficulty, 60.0)
        
        # Adjust based on content type
        if content_type == 'video':
            base_time *= 3.0
        elif content_type == 'image':
            base_time *= 1.5
        
        return base_time

    def _calculate_success_probability(self, suggestions: List[ImprovementSuggestion]) -> float:
        """Calculate overall success probability for improvement plan"""
        if not suggestions:
            return 0.0
        
        # Calculate weighted average based on impact
        total_weight = sum(s.estimated_impact for s in suggestions)
        if total_weight == 0:
            return 0.5
        
        weighted_prob = sum(
            s.success_probability * s.estimated_impact 
            for s in suggestions
        ) / total_weight
        
        return weighted_prob

    def _optimize_implementation_order(self, suggestions: List[ImprovementSuggestion]) -> List[str]:
        """Optimize implementation order for maximum effectiveness"""
        # Sort by: priority (critical first), then impact/difficulty ratio
        priority_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        difficulty_weights = {'easy': 3, 'medium': 2, 'hard': 1}
        
        def sort_key(s):
            priority_score = priority_weights.get(s.priority, 2)
            difficulty_score = difficulty_weights.get(s.implementation_difficulty, 2)
            impact_ratio = s.estimated_impact / max(1, s.processing_time_estimate / 60)
            
            return (priority_score, impact_ratio, difficulty_score)
        
        sorted_suggestions = sorted(suggestions, key=sort_key, reverse=True)
        return [s.suggestion_id for s in sorted_suggestions]

    def _generate_filter_fix_steps(self, filter_rule) -> List[str]:
        """Generate steps to fix filter rule violations"""
        fix_steps = {
            'profanity_check': [
                "Review content for inappropriate language",
                "Replace or remove offensive words",
                "Use alternative expressions"
            ],
            'spam_detection': [
                "Reduce promotional language density",
                "Remove excessive repetition",
                "Add value-focused content"
            ],
            'medical_claims': [
                "Add appropriate medical disclaimers",
                "Soften health-related claims",
                "Include 'consult your doctor' advisory"
            ]
        }
        
        return fix_steps.get(filter_rule.rule_id, ["Address filter rule violation"])

    async def apply_automated_improvements(self, improvement_plan: ImprovementPlan,
                                         content_path: str, max_attempts: int = 3) -> RetryResult:
        """
        Apply automated improvements and retry content validation
        
        Args:
            improvement_plan: Plan with automated improvements to apply
            content_path: Path to original content
            max_attempts: Maximum retry attempts
        
        Returns:
            RetryResult with outcome of improvement attempts
        """
        start_time = time.time()
        retry_id = f"retry_{improvement_plan.content_id}_{int(start_time)}"
        
        self.logger.info(f"Applying automated improvements: {retry_id}")
        
        # Initialize retry result
        result = RetryResult(
            retry_id=retry_id,
            original_content_id=improvement_plan.content_id,
            retry_attempt=1,
            improvements_applied=[],
            new_quality_result=None,
            new_filter_result=None,
            success=False,
            score_improvement=0.0,
            processing_time=0.0,
            cost=Decimal('0.00'),
            next_retry_recommended=False,
            next_retry_plan=None,
            created_at=datetime.utcnow()
        )
        
        try:
            # Apply automated improvements in order
            improved_content_path = content_path
            applied_improvements = []
            total_cost = Decimal('0.00')
            
            for suggestion_id in improvement_plan.implementation_order:
                suggestion = next(
                    (s for s in improvement_plan.automated_fixes if s.suggestion_id == suggestion_id),
                    None
                )
                
                if not suggestion:
                    continue
                
                # Apply improvement
                self.logger.info(f"Applying improvement: {suggestion.title}")
                
                improved_path = await self._apply_single_improvement(
                    improved_content_path, suggestion, improvement_plan.content_type
                )
                
                if improved_path and improved_path != improved_content_path:
                    improved_content_path = improved_path
                    applied_improvements.append(suggestion.title)
                    total_cost += suggestion.cost_estimate or Decimal('0.00')
                    
                    self.logger.info(f"Successfully applied: {suggestion.title}")
                else:
                    self.logger.warning(f"Failed to apply: {suggestion.title}")
            
            result.improvements_applied = applied_improvements
            result.cost = total_cost
            
            # Validate improved content
            if applied_improvements:
                config = ValidationConfig(
                    platform=improvement_plan.platform,
                    content_type=improvement_plan.content_type,
                    quality_tier='standard'
                )
                
                # Quality validation
                new_quality_result = await self.quality_validator.validate_content(
                    improved_content_path, improvement_plan.content_type, 
                    improvement_plan.platform, config
                )
                result.new_quality_result = new_quality_result
                
                # Filter validation
                new_filter_result = await self.content_filter.filter_content(
                    improved_content_path, improvement_plan.content_type,
                    improvement_plan.platform, new_quality_result
                )
                result.new_filter_result = new_filter_result
                
                # Calculate improvement
                original_score = improvement_plan.current_scores.get('overall', 0)
                new_score = new_quality_result.overall_score
                result.score_improvement = new_score - original_score
                
                # Determine success
                result.success = (
                    new_quality_result.passed_validation and
                    new_filter_result.final_decision == FilterResult.APPROVED and
                    result.score_improvement > 5.0
                )
                
                # Check if further improvements needed
                if not result.success and result.retry_attempt < max_attempts:
                    result.next_retry_recommended = True
                    # Generate new improvement plan based on updated results
                    result.next_retry_plan = await self.analyze_improvement_opportunities(
                        new_quality_result, new_filter_result, improved_content_path
                    )
                
                self.logger.info(
                    f"Improvement results: Score {original_score:.1f} -> {new_score:.1f} "
                    f"({result.score_improvement:+.1f}), Success: {result.success}"
                )
            
            # Calculate processing time
            result.processing_time = time.time() - start_time
            
            # Store retry result
            if self.db_manager:
                self._store_retry_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Automated improvement failed for {retry_id}: {e}")
            result.processing_time = time.time() - start_time
            return result

    async def _apply_single_improvement(self, content_path: str, 
                                      suggestion: ImprovementSuggestion,
                                      content_type: str) -> Optional[str]:
        """Apply a single improvement to content"""
        try:
            if content_type == 'image':
                return await self._apply_image_improvement(content_path, suggestion)
            elif content_type == 'video':
                return await self._apply_video_improvement(content_path, suggestion)
            elif content_type == 'text':
                return await self._apply_text_improvement(content_path, suggestion)
            
        except Exception as e:
            self.logger.error(f"Failed to apply improvement {suggestion.title}: {e}")
            return None

    async def _apply_image_improvement(self, image_path: str, 
                                     suggestion: ImprovementSuggestion) -> Optional[str]:
        """Apply image-specific improvements"""
        try:
            # Load image
            image = Image.open(image_path)
            params = suggestion.parameter_adjustments
            
            # Apply improvements based on type
            if 'brightness_adjustment' in params:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(params['brightness_adjustment'])
            
            if 'contrast_factor' in params:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(params['contrast_factor'])
            
            if 'sharpness_factor' in params:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(params['sharpness_factor'])
            
            if 'saturation_boost' in params:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(params['saturation_boost'])
            
            if 'upscale_factor' in params:
                factor = params['upscale_factor']
                new_size = (int(image.width * factor), int(image.height * factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            if 'crop_to_thirds' in params and params['crop_to_thirds']:
                # Apply rule of thirds cropping
                width, height = image.size
                # Simple center crop maintaining aspect ratio
                crop_width = int(width * 0.9)
                crop_height = int(height * 0.9)
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                image = image.crop((left, top, left + crop_width, top + crop_height))
            
            # Save improved image
            improved_path = os.path.join(self.temp_dir, f"improved_{os.path.basename(image_path)}")
            image.save(improved_path, quality=95)
            
            return improved_path
            
        except Exception as e:
            self.logger.error(f"Image improvement failed: {e}")
            return None

    async def _apply_video_improvement(self, video_path: str, 
                                     suggestion: ImprovementSuggestion) -> Optional[str]:
        """Apply video-specific improvements"""
        try:
            # Load video
            video = mp.VideoFileClip(video_path)
            params = suggestion.parameter_adjustments
            
            # Apply improvements
            if 'target_resolution' in params:
                target_res = params['target_resolution']
                video = video.resize(target_res)
            
            if 'target_fps' in params:
                target_fps = params['target_fps']
                video = video.set_fps(target_fps)
            
            if 'normalize_audio' in params and params['normalize_audio'] and video.audio:
                video = video.volumex(1.2)  # Simple volume boost
            
            # Save improved video
            improved_path = os.path.join(self.temp_dir, f"improved_{os.path.basename(video_path)}")
            video.write_videofile(improved_path, codec='libx264', audio_codec='aac')
            
            video.close()
            return improved_path
            
        except Exception as e:
            self.logger.error(f"Video improvement failed: {e}")
            return None

    async def _apply_text_improvement(self, text_path: str, 
                                    suggestion: ImprovementSuggestion) -> Optional[str]:
        """Apply text-specific improvements"""
        try:
            # Read text content
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = text_path  # Assume it's the text content itself
            
            params = suggestion.parameter_adjustments
            improved_text = text
            
            # Apply text improvements
            if 'add_engagement_words' in params and params['add_engagement_words']:
                engagement_words = ['amazing', 'incredible', 'must-see', 'exclusive']
                # Simple addition of engagement word if not present
                if not any(word in improved_text.lower() for word in engagement_words):
                    improved_text = f"{engagement_words[0].title()} " + improved_text
            
            if 'strengthen_cta' in params and params['strengthen_cta']:
                # Add or strengthen call-to-action
                if not any(cta in improved_text.lower() for cta in ['click', 'buy', 'get', 'try']):
                    improved_text += " Get yours today!"
            
            if 'platform_optimization' in params:
                # Apply platform-specific text optimization
                # This would include hashtag optimization, length adjustment, etc.
                pass
            
            # Save improved text
            if os.path.exists(text_path):
                improved_path = os.path.join(self.temp_dir, f"improved_{os.path.basename(text_path)}")
                with open(improved_path, 'w', encoding='utf-8') as f:
                    f.write(improved_text)
                return improved_path
            else:
                return improved_text  # Return improved text directly
            
        except Exception as e:
            self.logger.error(f"Text improvement failed: {e}")
            return None

    def _store_retry_result(self, result: RetryResult):
        """Store retry result for analytics and learning"""
        try:
            self.retry_history.append({
                'retry_id': result.retry_id,
                'original_content_id': result.original_content_id,
                'improvements_applied': result.improvements_applied,
                'score_improvement': result.score_improvement,
                'success': result.success,
                'cost': float(result.cost),
                'processing_time': result.processing_time,
                'timestamp': result.created_at
            })
            
            # Update effectiveness metrics
            for improvement in result.improvements_applied:
                if improvement not in self.effectiveness_metrics:
                    self.effectiveness_metrics[improvement] = {
                        'total_applications': 0,
                        'successes': 0,
                        'average_improvement': 0.0,
                        'total_cost': 0.0
                    }
                
                metrics = self.effectiveness_metrics[improvement]
                metrics['total_applications'] += 1
                if result.success:
                    metrics['successes'] += 1
                metrics['average_improvement'] = (
                    (metrics['average_improvement'] * (metrics['total_applications'] - 1) + 
                     result.score_improvement) / metrics['total_applications']
                )
                metrics['total_cost'] += float(result.cost)
            
            self.logger.info(f"Retry result stored: {result.retry_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store retry result: {e}")

    def get_improvement_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get improvement effectiveness analytics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_retries = [
            r for r in self.retry_history 
            if r['timestamp'] > cutoff_date
        ]
        
        if not recent_retries:
            return {'message': 'No recent improvement data available'}
        
        # Calculate analytics
        total_retries = len(recent_retries)
        successful_retries = len([r for r in recent_retries if r['success']])
        success_rate = successful_retries / total_retries if total_retries > 0 else 0
        
        average_improvement = np.mean([r['score_improvement'] for r in recent_retries])
        total_cost = sum(r['cost'] for r in recent_retries)
        average_processing_time = np.mean([r['processing_time'] for r in recent_retries])
        
        # Most effective improvements
        improvement_effectiveness = {}
        for improvement, metrics in self.effectiveness_metrics.items():
            if metrics['total_applications'] > 0:
                improvement_effectiveness[improvement] = {
                    'success_rate': metrics['successes'] / metrics['total_applications'],
                    'average_improvement': metrics['average_improvement'],
                    'cost_per_application': metrics['total_cost'] / metrics['total_applications'],
                    'total_applications': metrics['total_applications']
                }
        
        # Sort by effectiveness (success rate * average improvement)
        sorted_effectiveness = sorted(
            improvement_effectiveness.items(),
            key=lambda x: x[1]['success_rate'] * x[1]['average_improvement'],
            reverse=True
        )
        
        return {
            'period_days': days,
            'total_retries': total_retries,
            'success_rate': success_rate,
            'average_score_improvement': average_improvement,
            'total_cost': total_cost,
            'average_processing_time': average_processing_time,
            'most_effective_improvements': dict(sorted_effectiveness[:10]),
            'cost_per_success': total_cost / successful_retries if successful_retries > 0 else 0
        }

    def get_engine_status(self) -> Dict[str, Any]:
        """Get improvement engine status"""
        return {
            'ai_clients_available': {
                'openai': self.openai_client is not None,
                'anthropic': self.anthropic_client is not None
            },
            'supported_improvement_types': [t.value for t in ImprovementType],
            'improvement_templates_loaded': len(self.improvement_templates),
            'optimization_strategies': list(self.optimization_strategies.keys()),
            'database_connected': self.db_manager is not None,
            'improvement_history_size': len(self.improvement_history),
            'retry_history_size': len(self.retry_history),
            'effectiveness_metrics_tracked': len(self.effectiveness_metrics)
        }


# Example usage and testing functions
async def test_improvement_engine():
    """Test the quality improvement engine"""
    engine = QualityImprovementEngine()
    
    print("Quality Improvement Engine initialized successfully!")
    
    # Test system status
    status = engine.get_engine_status()
    print(f"\nImprovement Engine Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test analytics (empty initially)
    analytics = engine.get_improvement_analytics()
    print(f"\nImprovement Analytics: {analytics}")
    
    print("\nQuality Improvement Engine ready for automated content optimization!")


if __name__ == "__main__":
    import numpy as np
    
    # Run async test
    asyncio.run(test_improvement_engine())