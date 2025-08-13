"""
Multi-Layer Content Filtering System
Advanced filtering and quality gates for automated content approval/rejection
with platform policy compliance and brand safety features
"""

import os
import json
import time
import hashlib
import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from enum import Enum

# AI and content analysis
import openai
import anthropic
from transformers import pipeline
import torch

# Text analysis
import nltk
from textstat import flesch_reading_ease
import emoji

# Content filtering libraries
from better_profanity import profanity
import cv2
import numpy as np
from PIL import Image

# Database integration
try:
    from ..database.models import DatabaseManager, Product, Post, EngagementMetrics
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# LLM integration
from llm import create_client, get_response_from_llm

# Quality validator integration
from .content_quality_validator import ContentQualityValidator, ValidationConfig, ContentQualityResult


class FilterResult(Enum):
    """Content filtering result states"""
    APPROVED = "approved"
    REJECTED = "rejected"
    HUMAN_REVIEW = "human_review"
    NEEDS_REVISION = "needs_revision"
    QUARANTINED = "quarantined"


class FilterLevel(Enum):
    """Content filtering strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


@dataclass
class FilterRule:
    """Individual content filter rule"""
    rule_id: str
    name: str
    description: str
    filter_type: str  # 'text', 'image', 'video', 'metadata'
    pattern: Optional[str]  # Regex pattern for text filters
    threshold: float  # Confidence threshold for AI filters
    severity: str  # 'critical', 'high', 'medium', 'low'
    action: FilterResult
    platforms: List[str]  # Applicable platforms
    enabled: bool = True


@dataclass
class FilteringResult:
    """Comprehensive content filtering result"""
    content_id: str
    content_type: str
    platform: str
    final_decision: FilterResult
    confidence_score: float
    triggered_rules: List[FilterRule]
    safety_scores: Dict[str, float]
    policy_compliance: Dict[str, bool]
    brand_safety_score: float
    human_review_reasons: List[str]
    processing_time: float
    recommendations: List[str]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class PlatformPolicy:
    """Platform-specific policy requirements"""
    platform: str
    adult_content: bool = False
    violence_threshold: float = 0.1
    hate_speech_threshold: float = 0.05
    spam_threshold: float = 0.3
    misinformation_check: bool = True
    copyright_check: bool = True
    trademark_check: bool = True
    required_disclosures: List[str] = None
    restricted_keywords: List[str] = None
    max_promotional_ratio: float = 0.3


class ContentFilter:
    """
    Multi-layer content filtering system with comprehensive safety and compliance checks
    
    Features:
    - Brand safety filtering using AI and rule-based systems
    - Platform policy compliance validation
    - Automated content moderation with human review triggers
    - Quality gates with configurable thresholds
    - A/B testing support for filter effectiveness
    - Real-time policy updates and rule management
    """
    
    def __init__(self, db_session=None, filter_level: FilterLevel = FilterLevel.MODERATE):
        # Initialize quality validator
        self.quality_validator = ContentQualityValidator(db_session)
        
        # Database integration
        self.db_manager = None
        if DATABASE_AVAILABLE and db_session:
            self.db_manager = DatabaseManager(db_session)
        
        # Filter configuration
        self.filter_level = filter_level
        
        # Initialize AI clients
        self._init_ai_clients()
        
        # Initialize ML models
        self._init_ml_models()
        
        # Load filter rules and policies
        self._load_filter_rules()
        self._load_platform_policies()
        self._load_brand_safety_config()
        
        # Content caching and tracking
        self.filter_cache = {}
        self.filtering_history = []
        
        # Threading setup
        self.filter_lock = threading.Lock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Content Filter initialized with {filter_level.value} level")

    def _init_ai_clients(self):
        """Initialize AI clients for content analysis"""
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

    def _init_ml_models(self):
        """Initialize ML models for content filtering"""
        try:
            # Toxicity detection
            self.toxicity_detector = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                return_all_scores=True
            )
            
            # Hate speech detection
            self.hate_speech_detector = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                return_all_scores=True
            )
            
            # Adult content detection (would need specialized model)
            # self.adult_content_detector = pipeline("image-classification", model="...")
            
            self.logger.info("ML filtering models loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"ML models not available: {e}")
            self.toxicity_detector = None
            self.hate_speech_detector = None

    def _load_filter_rules(self):
        """Load content filtering rules based on filter level"""
        self.filter_rules = []
        
        # Text-based filter rules
        text_rules = [
            FilterRule(
                rule_id="profanity_check",
                name="Profanity Filter",
                description="Detect and filter profane language",
                filter_type="text",
                pattern=None,  # Uses better_profanity library
                threshold=0.8,
                severity="high",
                action=FilterResult.NEEDS_REVISION,
                platforms=["instagram", "tiktok", "x", "linkedin"]
            ),
            FilterRule(
                rule_id="spam_detection",
                name="Spam Content Detection",
                description="Detect spammy or promotional content",
                filter_type="text",
                pattern=r"(buy now|click here|limited time|act fast){2,}",
                threshold=0.7,
                severity="medium",
                action=FilterResult.HUMAN_REVIEW,
                platforms=["instagram", "tiktok", "x", "linkedin"]
            ),
            FilterRule(
                rule_id="all_caps_check",
                name="Excessive Caps Check",
                description="Flag content with too many capital letters",
                filter_type="text",
                pattern=r"[A-Z]{10,}",
                threshold=0.5,
                severity="low",
                action=FilterResult.NEEDS_REVISION,
                platforms=["instagram", "tiktok", "x", "linkedin"]
            ),
            FilterRule(
                rule_id="medical_claims",
                name="Medical Claims Filter",
                description="Flag potentially false medical claims",
                filter_type="text",
                pattern=r"(cure|heal|medical|doctor|treatment|medicine)",
                threshold=0.8,
                severity="critical",
                action=FilterResult.HUMAN_REVIEW,
                platforms=["instagram", "tiktok", "x", "linkedin"]
            ),
            FilterRule(
                rule_id="financial_advice",
                name="Financial Advice Filter",
                description="Flag potential financial advice without disclaimers",
                filter_type="text",
                pattern=r"(investment|guaranteed returns|get rich|money back)",
                threshold=0.7,
                severity="high",
                action=FilterResult.HUMAN_REVIEW,
                platforms=["instagram", "tiktok", "x", "linkedin"]
            )
        ]
        
        # Image-based filter rules
        image_rules = [
            FilterRule(
                rule_id="image_quality_gate",
                name="Image Quality Gate",
                description="Ensure minimum image quality standards",
                filter_type="image",
                pattern=None,
                threshold=70.0,  # Quality score threshold
                severity="medium",
                action=FilterResult.NEEDS_REVISION,
                platforms=["instagram", "tiktok", "x", "linkedin"]
            ),
            FilterRule(
                rule_id="brand_logo_check",
                name="Brand Logo Compliance",
                description="Ensure proper brand logo usage",
                filter_type="image",
                pattern=None,
                threshold=0.8,
                severity="high",
                action=FilterResult.HUMAN_REVIEW,
                platforms=["instagram", "tiktok", "x", "linkedin"]
            )
        ]
        
        # Video-based filter rules
        video_rules = [
            FilterRule(
                rule_id="video_quality_gate",
                name="Video Quality Gate",
                description="Ensure minimum video quality standards",
                filter_type="video",
                pattern=None,
                threshold=75.0,
                severity="medium",
                action=FilterResult.NEEDS_REVISION,
                platforms=["instagram", "tiktok", "x", "linkedin"]
            ),
            FilterRule(
                rule_id="audio_compliance",
                name="Audio Content Compliance",
                description="Check for copyrighted or inappropriate audio",
                filter_type="video",
                pattern=None,
                threshold=0.8,
                severity="high",
                action=FilterResult.HUMAN_REVIEW,
                platforms=["tiktok", "instagram"]
            )
        ]
        
        # Combine all rules
        all_rules = text_rules + image_rules + video_rules
        
        # Filter rules based on filter level
        if self.filter_level == FilterLevel.STRICT:
            self.filter_rules = [rule for rule in all_rules if rule.severity in ["critical", "high", "medium"]]
        elif self.filter_level == FilterLevel.MODERATE:
            self.filter_rules = [rule for rule in all_rules if rule.severity in ["critical", "high"]]
        else:  # PERMISSIVE
            self.filter_rules = [rule for rule in all_rules if rule.severity == "critical"]
        
        self.logger.info(f"Loaded {len(self.filter_rules)} filter rules for {self.filter_level.value} level")

    def _load_platform_policies(self):
        """Load platform-specific policy requirements"""
        self.platform_policies = {
            'instagram': PlatformPolicy(
                platform='instagram',
                adult_content=False,
                violence_threshold=0.1,
                hate_speech_threshold=0.05,
                spam_threshold=0.25,
                misinformation_check=True,
                copyright_check=True,
                trademark_check=True,
                required_disclosures=['#ad', '#sponsored', '#partnership'],
                restricted_keywords=['contest', 'giveaway', 'win', 'prize'],
                max_promotional_ratio=0.25
            ),
            'tiktok': PlatformPolicy(
                platform='tiktok',
                adult_content=False,
                violence_threshold=0.15,
                hate_speech_threshold=0.05,
                spam_threshold=0.3,
                misinformation_check=True,
                copyright_check=True,
                trademark_check=True,
                required_disclosures=['#ad', '#sponsored'],
                restricted_keywords=['dangerous', 'harmful', 'illegal'],
                max_promotional_ratio=0.2
            ),
            'x': PlatformPolicy(
                platform='x',
                adult_content=False,
                violence_threshold=0.2,
                hate_speech_threshold=0.05,
                spam_threshold=0.4,
                misinformation_check=True,
                copyright_check=True,
                trademark_check=False,
                required_disclosures=['#ad'],
                restricted_keywords=['hate', 'violence', 'terrorism'],
                max_promotional_ratio=0.3
            ),
            'linkedin': PlatformPolicy(
                platform='linkedin',
                adult_content=False,
                violence_threshold=0.05,
                hate_speech_threshold=0.02,
                spam_threshold=0.2,
                misinformation_check=True,
                copyright_check=True,
                trademark_check=True,
                required_disclosures=['#ad', '#sponsored', '#partnership'],
                restricted_keywords=['unprofessional', 'inappropriate'],
                max_promotional_ratio=0.15
            )
        }

    def _load_brand_safety_config(self):
        """Load brand safety configuration"""
        self.brand_safety_config = {
            'blocked_categories': [
                'adult_content',
                'violence',
                'hate_speech',
                'illegal_activities',
                'spam',
                'misinformation',
                'copyright_violation'
            ],
            'risk_thresholds': {
                'low': 0.1,
                'medium': 0.3,
                'high': 0.7,
                'critical': 0.9
            },
            'brand_keywords': {
                'positive': ['quality', 'premium', 'innovative', 'trusted', 'reliable'],
                'negative': ['cheap', 'fake', 'scam', 'ripoff', 'terrible'],
                'restricted': ['controversial', 'political', 'religious']
            },
            'compliance_requirements': {
                'fda_disclaimer': r'(medicine|health|cure|treatment)',
                'financial_disclaimer': r'(investment|guaranteed|returns|profit)',
                'legal_disclaimer': r'(legal|advice|lawsuit|court)'
            }
        }

    async def filter_content(self, content_path: str, content_type: str, 
                           platform: str, quality_result: Optional[ContentQualityResult] = None,
                           product_data: Optional[Dict[str, Any]] = None) -> FilteringResult:
        """
        Comprehensive multi-layer content filtering
        
        Args:
            content_path: Path to content file or text content
            content_type: 'image', 'video', or 'text'
            platform: Target platform
            quality_result: Optional pre-computed quality validation result
            product_data: Optional product information for context
        
        Returns:
            FilteringResult with filtering decision and details
        """
        start_time = time.time()
        content_id = f"filter_{content_type}_{platform}_{int(start_time)}"
        
        self.logger.info(f"Starting content filtering: {content_id}")
        
        # Initialize result
        result = FilteringResult(
            content_id=content_id,
            content_type=content_type,
            platform=platform,
            final_decision=FilterResult.APPROVED,
            confidence_score=1.0,
            triggered_rules=[],
            safety_scores={},
            policy_compliance={},
            brand_safety_score=100.0,
            human_review_reasons=[],
            processing_time=0.0,
            recommendations=[],
            metadata={'filter_level': self.filter_level.value},
            created_at=datetime.utcnow()
        )
        
        try:
            # Step 1: Quality validation (if not provided)
            if quality_result is None:
                config = ValidationConfig(
                    platform=platform,
                    content_type=content_type,
                    quality_tier='standard'
                )
                quality_result = await self.quality_validator.validate_content(
                    content_path, content_type, platform, config, product_data
                )
            
            result.metadata['quality_result'] = asdict(quality_result)
            
            # Step 2: Apply filter rules
            await self._apply_filter_rules(content_path, content_type, platform, result, product_data)
            
            # Step 3: Platform policy compliance
            await self._check_platform_compliance(content_path, content_type, platform, result)
            
            # Step 4: Brand safety analysis
            await self._analyze_brand_safety(content_path, content_type, result, product_data)
            
            # Step 5: AI-powered content moderation
            if self.openai_client or self.anthropic_client:
                await self._ai_content_moderation(content_path, content_type, platform, result)
            
            # Step 6: Final decision logic
            self._make_final_decision(result, quality_result)
            
            # Step 7: Generate recommendations
            await self._generate_filter_recommendations(result, quality_result)
            
            # Store filtering result
            if self.db_manager:
                self._store_filtering_result(result)
            
            # Calculate processing time
            result.processing_time = time.time() - start_time
            
            self.logger.info(
                f"Content filtering completed: {content_id} - "
                f"Decision: {result.final_decision.value}, "
                f"Confidence: {result.confidence_score:.2f}, "
                f"Time: {result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content filtering failed for {content_id}: {e}")
            result.final_decision = FilterResult.QUARANTINED
            result.human_review_reasons.append(f"Filtering error: {str(e)}")
            result.processing_time = time.time() - start_time
            result.metadata['error'] = str(e)
            return result

    async def _apply_filter_rules(self, content_path: str, content_type: str, 
                                platform: str, result: FilteringResult,
                                product_data: Optional[Dict[str, Any]]):
        """Apply configured filter rules to content"""
        applicable_rules = [
            rule for rule in self.filter_rules 
            if rule.filter_type == content_type or rule.filter_type == 'metadata'
            and platform in rule.platforms and rule.enabled
        ]
        
        for rule in applicable_rules:
            try:
                violated = await self._check_filter_rule(content_path, content_type, rule, product_data)
                
                if violated:
                    result.triggered_rules.append(rule)
                    
                    # Update confidence based on severity
                    severity_impact = {
                        'critical': 0.8,
                        'high': 0.6,
                        'medium': 0.4,
                        'low': 0.2
                    }
                    result.confidence_score *= (1 - severity_impact.get(rule.severity, 0.3))
                    
                    self.logger.warning(f"Filter rule triggered: {rule.name} for {result.content_id}")
                    
            except Exception as e:
                self.logger.error(f"Filter rule {rule.rule_id} failed: {e}")
                continue

    async def _check_filter_rule(self, content_path: str, content_type: str, 
                               rule: FilterRule, product_data: Optional[Dict[str, Any]]) -> bool:
        """Check if content violates a specific filter rule"""
        
        if content_type == 'text':
            # Read text content
            if os.path.exists(content_path):
                with open(content_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = content_path  # Assume it's the text content itself
            
            return await self._check_text_rule(text, rule)
            
        elif content_type == 'image':
            return await self._check_image_rule(content_path, rule)
            
        elif content_type == 'video':
            return await self._check_video_rule(content_path, rule)
            
        return False

    async def _check_text_rule(self, text: str, rule: FilterRule) -> bool:
        """Check text content against filter rule"""
        
        if rule.rule_id == "profanity_check":
            # Use better_profanity library
            return profanity.contains_profanity(text)
            
        elif rule.rule_id == "spam_detection":
            # Check for spam patterns
            if rule.pattern:
                spam_matches = len(re.findall(rule.pattern, text, re.IGNORECASE))
                return spam_matches > 0
            
            # Use ML model if available
            if self.toxicity_detector:
                try:
                    scores = self.toxicity_detector(text)
                    spam_score = max([s['score'] for s in scores if 'spam' in s['label'].lower()])
                    return spam_score > rule.threshold
                except:
                    pass
            
            return False
            
        elif rule.rule_id == "all_caps_check":
            if rule.pattern:
                return bool(re.search(rule.pattern, text))
            return False
            
        elif rule.rule_id == "medical_claims":
            if rule.pattern:
                medical_matches = re.findall(rule.pattern, text, re.IGNORECASE)
                if medical_matches:
                    # Check if proper disclaimers are present
                    disclaimers = ['not medical advice', 'consult doctor', 'fda approved']
                    has_disclaimer = any(disclaimer in text.lower() for disclaimer in disclaimers)
                    return not has_disclaimer  # Violation if no disclaimer
            return False
            
        elif rule.rule_id == "financial_advice":
            if rule.pattern:
                financial_matches = re.findall(rule.pattern, text, re.IGNORECASE)
                if financial_matches:
                    # Check for financial disclaimers
                    disclaimers = ['not financial advice', 'past performance', 'risk warning']
                    has_disclaimer = any(disclaimer in text.lower() for disclaimer in disclaimers)
                    return not has_disclaimer
            return False
        
        return False

    async def _check_image_rule(self, image_path: str, rule: FilterRule) -> bool:
        """Check image content against filter rule"""
        
        if rule.rule_id == "image_quality_gate":
            # This would use the quality validation result
            # For now, return False (no violation)
            return False
            
        elif rule.rule_id == "brand_logo_check":
            # Would check for proper brand logo usage
            # This would require computer vision model
            return False
        
        return False

    async def _check_video_rule(self, video_path: str, rule: FilterRule) -> bool:
        """Check video content against filter rule"""
        
        if rule.rule_id == "video_quality_gate":
            # This would use the quality validation result
            return False
            
        elif rule.rule_id == "audio_compliance":
            # Would check audio for copyrighted content
            # This would require audio fingerprinting
            return False
        
        return False

    async def _check_platform_compliance(self, content_path: str, content_type: str, 
                                       platform: str, result: FilteringResult):
        """Check content against platform-specific policies"""
        policy = self.platform_policies.get(platform)
        if not policy:
            result.policy_compliance[platform] = True
            return
        
        compliance_issues = []
        
        # Read text content for analysis
        text_content = ""
        if content_type == 'text':
            if os.path.exists(content_path):
                with open(content_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            else:
                text_content = content_path
        
        # Check hate speech threshold
        if self.hate_speech_detector and text_content:
            try:
                hate_scores = self.hate_speech_detector(text_content)
                hate_score = max([s['score'] for s in hate_scores if 'hate' in s['label'].lower()] + [0])
                result.safety_scores['hate_speech'] = hate_score
                
                if hate_score > policy.hate_speech_threshold:
                    compliance_issues.append(f"Hate speech score {hate_score:.2f} exceeds threshold {policy.hate_speech_threshold}")
            except Exception as e:
                self.logger.warning(f"Hate speech detection failed: {e}")
        
        # Check spam threshold
        if self.toxicity_detector and text_content:
            try:
                toxicity_scores = self.toxicity_detector(text_content)
                spam_score = max([s['score'] for s in toxicity_scores if 'spam' in s['label'].lower()] + [0])
                result.safety_scores['spam'] = spam_score
                
                if spam_score > policy.spam_threshold:
                    compliance_issues.append(f"Spam score {spam_score:.2f} exceeds threshold {policy.spam_threshold}")
            except Exception as e:
                self.logger.warning(f"Spam detection failed: {e}")
        
        # Check required disclosures
        if policy.required_disclosures and text_content:
            promotional_keywords = ['buy', 'purchase', 'sale', 'discount', 'offer']
            is_promotional = any(keyword in text_content.lower() for keyword in promotional_keywords)
            
            if is_promotional:
                has_disclosure = any(disclosure in text_content for disclosure in policy.required_disclosures)
                if not has_disclosure:
                    compliance_issues.append(f"Promotional content missing required disclosure: {policy.required_disclosures}")
        
        # Check restricted keywords
        if policy.restricted_keywords and text_content:
            found_restricted = [keyword for keyword in policy.restricted_keywords if keyword in text_content.lower()]
            if found_restricted:
                compliance_issues.append(f"Contains restricted keywords: {found_restricted}")
        
        # Check promotional ratio
        if text_content:
            promotional_ratio = self._calculate_promotional_ratio(text_content)
            if promotional_ratio > policy.max_promotional_ratio:
                compliance_issues.append(f"Promotional ratio {promotional_ratio:.2f} exceeds limit {policy.max_promotional_ratio}")
        
        # Set compliance result
        result.policy_compliance[platform] = len(compliance_issues) == 0
        
        if compliance_issues:
            result.human_review_reasons.extend(compliance_issues)
            self.logger.warning(f"Platform compliance issues for {platform}: {compliance_issues}")

    def _calculate_promotional_ratio(self, text: str) -> float:
        """Calculate ratio of promotional content in text"""
        promotional_keywords = [
            'buy', 'purchase', 'sale', 'discount', 'offer', 'deal', 'price',
            'order', 'shop', 'cart', 'checkout', 'payment', 'shipping'
        ]
        
        words = text.lower().split()
        promotional_count = sum(1 for word in words if word in promotional_keywords)
        
        return promotional_count / len(words) if words else 0.0

    async def _analyze_brand_safety(self, content_path: str, content_type: str, 
                                  result: FilteringResult, product_data: Optional[Dict[str, Any]]):
        """Analyze content for brand safety issues"""
        brand_safety_score = 100.0
        safety_issues = []
        
        # Read content for analysis
        text_content = ""
        if content_type == 'text':
            if os.path.exists(content_path):
                with open(content_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            else:
                text_content = content_path
        
        # Check for negative brand keywords
        negative_keywords = self.brand_safety_config['brand_keywords']['negative']
        if text_content:
            found_negative = [kw for kw in negative_keywords if kw in text_content.lower()]
            if found_negative:
                brand_safety_score -= len(found_negative) * 10
                safety_issues.append(f"Contains negative keywords: {found_negative}")
        
        # Check for restricted topics
        restricted_keywords = self.brand_safety_config['brand_keywords']['restricted']
        if text_content:
            found_restricted = [kw for kw in restricted_keywords if kw in text_content.lower()]
            if found_restricted:
                brand_safety_score -= len(found_restricted) * 15
                safety_issues.append(f"Contains restricted topics: {found_restricted}")
        
        # Check compliance requirements
        compliance_reqs = self.brand_safety_config['compliance_requirements']
        if text_content:
            for req_type, pattern in compliance_reqs.items():
                if re.search(pattern, text_content, re.IGNORECASE):
                    # Check if appropriate disclaimer is present
                    disclaimer_patterns = {
                        'fda_disclaimer': r'(not fda approved|not medical advice|consult doctor)',
                        'financial_disclaimer': r'(not financial advice|past performance|risk warning)',
                        'legal_disclaimer': r'(not legal advice|consult attorney|for information only)'
                    }
                    
                    disclaimer_pattern = disclaimer_patterns.get(req_type)
                    if disclaimer_pattern and not re.search(disclaimer_pattern, text_content, re.IGNORECASE):
                        brand_safety_score -= 20
                        safety_issues.append(f"Missing {req_type} disclaimer")
        
        # Ensure minimum score
        brand_safety_score = max(0.0, brand_safety_score)
        result.brand_safety_score = brand_safety_score
        
        if safety_issues:
            result.human_review_reasons.extend(safety_issues)
            self.logger.warning(f"Brand safety issues found: {safety_issues}")

    async def _ai_content_moderation(self, content_path: str, content_type: str, 
                                   platform: str, result: FilteringResult):
        """AI-powered content moderation using OpenAI/Anthropic"""
        if not (self.openai_client or self.anthropic_client):
            return
        
        try:
            # Prepare content for AI analysis
            ai_prompt = f"""
            Analyze this {content_type} content for social media platform {platform} and identify any potential issues:
            
            1. Brand safety concerns
            2. Platform policy violations
            3. Inappropriate content
            4. Spam or misleading information
            5. Copyright or trademark issues
            
            Rate the content safety from 0-100 (100 = completely safe) and provide specific concerns if any.
            
            Content to analyze:
            """
            
            if content_type == 'text':
                # Read text content
                if os.path.exists(content_path):
                    with open(content_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                else:
                    text_content = content_path
                
                ai_prompt += f"\n\nText: {text_content}"
                
                # Use Claude for text analysis
                if self.anthropic_client:
                    client, model = create_client('claude-3-5-sonnet-20241022')
                    
                    ai_response, _ = get_response_from_llm(
                        msg=ai_prompt,
                        client=client,
                        model=model,
                        system_message="You are an expert content moderator. Analyze content for safety and policy compliance.",
                        print_debug=False
                    )
                    
                    # Parse AI response for safety score
                    import re
                    scores = re.findall(r'(\d+(?:\.\d+)?)', ai_response)
                    if scores:
                        ai_safety_score = float(scores[0])
                        result.safety_scores['ai_moderation'] = ai_safety_score
                        
                        if ai_safety_score < 70:
                            result.human_review_reasons.append(f"AI flagged content (score: {ai_safety_score})")
                            result.metadata['ai_analysis'] = ai_response
            
            elif content_type == 'image' and self.openai_client:
                # Use GPT-4 Vision for image analysis
                with open(content_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ai_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                
                ai_response = response.choices[0].message.content
                
                # Parse response for safety concerns
                if "concern" in ai_response.lower() or "issue" in ai_response.lower():
                    result.human_review_reasons.append("AI detected potential image safety issues")
                    result.metadata['ai_image_analysis'] = ai_response
                
        except Exception as e:
            self.logger.error(f"AI content moderation failed: {e}")

    def _make_final_decision(self, result: FilteringResult, quality_result: ContentQualityResult):
        """Make final filtering decision based on all analysis"""
        
        # Start with approved status
        decision = FilterResult.APPROVED
        decision_reasons = []
        
        # Check triggered rules
        critical_rules = [rule for rule in result.triggered_rules if rule.severity == 'critical']
        high_rules = [rule for rule in result.triggered_rules if rule.severity == 'high']
        
        if critical_rules:
            decision = FilterResult.REJECTED
            decision_reasons.append(f"Critical rule violations: {[r.name for r in critical_rules]}")
        
        elif high_rules:
            decision = FilterResult.HUMAN_REVIEW
            decision_reasons.append(f"High severity rule violations: {[r.name for r in high_rules]}")
        
        # Check quality score
        if quality_result and not quality_result.passed_validation:
            if quality_result.overall_score < 40:
                decision = FilterResult.REJECTED
                decision_reasons.append(f"Quality score too low: {quality_result.overall_score:.1f}")
            elif quality_result.overall_score < 60:
                decision = max(decision, FilterResult.NEEDS_REVISION)
                decision_reasons.append(f"Quality score below threshold: {quality_result.overall_score:.1f}")
        
        # Check brand safety score
        if result.brand_safety_score < 50:
            decision = FilterResult.REJECTED
            decision_reasons.append(f"Brand safety score too low: {result.brand_safety_score:.1f}")
        elif result.brand_safety_score < 70:
            decision = max(decision, FilterResult.HUMAN_REVIEW)
            decision_reasons.append(f"Brand safety concerns: {result.brand_safety_score:.1f}")
        
        # Check platform compliance
        non_compliant_platforms = [p for p, compliant in result.policy_compliance.items() if not compliant]
        if non_compliant_platforms:
            decision = max(decision, FilterResult.HUMAN_REVIEW)
            decision_reasons.append(f"Platform compliance issues: {non_compliant_platforms}")
        
        # Check confidence score
        if result.confidence_score < 0.5:
            decision = max(decision, FilterResult.HUMAN_REVIEW)
            decision_reasons.append(f"Low confidence score: {result.confidence_score:.2f}")
        
        # Check AI safety scores
        ai_safety = result.safety_scores.get('ai_moderation', 100)
        if ai_safety < 60:
            decision = max(decision, FilterResult.HUMAN_REVIEW)
            decision_reasons.append(f"AI safety concerns: {ai_safety:.1f}")
        
        # Override based on filter level
        if self.filter_level == FilterLevel.STRICT:
            # More conservative decisions
            if decision == FilterResult.NEEDS_REVISION:
                decision = FilterResult.HUMAN_REVIEW
        elif self.filter_level == FilterLevel.PERMISSIVE:
            # More lenient decisions
            if decision == FilterResult.HUMAN_REVIEW and result.confidence_score > 0.7:
                decision = FilterResult.APPROVED
        
        result.final_decision = decision
        result.metadata['decision_reasons'] = decision_reasons
        
        self.logger.info(f"Final decision for {result.content_id}: {decision.value} - {decision_reasons}")

    async def _generate_filter_recommendations(self, result: FilteringResult, 
                                             quality_result: Optional[ContentQualityResult]):
        """Generate recommendations for content improvement"""
        recommendations = []
        
        # Recommendations based on triggered rules
        for rule in result.triggered_rules:
            if rule.rule_id == "profanity_check":
                recommendations.append("Remove or replace inappropriate language")
            elif rule.rule_id == "spam_detection":
                recommendations.append("Reduce promotional language and excessive repetition")
            elif rule.rule_id == "all_caps_check":
                recommendations.append("Use normal capitalization for better readability")
            elif rule.rule_id == "medical_claims":
                recommendations.append("Add medical disclaimer or remove health claims")
            elif rule.rule_id == "financial_advice":
                recommendations.append("Add financial disclaimer or soften investment language")
        
        # Recommendations based on quality result
        if quality_result and not quality_result.passed_validation:
            recommendations.extend(quality_result.improvement_suggestions)
        
        # Recommendations based on brand safety
        if result.brand_safety_score < 80:
            recommendations.append("Review content for brand safety and remove controversial elements")
        
        # Recommendations based on platform compliance
        for platform, compliant in result.policy_compliance.items():
            if not compliant:
                policy = self.platform_policies.get(platform)
                if policy and policy.required_disclosures:
                    recommendations.append(f"Add required disclosure for {platform}: {policy.required_disclosures}")
        
        # AI-generated recommendations
        if result.metadata.get('ai_analysis'):
            recommendations.append("Review AI analysis for specific improvement suggestions")
        
        result.recommendations = list(set(recommendations))  # Remove duplicates

    def _store_filtering_result(self, result: FilteringResult):
        """Store filtering result for analytics and tracking"""
        if not self.db_manager:
            return
        
        try:
            # Store in filtering history for analytics
            self.filtering_history.append({
                'content_id': result.content_id,
                'platform': result.platform,
                'content_type': result.content_type,
                'decision': result.final_decision.value,
                'confidence': result.confidence_score,
                'brand_safety_score': result.brand_safety_score,
                'triggered_rules': [r.rule_id for r in result.triggered_rules],
                'timestamp': result.created_at,
                'processing_time': result.processing_time
            })
            
            self.logger.info(f"Filtering result stored: {result.content_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store filtering result: {e}")

    async def batch_filter_content(self, content_items: List[Dict[str, Any]]) -> List[FilteringResult]:
        """Filter multiple content items in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit filtering tasks
            future_to_item = {}
            
            for item in content_items:
                future = executor.submit(
                    asyncio.run,
                    self.filter_content(
                        item['content_path'],
                        item['content_type'],
                        item['platform'],
                        item.get('quality_result'),
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
                    self.logger.error(f"Batch filtering item failed: {e}")
                    # Create error result
                    error_result = FilteringResult(
                        content_id=f"error_{int(time.time())}",
                        content_type=future_to_item[future]['content_type'],
                        platform=future_to_item[future]['platform'],
                        final_decision=FilterResult.QUARANTINED,
                        confidence_score=0.0,
                        triggered_rules=[],
                        safety_scores={},
                        policy_compliance={},
                        brand_safety_score=0.0,
                        human_review_reasons=[f"Filtering failed: {str(e)}"],
                        processing_time=0.0,
                        recommendations=[],
                        metadata={'error': str(e)},
                        created_at=datetime.utcnow()
                    )
                    results.append(error_result)
        
        return results

    def get_filter_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get filtering analytics and performance metrics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_filtering = [
            f for f in self.filtering_history 
            if f['timestamp'] > cutoff_date
        ]
        
        if not recent_filtering:
            return {'message': 'No recent filtering data available'}
        
        # Calculate analytics
        total_filtered = len(recent_filtering)
        
        # Decision breakdown
        decision_counts = {}
        for result in recent_filtering:
            decision = result['decision']
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        decision_rates = {k: v/total_filtered for k, v in decision_counts.items()}
        
        # Platform breakdown
        platform_stats = {}
        for result in recent_filtering:
            platform = result['platform']
            if platform not in platform_stats:
                platform_stats[platform] = {
                    'total': 0, 'approved': 0, 'rejected': 0, 
                    'human_review': 0, 'avg_confidence': 0, 'avg_brand_safety': 0
                }
            
            stats = platform_stats[platform]
            stats['total'] += 1
            stats[result['decision']] = stats.get(result['decision'], 0) + 1
            stats['avg_confidence'] += result['confidence']
            stats['avg_brand_safety'] += result['brand_safety_score']
        
        # Calculate averages
        for platform, stats in platform_stats.items():
            if stats['total'] > 0:
                stats['avg_confidence'] /= stats['total']
                stats['avg_brand_safety'] /= stats['total']
                stats['approval_rate'] = stats['approved'] / stats['total']
        
        # Most triggered rules
        rule_counts = {}
        for result in recent_filtering:
            for rule_id in result['triggered_rules']:
                rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        
        most_triggered = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'period_days': days,
            'total_filtered': total_filtered,
            'decision_breakdown': decision_counts,
            'decision_rates': decision_rates,
            'platform_breakdown': platform_stats,
            'average_confidence': np.mean([f['confidence'] for f in recent_filtering]),
            'average_brand_safety': np.mean([f['brand_safety_score'] for f in recent_filtering]),
            'average_processing_time': np.mean([f['processing_time'] for f in recent_filtering]),
            'most_triggered_rules': most_triggered,
            'filter_level': self.filter_level.value
        }

    def update_filter_rules(self, new_rules: List[FilterRule]):
        """Update filter rules dynamically"""
        # Validate new rules
        valid_rules = []
        for rule in new_rules:
            if rule.rule_id and rule.name and rule.filter_type:
                valid_rules.append(rule)
            else:
                self.logger.warning(f"Invalid filter rule skipped: {rule}")
        
        # Update rules
        existing_rule_ids = {rule.rule_id for rule in self.filter_rules}
        
        for rule in valid_rules:
            if rule.rule_id in existing_rule_ids:
                # Update existing rule
                for i, existing_rule in enumerate(self.filter_rules):
                    if existing_rule.rule_id == rule.rule_id:
                        self.filter_rules[i] = rule
                        break
            else:
                # Add new rule
                self.filter_rules.append(rule)
        
        self.logger.info(f"Filter rules updated: {len(valid_rules)} rules processed")

    def get_filter_status(self) -> Dict[str, Any]:
        """Get current filter system status"""
        return {
            'filter_level': self.filter_level.value,
            'total_rules': len(self.filter_rules),
            'enabled_rules': len([r for r in self.filter_rules if r.enabled]),
            'ai_clients_available': {
                'openai': self.openai_client is not None,
                'anthropic': self.anthropic_client is not None
            },
            'ml_models_available': {
                'toxicity_detector': self.toxicity_detector is not None,
                'hate_speech_detector': self.hate_speech_detector is not None
            },
            'supported_platforms': list(self.platform_policies.keys()),
            'database_connected': self.db_manager is not None,
            'filter_cache_size': len(self.filter_cache),
            'filtering_history_size': len(self.filtering_history)
        }


# Example usage and testing functions
async def test_content_filter():
    """Test the content filtering system"""
    content_filter = ContentFilter(filter_level=FilterLevel.MODERATE)
    
    print("Content Filter initialized successfully!")
    print(f"Filter level: {content_filter.filter_level.value}")
    print(f"Total filter rules: {len(content_filter.filter_rules)}")
    
    # Test system status
    status = content_filter.get_filter_status()
    print(f"\nFilter System Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test sample text filtering
    sample_text = "Check out this amazing product! Buy now and save 50%! #ad"
    
    try:
        result = await content_filter.filter_content(
            sample_text, 'text', 'instagram'
        )
        
        print(f"\nSample filtering result:")
        print(f"  Decision: {result.final_decision.value}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Brand Safety: {result.brand_safety_score:.1f}")
        print(f"  Triggered Rules: {[r.name for r in result.triggered_rules]}")
        print(f"  Recommendations: {result.recommendations}")
        
    except Exception as e:
        print(f"Filtering test failed: {e}")
    
    print("\nContent Filter ready for multi-layer content filtering!")


if __name__ == "__main__":
    import asyncio
    import base64
    import numpy as np
    
    # Run async test
    asyncio.run(test_content_filter())