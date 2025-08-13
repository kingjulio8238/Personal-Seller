"""
Comprehensive Content Generation Pipeline
Orchestrates image enhancement, video generation, and text creation for all 5 content types
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from decimal import Decimal

from .image_enhancer import ImageEnhancer
from .video_generator import VideoGenerator
from .text_generator import TextGenerator, ContentRequest
from .content_quality_validator import ContentQualityValidator, ValidationConfig
from .content_filter import ContentFilter, FilterLevel, FilterResult
from .quality_improvement_engine import QualityImprovementEngine
from ..database.models import Product, Post, EngagementMetrics, DatabaseManager


@dataclass
class ContentGenerationRequest:
    """Request structure for comprehensive content generation"""
    product_id: int
    base_image_url: str
    platforms: List[str]
    content_types: List[str]
    priority: str = 'normal'  # 'high', 'normal', 'low'
    deadline: Optional[datetime] = None
    budget_limit: Optional[Decimal] = None
    quality_tier: str = 'premium'  # 'premium', 'standard', 'basic'


@dataclass
class ContentGenerationResult:
    """Result structure for comprehensive content generation"""
    request_id: str
    product_id: int
    total_items_created: int
    images: Dict[str, List[str]]  # platform -> list of image paths
    videos: Dict[str, List[str]]  # platform -> list of video paths
    text_content: Dict[str, List[Dict[str, Any]]]  # platform -> list of text content
    quality_results: Dict[str, List[Dict[str, Any]]]  # platform -> quality validation results
    filter_results: Dict[str, List[Dict[str, Any]]]  # platform -> filter results
    approved_content: Dict[str, List[str]]  # platform -> approved content paths
    rejected_content: Dict[str, List[str]]  # platform -> rejected content paths
    needs_review: Dict[str, List[str]]  # platform -> content needing human review
    quality_scores: Dict[str, float]  # platform -> average quality scores
    metadata: Dict[str, Any]
    cost_summary: Dict[str, str]
    processing_time: float
    success_rate: float
    quality_pass_rate: float
    created_at: datetime


class ContentPipeline:
    """
    Comprehensive content generation pipeline manager
    Coordinates image enhancement, video generation, and text creation
    """
    
    def __init__(self, db_session=None, enable_quality_validation=True, enable_automated_improvements=True):
        # Initialize component generators
        self.image_enhancer = ImageEnhancer()
        self.video_generator = VideoGenerator()
        self.text_generator = TextGenerator()
        
        # Initialize quality validation system
        self.enable_quality_validation = enable_quality_validation
        self.enable_automated_improvements = enable_automated_improvements
        
        if enable_quality_validation:
            self.quality_validator = ContentQualityValidator(db_session)
            self.content_filter = ContentFilter(db_session, FilterLevel.MODERATE)
            
            if enable_automated_improvements:
                self.improvement_engine = QualityImprovementEngine(db_session)
            else:
                self.improvement_engine = None
        else:
            self.quality_validator = None
            self.content_filter = None
            self.improvement_engine = None
        
        # Database manager
        if db_session:
            self.db_manager = DatabaseManager(db_session)
        else:
            self.db_manager = None
        
        # Content type specifications
        self.content_type_specs = {
            'text_only': {
                'requires_image': False,
                'requires_video': False,
                'text_variants': 5,
                'processing_time': 30  # seconds
            },
            'text_image': {
                'requires_image': True,
                'requires_video': False,
                'image_variants': 10,
                'text_variants': 3,
                'processing_time': 180  # seconds
            },
            'text_video': {
                'requires_image': True,  # Video needs base image
                'requires_video': True,
                'video_variants': 5,
                'text_variants': 3,
                'processing_time': 300  # seconds
            },
            'image_carousel': {
                'requires_image': True,
                'requires_video': False,
                'image_variants': 15,
                'text_variants': 2,
                'processing_time': 240  # seconds
            },
            'story_series': {
                'requires_image': True,
                'requires_video': True,
                'image_variants': 8,
                'video_variants': 3,
                'text_variants': 8,  # One text per story frame
                'processing_time': 360  # seconds
            }
        }
        
        # Platform priorities for batch processing
        self.platform_priorities = {
            'high': ['instagram', 'tiktok', 'x'],
            'normal': ['instagram', 'tiktok', 'x', 'instagram_story', 'linkedin'],
            'low': ['x', 'linkedin']
        }
        
        self.temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_comprehensive_content(self, request: ContentGenerationRequest) -> ContentGenerationResult:
        """
        Create comprehensive content suite for all specified types and platforms
        Target: 10 enhanced image variants + 5 video variants per upload in under 60 seconds
        """
        start_time = time.time()
        request_id = f"content_{int(start_time)}"
        
        self.logger.info(f"Starting comprehensive content creation: {request_id}")
        
        # Initialize result structure
        result = ContentGenerationResult(
            request_id=request_id,
            product_id=request.product_id,
            total_items_created=0,
            images={},
            videos={},
            text_content={},
            quality_results={},
            filter_results={},
            approved_content={},
            rejected_content={},
            needs_review={},
            quality_scores={},
            metadata={'quality_validation_enabled': self.enable_quality_validation},
            cost_summary={},
            processing_time=0.0,
            success_rate=0.0,
            quality_pass_rate=0.0,
            created_at=datetime.utcnow()
        )
        
        try:
            # Get product data from database
            product_data = self._get_product_data(request.product_id)
            if not product_data:
                raise ValueError(f"Product {request.product_id} not found")
            
            # Determine target platforms based on priority
            target_platforms = self._get_target_platforms(request.platforms, request.priority)
            
            # Process each content type
            total_successes = 0
            total_attempts = 0
            
            for content_type in request.content_types:
                self.logger.info(f"Processing content type: {content_type}")
                
                # Process content type for all platforms
                content_result = self._process_content_type(
                    content_type, 
                    target_platforms, 
                    request, 
                    product_data
                )
                
                # Merge results
                self._merge_content_results(result, content_result)
                
                total_attempts += len(target_platforms)
                total_successes += content_result.get('successes', 0)
            
            # Calculate final metrics
            end_time = time.time()
            result.processing_time = end_time - start_time
            result.success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0
            result.cost_summary = self._get_total_cost_summary()
            result.total_items_created = self._count_total_items(result)
            
            # Calculate quality metrics
            if self.enable_quality_validation:
                result.quality_pass_rate = self._calculate_quality_pass_rate(result)
                result.quality_scores = self._calculate_average_quality_scores(result)
            
            # Store results in database if available
            if self.db_manager:
                self._store_content_results(result, product_data)
            
            self.logger.info(
                f"Content creation completed: {result.total_items_created} items in "
                f"{result.processing_time:.2f}s (Success rate: {result.success_rate:.2%})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content creation failed: {e}")
            result.processing_time = time.time() - start_time
            result.metadata['error'] = str(e)
            return result

    def _get_product_data(self, product_id: int) -> Optional[Dict[str, Any]]:
        """Get product data from database or return mock data"""
        if self.db_manager:
            # In production, fetch from database
            # product = self.db_manager.session.query(Product).get(product_id)
            # return product data...
            pass
        
        # Mock product data for development
        return {
            'id': product_id,
            'name': f'Product {product_id}',
            'description': 'High-quality product with premium features',
            'category': 'electronics',
            'features': 'Premium quality, innovative design, excellent value',
            'price': 299.99,
            'target_audience': 'tech-savvy consumers',
            'brand_voice': 'modern and innovative'
        }

    def _get_target_platforms(self, requested_platforms: List[str], priority: str) -> List[str]:
        """Get target platforms based on priority"""
        priority_platforms = self.platform_priorities.get(priority, self.platform_priorities['normal'])
        
        # Filter requested platforms by priority
        if requested_platforms:
            return [p for p in priority_platforms if p in requested_platforms]
        else:
            return priority_platforms

    def _process_content_type(self, content_type: str, platforms: List[str], 
                            request: ContentGenerationRequest, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific content type for all platforms"""
        content_spec = self.content_type_specs.get(content_type, self.content_type_specs['text_image'])
        
        result = {
            'images': {},
            'videos': {},
            'text_content': {},
            'successes': 0,
            'metadata': {'content_type': content_type}
        }
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(3, len(platforms))) as executor:
            # Submit platform processing tasks
            future_to_platform = {}
            
            for platform in platforms:
                future = executor.submit(
                    self._process_platform_content,
                    platform,
                    content_type,
                    content_spec,
                    request,
                    product_data
                )
                future_to_platform[future] = platform
            
            # Collect results
            for future in as_completed(future_to_platform):
                platform = future_to_platform[future]
                try:
                    platform_result = future.result(timeout=300)  # 5 minute timeout
                    
                    # Merge platform results
                    if platform_result.get('images'):
                        result['images'][platform] = platform_result['images']
                    if platform_result.get('videos'):
                        result['videos'][platform] = platform_result['videos']
                    if platform_result.get('text_content'):
                        result['text_content'][platform] = platform_result['text_content']
                    
                    if platform_result.get('success', False):
                        result['successes'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Platform {platform} processing failed: {e}")
                    continue
        
        return result

    def _process_platform_content(self, platform: str, content_type: str, content_spec: Dict[str, Any],
                                request: ContentGenerationRequest, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process content for a specific platform"""
        platform_result = {
            'images': [],
            'videos': [],
            'text_content': [],
            'success': False
        }
        
        try:
            # Generate images if required
            if content_spec.get('requires_image', False):
                image_variants = content_spec.get('image_variants', 5)
                
                # Use batch processing for multiple variants
                if image_variants > 5:
                    # Use comprehensive batch creation
                    batch_result = self.image_enhancer.batch_create_variants(
                        request.base_image_url, 
                        product_data, 
                        [platform], 
                        image_variants
                    )
                    platform_result['images'] = batch_result.get(platform, [])
                else:
                    # Use single variant creation
                    enhanced_image = self.image_enhancer.enhance_for_platform(
                        request.base_image_url, platform, product_data
                    )
                    platform_result['images'] = [enhanced_image] if enhanced_image else []
            
            # Generate videos if required
            if content_spec.get('requires_video', False):
                video_variants = content_spec.get('video_variants', 3)
                
                # Use the first enhanced image as base for video
                base_image_path = platform_result['images'][0] if platform_result['images'] else None
                
                if base_image_path:
                    # Run async video generation in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        video_results = loop.run_until_complete(
                            self.video_generator.create_video_variants(
                                base_image_path, product_data, [platform]
                            )
                        )
                        # Extract successful video paths
                        video_paths = []
                        for platform_key, result in video_results.items():
                            if result.success and result.video_path:
                                video_paths.append(result.video_path)
                        platform_result['videos'] = video_paths
                    finally:
                        loop.close()
            
            # Generate text content
            text_variants = content_spec.get('text_variants', 3)
            text_results = []
            
            for i in range(text_variants):
                content_request = ContentRequest(
                    platform=platform,
                    content_type=content_type,
                    product_data=product_data,
                    brand_voice=product_data.get('brand_voice', 'professional'),
                    target_audience=product_data.get('target_audience', 'general audience')
                )
                
                generated_content = self.text_generator.create_platform_optimized_content(content_request)
                text_results.append({
                    'text': generated_content.text,
                    'hashtags': generated_content.hashtags,
                    'cta': generated_content.call_to_action,
                    'metadata': generated_content.metadata
                })
            
            platform_result['text_content'] = text_results
            platform_result['success'] = True
            
            self.logger.info(f"Platform {platform} content created successfully")
            
        except Exception as e:
            self.logger.error(f"Platform {platform} content creation failed: {e}")
            platform_result['success'] = False
        
        return platform_result

    def _merge_content_results(self, main_result: ContentGenerationResult, content_result: Dict[str, Any]):
        """Merge content results into main result"""
        # Merge images
        for platform, images in content_result.get('images', {}).items():
            if platform not in main_result.images:
                main_result.images[platform] = []
            main_result.images[platform].extend(images)
        
        # Merge videos
        for platform, videos in content_result.get('videos', {}).items():
            if platform not in main_result.videos:
                main_result.videos[platform] = []
            main_result.videos[platform].extend(videos)
        
        # Merge text content
        for platform, texts in content_result.get('text_content', {}).items():
            if platform not in main_result.text_content:
                main_result.text_content[platform] = []
            main_result.text_content[platform].extend(texts)

    def _get_total_cost_summary(self) -> Dict[str, str]:
        """Get comprehensive cost summary from all generators"""
        costs = {
            'image_costs': self.image_enhancer.get_cost_summary(),
            'video_costs': self.video_generator.get_cost_summary(),  # Real video generator cost tracking
            'text_costs': {'total_cost': 0.00, 'daily_cost': 0.00, 'currency': 'USD'}   # Text generator cost tracking
        }
        
        # Calculate total (properly handle different formats)
        total_daily = Decimal('0.00')
        for cost_type, cost_data in costs.items():
            if isinstance(cost_data, dict):
                daily_cost = cost_data.get('daily_cost', 0.00)
                if isinstance(daily_cost, str):
                    daily_cost = daily_cost.replace('$', '')
                total_daily += Decimal(str(daily_cost))
        
        costs['total_daily_cost'] = float(total_daily)
        costs['currency'] = 'USD'
        return costs

    def _count_total_items(self, result: ContentGenerationResult) -> int:
        """Count total items created across all content types"""
        total = 0
        
        # Count images
        for platform_images in result.images.values():
            total += len(platform_images)
        
        # Count videos
        for platform_videos in result.videos.values():
            total += len(platform_videos)
        
        # Count text content
        for platform_texts in result.text_content.values():
            total += len(platform_texts)
        
        return total
    
    async def _validate_and_filter_content(self, content_paths: List[str], content_type: str,
                                         platform: str, request: ContentGenerationRequest,
                                         product_data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Validate and filter content with quality checks and automated improvements
        
        Returns dict with 'approved', 'rejected', 'needs_review', 'quality_results', 'filter_results'
        """
        result = {
            'approved': [],
            'rejected': [],
            'needs_review': [],
            'quality_results': [],
            'filter_results': []
        }
        
        if not self.quality_validator or not self.content_filter:
            # If validation disabled, approve all content
            result['approved'] = content_paths
            return result
        
        # Configure validation
        config = ValidationConfig(
            platform=platform,
            content_type=content_type,
            quality_tier=request.quality_tier,
            enable_ai_analysis=True,
            enable_technical_analysis=True,
            enable_brand_compliance=True,
            enable_engagement_prediction=True
        )
        
        for content_path in content_paths:
            try:
                # Quality validation
                quality_result = await self.quality_validator.validate_content(
                    content_path, content_type, platform, config, product_data
                )
                result['quality_results'].append(asdict(quality_result))
                
                # Content filtering
                filter_result = await self.content_filter.filter_content(
                    content_path, content_type, platform, quality_result, product_data
                )
                result['filter_results'].append(asdict(filter_result))
                
                # Determine content fate
                if (quality_result.passed_validation and 
                    filter_result.final_decision == FilterResult.APPROVED):
                    result['approved'].append(content_path)
                    
                elif (filter_result.final_decision == FilterResult.REJECTED or
                      quality_result.overall_score < 30):
                    result['rejected'].append(content_path)
                    
                    # Try automated improvement if enabled
                    if (self.enable_automated_improvements and self.improvement_engine and
                        quality_result.overall_score > 20):
                        
                        improved_content = await self._attempt_automated_improvement(
                            content_path, quality_result, filter_result, content_type, platform, product_data
                        )
                        
                        if improved_content:
                            result['approved'].append(improved_content)
                            self.logger.info(f"Content improved and approved: {content_path}")
                        else:
                            self.logger.warning(f"Content improvement failed: {content_path}")
                            
                else:
                    result['needs_review'].append(content_path)
                    
            except Exception as e:
                self.logger.error(f"Content validation failed for {content_path}: {e}")
                result['rejected'].append(content_path)
        
        return result
    
    async def _attempt_automated_improvement(self, content_path: str, quality_result, filter_result,
                                           content_type: str, platform: str, 
                                           product_data: Dict[str, Any]) -> Optional[str]:
        """
        Attempt automated content improvement
        
        Returns improved content path if successful, None if failed
        """
        if not self.improvement_engine:
            return None
        
        try:
            # Generate improvement plan
            improvement_plan = await self.improvement_engine.analyze_improvement_opportunities(
                quality_result, filter_result, content_path, product_data
            )
            
            # Apply automated improvements
            if improvement_plan.automated_fixes:
                retry_result = await self.improvement_engine.apply_automated_improvements(
                    improvement_plan, content_path, max_attempts=2
                )
                
                if retry_result.success and retry_result.new_quality_result:
                    self.logger.info(
                        f"Content improved: score {quality_result.overall_score:.1f} -> "
                        f"{retry_result.new_quality_result.overall_score:.1f}"
                    )
                    return retry_result.new_quality_result.content_id  # This would be the new path
            
        except Exception as e:
            self.logger.error(f"Automated improvement failed: {e}")
        
        return None
    
    def _calculate_quality_pass_rate(self, result: ContentGenerationResult) -> float:
        """Calculate overall quality pass rate"""
        total_items = 0
        passed_items = 0
        
        # Count approved vs total content
        for platform in result.approved_content:
            total_items += len(result.approved_content[platform])
            passed_items += len(result.approved_content[platform])
        
        for platform in result.rejected_content:
            total_items += len(result.rejected_content[platform])
        
        for platform in result.needs_review:
            total_items += len(result.needs_review[platform])
        
        return passed_items / total_items if total_items > 0 else 0.0
    
    def _calculate_average_quality_scores(self, result: ContentGenerationResult) -> Dict[str, float]:
        """Calculate average quality scores by platform"""
        platform_scores = {}
        
        for platform, quality_results in result.quality_results.items():
            if quality_results:
                scores = [qr.get('overall_score', 0) for qr in quality_results]
                platform_scores[platform] = sum(scores) / len(scores) if scores else 0.0
            else:
                platform_scores[platform] = 0.0
        
        return platform_scores

    def _store_content_results(self, result: ContentGenerationResult, product_data: Dict[str, Any]):
        """Store content results in database"""
        if not self.db_manager:
            return
        
        try:
            # Store posts for each platform/content combination
            for platform in result.images.keys():
                # Create post entries for tracking
                post = self.db_manager.create_post(
                    platform=platform,
                    product_id=result.product_id,
                    content_type='text_image',
                    agent_generation_id=1,  # Would be dynamic in production
                    image_url=result.images[platform][0] if result.images[platform] else None,
                    video_url=result.videos[platform][0] if result.videos.get(platform) else None,
                    caption=result.text_content[platform][0]['text'] if result.text_content.get(platform) else None
                )
                
            self.logger.info(f"Content results stored in database for product {result.product_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store content results: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and performance metrics"""
        return {
            'image_enhancer_status': self.image_enhancer.get_api_status(),
            'video_generator_status': {'status': 'operational'},  # Would be dynamic
            'text_generator_status': {'status': 'operational'},   # Would be dynamic
            'supported_content_types': list(self.content_type_specs.keys()),
            'platform_priorities': self.platform_priorities,
            'temp_directory': self.temp_dir,
            'database_connected': self.db_manager is not None
        }

    def cleanup_temporary_files(self, max_age_hours: int = 6):
        """Clean up temporary files from all generators"""
        try:
            # Cleanup image enhancer cache
            self.image_enhancer.cleanup_old_cache(max_age_hours)
            
            # Cleanup video generator temp files
            self.video_generator.cleanup_temp_files(max_age_hours)
            
            # Cleanup main temp directory
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        self.logger.info(f"Cleaned up temp file: {filename}")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def create_sample_content_request() -> ContentGenerationRequest:
    """Create sample content generation request for testing"""
    return ContentGenerationRequest(
        product_id=1,
        base_image_url="https://example.com/product-image.jpg",
        platforms=['instagram', 'tiktok', 'x'],
        content_types=['text_image', 'text_video'],
        priority='high',
        deadline=datetime.utcnow() + timedelta(hours=2),
        budget_limit=Decimal('50.00'),
        quality_tier='premium'
    )


if __name__ == "__main__":
    # Test the content pipeline
    pipeline = ContentPipeline()
    
    print("Content Pipeline initialized successfully!")
    print(f"Supported content types: {list(pipeline.content_type_specs.keys())}")
    print(f"Platform priorities: {pipeline.platform_priorities}")
    
    # Test pipeline status
    status = pipeline.get_pipeline_status()
    print(f"\nPipeline Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test sample request creation
    sample_request = create_sample_content_request()
    print(f"\nSample request: {sample_request.content_types} for {sample_request.platforms}")
    
    print("\nContent Pipeline ready for comprehensive content generation!")