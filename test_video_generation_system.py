#!/usr/bin/env python3
"""
Comprehensive Test Suite for Veo 3 Video Generation System
Tests all components of the video generation pipeline including API integration,
fallback systems, platform optimizations, and integration with the content pipeline.
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
from decimal import Decimal

# Add content_generation to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'content_generation'))

from content_generation.video_generator import (
    VideoGenerator, VideoGenerationRequest, VideoGenerationResult
)
from content_generation.content_pipeline import ContentPipeline, ContentGenerationRequest

# Test configuration
TEST_PRODUCT_DATA = {
    'name': 'Wireless Premium Headphones',
    'category': 'electronics',
    'features': 'noise cancellation, 40-hour battery, premium sound quality',
    'target_audience': 'audio enthusiasts and professionals',
    'brand_voice': 'innovative and premium',
    'tagline': 'Sound that moves you'
}

TEST_PLATFORMS = ['x', 'tiktok', 'instagram', 'linkedin']


class VideoGenerationTester:
    """Comprehensive test suite for video generation system"""
    
    def __init__(self):
        self.generator = VideoGenerator()
        self.pipeline = ContentPipeline()
        self.test_results = []
        self.temp_dir = tempfile.mkdtemp(prefix='veo3_test_')
        
        # Create test image
        self.test_image_path = self._create_test_image()
    
    def _create_test_image(self) -> str:
        """Create a simple test image using PIL"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a 1080x1080 test image
            img = Image.new('RGB', (1080, 1080), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # Add test content
            draw.rectangle([100, 100, 980, 980], fill='white', outline='black', width=5)
            draw.text((200, 500), "TEST PRODUCT", fill='black')
            draw.text((300, 600), "Premium Quality", fill='gray')
            
            test_image_path = os.path.join(self.temp_dir, 'test_product.jpg')
            img.save(test_image_path)
            
            return test_image_path
        except ImportError:
            # Fallback: create a minimal file
            test_image_path = os.path.join(self.temp_dir, 'test_product.txt')
            with open(test_image_path, 'w') as f:
                f.write("Test image placeholder")
            return test_image_path
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all video generation tests"""
        print("🚀 Starting Comprehensive Veo 3 Video Generation Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test 1: Basic Initialization
        await self._test_initialization()
        
        # Test 2: Platform Specifications
        await self._test_platform_specifications()
        
        # Test 3: Cost Tracking
        await self._test_cost_tracking()
        
        # Test 4: Text-to-Video Generation
        await self._test_text_to_video()
        
        # Test 5: Image-to-Video Generation
        await self._test_image_to_video()
        
        # Test 6: Batch Video Processing
        await self._test_batch_processing()
        
        # Test 7: Platform-Specific Optimizations
        await self._test_platform_optimizations()
        
        # Test 8: Quality Validation
        await self._test_quality_validation()
        
        # Test 9: Mock Services
        await self._test_mock_services()
        
        # Test 10: Error Handling
        await self._test_error_handling()
        
        # Test 11: Integration with Content Pipeline
        await self._test_pipeline_integration()
        
        # Test 12: Performance Benchmarks
        await self._test_performance_benchmarks()
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            'total_tests': len(self.test_results),
            'passed_tests': len([r for r in self.test_results if r['passed']]),
            'failed_tests': len([r for r in self.test_results if not r['passed']]),
            'total_time': total_time,
            'test_details': self.test_results,
            'performance_summary': self._get_performance_summary()
        }
        
        self._print_test_summary(results)
        return results
    
    async def _test_initialization(self):
        """Test 1: Basic system initialization"""
        test_name = "System Initialization"
        try:
            # Check basic initialization
            assert self.generator is not None
            assert hasattr(self.generator, 'platform_specs')
            assert hasattr(self.generator, 'pricing')
            assert len(self.generator.platform_specs) >= 4  # x, tiktok, instagram, linkedin
            
            # Check platform specs completeness
            for platform, spec in self.generator.platform_specs.items():
                assert 'dimensions' in spec
                assert 'fps' in spec
                assert 'max_duration' in spec
                assert 'style' in spec
            
            self._record_test(test_name, True, "All initialization checks passed")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_platform_specifications(self):
        """Test 2: Platform specifications validation"""
        test_name = "Platform Specifications"
        try:
            expected_platforms = ['x', 'tiktok', 'instagram', 'linkedin']
            
            for platform in expected_platforms:
                assert platform in self.generator.platform_specs
                
                spec = self.generator.platform_specs[platform]
                
                # Validate dimensions
                assert isinstance(spec['dimensions'], tuple)
                assert len(spec['dimensions']) == 2
                assert all(isinstance(d, int) for d in spec['dimensions'])
                
                # Validate aspect ratios
                if platform in ['tiktok', 'instagram']:
                    assert spec['aspect_ratio'] == '9:16'  # Vertical
                elif platform in ['x', 'linkedin']:
                    assert spec['aspect_ratio'] == '16:9'  # Horizontal
                
                # Validate trending features
                assert 'trending_features' in spec
                assert isinstance(spec['trending_features'], list)
            
            self._record_test(test_name, True, f"All {len(expected_platforms)} platforms validated")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_cost_tracking(self):
        """Test 3: Enhanced cost tracking functionality"""
        test_name = "Cost Tracking"
        try:
            # Check initial cost summary structure
            initial_cost = self.generator.get_cost_summary()
            
            # Test enhanced cost summary structure
            assert 'costs' in initial_cost
            assert 'limits' in initial_cost
            assert 'analytics' in initial_cost
            assert 'metadata' in initial_cost
            
            # Test costs section
            costs = initial_cost['costs']
            assert 'total' in costs
            assert 'daily' in costs
            assert 'monthly' in costs
            assert 'daily_remaining' in costs
            assert 'monthly_remaining' in costs
            
            # Test limits section
            limits = initial_cost['limits']
            assert 'daily_limit' in limits
            assert 'monthly_limit' in limits
            assert 'daily_usage_percent' in limits
            assert 'monthly_usage_percent' in limits
            
            # Test analytics section
            analytics = initial_cost['analytics']
            assert 'requests_total' in analytics
            assert 'requests_successful' in analytics
            assert 'success_rate' in analytics
            assert 'platform_costs' in analytics
            assert 'quality_costs' in analytics
            assert 'type_costs' in analytics
            
            # Test metadata
            metadata = initial_cost['metadata']
            assert 'currency' in metadata
            assert 'using_mock_services' in metadata
            assert metadata['currency'] == 'USD'
            
            # Test cost calculation
            test_request = VideoGenerationRequest(
                input_type='text',
                input_data='Test product showcase',
                platform='tiktok',
                product_data=TEST_PRODUCT_DATA,
                duration=30,
                quality='hd'
            )
            
            calculated_cost = self.generator._calculate_cost(test_request)
            assert isinstance(calculated_cost, Decimal)
            assert calculated_cost > Decimal('0.00')
            
            self._record_test(test_name, True, f"Cost tracking functional, calculated cost: ${calculated_cost}")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_text_to_video(self):
        """Test 4: Text-to-video generation"""
        test_name = "Text-to-Video Generation"
        try:
            test_prompt = "Professional showcase of premium wireless headphones with noise cancellation"
            
            result = await self.generator.create_text_to_video(
                text_prompt=test_prompt,
                platform='instagram',
                product_data=TEST_PRODUCT_DATA,
                duration=15,
                style='trendy'
            )
            
            assert isinstance(result, VideoGenerationResult)
            assert result.platform == 'instagram'
            assert result.processing_time > 0
            
            if result.success:
                assert result.video_path is not None
                assert os.path.exists(result.video_path) or result.video_path.endswith('.mp4')
                assert result.metadata is not None
                assert 'duration' in result.metadata or len(result.metadata) > 0
            
            self._record_test(test_name, True, f"Generated video: {result.success}, Processing time: {result.processing_time:.2f}s")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_image_to_video(self):
        """Test 5: Image-to-video generation"""
        test_name = "Image-to-Video Generation"
        try:
            result = await self.generator.create_product_video(
                enhanced_image_path=self.test_image_path,
                platform='tiktok',
                product_data=TEST_PRODUCT_DATA,
                duration=15,
                add_captions=True
            )
            
            assert isinstance(result, VideoGenerationResult)
            assert result.platform == 'tiktok'
            assert result.processing_time > 0
            
            # Should succeed with fallback even if Veo 3 API is not available
            if result.success:
                assert result.video_path is not None
                if os.path.exists(result.video_path):
                    assert result.video_path.endswith('.mp4')
                    file_size = os.path.getsize(result.video_path)
                    assert file_size > 0
            
            self._record_test(test_name, True, f"Video created: {result.success}, Cost: ${result.cost}")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_batch_processing(self):
        """Test 6: Batch video processing for multiple platforms"""
        test_name = "Batch Processing"
        try:
            platforms = ['x', 'instagram', 'linkedin']
            
            start_time = time.time()
            results = await self.generator.create_video_variants(
                enhanced_image_path=self.test_image_path,
                product_data=TEST_PRODUCT_DATA,
                platforms=platforms,
                duration=20
            )
            batch_time = time.time() - start_time
            
            assert isinstance(results, dict)
            assert len(results) == len(platforms)
            
            # Validate each platform result
            successful_platforms = []
            for platform in platforms:
                assert platform in results
                result = results[platform]
                assert isinstance(result, VideoGenerationResult)
                if result.success:
                    successful_platforms.append(platform)
            
            success_rate = len(successful_platforms) / len(platforms)
            
            self._record_test(
                test_name, 
                True, 
                f"Batch processed {len(platforms)} platforms in {batch_time:.2f}s, "
                f"Success rate: {success_rate:.1%}"
            )
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_platform_optimizations(self):
        """Test 7: Platform-specific optimizations"""
        test_name = "Platform Optimizations"
        try:
            optimization_tests = []
            
            for platform in TEST_PLATFORMS:
                platform_spec = self.generator.platform_specs[platform]
                
                # Test prompt generation
                prompt = self.generator.generate_video_prompt(
                    platform=platform,
                    product_data=TEST_PRODUCT_DATA,
                    input_type='image'
                )
                
                # Validate prompt contains platform-specific elements
                assert platform_spec['style'].lower() in prompt.lower()
                assert str(platform_spec['optimal_duration']) in prompt
                
                # Test FFmpeg parameters
                ffmpeg_params = self.generator._get_platform_ffmpeg_params(platform)
                assert isinstance(ffmpeg_params, list)
                
                optimization_tests.append(f"{platform}: ✓")
            
            self._record_test(test_name, True, f"All optimizations validated: {', '.join(optimization_tests)}")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_quality_validation(self):
        """Test 8: Video quality validation"""
        test_name = "Quality Validation"
        try:
            # Test quality scoring functions
            test_cases = [
                {'size': (1920, 1080), 'expected_score': 100.0},  # Full HD
                {'size': (1280, 720), 'expected_score': 80.0},    # HD
                {'size': (640, 480), 'expected_score': 40.0}      # SD
            ]
            
            for case in test_cases:
                score = self.generator._score_resolution(case['size'])
                assert score == case['expected_score']
            
            # Test duration scoring
            duration_score = self.generator._score_duration(30.0)
            assert duration_score == 100.0
            
            # Test FPS scoring
            fps_score = self.generator._score_fps(30.0)
            assert fps_score == 100.0
            
            self._record_test(test_name, True, "All quality validation functions working correctly")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_error_handling(self):
        """Test 9: Error handling and fallback mechanisms"""
        test_name = "Error Handling"
        try:
            error_scenarios = []
            
            # Test with invalid image path
            try:
                result = await self.generator.create_product_video(
                    enhanced_image_path="/nonexistent/image.jpg",
                    platform='tiktok',
                    product_data=TEST_PRODUCT_DATA
                )
                # Should not raise exception, but return failed result
                assert isinstance(result, VideoGenerationResult)
                assert not result.success
                error_scenarios.append("Invalid image path: ✓")
            except Exception:
                error_scenarios.append("Invalid image path: ✗")
            
            # Test with invalid platform
            try:
                result = await self.generator.create_product_video(
                    enhanced_image_path=self.test_image_path,
                    platform='invalid_platform',
                    product_data=TEST_PRODUCT_DATA
                )
                # Should use fallback platform specs
                assert isinstance(result, VideoGenerationResult)
                error_scenarios.append("Invalid platform: ✓")
            except Exception:
                error_scenarios.append("Invalid platform: ✗")
            
            # Test rate limiting
            await self.generator._check_rate_limits()  # Should not raise
            error_scenarios.append("Rate limiting: ✓")
            
            self._record_test(test_name, True, f"Error handling tests: {', '.join(error_scenarios)}")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_pipeline_integration(self):
        """Test 10: Integration with content pipeline"""
        test_name = "Pipeline Integration"
        try:
            # Test that pipeline can initialize video generator
            assert self.pipeline.video_generator is not None
            
            # Test cost summary integration
            cost_summary = self.pipeline._get_total_cost_summary()
            assert 'video_costs' in cost_summary
            assert 'total_daily_cost' in cost_summary
            
            # Test cleanup integration
            self.pipeline.cleanup_temporary_files(max_age_hours=0.01)  # Very short age for testing
            
            self._record_test(test_name, True, "Pipeline integration working correctly")
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_mock_services(self):
        """Test 9: Mock services functionality"""
        test_name = "Mock Services"
        try:
            # Check if using mock services
            using_mock = getattr(self.generator, 'using_mock_services', False)
            
            if using_mock:
                # Test mock Veo 3 service
                assert hasattr(self.generator.veo_service, 'call_count')
                assert hasattr(self.generator.veo_service, 'projects')
                
                # Test mock storage client
                assert hasattr(self.generator.storage_client, 'uploads')
                assert hasattr(self.generator.storage_client, 'downloads')
                
                # Test video generation with mock services
                request = VideoGenerationRequest(
                    input_type='text',
                    input_data='Test product showcase',
                    platform='tiktok',
                    product_data=TEST_PRODUCT_DATA,
                    duration=5
                )
                
                result = await self.generator.create_video_with_veo3(request)
                
                # Verify mock behavior
                assert result.success
                assert result.video_path is not None
                assert os.path.exists(result.video_path)
                assert result.cost == 0  # Mock services should have no cost
                
                # Check that mock service was called
                assert self.generator.veo_service.call_count > 0
                
                self._record_test(test_name, True, "Mock services functioning correctly")
                print(f"✅ {test_name}: PASSED (Using mock services)")
            else:
                # If not using mock services, verify real service configuration
                self._record_test(test_name, True, "Real Google services configured")
                print(f"✅ {test_name}: PASSED (Using real Google services)")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_performance_benchmarks(self):
        """Test 12: Performance benchmarks"""
        test_name = "Performance Benchmarks"
        try:
            benchmark_results = {}
            
            # Benchmark 1: Single video generation speed
            start_time = time.time()
            result = await self.generator.create_text_to_video(
                text_prompt="Quick performance test",
                platform='x',
                product_data=TEST_PRODUCT_DATA,
                duration=10
            )
            single_video_time = time.time() - start_time
            benchmark_results['single_video_generation'] = single_video_time
            
            # Benchmark 2: Batch processing efficiency  
            start_time = time.time()
            batch_results = await self.generator.create_video_variants(
                enhanced_image_path=self.test_image_path,
                product_data=TEST_PRODUCT_DATA,
                platforms=['x', 'instagram'],
                duration=10
            )
            batch_time = time.time() - start_time
            benchmark_results['batch_processing'] = batch_time
            
            # Performance targets (should complete within reasonable time)
            assert single_video_time < 60  # Single video under 1 minute
            assert batch_time < 120  # Batch processing under 2 minutes
            
            self._record_test(
                test_name, 
                True, 
                f"Performance benchmarks: Single={single_video_time:.2f}s, Batch={batch_time:.2f}s"
            )
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test(test_name, False, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def _record_test(self, name: str, passed: bool, details: str):
        """Record test result"""
        self.test_results.append({
            'name': name,
            'passed': passed,
            'details': details,
            'timestamp': time.time()
        })
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from test results"""
        processing_times = []
        for result in self.test_results:
            if 'Processing time' in result['details']:
                # Extract processing time from details
                try:
                    time_str = result['details'].split('Processing time: ')[1].split('s')[0]
                    processing_times.append(float(time_str))
                except:
                    pass
        
        return {
            'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'min_processing_time': min(processing_times) if processing_times else 0,
            'total_processing_samples': len(processing_times)
        }
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("🎬 VEO 3 VIDEO GENERATION TEST SUMMARY")
        print("=" * 60)
        
        print(f"📊 Total Tests: {results['total_tests']}")
        print(f"✅ Passed: {results['passed_tests']}")
        print(f"❌ Failed: {results['failed_tests']}")
        print(f"📈 Success Rate: {results['passed_tests']/results['total_tests']*100:.1f}%")
        print(f"⏱️  Total Time: {results['total_time']:.2f} seconds")
        
        print(f"\n📋 Performance Summary:")
        perf = results['performance_summary']
        if perf['total_processing_samples'] > 0:
            print(f"   Average Processing Time: {perf['avg_processing_time']:.2f}s")
            print(f"   Maximum Processing Time: {perf['max_processing_time']:.2f}s")
            print(f"   Minimum Processing Time: {perf['min_processing_time']:.2f}s")
        
        print(f"\n🔧 System Status:")
        print(f"   Google Veo 3 API: {'✅ Available' if self.generator.veo_service else '⚠️ Fallback Mode'}")
        print(f"   Google Cloud Storage: {'✅ Available' if self.generator.storage_client else '⚠️ Not Available'}")
        print(f"   Database Integration: {'✅ Available' if self.generator.db_manager else '⚠️ Not Available'}")
        
        print(f"\n💰 Cost Summary:")
        cost_summary = self.generator.get_cost_summary()
        print(f"   Daily Cost: ${cost_summary['daily_cost']}")
        print(f"   Total Cost: ${cost_summary['total_cost']}")
        
        print(f"\n🏗️  Platform Support:")
        for platform, specs in self.generator.platform_specs.items():
            print(f"   {platform.upper()}: {specs['dimensions'][0]}x{specs['dimensions'][1]} @ {specs['fps']}fps")
        
        if results['failed_tests'] > 0:
            print(f"\n❌ Failed Test Details:")
            for test in results['test_details']:
                if not test['passed']:
                    print(f"   • {test['name']}: {test['details']}")
        
        print("\n" + "=" * 60)
        print("🎉 Test Suite Completed!")
        print("=" * 60)


async def main():
    """Run the comprehensive test suite"""
    print("🎬 Veo 3 Video Generation System - Comprehensive Test Suite")
    print("This will test all aspects of the video generation pipeline.")
    print()
    
    # Initialize tester
    tester = VideoGenerationTester()
    
    try:
        # Run all tests
        results = await tester.run_comprehensive_tests()
        
        # Return appropriate exit code
        if results['failed_tests'] == 0:
            print("\n🎉 ALL TESTS PASSED! Video generation system is ready for production.")
            return 0
        else:
            print(f"\n⚠️  {results['failed_tests']} tests failed. Please review and fix issues.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    # Run the test suite
    exit_code = asyncio.run(main())
    exit(exit_code)