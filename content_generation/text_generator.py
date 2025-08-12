"""
Text Content Generation System
Creates platform-specific text content using LLMs with deep product understanding
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import openai
import anthropic
from llm import create_client, get_response_from_llm

@dataclass
class ContentRequest:
    """Content generation request structure"""
    platform: str
    content_type: str
    product_data: Dict[str, Any]
    brand_voice: str
    target_audience: str
    campaign_context: Optional[str] = None
    hashtag_strategy: Optional[str] = None
    call_to_action: Optional[str] = None

@dataclass
class GeneratedContent:
    """Generated content structure"""
    text: str
    hashtags: List[str]
    call_to_action: str
    character_count: int
    platform: str
    content_type: str
    created_at: datetime
    metadata: Dict[str, Any]

class TextGenerator:
    """
    LLM-based text content generation with platform optimization
    Creates compelling promotional text using deep product understanding
    """
    
    def __init__(self):
        # Initialize LLM clients with version compatibility
        try:
            # Try new OpenAI v1.0+ client
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except AttributeError:
            # Fallback for older OpenAI versions
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.openai_client = openai
        
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except Exception:
            self.anthropic_client = None
        
        # Load platform configurations
        self.load_platform_configs()
        
        # Content templates and strategies
        self.content_templates = self.load_content_templates()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_platform_configs(self):
        """Load platform-specific content requirements"""
        self.platform_configs = {
            'x': {
                'max_chars': 280,
                'hashtag_limit': 2,
                'optimal_length': 100,
                'tone': 'concise and informative',
                'emoji_usage': 'moderate',
                'link_handling': 'shortening_recommended',
                'engagement_tactics': ['questions', 'threads', 'replies']
            },
            'tiktok': {
                'max_chars': 2200,
                'hashtag_limit': 5,
                'optimal_length': 150,
                'tone': 'energetic and trendy',
                'emoji_usage': 'heavy',
                'link_handling': 'bio_link',
                'engagement_tactics': ['challenges', 'trends', 'duets']
            },
            'instagram': {
                'max_chars': 2200,
                'hashtag_limit': 30,
                'optimal_hashtags': 11,
                'optimal_length': 200,
                'tone': 'aesthetic and storytelling',
                'emoji_usage': 'moderate_to_heavy',
                'link_handling': 'bio_link_or_stories',
                'engagement_tactics': ['stories', 'user_generated_content', 'behind_scenes']
            }
        }

    def load_content_templates(self) -> Dict[str, Dict[str, str]]:
        """Load content generation templates for different scenarios"""
        return {
            'product_launch': {
                'hook': "🚀 Introducing {product_name}!",
                'body': "Experience {key_benefit} like never before. {product_name} delivers {main_features} for {target_audience}.",
                'cta': "Get yours today! Link in bio 👆"
            },
            'feature_highlight': {
                'hook': "Did you know {product_name} can {amazing_feature}?",
                'body': "Here's why {feature_name} is a game-changer: {feature_benefits}. Perfect for {use_case}.",
                'cta': "Try it yourself! {purchase_link}"
            },
            'social_proof': {
                'hook': "Our customers can't stop raving about {product_name}! ⭐⭐⭐⭐⭐",
                'body': "\"{customer_quote}\" - {customer_name}. Join thousands of satisfied customers who love {key_benefit}.",
                'cta': "See what the hype is about! {purchase_link}"
            },
            'problem_solution': {
                'hook': "Tired of {common_problem}?",
                'body': "{product_name} solves this by {solution_method}. No more {pain_points} - just {positive_outcome}.",
                'cta': "Make the switch today! {purchase_link}"
            },
            'lifestyle': {
                'hook': "Life's better with {product_name} ✨",
                'body': "Whether you're {lifestyle_scenario_1} or {lifestyle_scenario_2}, {product_name} fits seamlessly into your {lifestyle_type} lifestyle.",
                'cta': "Elevate your daily routine! {purchase_link}"
            }
        }

    def analyze_product_for_content(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product data to extract key content elements"""
        try:
            analysis_prompt = f"""
            Analyze this product for social media content creation:
            
            Product: {product_data.get('name', 'Unknown Product')}
            Description: {product_data.get('description', 'No description')}
            Features: {product_data.get('features', 'No features listed')}
            Category: {product_data.get('category', 'General')}
            Price: ${product_data.get('price', 0)}
            Target Audience: {product_data.get('target_audience', 'General audience')}
            Brand Voice: {product_data.get('brand_voice', 'Professional')}
            
            Extract and return in JSON format:
            1. key_benefits (top 3 benefits)
            2. emotional_hooks (3 emotional triggers)
            3. pain_points_solved (problems this product addresses)
            4. unique_selling_points (what makes it special)
            5. use_cases (different ways to use the product)
            6. target_emotions (emotions to evoke in content)
            7. content_angles (different narrative approaches)
            """
            
            client, model = create_client('claude-3-5-sonnet-20241022')
            response, _ = get_response_from_llm(
                msg=analysis_prompt,
                client=client,
                model=model,
                system_message="You are an expert content strategist. Always return valid JSON.",
                print_debug=False
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                # Fallback analysis
                return self.create_fallback_analysis(product_data)
                
        except Exception as e:
            self.logger.error(f"Product analysis failed: {e}")
            return self.create_fallback_analysis(product_data)

    def create_fallback_analysis(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic analysis when LLM analysis fails"""
        return {
            'key_benefits': [
                product_data.get('features', 'Great features'),
                'High quality',
                'Great value'
            ],
            'emotional_hooks': [
                'excitement',
                'satisfaction', 
                'confidence'
            ],
            'pain_points_solved': [
                'inconvenience',
                'inefficiency',
                'frustration'
            ],
            'unique_selling_points': [
                product_data.get('name', 'This product'),
                'Premium quality',
                'Trusted brand'
            ],
            'use_cases': [
                'daily use',
                'special occasions',
                'professional settings'
            ],
            'target_emotions': [
                'joy',
                'confidence',
                'satisfaction'
            ],
            'content_angles': [
                'product_launch',
                'feature_highlight',
                'lifestyle'
            ]
        }

    def generate_hashtags(self, platform: str, product_data: Dict[str, Any], 
                         content_analysis: Dict[str, Any]) -> List[str]:
        """Generate platform-specific hashtags"""
        platform_config = self.platform_configs.get(platform, self.platform_configs['x'])
        
        # Base hashtags from product data
        base_hashtags = []
        
        # Product-specific hashtags
        product_name = product_data.get('name', '').replace(' ', '')
        if product_name:
            base_hashtags.append(product_name)
        
        category = product_data.get('category', '')
        if category:
            base_hashtags.append(category)
        
        # Platform-specific trending hashtags
        platform_hashtags = {
            'x': ['tech', 'innovation', 'productivity', 'lifestyle'],
            'tiktok': ['viral', 'trending', 'fyp', 'musthave', 'ProductReview'],
            'instagram': ['instagood', 'lifestyle', 'quality', 'aesthetic', 'dailylife']
        }
        
        # Combine and limit based on platform
        all_hashtags = base_hashtags + platform_hashtags.get(platform, [])
        
        # Limit to platform requirements
        hashtag_limit = platform_config.get('optimal_hashtags', platform_config['hashtag_limit'])
        return all_hashtags[:hashtag_limit]

    def create_platform_optimized_content(self, request: ContentRequest) -> GeneratedContent:
        """Create content optimized for specific platform"""
        platform_config = self.platform_configs.get(request.platform, self.platform_configs['x'])
        
        # Analyze product for content creation
        content_analysis = self.analyze_product_for_content(request.product_data)
        
        # Generate content based on platform and type
        content_prompt = self.build_content_prompt(request, platform_config, content_analysis)
        
        try:
            # Use Claude for content generation
            client, model = create_client('claude-3-5-sonnet-20241022')
            
            response, _ = get_response_from_llm(
                msg=content_prompt,
                client=client,
                model=model,
                system_message="You are an expert social media content creator. Create engaging, conversion-focused content.",
                print_debug=False
            )
            
            # Parse the response
            content_parts = self.parse_content_response(response, request.platform)
            
            # Generate hashtags
            hashtags = self.generate_hashtags(request.platform, request.product_data, content_analysis)
            
            # Create final content object
            generated_content = GeneratedContent(
                text=content_parts['text'],
                hashtags=hashtags,
                call_to_action=content_parts.get('cta', 'Learn more!'),
                character_count=len(content_parts['text']),
                platform=request.platform,
                content_type=request.content_type,
                created_at=datetime.utcnow(),
                metadata={
                    'analysis_used': content_analysis,
                    'platform_config': platform_config,
                    'generation_method': 'claude_llm'
                }
            )
            
            self.logger.info(f"Generated {request.content_type} content for {request.platform}: {len(content_parts['text'])} chars")
            return generated_content
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            return self.create_fallback_content(request)

    def build_content_prompt(self, request: ContentRequest, platform_config: Dict[str, Any], 
                           analysis: Dict[str, Any]) -> str:
        """Build comprehensive content generation prompt"""
        
        prompt = f"""
        Create compelling {request.content_type} content for {request.platform.upper()} that drives engagement and conversions.
        
        PRODUCT INFORMATION:
        - Name: {request.product_data.get('name')}
        - Description: {request.product_data.get('description')}
        - Features: {request.product_data.get('features')}
        - Price: ${request.product_data.get('price', 0)}
        - Category: {request.product_data.get('category')}
        
        AUDIENCE & BRAND:
        - Target Audience: {request.target_audience}
        - Brand Voice: {request.brand_voice}
        
        PLATFORM REQUIREMENTS ({request.platform.upper()}):
        - Maximum characters: {platform_config['max_chars']}
        - Optimal length: {platform_config['optimal_length']} characters
        - Tone: {platform_config['tone']}
        - Emoji usage: {platform_config['emoji_usage']}
        
        CONTENT STRATEGY:
        - Key Benefits: {', '.join(analysis.get('key_benefits', []))}
        - Emotional Hooks: {', '.join(analysis.get('emotional_hooks', []))}
        - Pain Points Solved: {', '.join(analysis.get('pain_points_solved', []))}
        
        REQUIREMENTS:
        1. Create engaging hook that captures attention immediately
        2. Highlight key product benefits naturally
        3. Include emotional triggers that resonate with {request.target_audience}
        4. Add compelling call-to-action
        5. Stay within character limits
        6. Match the {request.brand_voice} brand voice
        7. Use appropriate emoji level: {platform_config['emoji_usage']}
        
        Return the content in this format:
        TEXT: [the main content]
        CTA: [call to action]
        """
        
        return prompt

    def parse_content_response(self, response: str, platform: str) -> Dict[str, str]:
        """Parse LLM response into structured content parts"""
        try:
            parts = {}
            
            # Extract main text
            text_match = response.split('TEXT:')[-1].split('CTA:')[0].strip()
            parts['text'] = text_match
            
            # Extract call to action
            if 'CTA:' in response:
                cta_match = response.split('CTA:')[-1].strip()
                parts['cta'] = cta_match
            else:
                parts['cta'] = 'Learn more!'
            
            return parts
            
        except Exception as e:
            self.logger.error(f"Failed to parse content response: {e}")
            return {
                'text': response[:200] + '...' if len(response) > 200 else response,
                'cta': 'Check it out!'
            }

    def create_fallback_content(self, request: ContentRequest) -> GeneratedContent:
        """Create basic content when AI generation fails"""
        product_name = request.product_data.get('name', 'our amazing product')
        
        fallback_texts = {
            'product_launch': f"🚀 Introducing {product_name}! Experience the difference with premium quality and innovative design. Perfect for {request.target_audience}.",
            'feature_highlight': f"✨ {product_name} delivers exceptional performance with cutting-edge features. See why customers love the quality and reliability.",
            'social_proof': f"⭐ Join thousands of satisfied customers who trust {product_name}. Quality you can count on, results you'll love.",
        }
        
        text = fallback_texts.get('product_launch', f"Discover {product_name} - quality and innovation combined!")
        
        return GeneratedContent(
            text=text,
            hashtags=['quality', 'innovation', 'lifestyle'],
            call_to_action='Learn more today!',
            character_count=len(text),
            platform=request.platform,
            content_type=request.content_type,
            created_at=datetime.utcnow(),
            metadata={'generation_method': 'fallback'}
        )

    def optimize_content_length(self, content: GeneratedContent, platform: str) -> GeneratedContent:
        """Optimize content length for platform requirements"""
        platform_config = self.platform_configs.get(platform, self.platform_configs['x'])
        max_chars = platform_config['max_chars']
        
        if content.character_count > max_chars:
            # Truncate content while preserving meaning
            truncated_text = content.text[:max_chars-20] + "..."
            
            content.text = truncated_text
            content.character_count = len(truncated_text)
            content.metadata['truncated'] = True
            
            self.logger.info(f"Content truncated for {platform}: {content.character_count} chars")
        
        return content

    def create_content_variants(self, request: ContentRequest, num_variants: int = 3) -> List[GeneratedContent]:
        """Create multiple content variants for A/B testing"""
        variants = []
        
        # Vary the content angle for each variant
        content_angles = ['product_launch', 'feature_highlight', 'social_proof', 'problem_solution', 'lifestyle']
        
        for i in range(min(num_variants, len(content_angles))):
            # Modify request for different angles
            variant_request = ContentRequest(
                platform=request.platform,
                content_type=request.content_type,
                product_data=request.product_data,
                brand_voice=request.brand_voice,
                target_audience=request.target_audience,
                campaign_context=content_angles[i]
            )
            
            variant = self.create_platform_optimized_content(variant_request)
            variant.metadata['variant_angle'] = content_angles[i]
            variant.metadata['variant_number'] = i + 1
            
            variants.append(variant)
        
        return variants

    def generate_content_for_platforms(self, product_data: Dict[str, Any], 
                                     platforms: List[str] = None,
                                     content_types: List[str] = None) -> Dict[str, List[GeneratedContent]]:
        """Generate content for multiple platforms and content types"""
        if platforms is None:
            platforms = ['x', 'tiktok', 'instagram']
        
        if content_types is None:
            content_types = ['text-only', 'text+image', 'text+video']
        
        all_content = {}
        
        for platform in platforms:
            platform_content = []
            
            for content_type in content_types:
                request = ContentRequest(
                    platform=platform,
                    content_type=content_type,
                    product_data=product_data,
                    brand_voice=product_data.get('brand_voice', 'professional'),
                    target_audience=product_data.get('target_audience', 'general audience')
                )
                
                content = self.create_platform_optimized_content(request)
                optimized_content = self.optimize_content_length(content, platform)
                platform_content.append(optimized_content)
            
            all_content[platform] = platform_content
        
        return all_content

if __name__ == "__main__":
    # Test the text generator
    generator = TextGenerator()
    
    # Sample product data
    sample_product = {
        'name': 'Wireless Noise-Canceling Headphones',
        'description': 'Premium wireless headphones with active noise cancellation',
        'features': 'Bluetooth 5.0, 30-hour battery, quick charge, premium sound quality',
        'category': 'electronics',
        'price': 299.99,
        'target_audience': 'music lovers and professionals',
        'brand_voice': 'modern and tech-savvy'
    }
    
    # Test content generation
    request = ContentRequest(
        platform='instagram',
        content_type='text+image',
        product_data=sample_product,
        brand_voice='modern and tech-savvy',
        target_audience='music lovers and professionals'
    )
    
    content = generator.create_platform_optimized_content(request)
    print(f"Generated content: {content.text}")
    print(f"Hashtags: {content.hashtags}")
    print(f"CTA: {content.call_to_action}")
    print(f"Character count: {content.character_count}")
    
    print("Text generator initialized successfully")