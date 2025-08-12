"""
Social Darwin Gödel Machine Agent
Adapts coding_agent.py for social media content distribution with self-evolution capabilities
"""

import argparse
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import os
import threading
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from llm_withtools import CLAUDE_MODEL, OPENAI_MODEL, chat_with_agent
from utils.eval_utils import get_report_score, msg_history_to_report, score_tie_breaker
from utils.git_utils import diff_versus_commit, reset_to_commit, apply_patch
from database.models import DatabaseManager, Product, Post, AgentGeneration
from tools.social_tools import get_all_social_tools
from content_generation.image_enhancer import ImageEnhancer
from content_generation.video_generator import VideoGenerator
from content_generation.text_generator import TextGenerator
from engagement_tracking.metrics_collector import MetricsCollector
from engagement_tracking.conversion_tracker import ConversionTracker
from engagement_tracking.reward_calculator import RewardCalculator

# Thread-local storage for logger instances (reuse from coding_agent.py)
thread_local = threading.local()

def get_thread_logger():
    """Get the logger instance specific to the current thread."""
    return getattr(thread_local, 'logger', None)

def set_thread_logger(logger):
    """Set the logger instance for the current thread."""
    thread_local.logger = logger

def setup_logger(log_file='./social_agent_history.md', level=logging.INFO):
    """Set up a logger with both file and console handlers."""
    # Create logger with a unique name based on thread ID
    logger = logging.getLogger(f'SocialAgentSystem-{threading.get_ident()}')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter('%(message)s')
    
    # Create and set up file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    # Store logger in thread-local storage
    set_thread_logger(logger)
    
    return logger

def safe_log(message, level=logging.INFO):
    """Thread-safe logging function that ensures messages go to the correct logger."""
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")

class SocialAgentSystem:
    """
    Social Media Content Distribution Agent with Self-Evolution
    Adapts AgenticSystem from coding_agent.py for social media tasks
    """
    
    def __init__(
        self,
        product_data: Dict[str, Any],
        database_session,
        agent_generation_id: int,
        chat_history_file='./social_agent_history.md',
        self_improve=False,
        platforms=['x', 'tiktok', 'instagram'],
        content_types=['text-only', 'text+image', 'image-only', 'text+video', 'video-only']
    ):
        self.product_data = product_data
        self.database_session = database_session
        self.db_manager = DatabaseManager(database_session)
        self.agent_generation_id = agent_generation_id
        self.chat_history_file = chat_history_file
        self.self_improve = self_improve
        self.platforms = platforms
        self.content_types = content_types
        self.code_model = CLAUDE_MODEL
        
        # Initialize content generation components
        self.image_enhancer = ImageEnhancer()
        self.video_generator = VideoGenerator()
        self.text_generator = TextGenerator()
        
        # Initialize tracking components
        self.metrics_collector = MetricsCollector(database_session)
        self.conversion_tracker = ConversionTracker(database_session)
        self.reward_calculator = RewardCalculator(database_session)
        
        # Initialize logger and store it in thread-local storage
        self.logger = setup_logger(chat_history_file)
        
        # Clear the log file
        with open(chat_history_file, 'w') as f:
            f.write('')
        
        # Load social tools
        self.social_tools = get_all_social_tools()
        
        safe_log(f"Social Agent initialized for agent generation {agent_generation_id}")
        safe_log(f"Product: {product_data.get('name', 'Unknown')}")
        safe_log(f"Platforms: {platforms}")
        safe_log(f"Content types: {content_types}")

    def get_current_edits(self):
        """Get current code modifications (adapted for social agent self-improvement)"""
        # This would track changes to the social agent's own code
        # For now, return placeholder
        return "# Social agent code modifications will be tracked here"

    def analyze_product_for_content(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product data to generate content strategy"""
        safe_log("Analyzing product for content generation...")
        
        instruction = f"""Analyze this product for social media content creation:

Product Information:
- Name: {product_data.get('name', 'Unknown Product')}
- Description: {product_data.get('description', 'No description available')}
- Features: {product_data.get('features', 'No features listed')}
- Target Audience: {product_data.get('target_audience', 'General audience')}
- Category: {product_data.get('category', 'Uncategorized')}
- Price: ${product_data.get('price', 0)}
- Brand Voice: {product_data.get('brand_voice', 'Professional and friendly')}

Your task is to create a comprehensive content strategy including:
1. Key selling points to highlight
2. Emotional hooks and benefits
3. Platform-specific angles (X/Twitter, TikTok, Instagram)
4. Hashtag recommendations
5. Content type recommendations
6. Timing and frequency suggestions

Provide specific, actionable recommendations for creating engaging social media content that drives conversions.
"""

        new_msg_history = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        content_strategy = new_msg_history[-1]
        try:
            content_strategy = content_strategy['content'][-1]['text']
        except:
            pass
        
        safe_log(f"Content strategy generated: {content_strategy[:200]}...")
        return {'strategy': content_strategy, 'analysis_timestamp': datetime.utcnow().isoformat()}

    def generate_content_for_platform(self, platform: str, content_type: str, 
                                    strategy_context: str) -> Dict[str, Any]:
        """Generate platform-specific content"""
        safe_log(f"Generating {content_type} content for {platform}")
        
        platform_specs = {
            'x': {'max_chars': 280, 'style': 'concise and engaging', 'hashtag_limit': 2},
            'tiktok': {'max_chars': 2200, 'style': 'trendy and energetic', 'hashtag_limit': 5},
            'instagram': {'max_chars': 2200, 'style': 'aesthetic and storytelling', 'hashtag_limit': 11}
        }
        
        spec = platform_specs.get(platform, platform_specs['x'])
        
        instruction = f"""Create {content_type} content for {platform.upper()} based on this strategy:

{strategy_context}

Platform Requirements:
- Maximum characters: {spec['max_chars']}
- Style: {spec['style']}
- Hashtag limit: {spec['hashtag_limit']}

Content Type: {content_type}

Product Information:
- Name: {self.product_data.get('name')}
- Key Features: {self.product_data.get('features')}
- Target Audience: {self.product_data.get('target_audience')}
- Brand Voice: {self.product_data.get('brand_voice')}

Generate:
1. Compelling caption/text
2. Recommended hashtags (within platform limit)
3. Call-to-action
4. Best posting time suggestion
5. If image/video content type, describe the visual elements needed

Make the content highly engaging and conversion-focused while staying authentic to the brand voice.
"""

        new_msg_history = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        generated_content = new_msg_history[-1]
        try:
            generated_content = generated_content['content'][-1]['text']
        except:
            pass
        
        safe_log(f"Generated content for {platform}: {generated_content[:100]}...")
        return {
            'platform': platform,
            'content_type': content_type,
            'generated_content': generated_content,
            'generated_at': datetime.utcnow().isoformat()
        }

    def enhance_product_image(self, base_image_path: str, platform: str) -> str:
        """Enhance product image using OpenAI Image Edit API"""
        safe_log(f"Enhancing image for {platform}")
        try:
            enhanced_url = self.image_enhancer.enhance_for_platform(
                base_image_path, 
                platform, 
                self.product_data
            )
            safe_log(f"Image enhanced successfully: {enhanced_url}")
            return enhanced_url
        except Exception as e:
            safe_log(f"Image enhancement failed: {e}")
            return base_image_path  # Return original if enhancement fails

    def generate_product_video(self, enhanced_image_url: str, platform: str) -> str:
        """Generate product video using Google Veo 3"""
        safe_log(f"Generating video for {platform}")
        try:
            video_url = self.video_generator.create_product_video(
                enhanced_image_url,
                platform,
                self.product_data
            )
            safe_log(f"Video generated successfully: {video_url}")
            return video_url
        except Exception as e:
            safe_log(f"Video generation failed: {e}")
            return None

    def create_post_content(self, platform: str, content_type: str, strategy_context: str) -> Dict[str, Any]:
        """Create complete post content including media"""
        safe_log(f"Creating {content_type} post for {platform}")
        
        # Generate text content
        content_data = self.generate_content_for_platform(platform, content_type, strategy_context)
        
        post_data = {
            'platform': platform,
            'content_type': content_type,
            'caption': content_data['generated_content'],
            'created_at': datetime.utcnow()
        }
        
        # Handle media based on content type
        if 'image' in content_type and self.product_data.get('base_image_url'):
            enhanced_image = self.enhance_product_image(
                self.product_data['base_image_url'], 
                platform
            )
            post_data['image_url'] = enhanced_image
            
        if 'video' in content_type:
            if post_data.get('image_url'):
                video_url = self.generate_product_video(post_data['image_url'], platform)
                post_data['video_url'] = video_url
            elif self.product_data.get('base_image_url'):
                # Generate video from base image
                enhanced_image = self.enhance_product_image(
                    self.product_data['base_image_url'], 
                    platform
                )
                video_url = self.generate_product_video(enhanced_image, platform)
                post_data['video_url'] = video_url
        
        return post_data

    def request_human_approval(self, post_data: Dict[str, Any]) -> bool:
        """Request human approval for post (mandatory as per plan)"""
        safe_log("Requesting human approval for post")
        
        # In production, this would trigger UI notification
        # For now, simulate approval process
        approval_request = {
            'post_data': post_data,
            'timestamp': datetime.utcnow().isoformat(),
            'requires_approval': True
        }
        
        # Store approval request in database or queue
        # Return True for demo (would be actual user response)
        safe_log("Human approval simulated as approved")
        return True

    def execute_posting_strategy(self, strategy_context: str) -> List[Dict[str, Any]]:
        """Execute complete posting strategy across platforms"""
        safe_log("Executing posting strategy across platforms")
        
        posted_content = []
        
        for platform in self.platforms:
            for content_type in self.content_types:
                # Skip invalid combinations (e.g., video-only on text platforms)
                if content_type == 'video-only' and platform == 'x':
                    continue
                    
                try:
                    # Create content
                    post_data = self.create_post_content(platform, content_type, strategy_context)
                    
                    # Request human approval (mandatory)
                    if not self.request_human_approval(post_data):
                        safe_log(f"Post rejected by human approval: {platform} {content_type}")
                        continue
                    
                    # Save to database
                    db_post = self.db_manager.create_post(
                        platform=platform,
                        product_id=self.product_data.get('id', 1),
                        content_type=content_type,
                        agent_generation_id=self.agent_generation_id,
                        caption=post_data.get('caption', ''),
                        image_url=post_data.get('image_url'),
                        video_url=post_data.get('video_url')
                    )
                    
                    # Execute actual posting (would use social_tools in production)
                    # For now, mark as posted
                    db_post.status = 'posted'
                    db_post.posted_time = datetime.utcnow()
                    db_post.approval_status = 'approved'
                    self.database_session.commit()
                    
                    posted_content.append({
                        'post_id': db_post.id,
                        'platform': platform,
                        'content_type': content_type,
                        'status': 'posted',
                        'posted_at': db_post.posted_time.isoformat()
                    })
                    
                    safe_log(f"Successfully posted to {platform}: {content_type}")
                    
                    # Brief delay between posts to respect rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    safe_log(f"Failed to post to {platform} ({content_type}): {e}")
                    continue
        
        safe_log(f"Posting strategy completed. Posted {len(posted_content)} pieces of content.")
        return posted_content

    def monitor_engagement_and_conversions(self, posted_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor engagement metrics and track conversions"""
        safe_log("Starting engagement and conversion monitoring")
        
        total_engagement = 0
        total_conversions = Decimal('0.00')
        performance_data = []
        
        for post_info in posted_content:
            post_id = post_info['post_id']
            
            # Collect engagement metrics (simulated for demo)
            engagement_metrics = self.metrics_collector.collect_metrics(post_id)
            if engagement_metrics:
                total_engagement += engagement_metrics.total_engagement
                performance_data.append({
                    'post_id': post_id,
                    'platform': post_info['platform'],
                    'engagement': engagement_metrics.total_engagement,
                    'weighted_engagement': engagement_metrics.weighted_engagement
                })
        
            # Check for conversions
            conversions = self.conversion_tracker.get_post_conversions(post_id)
            for conversion in conversions:
                if conversion.validated:
                    total_conversions += conversion.amount
        
        # Calculate RL rewards
        reward_score = self.reward_calculator.calculate_fitness_score(
            self.agent_generation_id,
            engagement_weight=1.0,
            conversion_weight=10.0
        )
        
        performance_summary = {
            'total_posts': len(posted_content),
            'total_engagement': total_engagement,
            'total_conversions': float(total_conversions),
            'fitness_score': reward_score,
            'performance_data': performance_data,
            'monitoring_timestamp': datetime.utcnow().isoformat()
        }
        
        safe_log(f"Performance monitoring completed: {performance_summary}")
        return performance_summary

    def diagnose_and_improve(self, performance_data: Dict[str, Any]) -> str:
        """Analyze performance and suggest self-improvements"""
        safe_log("Diagnosing performance for self-improvement")
        
        instruction = f"""Analyze this social media performance data and suggest improvements to the social agent's strategy:

Performance Data:
{json.dumps(performance_data, indent=2)}

Current Agent Configuration:
- Platforms: {self.platforms}
- Content Types: {self.content_types}
- Product: {self.product_data.get('name')}

Based on the performance data, identify:
1. What's working well (high engagement/conversion content)
2. What needs improvement (low performing content)
3. Specific code modifications to improve content generation algorithms
4. Platform-specific strategy adjustments
5. Content type optimization recommendations
6. Timing and frequency adjustments

Provide concrete, actionable improvements that could be implemented as code changes to this social agent.
Focus on modifications that would increase engagement rates and conversion rates based on the data.
"""

        new_msg_history = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        improvement_analysis = new_msg_history[-1]
        try:
            improvement_analysis = improvement_analysis['content'][-1]['text']
        except:
            pass
        
        safe_log(f"Self-improvement analysis completed: {improvement_analysis[:200]}...")
        return improvement_analysis

    def forward(self):
        """Main execution function for the social agent system"""
        safe_log("=== Social Agent Execution Starting ===")
        
        try:
            # Step 1: Analyze product and generate content strategy
            strategy_analysis = self.analyze_product_for_content(self.product_data)
            strategy_context = strategy_analysis['strategy']
            
            # Step 2: Execute posting strategy across platforms
            posted_content = self.execute_posting_strategy(strategy_context)
            
            if not posted_content:
                safe_log("No content was successfully posted")
                return
            
            # Step 3: Monitor engagement and conversions
            # In production, this would run continuously
            # For demo, simulate immediate metrics collection
            time.sleep(5)  # Brief wait to simulate real-world delay
            performance_data = self.monitor_engagement_and_conversions(posted_content)
            
            # Step 4: Self-improvement analysis (if enabled)
            if self.self_improve:
                improvement_suggestions = self.diagnose_and_improve(performance_data)
                safe_log(f"Self-improvement suggestions: {improvement_suggestions}")
                
                # In full DGM implementation, this would trigger code modification
                # and evaluation cycle similar to the coding agent
            
            # Step 5: Update agent generation performance
            agent = self.database_session.query(AgentGeneration).get(self.agent_generation_id)
            if agent:
                agent.fitness_score = performance_data['fitness_score']
                agent.engagement_score = performance_data['total_engagement']
                agent.conversion_score = performance_data['total_conversions']
                self.database_session.commit()
            
            safe_log("=== Social Agent Execution Completed Successfully ===")
            
        except Exception as e:
            safe_log(f"Social Agent execution failed: {e}", level=logging.ERROR)
            raise

def main():
    parser = argparse.ArgumentParser(description='Social media content distribution agent with self-evolution.')
    parser.add_argument('--product_data', required=True, help='JSON string of product data')
    parser.add_argument('--database_url', required=True, help='Database connection URL')
    parser.add_argument('--agent_generation_id', required=True, type=int, help='Agent generation ID')
    parser.add_argument('--chat_history_file', required=True, help='Path to chat history file')
    parser.add_argument('--platforms', default='x,tiktok,instagram', help='Comma-separated list of platforms')
    parser.add_argument('--content_types', default='text-only,text+image,image-only,text+video,video-only', 
                       help='Comma-separated list of content types')
    parser.add_argument('--self_improve', default=False, action='store_true', 
                       help='Enable self-improvement analysis')
    args = parser.parse_args()
    
    # Parse arguments
    product_data = json.loads(args.product_data)
    platforms = args.platforms.split(',')
    content_types = args.content_types.split(',')
    
    # Initialize database session (would use actual database in production)
    # For demo, this would be a real SQLAlchemy session
    database_session = None  # Placeholder
    
    # Create and run social agent
    social_agent = SocialAgentSystem(
        product_data=product_data,
        database_session=database_session,
        agent_generation_id=args.agent_generation_id,
        chat_history_file=args.chat_history_file,
        self_improve=args.self_improve,
        platforms=platforms,
        content_types=content_types
    )
    
    # Execute the social agent
    social_agent.forward()

if __name__ == "__main__":
    main()