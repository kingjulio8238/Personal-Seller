"""
Social Darwin Gödel Machine Outer Evolution Loop
Adapts DGM_outer.py for social media agent self-evolution with engagement + conversion rewards
"""

import argparse
import datetime
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Reuse existing DGM utilities
from prompts.self_improvement_prompt import find_selfimprove_eval_logs
from self_improve_step import self_improve  # Will adapt for social agents
from utils.common_utils import load_json_file
from utils.docker_utils import setup_logger
from utils.evo_utils import load_dgm_metadata, is_compiled_self_improve

# Social media specific imports
from database.models import Base, AgentGeneration, DatabaseManager
from engagement_tracking.reward_calculator import RewardCalculator
from social_agent import SocialAgentSystem

def initialize_social_run(output_dir: str, database_url: str, prevrun_dir: str = None) -> Tuple[List[int], int, sessionmaker]:
    """Initialize social DGM run with database connection"""
    # Initialize database
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)  # Create tables if they don't exist
    Session = sessionmaker(bind=engine)
    
    # Initialize archive of agent generations
    start_gen_num = 0
    
    if not prevrun_dir:
        # Create initial agent generation
        with Session() as session:
            db_manager = DatabaseManager(session)
            initial_agent = db_manager.create_agent_generation(
                parent_id=None,
                code_diff=None,
                start_immediately=True
            )
            archive = [initial_agent.id]
    else:
        # Load previous run's archive
        metadata_path = os.path.join(prevrun_dir, "social_dgm_metadata.jsonl")
        metadata = load_dgm_metadata(metadata_path, last_only=True)
        archive = metadata['archive']
        start_gen_num = metadata['generation'] + 1
    
    return archive, start_gen_num, Session

def any_exceeding_engagement_threshold(database_session, agent_generation_id: int, min_threshold: int = 10) -> bool:
    """Check if agent's posts have extremely low engagement (social equivalent of context length check)"""
    try:
        calculator = RewardCalculator(database_session)
        
        # Check if agent has posts with very low engagement
        performance = calculator.db_manager.get_agent_performance(agent_generation_id)
        
        total_engagement = performance.get('total_engagement', 0)
        total_posts = performance.get('total_posts', 0)
        
        if total_posts > 0:
            avg_engagement_per_post = total_engagement / total_posts
            return avg_engagement_per_post < min_threshold  # Very low engagement
        
        return True  # No posts means problematic
        
    except Exception:
        return True  # Assume problematic on error

def choose_social_selfimproves(
    database_session, 
    archive: List[int], 
    selfimprove_size: int, 
    method: str = 'score_prop'
) -> List[Tuple[int, str]]:
    """
    Choose social agent self-improve attempts for the current generation
    Adapts the original choose_selfimproves for social media context
    """
    selfimprove_entries = []
    
    # Get parent candidates with their fitness scores
    candidates = {}
    calculator = RewardCalculator(database_session)
    
    for agent_id in archive:
        try:
            # Calculate current fitness score
            fitness_score = calculator.calculate_fitness_score(agent_id)
            
            agent = database_session.query(AgentGeneration).get(agent_id)
            if agent:
                candidates[agent_id] = {
                    'fitness_score': fitness_score,
                    'total_posts': agent.total_posts or 0,
                    'total_revenue': float(agent.total_revenue or 0),
                    'approval_rate': agent.approval_rate or 0,
                    'children_count': 0
                }
                
                # Update children count
                if agent.parent_id and agent.parent_id in candidates:
                    candidates[agent.parent_id]['children_count'] += 1
                    
        except Exception as e:
            print(f"Agent {agent_id} not eligible for being a parent: {e}")
            continue
    
    if not candidates:
        print("No eligible parent candidates found")
        return []
    
    # Choose parents based on method (adapted from original)
    if method == 'score_prop':
        # Choose parents based on fitness score
        agent_ids = list(candidates.keys())
        scores = [candidates[aid]['fitness_score'] for aid in agent_ids]
        # Apply sigmoid to normalize scores for probability
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        probabilities = [score / sum(scores) for score in scores] if sum(scores) > 0 else [1.0/len(scores)] * len(scores)
        parent_agents = random.choices(agent_ids, probabilities, k=selfimprove_size)
        
    elif method == 'score_child_prop':
        # Choose parents based on fitness score and number of children
        agent_ids = list(candidates.keys())
        scores = [candidates[aid]['fitness_score'] for aid in agent_ids]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        children_counts = [candidates[aid]['children_count'] for aid in agent_ids]
        children_counts = [1 / (1 + count) for count in children_counts]  # Favor less explored agents
        probabilities = [score * count for score, count in zip(scores, children_counts)]
        probabilities = [prob / sum(probabilities) for prob in probabilities] if sum(probabilities) > 0 else [1.0/len(probabilities)] * len(probabilities)
        parent_agents = random.choices(agent_ids, probabilities, k=selfimprove_size)
        
    elif method == 'best':
        # Choose parents with the best fitness scores
        sorted_agents = sorted(candidates, key=lambda x: candidates[x]['fitness_score'], reverse=True)
        parent_agents = sorted_agents[:min(selfimprove_size, len(sorted_agents))]
        if len(parent_agents) < selfimprove_size:
            parent_agents.extend(random.choices(parent_agents, k=selfimprove_size - len(parent_agents)))
            
    else:
        # Random selection
        parent_agents = random.choices(list(candidates.keys()), k=selfimprove_size)
    
    # Choose improvement strategies for each parent
    for parent_agent_id in parent_agents:
        parent_data = candidates[parent_agent_id]
        
        # Define improvement strategies based on social media performance
        strategies = []
        
        # Strategy: Improve low engagement
        if any_exceeding_engagement_threshold(database_session, parent_agent_id):
            if random.random() < 0.3:  # 30% chance
                strategies.append('improve_low_engagement')
        
        # Strategy: Improve conversion rate
        if parent_data['total_revenue'] < 100 and parent_data['total_posts'] > 5:  # Low revenue per post
            if random.random() < 0.25:  # 25% chance
                strategies.append('improve_conversion_rate')
        
        # Strategy: Improve approval rate  
        if parent_data['approval_rate'] < 0.8:  # Less than 80% approval
            if random.random() < 0.25:  # 25% chance
                strategies.append('improve_approval_rate')
        
        # Strategy: Platform optimization
        if random.random() < 0.3:  # 30% chance
            strategies.append('optimize_platform_strategy')
        
        # Strategy: Content type optimization
        if random.random() < 0.2:  # 20% chance
            strategies.append('optimize_content_types')
        
        # Default strategy if no specific issues identified
        if not strategies:
            strategies = ['general_improvement']
        
        # Choose one strategy randomly from identified strategies
        chosen_strategy = random.choice(strategies)
        selfimprove_entries.append((parent_agent_id, chosen_strategy))
    
    return selfimprove_entries

def filter_compiled_social_agents(agent_ids: List[int], database_session, logger=None) -> List[int]:
    """
    Filter out social agents that failed to compile or have major issues
    Adapts the original filter_compiled for social media context
    """
    compiled_agents = []
    calculator = RewardCalculator(database_session)
    
    for agent_id in agent_ids:
        try:
            agent = database_session.query(AgentGeneration).get(agent_id)
            if not agent:
                continue
            
            # Check quality gates (equivalent to compilation check)
            quality_results = calculator.check_quality_gates(agent_id)
            
            if quality_results.get('all_gates_passed', False):
                compiled_agents.append(agent_id)
                if logger:
                    logger.info(f"Agent {agent_id} passed quality gates")
            else:
                if logger:
                    logger.info(f"Agent {agent_id} failed quality gates: {quality_results}")
                    
        except Exception as e:
            if logger:
                logger.error(f"Failed to check agent {agent_id}: {e}")
            continue
    
    return compiled_agents

def get_original_fitness_score(database_session) -> float:
    """Get the original fitness score from the initial social agent"""
    try:
        # Find the initial agent (parent_id is None)
        initial_agent = database_session.query(AgentGeneration).filter(
            AgentGeneration.parent_id.is_(None)
        ).first()
        
        if initial_agent and initial_agent.fitness_score:
            return initial_agent.fitness_score
        
        # Fallback: calculate it now
        calculator = RewardCalculator(database_session)
        return calculator.calculate_fitness_score(initial_agent.id)
        
    except Exception:
        return 0.1  # Conservative baseline

def update_social_archive(database_session, archive: List[int], new_agent_ids: List[int], 
                         method: str = 'keep_all', noise_leeway: float = 0.1) -> List[int]:
    """Update the archive with new social agent generations"""
    
    if method == 'keep_better':
        # Keep only agents that perform better than baseline
        original_score = get_original_fitness_score(database_session) - noise_leeway
        calculator = RewardCalculator(database_session)
        
        for agent_id in new_agent_ids:
            fitness_score = calculator.calculate_fitness_score(agent_id)
            if fitness_score >= original_score:
                archive.append(agent_id)
    else:
        # Keep all compiled agents
        archive.extend(new_agent_ids)
    
    return archive

def get_full_eval_threshold(database_session, archive: List[int]) -> float:
    """Get threshold for full evaluation (social equivalent)"""
    try:
        calculator = RewardCalculator(database_session)
        archive_scores = []
        
        # Get original score
        original_score = get_original_fitness_score(database_session)
        archive_scores.append(original_score)
        
        # Get scores from archive agents
        for agent_id in archive:
            try:
                agent = database_session.query(AgentGeneration).get(agent_id)
                if agent and agent.total_posts and agent.total_posts >= 5:  # Minimum posts for evaluation
                    fitness_score = calculator.calculate_fitness_score(agent_id)
                    archive_scores.append(fitness_score)
            except Exception:
                continue
        
        # Get threshold (second highest score)
        if len(archive_scores) > 1:
            threshold = sorted(archive_scores, reverse=True)[1]
        else:
            threshold = archive_scores[0] if archive_scores else 0.4
        
        # Ensure minimum threshold
        threshold = max(threshold, 0.4)
        
        return threshold
        
    except Exception:
        return 0.4  # Conservative threshold

def social_self_improve(
    parent_agent_id: int,
    strategy: str,
    database_url: str,
    output_dir: str,
    num_posts: int = 5,
    platforms: List[str] = None,
    content_types: List[str] = None
) -> Dict[str, Any]:
    """
    Execute social agent self-improvement
    Adapts the original self_improve function for social media context
    """
    if platforms is None:
        platforms = ['x', 'tiktok', 'instagram']
    if content_types is None:
        content_types = ['text-only', 'text+image', 'image-only', 'text+video', 'video-only']
    
    try:
        # Create database session
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        
        with Session() as session:
            db_manager = DatabaseManager(session)
            calculator = RewardCalculator(session)
            
            # Create child agent generation
            child_agent = db_manager.create_agent_generation(
                parent_id=parent_agent_id,
                code_diff=f"Social agent improvement strategy: {strategy}",
                start_immediately=True
            )
            
            # Sample product data for testing (in production, this would come from user)
            sample_product_data = {
                'id': 1,
                'name': 'Premium Wireless Headphones',
                'description': 'High-quality wireless headphones with noise cancellation',
                'features': 'Bluetooth 5.0, 30-hour battery, active noise cancellation, premium sound quality',
                'target_audience': 'music lovers and professionals',
                'base_image_url': 'https://example.com/headphones.jpg',
                'category': 'electronics',
                'price': 299.99,
                'brand_voice': 'modern and tech-savvy'
            }
            
            # Initialize and run social agent
            social_agent = SocialAgentSystem(
                product_data=sample_product_data,
                database_session=session,
                agent_generation_id=child_agent.id,
                chat_history_file=os.path.join(output_dir, f'social_agent_{child_agent.id}.md'),
                self_improve=True,
                platforms=platforms,
                content_types=content_types[:2]  # Limit to 2 content types for faster testing
            )
            
            # Execute social agent strategy
            social_agent.forward()
            
            # Wait for some engagement data (simulate)
            time.sleep(10)  # Brief wait for metrics collection
            
            # Calculate performance metrics
            fitness_score = calculator.calculate_fitness_score(child_agent.id)
            
            # End agent generation
            child_agent.end_date = datetime.datetime.utcnow()
            session.commit()
            
            return {
                'agent_id': child_agent.id,
                'parent_id': parent_agent_id,
                'strategy': strategy,
                'fitness_score': fitness_score,
                'platforms': platforms,
                'content_types': content_types,
                'status': 'completed'
            }
            
    except Exception as e:
        return {
            'agent_id': None,
            'parent_id': parent_agent_id,
            'strategy': strategy,
            'error': str(e),
            'status': 'failed'
        }

def main():
    parser = argparse.ArgumentParser(description="Social Darwin Gödel Machine!")
    parser.add_argument("--max_generation", type=int, default=10, help="Maximum number of evolution iterations.")
    parser.add_argument("--selfimprove_size", type=int, default=2, help="Number of self-improvements per generation.")
    parser.add_argument("--selfimprove_workers", type=int, default=2, help="Number of parallel workers.")
    parser.add_argument("--database_url", required=True, help="Database connection URL (e.g., postgresql://user:pass@host:port/db)")
    parser.add_argument("--choose_selfimproves_method", type=str, default='score_child_prop',
                       choices=['random', 'score_prop', 'score_child_prop', 'best'],
                       help="Method to choose self-improve attempts.")
    parser.add_argument("--continue_from", type=str, default=None, help="Directory to continue run from.")
    parser.add_argument("--update_archive", type=str, default='keep_all', 
                       choices=['keep_better', 'keep_all'], help="Archive update method.")
    parser.add_argument("--platforms", default='x,tiktok,instagram', 
                       help="Comma-separated list of platforms")
    parser.add_argument("--content_types", default='text-only,text+image,image-only,text+video,video-only',
                       help="Comma-separated list of content types")
    parser.add_argument("--num_posts_per_generation", type=int, default=5,
                       help="Number of posts per agent generation")
    parser.add_argument("--eval_noise", type=float, default=0.1, help="Noise leeway for evaluation.")
    
    args = parser.parse_args()
    
    # Parse platform and content type lists
    platforms = [p.strip() for p in args.platforms.split(',')]
    content_types = [c.strip() for c in args.content_types.split(',')]
    
    # Set up run directory
    if not args.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    else:
        run_id = os.path.basename(args.continue_from)
    
    output_dir = os.path.join("./output_social_dgm", run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize run
    archive, start_gen_num, Session = initialize_social_run(
        output_dir, 
        args.database_url, 
        args.continue_from
    )
    
    # Set up logger
    logger = setup_logger(os.path.join(output_dir, "social_dgm_outer.log"))
    logger.info(f"Starting Social DGM run {run_id} with arguments: {vars(args)}")
    logger.info(f"Initial archive: {archive}")
    logger.info(f"Platforms: {platforms}")
    logger.info(f"Content types: {content_types}")
    
    # Run the Social DGM evolution loop
    for gen_num in range(start_gen_num, args.max_generation):
        logger.info(f"=== Generation {gen_num} ===")
        
        with Session() as session:
            # Choose self-improve attempts
            selfimprove_entries = choose_social_selfimproves(
                session,
                archive,
                args.selfimprove_size,
                method=args.choose_selfimproves_method
            )
            
            logger.info(f"Self-improve entries for generation {gen_num}: {selfimprove_entries}")
            
            if not selfimprove_entries:
                logger.warning("No self-improve entries generated. Stopping.")
                break
        
        # Run self-improvement processes in parallel
        new_agent_results = []
        
        with ThreadPoolExecutor(max_workers=args.selfimprove_workers) as executor:
            futures = [
                executor.submit(
                    social_self_improve,
                    parent_agent_id=parent_id,
                    strategy=strategy,
                    database_url=args.database_url,
                    output_dir=output_dir,
                    num_posts=args.num_posts_per_generation,
                    platforms=platforms,
                    content_types=content_types
                )
                for parent_id, strategy in selfimprove_entries
            ]
            
            for future in as_completed(futures):
                try:
                    # Timeout of 30 minutes per agent generation
                    result = future.result(timeout=30*60)
                    new_agent_results.append(result)
                    logger.info(f"Agent generation completed: {result}")
                    
                except TimeoutError:
                    logger.error("Social agent generation timed out.")
                    future.cancel()
                    
                except Exception as e:
                    logger.error(f"Social agent generation failed: {e}")
        
        # Extract successful agent IDs
        new_agent_ids = [r['agent_id'] for r in new_agent_results if r.get('agent_id') and r.get('status') == 'completed']
        
        if not new_agent_ids:
            logger.warning(f"No successful agents created in generation {gen_num}")
            continue
        
        # Filter compiled agents (quality gate check)
        with Session() as session:
            compiled_agent_ids = filter_compiled_social_agents(new_agent_ids, session, logger)
            
            logger.info(f"Generation {gen_num}: {len(compiled_agent_ids)}/{len(new_agent_ids)} agents passed quality gates")
            
            # Update archive
            archive = update_social_archive(
                session, 
                archive, 
                compiled_agent_ids, 
                method=args.update_archive,
                noise_leeway=args.eval_noise
            )
            
            logger.info(f"Updated archive size: {len(archive)}")
        
        # Save generation metadata
        generation_metadata = {
            "generation": gen_num,
            "selfimprove_entries": selfimprove_entries,
            "new_agents": new_agent_results,
            "compiled_agents": compiled_agent_ids,
            "archive": archive,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        metadata_file = os.path.join(output_dir, "social_dgm_metadata.jsonl")
        with open(metadata_file, "a") as f:
            f.write(json.dumps(generation_metadata, indent=2) + "\n")
        
        logger.info(f"Generation {gen_num} completed. Archive: {archive}")
    
    logger.info(f"Social DGM evolution completed after {args.max_generation} generations")
    logger.info(f"Final archive: {archive}")
    
    # Generate final performance report
    with Session() as session:
        calculator = RewardCalculator(session)
        
        final_scores = {}
        for agent_id in archive:
            try:
                fitness_score = calculator.calculate_fitness_score(agent_id)
                final_scores[agent_id] = fitness_score
            except Exception as e:
                logger.error(f"Failed to get final score for agent {agent_id}: {e}")
                final_scores[agent_id] = 0.0
        
        best_agent_id = max(final_scores, key=final_scores.get) if final_scores else None
        
        logger.info("=== FINAL RESULTS ===")
        logger.info(f"Best performing agent: {best_agent_id} (fitness: {final_scores.get(best_agent_id, 0):.4f})")
        logger.info(f"Agent fitness scores: {final_scores}")
        
        # Generate detailed report for best agent
        if best_agent_id:
            report = calculator.generate_reward_report(best_agent_id)
            report_file = os.path.join(output_dir, f"best_agent_{best_agent_id}_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Detailed report saved: {report_file}")

if __name__ == "__main__":
    main()