"""
Cognitive Synergy Demonstration

Shows multiple OpenCog cognitive bots working together,
detecting synergies, and achieving emergent collective intelligence.
"""

import sys
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

from aiohttp import web
from aiohttp.web import Request, Response, json_response

# Import OpenCog cognitive components
sys.path.append('../..')
from agents.cognitive_bot import CognitiveBot, SpecializedCognitiveBot
from orchestrator.multi_bot_coordinator import MultiBotCoordinator, CoordinationTask, CoordinationStrategy
from config import DefaultConfig


class AnalystBot(SpecializedCognitiveBot):
    """Cognitive bot specialized in analysis and data processing"""
    
    def __init__(self):
        domain_knowledge = {
            "data_analysis": {"strength": 0.9, "confidence": 0.8},
            "pattern_recognition": {"strength": 0.8, "confidence": 0.9},
            "statistical_reasoning": {"strength": 0.7, "confidence": 0.8}
        }
        
        super().__init__(
            bot_id="analyst_bot",
            specialization="data_analysis",
            domain_knowledge=domain_knowledge
        )


class CreativeBot(SpecializedCognitiveBot):
    """Cognitive bot specialized in creative thinking and ideation"""
    
    def __init__(self):
        domain_knowledge = {
            "creative_thinking": {"strength": 0.9, "confidence": 0.7},
            "brainstorming": {"strength": 0.8, "confidence": 0.8},
            "innovation": {"strength": 0.7, "confidence": 0.7}
        }
        
        super().__init__(
            bot_id="creative_bot", 
            specialization="creative_thinking",
            domain_knowledge=domain_knowledge
        )
    
    async def _setup_initial_goals(self):
        """Setup creative-specific goals"""
        await super()._setup_initial_goals()
        
        from core.cognitive_engine import CognitiveGoal
        
        # Add creativity-specific goals
        creative_goal = CognitiveGoal(
            name="GenerateCreativeIdeas",
            description="Generate novel and creative solutions to problems",
            priority=0.8
        )
        await self.cognitive_engine.add_goal(creative_goal)


class LogicBot(SpecializedCognitiveBot):
    """Cognitive bot specialized in logical reasoning and validation"""
    
    def __init__(self):
        domain_knowledge = {
            "logical_reasoning": {"strength": 0.9, "confidence": 0.9},
            "validation": {"strength": 0.8, "confidence": 0.9},
            "systematic_thinking": {"strength": 0.8, "confidence": 0.8}
        }
        
        super().__init__(
            bot_id="logic_bot",
            specialization="logical_reasoning", 
            domain_knowledge=domain_knowledge
        )


class SynergyOrchestrator:
    """Orchestrates cognitive synergy demonstrations"""
    
    def __init__(self):
        self.coordinator = MultiBotCoordinator("synergy_coordinator")
        self.bots: Dict[str, CognitiveBot] = {}
        self.demonstration_scenarios = []
        self.running = False
        
    async def start(self):
        """Start the synergy demonstration system"""
        
        # Create specialized cognitive bots
        self.bots["analyst"] = AnalystBot()
        self.bots["creative"] = CreativeBot()  
        self.bots["logic"] = LogicBot()
        
        # Wait for bots to initialize
        await asyncio.sleep(2.0)
        
        # Start coordinator
        await self.coordinator.start()
        
        # Register bots with coordinator
        for bot_id, bot in self.bots.items():
            success = await self.coordinator.register_bot(bot)
            if success:
                print(f"Registered {bot_id} bot with coordinator")
            else:
                print(f"Failed to register {bot_id} bot")
        
        # Create demonstration scenarios
        await self._setup_demonstration_scenarios()
        
        self.running = True
        
        # Start demonstration loop
        asyncio.create_task(self._demonstration_loop())
        
        print("Cognitive Synergy Demonstration started with 3 specialized bots")
    
    async def _setup_demonstration_scenarios(self):
        """Setup various synergy demonstration scenarios"""
        
        # Scenario 1: Complementary analysis and creativity
        scenario1 = CoordinationTask(
            task_id="creative_analysis",
            description="Combine analytical and creative thinking for innovative solutions",
            required_capabilities=["data_analysis", "creative_thinking"],
            coordination_strategy=CoordinationStrategy.COLLABORATIVE,
            priority=0.8
        )
        await self.coordinator.add_coordination_task(scenario1)
        
        # Scenario 2: Logic validation of creative ideas
        scenario2 = CoordinationTask(
            task_id="validate_creativity",
            description="Use logical reasoning to validate and improve creative ideas",
            required_capabilities=["creative_thinking", "logical_reasoning"],
            coordination_strategy=CoordinationStrategy.COLLABORATIVE,
            priority=0.7
        )
        await self.coordinator.add_coordination_task(scenario2)
        
        # Scenario 3: Emergent collective intelligence
        scenario3 = CoordinationTask(
            task_id="collective_problem_solving",
            description="Leverage all three bots for emergent problem-solving capabilities",
            required_capabilities=["data_analysis", "creative_thinking", "logical_reasoning"],
            coordination_strategy=CoordinationStrategy.EMERGENT,
            priority=0.9
        )
        await self.coordinator.add_coordination_task(scenario3)
        
        self.demonstration_scenarios = [scenario1, scenario2, scenario3]
    
    async def _demonstration_loop(self):
        """Main demonstration loop showing synergy in action"""
        
        demo_cycle = 0
        
        while self.running:
            try:
                demo_cycle += 1
                print(f"\n=== Synergy Demonstration Cycle {demo_cycle} ===")
                
                # Show current bot states
                await self._show_bot_states()
                
                # Detect and display synergies
                synergies = await self.coordinator.synergy_detector.detect_synergies()
                if synergies:
                    print(f"\nDetected {len(synergies)} synergies:")
                    for synergy in synergies:
                        print(f"  - {synergy.synergy_type.value} between {synergy.agents} "
                              f"(strength: {synergy.strength:.3f})")
                
                # Show coordination status
                coord_status = self.coordinator.get_coordination_status()
                print(f"\nCoordination Status:")
                print(f"  - Active tasks: {coord_status['active_tasks']}")
                print(f"  - Collective performance: {coord_status['coordination_metrics']['collective_performance']:.3f}")
                
                # Simulate some cognitive activity
                await self._simulate_cognitive_activity()
                
                await asyncio.sleep(20.0)  # Demo cycle every 20 seconds
                
            except Exception as e:
                print(f"Demonstration loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _show_bot_states(self):
        """Display current states of all cognitive bots"""
        
        print("\nCognitive Bot States:")
        
        for bot_id, bot in self.bots.items():
            try:
                state = bot.get_cognitive_state()
                print(f"  {bot_id.upper()}:")
                print(f"    Goals: {len(state['active_goals'])}")
                print(f"    Attention: {len(state['attention_focus'])} items")
                print(f"    Energy: {state['energy_level']:.3f}")
                print(f"    Confidence: {state['confidence_level']:.3f}")
                print(f"    Conversations: {state['conversation_count']}")
                print(f"    Atomspace size: {state['atomspace_size']}")
                
            except Exception as e:
                print(f"    Error getting state: {e}")
    
    async def _simulate_cognitive_activity(self):
        """Simulate cognitive activity to trigger synergies"""
        
        # Simulate different types of cognitive inputs for each bot
        activities = [
            ("analyst", {
                "task": "analyze_data",
                "data_complexity": 0.7,
                "pattern_strength": 0.6
            }),
            ("creative", {
                "task": "generate_ideas", 
                "innovation_required": 0.8,
                "constraints": 0.3
            }),
            ("logic", {
                "task": "validate_reasoning",
                "logical_consistency": 0.9,
                "evidence_strength": 0.7
            })
        ]
        
        # Process activities through cognitive engines
        for bot_id, activity in activities:
            if bot_id in self.bots:
                try:
                    result = await self.bots[bot_id].cognitive_engine.process_input(activity)
                    print(f"    {bot_id} processed activity: {result.get('action', 'unknown')}")
                    
                except Exception as e:
                    print(f"    Error in {bot_id} activity: {e}")
    
    async def stop(self):
        """Stop the synergy demonstration"""
        self.running = False
        
        await self.coordinator.stop()
        
        for bot in self.bots.values():
            await bot.shutdown()
        
        print("Synergy demonstration stopped")
    
    def get_demonstration_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the demonstration"""
        
        coordinator_status = self.coordinator.get_coordination_status()
        
        bot_states = {}
        for bot_id, bot in self.bots.items():
            try:
                bot_states[bot_id] = bot.get_cognitive_state()
            except Exception as e:
                bot_states[bot_id] = {"error": str(e)}
        
        return {
            "demonstration_running": self.running,
            "coordinator_status": coordinator_status,
            "bot_states": bot_states,
            "detected_synergies": len(self.coordinator.synergy_detector.synergy_patterns),
            "synergy_patterns": [
                {
                    "type": synergy.synergy_type.value,
                    "agents": synergy.agents,
                    "strength": synergy.strength,
                    "confidence": synergy.confidence,
                    "benefits": synergy.benefits
                }
                for synergy in self.coordinator.synergy_detector.synergy_patterns
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


# Global orchestrator instance
ORCHESTRATOR = SynergyOrchestrator()

# Web endpoints for monitoring the demonstration

async def status_endpoint(request: Request) -> Response:
    """Get comprehensive demonstration status"""
    try:
        status = ORCHESTRATOR.get_demonstration_status()
        return json_response(status)
    except Exception as e:
        return json_response({"error": str(e)}, status=500)

async def bots_endpoint(request: Request) -> Response:
    """Get detailed information about all cognitive bots"""
    try:
        bots_info = {}
        
        for bot_id, bot in ORCHESTRATOR.bots.items():
            bots_info[bot_id] = {
                "cognitive_state": bot.get_cognitive_state(),
                "specialization": getattr(bot, 'specialization', None),
                "domain_knowledge": getattr(bot, 'domain_knowledge', {}),
                "conversation_history": len(bot.conversation_history),
                "user_models": len(bot.user_models)
            }
        
        return json_response(bots_info)
        
    except Exception as e:
        return json_response({"error": str(e)}, status=500)

async def synergies_endpoint(request: Request) -> Response:
    """Get current synergy information"""
    try:
        synergies = await ORCHESTRATOR.coordinator.synergy_detector.detect_synergies()
        
        synergy_info = {
            "active_synergies": len(synergies),
            "synergies": [
                {
                    "type": synergy.synergy_type.value,
                    "agents": synergy.agents,
                    "strength": synergy.strength,
                    "confidence": synergy.confidence,
                    "benefits": synergy.benefits,
                    "context": synergy.context
                }
                for synergy in synergies
            ],
            "coordinator_metrics": ORCHESTRATOR.coordinator.coordination_metrics
        }
        
        return json_response(synergy_info)
        
    except Exception as e:
        return json_response({"error": str(e)}, status=500)

async def trigger_activity_endpoint(request: Request) -> Response:
    """Manually trigger cognitive activity"""
    try:
        await ORCHESTRATOR._simulate_cognitive_activity()
        
        return json_response({
            "message": "Cognitive activity triggered",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return json_response({"error": str(e)}, status=500)

# Create web application
APP = web.Application()
APP.router.add_get("/status", status_endpoint)
APP.router.add_get("/bots", bots_endpoint)
APP.router.add_get("/synergies", synergies_endpoint)
APP.router.add_post("/trigger-activity", trigger_activity_endpoint)

# Startup and cleanup handlers
async def startup_handler(app):
    """Startup handler"""
    await ORCHESTRATOR.start()

async def cleanup_handler(app):
    """Cleanup handler"""
    await ORCHESTRATOR.stop()

APP.on_startup.append(startup_handler)
APP.on_cleanup.append(cleanup_handler)

if __name__ == "__main__":
    try:
        CONFIG = DefaultConfig()
        port = getattr(CONFIG, 'PORT', 3979)
        
        print("Starting Cognitive Synergy Demonstration...")
        print("- 3 specialized cognitive bots (Analyst, Creative, Logic)")
        print("- Multi-bot coordination system")
        print("- Real-time synergy detection")
        print("- Emergent collective intelligence")
        print(f"- Status endpoint: http://localhost:{port}/status")
        print(f"- Bots info: http://localhost:{port}/bots") 
        print(f"- Synergies: http://localhost:{port}/synergies")
        print(f"- Trigger activity: POST http://localhost:{port}/trigger-activity")
        
        web.run_app(APP, host="localhost", port=port)
        
    except Exception as error:
        print(f"Error starting synergy demonstration: {error}")
        raise error