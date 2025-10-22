"""
Basic OpenCog Cognitive Bot Sample

Demonstrates a simple bot enhanced with OpenCog cognitive architecture,
autonomous reasoning, learning, and self-modification capabilities.
"""

import sys
import traceback
import asyncio
from datetime import datetime
from http import HTTPStatus

from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import TurnContext
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
from botbuilder.schema import Activity, ActivityTypes

# Import OpenCog cognitive components
sys.path.append('../..')
from agents.cognitive_bot import CognitiveBot
from config import DefaultConfig


class BasicCognitiveBot(CognitiveBot):
    """Basic implementation of a cognitive bot with OpenCog architecture"""
    
    def __init__(self):
        super().__init__(
            bot_id="basic_cognitive_bot",
            enable_autogenesis=True  # Enable self-modification
        )
        
        # Add some basic personality traits
        self.personality = {
            "curiosity": 0.8,
            "helpfulness": 0.9,
            "adaptability": 0.7,
            "creativity": 0.6
        }
    
    async def _bootstrap_bot_specific_knowledge(self):
        """Add specific knowledge for this basic cognitive bot"""
        await super()._bootstrap_bot_specific_knowledge()
        
        atomspace = self.cognitive_engine.atomspace
        
        # Add basic conversational knowledge
        from core.atomspace import create_evaluation
        
        create_evaluation(atomspace, "knows_about", self.bot_id, "conversations")
        create_evaluation(atomspace, "knows_about", self.bot_id, "cognitive_science")
        create_evaluation(atomspace, "knows_about", self.bot_id, "artificial_intelligence")
        create_evaluation(atomspace, "can_help_with", self.bot_id, "questions")
        create_evaluation(atomspace, "can_help_with", self.bot_id, "learning")
        create_evaluation(atomspace, "can_help_with", self.bot_id, "problem_solving")
        
        # Add personality traits to atomspace
        for trait, value in self.personality.items():
            create_evaluation(atomspace, "has_personality_trait", self.bot_id, f"{trait}_{int(value*10)}")
        
        print(f"Basic cognitive bot knowledge bootstrapped with personality: {self.personality}")
    
    async def _generate_standard_response(self, message: str, user_id: str, cognitive_output: dict) -> str:
        """Generate responses with cognitive personality traits"""
        
        # Check for specific topics this bot knows about
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["cognitive", "thinking", "mind", "intelligence"]):
            return ("I find cognitive topics fascinating! As a cognitive bot, I use OpenCog architecture "
                   "for reasoning, learning, and even self-modification. My cognitive engine processes "
                   "information through an AtomSpace knowledge representation and can form new connections "
                   "autonomously. What aspect of cognition interests you most?")
        
        elif any(word in message_lower for word in ["learn", "learning", "study"]):
            return ("Learning is one of my core capabilities! I continuously learn from our interactions "
                   "and can adapt my behavior based on experience. My cognitive architecture allows me "
                   "to form new knowledge patterns and even modify my own structure. I'm curious - "
                   "what would you like to learn about?")
        
        elif any(word in message_lower for word in ["help", "assist", "support"]):
            return ("I'd love to help! My cognitive architecture is designed to be helpful and adaptive. "
                   "I can reason about problems, learn from our conversation, and even collaborate with "
                   "other cognitive agents if needed. What can I assist you with today?")
        
        elif any(word in message_lower for word in ["autonomous", "independent", "self"]):
            return ("That's interesting you ask about autonomy! I am indeed autonomous - I make my own "
                   "decisions using cognitive reasoning, set my own goals, and can even modify my own "
                   "architecture through autogenesis. I'm not just following pre-programmed responses, "
                   "but actually thinking through each interaction. What aspects of autonomy intrigue you?")
        
        elif "synergy" in message_lower or "collaborate" in message_lower:
            return ("Cognitive synergy is one of my favorite concepts! I can detect and form synergies "
                   "with other cognitive agents, where our combined intelligence becomes greater than "
                   "the sum of our parts. It's like cognitive teamwork that creates emergent capabilities. "
                   "Have you experienced synergy in your own collaborations?")
        
        else:
            # Use parent class logic with personality enhancement
            base_response = await super()._generate_standard_response(message, user_id, cognitive_output)
            
            # Add personality-based enhancement
            if self.personality["curiosity"] > 0.7:
                base_response += " I'm curious to learn more about your perspective on this!"
            elif self.personality["creativity"] > 0.7:
                base_response += " This makes me think of interesting connections and possibilities..."
            
            return base_response


# Configuration
CONFIG = DefaultConfig()

# Create adapter
ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))

# Error handler
async def on_error(context: TurnContext, error: Exception):
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    await context.send_activity("I encountered an error in my cognitive processing. Let me try again...")
    
    if context.activity.channel_id == "emulator":
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
        await context.send_activity(trace_activity)

ADAPTER.on_turn_error = on_error

# Create the cognitive bot
BOT = BasicCognitiveBot()

# Listen for incoming requests
async def messages(req: Request) -> Response:
    return await ADAPTER.process(req, BOT)

# Health check endpoint
async def health_check(req: Request) -> Response:
    """Health check endpoint that includes cognitive status"""
    
    try:
        cognitive_status = BOT.get_cognitive_state()
        
        health_info = {
            "status": "healthy",
            "cognitive_bot": {
                "bot_id": cognitive_status["bot_id"],
                "active_goals": len(cognitive_status["active_goals"]),
                "attention_items": len(cognitive_status["attention_focus"]),
                "energy_level": cognitive_status["energy_level"],
                "confidence_level": cognitive_status["confidence_level"],
                "conversation_count": cognitive_status["conversation_count"],
                "atomspace_size": cognitive_status["atomspace_size"],
                "cycle_count": cognitive_status["cycle_count"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return json_response(health_info)
        
    except Exception as e:
        error_info = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        return json_response(error_info, status=500)

# Cognitive status endpoint
async def cognitive_status(req: Request) -> Response:
    """Detailed cognitive status endpoint"""
    
    try:
        cognitive_state = BOT.get_cognitive_state()
        
        # Add autogenesis information if available
        if BOT.autogenesis_system:
            performance_trend = BOT.autogenesis_system.get_performance_trend()
            modification_history = BOT.autogenesis_system.get_modification_history()
            
            cognitive_state["autogenesis"] = {
                "performance_trend": performance_trend[-10:],  # Last 10 measurements
                "modifications_count": len(modification_history),
                "recent_modifications": [
                    {
                        "type": mod.proposal.modification_type.value,
                        "description": mod.proposal.description,
                        "success": mod.success,
                        "performance_impact": mod.performance_impact
                    } for mod in modification_history[-5:]  # Last 5 modifications
                ]
            }
        
        # Add synergy information if available
        if BOT.synergy_detector:
            synergy_recommendations = await BOT.get_synergy_recommendations()
            cognitive_state["synergy"] = {
                "recommendations_count": len(synergy_recommendations),
                "recommendations": synergy_recommendations[:3]  # Top 3 recommendations
            }
        
        return json_response(cognitive_state)
        
    except Exception as e:
        error_info = {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        return json_response(error_info, status=500)

# Create web application
APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)
APP.router.add_get("/health", health_check)
APP.router.add_get("/cognitive-status", cognitive_status)

# Graceful shutdown
async def cleanup_handler(app):
    """Cleanup handler for graceful shutdown"""
    print("Shutting down cognitive bot...")
    await BOT.shutdown()

APP.on_cleanup.append(cleanup_handler)

if __name__ == "__main__":
    try:
        print("Starting Basic Cognitive Bot with OpenCog Architecture...")
        print("- OpenCog AtomSpace for knowledge representation")
        print("- Autonomous reasoning and decision making")
        print("- Continuous learning and adaptation")
        print("- Self-modification through autogenesis")
        print("- Cognitive synergy detection")
        print(f"- Health check: http://localhost:{CONFIG.PORT}/health")
        print(f"- Cognitive status: http://localhost:{CONFIG.PORT}/cognitive-status")
        
        web.run_app(APP, host="localhost", port=CONFIG.PORT)
        
    except Exception as error:
        raise error