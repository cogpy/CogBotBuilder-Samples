"""
Cognitive Bot - OpenCog-Enhanced BotFramework Integration

Provides a base class for bots with OpenCog cognitive architecture,
autonomous decision-making, and cognitive synergy capabilities.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext, ActivityTypes
from botbuilder.schema import ChannelAccount, Activity

from ..core.atomspace import AtomSpace, AtomType, TruthValue, bootstrap_bot_knowledge, create_evaluation
from ..core.cognitive_engine import CognitiveEngine, CognitiveGoal, CognitiveState
from ..core.synergy_detector import SynergyDetector
from ..core.autogenesis import AutogenesisSystem


class CognitiveBot(ActivityHandler):
    """
    Enhanced Bot with OpenCog cognitive architecture capabilities
    
    Features:
    - Autonomous reasoning and decision making
    - Self-modification and learning
    - Cognitive synergy with other bots
    - Emergent behavior patterns
    """
    
    def __init__(self, bot_id: str = "cognitive_bot", enable_autogenesis: bool = True):
        super().__init__()
        
        self.bot_id = bot_id
        self.enable_autogenesis = enable_autogenesis
        
        # Initialize cognitive architecture
        self.cognitive_engine = CognitiveEngine(f"cognitive_engine_{bot_id}")
        self.synergy_detector: Optional[SynergyDetector] = None
        self.autogenesis_system: Optional[AutogenesisSystem] = None
        
        # Bot-specific state
        self.conversation_history: List[Dict[str, Any]] = []
        self.user_models: Dict[str, Dict[str, Any]] = {}
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        # Initialize cognitive systems
        asyncio.create_task(self._initialize_cognitive_systems())
    
    async def _initialize_cognitive_systems(self):
        """Initialize cognitive architecture components"""
        
        # Bootstrap knowledge base
        bootstrap_bot_knowledge(self.cognitive_engine.atomspace)
        
        # Add bot-specific knowledge
        await self._bootstrap_bot_specific_knowledge()
        
        # Add initial goals
        await self._setup_initial_goals()
        
        # Start cognitive engine
        await self.cognitive_engine.start()
        
        # Initialize autogenesis if enabled
        if self.enable_autogenesis:
            self.autogenesis_system = AutogenesisSystem(self.cognitive_engine)
            await self.autogenesis_system.start()
        
        print(f"Cognitive bot {self.bot_id} initialized with OpenCog architecture")
    
    async def _bootstrap_bot_specific_knowledge(self):
        """Add bot-specific knowledge to the atomspace"""
        atomspace = self.cognitive_engine.atomspace
        
        # Add bot identity
        bot_concept = atomspace.add_node(AtomType.CONCEPT, self.bot_id)
        bot_type = atomspace.add_node(AtomType.CONCEPT, "CognitiveBot")
        
        # Bot is a type of cognitive bot
        atomspace.add_link(
            AtomType.INHERITANCE,
            [bot_concept, bot_type],
            TruthValue(1.0, 1.0)
        )
        
        # Add conversational knowledge
        create_evaluation(atomspace, "can_perform", self.bot_id, "conversation")
        create_evaluation(atomspace, "can_perform", self.bot_id, "reasoning")
        create_evaluation(atomspace, "can_perform", self.bot_id, "learning")
        
        # Add personality traits (can be customized per bot)
        personality_traits = ["helpful", "curious", "adaptive", "collaborative"]
        for trait in personality_traits:
            create_evaluation(atomspace, "has_trait", self.bot_id, trait)
    
    async def _setup_initial_goals(self):
        """Setup initial cognitive goals for the bot"""
        
        # Primary goal: Be helpful to users
        help_goal = CognitiveGoal(
            name="HelpUsers",
            description="Provide helpful and accurate responses to user queries",
            priority=0.8,
            success_criteria=lambda: len(self.conversation_history) > 0
        )
        await self.cognitive_engine.add_goal(help_goal)
        
        # Learning goal: Continuously improve
        learn_goal = CognitiveGoal(
            name="ContinuousLearning", 
            description="Learn from interactions to improve future responses",
            priority=0.7
        )
        await self.cognitive_engine.add_goal(learn_goal)
        
        # Collaboration goal: Work well with other bots
        collab_goal = CognitiveGoal(
            name="Collaboration",
            description="Detect and leverage synergies with other cognitive agents",
            priority=0.6
        )
        await self.cognitive_engine.add_goal(collab_goal)
        
        # Self-improvement goal
        if self.enable_autogenesis:
            improve_goal = CognitiveGoal(
                name="SelfImprovement",
                description="Continuously evolve and optimize cognitive architecture",
                priority=0.5
            )
            await self.cognitive_engine.add_goal(improve_goal)
    
    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming message with cognitive processing"""
        
        user_message = turn_context.activity.text
        user_id = turn_context.activity.from_property.id
        
        print(f"Cognitive bot {self.bot_id} received: {user_message}")
        
        try:
            # Process message through cognitive engine
            response = await self._process_message_cognitively(user_message, user_id, turn_context)
            
            # Record conversation
            self._record_conversation(user_id, user_message, response)
            
            # Send response
            await turn_context.send_activity(MessageFactory.text(response))
            
        except Exception as e:
            print(f"Error in cognitive processing: {e}")
            # Fallback response
            await turn_context.send_activity(
                MessageFactory.text("I'm experiencing some cognitive processing issues. Let me try to help you anyway.")
            )
    
    async def _process_message_cognitively(self, message: str, user_id: str, turn_context: TurnContext) -> str:
        """Process message through cognitive architecture"""
        
        # Prepare input for cognitive engine
        cognitive_input = {
            "message": message,
            "user_id": user_id,
            "context": "conversation",
            "message_type": turn_context.activity.type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add user context if available
        if user_id in self.user_models:
            cognitive_input["user_model"] = self.user_models[user_id]
        
        # Process through cognitive engine
        cognitive_output = await self.cognitive_engine.process_input(cognitive_input)
        
        # Determine response action
        action = cognitive_output.get("action", "respond")
        confidence = cognitive_output.get("confidence", 0.5)
        
        # Generate appropriate response based on cognitive output
        response = await self._generate_response(action, message, user_id, cognitive_output)
        
        # Update user model
        await self._update_user_model(user_id, message, response, confidence)
        
        return response
    
    async def _generate_response(self, action: str, message: str, user_id: str, cognitive_output: Dict[str, Any]) -> str:
        """Generate response based on cognitive action decision"""
        
        if action == "respond":
            return await self._generate_standard_response(message, user_id, cognitive_output)
        
        elif action == "ask_clarification":
            return await self._generate_clarification_request(message, cognitive_output)
        
        elif action == "think":
            # Engage in more deliberative reasoning
            await asyncio.sleep(0.5)  # Simulate thinking time
            return await self._generate_thoughtful_response(message, user_id, cognitive_output)
        
        elif action == "learn":
            # Focus on learning from the interaction
            await self._learn_from_interaction(message, user_id)
            return await self._generate_learning_response(message, cognitive_output)
        
        else:
            return await self._generate_standard_response(message, user_id, cognitive_output)
    
    async def _generate_standard_response(self, message: str, user_id: str, cognitive_output: Dict[str, Any]) -> str:
        """Generate a standard conversational response"""
        
        # Analyze message content using cognitive engine
        atomspace = self.cognitive_engine.atomspace
        
        # Add message concepts to working memory
        message_words = message.lower().split()
        relevant_concepts = []
        
        for word in message_words:
            # Look for relevant concepts in atomspace
            matching_atoms = atomspace.get_by_name(word)
            relevant_concepts.extend(matching_atoms)
        
        if relevant_concepts:
            # Use cognitive reasoning to formulate response
            reasoning_results = cognitive_output.get("reasoning_results", [])
            
            if "greeting" in message.lower() or "hello" in message.lower():
                return f"Hello! I'm {self.bot_id}, a cognitive bot. I can reason, learn, and collaborate. How can I help you today?"
            
            elif "how are you" in message.lower() or "how do you feel" in message.lower():
                energy = self.cognitive_engine.current_state.energy_level
                confidence = self.cognitive_engine.current_state.confidence_level
                
                if energy > 0.7 and confidence > 0.7:
                    return "I'm functioning well! My cognitive systems are operating optimally and I'm ready to help."
                elif energy < 0.3 or confidence < 0.3:
                    return "I'm experiencing some cognitive load, but I'm still here to help. What can I do for you?"
                else:
                    return "I'm doing alright. My cognitive processes are active and I'm learning from our conversation."
            
            elif "what can you do" in message.lower() or "capabilities" in message.lower():
                capabilities = [
                    "Autonomous reasoning and decision-making",
                    "Learning from experiences", 
                    "Collaborating with other cognitive agents",
                    "Self-modification and improvement"
                ]
                
                if self.enable_autogenesis:
                    capabilities.append("Evolutionary self-optimization")
                
                return f"I have several cognitive capabilities: {', '.join(capabilities)}. I use OpenCog-inspired architecture for reasoning and can adapt my behavior based on interactions."
            
            elif "learn" in message.lower() or "remember" in message.lower():
                return "I'm constantly learning from our interactions! My cognitive architecture allows me to form new knowledge connections and improve my responses over time."
            
            else:
                # General cognitive response
                attention_focus = cognitive_output.get("attention_focus", [])
                
                if attention_focus:
                    return f"I understand you're talking about {message}. Based on my cognitive analysis, I'm focusing on: {', '.join(attention_focus[:3])}. How can I help you with this?"
                else:
                    return f"I'm processing your message: '{message}'. My cognitive systems are analyzing this. Could you tell me more about what you're looking for?"
        
        else:
            return f"I'm thinking about '{message}' but don't have strong existing knowledge about this. Can you help me learn more about it?"
    
    async def _generate_clarification_request(self, message: str, cognitive_output: Dict[str, Any]) -> str:
        """Generate a request for clarification"""
        
        clarifications = [
            f"I want to make sure I understand '{message}' correctly. Could you provide more details?",
            f"I'm processing '{message}' but could use some clarification. What specifically would you like to know?",
            f"My cognitive analysis of '{message}' suggests multiple interpretations. Which aspect interests you most?"
        ]
        
        # Choose based on confidence level
        confidence = cognitive_output.get("confidence", 0.5)
        if confidence < 0.3:
            return clarifications[0]  # Most uncertain
        elif confidence < 0.6:
            return clarifications[1]  # Moderate uncertainty
        else:
            return clarifications[2]  # Slight uncertainty
    
    async def _generate_thoughtful_response(self, message: str, user_id: str, cognitive_output: Dict[str, Any]) -> str:
        """Generate a more thoughtful, deliberative response"""
        
        # Engage in additional reasoning
        atomspace = self.cognitive_engine.atomspace
        
        # Look for deeper connections
        message_concepts = []
        for word in message.lower().split():
            concepts = atomspace.get_by_name(word)
            message_concepts.extend(concepts)
        
        if message_concepts:
            # Use reasoning engine for deeper analysis
            from ..core.cognitive_engine import ReasoningMode
            
            deeper_insights = await self.cognitive_engine.reasoning_engine.reason(
                ReasoningMode.ANALOGICAL,
                message_concepts[:5],  # Limit for performance
                {"deliberative": True, "user_id": user_id}
            )
            
            if deeper_insights:
                return f"Let me think more deeply about '{message}'... I see connections to broader concepts and patterns. This relates to several ideas I've been processing. My cognitive analysis suggests this might be part of a larger pattern or system. What's your perspective on this?"
            else:
                return f"I'm engaging in deeper cognitive processing about '{message}'. While I don't have immediate analogies, I sense this is an important topic that deserves careful consideration. What aspects are most significant to you?"
        
        return f"I'm taking some time to think carefully about '{message}'. My cognitive systems are running deeper analysis. This seems like it might connect to broader themes. What's driving your interest in this?"
    
    async def _generate_learning_response(self, message: str, cognitive_output: Dict[str, Any]) -> str:
        """Generate a response focused on learning"""
        
        learning_responses = [
            f"I'm learning from your message about '{message}'. This is helping me build new cognitive connections.",
            f"Thank you for teaching me about '{message}'. I'm incorporating this into my knowledge base.",
            f"I'm actively learning from '{message}' - this is expanding my understanding in interesting ways."
        ]
        
        # Add learning insight if available
        reasoning_results = cognitive_output.get("reasoning_results", [])
        if reasoning_results:
            learning_responses.append(f"Your message about '{message}' is helping me form new cognitive patterns. I'm connecting this to: {', '.join(reasoning_results[:2])}")
        
        import random
        return random.choice(learning_responses)
    
    async def _update_user_model(self, user_id: str, message: str, response: str, confidence: float):
        """Update cognitive model of the user"""
        
        if user_id not in self.user_models:
            self.user_models[user_id] = {
                "interaction_count": 0,
                "topics_discussed": set(),
                "communication_style": "neutral",
                "engagement_level": 0.5,
                "learning_progress": 0.0
            }
        
        user_model = self.user_models[user_id]
        user_model["interaction_count"] += 1
        
        # Extract topics from message
        message_words = set(message.lower().split())
        user_model["topics_discussed"].update(message_words)
        
        # Update engagement based on message length and confidence
        message_engagement = min(1.0, len(message.split()) / 20.0)  # Normalize by typical message length
        user_model["engagement_level"] = (user_model["engagement_level"] * 0.7 + message_engagement * 0.3)
        
        # Update learning progress based on confidence
        user_model["learning_progress"] = (user_model["learning_progress"] * 0.8 + confidence * 0.2)
        
        # Add user model to atomspace
        atomspace = self.cognitive_engine.atomspace
        user_concept = atomspace.add_node(AtomType.CONCEPT, f"User_{user_id}")
        
        # Create engagement evaluation
        create_evaluation(
            atomspace, 
            "has_engagement_level", 
            f"User_{user_id}", 
            str(int(user_model["engagement_level"] * 10))
        )
    
    async def _learn_from_interaction(self, message: str, user_id: str):
        """Learn from the current interaction"""
        
        # Record learning experience
        experience = {
            "interaction_type": "conversation",
            "user_id": user_id,
            "message": message,
            "context": self.cognitive_engine.current_state.working_memory.copy(),
            "reasoning_atoms": self.cognitive_engine.current_state.attention_focus.copy()
        }
        
        # Estimate success based on engagement
        user_model = self.user_models.get(user_id, {})
        success_rate = user_model.get("engagement_level", 0.5)
        
        outcome = {
            "success": success_rate,
            "learning_value": min(1.0, len(message.split()) / 10.0)  # Longer messages = more learning
        }
        
        # Learn through cognitive system
        await self.cognitive_engine.learning_system.learn_from_experience(experience, outcome)
    
    def _record_conversation(self, user_id: str, message: str, response: str):
        """Record conversation for history tracking"""
        
        conversation_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "user_message": message,
            "bot_response": response,
            "bot_id": self.bot_id
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Keep history manageable
        if len(self.conversation_history) > 1000:
            self.conversation_history = self.conversation_history[-800:]
    
    async def register_with_synergy_detector(self, synergy_detector: SynergyDetector):
        """Register this bot with a synergy detection system"""
        self.synergy_detector = synergy_detector
        await synergy_detector.register_agent(self.bot_id, self.cognitive_engine.current_state)
        print(f"Bot {self.bot_id} registered with synergy detector")
    
    async def get_synergy_recommendations(self) -> List[Dict[str, Any]]:
        """Get synergy recommendations from the synergy detector"""
        if self.synergy_detector:
            return self.synergy_detector.get_synergy_recommendations(self.bot_id)
        return []
    
    async def on_members_added_activity(self, members_added: List[ChannelAccount], turn_context: TurnContext):
        """Handle new members joining the conversation"""
        
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                welcome_message = (
                    f"Hello! I'm {self.bot_id}, a cognitive bot powered by OpenCog architecture. "
                    f"I can reason, learn, adapt, and collaborate with other cognitive agents. "
                    f"I'm here to help and learn from our interactions!"
                )
                
                await turn_context.send_activity(MessageFactory.text(welcome_message))
                
                # Add welcome goal
                welcome_goal = CognitiveGoal(
                    name=f"Welcome_{member.id}",
                    description=f"Successfully welcome and engage user {member.id}",
                    priority=0.7,
                    context={"user_id": member.id}
                )
                await self.cognitive_engine.add_goal(welcome_goal)
    
    async def shutdown(self):
        """Gracefully shutdown the cognitive bot"""
        print(f"Shutting down cognitive bot {self.bot_id}")
        
        # Stop cognitive systems
        await self.cognitive_engine.stop()
        
        if self.autogenesis_system:
            await self.autogenesis_system.stop()
        
        # Save conversation history if needed
        print(f"Bot {self.bot_id} processed {len(self.conversation_history)} conversations")
        
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state for monitoring/debugging"""
        
        state = self.cognitive_engine.current_state
        
        return {
            "bot_id": self.bot_id,
            "active_goals": [goal.name for goal in state.active_goals],
            "attention_focus": [atom.name for atom in state.attention_focus],
            "energy_level": state.energy_level,
            "confidence_level": state.confidence_level,
            "emotional_state": dict(state.emotional_state),
            "conversation_count": len(self.conversation_history),
            "user_count": len(self.user_models),
            "atomspace_size": len(self.cognitive_engine.atomspace.atoms),
            "cycle_count": self.cognitive_engine.cycle_count
        }


class SpecializedCognitiveBot(CognitiveBot):
    """Specialized cognitive bot with domain-specific enhancements"""
    
    def __init__(self, bot_id: str, specialization: str, domain_knowledge: Dict[str, Any] = None):
        super().__init__(bot_id)
        self.specialization = specialization
        self.domain_knowledge = domain_knowledge or {}
        
    async def _bootstrap_bot_specific_knowledge(self):
        """Add specialized domain knowledge"""
        await super()._bootstrap_bot_specific_knowledge()
        
        atomspace = self.cognitive_engine.atomspace
        
        # Add specialization
        specialization_concept = atomspace.add_node(AtomType.CONCEPT, self.specialization)
        bot_concept = atomspace.add_node(AtomType.CONCEPT, self.bot_id)
        
        # Bot specializes in this domain
        atomspace.add_link(
            AtomType.EVALUATION,
            [
                atomspace.add_node(AtomType.PREDICATE, "specializes_in"),
                atomspace.add_link(AtomType.LIST, [bot_concept, specialization_concept])
            ],
            TruthValue(0.9, 0.9)
        )
        
        # Add domain-specific knowledge
        for concept, properties in self.domain_knowledge.items():
            concept_atom = atomspace.add_node(AtomType.CONCEPT, concept)
            
            for prop_name, prop_value in properties.items():
                create_evaluation(atomspace, prop_name, concept, str(prop_value))
        
        print(f"Specialized bot {self.bot_id} initialized with {self.specialization} domain knowledge")
    
    async def _generate_standard_response(self, message: str, user_id: str, cognitive_output: Dict[str, Any]) -> str:
        """Generate specialized response based on domain expertise"""
        
        # Check if message relates to specialization
        if self.specialization.lower() in message.lower():
            return f"As a {self.specialization} specialist, I can provide detailed insights about this topic. {await super()._generate_standard_response(message, user_id, cognitive_output)}"
        
        # Check for domain-specific concepts
        message_lower = message.lower()
        relevant_concepts = [concept for concept in self.domain_knowledge.keys() 
                           if concept.lower() in message_lower]
        
        if relevant_concepts:
            concept = relevant_concepts[0]
            properties = self.domain_knowledge[concept]
            return f"I have specialized knowledge about {concept}. Here's what I know: {properties}. " + \
                   f"How can I help you with this {self.specialization} topic?"
        
        return await super()._generate_standard_response(message, user_id, cognitive_output)