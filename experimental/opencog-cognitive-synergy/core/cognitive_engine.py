"""
Cognitive Engine for OpenCog-based Bot Architecture

Implements core reasoning, decision-making, and learning capabilities
for autonomous cognitive agents and emergent intelligence systems.
"""

import asyncio
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

from .atomspace import AtomSpace, Atom, Link, AtomType, TruthValue, create_evaluation


class ReasoningMode(Enum):
    """Different modes of cognitive reasoning"""
    DEDUCTIVE = "deductive"     # Logic-based inference
    INDUCTIVE = "inductive"     # Pattern-based generalization  
    ABDUCTIVE = "abductive"     # Best explanation inference
    ANALOGICAL = "analogical"   # Similarity-based reasoning
    CREATIVE = "creative"       # Novel combination generation


@dataclass
class CognitiveGoal:
    """Represents a cognitive goal or objective"""
    name: str
    description: str
    priority: float = 0.5
    context: Optional[Dict[str, Any]] = None
    success_criteria: Optional[Callable] = None
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            import time
            self.created_at = time.time()


@dataclass
class CognitiveState:
    """Current state of the cognitive system"""
    attention_focus: List[Atom]
    active_goals: List[CognitiveGoal]
    working_memory: Dict[str, Any]
    emotional_state: Dict[str, float]
    confidence_level: float = 0.5
    energy_level: float = 1.0


class ReasoningEngine:
    """Core reasoning and inference engine"""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.inference_rules = {}
        self.reasoning_chains = deque(maxlen=100)
        self._setup_inference_rules()
    
    def _setup_inference_rules(self):
        """Initialize basic inference rules"""
        self.inference_rules = {
            "deduction": self._deductive_inference,
            "induction": self._inductive_inference,
            "abduction": self._abductive_inference,
            "analogy": self._analogical_inference,
            "creativity": self._creative_inference
        }
    
    async def reason(self, mode: ReasoningMode, premises: List[Atom], context: Dict[str, Any] = None) -> List[Atom]:
        """Execute reasoning in the specified mode"""
        context = context or {}
        
        try:
            inference_func = self.inference_rules.get(mode.value)
            if not inference_func:
                raise ValueError(f"Unknown reasoning mode: {mode}")
            
            conclusions = await inference_func(premises, context)
            
            # Record reasoning chain for learning
            self.reasoning_chains.append({
                "mode": mode.value,
                "premises": premises,
                "conclusions": conclusions,
                "context": context,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return conclusions
            
        except Exception as e:
            print(f"Reasoning error in {mode}: {e}")
            return []
    
    async def _deductive_inference(self, premises: List[Atom], context: Dict[str, Any]) -> List[Atom]:
        """Deductive reasoning: derive logical conclusions"""
        conclusions = []
        
        for premise in premises:
            # Look for inheritance chains
            if premise.atom_type == AtomType.INHERITANCE:
                child, parent = premise.outgoing
                
                # Find what else inherits from parent
                for atom in self.atomspace.get_by_type(AtomType.INHERITANCE):
                    if len(atom.outgoing) >= 2 and atom.outgoing[1] == parent:
                        sibling = atom.outgoing[0]
                        if sibling != child:
                            # Create similarity link between siblings
                            similarity = self.atomspace.add_link(
                                AtomType.SIMILARITY,
                                [child, sibling],
                                TruthValue(0.6, 0.7)
                            )
                            conclusions.append(similarity)
        
        return conclusions
    
    async def _inductive_inference(self, premises: List[Atom], context: Dict[str, Any]) -> List[Atom]:
        """Inductive reasoning: generalize from examples"""
        conclusions = []
        
        # Group premises by type and look for patterns
        evaluations = [p for p in premises if p.atom_type == AtomType.EVALUATION]
        
        if len(evaluations) >= 2:
            # Look for common predicates
            predicate_groups = {}
            for eval_link in evaluations:
                if len(eval_link.outgoing) >= 2:
                    predicate = eval_link.outgoing[0]
                    if predicate not in predicate_groups:
                        predicate_groups[predicate] = []
                    predicate_groups[predicate].append(eval_link)
            
            # Generate generalizations
            for predicate, evals in predicate_groups.items():
                if len(evals) >= 2:
                    # Create a general rule
                    general_concept = self.atomspace.add_node(
                        AtomType.CONCEPT,
                        f"GeneralRule_{predicate.name}"
                    )
                    
                    # Link the rule to the predicate
                    rule_link = self.atomspace.add_link(
                        AtomType.IMPLICATION,
                        [general_concept, predicate],
                        TruthValue(0.7, len(evals) / 10.0)
                    )
                    conclusions.append(rule_link)
        
        return conclusions
    
    async def _abductive_inference(self, premises: List[Atom], context: Dict[str, Any]) -> List[Atom]:
        """Abductive reasoning: find best explanations"""
        conclusions = []
        
        # Look for unexplained observations and generate hypotheses
        observations = [p for p in premises if p.atom_type == AtomType.EVALUATION]
        
        for observation in observations:
            # Generate potential explanations
            if len(observation.outgoing) >= 2:
                predicate = observation.outgoing[0]
                
                # Look for similar predicates that could explain this
                similar_predicates = self.atomspace.find_similar_concepts(predicate, threshold=0.5)
                
                for similar_pred, similarity in similar_predicates[:3]:  # Top 3 explanations
                    explanation = self.atomspace.add_node(
                        AtomType.CONCEPT,
                        f"Explanation_{observation.name}_{similar_pred.name}"
                    )
                    
                    explanation_link = self.atomspace.add_link(
                        AtomType.IMPLICATION,
                        [explanation, observation],
                        TruthValue(similarity * 0.6, 0.5)
                    )
                    conclusions.append(explanation_link)
        
        return conclusions
    
    async def _analogical_inference(self, premises: List[Atom], context: Dict[str, Any]) -> List[Atom]:
        """Analogical reasoning: transfer knowledge via similarity"""
        conclusions = []
        
        # Find structural similarities between different domains
        concepts = [p for p in premises if p.atom_type == AtomType.CONCEPT]
        
        for concept in concepts:
            similar_concepts = self.atomspace.find_similar_concepts(concept, threshold=0.6)
            
            for similar_concept, similarity in similar_concepts:
                # Transfer relationships from source to target domain
                for incoming_link in concept.incoming:
                    if incoming_link.atom_type in [AtomType.EVALUATION, AtomType.INHERITANCE]:
                        # Create analogous relationship for similar concept
                        new_outgoing = []
                        for atom in incoming_link.outgoing:
                            if atom == concept:
                                new_outgoing.append(similar_concept)
                            else:
                                new_outgoing.append(atom)
                        
                        analogy_link = self.atomspace.add_link(
                            incoming_link.atom_type,
                            new_outgoing,
                            TruthValue(similarity * 0.5, 0.4)
                        )
                        conclusions.append(analogy_link)
        
        return conclusions
    
    async def _creative_inference(self, premises: List[Atom], context: Dict[str, Any]) -> List[Atom]:
        """Creative reasoning: generate novel combinations"""
        conclusions = []
        
        # Combine unrelated concepts to create new ideas
        concepts = [p for p in premises if p.atom_type == AtomType.CONCEPT]
        
        if len(concepts) >= 2:
            # Create novel combinations
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    # Check if they're not already connected
                    if not any(concept2 in link.outgoing for link in concept1.incoming):
                        # Create creative combination
                        creative_concept = self.atomspace.add_node(
                            AtomType.CONCEPT,
                            f"Creative_{concept1.name}_{concept2.name}"
                        )
                        
                        # Link the combination to its components
                        combo_link = self.atomspace.add_link(
                            AtomType.AND,
                            [concept1, concept2, creative_concept],
                            TruthValue(0.3, 0.6)  # High uncertainty but interesting
                        )
                        conclusions.append(combo_link)
        
        return conclusions


class DecisionMaker:
    """Makes decisions based on goals, context, and available actions"""
    
    def __init__(self, reasoning_engine: ReasoningEngine):
        self.reasoning_engine = reasoning_engine
        self.decision_history = deque(maxlen=50)
        
    async def decide_action(self, 
                           available_actions: List[str], 
                           current_state: CognitiveState, 
                           context: Dict[str, Any] = None) -> Tuple[str, float]:
        """Choose the best action given current state and goals"""
        context = context or {}
        
        if not available_actions:
            return "wait", 0.1
        
        action_scores = {}
        
        for action in available_actions:
            score = await self._evaluate_action(action, current_state, context)
            action_scores[action] = score
        
        # Choose action with highest score
        best_action = max(action_scores.keys(), key=lambda x: action_scores[x])
        confidence = action_scores[best_action]
        
        # Record decision
        self.decision_history.append({
            "action": best_action,
            "confidence": confidence,
            "state": current_state,
            "context": context,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return best_action, confidence
    
    async def _evaluate_action(self, 
                               action: str, 
                               state: CognitiveState, 
                               context: Dict[str, Any]) -> float:
        """Evaluate how good an action is given current state"""
        score = 0.0
        
        # Goal alignment
        for goal in state.active_goals:
            if action.lower() in goal.description.lower():
                score += goal.priority * 0.5
        
        # Attention alignment
        action_atom = self.reasoning_engine.atomspace.add_node(AtomType.PROCEDURE, action)
        for focused_atom in state.attention_focus:
            if focused_atom.name.lower() in action.lower():
                score += focused_atom.attention_value * 0.3
        
        # Energy consideration
        if action in ["rest", "wait", "think"]:
            if state.energy_level < 0.3:
                score += 0.4  # Prefer low-energy actions when tired
        else:
            score *= state.energy_level  # Scale by available energy
        
        # Emotional state influence
        if "positive" in state.emotional_state:
            if action in ["help", "create", "explore"]:
                score += state.emotional_state.get("positive", 0) * 0.2
        
        # Context relevance
        if "urgent" in context:
            if action in ["respond", "act", "help"]:
                score += 0.3
        
        return min(1.0, max(0.0, score))


class LearningSystem:
    """Manages learning and adaptation of the cognitive system"""
    
    def __init__(self, atomspace: AtomSpace, reasoning_engine: ReasoningEngine):
        self.atomspace = atomspace
        self.reasoning_engine = reasoning_engine
        self.learning_rate = 0.1
        self.experience_buffer = deque(maxlen=1000)
        
    async def learn_from_experience(self, 
                                    experience: Dict[str, Any], 
                                    outcome: Dict[str, Any]):
        """Learn from a completed experience"""
        self.experience_buffer.append({
            "experience": experience,
            "outcome": outcome,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Update truth values based on outcome
        if "success" in outcome:
            success_rate = outcome["success"]
            await self._update_knowledge_confidence(experience, success_rate)
        
        # Extract patterns from successful experiences
        if outcome.get("success", 0) > 0.7:
            await self._extract_success_patterns(experience)
    
    async def _update_knowledge_confidence(self, 
                                           experience: Dict[str, Any], 
                                           success_rate: float):
        """Update confidence in knowledge based on experience outcomes"""
        if "reasoning_atoms" in experience:
            for atom in experience["reasoning_atoms"]:
                if hasattr(atom, "truth_value"):
                    # Adjust confidence based on success
                    old_confidence = atom.truth_value.confidence
                    confidence_delta = (success_rate - 0.5) * self.learning_rate
                    new_confidence = min(1.0, max(0.0, old_confidence + confidence_delta))
                    atom.truth_value.confidence = new_confidence
    
    async def _extract_success_patterns(self, experience: Dict[str, Any]):
        """Extract patterns from successful experiences for future use"""
        if "context" in experience and "action" in experience:
            context = experience["context"]
            action = experience["action"]
            
            # Create pattern atom
            pattern_name = f"SuccessPattern_{action}_{hash(str(context)) % 1000}"
            pattern_atom = self.atomspace.add_node(AtomType.CONCEPT, pattern_name)
            
            # Link pattern to action
            action_atom = self.atomspace.add_node(AtomType.PROCEDURE, action)
            pattern_link = self.atomspace.add_link(
                AtomType.IMPLICATION,
                [pattern_atom, action_atom],
                TruthValue(0.8, 0.7)
            )
            
            # Add context conditions
            for key, value in context.items():
                condition_atom = self.atomspace.add_node(AtomType.CONCEPT, f"{key}_{value}")
                condition_link = self.atomspace.add_link(
                    AtomType.AND,
                    [pattern_atom, condition_atom],
                    TruthValue(0.7, 0.6)
                )


class CognitiveEngine:
    """Main cognitive processing engine coordinating all cognitive functions"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.atomspace = AtomSpace(name)
        self.reasoning_engine = ReasoningEngine(self.atomspace)
        self.decision_maker = DecisionMaker(self.reasoning_engine)
        self.learning_system = LearningSystem(self.atomspace, self.reasoning_engine)
        
        self.current_state = CognitiveState(
            attention_focus=[],
            active_goals=[],
            working_memory={},
            emotional_state={"neutral": 1.0}
        )
        
        self.running = False
        self.cycle_count = 0
        
    async def start(self):
        """Start the cognitive engine"""
        self.running = True
        asyncio.create_task(self._cognitive_cycle())
    
    async def stop(self):
        """Stop the cognitive engine"""
        self.running = False
    
    async def _cognitive_cycle(self):
        """Main cognitive processing loop"""
        while self.running:
            try:
                await self._execute_cycle()
                await asyncio.sleep(0.1)  # 10 Hz cognitive cycle
                
            except Exception as e:
                print(f"Cognitive cycle error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_cycle(self):
        """Execute one cognitive cycle"""
        self.cycle_count += 1
        
        # Update attention
        self.atomspace.attention_bank.decay_attention()
        
        # Process goals
        await self._process_active_goals()
        
        # Reason about current situation
        if self.current_state.attention_focus:
            reasoning_results = await self.reasoning_engine.reason(
                ReasoningMode.DEDUCTIVE,
                self.current_state.attention_focus,
                self.current_state.working_memory
            )
            
            # Update attention with reasoning results
            if reasoning_results:
                self.atomspace.attention_bank.focus_on(reasoning_results[:3], 0.8)
        
        # Update energy and emotional state
        self._update_internal_state()
    
    async def _process_active_goals(self):
        """Process and prioritize active goals"""
        completed_goals = []
        
        for goal in self.current_state.active_goals:
            if goal.success_criteria and goal.success_criteria():
                completed_goals.append(goal)
        
        # Remove completed goals
        for goal in completed_goals:
            self.current_state.active_goals.remove(goal)
            
            # Learn from goal completion
            await self.learning_system.learn_from_experience(
                {"goal": goal.name, "context": self.current_state.working_memory},
                {"success": 1.0, "completion_time": asyncio.get_event_loop().time() - goal.created_at}
            )
    
    def _update_internal_state(self):
        """Update internal cognitive state"""
        # Energy gradually recovers
        self.current_state.energy_level = min(1.0, self.current_state.energy_level + 0.01)
        
        # Emotional state tends toward neutral
        for emotion in self.current_state.emotional_state:
            current_value = self.current_state.emotional_state[emotion]
            self.current_state.emotional_state[emotion] = current_value * 0.98
        
        # Confidence based on recent success
        recent_decisions = list(self.decision_maker.decision_history)[-10:]
        if recent_decisions:
            avg_confidence = sum(d.get("confidence", 0.5) for d in recent_decisions) / len(recent_decisions)
            self.current_state.confidence_level = avg_confidence
    
    async def add_goal(self, goal: CognitiveGoal):
        """Add a new cognitive goal"""
        self.current_state.active_goals.append(goal)
        
        # Focus attention on goal-related concepts
        goal_atom = self.atomspace.add_node(AtomType.CONCEPT, goal.name)
        self.atomspace.attention_bank.focus_on([goal_atom], goal.priority)
        
        # Update attention focus
        if goal_atom not in self.current_state.attention_focus:
            self.current_state.attention_focus.append(goal_atom)
    
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process external input through the cognitive system"""
        # Add input to working memory
        self.current_state.working_memory.update(input_data)
        
        # Create atoms for input concepts
        input_atoms = []
        for key, value in input_data.items():
            concept = self.atomspace.add_node(AtomType.CONCEPT, str(value))
            input_atoms.append(concept)
        
        # Focus attention on input
        self.atomspace.attention_bank.focus_on(input_atoms, 0.9)
        self.current_state.attention_focus.extend(input_atoms)
        
        # Reason about the input
        reasoning_results = await self.reasoning_engine.reason(
            ReasoningMode.ABDUCTIVE,
            input_atoms,
            input_data
        )
        
        # Make decision about response
        available_actions = ["respond", "ask_clarification", "think", "learn"]
        action, confidence = await self.decision_maker.decide_action(
            available_actions,
            self.current_state,
            input_data
        )
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning_results": [str(atom) for atom in reasoning_results],
            "attention_focus": [str(atom) for atom in self.atomspace.attention_bank.get_focused_atoms()]
        }