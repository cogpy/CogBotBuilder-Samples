"""
Cognitive Synergy Detection and Amplification System

Identifies emergent patterns, cognitive synergies between agents,
and opportunities for collaborative intelligence amplification.
"""

import asyncio
import math
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from .atomspace import AtomSpace, Atom, Link, AtomType, TruthValue
from .cognitive_engine import CognitiveEngine, CognitiveState


class SynergyType(Enum):
    """Types of cognitive synergies that can be detected"""
    COMPLEMENTARY = "complementary"     # Agents with different strengths
    RESONANT = "resonant"              # Agents thinking similarly
    EMERGENT = "emergent"              # New capabilities from combination
    AMPLIFYING = "amplifying"          # Mutual reinforcement
    CREATIVE = "creative"              # Novel idea generation
    COLLABORATIVE = "collaborative"    # Task coordination


@dataclass
class SynergyPattern:
    """Represents a detected synergy pattern"""
    synergy_type: SynergyType
    agents: List[str]
    strength: float
    confidence: float
    pattern_atoms: List[Atom]
    benefits: Dict[str, float]
    context: Dict[str, Any]
    detected_at: float
    duration: float = 0.0
    
    def __post_init__(self):
        if self.detected_at == 0.0:
            import time
            self.detected_at = time.time()


class SynergyDetector:
    """Detects and analyzes cognitive synergies between agents"""
    
    def __init__(self, master_atomspace: AtomSpace):
        self.master_atomspace = master_atomspace
        self.agent_states: Dict[str, CognitiveState] = {}
        self.synergy_patterns: List[SynergyPattern] = []
        self.synergy_history = deque(maxlen=1000)
        self.detection_thresholds = {
            SynergyType.COMPLEMENTARY: 0.6,
            SynergyType.RESONANT: 0.7,
            SynergyType.EMERGENT: 0.5,
            SynergyType.AMPLIFYING: 0.65,
            SynergyType.CREATIVE: 0.4,
            SynergyType.COLLABORATIVE: 0.7
        }
    
    async def register_agent(self, agent_id: str, cognitive_state: CognitiveState):
        """Register an agent for synergy monitoring"""
        self.agent_states[agent_id] = cognitive_state
        
        # Add agent concept to master atomspace
        agent_atom = self.master_atomspace.add_node(AtomType.CONCEPT, f"Agent_{agent_id}")
        
        # Create capabilities assessment
        await self._assess_agent_capabilities(agent_id, cognitive_state)
    
    async def detect_synergies(self) -> List[SynergyPattern]:
        """Detect all types of synergies between registered agents"""
        detected_synergies = []
        
        agent_ids = list(self.agent_states.keys())
        
        # Check all pairs of agents
        for i, agent1 in enumerate(agent_ids):
            for agent2 in agent_ids[i+1:]:
                synergies = await self._analyze_agent_pair(agent1, agent2)
                detected_synergies.extend(synergies)
        
        # Check group synergies (3+ agents)
        if len(agent_ids) >= 3:
            group_synergies = await self._analyze_agent_groups(agent_ids)
            detected_synergies.extend(group_synergies)
        
        # Filter by confidence thresholds and update patterns
        valid_synergies = []
        for synergy in detected_synergies:
            if synergy.confidence >= self.detection_thresholds.get(synergy.synergy_type, 0.5):
                valid_synergies.append(synergy)
                await self._record_synergy(synergy)
        
        self.synergy_patterns = valid_synergies
        return valid_synergies
    
    async def _analyze_agent_pair(self, agent1: str, agent2: str) -> List[SynergyPattern]:
        """Analyze synergy potential between two agents"""
        synergies = []
        state1 = self.agent_states[agent1]
        state2 = self.agent_states[agent2]
        
        # Complementary synergy detection
        complementary = await self._detect_complementary_synergy(agent1, agent2, state1, state2)
        if complementary:
            synergies.append(complementary)
        
        # Resonant synergy detection
        resonant = await self._detect_resonant_synergy(agent1, agent2, state1, state2)
        if resonant:
            synergies.append(resonant)
        
        # Amplifying synergy detection
        amplifying = await self._detect_amplifying_synergy(agent1, agent2, state1, state2)
        if amplifying:
            synergies.append(amplifying)
        
        # Creative synergy detection
        creative = await self._detect_creative_synergy(agent1, agent2, state1, state2)
        if creative:
            synergies.append(creative)
        
        return synergies
    
    async def _detect_complementary_synergy(self, 
                                            agent1: str, agent2: str,
                                            state1: CognitiveState, 
                                            state2: CognitiveState) -> Optional[SynergyPattern]:
        """Detect complementary capabilities between agents"""
        
        # Compare goal types and capabilities
        goals1 = {goal.name for goal in state1.active_goals}
        goals2 = {goal.name for goal in state2.active_goals}
        
        # Different but compatible goals indicate complementarity
        goal_overlap = len(goals1 & goals2) / max(len(goals1 | goals2), 1)
        complementarity_score = 1.0 - goal_overlap  # Higher when goals are different
        
        # Compare attention focus
        focus1 = {atom.name for atom in state1.attention_focus}
        focus2 = {atom.name for atom in state2.attention_focus}
        
        focus_overlap = len(focus1 & focus2) / max(len(focus1 | focus2), 1)
        attention_complementarity = 1.0 - focus_overlap
        
        # Compare emotional states for balance
        emotion_balance = self._calculate_emotional_balance(
            state1.emotional_state, 
            state2.emotional_state
        )
        
        # Overall complementarity strength
        strength = (complementarity_score * 0.4 + 
                   attention_complementarity * 0.4 + 
                   emotion_balance * 0.2)
        
        confidence = min(state1.confidence_level, state2.confidence_level) * 0.8
        
        if strength >= self.detection_thresholds[SynergyType.COMPLEMENTARY]:
            return SynergyPattern(
                synergy_type=SynergyType.COMPLEMENTARY,
                agents=[agent1, agent2],
                strength=strength,
                confidence=confidence,
                pattern_atoms=state1.attention_focus + state2.attention_focus,
                benefits={
                    "capability_coverage": strength,
                    "task_efficiency": strength * 0.8,
                    "robustness": emotion_balance
                },
                context={
                    "goal_overlap": goal_overlap,
                    "focus_overlap": focus_overlap,
                    "emotion_balance": emotion_balance
                },
                detected_at=0.0
            )
        
        return None
    
    async def _detect_resonant_synergy(self,
                                       agent1: str, agent2: str,
                                       state1: CognitiveState,
                                       state2: CognitiveState) -> Optional[SynergyPattern]:
        """Detect resonant thinking patterns between agents"""
        
        # Compare attention patterns
        focus1 = {atom.name for atom in state1.attention_focus}
        focus2 = {atom.name for atom in state2.attention_focus}
        
        focus_resonance = len(focus1 & focus2) / max(len(focus1 | focus2), 1)
        
        # Compare working memory content
        memory1_concepts = set(str(v) for v in state1.working_memory.values())
        memory2_concepts = set(str(v) for v in state2.working_memory.values())
        
        memory_resonance = len(memory1_concepts & memory2_concepts) / max(len(memory1_concepts | memory2_concepts), 1)
        
        # Compare emotional states for alignment
        emotion_alignment = self._calculate_emotional_alignment(
            state1.emotional_state,
            state2.emotional_state
        )
        
        # Overall resonance strength
        strength = (focus_resonance * 0.5 + 
                   memory_resonance * 0.3 + 
                   emotion_alignment * 0.2)
        
        confidence = (state1.confidence_level + state2.confidence_level) / 2
        
        if strength >= self.detection_thresholds[SynergyType.RESONANT]:
            return SynergyPattern(
                synergy_type=SynergyType.RESONANT,
                agents=[agent1, agent2],
                strength=strength,
                confidence=confidence,
                pattern_atoms=list(set(state1.attention_focus + state2.attention_focus)),
                benefits={
                    "coordination_efficiency": strength,
                    "decision_speed": strength * 0.9,
                    "consistency": emotion_alignment
                },
                context={
                    "focus_resonance": focus_resonance,
                    "memory_resonance": memory_resonance,
                    "emotion_alignment": emotion_alignment
                },
                detected_at=0.0
            )
        
        return None
    
    async def _detect_amplifying_synergy(self,
                                         agent1: str, agent2: str,
                                         state1: CognitiveState,
                                         state2: CognitiveState) -> Optional[SynergyPattern]:
        """Detect mutually amplifying cognitive patterns"""
        
        # Check if agents have similar goals that could be mutually reinforced
        goals1 = {goal.name for goal in state1.active_goals}
        goals2 = {goal.name for goal in state2.active_goals}
        
        shared_goals = goals1 & goals2
        goal_synergy = len(shared_goals) / max(len(goals1 | goals2), 1)
        
        # Check confidence levels - higher confidence can amplify lower confidence
        confidence_amplification = min(state1.confidence_level, state2.confidence_level) * \
                                  max(state1.confidence_level, state2.confidence_level)
        
        # Check energy levels for mutual support
        energy_synergy = (state1.energy_level + state2.energy_level) / 2
        
        # Overall amplification potential
        strength = (goal_synergy * 0.4 + 
                   confidence_amplification * 0.4 + 
                   energy_synergy * 0.2)
        
        confidence = math.sqrt(state1.confidence_level * state2.confidence_level)
        
        if strength >= self.detection_thresholds[SynergyType.AMPLIFYING]:
            return SynergyPattern(
                synergy_type=SynergyType.AMPLIFYING,
                agents=[agent1, agent2],
                strength=strength,
                confidence=confidence,
                pattern_atoms=state1.attention_focus + state2.attention_focus,
                benefits={
                    "performance_boost": strength * 1.2,
                    "confidence_increase": confidence_amplification,
                    "energy_efficiency": energy_synergy
                },
                context={
                    "shared_goals": list(shared_goals),
                    "confidence_diff": abs(state1.confidence_level - state2.confidence_level),
                    "energy_synergy": energy_synergy
                },
                detected_at=0.0
            )
        
        return None
    
    async def _detect_creative_synergy(self,
                                       agent1: str, agent2: str,
                                       state1: CognitiveState,
                                       state2: CognitiveState) -> Optional[SynergyPattern]:
        """Detect potential for creative collaboration"""
        
        # Check diversity in attention focus (different perspectives for creativity)
        focus1 = set(atom.name for atom in state1.attention_focus)
        focus2 = set(atom.name for atom in state2.attention_focus)
        
        focus_diversity = 1.0 - (len(focus1 & focus2) / max(len(focus1 | focus2), 1))
        
        # Check for complementary emotional states that enhance creativity
        creativity_emotions1 = state1.emotional_state.get("curiosity", 0) + \
                             state1.emotional_state.get("positive", 0)
        creativity_emotions2 = state2.emotional_state.get("curiosity", 0) + \
                             state2.emotional_state.get("positive", 0)
        
        creative_potential = (creativity_emotions1 + creativity_emotions2) / 2
        
        # Check for high energy levels (needed for creative work)
        energy_for_creativity = min(state1.energy_level, state2.energy_level)
        
        # Overall creative synergy
        strength = (focus_diversity * 0.4 + 
                   creative_potential * 0.4 + 
                   energy_for_creativity * 0.2)
        
        confidence = (state1.confidence_level + state2.confidence_level) / 2 * 0.7  # Creative work has inherent uncertainty
        
        if strength >= self.detection_thresholds[SynergyType.CREATIVE]:
            return SynergyPattern(
                synergy_type=SynergyType.CREATIVE,
                agents=[agent1, agent2],
                strength=strength,
                confidence=confidence,
                pattern_atoms=list(set(state1.attention_focus + state2.attention_focus)),
                benefits={
                    "innovation_potential": strength * 1.5,
                    "novel_solutions": focus_diversity,
                    "creative_energy": creative_potential
                },
                context={
                    "focus_diversity": focus_diversity,
                    "creative_emotions": creative_potential,
                    "energy_level": energy_for_creativity
                },
                detected_at=0.0
            )
        
        return None
    
    async def _analyze_agent_groups(self, agent_ids: List[str]) -> List[SynergyPattern]:
        """Analyze synergies in groups of 3+ agents"""
        group_synergies = []
        
        # For now, analyze groups of 3 (can be extended)
        for i in range(len(agent_ids)):
            for j in range(i+1, len(agent_ids)):
                for k in range(j+1, len(agent_ids)):
                    group = [agent_ids[i], agent_ids[j], agent_ids[k]]
                    synergy = await self._detect_group_synergy(group)
                    if synergy:
                        group_synergies.append(synergy)
        
        return group_synergies
    
    async def _detect_group_synergy(self, agent_group: List[str]) -> Optional[SynergyPattern]:
        """Detect emergent synergy in a group of agents"""
        
        states = [self.agent_states[agent_id] for agent_id in agent_group]
        
        # Check for diverse but coordinated goals
        all_goals = set()
        for state in states:
            all_goals.update(goal.name for goal in state.active_goals)
        
        goal_diversity = len(all_goals) / len(agent_group) if agent_group else 0
        
        # Check for distributed attention coverage
        all_focus = set()
        for state in states:
            all_focus.update(atom.name for atom in state.attention_focus)
        
        attention_coverage = len(all_focus) / max(sum(len(state.attention_focus) for state in states), 1)
        
        # Check for balanced energy and confidence
        avg_energy = sum(state.energy_level for state in states) / len(states)
        avg_confidence = sum(state.confidence_level for state in states) / len(states)
        
        energy_balance = 1.0 - np.std([state.energy_level for state in states])
        confidence_balance = 1.0 - np.std([state.confidence_level for state in states])
        
        # Overall group synergy
        strength = (goal_diversity * 0.3 + 
                   attention_coverage * 0.3 + 
                   avg_energy * 0.2 + 
                   (energy_balance + confidence_balance) * 0.2)
        
        confidence = avg_confidence * min(1.0, len(agent_group) / 5.0)  # Group confidence scaled by size
        
        if strength >= self.detection_thresholds[SynergyType.EMERGENT]:
            all_atoms = []
            for state in states:
                all_atoms.extend(state.attention_focus)
            
            return SynergyPattern(
                synergy_type=SynergyType.EMERGENT,
                agents=agent_group,
                strength=strength,
                confidence=confidence,
                pattern_atoms=list(set(all_atoms)),
                benefits={
                    "emergent_intelligence": strength * 1.3,
                    "problem_solving": goal_diversity,
                    "resource_efficiency": attention_coverage,
                    "group_coherence": (energy_balance + confidence_balance) / 2
                },
                context={
                    "group_size": len(agent_group),
                    "goal_diversity": goal_diversity,
                    "attention_coverage": attention_coverage,
                    "balance_metrics": {
                        "energy": energy_balance,
                        "confidence": confidence_balance
                    }
                },
                detected_at=0.0
            )
        
        return None
    
    def _calculate_emotional_balance(self, 
                                     emotions1: Dict[str, float], 
                                     emotions2: Dict[str, float]) -> float:
        """Calculate how well two emotional states balance each other"""
        
        # Look for complementary emotional patterns
        complementary_pairs = [
            ("calm", "excited"),
            ("analytical", "creative"),
            ("confident", "curious"),
            ("focused", "exploratory")
        ]
        
        balance_score = 0.0
        pair_count = 0
        
        for emotion_a, emotion_b in complementary_pairs:
            val_a1 = emotions1.get(emotion_a, 0.0)
            val_b1 = emotions1.get(emotion_b, 0.0)
            val_a2 = emotions2.get(emotion_a, 0.0)
            val_b2 = emotions2.get(emotion_b, 0.0)
            
            # Check if agents have complementary strengths in this pair
            if (val_a1 > val_b1 and val_b2 > val_a2) or (val_b1 > val_a1 and val_a2 > val_b2):
                balance_score += 1.0
            
            pair_count += 1
        
        return balance_score / max(pair_count, 1)
    
    def _calculate_emotional_alignment(self, 
                                       emotions1: Dict[str, float], 
                                       emotions2: Dict[str, float]) -> float:
        """Calculate emotional alignment between two agents"""
        
        all_emotions = set(emotions1.keys()) | set(emotions2.keys())
        if not all_emotions:
            return 0.5  # Neutral alignment
        
        alignment_sum = 0.0
        for emotion in all_emotions:
            val1 = emotions1.get(emotion, 0.0)
            val2 = emotions2.get(emotion, 0.0)
            
            # Calculate similarity (inverse of difference)
            alignment_sum += 1.0 - abs(val1 - val2)
        
        return alignment_sum / len(all_emotions)
    
    async def _assess_agent_capabilities(self, agent_id: str, state: CognitiveState):
        """Assess and record an agent's capabilities in the master atomspace"""
        
        # Create capability atoms based on goals and focus
        capabilities = []
        
        for goal in state.active_goals:
            capability_atom = self.master_atomspace.add_node(
                AtomType.CONCEPT, 
                f"Capability_{goal.name}"
            )
            capabilities.append(capability_atom)
        
        # Link agent to capabilities
        agent_atom = self.master_atomspace.add_node(AtomType.CONCEPT, f"Agent_{agent_id}")
        
        for capability in capabilities:
            self.master_atomspace.add_link(
                AtomType.EVALUATION,
                [
                    self.master_atomspace.add_node(AtomType.PREDICATE, "has_capability"),
                    self.master_atomspace.add_link(AtomType.LIST, [agent_atom, capability])
                ],
                TruthValue(0.8, state.confidence_level)
            )
    
    async def _record_synergy(self, synergy: SynergyPattern):
        """Record a detected synergy in the master atomspace and history"""
        
        # Add to history
        self.synergy_history.append(synergy)
        
        # Create synergy atoms in master atomspace
        synergy_atom = self.master_atomspace.add_node(
            AtomType.CONCEPT,
            f"Synergy_{synergy.synergy_type.value}_{hash(tuple(synergy.agents)) % 1000}"
        )
        
        # Link synergy to participating agents
        agent_atoms = [
            self.master_atomspace.add_node(AtomType.CONCEPT, f"Agent_{agent_id}")
            for agent_id in synergy.agents
        ]
        
        synergy_link = self.master_atomspace.add_link(
            AtomType.EVALUATION,
            [
                self.master_atomspace.add_node(AtomType.PREDICATE, "exhibits_synergy"),
                self.master_atomspace.add_link(AtomType.LIST, [synergy_atom] + agent_atoms)
            ],
            TruthValue(synergy.strength, synergy.confidence)
        )
    
    def get_synergy_recommendations(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get synergy recommendations for a specific agent"""
        recommendations = []
        
        for synergy in self.synergy_patterns:
            if agent_id in synergy.agents:
                recommendations.append({
                    "synergy_type": synergy.synergy_type.value,
                    "partners": [a for a in synergy.agents if a != agent_id],
                    "strength": synergy.strength,
                    "benefits": synergy.benefits,
                    "suggested_actions": self._generate_synergy_actions(synergy)
                })
        
        return sorted(recommendations, key=lambda x: x["strength"], reverse=True)
    
    def _generate_synergy_actions(self, synergy: SynergyPattern) -> List[str]:
        """Generate suggested actions to leverage a synergy"""
        actions = []
        
        if synergy.synergy_type == SynergyType.COMPLEMENTARY:
            actions.extend([
                "coordinate_task_division",
                "share_specialized_knowledge",
                "provide_mutual_support"
            ])
        elif synergy.synergy_type == SynergyType.RESONANT:
            actions.extend([
                "synchronize_actions",
                "share_decision_making",
                "coordinate_responses"
            ])
        elif synergy.synergy_type == SynergyType.AMPLIFYING:
            actions.extend([
                "reinforce_shared_goals",
                "boost_confidence",
                "coordinate_energy_usage"
            ])
        elif synergy.synergy_type == SynergyType.CREATIVE:
            actions.extend([
                "brainstorm_together",
                "combine_perspectives",
                "generate_novel_solutions"
            ])
        elif synergy.synergy_type == SynergyType.EMERGENT:
            actions.extend([
                "form_collaborative_group",
                "distribute_complex_tasks",
                "leverage_collective_intelligence"
            ])
        
        return actions