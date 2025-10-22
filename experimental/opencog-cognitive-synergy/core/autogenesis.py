"""
Autogenesis System for Self-Modifying Cognitive Architecture

Enables autonomous self-improvement, architectural evolution,
and emergent capability development through recursive self-modification.
"""

import asyncio
import json
import math
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import copy

from .atomspace import AtomSpace, Atom, Link, AtomType, TruthValue
from .cognitive_engine import CognitiveEngine, CognitiveState, CognitiveGoal
from .synergy_detector import SynergyDetector, SynergyPattern, SynergyType


class ModificationType(Enum):
    """Types of self-modifications the system can perform"""
    STRUCTURAL = "structural"        # Modify atomspace structure
    BEHAVIORAL = "behavioral"        # Modify decision patterns
    GOAL_EVOLUTION = "goal_evolution"  # Evolve goal hierarchies
    CAPABILITY_EXPANSION = "capability_expansion"  # Add new capabilities
    EFFICIENCY_OPTIMIZATION = "efficiency_optimization"  # Optimize performance
    SYNERGY_ENHANCEMENT = "synergy_enhancement"  # Improve collaboration
    EMERGENT_ADAPTATION = "emergent_adaptation"  # Spontaneous adaptations


@dataclass
class ModificationProposal:
    """Represents a proposed self-modification"""
    modification_type: ModificationType
    description: str
    target_component: str
    proposed_changes: Dict[str, Any]
    expected_benefits: Dict[str, float]
    risks: Dict[str, float]
    confidence: float
    priority: float
    prerequisites: List[str] = field(default_factory=list)
    created_at: float = field(default=0.0)
    
    def __post_init__(self):
        if self.created_at == 0.0:
            import time
            self.created_at = time.time()


@dataclass
class ModificationResult:
    """Results of an executed modification"""
    proposal: ModificationProposal
    success: bool
    actual_changes: Dict[str, Any]
    measured_benefits: Dict[str, float]
    unexpected_effects: Dict[str, Any]
    performance_impact: float
    executed_at: float = field(default=0.0)
    
    def __post_init__(self):
        if self.executed_at == 0.0:
            import time
            self.executed_at = time.time()


class EvolutionaryOptimizer:
    """Manages evolutionary optimization of cognitive architectures"""
    
    def __init__(self, target_engine: CognitiveEngine):
        self.target_engine = target_engine
        self.modification_history: List[ModificationResult] = []
        self.active_proposals: List[ModificationProposal] = []
        self.performance_metrics = deque(maxlen=1000)
        self.baseline_performance = 0.5
        
    async def generate_modification_proposals(self) -> List[ModificationProposal]:
        """Generate potential self-modification proposals"""
        proposals = []
        
        # Analyze current performance
        current_performance = await self._assess_performance()
        self.performance_metrics.append(current_performance)
        
        # Generate different types of modifications
        proposals.extend(await self._propose_structural_modifications())
        proposals.extend(await self._propose_behavioral_modifications())
        proposals.extend(await self._propose_goal_evolution())
        proposals.extend(await self._propose_capability_expansion())
        proposals.extend(await self._propose_efficiency_optimizations())
        
        # Prioritize proposals
        proposals = await self._prioritize_proposals(proposals)
        
        self.active_proposals = proposals
        return proposals
    
    async def _assess_performance(self) -> float:
        """Assess current performance of the cognitive engine"""
        metrics = {}
        
        # Decision quality (based on confidence and success rate)
        recent_decisions = list(self.target_engine.decision_maker.decision_history)[-20:]
        if recent_decisions:
            avg_confidence = sum(d.get("confidence", 0.5) for d in recent_decisions) / len(recent_decisions)
            metrics["decision_confidence"] = avg_confidence
        else:
            metrics["decision_confidence"] = 0.5
        
        # Goal completion rate
        if hasattr(self.target_engine.learning_system, 'experience_buffer'):
            recent_experiences = list(self.target_engine.learning_system.experience_buffer)[-50:]
            if recent_experiences:
                success_rate = sum(1 for exp in recent_experiences 
                                 if exp.get("outcome", {}).get("success", 0) > 0.7) / len(recent_experiences)
                metrics["goal_success_rate"] = success_rate
            else:
                metrics["goal_success_rate"] = 0.5
        else:
            metrics["goal_success_rate"] = 0.5
        
        # Attention efficiency
        attention_atoms = self.target_engine.atomspace.attention_bank.get_focused_atoms()
        attention_efficiency = len(attention_atoms) / max(len(self.target_engine.current_state.attention_focus), 1)
        metrics["attention_efficiency"] = min(1.0, attention_efficiency)
        
        # Energy utilization
        metrics["energy_utilization"] = self.target_engine.current_state.energy_level
        
        # Overall performance score
        performance = (
            metrics["decision_confidence"] * 0.3 +
            metrics["goal_success_rate"] * 0.3 +
            metrics["attention_efficiency"] * 0.2 +
            metrics["energy_utilization"] * 0.2
        )
        
        return performance
    
    async def _propose_structural_modifications(self) -> List[ModificationProposal]:
        """Propose modifications to atomspace structure"""
        proposals = []
        
        # Analyze atomspace connectivity
        graph = self.target_engine.atomspace.graph
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        
        if node_count > 0:
            connectivity_ratio = edge_count / node_count
            
            # Propose structure optimization if connectivity is low
            if connectivity_ratio < 0.5:
                proposals.append(ModificationProposal(
                    modification_type=ModificationType.STRUCTURAL,
                    description="Increase atomspace connectivity for better knowledge integration",
                    target_component="atomspace.graph",
                    proposed_changes={
                        "add_similarity_links": True,
                        "strengthen_weak_connections": True,
                        "target_connectivity": 0.7
                    },
                    expected_benefits={
                        "reasoning_speed": 0.3,
                        "knowledge_integration": 0.4,
                        "pattern_recognition": 0.25
                    },
                    risks={
                        "computational_overhead": 0.2,
                        "noise_introduction": 0.1
                    },
                    confidence=0.7,
                    priority=0.6
                ))
            
            # Propose structure pruning if connectivity is too high
            elif connectivity_ratio > 2.0:
                proposals.append(ModificationProposal(
                    modification_type=ModificationType.STRUCTURAL,
                    description="Prune excessive connections to reduce noise",
                    target_component="atomspace.graph",
                    proposed_changes={
                        "remove_weak_links": True,
                        "confidence_threshold": 0.3,
                        "target_connectivity": 1.2
                    },
                    expected_benefits={
                        "processing_speed": 0.4,
                        "focus_quality": 0.3,
                        "memory_efficiency": 0.2
                    },
                    risks={
                        "information_loss": 0.3,
                        "reduced_flexibility": 0.2
                    },
                    confidence=0.6,
                    priority=0.5
                ))
        
        return proposals
    
    async def _propose_behavioral_modifications(self) -> List[ModificationProposal]:
        """Propose modifications to decision-making behavior"""
        proposals = []
        
        # Analyze decision patterns
        recent_decisions = list(self.target_engine.decision_maker.decision_history)[-50:]
        
        if recent_decisions:
            # Calculate decision diversity
            actions = [d.get("action", "unknown") for d in recent_decisions]
            action_diversity = len(set(actions)) / len(actions)
            
            # Propose behavior diversification if too repetitive
            if action_diversity < 0.3:
                proposals.append(ModificationProposal(
                    modification_type=ModificationType.BEHAVIORAL,
                    description="Increase behavioral diversity to explore new strategies",
                    target_component="decision_maker",
                    proposed_changes={
                        "exploration_factor": 0.3,
                        "random_action_probability": 0.1,
                        "novelty_bonus": 0.2
                    },
                    expected_benefits={
                        "strategy_discovery": 0.4,
                        "adaptation_speed": 0.3,
                        "creativity": 0.3
                    },
                    risks={
                        "performance_variability": 0.3,
                        "temporary_confusion": 0.2
                    },
                    confidence=0.6,
                    priority=0.4
                ))
            
            # Propose behavior stabilization if too chaotic
            elif action_diversity > 0.8:
                proposals.append(ModificationProposal(
                    modification_type=ModificationType.BEHAVIORAL,
                    description="Stabilize behavior by reinforcing successful patterns",
                    target_component="decision_maker",
                    proposed_changes={
                        "success_reinforcement": 0.4,
                        "pattern_memory_weight": 0.6,
                        "consistency_bonus": 0.3
                    },
                    expected_benefits={
                        "reliability": 0.4,
                        "performance_consistency": 0.4,
                        "efficiency": 0.2
                    },
                    risks={
                        "reduced_adaptability": 0.3,
                        "strategy_staleness": 0.2
                    },
                    confidence=0.7,
                    priority=0.5
                ))
        
        return proposals
    
    async def _propose_goal_evolution(self) -> List[ModificationProposal]:
        """Propose evolution of goal hierarchies"""
        proposals = []
        
        current_goals = self.target_engine.current_state.active_goals
        
        # Propose meta-goals if missing
        has_meta_goals = any("meta" in goal.name.lower() for goal in current_goals)
        
        if not has_meta_goals:
            proposals.append(ModificationProposal(
                modification_type=ModificationType.GOAL_EVOLUTION,
                description="Add meta-cognitive goals for self-awareness",
                target_component="goal_system",
                proposed_changes={
                    "add_goal": {
                        "name": "MetaCognition",
                        "description": "Monitor and improve own thinking processes",
                        "priority": 0.8
                    }
                },
                expected_benefits={
                    "self_awareness": 0.5,
                    "performance_monitoring": 0.4,
                    "adaptive_learning": 0.3
                },
                risks={
                    "computational_overhead": 0.2,
                    "goal_conflict": 0.1
                },
                confidence=0.8,
                priority=0.7
            ))
        
        # Propose goal prioritization adjustment
        if current_goals:
            avg_priority = sum(goal.priority for goal in current_goals) / len(current_goals)
            
            if avg_priority < 0.3:  # Goals have low priority
                proposals.append(ModificationProposal(
                    modification_type=ModificationType.GOAL_EVOLUTION,
                    description="Increase goal priorities to improve motivation",
                    target_component="goal_system",
                    proposed_changes={
                        "priority_boost": 0.3,
                        "dynamic_prioritization": True
                    },
                    expected_benefits={
                        "motivation": 0.4,
                        "goal_pursuit": 0.3,
                        "energy_focus": 0.2
                    },
                    risks={
                        "goal_competition": 0.2,
                        "resource_conflicts": 0.1
                    },
                    confidence=0.6,
                    priority=0.5
                ))
        
        return proposals
    
    async def _propose_capability_expansion(self) -> List[ModificationProposal]:
        """Propose new capability development"""
        proposals = []
        
        # Analyze current capabilities
        atomspace = self.target_engine.atomspace
        capability_atoms = atomspace.get_by_type(AtomType.PROCEDURE)
        
        # Propose missing basic capabilities
        basic_capabilities = ["learn", "reason", "communicate", "adapt", "collaborate"]
        current_capability_names = {atom.name.lower() for atom in capability_atoms}
        
        missing_capabilities = [cap for cap in basic_capabilities 
                              if cap not in current_capability_names]
        
        for capability in missing_capabilities:
            proposals.append(ModificationProposal(
                modification_type=ModificationType.CAPABILITY_EXPANSION,
                description=f"Add {capability} capability to cognitive repertoire",
                target_component="capability_system",
                proposed_changes={
                    "new_capability": {
                        "name": capability,
                        "type": "basic_cognitive",
                        "implementation": "adaptive_learning"
                    }
                },
                expected_benefits={
                    "functional_completeness": 0.3,
                    "problem_solving": 0.2,
                    "adaptability": 0.2
                },
                risks={
                    "complexity_increase": 0.2,
                    "integration_challenges": 0.1
                },
                confidence=0.7,
                priority=0.6
            ))
        
        return proposals
    
    async def _propose_efficiency_optimizations(self) -> List[ModificationProposal]:
        """Propose performance efficiency improvements"""
        proposals = []
        
        # Analyze processing efficiency
        recent_performance = self.performance_metrics[-10:] if self.performance_metrics else [0.5]
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        if avg_performance < self.baseline_performance:
            proposals.append(ModificationProposal(
                modification_type=ModificationType.EFFICIENCY_OPTIMIZATION,
                description="Optimize cognitive processing for better performance",
                target_component="processing_engine",
                proposed_changes={
                    "attention_focus_optimization": True,
                    "memory_access_caching": True,
                    "decision_shortcutting": True,
                    "parallel_processing": True
                },
                expected_benefits={
                    "processing_speed": 0.4,
                    "response_time": 0.3,
                    "energy_efficiency": 0.2,
                    "throughput": 0.3
                },
                risks={
                    "accuracy_reduction": 0.2,
                    "complexity_increase": 0.1
                },
                confidence=0.8,
                priority=0.8
            ))
        
        return proposals
    
    async def _prioritize_proposals(self, proposals: List[ModificationProposal]) -> List[ModificationProposal]:
        """Prioritize modification proposals based on multiple criteria"""
        
        def calculate_priority_score(proposal: ModificationProposal) -> float:
            # Base priority from proposal
            base_score = proposal.priority
            
            # Adjust by confidence
            confidence_factor = proposal.confidence
            
            # Adjust by expected benefits vs risks
            total_benefits = sum(proposal.expected_benefits.values())
            total_risks = sum(proposal.risks.values())
            benefit_risk_ratio = total_benefits / max(total_risks, 0.1)
            
            # Combine factors
            priority_score = base_score * confidence_factor * min(benefit_risk_ratio, 2.0)
            
            return priority_score
        
        # Calculate priority scores
        for proposal in proposals:
            proposal.priority = calculate_priority_score(proposal)
        
        # Sort by priority score
        return sorted(proposals, key=lambda p: p.priority, reverse=True)
    
    async def execute_modification(self, proposal: ModificationProposal) -> ModificationResult:
        """Execute a proposed modification"""
        
        try:
            # Record pre-modification state
            pre_performance = await self._assess_performance()
            
            # Execute the modification
            actual_changes = await self._apply_modification(proposal)
            
            # Wait for system to stabilize
            await asyncio.sleep(0.5)
            
            # Measure post-modification performance
            post_performance = await self._assess_performance()
            performance_impact = post_performance - pre_performance
            
            # Measure actual benefits
            measured_benefits = await self._measure_benefits(proposal, actual_changes)
            
            # Detect unexpected effects
            unexpected_effects = await self._detect_unexpected_effects(proposal, actual_changes)
            
            result = ModificationResult(
                proposal=proposal,
                success=True,
                actual_changes=actual_changes,
                measured_benefits=measured_benefits,
                unexpected_effects=unexpected_effects,
                performance_impact=performance_impact
            )
            
            self.modification_history.append(result)
            return result
            
        except Exception as e:
            # Handle modification failure
            result = ModificationResult(
                proposal=proposal,
                success=False,
                actual_changes={},
                measured_benefits={},
                unexpected_effects={"error": str(e)},
                performance_impact=0.0
            )
            
            self.modification_history.append(result)
            return result
    
    async def _apply_modification(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """Apply the actual changes specified in a modification proposal"""
        actual_changes = {}
        
        if proposal.modification_type == ModificationType.STRUCTURAL:
            actual_changes = await self._apply_structural_modification(proposal)
            
        elif proposal.modification_type == ModificationType.BEHAVIORAL:
            actual_changes = await self._apply_behavioral_modification(proposal)
            
        elif proposal.modification_type == ModificationType.GOAL_EVOLUTION:
            actual_changes = await self._apply_goal_modification(proposal)
            
        elif proposal.modification_type == ModificationType.CAPABILITY_EXPANSION:
            actual_changes = await self._apply_capability_modification(proposal)
            
        elif proposal.modification_type == ModificationType.EFFICIENCY_OPTIMIZATION:
            actual_changes = await self._apply_efficiency_modification(proposal)
        
        return actual_changes
    
    async def _apply_structural_modification(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """Apply structural modifications to the atomspace"""
        changes = {}
        atomspace = self.target_engine.atomspace
        
        if proposal.proposed_changes.get("add_similarity_links"):
            # Add similarity links between concepts
            concepts = list(atomspace.get_by_type(AtomType.CONCEPT))
            links_added = 0
            
            for i, concept1 in enumerate(concepts):
                similar_concepts = atomspace.find_similar_concepts(concept1, threshold=0.6)
                for similar_concept, similarity in similar_concepts[:3]:  # Limit to top 3
                    link = atomspace.add_link(
                        AtomType.SIMILARITY,
                        [concept1, similar_concept],
                        TruthValue(similarity, 0.7)
                    )
                    links_added += 1
            
            changes["similarity_links_added"] = links_added
        
        if proposal.proposed_changes.get("strengthen_weak_connections"):
            # Strengthen weak but potentially important connections
            strengthened = 0
            for atom in atomspace.atoms.values():
                if hasattr(atom, 'truth_value') and atom.truth_value.confidence < 0.3:
                    if atom.truth_value.strength > 0.6:  # High strength, low confidence
                        atom.truth_value.confidence = min(1.0, atom.truth_value.confidence + 0.2)
                        strengthened += 1
            
            changes["connections_strengthened"] = strengthened
        
        return changes
    
    async def _apply_behavioral_modification(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """Apply behavioral modifications to decision making"""
        changes = {}
        decision_maker = self.target_engine.decision_maker
        
        if "exploration_factor" in proposal.proposed_changes:
            # This would modify the decision maker's exploration behavior
            # In a full implementation, this would modify the decision algorithm
            changes["exploration_factor"] = proposal.proposed_changes["exploration_factor"]
        
        if "success_reinforcement" in proposal.proposed_changes:
            # This would modify how successful decisions are reinforced
            changes["success_reinforcement"] = proposal.proposed_changes["success_reinforcement"]
        
        return changes
    
    async def _apply_goal_modification(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """Apply goal system modifications"""
        changes = {}
        
        if "add_goal" in proposal.proposed_changes:
            goal_spec = proposal.proposed_changes["add_goal"]
            new_goal = CognitiveGoal(
                name=goal_spec["name"],
                description=goal_spec["description"],
                priority=goal_spec.get("priority", 0.5)
            )
            
            await self.target_engine.add_goal(new_goal)
            changes["goal_added"] = goal_spec["name"]
        
        if "priority_boost" in proposal.proposed_changes:
            boost = proposal.proposed_changes["priority_boost"]
            goals_modified = 0
            
            for goal in self.target_engine.current_state.active_goals:
                goal.priority = min(1.0, goal.priority + boost)
                goals_modified += 1
            
            changes["goals_priority_boosted"] = goals_modified
        
        return changes
    
    async def _apply_capability_modification(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """Apply capability expansion modifications"""
        changes = {}
        atomspace = self.target_engine.atomspace
        
        if "new_capability" in proposal.proposed_changes:
            capability_spec = proposal.proposed_changes["new_capability"]
            capability_atom = atomspace.add_node(
                AtomType.PROCEDURE,
                capability_spec["name"],
                TruthValue(0.7, 0.6)
            )
            
            changes["capability_added"] = capability_spec["name"]
        
        return changes
    
    async def _apply_efficiency_modification(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """Apply efficiency optimization modifications"""
        changes = {}
        
        # These would involve modifying internal algorithms
        # For demonstration, we'll just record the intended optimizations
        
        for optimization, enabled in proposal.proposed_changes.items():
            if enabled:
                changes[f"optimization_{optimization}"] = True
        
        return changes
    
    async def _measure_benefits(self, proposal: ModificationProposal, changes: Dict[str, Any]) -> Dict[str, float]:
        """Measure actual benefits achieved by a modification"""
        # This would involve comparing performance metrics before and after
        # For now, we'll estimate based on the changes made
        
        measured_benefits = {}
        
        for benefit, expected_value in proposal.expected_benefits.items():
            # Simulate measurement with some variance
            actual_value = expected_value * np.random.normal(1.0, 0.2)
            measured_benefits[benefit] = max(0.0, actual_value)
        
        return measured_benefits
    
    async def _detect_unexpected_effects(self, proposal: ModificationProposal, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Detect any unexpected side effects of a modification"""
        # This would involve monitoring for changes not anticipated in the proposal
        
        unexpected_effects = {}
        
        # For demonstration, randomly generate some potential effects
        if np.random.random() < 0.1:  # 10% chance of unexpected effect
            unexpected_effects["performance_fluctuation"] = np.random.uniform(-0.1, 0.1)
        
        if np.random.random() < 0.05:  # 5% chance of emergent behavior
            unexpected_effects["emergent_behavior"] = "New reasoning pattern discovered"
        
        return unexpected_effects


class AutogenesisSystem:
    """Main autogenesis system coordinating all self-modification capabilities"""
    
    def __init__(self, cognitive_engine: CognitiveEngine):
        self.cognitive_engine = cognitive_engine
        self.evolutionary_optimizer = EvolutionaryOptimizer(cognitive_engine)
        self.modification_queue = asyncio.Queue()
        self.running = False
        
        # Self-modification parameters
        self.modification_interval = 30.0  # seconds
        self.max_modifications_per_cycle = 3
        self.safety_threshold = 0.3  # Minimum acceptable performance
        
    async def start(self):
        """Start the autogenesis system"""
        self.running = True
        asyncio.create_task(self._autogenesis_loop())
    
    async def stop(self):
        """Stop the autogenesis system"""
        self.running = False
    
    async def _autogenesis_loop(self):
        """Main self-modification loop"""
        while self.running:
            try:
                await self._execute_autogenesis_cycle()
                await asyncio.sleep(self.modification_interval)
                
            except Exception as e:
                print(f"Autogenesis error: {e}")
                await asyncio.sleep(5.0)
    
    async def _execute_autogenesis_cycle(self):
        """Execute one cycle of self-modification"""
        
        # Generate modification proposals
        proposals = await self.evolutionary_optimizer.generate_modification_proposals()
        
        if not proposals:
            return
        
        # Check safety constraints
        current_performance = await self.evolutionary_optimizer._assess_performance()
        if current_performance < self.safety_threshold:
            print(f"Performance below safety threshold ({current_performance:.3f} < {self.safety_threshold})")
            return
        
        # Execute top modifications
        modifications_executed = 0
        
        for proposal in proposals[:self.max_modifications_per_cycle]:
            if modifications_executed >= self.max_modifications_per_cycle:
                break
            
            # Safety check: don't execute risky modifications if performance is borderline
            total_risk = sum(proposal.risks.values())
            if current_performance < 0.6 and total_risk > 0.3:
                continue
            
            print(f"Executing modification: {proposal.description}")
            result = await self.evolutionary_optimizer.execute_modification(proposal)
            
            if result.success:
                print(f"Modification successful. Performance impact: {result.performance_impact:.3f}")
                modifications_executed += 1
                
                # If modification caused significant performance drop, consider reverting
                if result.performance_impact < -0.2:
                    print("Significant performance drop detected - modification monitoring required")
            else:
                print(f"Modification failed: {result.unexpected_effects}")
        
        print(f"Autogenesis cycle completed. {modifications_executed} modifications executed.")
    
    async def propose_emergency_modification(self, issue_description: str) -> List[ModificationProposal]:
        """Generate emergency modifications to address critical issues"""
        
        proposals = []
        
        if "performance" in issue_description.lower():
            # Emergency performance optimization
            proposal = ModificationProposal(
                modification_type=ModificationType.EFFICIENCY_OPTIMIZATION,
                description="Emergency performance optimization",
                target_component="all_systems",
                proposed_changes={
                    "reduce_processing_overhead": True,
                    "focus_attention": True,
                    "prioritize_critical_goals": True
                },
                expected_benefits={
                    "immediate_performance": 0.5,
                    "stability": 0.3
                },
                risks={
                    "feature_reduction": 0.2
                },
                confidence=0.7,
                priority=1.0  # Maximum priority
            )
            proposals.append(proposal)
        
        if "error" in issue_description.lower() or "failure" in issue_description.lower():
            # Emergency stability enhancement
            proposal = ModificationProposal(
                modification_type=ModificationType.BEHAVIORAL,
                description="Emergency stability enhancement",
                target_component="error_handling",
                proposed_changes={
                    "increase_error_tolerance": True,
                    "enable_safe_mode": True,
                    "reduce_complexity": True
                },
                expected_benefits={
                    "stability": 0.6,
                    "error_recovery": 0.4
                },
                risks={
                    "reduced_capabilities": 0.3
                },
                confidence=0.8,
                priority=1.0
            )
            proposals.append(proposal)
        
        return proposals
    
    def get_modification_history(self) -> List[ModificationResult]:
        """Get history of all modifications performed"""
        return self.evolutionary_optimizer.modification_history.copy()
    
    def get_performance_trend(self) -> List[float]:
        """Get recent performance trend data"""
        return list(self.evolutionary_optimizer.performance_metrics)
    
    async def force_evolution(self, target_capability: str) -> bool:
        """Force evolution toward a specific capability"""
        
        # Generate targeted proposals
        proposal = ModificationProposal(
            modification_type=ModificationType.CAPABILITY_EXPANSION,
            description=f"Forced evolution toward {target_capability}",
            target_component="capability_system",
            proposed_changes={
                "target_capability": target_capability,
                "accelerated_learning": True,
                "focused_development": True
            },
            expected_benefits={
                target_capability: 0.8,
                "adaptability": 0.3
            },
            risks={
                "resource_concentration": 0.4,
                "other_capability_neglect": 0.3
            },
            confidence=0.6,
            priority=0.9
        )
        
        result = await self.evolutionary_optimizer.execute_modification(proposal)
        return result.success