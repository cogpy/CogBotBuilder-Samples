"""
Multi-Bot Coordination System for Cognitive Synergy Architecture

Manages multiple cognitive bots, orchestrates their interactions,
and facilitates emergent collective intelligence through synergy detection.
"""

import asyncio
import json
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from ..core.atomspace import AtomSpace, AtomType, TruthValue, create_evaluation
from ..core.synergy_detector import SynergyDetector, SynergyPattern, SynergyType
from ..agents.cognitive_bot import CognitiveBot


class CoordinationStrategy(Enum):
    """Different strategies for coordinating multiple bots"""
    AUTONOMOUS = "autonomous"           # Minimal coordination, bots act independently
    COLLABORATIVE = "collaborative"    # Active collaboration on shared tasks
    HIERARCHICAL = "hierarchical"      # Master-worker coordination
    EMERGENT = "emergent"             # Emergent coordination through synergy
    ADAPTIVE = "adaptive"             # Dynamically adapt coordination strategy


@dataclass
class CoordinationTask:
    """Represents a task requiring coordination between bots"""
    task_id: str
    description: str
    required_capabilities: List[str]
    assigned_bots: List[str] = field(default_factory=list)
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.COLLABORATIVE
    priority: float = 0.5
    deadline: Optional[float] = None
    progress: float = 0.0
    status: str = "pending"
    created_at: float = field(default=0.0)
    
    def __post_init__(self):
        if self.created_at == 0.0:
            import time
            self.created_at = time.time()


class MultiBotCoordinator:
    """Coordinates multiple cognitive bots for collective intelligence"""
    
    def __init__(self, coordinator_id: str = "multi_bot_coordinator"):
        self.coordinator_id = coordinator_id
        self.registered_bots: Dict[str, CognitiveBot] = {}
        self.master_atomspace = AtomSpace("master_coordination")
        self.synergy_detector = SynergyDetector(self.master_atomspace)
        
        # Coordination state
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.bot_capabilities: Dict[str, Set[str]] = {}
        self.bot_workloads: Dict[str, float] = {}
        self.coordination_history = deque(maxlen=1000)
        
        # Coordination parameters
        self.coordination_interval = 5.0  # seconds
        self.synergy_detection_interval = 10.0  # seconds
        self.max_bots_per_task = 5
        
        # System state
        self.running = False
        self.coordination_metrics = {
            "tasks_completed": 0,
            "synergies_detected": 0,
            "coordination_efficiency": 0.5,
            "collective_performance": 0.5
        }
        
    async def start(self):
        """Start the multi-bot coordination system"""
        self.running = True
        
        # Start coordination processes
        asyncio.create_task(self._coordination_loop())
        asyncio.create_task(self._synergy_detection_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        
        print(f"Multi-bot coordinator {self.coordinator_id} started")
    
    async def stop(self):
        """Stop the coordination system"""
        self.running = False
        print(f"Multi-bot coordinator {self.coordinator_id} stopped")
    
    async def register_bot(self, bot: CognitiveBot) -> bool:
        """Register a cognitive bot with the coordinator"""
        
        try:
            bot_id = bot.bot_id
            self.registered_bots[bot_id] = bot
            
            # Initialize bot state tracking
            self.bot_workloads[bot_id] = 0.0
            
            # Assess bot capabilities
            capabilities = await self._assess_bot_capabilities(bot)
            self.bot_capabilities[bot_id] = capabilities
            
            # Register bot with synergy detector
            await bot.register_with_synergy_detector(self.synergy_detector)
            
            # Add bot to master atomspace
            bot_atom = self.master_atomspace.add_node(AtomType.CONCEPT, f"Bot_{bot_id}")
            coordinator_atom = self.master_atomspace.add_node(AtomType.CONCEPT, f"Coordinator_{self.coordinator_id}")
            
            # Create coordination relationship
            coordination_link = self.master_atomspace.add_link(
                AtomType.EVALUATION,
                [
                    self.master_atomspace.add_node(AtomType.PREDICATE, "coordinates"),
                    self.master_atomspace.add_link(AtomType.LIST, [coordinator_atom, bot_atom])
                ],
                TruthValue(0.8, 0.9)
            )
            
            print(f"Bot {bot_id} registered with coordinator. Capabilities: {capabilities}")
            return True
            
        except Exception as e:
            print(f"Error registering bot {bot.bot_id}: {e}")
            return False
    
    async def unregister_bot(self, bot_id: str) -> bool:
        """Unregister a bot from the coordinator"""
        
        if bot_id not in self.registered_bots:
            return False
        
        try:
            # Remove from active tasks
            for task in self.active_tasks.values():
                if bot_id in task.assigned_bots:
                    task.assigned_bots.remove(bot_id)
            
            # Clean up state
            del self.registered_bots[bot_id]
            del self.bot_capabilities[bot_id]
            del self.bot_workloads[bot_id]
            
            print(f"Bot {bot_id} unregistered from coordinator")
            return True
            
        except Exception as e:
            print(f"Error unregistering bot {bot_id}: {e}")
            return False
    
    async def _assess_bot_capabilities(self, bot: CognitiveBot) -> Set[str]:
        """Assess the capabilities of a cognitive bot"""
        
        capabilities = set()
        
        # Basic capabilities for all cognitive bots
        capabilities.update(["conversation", "reasoning", "learning"])
        
        # Check for specialization
        if hasattr(bot, 'specialization'):
            capabilities.add(bot.specialization)
        
        # Assess from bot's cognitive state
        cognitive_state = bot.get_cognitive_state()
        
        # Infer capabilities from goals
        for goal_name in cognitive_state.get("active_goals", []):
            if "help" in goal_name.lower():
                capabilities.add("assistance")
            if "learn" in goal_name.lower():
                capabilities.add("adaptive_learning")
            if "collab" in goal_name.lower():
                capabilities.add("collaboration")
        
        # Infer from atomspace content
        atomspace_size = cognitive_state.get("atomspace_size", 0)
        if atomspace_size > 100:
            capabilities.add("knowledge_processing")
        
        # Check energy and confidence levels
        energy = cognitive_state.get("energy_level", 0.5)
        confidence = cognitive_state.get("confidence_level", 0.5)
        
        if energy > 0.7 and confidence > 0.7:
            capabilities.add("high_performance")
        elif energy > 0.8:
            capabilities.add("high_energy")
        elif confidence > 0.8:
            capabilities.add("high_confidence")
        
        return capabilities
    
    async def add_coordination_task(self, task: CoordinationTask) -> bool:
        """Add a new coordination task"""
        
        try:
            self.active_tasks[task.task_id] = task
            
            # Assign bots to task
            await self._assign_bots_to_task(task)
            
            # Add task to master atomspace
            task_atom = self.master_atomspace.add_node(AtomType.CONCEPT, f"Task_{task.task_id}")
            
            # Link task to required capabilities
            for capability in task.required_capabilities:
                capability_atom = self.master_atomspace.add_node(AtomType.CONCEPT, f"Capability_{capability}")
                self.master_atomspace.add_link(
                    AtomType.EVALUATION,
                    [
                        self.master_atomspace.add_node(AtomType.PREDICATE, "requires"),
                        self.master_atomspace.add_link(AtomType.LIST, [task_atom, capability_atom])
                    ],
                    TruthValue(0.9, 0.8)
                )
            
            print(f"Coordination task {task.task_id} added: {task.description}")
            return True
            
        except Exception as e:
            print(f"Error adding coordination task: {e}")
            return False
    
    async def _assign_bots_to_task(self, task: CoordinationTask):
        """Assign appropriate bots to a coordination task"""
        
        # Find bots with required capabilities
        candidate_bots = []
        
        for bot_id, capabilities in self.bot_capabilities.items():
            # Check capability match
            capability_match = len(set(task.required_capabilities) & capabilities) / len(task.required_capabilities)
            
            if capability_match > 0.3:  # At least 30% capability match
                # Consider workload
                workload = self.bot_workloads.get(bot_id, 0.0)
                availability = 1.0 - workload
                
                # Calculate assignment score
                assignment_score = capability_match * 0.7 + availability * 0.3
                
                candidate_bots.append((bot_id, assignment_score, capability_match))
        
        # Sort by assignment score
        candidate_bots.sort(key=lambda x: x[1], reverse=True)
        
        # Assign top candidates
        max_assignments = min(len(candidate_bots), self.max_bots_per_task)
        
        for i in range(max_assignments):
            bot_id, score, capability_match = candidate_bots[i]
            
            if score > 0.4:  # Minimum assignment threshold
                task.assigned_bots.append(bot_id)
                
                # Update workload
                workload_increase = task.priority * 0.2
                self.bot_workloads[bot_id] = min(1.0, self.bot_workloads[bot_id] + workload_increase)
                
                print(f"Assigned bot {bot_id} to task {task.task_id} (score: {score:.3f})")
        
        if not task.assigned_bots:
            print(f"Warning: No bots assigned to task {task.task_id}")
    
    async def _coordination_loop(self):
        """Main coordination processing loop"""
        
        while self.running:
            try:
                await self._execute_coordination_cycle()
                await asyncio.sleep(self.coordination_interval)
                
            except Exception as e:
                print(f"Coordination loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_coordination_cycle(self):
        """Execute one coordination cycle"""
        
        # Update bot states
        await self._update_bot_states()
        
        # Process active tasks
        await self._process_active_tasks()
        
        # Rebalance workloads if needed
        await self._rebalance_workloads()
        
        # Update coordination metrics
        self._update_coordination_metrics()
    
    async def _update_bot_states(self):
        """Update tracking of all bot states"""
        
        for bot_id, bot in self.registered_bots.items():
            try:
                # Get current cognitive state
                cognitive_state = bot.get_cognitive_state()
                
                # Update capabilities assessment
                new_capabilities = await self._assess_bot_capabilities(bot)
                
                if new_capabilities != self.bot_capabilities.get(bot_id, set()):
                    self.bot_capabilities[bot_id] = new_capabilities
                    print(f"Bot {bot_id} capabilities updated: {new_capabilities}")
                
                # Update synergy detector registration
                await self.synergy_detector.register_agent(bot_id, bot.cognitive_engine.current_state)
                
            except Exception as e:
                print(f"Error updating bot {bot_id} state: {e}")
    
    async def _process_active_tasks(self):
        """Process and update active coordination tasks"""
        
        completed_tasks = []
        
        for task_id, task in self.active_tasks.items():
            try:
                # Update task progress (simplified simulation)
                if task.assigned_bots:
                    # Calculate progress based on bot capabilities and workloads
                    progress_increment = 0.0
                    
                    for bot_id in task.assigned_bots:
                        if bot_id in self.registered_bots:
                            bot_capabilities = self.bot_capabilities.get(bot_id, set())
                            capability_match = len(set(task.required_capabilities) & bot_capabilities) / max(len(task.required_capabilities), 1)
                            
                            bot_workload = self.bot_workloads.get(bot_id, 1.0)
                            bot_availability = 1.0 - bot_workload
                            
                            bot_contribution = capability_match * bot_availability * task.priority * 0.1
                            progress_increment += bot_contribution
                    
                    task.progress = min(1.0, task.progress + progress_increment)
                    
                    # Update task status
                    if task.progress >= 1.0:
                        task.status = "completed"
                        completed_tasks.append(task_id)
                    elif task.progress > 0.1:
                        task.status = "in_progress"
                
                # Check for deadline
                if task.deadline:
                    import time
                    if time.time() > task.deadline and task.status != "completed":
                        task.status = "overdue"
                
            except Exception as e:
                print(f"Error processing task {task_id}: {e}")
        
        # Remove completed tasks and update workloads
        for task_id in completed_tasks:
            task = self.active_tasks[task_id]
            
            # Reduce workloads for assigned bots
            for bot_id in task.assigned_bots:
                if bot_id in self.bot_workloads:
                    workload_reduction = task.priority * 0.2
                    self.bot_workloads[bot_id] = max(0.0, self.bot_workloads[bot_id] - workload_reduction)
            
            del self.active_tasks[task_id]
            self.coordination_metrics["tasks_completed"] += 1
            print(f"Task {task_id} completed")
    
    async def _rebalance_workloads(self):
        """Rebalance workloads across bots if needed"""
        
        if not self.bot_workloads:
            return
        
        # Calculate workload statistics
        workloads = list(self.bot_workloads.values())
        avg_workload = sum(workloads) / len(workloads)
        workload_std = np.std(workloads) if len(workloads) > 1 else 0.0
        
        # If workload distribution is very uneven, try to rebalance
        if workload_std > 0.3:
            # Find overloaded and underloaded bots
            overloaded_bots = [(bot_id, workload) for bot_id, workload in self.bot_workloads.items() 
                             if workload > avg_workload + 0.2]
            underloaded_bots = [(bot_id, workload) for bot_id, workload in self.bot_workloads.items() 
                              if workload < avg_workload - 0.2]
            
            if overloaded_bots and underloaded_bots:
                # Try to reassign tasks
                await self._reassign_tasks_for_balance(overloaded_bots, underloaded_bots)
    
    async def _reassign_tasks_for_balance(self, overloaded_bots: List[tuple], underloaded_bots: List[tuple]):
        """Reassign tasks to balance workloads"""
        
        # Find tasks that could be reassigned
        reassignable_tasks = []
        
        for task_id, task in self.active_tasks.items():
            if len(task.assigned_bots) > 1 and task.status == "in_progress":
                reassignable_tasks.append((task_id, task))
        
        # Try reassignments
        for task_id, task in reassignable_tasks:
            for overloaded_bot_id, _ in overloaded_bots:
                if overloaded_bot_id in task.assigned_bots:
                    # Try to find an underloaded bot that can handle this task
                    for underloaded_bot_id, _ in underloaded_bots:
                        if underloaded_bot_id not in task.assigned_bots:
                            underloaded_capabilities = self.bot_capabilities.get(underloaded_bot_id, set())
                            capability_match = len(set(task.required_capabilities) & underloaded_capabilities) / len(task.required_capabilities)
                            
                            if capability_match > 0.5:  # Good capability match
                                # Perform reassignment
                                task.assigned_bots.remove(overloaded_bot_id)
                                task.assigned_bots.append(underloaded_bot_id)
                                
                                # Update workloads
                                workload_transfer = task.priority * 0.2
                                self.bot_workloads[overloaded_bot_id] -= workload_transfer
                                self.bot_workloads[underloaded_bot_id] += workload_transfer
                                
                                print(f"Reassigned task {task_id} from {overloaded_bot_id} to {underloaded_bot_id}")
                                break
    
    async def _synergy_detection_loop(self):
        """Loop for detecting and leveraging synergies"""
        
        while self.running:
            try:
                await self._detect_and_leverage_synergies()
                await asyncio.sleep(self.synergy_detection_interval)
                
            except Exception as e:
                print(f"Synergy detection loop error: {e}")
                await asyncio.sleep(2.0)
    
    async def _detect_and_leverage_synergies(self):
        """Detect synergies and create coordination opportunities"""
        
        # Detect synergies between registered bots
        synergies = await self.synergy_detector.detect_synergies()
        
        if synergies:
            self.coordination_metrics["synergies_detected"] += len(synergies)
            
            # Process each synergy
            for synergy in synergies:
                await self._leverage_synergy(synergy)
    
    async def _leverage_synergy(self, synergy: SynergyPattern):
        """Leverage a detected synergy for improved coordination"""
        
        synergy_actions = self.synergy_detector._generate_synergy_actions(synergy)
        
        # Create coordination tasks based on synergy
        if synergy.synergy_type in [SynergyType.COMPLEMENTARY, SynergyType.COLLABORATIVE]:
            # Create collaborative task
            task = CoordinationTask(
                task_id=f"synergy_task_{len(self.active_tasks)}",
                description=f"Leverage {synergy.synergy_type.value} synergy between {', '.join(synergy.agents)}",
                required_capabilities=list(synergy.benefits.keys()),
                coordination_strategy=CoordinationStrategy.COLLABORATIVE,
                priority=synergy.strength,
                assigned_bots=synergy.agents.copy()
            )
            
            await self.add_coordination_task(task)
        
        elif synergy.synergy_type == SynergyType.EMERGENT:
            # Create emergent coordination task
            task = CoordinationTask(
                task_id=f"emergent_task_{len(self.active_tasks)}",
                description=f"Facilitate emergent intelligence from group of {len(synergy.agents)} agents",
                required_capabilities=["collective_intelligence", "emergence_facilitation"],
                coordination_strategy=CoordinationStrategy.EMERGENT,
                priority=synergy.strength * 1.2,  # Boost priority for emergent synergies
                assigned_bots=synergy.agents.copy()
            )
            
            await self.add_coordination_task(task)
        
        print(f"Leveraging {synergy.synergy_type.value} synergy between {synergy.agents} (strength: {synergy.strength:.3f})")
    
    async def _performance_monitoring_loop(self):
        """Monitor overall coordination system performance"""
        
        while self.running:
            try:
                await self._monitor_performance()
                await asyncio.sleep(15.0)  # Monitor every 15 seconds
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                await asyncio.sleep(3.0)
    
    async def _monitor_performance(self):
        """Monitor and log coordination system performance"""
        
        # Calculate collective performance
        if self.registered_bots:
            bot_performances = []
            
            for bot_id, bot in self.registered_bots.items():
                try:
                    cognitive_state = bot.get_cognitive_state()
                    bot_performance = (cognitive_state.get("confidence_level", 0.5) + 
                                     cognitive_state.get("energy_level", 0.5)) / 2
                    bot_performances.append(bot_performance)
                    
                except Exception:
                    bot_performances.append(0.5)  # Default if can't assess
            
            self.coordination_metrics["collective_performance"] = sum(bot_performances) / len(bot_performances)
        
        # Calculate coordination efficiency
        if self.active_tasks:
            completed_ratio = self.coordination_metrics["tasks_completed"] / (
                self.coordination_metrics["tasks_completed"] + len(self.active_tasks)
            )
            
            avg_progress = sum(task.progress for task in self.active_tasks.values()) / len(self.active_tasks)
            
            self.coordination_metrics["coordination_efficiency"] = (completed_ratio + avg_progress) / 2
        
        # Log performance summary periodically
        if hasattr(self, '_last_performance_log'):
            import time
            if time.time() - self._last_performance_log > 60:  # Log every minute
                await self._log_performance_summary()
                self._last_performance_log = time.time()
        else:
            import time
            self._last_performance_log = time.time()
    
    def _update_coordination_metrics(self):
        """Update coordination system metrics"""
        
        # Calculate workload balance
        if self.bot_workloads:
            workloads = list(self.bot_workloads.values())
            workload_balance = 1.0 - np.std(workloads) if len(workloads) > 1 else 1.0
            self.coordination_metrics["workload_balance"] = max(0.0, workload_balance)
    
    async def _log_performance_summary(self):
        """Log a comprehensive performance summary"""
        
        summary = {
            "coordinator_id": self.coordinator_id,
            "registered_bots": len(self.registered_bots),
            "active_tasks": len(self.active_tasks),
            "metrics": self.coordination_metrics.copy(),
            "bot_workloads": dict(self.bot_workloads),
            "synergies_active": len(self.synergy_detector.synergy_patterns)
        }
        
        print(f"Coordination Performance Summary: {json.dumps(summary, indent=2)}")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination system status"""
        
        return {
            "coordinator_id": self.coordinator_id,
            "registered_bots": list(self.registered_bots.keys()),
            "bot_capabilities": {bot_id: list(caps) for bot_id, caps in self.bot_capabilities.items()},
            "active_tasks": len(self.active_tasks),
            "task_details": {task_id: {
                "description": task.description,
                "progress": task.progress,
                "status": task.status,
                "assigned_bots": task.assigned_bots
            } for task_id, task in self.active_tasks.items()},
            "coordination_metrics": self.coordination_metrics.copy(),
            "bot_workloads": dict(self.bot_workloads),
            "detected_synergies": len(self.synergy_detector.synergy_patterns),
            "running": self.running
        }