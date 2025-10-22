"""
OpenCog AtomSpace Implementation for Cognitive Synergy Architecture

Provides knowledge representation, reasoning, and memory management
for autonomous bot orchestration and emergent cognitive behaviors.
"""

import asyncio
import json
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import numpy as np


class AtomType(Enum):
    """Fundamental atom types in the cognitive architecture"""
    CONCEPT = "ConceptNode"
    PREDICATE = "PredicateNode"  
    PROCEDURE = "ProcedureNode"
    EVALUATION = "EvaluationLink"
    INHERITANCE = "InheritanceLink"
    SIMILARITY = "SimilarityLink"
    IMPLICATION = "ImplicationLink"
    AND = "AndLink"
    OR = "OrLink"
    NOT = "NotLink"
    EXECUTION = "ExecutionLink"
    LIST = "ListLink"
    VARIABLE = "VariableNode"
    CONTEXT = "ContextLink"


@dataclass
class TruthValue:
    """Truth value with strength and confidence for probabilistic reasoning"""
    strength: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))


class Atom:
    """Base atom class for all knowledge representations"""
    
    def __init__(self, atom_type: AtomType, name: str, truth_value: Optional[TruthValue] = None):
        self.atom_type = atom_type
        self.name = name
        self.truth_value = truth_value or TruthValue()
        self.incoming = set()  # Links pointing to this atom
        self.outgoing = []     # Atoms this links to
        self.attention_value = 0.5
        self.importance = 0.5
        
    def __hash__(self):
        return hash((self.atom_type, self.name))
    
    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        return self.atom_type == other.atom_type and self.name == other.name
    
    def __repr__(self):
        return f"{self.atom_type.value}({self.name})[{self.truth_value.strength:.2f},{self.truth_value.confidence:.2f}]"


class Link(Atom):
    """Link atom that connects other atoms"""
    
    def __init__(self, atom_type: AtomType, outgoing: List[Atom], name: str = "", truth_value: Optional[TruthValue] = None):
        super().__init__(atom_type, name, truth_value)
        self.outgoing = outgoing
        
        # Update incoming sets of target atoms
        for atom in outgoing:
            atom.incoming.add(self)
    
    def __hash__(self):
        return hash((self.atom_type, tuple(self.outgoing)))
    
    def __eq__(self, other):
        if not isinstance(other, Link):
            return False
        return self.atom_type == other.atom_type and self.outgoing == other.outgoing


class AtomSpace:
    """Central knowledge base and reasoning engine for cognitive architecture"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.atoms: Dict[Tuple, Atom] = {}
        self.type_index: Dict[AtomType, Set[Atom]] = defaultdict(set)
        self.name_index: Dict[str, Set[Atom]] = defaultdict(set)
        self.graph = nx.DiGraph()
        self.attention_bank = AttentionBank()
        self.pattern_matcher = PatternMatcher(self)
        
    def add_node(self, atom_type: AtomType, name: str, truth_value: Optional[TruthValue] = None) -> Atom:
        """Add a node atom to the atomspace"""
        key = (atom_type, name)
        if key in self.atoms:
            return self.atoms[key]
            
        atom = Atom(atom_type, name, truth_value)
        self.atoms[key] = atom
        self.type_index[atom_type].add(atom)
        self.name_index[name].add(atom)
        self.graph.add_node(atom)
        
        return atom
    
    def add_link(self, atom_type: AtomType, outgoing: List[Atom], truth_value: Optional[TruthValue] = None) -> Link:
        """Add a link atom connecting other atoms"""
        key = (atom_type, tuple(outgoing))
        if key in self.atoms:
            return self.atoms[key]
            
        link = Link(atom_type, outgoing, "", truth_value)
        self.atoms[key] = link
        self.type_index[atom_type].add(link)
        
        # Add edges to graph
        for i, source in enumerate(outgoing):
            for target in outgoing[i+1:]:
                self.graph.add_edge(source, target, link=link)
                
        return link
    
    def get_by_name(self, name: str) -> Set[Atom]:
        """Get all atoms with the given name"""
        return self.name_index.get(name, set())
    
    def get_by_type(self, atom_type: AtomType) -> Set[Atom]:
        """Get all atoms of the given type"""
        return self.type_index.get(atom_type, set())
    
    def find_similar_concepts(self, concept: Atom, threshold: float = 0.7) -> List[Tuple[Atom, float]]:
        """Find concepts similar to the given concept using graph structure"""
        if concept not in self.graph:
            return []
            
        similar = []
        for atom in self.get_by_type(AtomType.CONCEPT):
            if atom == concept:
                continue
                
            # Calculate similarity based on shared connections
            concept_neighbors = set(self.graph.neighbors(concept))
            atom_neighbors = set(self.graph.neighbors(atom))
            
            if concept_neighbors or atom_neighbors:
                jaccard_similarity = len(concept_neighbors & atom_neighbors) / len(concept_neighbors | atom_neighbors)
                if jaccard_similarity >= threshold:
                    similar.append((atom, jaccard_similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def propagate_activation(self, source_atoms: List[Atom], decay_rate: float = 0.9) -> Dict[Atom, float]:
        """Spread activation through the knowledge network"""
        activation = {atom: 1.0 for atom in source_atoms}
        
        for _ in range(3):  # 3 propagation steps
            new_activation = activation.copy()
            
            for atom, strength in activation.items():
                if strength < 0.1:  # Skip weak activations
                    continue
                    
                # Propagate to connected atoms
                for connected in self.graph.neighbors(atom):
                    current = new_activation.get(connected, 0.0)
                    new_activation[connected] = max(current, strength * decay_rate)
            
            activation = new_activation
        
        return {atom: strength for atom, strength in activation.items() if strength > 0.1}


class AttentionBank:
    """Manages attention allocation and focus of the cognitive system"""
    
    def __init__(self):
        self.focus_atoms: Set[Atom] = set()
        self.attention_values: Dict[Atom, float] = {}
        
    def focus_on(self, atoms: List[Atom], strength: float = 1.0):
        """Focus attention on specific atoms"""
        for atom in atoms:
            self.focus_atoms.add(atom)
            self.attention_values[atom] = strength
            atom.attention_value = strength
    
    def decay_attention(self, decay_rate: float = 0.95):
        """Gradually decay attention values over time"""
        to_remove = []
        for atom in self.focus_atoms:
            self.attention_values[atom] *= decay_rate
            atom.attention_value = self.attention_values[atom]
            
            if self.attention_values[atom] < 0.1:
                to_remove.append(atom)
        
        for atom in to_remove:
            self.focus_atoms.discard(atom)
            del self.attention_values[atom]
    
    def get_focused_atoms(self, threshold: float = 0.5) -> List[Atom]:
        """Get atoms currently in focus above threshold"""
        return [atom for atom in self.focus_atoms 
                if self.attention_values.get(atom, 0) >= threshold]


class PatternMatcher:
    """Pattern matching and query engine for the atomspace"""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
    
    def match_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, Atom]]:
        """Match a pattern against the atomspace and return bindings"""
        # Simplified pattern matching - in a full implementation this would be much more sophisticated
        results = []
        
        if pattern.get("type") == "concept":
            name_pattern = pattern.get("name", "*")
            if name_pattern == "*":
                concepts = self.atomspace.get_by_type(AtomType.CONCEPT)
                for concept in concepts:
                    results.append({"match": concept})
            else:
                matches = self.atomspace.get_by_name(name_pattern)
                concepts = [atom for atom in matches if atom.atom_type == AtomType.CONCEPT]
                for concept in concepts:
                    results.append({"match": concept})
        
        return results
    
    async def query_knowledge(self, query: str) -> List[Atom]:
        """Query the knowledge base using natural language"""
        # This would integrate with NLP processing in a full implementation
        # For now, simple keyword matching
        results = []
        
        for name, atoms in self.atomspace.name_index.items():
            if query.lower() in name.lower():
                results.extend(atoms)
        
        return results[:10]  # Limit results


# Utility functions for creating common knowledge patterns

def create_concept_hierarchy(atomspace: AtomSpace, parent: str, children: List[str]) -> List[Link]:
    """Create inheritance relationships between concepts"""
    parent_node = atomspace.add_node(AtomType.CONCEPT, parent)
    inheritance_links = []
    
    for child_name in children:
        child_node = atomspace.add_node(AtomType.CONCEPT, child_name)
        inheritance_link = atomspace.add_link(
            AtomType.INHERITANCE, 
            [child_node, parent_node],
            TruthValue(0.8, 0.9)
        )
        inheritance_links.append(inheritance_link)
    
    return inheritance_links


def create_evaluation(atomspace: AtomSpace, predicate: str, subject: str, object_name: str = None) -> Link:
    """Create an evaluation link representing a fact"""
    pred_node = atomspace.add_node(AtomType.PREDICATE, predicate)
    subj_node = atomspace.add_node(AtomType.CONCEPT, subject)
    
    if object_name:
        obj_node = atomspace.add_node(AtomType.CONCEPT, object_name)
        list_link = atomspace.add_link(AtomType.LIST, [subj_node, obj_node])
    else:
        list_link = atomspace.add_link(AtomType.LIST, [subj_node])
    
    return atomspace.add_link(
        AtomType.EVALUATION,
        [pred_node, list_link],
        TruthValue(0.9, 0.8)
    )


def bootstrap_bot_knowledge(atomspace: AtomSpace):
    """Bootstrap basic bot-related knowledge into the atomspace"""
    # Create basic bot concepts
    create_concept_hierarchy(atomspace, "Entity", ["Bot", "User", "Message", "Action"])
    create_concept_hierarchy(atomspace, "Bot", ["EchoBot", "ChatBot", "SkillBot", "CognitiveBot"])
    create_concept_hierarchy(atomspace, "Action", ["Respond", "Listen", "Reason", "Learn"])
    
    # Create basic relationships
    create_evaluation(atomspace, "can_perform", "Bot", "Action")
    create_evaluation(atomspace, "interacts_with", "Bot", "User")
    create_evaluation(atomspace, "processes", "Bot", "Message")
    
    return atomspace