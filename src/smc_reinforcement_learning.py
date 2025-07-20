"""
Sequential Monte Carlo Reinforcement Learning Implementation
"""
import random
import math
import threading
import time
from typing import Dict, List, Tuple, Optional
import pickle
import logging
import numpy as np
import os

class Particle:
    """Individual particle in SMC filter"""
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weight = 1.0
        self.fitness = 0.0
        self.age = 0
        # Initialize policy with smaller values for stability
        self.policy = [[random.gauss(0, 0.01) for _ in range(action_dim)] 
                      for _ in range(state_dim)]
        self.value_function = [0.0] * state_dim
        
    def predict_action(self, state: List[float]) -> Tuple[int, float]:
        """Predict action and return confidence"""
        action_values = []
        for j in range(self.action_dim):
            value = 0
            for i in range(len(state)):
                if i < len(self.policy):
                    value += state[i] * self.policy[i][j]
            action_values.append(value)
        
        # Handle case with no valid action values
        if not action_values:
            return 0, 0.0
            
        # Find argmax and max value
        max_idx = 0
        max_val = action_values[0]
        for i, val in enumerate(action_values):
            if val > max_val:
                max_val = val
                max_idx = i
        
        # Numerical stabilization for softmax
        max_val = max(action_values)
        exp_values = [math.exp(v - max_val) for v in action_values]
        exp_sum = sum(exp_values)
        confidence = exp_values[max_idx] / exp_sum if exp_sum > 0 else 1.0/len(action_values)
        
        return max_idx, confidence
    
    def update_policy(self, gradient: List[List[float]], learning_rate: float = 0.01):
        """Update policy using gradient"""
        for i in range(len(self.policy)):
            for j in range(len(self.policy[i])):
                if i < len(gradient) and j < len(gradient[i]):
                    self.policy[i][j] += learning_rate * gradient[i][j]
    
    def mutate(self, mutation_rate: float = 0.1):
        """Apply mutation to particle"""
        for i in range(len(self.policy)):
            for j in range(len(self.policy[i])):
                if random.random() < mutation_rate:
                    self.policy[i][j] += random.gauss(0, mutation_rate)
    
    def copy(self):
        """Create a copy of this particle"""
        new_particle = Particle(self.state_dim, self.action_dim)
        new_particle.policy = [row[:] for row in self.policy]
        new_particle.weight = self.weight
        new_particle.fitness = self.fitness
        new_particle.age = self.age
        new_particle.value_function = self.value_function[:]
        return new_particle

class SMCReinforcementLearning:
    """Sequential Monte Carlo Reinforcement Learning Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, num_particles: int = 100,
                 load_path: str = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_particles = num_particles
        
        if load_path and os.path.exists(load_path):
            self.load_model(load_path)
        else:
            # Create initial particle population
            self.particles = []
            for _ in range(self.num_particles):
                particle = Particle(state_dim, action_dim)
                self.particles.append(particle)
            
            self.best_particle = self.particles[0].copy() if self.particles else None
        
        self.learning_history = []
        self.performance_threshold = 0.1
        self.lock = threading.Lock()
        
    def select_action(self, state: List[float]) -> Tuple[int, Dict]:
        """Select action using weighted ensemble of particles"""
        if not self.particles:
            return 0, {'confidence': 0.0}
        
        # Ensure state is always a list of floats
        if isinstance(state, (np.ndarray, tuple)):
            state = state.tolist() if isinstance(state, np.ndarray) else list(state)
        state = [float(x) for x in state]
        
        weights = [p.weight for p in self.particles]
        weight_sum = sum(weights)
        if weight_sum == 0:
            weights = [1.0] * len(weights)
            weight_sum = len(weights)
        
        # Normalize weights
        weights = [w / weight_sum for w in weights]
        
        # Calculate weighted average action
        action_probs = [0.0] * self.action_dim
        total_confidence = 0.0
        weighted_confidence = 0.0
        
        for i, particle in enumerate(self.particles):
            action_idx, confidence = particle.predict_action(state)
            if action_idx < len(action_probs):
                action_probs[action_idx] += weights[i]
                total_confidence += confidence
                weighted_confidence += confidence * weights[i]
        
        # Find argmax and confidence
        max_idx = 0
        max_val = action_probs[0]
        for i, val in enumerate(action_probs):
            if val > max_val:
                max_val = val
                max_idx = i
        
        avg_confidence = total_confidence / len(self.particles) if self.particles else 0.0
        return max_idx, {
            'confidence': max_val,
            'particle_confidence': avg_confidence,
            'weighted_confidence': weighted_confidence
        }
    
    def update(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool):
        """Wrapper for update_particles to match expected interface"""
        self.update_particles(state, action, reward, next_state)
    
    def update_particles(self, state: List[float], action: int, reward: float, next_state: List[float]):
        """Update particle weights and policies based on experience"""
        # Convert states to lists
        state = state.tolist() if isinstance(state, np.ndarray) else list(state)
        next_state = next_state.tolist() if isinstance(next_state, np.ndarray) else list(next_state)

        with self.lock:
            for particle in self.particles:
                # Get both action index and confidence
                predicted_action, confidence = particle.predict_action(state)
                
                # Calculate prediction error
                prediction_error = 1.0 if predicted_action == action else 0.0
                
                # Update particle weight
                particle.weight *= (1.0 + reward * prediction_error * confidence)
                particle.fitness = particle.weight
                
                # Update policy
                if prediction_error > 0:
                    gradient = []
                    for i in range(len(state)):
                        row = [0.0] * self.action_dim
                        if predicted_action < len(row):
                            row[predicted_action] = state[i] * reward * confidence
                        gradient.append(row)
                    particle.update_policy(gradient)
            
            # Update best particle
            current_best = max(self.particles, key=lambda p: p.fitness, default=None)
            if current_best and (not self.best_particle or current_best.fitness > self.best_particle.fitness):
                self.best_particle = current_best.copy()
    
    def resample_particles(self):
        """Resample particles based on weights"""
        weights = [p.weight for p in self.particles]
        weight_sum = sum(weights)
        if weight_sum == 0:
            return
        
        # Normalize weights
        weights = [w / weight_sum for w in weights]
        
        # Calculate effective sample size
        ess = 1.0 / sum(w ** 2 for w in weights)
        
        if ess < self.num_particles / 2:
            # Resample particles
            new_particles = []
            
            # Simple resampling
            for _ in range(self.num_particles):
                # Weighted random selection
                r = random.random()
                cumsum = 0
                selected_idx = 0
                for i, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        selected_idx = i
                        break
                
                # Copy selected particle
                original = self.particles[selected_idx]
                new_particle = original.copy()
                new_particle.mutate()
                new_particles.append(new_particle)
            
            self.particles = new_particles
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        if not self.particles:
            return {'avg_weight': 0, 'max_fitness': 0, 'diversity': 0}
        
        weights = [p.weight for p in self.particles]
        fitness_scores = [p.fitness for p in self.particles]
        
        return {
            'avg_weight': sum(weights) / len(weights),
            'max_fitness': max(fitness_scores),
            'diversity': math.sqrt(sum((w - sum(weights)/len(weights))**2 for w in weights) / len(weights))
        }
    
    def save_model(self, filepath: str):
        """Save the current model state to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'particles': self.particles,
                'best_particle': self.best_particle,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'num_particles': self.num_particles
            }, f)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.particles = data['particles']
            self.best_particle = data['best_particle']
            self.state_dim = data['state_dim']
            self.action_dim = data['action_dim']
            self.num_particles = data['num_particles']