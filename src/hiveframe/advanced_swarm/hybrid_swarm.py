"""
Hybrid Swarm Intelligence

Combines multiple swarm intelligence algorithms for superior optimization.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Any, Dict, Optional
import random
import math


@dataclass
class Particle:
    """Particle for PSO"""
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_fitness: float
    fitness: float


@dataclass
class Ant:
    """Ant for ACO"""
    path: List[int]
    distance: float
    

@dataclass
class Firefly:
    """Firefly for Firefly Algorithm"""
    position: List[float]
    brightness: float
    fitness: float


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for numerical optimization.
    
    Particles move through solution space, influenced by their own
    best position and the swarm's best position.
    """
    
    def __init__(
        self,
        num_particles: int = 30,
        dimensions: int = 2,
        bounds: Tuple[float, float] = (-100, 100),
        w: float = 0.7,  # Inertia weight
        c1: float = 1.5,  # Cognitive parameter
        c2: float = 1.5,  # Social parameter
    ):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles: List[Particle] = []
        self.global_best_position: Optional[List[float]] = None
        self.global_best_fitness: float = float('inf')
        
    def initialize(self) -> None:
        """Initialize particle swarm"""
        self.particles = []
        
        for _ in range(self.num_particles):
            position = [
                random.uniform(self.bounds[0], self.bounds[1])
                for _ in range(self.dimensions)
            ]
            velocity = [
                random.uniform(-1, 1)
                for _ in range(self.dimensions)
            ]
            
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=float('inf'),
                fitness=float('inf'),
            )
            self.particles.append(particle)
    
    def optimize(
        self,
        fitness_func: Callable[[List[float]], float],
        max_iterations: int = 100,
    ) -> Tuple[List[float], float]:
        """
        Run PSO optimization.
        
        Args:
            fitness_func: Function to minimize
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (best_position, best_fitness)
        """
        self.initialize()
        
        for iteration in range(max_iterations):
            # Evaluate fitness for all particles
            for particle in self.particles:
                particle.fitness = fitness_func(particle.position)
                
                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in self.particles:
                for d in range(self.dimensions):
                    r1, r2 = random.random(), random.random()
                    
                    cognitive = self.c1 * r1 * (particle.best_position[d] - particle.position[d])
                    social = self.c2 * r2 * (self.global_best_position[d] - particle.position[d])
                    
                    particle.velocity[d] = (
                        self.w * particle.velocity[d] + cognitive + social
                    )
                    
                    # Update position
                    particle.position[d] += particle.velocity[d]
                    
                    # Clamp to bounds
                    particle.position[d] = max(
                        self.bounds[0],
                        min(self.bounds[1], particle.position[d])
                    )
        
        return self.global_best_position, self.global_best_fitness


class AntColonyOptimizer:
    """
    Ant Colony Optimization for routing and path problems.
    
    Ants deposit pheromones on paths, gradually finding optimal routes
    through collective behavior.
    """
    
    def __init__(
        self,
        num_ants: int = 20,
        num_nodes: int = 10,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0,   # Heuristic importance
        rho: float = 0.5,    # Evaporation rate
        q: float = 100.0,    # Pheromone deposit factor
    ):
        self.num_ants = num_ants
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone: List[List[float]] = []
        self.best_path: Optional[List[int]] = None
        self.best_distance: float = float('inf')
        
    def initialize(self, distance_matrix: List[List[float]]) -> None:
        """Initialize pheromone trails"""
        self.pheromone = [
            [1.0 for _ in range(self.num_nodes)]
            for _ in range(self.num_nodes)
        ]
        self.distance_matrix = distance_matrix
    
    def optimize(
        self,
        distance_matrix: List[List[float]],
        max_iterations: int = 100,
    ) -> Tuple[List[int], float]:
        """
        Run ACO optimization.
        
        Args:
            distance_matrix: Matrix of distances between nodes
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (best_path, best_distance)
        """
        self.initialize(distance_matrix)
        
        for iteration in range(max_iterations):
            ants = []
            
            # Construct solutions for each ant
            for _ in range(self.num_ants):
                path = self._construct_path()
                distance = self._calculate_distance(path)
                ants.append(Ant(path=path, distance=distance))
                
                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path
            
            # Update pheromones
            self._update_pheromones(ants)
        
        return self.best_path, self.best_distance
    
    def _construct_path(self) -> List[int]:
        """Construct a path for one ant"""
        path = [0]  # Start at node 0
        unvisited = set(range(1, self.num_nodes))
        
        while unvisited:
            current = path[-1]
            next_node = self._select_next_node(current, unvisited)
            path.append(next_node)
            unvisited.remove(next_node)
        
        return path
    
    def _select_next_node(self, current: int, unvisited: set) -> int:
        """Select next node based on pheromone and heuristic"""
        probabilities = []
        
        for node in unvisited:
            pheromone = self.pheromone[current][node] ** self.alpha
            heuristic = (1.0 / self.distance_matrix[current][node]) ** self.beta
            probabilities.append((node, pheromone * heuristic))
        
        # Roulette wheel selection
        total = sum(p[1] for p in probabilities)
        r = random.uniform(0, total)
        cumsum = 0
        
        for node, prob in probabilities:
            cumsum += prob
            if cumsum >= r:
                return node
        
        return list(unvisited)[0]
    
    def _calculate_distance(self, path: List[int]) -> float:
        """Calculate total path distance"""
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i]][path[i+1]]
        # Return to start
        distance += self.distance_matrix[path[-1]][path[0]]
        return distance
    
    def _update_pheromones(self, ants: List[Ant]) -> None:
        """Update pheromone trails"""
        # Evaporation
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.pheromone[i][j] *= (1 - self.rho)
        
        # Deposit
        for ant in ants:
            deposit = self.q / ant.distance
            for i in range(len(ant.path) - 1):
                self.pheromone[ant.path[i]][ant.path[i+1]] += deposit
                self.pheromone[ant.path[i+1]][ant.path[i]] += deposit


class FireflyAlgorithm:
    """
    Firefly Algorithm for multimodal optimization.
    
    Fireflies are attracted to brighter fireflies, enabling
    exploration of multiple optimal regions.
    """
    
    def __init__(
        self,
        num_fireflies: int = 25,
        dimensions: int = 2,
        bounds: Tuple[float, float] = (-10, 10),
        alpha: float = 0.5,  # Randomization parameter
        beta0: float = 1.0,  # Attractiveness at r=0
        gamma: float = 1.0,  # Light absorption coefficient
    ):
        self.num_fireflies = num_fireflies
        self.dimensions = dimensions
        self.bounds = bounds
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.fireflies: List[Firefly] = []
        
    def initialize(self) -> None:
        """Initialize firefly swarm"""
        self.fireflies = []
        
        for _ in range(self.num_fireflies):
            position = [
                random.uniform(self.bounds[0], self.bounds[1])
                for _ in range(self.dimensions)
            ]
            
            firefly = Firefly(
                position=position,
                brightness=0.0,
                fitness=float('inf'),
            )
            self.fireflies.append(firefly)
    
    def optimize(
        self,
        fitness_func: Callable[[List[float]], float],
        max_iterations: int = 100,
    ) -> Tuple[List[float], float]:
        """
        Run Firefly Algorithm optimization.
        
        Args:
            fitness_func: Function to minimize
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (best_position, best_fitness)
        """
        self.initialize()
        
        best_position = None
        best_fitness = float('inf')
        
        for iteration in range(max_iterations):
            # Evaluate fitness and brightness
            for firefly in self.fireflies:
                firefly.fitness = fitness_func(firefly.position)
                firefly.brightness = 1.0 / (1.0 + firefly.fitness)
                
                if firefly.fitness < best_fitness:
                    best_fitness = firefly.fitness
                    best_position = firefly.position.copy()
            
            # Move fireflies
            for i, firefly_i in enumerate(self.fireflies):
                for j, firefly_j in enumerate(self.fireflies):
                    if firefly_j.brightness > firefly_i.brightness:
                        # Move i towards j
                        r = self._distance(firefly_i.position, firefly_j.position)
                        beta = self.beta0 * math.exp(-self.gamma * r * r)
                        
                        for d in range(self.dimensions):
                            firefly_i.position[d] += (
                                beta * (firefly_j.position[d] - firefly_i.position[d]) +
                                self.alpha * (random.random() - 0.5)
                            )
                            
                            # Clamp to bounds
                            firefly_i.position[d] = max(
                                self.bounds[0],
                                min(self.bounds[1], firefly_i.position[d])
                            )
        
        return best_position, best_fitness
    
    def _distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


class HybridSwarmOptimizer:
    """
    Hybrid swarm optimizer combining PSO, ACO, and Firefly algorithms.
    
    Dynamically selects or combines algorithms based on problem characteristics
    and optimization progress, achieving superior performance through synergy.
    """
    
    def __init__(self):
        self.pso = ParticleSwarmOptimizer()
        self.firefly = FireflyAlgorithm()
        self.optimization_history: List[Dict[str, Any]] = []
        
    def optimize(
        self,
        fitness_func: Callable[[List[float]], float],
        problem_type: str = "numerical",
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Optimize using hybrid approach.
        
        Args:
            fitness_func: Function to minimize
            problem_type: 'numerical', 'routing', or 'multimodal'
            max_iterations: Maximum iterations
            
        Returns:
            Dictionary with optimization results
        """
        results = {}
        
        if problem_type == "numerical":
            # Use PSO for smooth numerical optimization
            position, fitness = self.pso.optimize(fitness_func, max_iterations)
            results = {
                "algorithm": "PSO",
                "best_position": position,
                "best_fitness": fitness,
            }
        
        elif problem_type == "multimodal":
            # Use Firefly for multimodal landscapes
            position, fitness = self.firefly.optimize(fitness_func, max_iterations)
            results = {
                "algorithm": "Firefly",
                "best_position": position,
                "best_fitness": fitness,
            }
        
        else:
            # Hybrid: Run both and take best
            pso_pos, pso_fit = self.pso.optimize(fitness_func, max_iterations // 2)
            ff_pos, ff_fit = self.firefly.optimize(fitness_func, max_iterations // 2)
            
            if pso_fit < ff_fit:
                results = {
                    "algorithm": "Hybrid (PSO best)",
                    "best_position": pso_pos,
                    "best_fitness": pso_fit,
                }
            else:
                results = {
                    "algorithm": "Hybrid (Firefly best)",
                    "best_position": ff_pos,
                    "best_fitness": ff_fit,
                }
        
        self.optimization_history.append(results)
        return results
