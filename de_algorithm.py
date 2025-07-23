"""
差分进化算法模块
只包含差分进化算法的核心实现
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from metabolic_model import MetabolicModel, PersonProfile
from fitness_evaluator import FitnessEvaluator
from solution_generator import Solution, SolutionGenerator
from config import ConfigManager

logger = logging.getLogger(__name__)


class DifferentialEvolution:
    """差分进化算法实现 - 只负责核心算法逻辑"""
    
    def __init__(self, person: PersonProfile, config: ConfigManager):
        self.person = person
        self.config = config
        self.metabolic_model = MetabolicModel()
        self.fitness_evaluator = FitnessEvaluator()
        self.solution_generator = SolutionGenerator()
        
        # 记录进化历史
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solutions_history = []
        self.population_history = []
        self.fitness_history = []
        
    def mutate(self, population: List[Solution], target_idx: int) -> Solution:
        """变异操作 - DE/rand/1策略"""
        # 随机选择三个不同的个体
        indices = list(range(len(population)))
        indices.remove(target_idx)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        
        # 获取向量
        v_r1 = population[r1].to_vector()
        v_r2 = population[r2].to_vector()
        v_r3 = population[r3].to_vector()
        
        # 差分变异
        F = self._get_scaling_factor()
        mutant_vector = v_r1 + F * (v_r2 - v_r3)
        
        # 使用solution_generator处理边界
        mutant_vector = self.solution_generator.handle_bounds(mutant_vector)
        
        return Solution(mutant_vector)
    
    def crossover(self, target: Solution, mutant: Solution) -> Solution:
        """交叉操作 - 二项交叉"""
        target_vector = target.to_vector()
        mutant_vector = mutant.to_vector()
        trial_vector = target_vector.copy()
        
        CR = self._get_crossover_rate()
        
        # 确保至少有一个维度来自变异向量
        j_rand = np.random.randint(0, len(trial_vector))
        
        for j in range(len(trial_vector)):
            if np.random.rand() < CR or j == j_rand:
                trial_vector[j] = mutant_vector[j]
        
        # 使用solution_generator进行规范化
        trial_vector = self.solution_generator.normalize_solution_vector(trial_vector)
        
        return Solution(trial_vector)
    
    def _get_scaling_factor(self) -> float:
        """获取缩放因子（支持自适应）"""
        if self.config.algorithm.adaptive_parameters:
            # 自适应F值
            return np.random.uniform(*self.config.algorithm.scaling_factor_range)
        return self.config.algorithm.scaling_factor
    
    def _get_crossover_rate(self) -> float:
        """获取交叉率（支持自适应）"""
        if self.config.algorithm.adaptive_parameters:
            # 自适应CR值
            return np.random.uniform(*self.config.algorithm.crossover_rate_range)
        return self.config.algorithm.crossover_rate
    
    def evaluate_population(self, population: List[Solution], week: int):
        """评估种群适应度"""
        for solution in population:
            if solution.fitness is None:  # 避免重复评估
                # 模拟一周的执行
                results = self.metabolic_model.simulate_week(
                    self.person, solution, week
                )
                
                # 计算适应度
                fitness, components = self.fitness_evaluator.calculate_fitness(
                    results, solution, self.person
                )
                
                solution.fitness = fitness
                solution.fitness_components = components
    
    def selection(self, target: Solution, trial: Solution) -> Solution:
        """选择操作"""
        if trial.fitness < target.fitness:
            logger.info(f"改进: {target.fitness:.3f} -> {trial.fitness:.3f}")
            return trial
        return target
    
    def check_termination(self, iteration: int, best_fitness: float) -> bool:
        """检查终止条件"""
        # 达到最大迭代次数
        if iteration >= self.config.algorithm.max_iterations:
            return True
        
        # 早停条件
        if self.config.algorithm.early_stopping and len(self.best_fitness_history) >= self.config.algorithm.patience:
            recent_improvements = []
            for i in range(1, self.config.algorithm.patience + 1):
                if i <= len(self.best_fitness_history):
                    improvement = self.best_fitness_history[-i] - best_fitness
                    recent_improvements.append(improvement)
            
            # 如果最近几代改善都小于阈值，则停止
            if all(imp < self.config.algorithm.min_improvement for imp in recent_improvements):
                logger.info(f"触发早停条件，连续{self.config.algorithm.patience}代改善小于{self.config.algorithm.min_improvement}")
                return True
        
        return False
    
    def optimize(self) -> Tuple[Solution, Dict]:
        """执行优化过程 - 核心算法"""
        logger.info("开始差分进化优化...")
        logger.info(f"目标用户: {self.person}")
        
        # 初始化种群（使用solution_generator）
        bmr = self.metabolic_model.calculate_bmr(self.person)
        tdee = bmr * self.person.activity_level
        population = [Solution(vec) for vec in self.solution_generator.generate_diverse_population(
            self.config.algorithm.population_size, tdee
        )]
        
        # 评估初始种群
        self.evaluate_population(population, week=0)
        
        # 找到初始最佳方案
        best_solution = min(population, key=lambda x: x.fitness)
        logger.info(f"初始最佳方案: {best_solution}, 适应度: {best_solution.fitness:.3f}")
        
        # 进化主循环
        iteration = 0
        while not self.check_termination(iteration, best_solution.fitness):
            logger.info(f"\n--- 第 {iteration + 1} 周 ---")
            
            new_population = []
            
            for i, target in enumerate(population):
                # 变异
                mutant = self.mutate(population, i)
                
                # 交叉
                trial = self.crossover(target, mutant)
                
                # 验证约束
                if not self.solution_generator.validate_solution(trial.to_vector(), self.person.weight):
                    new_population.append(target)
                    continue
                
                # 评估试验方案
                self.evaluate_population([trial], week=iteration+1)
                
                # 选择
                selected = self.selection(target, trial)
                new_population.append(selected)
            
            population = new_population
            
            # 更新最佳方案
            current_best = min(population, key=lambda x: x.fitness)
            if current_best.fitness < best_solution.fitness:
                best_solution = current_best
                logger.info(f"发现更优方案! 适应度: {best_solution.fitness:.3f}")
            
            # 记录历史
            self.best_fitness_history.append(best_solution.fitness)
            avg_fitness = np.mean([s.fitness for s in population])
            self.avg_fitness_history.append(avg_fitness)
            self.best_solutions_history.append(best_solution)
            self.population_history.append(population.copy())
            self.fitness_history.append([s.fitness for s in population])
            
            # 更新人体状态（模拟代谢适应）
            self.person = self.metabolic_model.update_person_state(
                self.person, best_solution, week=iteration+1
            )
            
            iteration += 1
        
        # 返回结果
        results = {
            'best_solution': best_solution,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'best_solutions_history': self.best_solutions_history,
            'population_history': self.population_history,
            'fitness_history': self.fitness_history,
            'final_person_state': self.person,
            'initial_weight': self.person.initial_weight,
            'total_iterations': iteration
        }
        
        return best_solution, results