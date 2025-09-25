"""
差分进化算法模块（修复版）
分离优化和模拟过程，确保公平比较
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import copy
from metabolic_model import MetabolicModel, PersonProfile
from fitness_evaluator import FitnessEvaluator
from solution_generator import Solution, SolutionGenerator
from config import ConfigManager

logger = logging.getLogger(__name__)


class DifferentialEvolution:
    """修复后的差分进化算法 - 正确分离优化和模拟"""
    
    def __init__(self, person: PersonProfile, config: ConfigManager):
        self.initial_person = copy.deepcopy(person)  # 保存初始状态
        self.config = config
        # 传递配置给MetabolicModel
        self.metabolic_model = MetabolicModel(config)  # ✅ 传递config
        self.fitness_evaluator = FitnessEvaluator(config=config)  # ✅ 传递config
        self.solution_generator = SolutionGenerator(config=config)  # ✅ 传递config
        
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
        """交叉操作"""
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
        """获取缩放因子"""
        if self.config.algorithm.adaptive_parameters:
            # 自适应F值
            return np.random.uniform(*self.config.algorithm.scaling_factor_range)
        return self.config.algorithm.scaling_factor
    
    def _get_crossover_rate(self) -> float:
        """获取交叉率"""
        if self.config.algorithm.adaptive_parameters:
            # 自适应CR值
            return np.random.uniform(*self.config.algorithm.crossover_rate_range)
        return self.config.algorithm.crossover_rate
    
    def evaluate_population(self, solution: Solution, person: PersonProfile, 
                                    weeks: int) -> Tuple[float, Dict]:
        """
        评估方案在指定周期内的效果（不改变原始person）
        这是核心修复：评估时模拟整个周期，但不改变优化过程中的身体状态
        """
        # 复制一份person进行模拟
        sim_person = copy.deepcopy(person)
        
        total_weight_loss = 0
        total_fat_loss = 0
        total_muscle_loss = 0
        final_metabolic_factor = 1.0
        
        # 模拟整个周期
        for week in range(weeks):
            results = self.metabolic_model.simulate_week(sim_person, solution, week)
            
            total_weight_loss += results['total_weight_loss']
            total_fat_loss += results['fat_loss']
            total_muscle_loss += results['muscle_loss']
            final_metabolic_factor = results['metabolic_adaptation_factor']
            
            # 更新模拟人物状态
            sim_person = self.metabolic_model.update_person_state(sim_person, solution, week)
        
        # 计算综合适应度
        muscle_loss_rate = total_muscle_loss / total_weight_loss if total_weight_loss > 0 else 0
        fat_loss_rate = total_fat_loss / total_weight_loss if total_weight_loss > 0 else 1
        
        simulation_results = {
            'total_weight_loss': total_weight_loss,
            'muscle_loss_rate': muscle_loss_rate,
            'fat_loss_rate': fat_loss_rate,
            'energy_deficit': 500,  # 示例值
            'final_weight': sim_person.weight,
            'final_body_fat': sim_person.body_fat_percentage,
            'metabolic_adaptation': final_metabolic_factor
        }
        
        fitness, components = self.fitness_evaluator.calculate_fitness(
            simulation_results, solution, person
        )
        
        return fitness, simulation_results
    
    def evaluate_population(self, population: List[Solution], person: PersonProfile, 
                          simulation_weeks: int):
        """评估种群（基于未来模拟）"""
        for solution in population:
            if solution.fitness is None:
                fitness, results = self.evaluate_population(
                    solution, person, simulation_weeks
                )
                solution.fitness = fitness
                solution.fitness_components = results
    
    def optimize(self, weeks: int) -> Tuple[Solution, Dict]:
        """
        为指定周期找到最优方案
        注意：优化过程不改变身体状态，只是寻找最优方案
        """
        logger.info(f"开始为{weeks}周期寻找最优方案...")
        
        # 使用初始身体状态进行优化
        optimization_person = copy.deepcopy(self.initial_person)
        
        # 初始化种群
        bmr = self.metabolic_model.calculate_bmr(optimization_person)
        tdee = bmr * optimization_person.activity_level
        population = [Solution(vec) for vec in self.solution_generator.generate_diverse_population(
            self.config.algorithm.population_size, tdee
        )]
        
        # 评估初始种群（模拟weeks周的效果）
        self.evaluate_population(population, optimization_person, weeks)
        
        # 找到初始最佳方案
        best_solution = min(population, key=lambda x: x.fitness)
        logger.info(f"初始最佳适应度: {best_solution.fitness:.3f}")
        
        # 优化迭代（注意：这里是算法迭代，不是周数）
        max_iterations = self.config.algorithm.max_iterations  # 使用配置的迭代次数
        
        for iteration in range(max_iterations):
            if iteration % 10 == 0:
                logger.info(f"优化迭代 {iteration}/{max_iterations}, 当前最佳: {best_solution.fitness:.3f}")
            
            new_population = []
            
            for i, target in enumerate(population):
                # 变异
                mutant = self.mutate(population, i)
                
                # 交叉
                trial = self.crossover(target, mutant)
                
                # 验证约束
                if not self.solution_generator.validate_solution(trial.to_vector(), optimization_person.weight):
                    new_population.append(target)
                    continue
                
                # 评估试验方案（模拟weeks周）
                trial.fitness, trial.fitness_components = self.evaluate_population(
                    trial, optimization_person, weeks
                )
                
                # 选择
                if trial.fitness < target.fitness:
                    new_population.append(trial)
                else:
                    new_population.append(target)
            
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
            
            # 检查早停条件
            if self._check_convergence(iteration):
                logger.info(f"算法收敛，在第{iteration}次迭代停止")
                break
        
        optimization_results = {
            'best_solution': best_solution,
            'iterations': iteration + 1,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }
        
        return best_solution, optimization_results
    
    def simulate_solution(self, solution: Solution, person: PersonProfile, 
                         weeks: int) -> Dict:
        """
        模拟方案执行（用于最终效果评估）
        这是实际执行方案的过程
        """
        sim_person = copy.deepcopy(person)
        week_results = []
        
        for week in range(weeks):
            results = self.metabolic_model.simulate_week(sim_person, solution, week)
            week_results.append({
                'week': week,
                'weight': sim_person.weight,
                'body_fat': sim_person.body_fat_percentage,
                'metabolic_factor': sim_person.metabolic_adaptation_factor,
                **results
            })
            
            sim_person = self.metabolic_model.update_person_state(sim_person, solution, week)
        
        return {
            'initial_weight': person.weight,
            'final_weight': sim_person.weight,
            'total_weight_loss': person.weight - sim_person.weight,
            'final_person': sim_person,
            'weekly_results': week_results
        }
    
    def _check_convergence(self, iteration: int) -> bool:
        """检查收敛条件"""
        if not self.config.algorithm.early_stopping:
            return False
        
        if len(self.best_fitness_history) >= self.config.algorithm.patience:
            recent = self.best_fitness_history[-self.config.algorithm.patience:]
            improvements = [abs(recent[i] - recent[i+1]) for i in range(len(recent)-1)]
            
            if all(imp < self.config.algorithm.min_improvement for imp in improvements):
                return True
        
        return False


class AdaptiveDifferentialEvolution(DifferentialEvolution):
    """自适应差分进化 - 可以生成周期性变化的方案"""
    
    def optimize_adaptive_plan(self, weeks: int) -> Tuple[List[Solution], Dict]:
        """
        优化自适应方案（每周可以不同）
        返回每周的最优方案序列
        """
        logger.info(f"开始优化{weeks}周的自适应方案...")
        
        # 将周期分段
        phases = self._divide_into_phases(weeks)
        weekly_solutions = []
        current_person = copy.deepcopy(self.initial_person)
        
        for phase_name, phase_weeks in phases:
            logger.info(f"优化{phase_name}阶段（{len(phase_weeks)}周）...")
            
            # 为这个阶段找到最优方案
            phase_solution, _ = self.optimize(len(phase_weeks))
            
            # 应用到每周
            for week in phase_weeks:
                # 可以根据周数微调方案
                adjusted_solution = self._adjust_solution_for_week(
                    phase_solution, week, current_person
                )
                weekly_solutions.append(adjusted_solution)
                
                # 更新身体状态（用于下一阶段）
                results = self.metabolic_model.simulate_week(
                    current_person, adjusted_solution, week
                )
                current_person = self.metabolic_model.update_person_state(
                    current_person, adjusted_solution, week
                )
        
        # 评估整体效果
        total_results = self.evaluate_solution_sequence(
            weekly_solutions, self.initial_person
        )
        
        return weekly_solutions, total_results
    
    def _divide_into_phases(self, weeks: int) -> List[Tuple[str, List[int]]]:
        """将周期分为不同阶段"""
        phases = []
        
        if weeks <= 4:
            phases.append(("快速启动", list(range(weeks))))
        elif weeks <= 8:
            phases.append(("快速启动", list(range(2))))
            phases.append(("稳定推进", list(range(2, weeks))))
        else:
            # 初期（2周）
            phases.append(("适应期", list(range(2))))
            # 中期
            mid_weeks = (weeks - 2) // 2
            phases.append(("推进期", list(range(2, 2 + mid_weeks))))
            # 后期
            phases.append(("巩固期", list(range(2 + mid_weeks, weeks))))
        
        return phases
    
    def _adjust_solution_for_week(self, base_solution: Solution, 
                                 week: int, person: PersonProfile) -> Solution:
        """根据周数和身体状态调整方案"""
        adjusted = copy.deepcopy(base_solution)
        
        # 检测代谢适应
        if hasattr(person, 'metabolic_adaptation_factor'):
            if person.metabolic_adaptation_factor < 0.85:
                # 代谢适应严重，进行调整
                # 提高热量（反向节食）
                adjusted.calories *= 1.05
                # 增加碳水
                adjusted.carb_ratio = min(0.5, adjusted.carb_ratio * 1.1)
                # 调整其他比例
                total = adjusted.protein_ratio + adjusted.carb_ratio + adjusted.fat_ratio
                adjusted.protein_ratio /= total
                adjusted.carb_ratio /= total
                adjusted.fat_ratio /= total
        
        # 周期性变化（如碳水循环）
        if week % 4 == 3:  # 每4周的最后一周
            # Refeed day
            adjusted.calories *= 1.1
            adjusted.carb_ratio = min(0.5, adjusted.carb_ratio * 1.2)
        
        return adjusted
    
    def evaluate_solution_sequence(self, solutions: List[Solution], 
                                  person: PersonProfile) -> Dict:
        """评估方案序列的整体效果"""
        sim_person = copy.deepcopy(person)
        total_weight_loss = 0
        weekly_results = []
        
        for week, solution in enumerate(solutions):
            results = self.metabolic_model.simulate_week(sim_person, solution, week)
            total_weight_loss += results['total_weight_loss']
            
            weekly_results.append({
                'week': week,
                'weight': sim_person.weight,
                'solution': solution.to_vector().tolist()
            })
            
            sim_person = self.metabolic_model.update_person_state(
                sim_person, solution, week
            )
        
        return {
            'total_weight_loss': person.weight - sim_person.weight,
            'final_weight': sim_person.weight,
            'final_body_fat': sim_person.body_fat_percentage,
            'weekly_results': weekly_results
        }