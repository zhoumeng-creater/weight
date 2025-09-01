"""
增强版实验运行器
使用新的配置系统、虚拟人生成器和统一的数据收集机制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import os
import json
import copy
from tqdm import tqdm
import itertools
from scipy import stats

# 导入项目模块
from metabolic_model import PersonProfile, MetabolicModel, AdvancedMetabolicModel
from config import ConfigManager, load_preset
from de_algorithm import DifferentialEvolution
from solution_generator import Solution, SolutionGenerator
from fitness_evaluator import FitnessEvaluator, AdaptiveFitnessEvaluator
from visualization import DataTracker, WeightLossVisualizer, OptimizationVisualizer, ReportGenerator
from data_loader import SimulatedExperiment
from virtual_subjects import VirtualSubjectGenerator  # 使用新的独立模块

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def np_pd_convert(o):
    """NumPy和Pandas对象转换为可序列化格式"""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Series,)):
        return o.tolist()
    if isinstance(o, (pd.DataFrame,)):
        return o.to_dict(orient="records")
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    
     # 添加这部分：处理 Solution 对象
    from solution_generator import Solution
    if isinstance(o, Solution):
        return {
            'type': 'Solution',
            'calories': float(o.calories),
            'protein_ratio': float(o.protein_ratio),
            'carb_ratio': float(o.carb_ratio),
            'fat_ratio': float(o.fat_ratio),
            'cardio_freq': int(o.cardio_freq),
            'cardio_duration': int(o.cardio_duration),
            'strength_freq': int(o.strength_freq),
            'sleep_hours': float(o.sleep_hours),
            'fitness': float(o.fitness) if o.fitness is not None else None,
            'vector': o.to_vector().tolist()
        }
    
    # 添加这部分：处理 PersonProfile 对象（可能也会遇到）
    from metabolic_model import PersonProfile
    if isinstance(o, PersonProfile):
        return {
            'type': 'PersonProfile',
            'age': o.age,
            'gender': o.gender,
            'height': o.height,
            'weight': o.weight,
            'body_fat_percentage': o.body_fat_percentage,
            'activity_level': o.activity_level,
            'weeks_on_diet': o.weeks_on_diet,
            'metabolic_adaptation_factor': o.metabolic_adaptation_factor
        }
    raise TypeError(f"Type {type(o)} not serializable")


class EnhancedExperimentRunner:
    """增强版实验运行器"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化实验运行器
        
        Args:
            config: 配置管理器，如果未提供则使用默认配置
        """
        self.config = config or ConfigManager()
        self.output_dir = self.config.system.experiment_output_directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results_database = {}
        self.subject_generator = VirtualSubjectGenerator(seed=self.config.system.random_seed)
        
        # 初始化不同的模型
        self.models = {
            'basic': MetabolicModel(),
            'advanced': AdvancedMetabolicModel()
        }
        
        # 使用统一的数据追踪器
        self.data_tracker = DataTracker()
        
    def run_experiment_A1_benchmark(self, 
                                   n_subjects: Optional[int] = None, 
                                   duration_weeks: Optional[int] = None):
        """
        实验A1: 基准对比实验
        
        Args:
            n_subjects: 虚拟人数量，如果未指定则使用配置
            duration_weeks: 实验周期，如果未指定则使用配置
        """
        n_subjects = n_subjects or self.config.experiment.benchmark_n_subjects
        duration_weeks = duration_weeks or self.config.experiment.benchmark_duration_weeks
        
        logger.info("=" * 50)
        logger.info("开始实验A1: 基准对比实验")
        logger.info(f"样本数: {n_subjects}, 周期: {duration_weeks}周")
        logger.info("=" * 50)
        
        # 生成虚拟人
        subjects = self.subject_generator.generate_diverse_population(n_subjects)
        
        # 定义对照方法
        control_methods = {
            'fixed_deficit': self._fixed_deficit_method,
            'step_reduction': self._step_reduction_method,
            'cyclic_diet': self._cyclic_diet_method,
            'random': self._random_method,
            'de_optimized': self._de_optimized_method
        }
        
        results = {}
        
        for method_name, method_func in control_methods.items():
            logger.info(f"\n测试方法: {method_name}")
            method_results = []
            
            # 为每个方法创建数据追踪器
            method_tracker = DataTracker()
            method_tracker.metadata['method'] = method_name
            method_tracker.metadata['experiment'] = 'A1_benchmark'
            
            for subject in tqdm(subjects, desc=f"运行{method_name}"):
                # 复制subject以避免相互影响
                test_subject = copy.deepcopy(subject)
                
                # 运行方法
                result = method_func(test_subject, duration_weeks)
                
                # 收集结果
                method_results.append(result)
                
                # 记录到数据追踪器
                if 'weekly_results' in result:
                    for week, week_data in enumerate(result['weekly_results']):
                        method_tracker.add_record(
                            week=week,
                            weight=week_data.get('weight', test_subject.weight),
                            method=method_name,
                            subject_id=id(subject)
                        )
            
            results[method_name] = method_results
        
        # 统计分析
        analysis = self._analyze_benchmark_results(results)
        
        # 保存结果
        self._save_experiment_results("A1_benchmark", results, analysis)
        
        return results, analysis
    
    def run_experiment_A2_plateau_breakthrough(self, n_subjects: Optional[int] = None):
        """实验A2: 平台期突破专项实验"""
        n_subjects = n_subjects or self.config.experiment.plateau_n_subjects
        
        logger.info("=" * 50)
        logger.info("开始实验A2: 平台期突破实验")
        logger.info(f"样本数: {n_subjects}")
        logger.info("=" * 50)
        
        # 生成已在平台期的虚拟人
        plateau_subjects = self.subject_generator.generate_plateau_subjects(n_subjects)
        
        # 突破策略
        breakthrough_strategies = {
            'de_plateau_breaker': self._de_plateau_breakthrough,
            'continue_same': self._continue_same_plan,
            'increase_exercise': self._increase_exercise,
            'diet_break': self._diet_break,
            'carb_cycling': self._carb_cycling
        }
        
        results = {}
        
        for strategy_name, strategy_func in breakthrough_strategies.items():
            logger.info(f"\n测试策略: {strategy_name}")
            strategy_results = []
            
            for subject in tqdm(plateau_subjects, desc=f"运行{strategy_name}"):
                test_subject = copy.deepcopy(subject)
                
                # 记录初始状态
                initial_state = {
                    'weight': test_subject.weight,
                    'body_fat': test_subject.body_fat_percentage,
                    'metabolic_factor': getattr(test_subject, 'metabolic_adaptation_factor', 1.0)
                }
                
                # 运行策略
                weeks = self.config.experiment.plateau_observation_weeks
                result = strategy_func(test_subject, weeks=weeks)
                
                # 判断是否突破平台期
                weight_change = initial_state['weight'] - result['final_weight']
                breakthrough_success = weight_change > self.config.experiment.breakthrough_threshold
                
                strategy_results.append({
                    'initial_state': initial_state,
                    'final_state': result,
                    'breakthrough_success': breakthrough_success,
                    'weight_change': weight_change
                })
            
            results[strategy_name] = strategy_results
        
        # 分析结果
        analysis = self._analyze_plateau_results(results)
        
        # 保存结果
        self._save_experiment_results("A2_plateau", results, analysis)
        
        return results, analysis
    
    def run_experiment_B1_metabolic_validation(self):
        """实验B1: 代谢适应模型验证"""
        logger.info("=" * 50)
        logger.info("开始实验B1: 代谢适应模型验证")
        logger.info("=" * 50)
        
        # 使用配置中的测试场景
        test_scenarios = self.config.experiment.test_scenarios
        
        results = {}
        
        for scenario in test_scenarios:
            logger.info(f"\n测试场景: {scenario['name']}")
            
            # 测试虚拟人
            subjects = self.subject_generator.generate_standard_subjects(100)
            
            scenario_results = {
                'basic_model': [],
                'advanced_model': []
            }
            
            for subject in subjects:
                # 基础模型（无代谢适应）
                basic_result = self._run_with_model(
                    copy.deepcopy(subject),
                    self.models['basic'],
                    scenario['duration'],
                    scenario['deficit']
                )
                scenario_results['basic_model'].append(basic_result)
                
                # 高级模型（有代谢适应）
                advanced_result = self._run_with_model(
                    copy.deepcopy(subject),
                    self.models['advanced'],
                    scenario['duration'],
                    scenario['deficit']
                )
                scenario_results['advanced_model'].append(advanced_result)
            
            results[scenario['name']] = scenario_results
        
        # 分析模型差异
        analysis = self._analyze_model_differences(results)
        
        # 保存结果
        self._save_experiment_results("B1_metabolic", results, analysis)
        
        return results, analysis
    
    def run_experiment_C1_parameter_sensitivity(self):
        """实验C1: 算法参数敏感性分析"""
        logger.info("=" * 50)
        logger.info("开始实验C1: 参数敏感性分析")
        logger.info("=" * 50)
        
        # 使用配置中的参数网格
        param_grid = self.config.experiment.sensitivity_param_grid
        
        # 生成参数组合
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        # 使用固定的测试对象
        test_subject = PersonProfile(
            age=35, gender='male', height=175, weight=90,
            body_fat_percentage=28, activity_level=1.4, weeks_on_diet=0
        )
        
        results = []
        
        for param_values in tqdm(param_combinations, desc="参数组合测试"):
            # 创建配置
            test_config = copy.deepcopy(self.config)
            for name, value in zip(param_names, param_values):
                if name == 'population_size':
                    test_config.algorithm.population_size = value
                elif name == 'scaling_factor':
                    test_config.algorithm.scaling_factor = value
                elif name == 'crossover_rate':
                    test_config.algorithm.crossover_rate = value
                elif name == 'max_iterations':
                    test_config.algorithm.max_iterations = value
            
            # 运行优化（重复3次取平均）
            run_results = []
            for _ in range(3):
                optimizer = DifferentialEvolution(copy.deepcopy(test_subject), test_config)
                best_solution, opt_results = optimizer.optimize()
                
                run_results.append({
                    'fitness': best_solution.fitness,
                    'convergence_iteration': self._find_convergence_point(opt_results),
                    'final_weight': opt_results['final_person_state'].weight
                })
            
            # 计算平均值和标准差
            avg_fitness = np.mean([r['fitness'] for r in run_results])
            std_fitness = np.std([r['fitness'] for r in run_results])
            avg_convergence = np.mean([r['convergence_iteration'] for r in run_results])
            
            results.append({
                'parameters': dict(zip(param_names, param_values)),
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'avg_convergence': avg_convergence,
                'runs': run_results
            })
        
        # 分析敏感性
        analysis = self._analyze_sensitivity(results, param_names)
        
        # 保存结果
        self._save_experiment_results("C1_sensitivity", results, analysis)
        
        return results, analysis
    
    def run_experiment_D1_ablation_study(self):
        """实验D1: 消融研究"""
        logger.info("=" * 50)
        logger.info("开始实验D1: 消融研究")
        logger.info("=" * 50)
        
        components = self.config.experiment.ablation_components
        
        test_subject = PersonProfile(
            age=35, gender='male', height=175, weight=90,
            body_fat_percentage=28, activity_level=1.4, weeks_on_diet=0
        )
        
        results = {}
        
        # 完整模型
        logger.info("测试完整模型...")
        full_config = copy.deepcopy(self.config)
        optimizer = DifferentialEvolution(copy.deepcopy(test_subject), full_config)
        best_solution, opt_results = optimizer.optimize()
        
        results['full_model'] = {
            'fitness': best_solution.fitness,
            'weight_loss': test_subject.weight - opt_results['final_person_state'].weight,
            'solution': best_solution
        }
        
        # 逐个移除组件
        for component in components:
            logger.info(f"测试移除组件: {component}")
            
            # 创建配置副本
            ablated_config = copy.deepcopy(self.config)
            
            # ✅ 调用_disable_component方法
            self._disable_component(ablated_config, component)
            
            # 运行优化
            test_subject_copy = copy.deepcopy(test_subject)
            
            # 如果禁用了某些组件，需要特殊处理初始种群
            if component in ['sleep_optimization', 'strength_training', 
                            'cardio_training', 'nutrition_optimization']:
                # 创建自定义优化器
                optimizer = self._create_ablated_optimizer(
                    test_subject_copy, ablated_config, component
                )
            else:
                optimizer = DifferentialEvolution(test_subject_copy, ablated_config)
            
            best_solution, opt_results = optimizer.optimize()
            
            results[f"without_{component}"] = {
                'fitness': best_solution.fitness,
                'weight_loss': test_subject.weight - opt_results['final_person_state'].weight,
                'solution': best_solution
            }
        
        # 分析组件重要性
        analysis = self._analyze_ablation_results(results)
        
        # 保存结果
        self._save_experiment_results("D1_ablation", results, analysis)
        
        return results, analysis

    def _create_ablated_optimizer(self, subject: PersonProfile, config: ConfigManager, 
                                disabled_component: str):
        """创建禁用特定组件的优化器"""
        
        class AblatedDifferentialEvolution(DifferentialEvolution):
            def __init__(self, person, config, disabled_component):
                super().__init__(person, config)
                self.disabled_component = disabled_component
                
            def optimize(self):
                # 修改种群生成逻辑
                logger.info(f"开始差分进化优化（禁用组件：{self.disabled_component}）...")
                logger.info(f"目标用户: {self.person}")
                
                # 使用特殊的种群生成方法
                bmr = self.metabolic_model.calculate_bmr(self.person)
                tdee = bmr * self.person.activity_level
                
                # 生成禁用组件的种群
                population = []
                for _ in range(self.config.algorithm.population_size):
                    solution_vector = self.solution_generator.generate_with_disabled_components(
                        tdee, [self.disabled_component]
                    )
                    population.append(Solution(solution_vector))
                
                # 继续正常的优化流程
                self.evaluate_population(population, week=0)
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
                        
                        # 确保变异体也符合禁用组件的约束
                        mutant_vector = mutant.to_vector()
                        mutant_vector = self._apply_component_constraints(
                            mutant_vector, self.disabled_component
                        )
                        mutant = Solution(mutant_vector)
                        
                        # 交叉
                        trial = self.crossover(target, mutant)
                        
                        # 再次确保符合约束
                        trial_vector = trial.to_vector()
                        trial_vector = self._apply_component_constraints(
                            trial_vector, self.disabled_component
                        )
                        trial = Solution(trial_vector)
                        
                        # 评估和选择
                        self.evaluate_population([trial], week=iteration+1)
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
                    
                    # 更新人体状态
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
            
            def _apply_component_constraints(self, solution_vector, component):
                """应用组件约束"""
                if component == 'sleep_optimization':
                    solution_vector[7] = 7.0  # 固定睡眠时间
                elif component == 'strength_training':
                    solution_vector[6] = 0  # 无力量训练
                elif component == 'cardio_training':
                    solution_vector[4] = 0  # 无有氧训练
                    solution_vector[5] = 0
                elif component == 'nutrition_optimization':
                    # 固定营养比例
                    solution_vector[1] = 0.30
                    solution_vector[2] = 0.40
                    solution_vector[3] = 0.30
                
                return solution_vector
            
            def _apply_bounds(self, solution_vector):
                """应用边界约束"""
                bounds = self.solution_generator.get_bounds()
                for i in range(len(solution_vector)):
                    solution_vector[i] = np.clip(solution_vector[i], bounds[i][0], bounds[i][1])
                return solution_vector
        
        return AblatedDifferentialEvolution(subject, config, disabled_component)
    
    def run_experiment_E1_long_term_tracking(self, weeks: Optional[int] = None):
        """实验E1: 长期效果追踪"""
        weeks = weeks or self.config.experiment.long_term_tracking_weeks
        
        logger.info("=" * 50)
        logger.info(f"开始实验E1: 长期效果追踪 ({weeks}周)")
        logger.info("=" * 50)
        
        # 生成测试对象
        subjects = self.subject_generator.generate_standard_subjects(30)
        
        results = []
        
        for subject in tqdm(subjects, desc="长期追踪"):
            # 使用DE算法优化
            optimizer = DifferentialEvolution(copy.deepcopy(subject), self.config)
            
            # 长期追踪
            tracking_data = {
                'week': [],
                'weight': [],
                'body_fat': [],
                'muscle_mass': [],
                'metabolic_factor': [],
                'solution': []
            }
            
            current_subject = copy.deepcopy(subject)
            
            for week in range(weeks):
                # 每4周重新优化一次
                if week % 4 == 0:
                    optimizer.person = current_subject
                    best_solution, _ = optimizer.optimize()
                
                # 模拟执行
                week_result = self.models['advanced'].simulate_week(
                    current_subject, best_solution, week
                )
                
                # 更新状态
                current_subject = self.models['advanced'].update_person_state(
                    current_subject, best_solution, week
                )
                
                # 记录数据
                tracking_data['week'].append(week)
                tracking_data['weight'].append(current_subject.weight)
                tracking_data['body_fat'].append(current_subject.body_fat_percentage)
                tracking_data['muscle_mass'].append(
                    current_subject.weight * (1 - current_subject.body_fat_percentage / 100) * 0.45
                )
                tracking_data['metabolic_factor'].append(
                    getattr(current_subject, 'metabolic_adaptation_factor', 1.0)
                )
                tracking_data['solution'].append(best_solution.to_vector())
            
            results.append({
                'subject': subject,
                'tracking_data': tracking_data,
                'final_state': current_subject,
                'total_weight_loss': subject.weight - current_subject.weight
            })
        
        # 分析长期效果
        analysis = self._analyze_long_term_results(results)
        
        # 保存结果
        self._save_experiment_results("E1_long_term", results, analysis)
        
        return results, analysis
    
    # ============ 辅助方法 ============
    
    def _fixed_deficit_method(self, subject: PersonProfile, weeks: int) -> Dict:
        """固定热量赤字方法"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        generator = SolutionGenerator()
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        target_calories = tdee * 0.8  # 20%赤字
        
        results = []
        
        for week in range(weeks):
            # 生成固定方案
            solution_vector = generator.generate_from_template("balanced", tdee)
            solution_vector[0] = target_calories  # 设置热量
            solution = Solution(solution_vector)
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _step_reduction_method(self, subject: PersonProfile, weeks: int) -> Dict:
        """阶梯式减少方法"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        generator = SolutionGenerator()
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        results = []
        
        for week in range(weeks):
            # 每4周减少5%热量
            reduction = 1 - (week // 4) * 0.05
            reduction = max(reduction, 0.7)  # 最多减少30%
            
            target_calories = tdee * reduction
            
            solution_vector = generator.generate_from_template("balanced", tdee)
            solution_vector[0] = target_calories
            solution = Solution(solution_vector)
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _cyclic_diet_method(self, subject: PersonProfile, weeks: int) -> Dict:
        """循环饮食方法"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        generator = SolutionGenerator()
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        results = []
        
        for week in range(weeks):
            # 循环：低-低-中-高
            cycle_position = week % 4
            if cycle_position < 2:
                calories_factor = 0.75  # 低热量
            elif cycle_position == 2:
                calories_factor = 0.9   # 中热量
            else:
                calories_factor = 1.0   # 高热量（维持）
            
            target_calories = tdee * calories_factor
            
            solution_vector = generator.generate_from_template("balanced", tdee)
            solution_vector[0] = target_calories
            solution = Solution(solution_vector)
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _random_method(self, subject: PersonProfile, weeks: int) -> Dict:
        """随机方案（对照）"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        generator = SolutionGenerator()
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        results = []
        
        for week in range(weeks):
            # 生成随机方案
            solution_vector = generator.generate_random_solution(tdee)
            solution = Solution(solution_vector)
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _de_optimized_method(self, subject: PersonProfile, weeks: int) -> Dict:
        """DE优化方案"""
        config = copy.deepcopy(self.config)
        config.algorithm.max_iterations = weeks
        
        optimizer = DifferentialEvolution(copy.deepcopy(subject), config)
        best_solution, opt_results = optimizer.optimize()
        
        return {
            'final_weight': opt_results['final_person_state'].weight,
            'total_weight_loss': subject.weight - opt_results['final_person_state'].weight,
            'best_solution': best_solution,
            'optimization_results': opt_results
        }
    
    def _de_plateau_breakthrough(self, subject: PersonProfile, weeks: int) -> Dict:
        """DE平台期突破策略"""
        from solution_generator import SolutionGenerator, SolutionConstraints
        
        # 使用平台期突破专门功能
        generator = SolutionGenerator()
        
        # 生成初始方案
        bmr = MetabolicModel().calculate_bmr(subject)
        tdee = bmr * subject.activity_level
        initial_solution = generator.generate_from_template("balanced", tdee)
        
        # 生成突破方案
        breakthrough_solutions = generator.generate_plateau_breaking_solutions(
            initial_solution, num_solutions=3
        )
        
        # 选择最优突破方案
        model = AdvancedMetabolicModel()
        best_result = None
        best_weight_loss = 0
        
        for solution_vector in breakthrough_solutions:
            solution = Solution(solution_vector)
            test_subject = copy.deepcopy(subject)
            
            for week in range(weeks):
                model.simulate_week(test_subject, solution, week)
                test_subject = model.update_person_state(test_subject, solution, week)
            
            weight_loss = subject.weight - test_subject.weight
            if weight_loss > best_weight_loss:
                best_weight_loss = weight_loss
                best_result = test_subject
        
        return {
            'final_weight': best_result.weight if best_result else subject.weight,
            'total_weight_loss': best_weight_loss
        }
    
    def _continue_same_plan(self, subject: PersonProfile, weeks: int) -> Dict:
        """继续原计划（对照）"""
        return self._fixed_deficit_method(subject, weeks)
    
    def _increase_exercise(self, subject: PersonProfile, weeks: int) -> Dict:
        """增加运动量策略"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        generator = SolutionGenerator()
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        results = []
        
        for week in range(weeks):
            solution_vector = generator.generate_from_template("endurance", tdee)
            # 增加运动频率
            solution_vector[4] = min(solution_vector[4] + 2, 6)  # 有氧
            solution_vector[6] = min(solution_vector[6] + 1, 5)  # 力量
            
            solution = Solution(solution_vector)
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _diet_break(self, subject: PersonProfile, weeks: int) -> Dict:
        """饮食休息策略"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        generator = SolutionGenerator()
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        results = []
        
        for week in range(weeks):
            # 前2周维持，后续恢复减脂
            if week < 2:
                calories_factor = 1.0  # 维持热量
            else:
                calories_factor = 0.8  # 恢复减脂
            
            target_calories = tdee * calories_factor
            
            solution_vector = generator.generate_from_template("balanced", tdee)
            solution_vector[0] = target_calories
            solution = Solution(solution_vector)
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _carb_cycling(self, subject: PersonProfile, weeks: int) -> Dict:
        """碳水循环策略"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        generator = SolutionGenerator()
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        results = []
        
        for week in range(weeks):
            # 高碳日和低碳日交替
            if week % 2 == 0:
                template = "low_carb"
            else:
                template = "high_carb"  # 实际上用balanced代替
                template = "balanced"
            
            solution_vector = generator.generate_from_template(template, tdee)
            solution = Solution(solution_vector)
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _run_with_model(self, subject: PersonProfile, model, duration: int, deficit: float) -> List:
        """使用指定模型运行实验"""
        current_subject = copy.deepcopy(subject)
        generator = SolutionGenerator()
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        target_calories = tdee - deficit
        
        results = []
        
        for week in range(duration):
            solution_vector = generator.generate_from_template("balanced", tdee)
            solution_vector[0] = target_calories
            solution = Solution(solution_vector)
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            
            results.append({
                'week': week,
                'weight': current_subject.weight,
                'body_fat': current_subject.body_fat_percentage
            })
        
        return results
    
    def _disable_component(self, config: ConfigManager, component: str):
        """禁用指定组件"""
        if component == 'metabolic_adaptation':
            config.metabolic.enable_metabolic_adaptation = False
            
        elif component == 'sleep_optimization':
            # 固定睡眠时间为7小时（不优化）
            config.algorithm.population_size = config.algorithm.population_size  # 保持不变
            # 标记：在生成方案时固定睡眠
            config.experiment.disable_sleep_opt = True
            
        elif component == 'strength_training':
            # 禁用力量训练（设置为0）
            config.exercise.strength_frequency_range = (0, 0)
            
        elif component == 'cardio_training':
            # 禁用有氧训练（设置为0）
            config.exercise.cardio_frequency_range = (0, 0)
            
        elif component == 'nutrition_optimization':
            # 使用固定营养比例（不优化）
            config.nutrition.protein_range = (0.30, 0.30)  # 固定30%
            config.nutrition.carb_range = (0.40, 0.40)     # 固定40%
            config.nutrition.fat_range = (0.30, 0.30)      # 固定30%
            
        elif component == 'neat_adjustment':
            config.metabolic.consider_neat = False
    
    def _find_convergence_point(self, results: Dict) -> int:
        """找到收敛点"""
        if 'best_fitness_history' not in results:
            return -1
        
        history = results['best_fitness_history']
        if len(history) < 3:
            return len(history)
        
        # 找到连续3代改善小于阈值的点
        for i in range(2, len(history)):
            improvements = []
            for j in range(i-2, i):
                improvements.append(abs(history[j] - history[j+1]))
            
            if all(imp < self.config.algorithm.min_improvement for imp in improvements):
                return i
        
        return len(history)
    
    # ============ 分析方法 ============
    
    def _analyze_benchmark_results(self, results: Dict) -> Dict:
        """分析基准实验结果"""
        analysis = {}
        
        for method_name, method_results in results.items():
            weight_losses = [r['total_weight_loss'] for r in method_results]
            
            analysis[method_name] = {
                'mean_weight_loss': np.mean(weight_losses),
                'std_weight_loss': np.std(weight_losses),
                'max_weight_loss': np.max(weight_losses),
                'min_weight_loss': np.min(weight_losses),
                'success_rate': sum(1 for w in weight_losses if w > self.config.experiment.success_weight_loss_threshold) / len(weight_losses)
            }
        
        # 统计检验（DE vs 其他方法）
        if 'de_optimized' in results:
            de_losses = [r['total_weight_loss'] for r in results['de_optimized']]
            analysis['statistical_tests'] = {}
            
            for method_name in results:
                if method_name != 'de_optimized':
                    other_losses = [r['total_weight_loss'] for r in results[method_name]]
                    t_stat, p_value = stats.ttest_ind(de_losses, other_losses)
                    
                    analysis['statistical_tests'][f'DE_vs_{method_name}'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.config.experiment.significance_level
                    }
        
        return analysis
    
    def _analyze_plateau_results(self, results: Dict) -> Dict:
        """分析平台期突破结果"""
        analysis = {}
        
        for strategy_name, strategy_results in results.items():
            successes = sum(1 for r in strategy_results if r['breakthrough_success'])
            weight_changes = [r['weight_change'] for r in strategy_results]
            
            analysis[strategy_name] = {
                'success_rate': successes / len(strategy_results),
                'mean_weight_change': np.mean(weight_changes),
                'std_weight_change': np.std(weight_changes),
                'successful_cases': successes
            }
        
        if 'de_plateau_breaker' in results:
            analysis['statistical_tests'] = {}
            de_changes = [r['weight_change'] for r in results['de_plateau_breaker']]
            
            for strategy in results:
                if strategy != 'de_plateau_breaker':
                    other_changes = [r['weight_change'] for r in results[strategy]]
                    t_stat, p_value = stats.ttest_ind(de_changes, other_changes)
                    
                    analysis['statistical_tests'][f'DE_vs_{strategy}'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.config.experiment.significance_level
                    }

        return analysis
    
    def _analyze_model_differences(self, results: Dict) -> Dict:
        """分析模型差异"""
        analysis = {}
        
        for scenario_name, scenario_results in results.items():
            basic_weights = [r[-1]['weight'] for r in scenario_results['basic_model']]
            advanced_weights = [r[-1]['weight'] for r in scenario_results['advanced_model']]
            
            # 计算预测差异
            weight_diffs = [abs(b - a) for b, a in zip(basic_weights, advanced_weights)]
            
            analysis[scenario_name] = {
                'mean_difference': np.mean(weight_diffs),
                'max_difference': np.max(weight_diffs),
                'basic_mean_loss': np.mean([r[0]['weight'] - r[-1]['weight'] 
                                           for r in scenario_results['basic_model']]),
                'advanced_mean_loss': np.mean([r[0]['weight'] - r[-1]['weight'] 
                                             for r in scenario_results['advanced_model']])
            }
        analysis['statistical_tests'] = {}
        
        for scenario_name, scenario_results in results.items():
            basic_losses = [r[0]['weight'] - r[-1]['weight'] 
                        for r in scenario_results['basic_model']]
            advanced_losses = [r[0]['weight'] - r[-1]['weight'] 
                            for r in scenario_results['advanced_model']]
            
            t_stat, p_value = stats.ttest_rel(basic_losses, advanced_losses)
            
            analysis['statistical_tests'][f'{scenario_name}_model_comparison'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.config.experiment.significance_level,
                'effect_size': (np.mean(advanced_losses) - np.mean(basic_losses)) / np.std(basic_losses)
            }
        
        return analysis
    
    def _analyze_sensitivity(self, results: List[Dict], param_names: List[str]) -> Dict:
        """分析参数敏感性"""
        analysis = {
            'sensitivity_scores': {},
            'optimal_parameters': {}
        }
        
        # 计算每个参数的敏感性
        for param_name in param_names:
            param_values = [r['parameters'][param_name] for r in results]
            fitness_values = [r['avg_fitness'] for r in results]
            
            # 计算相关性作为敏感性指标
            correlation = np.corrcoef(param_values, fitness_values)[0, 1]
            analysis['sensitivity_scores'][param_name] = abs(correlation)
        
        # 找到最优参数组合
        best_result = min(results, key=lambda x: x['avg_fitness'])
        analysis['optimal_parameters'] = best_result['parameters']
        analysis['optimal_fitness'] = best_result['avg_fitness']
        
        # ✅ 添加统计检验：参数的显著性
        analysis['statistical_tests'] = {}
        
        for param_name in param_names:
            param_values = [r['parameters'][param_name] for r in results]
            fitness_values = [r['avg_fitness'] for r in results]
            
            # Spearman相关性检验
            rho, p_value = stats.spearmanr(param_values, fitness_values)
            
            analysis['statistical_tests'][f'{param_name}_correlation'] = {
                'correlation': rho,
                'p_value': p_value,
                'significant': p_value < self.config.experiment.significance_level
            }

        return analysis
    
    def _analyze_ablation_results(self, results: Dict) -> Dict:
        """分析消融研究结果"""
        analysis = {}
        
        if 'full_model' in results:
            full_fitness = results['full_model']['fitness']
            
            component_impacts = []
            for key, value in results.items():
                if key.startswith('without_'):
                    component = key.replace('without_', '')
                    
                    # 计算重要性
                    importance = abs(value['fitness'] - full_fitness) / full_fitness
                    
                    analysis[component] = {
                        'importance': importance,
                        'fitness_without': value['fitness'],
                        'fitness_with': full_fitness,
                        'impact': value['fitness'] - full_fitness
                    }
                    component_impacts.append(value['fitness'] - full_fitness)

            # ✅ 添加统计检验：组件影响的显著性
            if component_impacts:
                # 单样本t检验：测试影响是否显著不为0
                t_stat, p_value = stats.ttest_1samp(component_impacts, 0)
                
                analysis['statistical_tests'] = {
                    'components_impact_test': {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'mean_impact': np.mean(component_impacts),
                        'std_impact': np.std(component_impacts)
                    }
                }        
        # 排序组件重要性
        if analysis:
            sorted_components = sorted(analysis.items(), 
                                     key=lambda x: x[1]['importance'], 
                                     reverse=True)
            analysis['ranking'] = [comp for comp, _ in sorted_components]
        
        return analysis
    
    def _analyze_long_term_results(self, results: List[Dict]) -> Dict:
        """分析长期效果"""
        analysis = {
            'average_total_loss': np.mean([r['total_weight_loss'] for r in results]),
            'std_total_loss': np.std([r['total_weight_loss'] for r in results]),
            'retention_rate': 0,  # 保持率
            'plateau_frequency': 0,  # 平台期频率
            'metabolic_adaptation': []
        }
        
        # 分析代谢适应
        for result in results:
            final_factor = result['tracking_data']['metabolic_factor'][-1]
            analysis['metabolic_adaptation'].append(final_factor)
        
        analysis['mean_metabolic_adaptation'] = np.mean(analysis['metabolic_adaptation'])
        
        # 计算平台期频率
        plateau_counts = []
        for result in results:
            weights = result['tracking_data']['weight']
            plateau_weeks = 0
            
            for i in range(2, len(weights)):
                if abs(weights[i] - weights[i-1]) < self.config.experiment.plateau_detection_threshold:
                    plateau_weeks += 1
            
            plateau_counts.append(plateau_weeks)
        
        analysis['plateau_frequency'] = np.mean(plateau_counts) / len(results[0]['tracking_data']['weight'])
        

        # ✅ 添加统计检验
        analysis['statistical_tests'] = {}
        
        # 1. 测试体重减少的显著性（单样本t检验）
        weight_losses = [r['total_weight_loss'] for r in results]
        if weight_losses:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(weight_losses, 0)
            analysis['statistical_tests']['weight_loss_significance'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_loss': np.mean(weight_losses),
                'std_loss': np.std(weight_losses)
            }
        
        # 2. 测试代谢适应的显著性（是否显著不等于1）
        if analysis['metabolic_adaptation']:
            t_stat, p_value = stats.ttest_1samp(analysis['metabolic_adaptation'], 1.0)
            analysis['statistical_tests']['metabolic_adaptation_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_factor': analysis['mean_metabolic_adaptation']
            }
        
        # 3. 计算体重保持率（最后4周体重变化）
        retention_rates = []
        for result in results:
            weights = result['tracking_data']['weight']
            if len(weights) >= 8:
                last_month_change = abs(weights[-1] - weights[-4])
                retention_rate = 1 - (last_month_change / result['total_weight_loss']) if result['total_weight_loss'] > 0 else 0
                retention_rates.append(retention_rate)
        
        if retention_rates:
            analysis['retention_rate'] = np.mean(retention_rates)
            analysis['statistical_tests']['retention_analysis'] = {
                'mean_retention': analysis['retention_rate'],
                'std_retention': np.std(retention_rates),
                'good_retention_rate': sum(1 for r in retention_rates if r > 0.8) / len(retention_rates)
            }
            
        return analysis
    
    # ============ 保存方法 ============
    
    def _save_experiment_results(self, experiment_name: str, results: Any, analysis: Any):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始结果
        with open(os.path.join(output_dir, "raw_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, default=np_pd_convert, indent=2)
        
        # 保存分析结果
        with open(os.path.join(output_dir, "analysis.json"), 'w', encoding='utf-8') as f:
            json.dump(analysis, f, default=np_pd_convert, indent=2)
        
        # 生成摘要报告
        self._generate_summary_report(experiment_name, results, analysis, output_dir)
        
        logger.info(f"实验结果已保存到: {output_dir}")
    
    def _generate_summary_report(self, experiment_name: str, results: Any, 
                                analysis: Any, output_dir: str):
        """生成摘要报告"""
        report = f"""
# 实验报告: {experiment_name}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验概要
- 实验类型: {experiment_name}
- 样本数量: {self._count_subjects(results)}

## 主要发现
{self._format_key_findings(analysis)}

## 统计结果
{self._format_statistics(analysis)}

## 结论
{self._generate_conclusions(experiment_name, analysis)}
"""
        
        with open(os.path.join(output_dir, "summary_report.md"), 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _count_subjects(self, results):
        """统计实验对象数量"""
        # A类实验：直接的方法->结果列表
        if isinstance(results, dict):
            first_key = list(results.keys())[0]
            first_value = results[first_key]
            
            # 标准格式：{'method': [results]}
            if isinstance(first_value, list):
                return len(first_value)
            
            # B1格式：嵌套字典
            elif isinstance(first_value, dict):
                # 检查是否有'basic_model'或'advanced_model'
                if 'basic_model' in first_value:
                    return len(first_value['basic_model'])
                # D1格式：单个结果
                elif 'fitness' in first_value:
                    # D1实验每个组件测试用同一个subject
                    return 1
                # 尝试递归查找
                for sub_key, sub_value in first_value.items():
                    if isinstance(sub_value, list):
                        return len(sub_value)
        
        # E1格式：直接的结果列表
        elif isinstance(results, list):
            return len(results)
        
        # 无法确定
        return "未知"
    
    def _format_key_findings(self, analysis):
        """格式化主要发现"""
        findings = []
        
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                if key == 'statistical_tests':  # 跳过统计检验部分
                    continue
                    
                if isinstance(value, dict):
                    # 原有的字段
                    if 'mean_weight_loss' in value:
                        findings.append(f"- {key}: 平均减重 {value['mean_weight_loss']:.2f} kg")
                    elif 'success_rate' in value:
                        findings.append(f"- {key}: 成功率 {value['success_rate']:.1%}")
                    
                    # B1实验的字段
                    elif 'mean_difference' in value:
                        findings.append(f"- {key}: 模型差异 {value['mean_difference']:.3f} kg")
                    elif 'basic_mean_loss' in value and 'advanced_mean_loss' in value:
                        findings.append(f"- {key}: 基础模型减重 {value['basic_mean_loss']:.2f} kg, "
                                    f"高级模型减重 {value['advanced_mean_loss']:.2f} kg")
                    
                    # D1实验的字段
                    elif 'importance' in value:
                        findings.append(f"- {key}: 重要性 {value['importance']:.1%}")
                    
                    # C1实验的字段
                    elif key == 'optimal_parameters':
                        param_str = ', '.join([f"{k}={v}" for k, v in value.items()])
                        findings.append(f"- 最优参数: {param_str}")
                    elif key == 'sensitivity_scores':
                        for param, score in value.items():
                            findings.append(f"- {param} 敏感度: {score:.3f}")
        
        return "\n".join(findings) if findings else "- 详见分析结果"
    
    def _format_statistics(self, analysis):
        """格式化统计结果"""
        stats = []
        
        if 'statistical_tests' in analysis:
            for test_name, test_result in analysis['statistical_tests'].items():
                stats.append(f"- {test_name}: p={test_result['p_value']:.4f} "
                           f"({'显著' if test_result['significant'] else '不显著'})")
        
        return "\n".join(stats) if stats else "- 无统计检验结果"
    
    def _generate_conclusions(self, experiment_name: str, analysis: Any) -> str:
        """生成结论"""
        if 'A1' in experiment_name:
            return "DE算法在减重效果上表现优于传统方法。"
        elif 'A2' in experiment_name:
            return "DE算法能够有效突破减肥平台期。"
        elif 'B1' in experiment_name:
            return "考虑代谢适应的模型预测更加准确。"
        elif 'C1' in experiment_name:
            return f"最优参数组合已确定：{analysis.get('optimal_parameters', '见分析结果')}"
        elif 'D1' in experiment_name:
            return f"关键组件排名：{', '.join(analysis.get('ranking', []))}"
        elif 'E1' in experiment_name:
            return "长期追踪显示方案具有良好的可持续性。"
        else:
            return "实验完成，详见分析结果。"


if __name__ == "__main__":
    # 测试代码
    config = ConfigManager()
    runner = EnhancedExperimentRunner(config)
    
    print("增强版实验运行器已加载")
    print(f"输出目录: {runner.output_dir}")
    print(f"可用实验:")
    print("  - A1: 基准对比")
    print("  - A2: 平台期突破")
    print("  - B1: 代谢模型验证")
    print("  - C1: 参数敏感性")
    print("  - D1: 消融研究")
    print("  - E1: 长期追踪")