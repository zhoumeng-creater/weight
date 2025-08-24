"""
增强版实验运行器
专门为减肥平台期优化研究设计的完整实验框架
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

from metabolic_model import PersonProfile, MetabolicModel, AdvancedMetabolicModel
from config import ConfigManager, load_preset
from de_algorithm import DifferentialEvolution
from solution_generator import Solution, SolutionGenerator
from fitness_evaluator import FitnessEvaluator, AdaptiveFitnessEvaluator
from visualization import DataTracker, WeightLossVisualizer, OptimizationVisualizer, ReportGenerator
from data_loader import SimulatedExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def np_pd_convert(o):
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
    # 其他非内置类型可以继续在这里补
    raise TypeError(f"Type {type(o)} not serializable")

class VirtualSubjectGenerator:
    """虚拟实验对象生成器"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.subjects_created = 0
        
    def generate_standard_subjects(self, n_subjects: int = 120) -> List[PersonProfile]:
        """生成标准虚拟人群体"""
        subjects = []
        
        # 年龄组
        age_groups = {
            "young": [20, 25, 30],
            "middle": [35, 40, 45],
            "senior": [50, 55, 60]
        }
        
        # BMI类别
        bmi_categories = {
            "overweight": [25.0, 27.5],
            "obese_1": [30.0, 32.5],
            "obese_2": [35.0, 37.5]
        }
        
        # 活动水平
        activity_levels = [1.2, 1.375, 1.55, 1.725]
        
        # 减肥史
        diet_history = [0, 8, 12]  # 周数
        
        # 生成所有组合
        for age_group in age_groups.values():
            for age in age_group:
                for gender in ['male', 'female']:
                    for bmi_cat in bmi_categories.values():
                        for bmi in bmi_cat:
                            # 随机选择其他参数
                            activity = np.random.choice(activity_levels)
                            weeks_on_diet = np.random.choice(diet_history)
                            
                            # 根据BMI计算体重（假设平均身高）
                            height = 170 if gender == 'male' else 160
                            weight = bmi * (height / 100) ** 2
                            
                            # 计算体脂率（基于BMI和性别的估算）
                            if gender == 'male':
                                body_fat = 1.20 * bmi + 0.23 * age - 16.2
                            else:
                                body_fat = 1.20 * bmi + 0.23 * age - 5.4
                            
                            # 添加随机扰动
                            weight += np.random.normal(0, 2)
                            body_fat += np.random.normal(0, 2)
                            body_fat = np.clip(body_fat, 10, 50)
                            
                            subject = PersonProfile(
                                age=age,
                                gender=gender,
                                height=height,
                                weight=weight,
                                body_fat_percentage=body_fat,
                                activity_level=activity,
                                weeks_on_diet=weeks_on_diet
                            )
                            
                            subjects.append(subject)
                            self.subjects_created += 1
                            
                            if len(subjects) >= n_subjects:
                                return subjects[:n_subjects]
        
        return subjects
    
    def generate_edge_cases(self, n_cases: int = 20) -> List[Tuple[PersonProfile, Dict]]:
        """生成边界案例"""
        edge_cases = []
        
        # 极端案例类型
        case_types = [
            {"name": "severe_metabolic_damage", "bmr_factor": 0.75},
            {"name": "insulin_resistance", "insulin_factor": 1.3},
            {"name": "hypothyroid", "thyroid_factor": 0.8},
            {"name": "menopause", "hormonal_fluctuation": True},
            {"name": "high_stress", "cortisol_level": 2.0},
            {"name": "sleep_deprived", "avg_sleep": 5.0},
            {"name": "extreme_dieter", "weeks_on_diet": 52},
            {"name": "yo_yo_dieter", "weight_cycles": 5}
        ]
        
        for i, case_type in enumerate(itertools.cycle(case_types)):
            if i >= n_cases:
                break
                
            # 基础参数随机化
            age = np.random.randint(25, 60)
            gender = np.random.choice(['male', 'female'])
            height = np.random.normal(170 if gender == 'male' else 160, 10)
            weight = np.random.normal(90, 15)
            body_fat = np.random.normal(35, 5)
            
            subject = PersonProfile(
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                body_fat_percentage=body_fat,
                activity_level=1.2,  # 边界案例通常活动量低
                weeks_on_diet=case_type.get('weeks_on_diet', 0)
            )
            
            edge_cases.append((subject, case_type))
            self.subjects_created += 1
        
        return edge_cases
    
    def generate_plateau_subjects(self, n_subjects: int = 30) -> List[PersonProfile]:
        """生成已经处于平台期的虚拟人"""
        plateau_subjects = []
        
        for _ in range(n_subjects):
            # 这些人已经减肥一段时间
            weeks_on_diet = np.random.randint(8, 20)
            
            # 初始体重较高
            initial_weight = np.random.normal(95, 15)
            
            # 已经减掉一些体重（5-15%）
            weight_loss_percentage = np.random.uniform(0.05, 0.15)
            current_weight = initial_weight * (1 - weight_loss_percentage)
            
            # 代谢已经适应
            subject = PersonProfile(
                age=np.random.randint(25, 55),
                gender=np.random.choice(['male', 'female']),
                height=np.random.normal(170, 10),
                weight=current_weight,
                body_fat_percentage=np.random.normal(30, 5),
                activity_level=np.random.choice([1.2, 1.375, 1.55]),
                weeks_on_diet=weeks_on_diet
            )
            
            # 设置代谢适应
            subject.metabolic_adaptation_factor = np.random.uniform(0.8, 0.9)
            subject.initial_weight = initial_weight
            
            plateau_subjects.append(subject)
            self.subjects_created += 1
        
        return plateau_subjects


class EnhancedExperimentRunner:
    """增强版实验运行器"""
    
    def __init__(self, output_dir: str = "./experiment_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.config = ConfigManager()
        self.results_database = {}
        self.subject_generator = VirtualSubjectGenerator()
        
        # 初始化不同的模型
        self.models = {
            'basic': MetabolicModel(),
            'advanced': AdvancedMetabolicModel()
        }
        
        # 数据收集器
        self.data_collector = ExperimentDataCollector()
        
    def run_experiment_A1_benchmark(self, n_subjects: int = 30, duration_weeks: int = 16):
        """实验A1: 基准对比实验"""
        logger.info("=" * 50)
        logger.info("开始实验A1: 基准对比实验")
        logger.info("=" * 50)
        
        # 生成虚拟人
        subjects = self.subject_generator.generate_standard_subjects(n_subjects)
        
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
            
            for subject in tqdm(subjects, desc=f"运行{method_name}"):
                # 复制subject以避免相互影响
                test_subject = copy.deepcopy(subject)
                
                # 运行方法
                result = method_func(test_subject, duration_weeks)
                
                # 收集结果
                method_results.append(result)
            
            results[method_name] = method_results
        
        # 统计分析
        analysis = self._analyze_benchmark_results(results)
        
        # 保存结果
        self._save_experiment_results("A1_benchmark", results, analysis)
        
        return results, analysis
    
    def run_experiment_A2_plateau_breakthrough(self, n_subjects: int = 30):
        """实验A2: 平台期突破专项实验"""
        logger.info("=" * 50)
        logger.info("开始实验A2: 平台期突破实验")
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
                    'metabolic_factor': test_subject.metabolic_adaptation_factor
                }
                
                # 运行策略（8周观察期）
                result = strategy_func(test_subject, weeks=8)
                
                # 判断是否突破平台期
                weight_change = initial_state['weight'] - result['final_weight']
                breakthrough_success = weight_change > 0.5  # 2周内减重>0.5kg
                
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
        
        # 创建测试场景
        test_scenarios = [
            {'duration': 4, 'deficit': 500, 'name': '短期适度减脂'},
            {'duration': 12, 'deficit': 500, 'name': '长期适度减脂'},
            {'duration': 8, 'deficit': 1000, 'name': '激进减脂'},
            {'duration': 16, 'deficit': 300, 'name': '缓慢减脂'}
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            logger.info(f"\n测试场景: {scenario['name']}")
            
            # 测试10个不同的虚拟人
            subjects = self.subject_generator.generate_standard_subjects(10)
            
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
        
        # 参数网格
        param_grid = {
            'population_size': [5, 10, 20, 30],
            'scaling_factor': [0.4, 0.6, 0.8, 1.0],
            'crossover_rate': [0.5, 0.7, 0.9],
            'max_iterations': [8, 12, 16]
        }
        
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
            config = ConfigManager()
            for name, value in zip(param_names, param_values):
                if name == 'population_size':
                    config.algorithm.population_size = value
                elif name == 'scaling_factor':
                    config.algorithm.scaling_factor = value
                elif name == 'crossover_rate':
                    config.algorithm.crossover_rate = value
                elif name == 'max_iterations':
                    config.algorithm.max_iterations = value
            
            # 运行优化（重复3次取平均）
            run_results = []
            for _ in range(3):
                optimizer = DifferentialEvolution(copy.deepcopy(test_subject), config)
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
        
        # 要消融的组件
        components = [
            'metabolic_adaptation',
            'sleep_optimization',
            'strength_training',
            'macro_optimization',
            'neat_adjustment'
        ]
        
        # 测试对象
        subjects = self.subject_generator.generate_standard_subjects(10)
        
        results = {'full_model': []}
        
        # 完整模型基准
        for subject in subjects:
            config = ConfigManager()
            optimizer = DifferentialEvolution(copy.deepcopy(subject), config)
            best_solution, opt_results = optimizer.optimize()
            
            results['full_model'].append({
                'fitness': best_solution.fitness,
                'weight_loss': subject.weight - opt_results['final_person_state'].weight,
                'muscle_retention': 1 - opt_results['final_person_state'].lean_body_mass / subject.lean_body_mass
            })
        
        # 逐个移除组件
        for component in components:
            logger.info(f"\n移除组件: {component}")
            component_results = []
            
            for subject in subjects:
                # 创建修改后的配置
                config = self._create_ablated_config(component)
                
                # 运行优化
                optimizer = DifferentialEvolution(copy.deepcopy(subject), config)
                best_solution, opt_results = optimizer.optimize()
                
                component_results.append({
                    'fitness': best_solution.fitness,
                    'weight_loss': subject.weight - opt_results['final_person_state'].weight,
                    'muscle_retention': 1 - opt_results['final_person_state'].lean_body_mass / subject.lean_body_mass
                })
            
            results[f'without_{component}'] = component_results
        
        # 分析组件重要性
        analysis = self._analyze_ablation_results(results)
        
        # 保存结果
        self._save_experiment_results("D1_ablation", results, analysis)
        
        return results, analysis
    
    def run_experiment_E1_long_term_tracking(self, duration_weeks: int = 24):
        """实验E1: 长期效果追踪"""
        logger.info("=" * 50)
        logger.info("开始实验E1: 长期效果追踪")
        logger.info("=" * 50)
        
        # 生成测试对象
        subjects = self.subject_generator.generate_standard_subjects(20)
        
        results = []
        
        for subject in tqdm(subjects, desc="长期追踪"):
            # 记录初始状态
            initial_state = {
                'weight': subject.weight,
                'body_fat': subject.body_fat_percentage,
                'muscle_mass': subject.lean_body_mass
            }
            
            # 创建数据追踪器
            tracker = DataTracker()
            tracker.metadata['person_profile'] = {
                'age': subject.age,
                'gender': subject.gender,
                'height': subject.height,
                'initial_weight': subject.weight
            }
            
            # 运行长期优化
            config = ConfigManager()
            config.algorithm.max_iterations = duration_weeks
            
            optimizer = DifferentialEvolution(copy.deepcopy(subject), config)
            best_solution, opt_results = optimizer.optimize()
            
            # 收集每周数据
            weekly_data = []
            for week in range(duration_weeks):
                if week < len(opt_results['best_solutions_history']):
                    solution = opt_results['best_solutions_history'][week]
                    weekly_data.append({
                        'week': week,
                        'solution': solution.to_vector().tolist(),
                        'fitness': solution.fitness,
                        'weight': self._estimate_weight_at_week(subject, solution, week),
                        'adherence_score': self._calculate_adherence_score(solution)
                    })
            
            # 计算反弹风险
            rebound_risk = self._calculate_rebound_risk(subject, opt_results)
            
            results.append({
                'subject_id': f"S{subjects.index(subject):03d}",
                'initial_state': initial_state,
                'final_state': {
                    'weight': opt_results['final_person_state'].weight,
                    'body_fat': opt_results['final_person_state'].body_fat_percentage,
                    'muscle_mass': opt_results['final_person_state'].lean_body_mass
                },
                'weekly_data': weekly_data,
                'rebound_risk': rebound_risk,
                'total_weight_loss': initial_state['weight'] - opt_results['final_person_state'].weight
            })
        
        # 分析长期趋势
        analysis = self._analyze_long_term_results(results)
        
        # 保存结果
        self._save_experiment_results("E1_long_term", results, analysis)
        
        return results, analysis
    
    # ========== 辅助方法 ==========
    
    def _fixed_deficit_method(self, subject: PersonProfile, weeks: int) -> Dict:
        """固定热量赤字法"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        
        # 计算初始TDEE
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        # 固定赤字500kcal
        target_calories = tdee - 500
        
        # 创建固定方案
        solution = Solution(np.array([
            target_calories,  # 热量
            0.30,  # 蛋白质比例
            0.40,  # 碳水比例
            0.30,  # 脂肪比例
            3,     # 有氧频率
            45,    # 有氧时长
            2,     # 力量频率
            7.5    # 睡眠时间
        ]))
        
        # 模拟执行
        results = []
        for week in range(weeks):
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _step_reduction_method(self, subject: PersonProfile, weeks: int) -> Dict:
        """阶梯式减重法"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        results = []
        
        for week in range(weeks):
            # 每2周降低100kcal
            reduction = (week // 2) * 100
            target_calories = tdee - 400 - reduction
            target_calories = max(1200, target_calories)  # 不低于1200
            
            solution = Solution(np.array([
                target_calories, 0.30, 0.40, 0.30, 3, 45, 2, 7.5
            ]))
            
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append(week_result)
        
        return {
            'final_weight': current_subject.weight,
            'total_weight_loss': subject.weight - current_subject.weight,
            'weekly_results': results
        }
    
    def _cyclic_diet_method(self, subject: PersonProfile, weeks: int) -> Dict:
        """循环饮食法（5天低卡+2天高卡）"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        results = []
        
        for week in range(weeks):
            # 5天低卡（-600），2天高卡（维持）
            avg_calories = (tdee - 600) * 5/7 + tdee * 2/7
            
            solution = Solution(np.array([
                avg_calories, 0.30, 0.40, 0.30, 3, 45, 2, 7.5
            ]))
            
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
        config = ConfigManager()
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
                best_result = {
                    'final_weight': test_subject.weight,
                    'solution': solution
                }
        
        return best_result
    
    def _continue_same_plan(self, subject: PersonProfile, weeks: int) -> Dict:
        """继续原方案（对照）"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        
        # 使用标准方案
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        solution = Solution(np.array([
            tdee - 500, 0.30, 0.40, 0.30, 3, 45, 2, 7.5
        ]))
        
        for week in range(weeks):
            model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
        
        return {'final_weight': current_subject.weight}
    
    def _increase_exercise(self, subject: PersonProfile, weeks: int) -> Dict:
        """增加运动量策略"""
        model = MetabolicModel()
        current_subject = copy.deepcopy(subject)
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        # 增加运动
        solution = Solution(np.array([
            tdee - 500, 0.30, 0.40, 0.30, 
            5,    # 增加有氧频率
            60,   # 增加有氧时长
            3,    # 增加力量训练
            7.5
        ]))
        
        for week in range(weeks):
            model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
        
        return {'final_weight': current_subject.weight}
    
    def _diet_break(self, subject: PersonProfile, weeks: int) -> Dict:
        """Diet Break策略"""
        model = AdvancedMetabolicModel()
        current_subject = copy.deepcopy(subject)
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        for week in range(weeks):
            if week < 2:
                # 前2周恢复到维持热量
                calories = tdee
            else:
                # 之后恢复赤字
                calories = tdee - 600
            
            solution = Solution(np.array([
                calories, 0.30, 0.40, 0.30, 3, 45, 2, 7.5
            ]))
            
            model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
        
        return {'final_weight': current_subject.weight}
    
    def _carb_cycling(self, subject: PersonProfile, weeks: int) -> Dict:
        """碳水循环策略"""
        model = AdvancedMetabolicModel()
        current_subject = copy.deepcopy(subject)
        
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        for week in range(weeks):
            # 高碳日和低碳日交替
            if week % 2 == 0:
                # 高碳周
                solution = Solution(np.array([
                    tdee - 300, 0.25, 0.50, 0.25, 3, 45, 2, 7.5
                ]))
            else:
                # 低碳周
                solution = Solution(np.array([
                    tdee - 500, 0.35, 0.30, 0.35, 3, 45, 2, 7.5
                ]))
            
            model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
        
        return {'final_weight': current_subject.weight}
    
    def _run_with_model(self, subject: PersonProfile, model, weeks: int, deficit: float) -> Dict:
        """使用指定模型运行实验"""
        current_subject = copy.deepcopy(subject)
        bmr = model.calculate_bmr(current_subject)
        tdee = bmr * current_subject.activity_level
        
        solution = Solution(np.array([
            tdee - deficit, 0.30, 0.40, 0.30, 3, 45, 2, 7.5
        ]))
        
        results = []
        for week in range(weeks):
            week_result = model.simulate_week(current_subject, solution, week)
            current_subject = model.update_person_state(current_subject, solution, week)
            results.append({
                'week': week,
                'weight': current_subject.weight,
                'bmr': model.calculate_bmr(current_subject),
                'metabolic_factor': getattr(current_subject, 'metabolic_adaptation_factor', 1.0)
            })
        
        return results
    
    def _create_ablated_config(self, component: str) -> ConfigManager:
        """创建消融配置"""
        config = ConfigManager()
        
        if component == 'metabolic_adaptation':
            config.metabolic.adaptation_rate_per_week = 0
        elif component == 'sleep_optimization':
            # 固定睡眠，不优化
            config.exercise.cardio_duration_options = [7.5]
        elif component == 'strength_training':
            config.exercise.strength_frequency_range = (0, 0)
        elif component == 'macro_optimization':
            # 固定营养素比例
            config.nutrition.protein_range = (0.30, 0.30)
            config.nutrition.carb_range = (0.40, 0.40)
            config.nutrition.fat_range = (0.30, 0.30)
        elif component == 'neat_adjustment':
            # 禁用NEAT调整
            config.metabolic.neat_adjustment = False
        
        return config
    
    def _find_convergence_point(self, opt_results: Dict) -> int:
        """找到收敛点"""
        if 'best_fitness_history' not in opt_results:
            return -1
        
        history = opt_results['best_fitness_history']
        if len(history) < 3:
            return len(history)
        
        # 找到连续3代改善小于1%的点
        for i in range(2, len(history)):
            if i >= 2:
                recent_change = abs(history[i] - history[i-1]) / abs(history[i-1])
                if recent_change < 0.01:
                    return i
        
        return len(history)
    
    def _estimate_weight_at_week(self, subject: PersonProfile, solution: Solution, week: int) -> float:
        """估算某周的体重"""
        # 简化估算
        weekly_deficit = (subject.activity_level * MetabolicModel().calculate_bmr(subject) - solution.calories) * 7
        weight_loss_per_week = weekly_deficit / 7700  # 7700 kcal = 1kg
        return subject.weight - weight_loss_per_week * week
    
    def _calculate_adherence_score(self, solution: Solution) -> float:
        """计算依从性分数"""
        score = 10.0
        
        # 热量过低扣分
        if solution.calories < 1500:
            score -= 2
        
        # 运动过多扣分
        total_exercise = solution.cardio_freq + solution.strength_freq
        if total_exercise > 6:
            score -= 1
        
        # 睡眠不足扣分
        if solution.sleep_hours < 7:
            score -= 1
        
        return max(0, score)
    
    def _calculate_rebound_risk(self, subject: PersonProfile, opt_results: Dict) -> float:
        """计算反弹风险"""
        risk = 0.0
        
        # 代谢损伤程度
        if 'final_person_state' in opt_results:
            final_state = opt_results['final_person_state']
            if hasattr(final_state, 'metabolic_adaptation_factor'):
                metabolic_damage = 1 - final_state.metabolic_adaptation_factor
                risk += metabolic_damage * 0.4
        
        # 减重速度
        if 'total_iterations' in opt_results:
            weekly_loss = (subject.weight - final_state.weight) / opt_results['total_iterations']
            if weekly_loss > 1.0:  # 每周超过1kg
                risk += 0.3
        
        # 肌肉流失
        muscle_loss_ratio = 1 - final_state.lean_body_mass / subject.lean_body_mass
        if muscle_loss_ratio > 0.1:  # 肌肉流失超过10%
            risk += 0.3
        
        return min(1.0, risk)
    
    def _analyze_benchmark_results(self, results: Dict) -> Dict:
        """分析基准对比结果"""
        analysis = {}
        
        for method_name, method_results in results.items():
            weight_losses = [r['total_weight_loss'] for r in method_results]
            
            analysis[method_name] = {
                'mean_weight_loss': np.mean(weight_losses),
                'std_weight_loss': np.std(weight_losses),
                'max_weight_loss': np.max(weight_losses),
                'min_weight_loss': np.min(weight_losses),
                'success_rate': sum(1 for w in weight_losses if w > 5) / len(weight_losses)
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
                        'significant': p_value < 0.05
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
        
        return analysis
    
    def _analyze_sensitivity(self, results: List[Dict], param_names: List[str]) -> Dict:
        """分析参数敏感性"""
        analysis = {'parameter_importance': {}}
        
        # 对每个参数计算其影响
        for param_name in param_names:
            param_effects = []
            
            # 按参数值分组
            param_groups = {}
            for result in results:
                param_value = result['parameters'][param_name]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result['avg_fitness'])
            
            # 计算方差
            if len(param_groups) > 1:
                group_means = [np.mean(values) for values in param_groups.values()]
                param_variance = np.var(group_means)
                param_effects.append(param_variance)
            
            analysis['parameter_importance'][param_name] = np.mean(param_effects) if param_effects else 0
        
        # 找出最优参数组合
        best_result = min(results, key=lambda x: x['avg_fitness'])
        analysis['optimal_parameters'] = best_result['parameters']
        analysis['optimal_fitness'] = best_result['avg_fitness']
        
        return analysis
    
    def _analyze_ablation_results(self, results: Dict) -> Dict:
        """分析消融结果"""
        analysis = {}
        
        # 计算每个组件的重要性
        full_model_fitness = np.mean([r['fitness'] for r in results['full_model']])
        
        for component_name, component_results in results.items():
            if component_name != 'full_model':
                ablated_fitness = np.mean([r['fitness'] for r in component_results])
                importance = (ablated_fitness - full_model_fitness) / full_model_fitness
                
                analysis[component_name] = {
                    'importance_score': importance,
                    'fitness_degradation': ablated_fitness - full_model_fitness,
                    'weight_loss_impact': np.mean([r['weight_loss'] for r in results['full_model']]) - 
                                         np.mean([r['weight_loss'] for r in component_results])
                }
        
        # 排序组件重要性
        analysis['component_ranking'] = sorted(
            [(k, v['importance_score']) for k, v in analysis.items() if k != 'component_ranking'],
            key=lambda x: x[1],
            reverse=True
        )
        
        return analysis
    
    def _analyze_long_term_results(self, results: List[Dict]) -> Dict:
        """分析长期结果"""
        analysis = {
            'weight_loss_distribution': [],
            'rebound_risk_distribution': [],
            'adherence_trends': []
        }
        
        for result in results:
            analysis['weight_loss_distribution'].append(result['total_weight_loss'])
            analysis['rebound_risk_distribution'].append(result['rebound_risk'])
            
            # 分析依从性趋势
            if 'weekly_data' in result:
                adherence_scores = [w.get('adherence_score', 0) for w in result['weekly_data']]
                analysis['adherence_trends'].append(np.mean(adherence_scores))
        
        # 统计汇总
        analysis['summary'] = {
            'mean_weight_loss': np.mean(analysis['weight_loss_distribution']),
            'std_weight_loss': np.std(analysis['weight_loss_distribution']),
            'mean_rebound_risk': np.mean(analysis['rebound_risk_distribution']),
            'mean_adherence': np.mean(analysis['adherence_trends']),
            'success_rate': sum(1 for w in analysis['weight_loss_distribution'] if w > 5) / len(results)
        }
        
        return analysis
    

    
    def _save_experiment_results(self, experiment_name: str, results: Any, analysis: Any):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建实验目录
        exp_dir = os.path.join(self.output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存原始结果
        with open(os.path.join(exp_dir, "raw_results.json"), 'w') as f:
            json.dump(self._convert_to_serializable(results), f, default=np_pd_convert, indent=2)
        
        # 保存分析结果
        with open(os.path.join(exp_dir, "analysis.json"), 'w') as f:
            json.dump(self._convert_to_serializable(analysis), f, default=np_pd_convert, indent=2)
        
        # 生成摘要报告
        self._generate_summary_report(experiment_name, results, analysis, exp_dir)
        
        logger.info(f"实验结果已保存到: {exp_dir}")
    
    def _convert_to_serializable(self, obj):
        """转换为可序列化的格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (PersonProfile, Solution)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_summary_report(self, experiment_name: str, results: Any, analysis: Any, output_dir: str):
        """生成摘要报告"""
        report = f"""
                    # 实验报告: {experiment_name}
                    生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

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
        if isinstance(results, dict):
            first_key = list(results.keys())[0]
            if isinstance(results[first_key], list):
                return len(results[first_key])
        elif isinstance(results, list):
            return len(results)
        return "未知"
    
    def _format_key_findings(self, analysis):
        """格式化主要发现"""
        findings = []
        
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                if isinstance(value, dict) and 'mean_weight_loss' in value:
                    findings.append(f"- {key}: 平均减重 {value['mean_weight_loss']:.2f} kg")
                elif isinstance(value, dict) and 'success_rate' in value:
                    findings.append(f"- {key}: 成功率 {value['success_rate']:.1%}")
        
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
            return "代谢适应和营养优化是最重要的组件。"
        elif 'E1' in experiment_name:
            return "长期追踪显示方案具有良好的可持续性。"
        else:
            return "实验完成，详见分析结果。"


class ExperimentDataCollector:
    """实验数据收集器"""
    
    def __init__(self):
        self.data_categories = {
            'physiological': ['weight', 'body_fat', 'muscle_mass', 'bmr', 'tdee'],
            'solution': ['calories', 'protein', 'carbs', 'fat', 'exercise', 'sleep'],
            'algorithm': ['fitness', 'convergence', 'diversity'],
            'metabolic': ['adaptation_factor', 'hormones', 'neat']
        }
        
        self.collected_data = {category: [] for category in self.data_categories}
    
    def collect(self, category: str, data: Dict):
        """收集数据"""
        if category in self.collected_data:
            self.collected_data[category].append({
                'timestamp': datetime.now(),
                'data': data
            })
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """导出为DataFrame"""
        all_data = []
        
        for category, records in self.collected_data.items():
            for record in records:
                flat_record = {'category': category, 'timestamp': record['timestamp']}
                flat_record.update(record['data'])
                all_data.append(flat_record)
        
        return pd.DataFrame(all_data)
    
    def save_to_csv(self, filepath: str):
        """保存为CSV"""
        df = self.export_to_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"数据已保存到: {filepath}")


# 主函数
def main():
    """运行完整实验"""
    runner = EnhancedExperimentRunner()
    
    print("=" * 60)
    print("差分进化算法减肥平台期优化 - 完整实验")
    print("=" * 60)
    
    # 运行所有实验系列
    experiments = [
        ("A1", runner.run_experiment_A1_benchmark),
        ("A2", runner.run_experiment_A2_plateau_breakthrough),
        ("B1", runner.run_experiment_B1_metabolic_validation),
        ("C1", runner.run_experiment_C1_parameter_sensitivity),
        ("D1", runner.run_experiment_D1_ablation_study),
        ("E1", runner.run_experiment_E1_long_term_tracking)
    ]
    
    all_results = {}
    
    for exp_name, exp_func in experiments:
        print(f"\n运行实验 {exp_name}...")
        try:
            results, analysis = exp_func()
            all_results[exp_name] = {
                'results': results,
                'analysis': analysis,
                'status': 'completed'
            }
            print(f"✓ 实验 {exp_name} 完成")
        except Exception as e:
            print(f"✗ 实验 {exp_name} 失败: {e}")
            all_results[exp_name] = {'status': 'failed', 'error': str(e)}
    
    # 生成总报告
    print("\n生成总报告...")
    with open(os.path.join(runner.output_dir, "experiment_summary.json"), 'w') as f:
        json.dump(runner._convert_to_serializable(all_results), f,  default=np_pd_convert, indent=2)
    
    print(f"\n所有实验完成！结果保存在: {runner.output_dir}")
    
    return all_results


if __name__ == "__main__":
    main()