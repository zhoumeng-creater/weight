"""
实验运行示例
展示如何使用系统进行不同类型的实验
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime
import os
import copy

from metabolic_model import PersonProfile, MetabolicModel, AdvancedMetabolicModel
from config import ConfigManager, load_preset
from de_algorithm import DifferentialEvolution
from visualization import (
    DataTracker, 
    WeightLossVisualizer, 
    OptimizationVisualizer,
    ReportGenerator
)
from data_loader import ExperimentDataLoader, SimulatedExperiment, ExperimentDesigner
from solution_generator import Solution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 替换原有的字体设置代码
from font_manager import setup_chinese_font

# 设置中文字体
setup_chinese_font()


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.results = {}
        # 初始化可视化器
        self.weight_viz = WeightLossVisualizer()
        self.opt_viz = OptimizationVisualizer()
        
    def _track_optimization_results(self, person: PersonProfile, optimization_results: Dict) -> DataTracker:
        """将优化结果转换为DataTracker（复用main.py的逻辑）"""
        tracker = DataTracker()
        tracker.metadata['start_date'] = datetime.now()
        tracker.metadata['person_profile'] = {
            'age': person.age,
            'gender': person.gender,
            'height': person.height,
            'initial_weight': person.initial_weight,
            'initial_body_fat': person.body_fat_percentage,
            'activity_level': person.activity_level
        }
        
        # 模拟每周的数据
        current_person = PersonProfile(
            age=person.age,
            gender=person.gender,
            height=person.height,
            weight=person.initial_weight,
            body_fat_percentage=person.body_fat_percentage,
            activity_level=person.activity_level,
            weeks_on_diet=person.weeks_on_diet
        )
        
        metabolic_model = AdvancedMetabolicModel()
        
        # 如果有历史方案，使用它们
        if 'best_solutions_history' in optimization_results:
            for i, solution in enumerate(optimization_results['best_solutions_history']):
                # 使用代谢模型模拟一周
                week_results = metabolic_model.simulate_week(current_person, solution, i)
                
                # 记录数据
                week_data = {
                    'weight': current_person.weight,
                    'body_fat_percentage': current_person.body_fat_percentage,
                    'muscle_mass': current_person.lean_body_mass,
                    'fat_mass': current_person.fat_mass,
                    'bmr': metabolic_model.calculate_bmr(current_person),
                    'tdee': metabolic_model.calculate_tdee(current_person, solution),
                    'metabolic_adaptation_factor': current_person.metabolic_adaptation_factor,
                    'calories_consumed': solution.calories,
                    'protein_grams': solution.calories * solution.protein_ratio / 4,
                    'carb_grams': solution.calories * solution.carb_ratio / 4,
                    'fat_grams': solution.calories * solution.fat_ratio / 9,
                    'cardio_minutes': solution.cardio_freq * solution.cardio_duration,
                    'strength_minutes': solution.strength_freq * 60,
                    'sleep_hours': solution.sleep_hours,
                    'fitness_score': solution.fitness,
                    'muscle_retention_rate': 1 - week_results['muscle_loss_rate'],
                    'fat_loss_rate': week_results['fat_loss_rate']
                }
                
                tracker.add_record(i, week_data)
                
                # 更新人体状态
                current_person = metabolic_model.update_person_state(
                    current_person, solution, i
                )
        
        return tracker
        
    def run_simulation_experiment(self, 
                                 person: PersonProfile,
                                 experiment_name: str = "simulation_exp",
                                 generate_viz: bool = True):
        """运行纯仿真实验"""
        logger.info(f"开始仿真实验: {experiment_name}")
        
        # 创建优化器
        optimizer = DifferentialEvolution(person, self.config)
        
        # 运行优化
        best_solution, results = optimizer.optimize()
        
        # 保存结果
        self.results[experiment_name] = {
            'type': 'simulation',
            'person': person,
            'best_solution': best_solution,
            'optimization_results': results,
            'timestamp': datetime.now()
        }
        
        # 生成可视化（使用已有的visualization功能）
        if generate_viz:
            self._generate_experiment_visualizations(experiment_name)
        
        return best_solution, results
    
    def run_validation_experiment(self,
                                real_data_file: str,
                                experiment_name: str = "validation_exp",
                                generate_viz: bool = True):
        """运行验证实验（使用真实数据验证算法）"""
        logger.info(f"开始验证实验: {experiment_name}")
        
        # 加载真实数据
        loader = ExperimentDataLoader()
        experiment_data = loader.load_experiment_data(real_data_file)
        
        # 获取第一个参与者
        participant = experiment_data['participants'][0]
        person = loader.create_person_profile(participant)
        
        # 运行优化
        optimizer = DifferentialEvolution(person, self.config)
        best_solution, opt_results = optimizer.optimize()
        
        # 比较预测结果与真实数据
        real_tracker = loader.convert_to_tracker(experiment_data)
        real_df = real_tracker.get_dataframe()
        
        # 使用最优方案生成预测数据
        sim = SimulatedExperiment()
        predicted_df = sim.generate_synthetic_data(
            person, best_solution, len(real_df), noise_level=0
        )
        
        # 计算误差
        mae_weight = np.mean(np.abs(real_df['weight'] - predicted_df['weight']))
        rmse_weight = np.sqrt(np.mean((real_df['weight'] - predicted_df['weight'])**2))
        
        self.results[experiment_name] = {
            'type': 'validation',
            'real_data': real_df,
            'predicted_data': predicted_df,
            'mae_weight': mae_weight,
            'rmse_weight': rmse_weight,
            'best_solution': best_solution,
            'optimization_results': opt_results,
            'person': person,
            'timestamp': datetime.now()
        }
        
        logger.info(f"验证结果 - MAE: {mae_weight:.2f}kg, RMSE: {rmse_weight:.2f}kg")
        
        # 生成可视化
        if generate_viz:
            self._generate_experiment_visualizations(experiment_name)
        
        return self.results[experiment_name]
    
    def run_ablation_study(self, 
                          person: PersonProfile,
                          components: List[str],
                          experiment_name: str = "ablation_study",
                          generate_viz: bool = True):
        """运行消融实验（测试各组件的重要性）"""
        logger.info(f"开始消融实验: {experiment_name}")
        
        ablation_results = {}
        
        # 完整模型
        logger.info("运行完整模型...")
        full_optimizer = DifferentialEvolution(person, self.config)
        full_solution, full_results = full_optimizer.optimize()
        ablation_results['full_model'] = {
            'fitness': full_solution.fitness,
            'weight_loss': person.weight - full_results['final_person_state'].weight,
            'solution': full_solution,
            'optimization_results': full_results
        }
        
        # 逐个移除组件
        for component in components:
            logger.info(f"移除组件: {component}")
            
            # 创建修改后的配置
            modified_config = ConfigManager()
            
            if component == 'metabolic_adaptation':
                # 禁用代谢适应
                modified_config.metabolic.adaptation_rate_per_week = 0
            elif component == 'sleep':
                # 固定睡眠时间
                modified_config.exercise.cardio_duration_options = [7.5]
            elif component == 'strength_training':
                # 移除力量训练
                modified_config.exercise.strength_frequency_range = (0, 0)
            
            # 运行优化
            ablation_optimizer = DifferentialEvolution(copy.deepcopy(person), modified_config)
            ablation_solution, ablation_opt_results = ablation_optimizer.optimize()
            
            ablation_results[f'without_{component}'] = {
                'fitness': ablation_solution.fitness,
                'weight_loss': person.weight - ablation_opt_results['final_person_state'].weight,
                'difference': ablation_solution.fitness - full_solution.fitness,
                'solution': ablation_solution,
                'optimization_results': ablation_opt_results
            }
        
        self.results[experiment_name] = {
            'type': 'ablation',
            'results': ablation_results,
            'person': person,
            'timestamp': datetime.now()
        }
        
        # 生成可视化
        if generate_viz:
            self._generate_experiment_visualizations(experiment_name)
        
        return ablation_results
    
    def run_parameter_sensitivity_analysis(self,
                                         person: PersonProfile,
                                         parameter_ranges: Dict,
                                         experiment_name: str = "sensitivity_analysis",
                                         generate_viz: bool = True):
        """运行参数敏感性分析"""
        logger.info(f"开始参数敏感性分析: {experiment_name}")
        
        sensitivity_results = {}
        
        for param_name, param_range in parameter_ranges.items():
            logger.info(f"分析参数: {param_name}")
            param_results = []
            
            for param_value in param_range:
                # 创建配置
                config = ConfigManager()
                
                # 设置参数值
                if param_name == 'population_size':
                    config.algorithm.population_size = param_value
                elif param_name == 'scaling_factor':
                    config.algorithm.scaling_factor = param_value
                elif param_name == 'crossover_rate':
                    config.algorithm.crossover_rate = param_value
                
                # 运行优化
                optimizer = DifferentialEvolution(copy.deepcopy(person), config)
                solution, results = optimizer.optimize()
                
                param_results.append({
                    'value': param_value,
                    'fitness': solution.fitness,
                    'iterations': results['total_iterations'],
                    'solution': solution,
                    'optimization_results': results
                })
            
            sensitivity_results[param_name] = param_results
        
        self.results[experiment_name] = {
            'type': 'sensitivity',
            'results': sensitivity_results,
            'person': person,
            'timestamp': datetime.now()
        }
        
        # 生成可视化
        if generate_viz:
            self._generate_experiment_visualizations(experiment_name)
        
        return sensitivity_results
    
    def run_comparative_experiment(self,
                                 person: PersonProfile,
                                 presets: List[str],
                                 experiment_name: str = "comparative_exp",
                                 generate_viz: bool = True):
        """运行比较实验（比较不同预设配置）"""
        logger.info(f"开始比较实验: {experiment_name}")
        
        comparative_results = {}
        all_trackers = {}  # 保存所有tracker用于对比
        
        for preset in presets:
            logger.info(f"测试预设: {preset}")
            
            # 加载预设配置
            config = load_preset(preset)
            
            # 运行优化
            optimizer = DifferentialEvolution(copy.deepcopy(person), config)
            solution, results = optimizer.optimize()
            
            # 创建tracker
            tracker = self._track_optimization_results(copy.deepcopy(person), results)
            all_trackers[preset] = tracker
            
            comparative_results[preset] = {
                'best_solution': solution,
                'fitness': solution.fitness,
                'weight_loss': person.weight - results['final_person_state'].weight,
                'iterations': results['total_iterations'],
                'final_state': results['final_person_state'],
                'optimization_results': results,
                'tracker': tracker
            }
        
        self.results[experiment_name] = {
            'type': 'comparative',
            'results': comparative_results,
            'trackers': all_trackers,
            'person': person,
            'timestamp': datetime.now()
        }
        
        # 生成可视化
        if generate_viz:
            self._generate_experiment_visualizations(experiment_name)
        
        return comparative_results
    
    def _generate_experiment_visualizations(self, experiment_name: str):
        """为实验生成可视化（使用已有的visualization模块）"""
        if experiment_name not in self.results:
            logger.error(f"未找到实验: {experiment_name}")
            return
        
        exp_data = self.results[experiment_name]
        exp_type = exp_data['type']
        
        # 创建输出目录
        output_dir = os.path.join('./results', experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            if exp_type == 'simulation' or exp_type == 'validation':
                # 创建DataTracker
                tracker = self._track_optimization_results(
                    exp_data['person'], 
                    exp_data['optimization_results']
                )
                
                # 1. 生成综合仪表板（使用WeightLossVisualizer）
                self.weight_viz.create_dashboard(
                    tracker,
                    save_path=os.path.join(output_dir, 'dashboard.png'),
                    show=False
                )
                
                # 2. 生成优化结果图（使用OptimizationVisualizer）
                self.opt_viz.plot_optimization_results(
                    exp_data['optimization_results'],
                    save_path=os.path.join(output_dir, 'optimization_results.png')
                )
                
                # 3. 生成优化进展图
                if 'population_history' in exp_data['optimization_results']:
                    self.opt_viz.plot_optimization_progress(
                        exp_data['optimization_results'].get('population_history', []),
                        exp_data['optimization_results'].get('fitness_history', []),
                        exp_data['optimization_results'].get('best_solutions_history', []),
                        save_path=os.path.join(output_dir, 'optimization_progress.png')
                    )
                
                # 4. 生成解空间分布图
                if 'best_solutions_history' in exp_data['optimization_results']:
                    self.opt_viz.plot_solution_space(
                        exp_data['optimization_results']['best_solutions_history'],
                        highlight_best=True,
                        save_path=os.path.join(output_dir, 'solution_space.png')
                    )
                
                # 5. 生成HTML报告（使用ReportGenerator）
                report_gen = ReportGenerator(self.weight_viz)
                report_gen.generate_html_report(
                    tracker,
                    exp_data['best_solution'],
                    exp_data['optimization_results'],
                    save_path=os.path.join(output_dir, 'report.html')
                )
                
                # 6. 保存tracker数据
                tracker.save_to_file(os.path.join(output_dir, 'tracking_data.json'))
                
                logger.info(f"完整可视化已生成: {output_dir}")
                
            elif exp_type == 'comparative':
                # 使用WeightLossVisualizer的比较报告功能
                trackers = exp_data.get('trackers', {})
                if trackers:
                    comparison_fig = self.weight_viz.create_comparison_report(
                        trackers,
                        save_path=os.path.join(output_dir, 'comparison_report.png')
                    )
                    
                # 为每个策略生成单独的报告
                for preset_name, result in exp_data['results'].items():
                    preset_dir = os.path.join(output_dir, preset_name)
                    os.makedirs(preset_dir, exist_ok=True)
                    
                    if 'tracker' in result:
                        # 生成仪表板
                        self.weight_viz.create_dashboard(
                            result['tracker'],
                            save_path=os.path.join(preset_dir, 'dashboard.png'),
                            show=False
                        )
                        
                        # 生成HTML报告
                        report_gen = ReportGenerator(self.weight_viz)
                        report_gen.generate_html_report(
                            result['tracker'],
                            result['best_solution'],
                            result['optimization_results'],
                            save_path=os.path.join(preset_dir, 'report.html')
                        )
                
            elif exp_type == 'ablation':
                # 为消融实验创建专门的可视化
                # 创建一个综合的tracker来展示所有结果
                results = exp_data['results']
                
                # 生成对比图表
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 1. 适应度对比
                components = list(results.keys())
                fitness_values = [results[c]['fitness'] for c in components]
                
                ax = axes[0, 0]
                bars = ax.bar(components, fitness_values, color='#2196f3')
                ax.set_ylabel('适应度值')
                ax.set_title('组件消融 - 适应度对比')
                ax.grid(True, alpha=0.3)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
                
                # 2. 减重效果对比
                ax = axes[0, 1]
                weight_losses = [results[c]['weight_loss'] for c in components]
                bars = ax.bar(components, weight_losses, color='#4caf50')
                ax.set_ylabel('减重量 (kg)')
                ax.set_title('组件消融 - 减重效果')
                ax.grid(True, alpha=0.3)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
                
                # 3. 组件重要性（差异）
                ax = axes[1, 0]
                importance = []
                labels = []
                for key in results:
                    if key != 'full_model' and 'difference' in results[key]:
                        labels.append(key.replace('without_', ''))
                        importance.append(abs(results[key]['difference']))
                
                if importance:
                    bars = ax.bar(labels, importance, color='#ff9800')
                    ax.set_ylabel('适应度影响（绝对值）')
                    ax.set_title('组件重要性排序')
                    ax.grid(True, alpha=0.3)
                
                # 4. 使用full_model的优化曲线
                ax = axes[1, 1]
                if 'optimization_results' in results['full_model']:
                    opt_results = results['full_model']['optimization_results']
                    if 'best_fitness_history' in opt_results:
                        iterations = range(1, len(opt_results['best_fitness_history']) + 1)
                        ax.plot(iterations, opt_results['best_fitness_history'], 'b-', linewidth=2)
                        ax.set_xlabel('迭代次数')
                        ax.set_ylabel('适应度值')
                        ax.set_title('完整模型优化曲线')
                        ax.grid(True, alpha=0.3)
                
                plt.suptitle(f'消融实验结果 - {experiment_name}', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'ablation_analysis.png'), dpi=150)
                plt.close()
                
            elif exp_type == 'sensitivity':
                # 为敏感性分析创建图表
                results = exp_data['results']
                
                import matplotlib.pyplot as plt
                n_params = len(results)
                fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
                if n_params == 1:
                    axes = [axes]
                
                for idx, (param_name, param_results) in enumerate(results.items()):
                    ax = axes[idx]
                    values = [r['value'] for r in param_results]
                    fitness_values = [r['fitness'] for r in param_results]
                    
                    ax.plot(values, fitness_values, 'o-', linewidth=2, markersize=8, color='#e91e63')
                    ax.set_xlabel(param_name)
                    ax.set_ylabel('适应度值')
                    ax.set_title(f'参数敏感性: {param_name}')
                    ax.grid(True, alpha=0.3)
                    
                    # 标记最优值
                    best_idx = np.argmin(fitness_values)
                    ax.plot(values[best_idx], fitness_values[best_idx], 
                           'r*', markersize=15, label=f'最优: {values[best_idx]}')
                    ax.legend()
                
                plt.suptitle(f'参数敏感性分析 - {experiment_name}', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'sensitivity_analysis.png'), dpi=150)
                plt.close()
                
        except Exception as e:
            logger.error(f"生成可视化时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_experiment_report(self, experiment_name: str, save_path: str = None):
        """生成实验报告（保持原有功能）"""
        if experiment_name not in self.results:
            logger.error(f"未找到实验: {experiment_name}")
            return
        
        exp_data = self.results[experiment_name]
        exp_type = exp_data['type']
        
        # 创建报告
        report = f"""
# 实验报告: {experiment_name}

**实验类型**: {exp_type}
**时间**: {exp_data['timestamp']}

## 结果摘要
"""
        
        if exp_type == 'simulation':
            report += f"""
- 最优适应度: {exp_data['best_solution'].fitness:.3f}
- 最优方案: {exp_data['best_solution']}
"""
        
        elif exp_type == 'validation':
            report += f"""
- 体重预测MAE: {exp_data['mae_weight']:.2f} kg
- 体重预测RMSE: {exp_data['rmse_weight']:.2f} kg
"""
        
        elif exp_type == 'ablation':
            report += "\n### 消融实验结果\n"
            for component, result in exp_data['results'].items():
                report += f"- {component}: 适应度={result['fitness']:.3f}, "
                report += f"减重={result.get('weight_loss', 'N/A'):.1f}kg\n"
        
        elif exp_type == 'comparative':
            report += "\n### 比较实验结果\n"
            for preset, result in exp_data['results'].items():
                report += f"- {preset}: 适应度={result['fitness']:.3f}, "
                report += f"减重={result['weight_loss']:.1f}kg\n"
        
        # 保存报告
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告已保存到: {save_path}")
        
        return report


def example_experiments():
    """运行示例实验"""
    # 创建实验运行器
    runner = ExperimentRunner()
    
    # 创建测试用户
    test_person = PersonProfile(
        age=30,
        gender='male',
        height=175,
        weight=85,
        body_fat_percentage=25,
        activity_level=1.4,
        weeks_on_diet=0
    )
    
    # 1. 仿真实验
    print("\n=== 运行仿真实验 ===")
    runner.run_simulation_experiment(test_person, "sim_exp_001")
    
    # 2. 生成合成数据用于验证
    print("\n=== 生成合成数据 ===")
    sim = SimulatedExperiment()
    sim.save_synthetic_dataset(5, 12, "synthetic_data.csv")
    
    # 3. 验证实验
    print("\n=== 运行验证实验 ===")
    runner.run_validation_experiment("synthetic_data.csv", "val_exp_001")
    
    # 4. 消融实验
    print("\n=== 运行消融实验 ===")
    runner.run_ablation_study(
        test_person,
        ['metabolic_adaptation', 'sleep', 'strength_training'],
        "ablation_exp_001"
    )
    
    # 5. 参数敏感性分析
    print("\n=== 运行参数敏感性分析 ===")
    runner.run_parameter_sensitivity_analysis(
        test_person,
        {
            'population_size': [5, 10, 15, 20],
            'scaling_factor': [0.4, 0.6, 0.8, 1.0]
        },
        "sensitivity_exp_001"
    )
    
    # 6. 比较实验
    print("\n=== 运行比较实验 ===")
    runner.run_comparative_experiment(
        test_person,
        ['aggressive', 'balanced', 'conservative'],
        "comparative_exp_001"
    )
    
    # 生成所有报告
    print("\n=== 生成实验报告 ===")
    for exp_name in runner.results:
        runner.generate_experiment_report(exp_name, f"./results/{exp_name}/report_{exp_name}.md")
    
    print("\n所有实验完成！")
    print("查看 ./results/ 目录下各实验文件夹，包含：")
    print("  - dashboard.png: 综合仪表板")
    print("  - optimization_*.png: 优化过程图")
    print("  - report.html: HTML报告")
    print("  - report_*.md: Markdown报告")
    print("  - tracking_data.json: 详细数据")


def example_real_data_workflow():
    """使用真实数据的工作流程示例"""
    print("\n=== 真实数据工作流程示例 ===")
    
    # 假设您有一个CSV文件包含真实的减肥数据
    # 格式应该包含: week, weight, body_fat_percentage等列
    
    # 创建示例CSV文件
    real_data = pd.DataFrame({
        'week': range(12),
        'weight': [85, 84.2, 83.5, 82.9, 82.3, 81.8, 81.4, 81.1, 80.8, 80.5, 80.3, 80.0],
        'body_fat_percentage': [25, 24.5, 24.1, 23.7, 23.3, 23.0, 22.7, 22.4, 22.2, 22.0, 21.8, 21.5],
        'calories_consumed': [1800] * 12,
        'cardio_minutes': [150] * 12,
        'sleep_hours': [7.5] * 12
    })
    real_data.to_csv('real_weight_loss_data.csv', index=False)
    
    # 使用真实数据
    loader = ExperimentDataLoader()
    experiment_data = loader.load_experiment_data('real_weight_loss_data.csv')
    
    # 转换为DataTracker
    tracker = loader.convert_to_tracker(experiment_data)
    
    # 可视化（使用已有的WeightLossVisualizer）
    viz = WeightLossVisualizer()
    viz.create_dashboard(tracker, save_path='real_data_dashboard.png', show=False)
    
    print("真实数据分析完成！")


if __name__ == "__main__":
    # 运行示例实验
    example_experiments()
    
    # 运行真实数据工作流程
    example_real_data_workflow()