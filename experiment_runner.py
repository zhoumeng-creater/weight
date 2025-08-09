"""
实验运行示例
展示如何使用系统进行不同类型的实验
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime

from metabolic_model import PersonProfile
from config import ConfigManager, load_preset
from de_algorithm import DifferentialEvolution
from visualization import DataTracker, WeightLossVisualizer, OptimizationVisualizer
from data_loader import ExperimentDataLoader, SimulatedExperiment, ExperimentDesigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.results = {}
        
    def run_simulation_experiment(self, 
                                 person: PersonProfile,
                                 experiment_name: str = "simulation_exp"):
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
        
        return best_solution, results
    
    def run_validation_experiment(self,
                                real_data_file: str,
                                experiment_name: str = "validation_exp"):
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
            'timestamp': datetime.now()
        }
        
        logger.info(f"验证结果 - MAE: {mae_weight:.2f}kg, RMSE: {rmse_weight:.2f}kg")
        
        return self.results[experiment_name]
    
    def run_ablation_study(self, 
                          person: PersonProfile,
                          components: List[str],
                          experiment_name: str = "ablation_study"):
        """运行消融实验（测试各组件的重要性）"""
        logger.info(f"开始消融实验: {experiment_name}")
        
        ablation_results = {}
        
        # 完整模型
        logger.info("运行完整模型...")
        full_optimizer = DifferentialEvolution(person, self.config)
        full_solution, full_results = full_optimizer.optimize()
        ablation_results['full_model'] = {
            'fitness': full_solution.fitness,
            'weight_loss': person.weight - full_results['final_person_state'].weight
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
            ablation_optimizer = DifferentialEvolution(person, modified_config)
            ablation_solution, ablation_results = ablation_optimizer.optimize()
            
            ablation_results[f'without_{component}'] = {
                'fitness': ablation_solution.fitness,
                'weight_loss': person.weight - ablation_results['final_person_state'].weight,
                'difference': ablation_solution.fitness - full_solution.fitness
            }
        
        self.results[experiment_name] = {
            'type': 'ablation',
            'results': ablation_results,
            'timestamp': datetime.now()
        }
        
        return ablation_results
    
    def run_parameter_sensitivity_analysis(self,
                                         person: PersonProfile,
                                         parameter_ranges: Dict,
                                         experiment_name: str = "sensitivity_analysis"):
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
                optimizer = DifferentialEvolution(person, config)
                solution, results = optimizer.optimize()
                
                param_results.append({
                    'value': param_value,
                    'fitness': solution.fitness,
                    'iterations': results['total_iterations']
                })
            
            sensitivity_results[param_name] = param_results
        
        self.results[experiment_name] = {
            'type': 'sensitivity',
            'results': sensitivity_results,
            'timestamp': datetime.now()
        }
        
        return sensitivity_results
    
    def run_comparative_experiment(self,
                                 person: PersonProfile,
                                 presets: List[str],
                                 experiment_name: str = "comparative_exp"):
        """运行比较实验（比较不同预设配置）"""
        logger.info(f"开始比较实验: {experiment_name}")
        
        comparative_results = {}
        
        for preset in presets:
            logger.info(f"测试预设: {preset}")
            
            # 加载预设配置
            config = load_preset(preset)
            
            # 运行优化
            optimizer = DifferentialEvolution(person.copy(), config)
            solution, results = optimizer.optimize()
            
            comparative_results[preset] = {
                'best_solution': solution,
                'fitness': solution.fitness,
                'weight_loss': person.weight - results['final_person_state'].weight,
                'iterations': results['total_iterations'],
                'final_state': results['final_person_state']
            }
        
        self.results[experiment_name] = {
            'type': 'comparative',
            'results': comparative_results,
            'timestamp': datetime.now()
        }
        
        return comparative_results
    
    def generate_experiment_report(self, experiment_name: str, save_path: str = None):
        """生成实验报告"""
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
        runner.generate_experiment_report(exp_name, f"report_{exp_name}.md")
    
    print("\n所有实验完成！")


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
    
    # 可视化
    viz = WeightLossVisualizer()
    viz.create_dashboard(tracker, save_path='real_data_dashboard.png', show=False)
    
    print("真实数据分析完成！")


if __name__ == "__main__":
    # 运行示例实验
    example_experiments()
    
    # 运行真实数据工作流程
    example_real_data_workflow()
