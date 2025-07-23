"""
差分进化算法在减肥平台期优化中的应用 - 主程序
Author: Meng Zhou
Date: 2025-07-19
"""

import logging
import argparse
import os
import sys
from datetime import datetime
import numpy as np

# 导入项目模块
from metabolic_model import PersonProfile
from config import ConfigManager, load_preset
from solution_generator import SolutionGenerator, SolutionConstraints
from de_algorithm import DifferentialEvolution
from visualization import WeightLossVisualizer, DataTracker, OptimizationVisualizer, ReportGenerator
from fitness_evaluator import AdaptiveFitnessEvaluator
from metabolic_model import AdvancedMetabolicModel

# 设置中文字体支持
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def setup_logging(log_level="INFO", log_file=None):
    """设置日志配置"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def create_user_profile():
    """交互式创建用户档案"""
    print("\n=== 创建用户档案 ===")
    
    try:
        age = int(input("请输入年龄: "))
        gender = input("请输入性别 (male/female): ").lower()
        height = float(input("请输入身高 (cm): "))
        weight = float(input("请输入体重 (kg): "))
        body_fat = float(input("请输入体脂率 (%): "))
        
        print("\n活动水平:")
        print("1. 久坐 (1.2)")
        print("2. 轻度活动 (1.375)")
        print("3. 中度活动 (1.55)")
        print("4. 高度活动 (1.725)")
        print("5. 极度活动 (1.9)")
        
        activity_choice = int(input("请选择活动水平 (1-5): "))
        activity_levels = [1.2, 1.375, 1.55, 1.725, 1.9]
        activity_level = activity_levels[activity_choice - 1]
        
        weeks_on_diet = int(input("已经减肥多少周? (如果刚开始请输入0): "))
        
        return PersonProfile(
            age=age,
            gender=gender,
            height=height,
            weight=weight,
            body_fat_percentage=body_fat,
            activity_level=activity_level,
            weeks_on_diet=weeks_on_diet
        )
    
    except (ValueError, IndexError) as e:
        print(f"输入错误: {e}")
        print("使用默认用户档案")
        return PersonProfile(
            age=30,
            gender='male',
            height=175,
            weight=85,
            body_fat_percentage=25,
            activity_level=1.4,
            weeks_on_diet=8
        )


def run_optimization(person, config):
    """运行优化过程"""
    print("\n=== 开始优化 ===")
    print(f"配置: 种群大小={config.algorithm.population_size}, "
          f"最大迭代={config.algorithm.max_iterations}, "
          f"自适应参数={'是' if config.algorithm.adaptive_parameters else '否'}")
    
    # 创建优化器
    optimizer = DifferentialEvolution(person, config)
    
    # 运行优化
    best_solution, results = optimizer.optimize()
    
    # 输出结果摘要
    print("\n=== 优化结果摘要 ===")
    print(f"最佳方案: {best_solution}")
    print(f"最终适应度: {best_solution.fitness:.3f}")
    print(f"初始体重: {results['initial_weight']:.1f} kg")
    print(f"最终体重: {results['final_person_state'].weight:.1f} kg")
    print(f"总减重: {results['initial_weight'] - results['final_person_state'].weight:.1f} kg")
    print(f"总迭代次数: {results['total_iterations']}")
    
    # 打印最优方案详情
    print("\n=== 最优方案详情 ===")
    print(f"每日热量: {best_solution.calories:.0f} kcal")
    print(f"营养素比例 - 蛋白质: {best_solution.protein_ratio:.1%}, "
          f"碳水: {best_solution.carb_ratio:.1%}, "
          f"脂肪: {best_solution.fat_ratio:.1%}")
    print(f"有氧运动: 每周{best_solution.cardio_freq}次, 每次{best_solution.cardio_duration}分钟")
    print(f"力量训练: 每周{best_solution.strength_freq}次")
    print(f"建议睡眠: {best_solution.sleep_hours:.1f}小时/晚")
    
    # 适应度组成分析
    if best_solution.fitness_components:
        print("\n=== 适应度组成分析 ===")
        for key, value in best_solution.fitness_components.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
    
    return best_solution, results


def create_visualizations(results, tracker, output_dir):
    """创建所有可视化图表"""
    print("\n=== 生成可视化 ===")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 优化过程可视化
    opt_viz = OptimizationVisualizer()
    
    # 优化结果图
    opt_viz.plot_optimization_results(
        results, 
        save_path=os.path.join(output_dir, "optimization_results.png")
    )
    print("✓ 优化结果图已生成")
    
    # 优化进展图
    if 'population_history' in results and 'fitness_history' in results:
        opt_viz.plot_optimization_progress(
            results['population_history'],
            results['fitness_history'],
            results['best_solutions_history'],
            save_path=os.path.join(output_dir, "optimization_progress.png")
        )
        print("✓ 优化进展图已生成")
    
    # 解空间分布图
    if 'best_solutions_history' in results:
        opt_viz.plot_solution_space(
            results['best_solutions_history'],
            highlight_best=True,
            save_path=os.path.join(output_dir, "solution_space.png")
        )
        print("✓ 解空间分布图已生成")
    
    # 2. 减肥进度可视化
    viz = WeightLossVisualizer()
    viz.create_dashboard(
        tracker, 
        save_path=os.path.join(output_dir, "weight_loss_dashboard.png"),
        show=False
    )
    print("✓ 综合仪表板已生成")
    
    # 3. 动画（可选）
    try:
        anim = viz.create_animation(
            tracker,
            save_path=os.path.join(output_dir, "weight_loss_animation.gif")
        )
        print("✓ 动画已生成")
    except Exception as e:
        print(f"⚠ 动画生成失败: {e}")


def track_results(person, results):
    """将优化结果记录到数据追踪器"""
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
    tracker.metadata['goals'] = {
        'target_weight': person.weight - 10,  # 假设目标减重10kg
        'target_body_fat': max(10 if person.gender == 'male' else 18, 
                              person.body_fat_percentage - 10)
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
    
    for i, solution in enumerate(results['best_solutions_history']):
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
            'fat_loss_rate': week_results['fat_loss_rate'],
            'solution': solution.to_vector().tolist()
        }
        
        tracker.add_record(i, week_data)
        
        # 更新人体状态
        current_person = metabolic_model.update_person_state(
            current_person, solution, i
        )
    
    return tracker


def generate_report(tracker, best_solution, results, output_dir):
    """生成HTML报告"""
    print("\n=== 生成报告 ===")
    
    viz = WeightLossVisualizer()
    report_gen = ReportGenerator(viz)
    
    report_path = os.path.join(output_dir, "weight_loss_report.html")
    
    report_gen.generate_html_report(
        tracker,
        best_solution,
        {
            'population_history': results.get('population_history', []),
            'fitness_history': results.get('fitness_history', []),
            'best_history': results['best_solutions_history']
        },
        save_path=report_path
    )
    
    print(f"✓ HTML报告已生成: {report_path}")
    return report_path


def check_plateau_and_suggest(results, config):
    """检查是否存在平台期并提供建议"""
    if len(results['best_fitness_history']) < 3:
        return
    
    # 检查最近的改善情况
    recent_improvements = []
    for i in range(len(results['best_fitness_history'])-3, len(results['best_fitness_history'])):
        if i > 0:
            improvement = results['best_fitness_history'][i-1] - results['best_fitness_history'][i]
            recent_improvements.append(improvement)
    
    # 如果改善很小，可能是平台期
    if all(imp < 0.01 for imp in recent_improvements):
        print("\n⚠ 检测到可能的减肥平台期！")
        print("\n=== 平台期突破建议 ===")
        
        # 使用方案生成器生成突破方案
        constraint_config = config.get_constraint_config()
        constraints = SolutionConstraints(**constraint_config)
        generator = SolutionGenerator(constraints)
        
        best_solution = results['best_solution']
        plateau_solutions = generator.generate_plateau_breaking_solutions(
            best_solution.to_vector(),
            num_solutions=3
        )
        
        print("\n推荐的平台期突破方案：")
        for i, sol_vector in enumerate(plateau_solutions, 1):
            from solution_generator import Solution
            sol = Solution(sol_vector)
            print(f"\n方案 {i}:")
            print(f"  热量: {sol.calories:.0f} kcal")
            print(f"  营养素: P {sol.protein_ratio:.0%} / C {sol.carb_ratio:.0%} / F {sol.fat_ratio:.0%}")
            print(f"  运动: 有氧{sol.cardio_freq}次×{sol.cardio_duration}分, 力量{sol.strength_freq}次")
            print(f"  睡眠: {sol.sleep_hours:.1f}小时")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='差分进化算法减肥优化系统')
    parser.add_argument('--mode', choices=['interactive', 'default', 'test'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--preset', choices=['aggressive', 'balanced', 'conservative'], 
                       default='balanced', help='配置预设')
    parser.add_argument('--output', default='./results', help='输出目录')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    parser.add_argument('--advanced', action='store_true', help='使用高级功能')
    
    args = parser.parse_args()
    
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.output, f'optimization_{timestamp}.log')
    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("差分进化算法减肥优化系统启动")
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"输出目录: {args.output}")
    logger.info("="*50)
    
    # 1. 创建或加载用户档案
    if args.mode == 'interactive':
        person = create_user_profile()
    else:
        # 使用默认用户档案
        person = PersonProfile(
            age=30,
            gender='male',
            height=175,
            weight=85,
            body_fat_percentage=25,
            activity_level=1.4,
            weeks_on_diet=8
        )
    
    # 2. 加载配置
    if args.preset:
        config = load_preset(args.preset)
        print(f"\n使用预设配置: {args.preset}")
    else:
        config = ConfigManager()
    
    # 如果是测试模式，使用较小的参数
    if args.mode == 'test':
        config.algorithm.population_size = 5
        config.algorithm.max_iterations = 3
        config.system.show_plots = False
    
    # 3. 运行优化
    best_solution, results = run_optimization(person, config)
    
    # 4. 追踪和记录结果
    tracker = track_results(person, results)
    
    # 保存追踪数据
    tracker_file = os.path.join(args.output, f'tracking_data_{timestamp}.json')
    tracker.save_to_file(tracker_file)
    print(f"\n✓ 追踪数据已保存: {tracker_file}")
    
    # 5. 生成可视化
    create_visualizations(results, tracker, args.output)
    
    # 6. 生成报告
    report_path = generate_report(tracker, best_solution, results, args.output)
    
    # 7. 保存配置
    config_file = os.path.join(args.output, f'config_{timestamp}.json')
    config.save_config()
    os.rename(config.config_file, config_file)
    print(f"✓ 配置已保存: {config_file}")
    
    # 8. 检查平台期并提供建议
    check_plateau_and_suggest(results, config)
    
    # 9. 高级功能演示（如果启用）
    if args.advanced:
        print("\n=== 高级功能演示 ===")
        from fitness_evaluator import AdaptiveFitnessEvaluator
        adaptive_evaluator = AdaptiveFitnessEvaluator()
        print("✓ 自适应适应度评估器已启用")
        print("✓ 高级代谢模型已启用")
    
    print("\n=== 优化完成 ===")
    print(f"所有结果已保存到: {args.output}")
    print(f"查看HTML报告: {report_path}")
    
    # 如果不是测试模式，询问是否打开报告
    if args.mode != 'test' and config.system.show_plots:
        try:
            import webbrowser
            if input("\n是否在浏览器中打开报告? (y/n): ").lower() == 'y':
                webbrowser.open(report_path)
        except:
            pass


if __name__ == "__main__":
    main()