import traceback
from experiment_runner import EnhancedExperimentRunner
from config import ConfigManager

config = ConfigManager()
runner = EnhancedExperimentRunner(config)

# 测试D1
print("="*60)
print("测试D1实验...")
try:
    results, analysis = runner.run_experiment_D1_ablation_study()
    print("D1成功!")
except Exception as e:
    print(f"D1失败: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("测试E1实验...")
# 测试E1
try:
    results, analysis = runner.run_experiment_E1_long_term_tracking(weeks=4)  # 用短周期测试
    print("E1成功!")
    print(f"分析结果键: {list(analysis.keys())}")
    if 'statistical_tests' in analysis:
        print(f"统计测试: {analysis['statistical_tests']}")
except Exception as e:
    print(f"E1失败: {e}")
    traceback.print_exc()