"""
配置文件
统一管理所有参数配置，不包含任何业务逻辑
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmConfig:
    """差分进化算法配置"""
    # 基础参数
    population_size: int = 100
    max_iterations: int = 12000
    scaling_factor: float = 0.8
    crossover_rate: float = 0.9
    
    # 高级参数
    adaptive_parameters: bool = True  # 是否使用自适应参数
    scaling_factor_range: tuple = (0.4, 1.0)
    crossover_rate_range: tuple = (0.5, 1.0)
    
    # 早停条件
    early_stopping: bool = True
    patience: int = 3  # 连续n代无改善则停止
    min_improvement: float = 0.001  # 最小改善阈值
    
    # 多目标优化
    multi_objective: bool = False
    pareto_front_size: int = 20


@dataclass
class NutritionConfig:
    """营养配置"""
    # 热量范围
    calorie_deficit_range: tuple = (0.15, 0.35)  # TDEE的15-35%赤字
    min_absolute_calories: int = 1200
    max_absolute_calories: int = 3000
    
    # 营养素比例范围
    protein_range: tuple = (0.25, 0.45)
    carb_range: tuple = (0.30, 0.50)
    fat_range: tuple = (0.20, 0.35)
    
    # 营养素热量密度（kcal/g）
    protein_calories_per_gram: float = 4.0
    carb_calories_per_gram: float = 4.0
    fat_calories_per_gram: float = 9.0
    
    # 特殊需求
    min_protein_per_kg: float = 1.2  # 最低蛋白质摄入（g/kg体重）
    max_protein_per_kg: float = 2.5
    min_fiber_grams: float = 25  # 最低纤维摄入
    
    # 餐次分配（可选）
    meal_distribution: Dict[str, float] = field(default_factory=lambda: {
        "breakfast": 0.25,
        "lunch": 0.35,
        "dinner": 0.30,
        "snacks": 0.10
    })


@dataclass
class ExerciseConfig:
    """运动配置"""
    # 有氧运动
    cardio_frequency_range: tuple = (2, 6)  # 每周次数
    cardio_duration_options: List[int] = field(default_factory=lambda: [20, 30, 45, 60, 90])
    cardio_intensity_levels: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.5,      # 50% HRmax
        "moderate": 0.7,  # 70% HRmax
        "high": 0.85     # 85% HRmax
    })
    
    # 力量训练
    strength_frequency_range: tuple = (2, 5)
    strength_duration_minutes: int = 60  # 假设每次时长
    strength_muscle_groups: List[str] = field(default_factory=lambda: [
        "胸部", "背部", "腿部", "肩部", "手臂", "核心"
    ])
    
    # 运动限制
    max_weekly_hours: float = 15
    min_rest_days: int = 1
    
    # 能量消耗估算（kcal/分钟/kg体重）
    cardio_calorie_burn_rate: Dict[str, float] = field(default_factory=lambda: {
        "walking": 0.05,
        "jogging": 0.10,
        "running": 0.15,
        "cycling": 0.08,
        "swimming": 0.12,
        "hiit": 0.14
    })
    
    strength_calorie_burn_per_session: float = 300  # 平均每次力量训练消耗


@dataclass
class MetabolicConfig:
    """代谢模型配置"""
    # BMR计算方法
    bmr_equation: str = "mifflin"  # 可选: "mifflin", "harris", "katch"
    
    # 活动系数
    activity_levels: Dict[str, float] = field(default_factory=lambda: {
        "sedentary": 1.2,      # 久坐
        "lightly_active": 1.375,  # 轻度活动
        "moderately_active": 1.55,  # 中度活动
        "very_active": 1.725,    # 高度活动
        "extra_active": 1.9      # 极度活动
    })
    
    # 代谢适应参数
    adaptation_rate_per_week: float = 0.05  # 每周适应率
    min_adaptation_factor: float = 0.75  # 最低代谢率（相对于初始）
    adaptation_recovery_weeks: int = 2  # 恢复正常代谢所需周数
    
    # 体成分变化
    calories_per_kg_fat: float = 7700
    calories_per_kg_muscle: float = 1800
    
    # 激素影响因子
    leptin_sensitivity: float = 0.1
    cortisol_impact: float = 0.05
    thyroid_impact: float = 0.1
    
    # 睡眠影响
    optimal_sleep_hours: float = 8.0
    sleep_deficit_penalty: float = 0.05  # 每少1小时的代谢惩罚


@dataclass
class FitnessConfig:
    """适应度评估配置"""
    # 基础权重
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        "muscle_loss": 0.4,
        "fat_loss": 0.4,
        "sustainability": 0.2
    })
    
    # 阶段性权重调整
    phase_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "initial": {"muscle_loss": 0.3, "fat_loss": 0.5, "sustainability": 0.2},
        "middle": {"muscle_loss": 0.4, "fat_loss": 0.4, "sustainability": 0.2},
        "final": {"muscle_loss": 0.5, "fat_loss": 0.25, "sustainability": 0.25}
    })
    
    # 理想指标
    ideal_weekly_weight_loss: float = 0.75  # kg
    ideal_muscle_retention_rate: float = 0.9  # 90%肌肉保留
    max_acceptable_deficit: float = 0.35  # 最大可接受能量赤字
    
    # 惩罚因子
    plateau_penalty: float = 0.1  # 每周平台期惩罚
    extreme_deficit_penalty: float = 0.5  # 极端赤字惩罚
    muscle_loss_threshold: float = 0.25  # 肌肉流失警戒线


@dataclass
class UserPreferences:
    """用户偏好设置"""
    # 饮食偏好
    dietary_restrictions: List[str] = field(default_factory=list)  # 如：vegetarian, vegan, keto
    preferred_meal_frequency: int = 3  # 每日餐次
    intermittent_fasting: Optional[str] = None  # 如："16:8", "5:2"
    
    # 运动偏好
    preferred_exercises: List[str] = field(default_factory=list)
    avoided_exercises: List[str] = field(default_factory=list)
    gym_access: bool = True
    home_equipment: List[str] = field(default_factory=list)  # 如：dumbbells, resistance_bands
    
    # 生活方式
    work_schedule: str = "regular"  # regular, shift, flexible
    stress_level: str = "moderate"  # low, moderate, high
    social_eating_frequency: int = 2  # 每周社交聚餐次数
    
    # 目标设置
    primary_goal: str = "fat_loss"  # fat_loss, muscle_gain, health
    target_date: Optional[str] = None  # 目标达成日期
    acceptable_muscle_loss: float = 0.1  # 可接受的肌肉流失比例


@dataclass
class SystemConfig:
    """系统配置"""
    # 日志设置
    log_level: str = "INFO"
    log_file: Optional[str] = "de_weight_loss.log"
    
    # 数据存储
    data_directory: str = "./data"
    results_directory: str = "./results"
    save_intermediate_results: bool = True
    
    # 可视化设置
    plot_style: str = "seaborn"
    figure_dpi: int = 300
    save_plots: bool = True
    show_plots: bool = True
    real_time_monitoring: bool = False
    
    # 性能设置
    parallel_evaluation: bool = False
    num_threads: int = 4
    random_seed: Optional[int] = 42
    
    # 导出设置
    export_format: List[str] = field(default_factory=lambda: ["json", "csv", "html"])
    include_detailed_history: bool = True


class ConfigManager:
    """配置管理器 - 只负责配置的加载、保存和验证"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.algorithm = AlgorithmConfig()
        self.nutrition = NutritionConfig()
        self.exercise = ExerciseConfig()
        self.metabolic = MetabolicConfig()
        self.fitness = FitnessConfig()
        self.user_preferences = UserPreferences()
        self.system = SystemConfig()
        
        # 尝试加载现有配置
        self.load_config()
        
    def load_config(self):
        """从文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 更新各个配置对象
                if 'algorithm' in data:
                    self.algorithm = AlgorithmConfig(**data['algorithm'])
                if 'nutrition' in data:
                    self.nutrition = NutritionConfig(**data['nutrition'])
                if 'exercise' in data:
                    self.exercise = ExerciseConfig(**data['exercise'])
                if 'metabolic' in data:
                    self.metabolic = MetabolicConfig(**data['metabolic'])
                if 'fitness' in data:
                    self.fitness = FitnessConfig(**data['fitness'])
                if 'user_preferences' in data:
                    self.user_preferences = UserPreferences(**data['user_preferences'])
                if 'system' in data:
                    self.system = SystemConfig(**data['system'])
                
                logger.info(f"配置已从 {self.config_file} 加载")
            except Exception as e:
                logger.warning(f"加载配置失败: {e}，使用默认配置")
    
    def save_config(self):
        """保存配置到文件"""
        data = {
            'algorithm': asdict(self.algorithm),
            'nutrition': asdict(self.nutrition),
            'exercise': asdict(self.exercise),
            'metabolic': asdict(self.metabolic),
            'fitness': asdict(self.fitness),
            'user_preferences': asdict(self.user_preferences),
            'system': asdict(self.system)
        }
        
        os.makedirs(os.path.dirname(self.config_file) or '.', exist_ok=True)
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"配置已保存到 {self.config_file}")
    
    def update_from_dict(self, updates: Dict[str, Dict]):
        """从字典更新配置"""
        for section, values in updates.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def validate_config(self) -> List[str]:
        """验证配置的合理性"""
        errors = []
        
        # 验证营养素比例
        min_total = (self.nutrition.protein_range[0] + 
                    self.nutrition.carb_range[0] + 
                    self.nutrition.fat_range[0])
        max_total = (self.nutrition.protein_range[1] + 
                    self.nutrition.carb_range[1] + 
                    self.nutrition.fat_range[1])
        
        if min_total > 1.0:
            errors.append("营养素最小比例之和超过100%")
        if max_total < 1.0:
            errors.append("营养素最大比例之和小于100%")
        
        # 验证热量范围
        if self.nutrition.min_absolute_calories < 800:
            errors.append("最低热量设置过低，可能危害健康")
        
        # 验证运动时间
        if self.exercise.max_weekly_hours > 20:
            errors.append("每周最大运动时间设置过高")
        
        # 验证算法参数
        if not (0 < self.algorithm.scaling_factor <= 2):
            errors.append("缩放因子应在(0, 2]范围内")
        
        if not (0 < self.algorithm.crossover_rate <= 1):
            errors.append("交叉率应在(0, 1]范围内")
        
        return errors
    
    def get_constraint_config(self) -> Dict:
        """获取约束配置（供SolutionGenerator使用）"""
        return {
            'min_calories': self.nutrition.min_absolute_calories,
            'max_calories': self.nutrition.max_absolute_calories,
            'min_protein_ratio': self.nutrition.protein_range[0],
            'max_protein_ratio': self.nutrition.protein_range[1],
            'min_carb_ratio': self.nutrition.carb_range[0],
            'max_carb_ratio': self.nutrition.carb_range[1],
            'min_fat_ratio': self.nutrition.fat_range[0],
            'max_fat_ratio': self.nutrition.fat_range[1],
            'min_cardio_freq': self.exercise.cardio_frequency_range[0],
            'max_cardio_freq': self.exercise.cardio_frequency_range[1],
            'cardio_duration_options': self.exercise.cardio_duration_options,
            'min_strength_freq': self.exercise.strength_frequency_range[0],
            'max_strength_freq': self.exercise.strength_frequency_range[1],
            'max_weekly_exercise_hours': self.exercise.max_weekly_hours,
            'min_sleep_hours': 6.5,  # 可以从用户偏好中调整
            'max_sleep_hours': 8.5,
            'min_protein_per_kg': self.nutrition.min_protein_per_kg,
            'max_protein_per_kg': self.nutrition.max_protein_per_kg
        }


# 创建默认配置实例
default_config = ConfigManager()


# 预设配置模板
PRESET_CONFIGS = {
    "aggressive": {
        "algorithm": {
            "scaling_factor": 1.0,
            "crossover_rate": 0.95
        },
        "nutrition": {
            "calorie_deficit_range": (0.25, 0.40),
            "protein_range": (0.35, 0.45)
        },
        "fitness": {
            "default_weights": {
                "muscle_loss": 0.3,
                "fat_loss": 0.5,
                "sustainability": 0.2
            }
        }
    },
    "balanced": {
        "algorithm": {
            "scaling_factor": 0.8,
            "crossover_rate": 0.9
        },
        "nutrition": {
            "calorie_deficit_range": (0.15, 0.30),
            "protein_range": (0.30, 0.40)
        },
        "fitness": {
            "default_weights": {
                "muscle_loss": 0.4,
                "fat_loss": 0.4,
                "sustainability": 0.2
            }
        }
    },
    "conservative": {
        "algorithm": {
            "scaling_factor": 0.6,
            "crossover_rate": 0.8
        },
        "nutrition": {
            "calorie_deficit_range": (0.10, 0.20),
            "protein_range": (0.25, 0.35)
        },
        "fitness": {
            "default_weights": {
                "muscle_loss": 0.5,
                "fat_loss": 0.25,
                "sustainability": 0.25
            }
        }
    }
}


def load_preset(preset_name: str) -> ConfigManager:
    """加载预设配置"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"未知的预设配置: {preset_name}")
    
    config = ConfigManager()
    config.update_from_dict(PRESET_CONFIGS[preset_name])
    
    logger.info(f"已加载预设配置: {preset_name}")
    return config


if __name__ == "__main__":
    # 测试配置管理器
    config = ConfigManager()
    
    # 保存默认配置
    config.save_config()
    
    # 验证配置
    errors = config.validate_config()
    if errors:
        print("配置验证错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置验证通过")
    
    # 测试预设加载
    aggressive_config = load_preset("aggressive")
    print(f"\n激进模式热量赤字范围: {aggressive_config.nutrition.calorie_deficit_range}")
    
    # 测试约束配置获取
    constraints = config.get_constraint_config()
    print(f"\n约束配置示例 - 热量范围: {constraints['min_calories']}-{constraints['max_calories']}")