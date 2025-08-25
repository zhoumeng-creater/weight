"""
方案生成器模块
负责所有与方案相关的操作：Solution类定义、生成、验证、编码/解码、约束处理
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import json
from config import ConfigManager

logger = logging.getLogger(__name__)


class Solution:
    """减肥方案类"""
    def __init__(self, vector: np.ndarray):
        """
        初始化方案
        vector: [C, P, H, F, Cardio_Freq, Cardio_Dur, Strength_Freq, Sleep_Dur]
        """
        self.calories = vector[0]
        self.protein_ratio = vector[1]
        self.carb_ratio = vector[2]
        self.fat_ratio = vector[3]
        self.cardio_freq = int(vector[4])
        self.cardio_duration = int(vector[5])
        self.strength_freq = int(vector[6])
        self.sleep_hours = vector[7]
        
        self.fitness = None
        self.fitness_components = None
        
    def to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        return np.array([
            self.calories, self.protein_ratio, self.carb_ratio, self.fat_ratio,
            self.cardio_freq, self.cardio_duration, self.strength_freq, self.sleep_hours
        ])
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"方案: 热量={self.calories:.0f}kcal, "
                f"P/C/F={self.protein_ratio:.0%}/{self.carb_ratio:.0%}/{self.fat_ratio:.0%}, "
                f"有氧={self.cardio_freq}次×{self.cardio_duration}分, "
                f"力量={self.strength_freq}次, 睡眠={self.sleep_hours:.1f}小时")


@dataclass
class SolutionConstraints:
    """方案约束条件"""
    # 热量约束
    min_calories: float = 1200
    max_calories: float = 2500
    
    # 营养素比例约束
    min_protein_ratio: float = 0.20
    max_protein_ratio: float = 0.50
    min_carb_ratio: float = 0.25
    max_carb_ratio: float = 0.60
    min_fat_ratio: float = 0.15
    max_fat_ratio: float = 0.40
    
    # 运动约束
    min_cardio_freq: int = 0
    max_cardio_freq: int = 7
    cardio_duration_options: List[int] = None
    min_strength_freq: int = 0
    max_strength_freq: int = 6
    max_weekly_exercise_hours: float = 20
    
    # 睡眠约束
    min_sleep_hours: float = 6.0
    max_sleep_hours: float = 9.0
    
    # 蛋白质摄入约束（g/kg体重）
    min_protein_per_kg: float = 1.0
    max_protein_per_kg: float = 3.0
    
    def __post_init__(self):
        if self.cardio_duration_options is None:
            self.cardio_duration_options = [20, 30, 45, 60, 90]


class SolutionGenerator:
    """方案生成器"""
    
    def __init__(self, constraints: Optional[SolutionConstraints] = None,
                 config: Optional[ConfigManager] = None):
        """
        初始化方案生成器
        
        Args:
            constraints: 约束条件
            config: 配置管理器
        """
        self.config = config or ConfigManager()
        
        # 如果没有提供约束，从配置生成
        if constraints is None:
            constraint_dict = self.config.get_constraint_config()
            self.constraints = SolutionConstraints(**constraint_dict)
        else:
            self.constraints = constraints
        
        # 预定义的方案模板
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Dict]:
        """加载预定义的方案模板"""
        templates = {
            "balanced": {
                "name": "均衡型",
                "description": "平衡的饮食和运动方案",
                "calories_factor": 0.85,  # TDEE的85%
                "protein_ratio": 0.30,
                "carb_ratio": 0.40,
                "fat_ratio": 0.30,
                "cardio_freq": 3,
                "cardio_duration": 45,
                "strength_freq": 3,
                "sleep_hours": 7.5
            },
            "low_carb": {
                "name": "低碳型",
                "description": "低碳水化合物方案",
                "calories_factor": 0.80,
                "protein_ratio": 0.35,
                "carb_ratio": 0.30,
                "fat_ratio": 0.35,
                "cardio_freq": 4,
                "cardio_duration": 30,
                "strength_freq": 3,
                "sleep_hours": 7.5
            },
            "high_protein": {
                "name": "高蛋白型",
                "description": "高蛋白保肌方案",
                "calories_factor": 0.80,
                "protein_ratio": 0.40,
                "carb_ratio": 0.35,
                "fat_ratio": 0.25,
                "cardio_freq": 3,
                "cardio_duration": 30,
                "strength_freq": 4,
                "sleep_hours": 8.0
            },
            "endurance": {
                "name": "耐力型",
                "description": "重视有氧运动的方案",
                "calories_factor": 0.85,
                "protein_ratio": 0.25,
                "carb_ratio": 0.50,
                "fat_ratio": 0.25,
                "cardio_freq": 5,
                "cardio_duration": 60,
                "strength_freq": 2,
                "sleep_hours": 8.0
            },
            "strength": {
                "name": "力量型",
                "description": "重视力量训练的方案",
                "calories_factor": 0.90,
                "protein_ratio": 0.35,
                "carb_ratio": 0.40,
                "fat_ratio": 0.25,
                "cardio_freq": 2,
                "cardio_duration": 30,
                "strength_freq": 5,
                "sleep_hours": 8.0
            }
        }
        return templates
    
    def handle_bounds(self, vector: np.ndarray) -> np.ndarray:
        """处理边界约束"""
        # 连续变量
        vector[0] = np.clip(vector[0], self.constraints.min_calories, self.constraints.max_calories)
        vector[1] = np.clip(vector[1], self.constraints.min_protein_ratio, self.constraints.max_protein_ratio)
        vector[2] = np.clip(vector[2], self.constraints.min_carb_ratio, self.constraints.max_carb_ratio)
        vector[3] = np.clip(vector[3], self.constraints.min_fat_ratio, self.constraints.max_fat_ratio)
        vector[7] = np.clip(vector[7], self.constraints.min_sleep_hours, self.constraints.max_sleep_hours)
        
        # 离散变量
        vector[4] = min(range(self.constraints.min_cardio_freq, self.constraints.max_cardio_freq + 1), 
                       key=lambda x: abs(x - vector[4]))
        vector[5] = min(self.constraints.cardio_duration_options, 
                       key=lambda x: abs(x - vector[5]))
        vector[6] = min(range(self.constraints.min_strength_freq, self.constraints.max_strength_freq + 1), 
                       key=lambda x: abs(x - vector[6]))
        
        return vector
    
    def normalize_solution_vector(self, vector: np.ndarray) -> np.ndarray:
        """规范化方案向量（确保营养素比例和为1）"""
        # 确保营养素比例和为1
        protein = vector[1]
        carb = vector[2]
        fat = vector[3]
        total = protein + carb + fat
        
        if total > 0:
            vector[1] = protein / total
            vector[2] = carb / total
            vector[3] = fat / total
        
        # 再次确保在约束范围内
        return self.handle_bounds(vector)
    
    def validate_solution(self, solution: np.ndarray, person_weight: Optional[float] = None) -> bool:
        """验证方案是否满足所有约束条件"""
        calories, protein_ratio, carb_ratio, fat_ratio, \
        cardio_freq, cardio_duration, strength_freq, sleep_hours = solution
        
        # 热量约束
        if not (self.constraints.min_calories <= calories <= self.constraints.max_calories):
            return False
        
        # 营养素比例约束
        if not (self.constraints.min_protein_ratio <= protein_ratio <= self.constraints.max_protein_ratio):
            return False
        if not (self.constraints.min_carb_ratio <= carb_ratio <= self.constraints.max_carb_ratio):
            return False
        if not (self.constraints.min_fat_ratio <= fat_ratio <= self.constraints.max_fat_ratio):
            return False
        
        # 营养素比例和必须接近1
        if abs(protein_ratio + carb_ratio + fat_ratio - 1.0) > 0.01:
            return False
        
        # 运动约束
        if not (self.constraints.min_cardio_freq <= cardio_freq <= self.constraints.max_cardio_freq):
            return False
        if cardio_duration not in self.constraints.cardio_duration_options:
            return False
        if not (self.constraints.min_strength_freq <= strength_freq <= self.constraints.max_strength_freq):
            return False
        
        # 每周总运动时间约束
        total_exercise_hours = (cardio_freq * cardio_duration / 60 + strength_freq * 1.0)
        if total_exercise_hours > self.constraints.max_weekly_exercise_hours:
            return False
        
        # 睡眠约束
        if not (self.constraints.min_sleep_hours <= sleep_hours <= self.constraints.max_sleep_hours):
            return False
        
        # 蛋白质摄入量约束（如果提供了体重）
        if person_weight is not None:
            protein_grams = calories * protein_ratio / 4  # 蛋白质每克4卡
            protein_per_kg = protein_grams / person_weight
            if not (self.constraints.min_protein_per_kg <= protein_per_kg <= self.constraints.max_protein_per_kg):
                return False
        
        return True
    
    def generate_random_solution(self, person_tdee: float) -> np.ndarray:
        """生成随机方案"""
        max_attempts = 100
        
        for _ in range(max_attempts):
        # 生成热量 - 修复：确保范围有效
            cal_lower = max(self.constraints.min_calories, person_tdee * 0.6)
            cal_upper = min(self.constraints.max_calories, person_tdee * 0.95)
            
            # 关键修复：如果上界小于下界，调整范围
            if cal_upper <= cal_lower:
                # 方案1：使用最小热量附近的范围
                cal_lower = self.constraints.min_calories
                cal_upper = min(self.constraints.max_calories, 
                            max(self.constraints.min_calories + 300, person_tdee * 1.1))
                
            calories = np.random.uniform(cal_lower, cal_upper)
            
            # 生成营养素比例（使用Dirichlet分布确保和为1）
            alpha = [3, 4, 3]  # 控制分布的参数
            ratios = np.random.dirichlet(alpha)
            protein_ratio, carb_ratio, fat_ratio = ratios
            
            # 调整到约束范围内
            protein_ratio = np.clip(protein_ratio, 
                                   self.constraints.min_protein_ratio,
                                   self.constraints.max_protein_ratio)
            carb_ratio = np.clip(carb_ratio,
                                self.constraints.min_carb_ratio,
                                self.constraints.max_carb_ratio)
            fat_ratio = 1 - protein_ratio - carb_ratio
            
            # 生成运动参数
            cardio_freq = np.random.randint(
                self.constraints.min_cardio_freq,
                self.constraints.max_cardio_freq + 1
            )
            cardio_duration = np.random.choice(self.constraints.cardio_duration_options)
            strength_freq = np.random.randint(
                self.constraints.min_strength_freq,
                self.constraints.max_strength_freq + 1
            )
            
            # 生成睡眠时间
            sleep_hours = np.random.uniform(
                self.constraints.min_sleep_hours,
                self.constraints.max_sleep_hours
            )
            
            solution = np.array([
                calories, protein_ratio, carb_ratio, fat_ratio,
                cardio_freq, cardio_duration, strength_freq, sleep_hours
            ])
            
            # 验证方案
            if self.validate_solution(solution):
                return solution
        
        # 如果无法生成有效方案，返回一个保守的方案
        logger.warning("无法生成满足所有约束的随机方案，返回默认方案")
        return self.generate_from_template("balanced", person_tdee)
    
    def generate_from_template(self, template_name: str, person_tdee: float) -> np.ndarray:
        """从模板生成方案"""
        if template_name not in self.templates:
            logger.error(f"未知的模板: {template_name}")
            template_name = "balanced"
        
        template = self.templates[template_name]
        
        calories = person_tdee * template["calories_factor"]
        calories = np.clip(calories, self.constraints.min_calories, self.constraints.max_calories)
        
        solution = np.array([
            calories,
            template["protein_ratio"],
            template["carb_ratio"],
            template["fat_ratio"],
            template["cardio_freq"],
            template["cardio_duration"],
            template["strength_freq"],
            template["sleep_hours"]
        ])
        
        return solution
    
    def generate_diverse_population(self, population_size: int, person_tdee: float) -> List[np.ndarray]:
        """生成多样化的初始种群"""
        population = []
        
        # 首先添加所有模板方案
        for template_name in self.templates:
            if len(population) < population_size:
                solution = self.generate_from_template(template_name, person_tdee)
                population.append(solution)
        
        # 然后生成随机方案
        while len(population) < population_size:
            solution = self.generate_random_solution(person_tdee)
            
            # 确保方案之间有足够的差异
            if self._is_diverse_enough(solution, population):
                population.append(solution)
        
        logger.info(f"生成了 {len(population)} 个多样化的初始方案")
        return population
    
    def _is_diverse_enough(self, new_solution: np.ndarray, 
                           existing_solutions: List[np.ndarray],
                           min_distance: float = 0.1) -> bool:
        """检查新方案是否与现有方案有足够的差异"""
        if not existing_solutions:
            return True
        
        # 归一化后计算距离
        new_norm = self._normalize_solution(new_solution)
        
        for existing in existing_solutions:
            existing_norm = self._normalize_solution(existing)
            distance = np.linalg.norm(new_norm - existing_norm)
            
            if distance < min_distance:
                return False
        
        return True
    
    def _normalize_solution(self, solution: np.ndarray) -> np.ndarray:
        """归一化方案向量（用于比较）"""
        norm_solution = solution.copy()
        
        # 归一化各个维度到[0,1]
        # 热量
        norm_solution[0] = (solution[0] - self.constraints.min_calories) / \
                          (self.constraints.max_calories - self.constraints.min_calories)
        
        # 营养素比例已经在[0,1]范围内
        
        # 运动频率
        norm_solution[4] = solution[4] / self.constraints.max_cardio_freq
        norm_solution[5] = solution[5] / max(self.constraints.cardio_duration_options)
        norm_solution[6] = solution[6] / self.constraints.max_strength_freq
        
        # 睡眠
        norm_solution[7] = (solution[7] - self.constraints.min_sleep_hours) / \
                          (self.constraints.max_sleep_hours - self.constraints.min_sleep_hours)
        
        return norm_solution
    
    def mutate_solution(self, solution: np.ndarray, mutation_rate: float = 0.3) -> np.ndarray:
        """对方案进行小幅度变异（用于局部搜索）"""
        mutated = solution.copy()
        
        for i in range(len(solution)):
            if np.random.rand() < mutation_rate:
                # 根据不同维度采用不同的变异策略
                if i == 0:  # 热量
                    mutated[i] += np.random.normal(0, 50)  # ±50卡的变化
                elif i in [1, 2, 3]:  # 营养素比例
                    mutated[i] += np.random.normal(0, 0.05)  # ±5%的变化
                elif i in [4, 6]:  # 频率
                    mutated[i] += np.random.randint(-1, 2)  # ±1次的变化
                elif i == 5:  # 有氧时长
                    # 随机选择相邻的时长选项
                    current_idx = self.constraints.cardio_duration_options.index(int(solution[i]))
                    new_idx = np.clip(current_idx + np.random.randint(-1, 2), 
                                     0, len(self.constraints.cardio_duration_options) - 1)
                    mutated[i] = self.constraints.cardio_duration_options[new_idx]
                elif i == 7:  # 睡眠
                    mutated[i] += np.random.normal(0, 0.25)  # ±15分钟的变化
        
        # 规范化结果
        return self.normalize_solution_vector(mutated)
    
    def encode_solution(self, solution: np.ndarray) -> str:
        """将方案编码为字符串（便于存储和传输）"""
        solution_dict = {
            "calories": float(solution[0]),
            "protein_ratio": float(solution[1]),
            "carb_ratio": float(solution[2]),
            "fat_ratio": float(solution[3]),
            "cardio_freq": int(solution[4]),
            "cardio_duration": int(solution[5]),
            "strength_freq": int(solution[6]),
            "sleep_hours": float(solution[7])
        }
        return json.dumps(solution_dict, ensure_ascii=False)
    
    def decode_solution(self, encoded: str) -> np.ndarray:
        """从字符串解码方案"""
        solution_dict = json.loads(encoded)
        return np.array([
            solution_dict["calories"],
            solution_dict["protein_ratio"],
            solution_dict["carb_ratio"],
            solution_dict["fat_ratio"],
            solution_dict["cardio_freq"],
            solution_dict["cardio_duration"],
            solution_dict["strength_freq"],
            solution_dict["sleep_hours"]
        ])
    
    def generate_plateau_breaking_solutions(self, current_solution: np.ndarray, 
                                          num_solutions: int = 5) -> List[np.ndarray]:
        """生成突破平台期的方案变体"""
        plateau_breakers = []
        
        # 策略1：热量循环（Calorie Cycling）
        high_cal = current_solution.copy()
        high_cal[0] *= 1.15  # 提高15%热量
        plateau_breakers.append(high_cal)
        
        low_cal = current_solution.copy()
        low_cal[0] *= 0.85  # 降低15%热量
        plateau_breakers.append(low_cal)
        
        # 策略2：碳水循环（Carb Cycling）
        high_carb = current_solution.copy()
        high_carb[2] = min(0.50, high_carb[2] * 1.3)  # 提高碳水
        high_carb[3] = 1 - high_carb[1] - high_carb[2]  # 调整脂肪
        plateau_breakers.append(high_carb)
        
        # 策略3：增加运动强度
        high_exercise = current_solution.copy()
        high_exercise[4] = min(self.constraints.max_cardio_freq, high_exercise[4] + 1)
        high_exercise[6] = min(self.constraints.max_strength_freq, high_exercise[6] + 1)
        plateau_breakers.append(high_exercise)
        
        # 策略4：间歇性禁食模拟（通过降低总热量）
        if_solution = current_solution.copy()
        if_solution[0] *= 0.75  # 模拟16:8间歇性禁食
        plateau_breakers.append(if_solution)
        
        # 验证并返回有效方案
        valid_solutions = []
        for solution in plateau_breakers[:num_solutions]:
            if self.validate_solution(solution):
                valid_solutions.append(solution)
            else:
                # 如果无效，生成一个随机变异
                mutated = self.mutate_solution(current_solution, mutation_rate=0.5)
                if self.validate_solution(mutated):
                    valid_solutions.append(mutated)
        
        return valid_solutions

    def generate_with_disabled_components(self, person_tdee: float, 
                                        disabled_components: List[str]) -> np.ndarray:
        """生成考虑禁用组件的方案"""
        solution = self.generate_random_solution(person_tdee)
        
        # 根据禁用的组件固定相应的值
        for component in disabled_components:
            if component == 'sleep_optimization':
                solution[7] = 7.0  # 固定睡眠7小时
            elif component == 'strength_training':
                solution[6] = 0  # 无力量训练
            elif component == 'cardio_training':
                solution[4] = 0  # 无有氧训练
                solution[5] = 0  # 无有氧时长
            elif component == 'nutrition_optimization':
                # 固定营养比例
                solution[1] = 0.30  # 蛋白质30%
                solution[2] = 0.40  # 碳水40%
                solution[3] = 0.30  # 脂肪30%
        
        return solution

class SolutionLibrary:
    """方案库管理"""
    
    def __init__(self, filepath: str = "solution_library.json"):
        self.filepath = filepath
        self.library = self._load_library()
        
    def _load_library(self) -> Dict:
        """加载方案库"""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"solutions": [], "metadata": {}}
    
    def save_solution(self, solution: np.ndarray, 
                     performance: Dict,
                     tags: List[str] = None):
        """保存成功的方案"""
        generator = SolutionGenerator()
        
        solution_record = {
            "id": len(self.library["solutions"]) + 1,
            "solution": generator.encode_solution(solution),
            "performance": performance,
            "tags": tags or [],
            "timestamp": str(np.datetime64('now'))
        }
        
        self.library["solutions"].append(solution_record)
        self._save_library()
        
        logger.info(f"保存方案 #{solution_record['id']} 到方案库")
    
    def get_top_solutions(self, n: int = 10, 
                         metric: str = "total_weight_loss") -> List[Dict]:
        """获取表现最好的方案"""
        sorted_solutions = sorted(
            self.library["solutions"],
            key=lambda x: x["performance"].get(metric, 0),
            reverse=True
        )
        return sorted_solutions[:n]
    
    def _save_library(self):
        """保存方案库"""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.library, f, ensure_ascii=False, indent=2)