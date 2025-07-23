"""
适应度评估模块
计算肌肉流失率、体脂下降率、可持续性评分和综合适应度
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FitnessWeights:
    """适应度权重配置"""
    muscle_loss_weight: float = 0.4  # 肌肉流失权重
    fat_loss_weight: float = 0.4     # 体脂下降权重
    sustainability_weight: float = 0.2  # 可持续性权重
    
    def normalize(self):
        """归一化权重"""
        total = self.muscle_loss_weight + self.fat_loss_weight + self.sustainability_weight
        self.muscle_loss_weight /= total
        self.fat_loss_weight /= total
        self.sustainability_weight /= total


class FitnessEvaluator:
    """适应度评估器"""
    
    def __init__(self, weights: Optional[FitnessWeights] = None):
        self.weights = weights or FitnessWeights()
        self.weights.normalize()
        
    def calculate_muscle_loss_score(self, muscle_loss_rate: float) -> float:
        """
        计算肌肉流失评分（0-1，越低越好）
        muscle_loss_rate: 肌肉流失占总减重的比例
        """
        # 理想情况：肌肉流失率应该低于10%
        # 使用sigmoid函数进行平滑映射
        ideal_rate = 0.1
        
        if muscle_loss_rate <= ideal_rate:
            score = muscle_loss_rate / ideal_rate * 0.5
        else:
            # 超过理想值后，分数快速增加
            excess = muscle_loss_rate - ideal_rate
            score = 0.5 + (1 - np.exp(-excess * 5)) * 0.5
        
        return min(1.0, score)
    
    def calculate_fat_loss_score(self, fat_loss_rate: float, total_weight_loss: float) -> float:
        """
        计算体脂下降评分（0-1，越低越好，因为我们要最大化脂肪流失）
        fat_loss_rate: 脂肪流失占总减重的比例
        total_weight_loss: 总减重量（kg/周）
        """
        # 理想的每周减重：0.5-1kg，其中90%应该是脂肪
        ideal_weekly_loss = 0.75  # kg
        ideal_fat_ratio = 0.9
        
        # 评估减重速度
        if total_weight_loss <= 0:
            speed_score = 1.0  # 没有减重，最差
        elif total_weight_loss <= ideal_weekly_loss:
            speed_score = 1 - (total_weight_loss / ideal_weekly_loss) * 0.7
        else:
            # 减重过快也不好
            excess = total_weight_loss - ideal_weekly_loss
            speed_score = 0.3 + (1 - np.exp(-excess * 2)) * 0.7
        
        # 评估脂肪流失比例
        if fat_loss_rate >= ideal_fat_ratio:
            ratio_score = 0  # 理想情况
        else:
            ratio_score = (ideal_fat_ratio - fat_loss_rate) / ideal_fat_ratio
        
        # 综合评分
        score = 0.6 * speed_score + 0.4 * ratio_score
        
        return min(1.0, score)
    
    def calculate_sustainability_score(self, solution, person, energy_deficit: float) -> float:
        """
        计算可持续性评分（0-1，越低越好）
        基于多个因素评估方案的可持续性
        """
        scores = []
        
        # 1. 热量限制程度（过度限制难以坚持）
        deficit_ratio = energy_deficit / (energy_deficit + solution.calories)
        if deficit_ratio <= 0.2:  # 20%以内的缺口
            calorie_score = deficit_ratio / 0.2 * 0.3
        elif deficit_ratio <= 0.35:  # 20-35%的缺口
            calorie_score = 0.3 + (deficit_ratio - 0.2) / 0.15 * 0.4
        else:  # 超过35%，非常难坚持
            calorie_score = 0.7 + (deficit_ratio - 0.35) / 0.15 * 0.3
        scores.append(min(1.0, calorie_score))
        
        # 2. 运动强度（过多运动导致疲劳）
        total_exercise_hours = (solution.cardio_freq * solution.cardio_duration / 60 + 
                               solution.strength_freq * 1.0)  # 假设力量训练1小时
        
        if total_exercise_hours <= 5:  # 每周5小时以内
            exercise_score = total_exercise_hours / 5 * 0.4
        elif total_exercise_hours <= 10:  # 5-10小时
            exercise_score = 0.4 + (total_exercise_hours - 5) / 5 * 0.4
        else:  # 超过10小时
            exercise_score = 0.8 + (total_exercise_hours - 10) / 5 * 0.2
        scores.append(min(1.0, exercise_score))
        
        # 3. 睡眠质量（睡眠不足影响恢复和意志力）
        if solution.sleep_hours >= 7.5:
            sleep_score = 0.1
        elif solution.sleep_hours >= 7:
            sleep_score = 0.1 + (7.5 - solution.sleep_hours) / 0.5 * 0.3
        else:
            sleep_score = 0.4 + (7 - solution.sleep_hours) / 0.5 * 0.6
        scores.append(min(1.0, sleep_score))
        
        # 4. 饮食限制程度（极端的营养素比例难以坚持）
        # 蛋白质过高
        protein_score = 0
        if solution.protein_ratio > 0.35:
            protein_score = (solution.protein_ratio - 0.35) / 0.1 * 0.3
        
        # 碳水过低
        carb_score = 0
        if solution.carb_ratio < 0.35:
            carb_score = (0.35 - solution.carb_ratio) / 0.15 * 0.4
        
        diet_restriction_score = min(1.0, protein_score + carb_score)
        scores.append(diet_restriction_score)
        
        # 5. 时间因素（长期执行同一方案会降低依从性）
        if person.weeks_on_diet < 4:
            time_score = 0.1
        elif person.weeks_on_diet < 8:
            time_score = 0.1 + (person.weeks_on_diet - 4) / 4 * 0.3
        else:
            time_score = 0.4 + (person.weeks_on_diet - 8) / 8 * 0.6
        scores.append(min(1.0, time_score))
        
        # 综合评分（加权平均）
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # 各因素权重
        sustainability_score = sum(s * w for s, w in zip(scores, weights))
        
        logger.debug(f"可持续性评分详情: 热量={scores[0]:.2f}, 运动={scores[1]:.2f}, "
                    f"睡眠={scores[2]:.2f}, 饮食={scores[3]:.2f}, 时间={scores[4]:.2f}, "
                    f"综合={sustainability_score:.2f}")
        
        return sustainability_score
    
    def calculate_fitness(self, simulation_results: Dict, solution, person) -> Tuple[float, Dict]:
        """
        计算综合适应度值
        返回：(适应度值, 各组成部分的详细信息)
        """
        # 提取所需数据
        muscle_loss_rate = simulation_results['muscle_loss_rate']
        fat_loss_rate = simulation_results['fat_loss_rate']
        total_weight_loss = simulation_results['total_weight_loss']
        energy_deficit = simulation_results['energy_deficit']
        
        # 计算各项评分
        muscle_score = self.calculate_muscle_loss_score(muscle_loss_rate)
        fat_score = self.calculate_fat_loss_score(fat_loss_rate, total_weight_loss)
        sustainability_score = self.calculate_sustainability_score(solution, person, energy_deficit)
        
        # 加权计算总适应度
        fitness = (self.weights.muscle_loss_weight * muscle_score +
                  self.weights.fat_loss_weight * fat_score +
                  self.weights.sustainability_weight * sustainability_score)
        
        # 特殊情况处理
        # 1. 如果完全没有减重，增加惩罚
        if total_weight_loss <= 0:
            fitness += 0.5
        
        # 2. 如果肌肉流失过多（超过30%），增加惩罚
        if muscle_loss_rate > 0.3:
            fitness += 0.3
        
        # 3. 如果违反基本健康原则，增加惩罚
        if solution.calories < 1200:  # 热量过低
            fitness += 0.5
        
        # 确保适应度值在合理范围内
        fitness = max(0, min(10, fitness))
        
        # 返回详细信息
        components = {
            'muscle_loss_score': muscle_score,
            'fat_loss_score': fat_score,
            'sustainability_score': sustainability_score,
            'muscle_loss_rate': muscle_loss_rate,
            'fat_loss_rate': fat_loss_rate,
            'total_weight_loss': total_weight_loss,
            'weighted_fitness': fitness
        }
        
        logger.info(f"适应度计算: 肌肉评分={muscle_score:.3f}, "
                   f"脂肪评分={fat_score:.3f}, 可持续性={sustainability_score:.3f}, "
                   f"总适应度={fitness:.3f}")
        
        return fitness, components


class AdaptiveFitnessEvaluator(FitnessEvaluator):
    """自适应适应度评估器（根据不同阶段调整权重）"""
    
    def __init__(self):
        super().__init__()
        
    def update_weights(self, person, week: int):
        """根据减肥阶段动态调整权重"""
        # 初期（前4周）：重视快速减脂
        if week < 4:
            self.weights.muscle_loss_weight = 0.3
            self.weights.fat_loss_weight = 0.5
            self.weights.sustainability_weight = 0.2
        
        # 中期（4-8周）：平衡各方面
        elif week < 8:
            self.weights.muscle_loss_weight = 0.4
            self.weights.fat_loss_weight = 0.4
            self.weights.sustainability_weight = 0.2
        
        # 后期（8周后）：重视肌肉保护和可持续性
        else:
            self.weights.muscle_loss_weight = 0.5
            self.weights.fat_loss_weight = 0.25
            self.weights.sustainability_weight = 0.25
        
        # 特殊情况：如果体脂率已经很低，更重视肌肉保护
        if person.body_fat_percentage < 15 and person.gender == 'male':
            self.weights.muscle_loss_weight = 0.6
            self.weights.fat_loss_weight = 0.2
            self.weights.sustainability_weight = 0.2
        elif person.body_fat_percentage < 23 and person.gender == 'female':
            self.weights.muscle_loss_weight = 0.6
            self.weights.fat_loss_weight = 0.2
            self.weights.sustainability_weight = 0.2
        
        self.weights.normalize()
        
        logger.info(f"更新权重（第{week}周）: 肌肉={self.weights.muscle_loss_weight:.2f}, "
                   f"脂肪={self.weights.fat_loss_weight:.2f}, "
                   f"可持续性={self.weights.sustainability_weight:.2f}")
    
    def calculate_plateau_penalty(self, person, week: int) -> float:
        """计算平台期惩罚（鼓励突破平台期）"""
        # 如果连续两周体重变化小于0.2kg，认为是平台期
        if hasattr(person, 'weight_history') and len(person.weight_history) >= 2:
            recent_change = abs(person.weight_history[-1] - person.weight_history[-2])
            if recent_change < 0.2:
                # 根据平台期持续时间增加惩罚
                plateau_weeks = getattr(person, 'plateau_weeks', 0) + 1
                penalty = min(0.5, plateau_weeks * 0.1)
                logger.info(f"检测到平台期（第{plateau_weeks}周），增加惩罚: {penalty:.2f}")
                return penalty
        
        return 0
    
    def calculate_fitness(self, simulation_results: Dict, solution, person) -> Tuple[float, Dict]:
        """计算适应度（包含自适应调整）"""
        # 获取当前周数
        week = person.weeks_on_diet
        
        # 更新权重
        self.update_weights(person, week)
        
        # 计算基础适应度
        base_fitness, components = super().calculate_fitness(simulation_results, solution, person)
        
        # 添加平台期惩罚
        plateau_penalty = self.calculate_plateau_penalty(person, week)
        adjusted_fitness = base_fitness + plateau_penalty
        
        components['plateau_penalty'] = plateau_penalty
        components['adjusted_fitness'] = adjusted_fitness
        
        return adjusted_fitness, components