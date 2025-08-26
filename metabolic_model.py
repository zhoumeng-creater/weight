"""
人体代谢模型
实现BMR计算、TDEE计算、代谢适应模拟和体重变化预测
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import logging
from config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class PersonProfile:
    """个人档案"""
    age: int
    gender: str  # 'male' or 'female'
    height: float  # cm
    weight: float  # kg
    body_fat_percentage: float  # 百分比，如25表示25%
    activity_level: float  # 活动系数，1.2-1.9
    weeks_on_diet: int = 0  # 已经进行减肥的周数
    
    # 代谢适应相关参数
    initial_weight: float = field(init=False)
    initial_bmr: float = field(init=False)
    metabolic_adaptation_factor: float = 1.0  # 代谢适应因子
    
    def __post_init__(self):
        self.initial_weight = self.weight
        self.initial_bmr = self._calculate_bmr()
    
    def _calculate_bmr(self) -> float:
        """使用Mifflin-St Jeor方程计算基础代谢率"""
        if self.gender.lower() == 'male':
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        return bmr
    
    @property
    def lean_body_mass(self) -> float:
        """瘦体重（kg）"""
        return self.weight * (1 - self.body_fat_percentage / 100)
    
    @property
    def fat_mass(self) -> float:
        """脂肪重量（kg）"""
        return self.weight * (self.body_fat_percentage / 100)


class MetabolicModel:
    """人体代谢模型"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        # 能量转换常数
        self.CALORIES_PER_KG_FAT = 7700  # 1kg脂肪约等于7700千卡
        self.CALORIES_PER_KG_MUSCLE = 1800  # 1kg肌肉约等于1800千卡
        
        # 使用配置中的代谢适应参数
        self.ADAPTATION_RATE = self.config.metabolic.adaptation_rate if self.config else 0.05
        self.MIN_ADAPTATION_FACTOR = self.config.metabolic.max_adaptation_factor if self.config else 0.75
        self.enable_adaptation = self.config.metabolic.enable_metabolic_adaptation if self.config else True
        
    def calculate_bmr(self, person: PersonProfile) -> float:
        """计算当前BMR（考虑代谢适应）"""
        # 基础BMR（Mifflin-St Jeor方程）
        if person.gender.lower() == 'male':
            base_bmr = 10 * person.weight + 6.25 * person.height - 5 * person.age + 5
        else:
            base_bmr = 10 * person.weight + 6.25 * person.height - 5 * person.age - 161
        
        # 考虑体重变化的影响
        weight_factor = (person.weight / person.initial_weight) ** 0.75
        
        # 应用代谢适应
        adapted_bmr = base_bmr * weight_factor * person.metabolic_adaptation_factor
        
        logger.debug(f"BMR计算: 基础={base_bmr:.0f}, 体重因子={weight_factor:.3f}, "
                    f"适应因子={person.metabolic_adaptation_factor:.3f}, "
                    f"最终={adapted_bmr:.0f}")
        
        return adapted_bmr
    
    def calculate_tdee(self, person: PersonProfile, solution) -> float:
        """计算每日总能量消耗"""
        bmr = self.calculate_bmr(person)
        
        # 计算运动消耗
        # 有氧运动：假设中等强度，每分钟消耗体重(kg)*0.1千卡
        cardio_calories_per_week = (solution.cardio_freq * solution.cardio_duration * 
                                   person.weight * 0.1)
        
        # 力量训练：假设每次训练消耗300千卡
        strength_calories_per_week = solution.strength_freq * 300
        
        # 每日运动消耗
        daily_exercise_calories = (cardio_calories_per_week + strength_calories_per_week) / 7
        
        # TDEE = BMR * 活动系数 + 运动消耗
        tdee = bmr * person.activity_level + daily_exercise_calories
        
        # 睡眠对代谢的影响（睡眠不足会降低代谢）
        sleep_factor = 1.0
        if solution.sleep_hours < 7:
            sleep_factor = 0.95 + (solution.sleep_hours - 6) * 0.05
        tdee *= sleep_factor
        
        logger.debug(f"TDEE计算: BMR={bmr:.0f}, 运动消耗={daily_exercise_calories:.0f}, "
                    f"睡眠因子={sleep_factor:.3f}, 总计={tdee:.0f}")
        
        return tdee
    
    def calculate_metabolic_adaptation(self, person: PersonProfile, 
                                     energy_deficit: float, week: int) -> float:
        """计算代谢适应"""
        if not self.enable_adaptation:
            return 1.0  # 返回1.0表示没有适应
        
        # 基于能量缺口和时间的代谢适应
        if energy_deficit > 0:
            # 每周根据能量缺口程度进行适应
            deficit_percentage = energy_deficit / self.calculate_bmr(person)
            weekly_adaptation = self.ADAPTATION_RATE * deficit_percentage
            
            # 累积适应效应
            new_factor = person.metabolic_adaptation_factor * (1 - weekly_adaptation)
            
            # 确保不低于最小值
            new_factor = max(new_factor, self.MIN_ADAPTATION_FACTOR)
            
            logger.debug(f"代谢适应: 缺口比例={deficit_percentage:.2%}, "
                        f"周适应={weekly_adaptation:.3f}, 新因子={new_factor:.3f}")
            
            return new_factor
        
        return person.metabolic_adaptation_factor
    
    def predict_body_composition_changes(self, person: PersonProfile, 
                                       energy_deficit: float, 
                                       solution) -> Tuple[float, float]:
        """预测体成分变化（脂肪和肌肉）"""
        if energy_deficit <= 0:
            return 0, 0
        
        # 蛋白质摄入对肌肉保护的影响
        protein_grams = solution.calories * solution.protein_ratio / 4
        protein_per_kg = protein_grams / person.weight
        
        # 肌肉保护因子（蛋白质摄入越高，肌肉流失越少）
        muscle_protection = min(1.0, protein_per_kg / 2.2)  # 2.2g/kg为理想值
        
        # 力量训练对肌肉保护的影响
        strength_protection = min(1.0, solution.strength_freq / 3)  # 3次/周为理想值
        
        # 总肌肉保护因子
        total_protection = 0.6 * muscle_protection + 0.4 * strength_protection
        
        # 能量缺口分配
        # 基础：75%来自脂肪，25%来自肌肉
        base_fat_ratio = 0.75
        base_muscle_ratio = 0.25
        
        # 根据保护因子调整
        actual_muscle_ratio = base_muscle_ratio * (1 - total_protection)
        actual_fat_ratio = 1 - actual_muscle_ratio
        
        # 计算实际损失
        total_energy_deficit_week = energy_deficit * 7
        fat_loss = (total_energy_deficit_week * actual_fat_ratio) / self.CALORIES_PER_KG_FAT
        muscle_loss = (total_energy_deficit_week * actual_muscle_ratio) / self.CALORIES_PER_KG_MUSCLE
        
        logger.debug(f"体成分变化: 蛋白质={protein_per_kg:.2f}g/kg, "
                    f"保护因子={total_protection:.2f}, "
                    f"脂肪损失={fat_loss:.3f}kg, 肌肉损失={muscle_loss:.3f}kg")
        
        return fat_loss, muscle_loss
    
    def simulate_week(self, person: PersonProfile, solution, week: int) -> Dict:
        """模拟一周的执行结果"""
        # 计算TDEE
        tdee = self.calculate_tdee(person, solution)
        
        # 计算能量缺口
        energy_deficit = tdee - solution.calories
        
        # 预测体成分变化
        fat_loss, muscle_loss = self.predict_body_composition_changes(
            person, energy_deficit, solution
        )
        
        # 计算新体重
        weight_loss = fat_loss + muscle_loss
        new_weight = person.weight - weight_loss
        
        # 计算新体脂率
        new_fat_mass = person.fat_mass - fat_loss
        new_lean_mass = person.lean_body_mass - muscle_loss
        new_body_fat_percentage = (new_fat_mass / new_weight) * 100
        
        # 计算代谢适应
        new_adaptation_factor = self.calculate_metabolic_adaptation(
            person, energy_deficit, week
        )
        
        results = {
            'tdee': tdee,
            'energy_deficit': energy_deficit,
            'fat_loss': fat_loss,
            'muscle_loss': muscle_loss,
            'total_weight_loss': weight_loss,
            'final_weight': new_weight,
            'final_body_fat_percentage': new_body_fat_percentage,
            'metabolic_adaptation_factor': new_adaptation_factor,
            'muscle_loss_rate': muscle_loss / weight_loss if weight_loss > 0 else 0,
            'fat_loss_rate': fat_loss / weight_loss if weight_loss > 0 else 0
        }
        
        return results
    
    def update_person_state(self, person: PersonProfile, solution, week: int) -> PersonProfile:
        """更新人体状态（用于模拟多周）"""
        results = self.simulate_week(person, solution, week)
        
        # 创建新的PersonProfile
        updated_person = PersonProfile(
            age=person.age,
            gender=person.gender,
            height=person.height,
            weight=results['final_weight'],
            body_fat_percentage=results['final_body_fat_percentage'],
            activity_level=person.activity_level,
            weeks_on_diet=person.weeks_on_diet + 1
        )
        
        # 保留初始值和适应因子
        updated_person.initial_weight = person.initial_weight
        updated_person.initial_bmr = person.initial_bmr
        updated_person.metabolic_adaptation_factor = results['metabolic_adaptation_factor']
        
        return updated_person
    
    def calculate_plateau_risk(self, person: PersonProfile, solution) -> float:
        """计算进入平台期的风险（0-1）"""
        risk_factors = []
        
        # 1. 代谢适应程度
        adaptation_risk = 1 - person.metabolic_adaptation_factor
        risk_factors.append(adaptation_risk * 0.3)
        
        # 2. 减重时长
        time_risk = min(1.0, person.weeks_on_diet / 12)  # 12周后风险最大
        risk_factors.append(time_risk * 0.2)
        
        # 3. 能量缺口过大
        tdee = self.calculate_tdee(person, solution)
        deficit_ratio = (tdee - solution.calories) / tdee
        if deficit_ratio > 0.25:  # 缺口超过25%
            deficit_risk = min(1.0, (deficit_ratio - 0.25) / 0.25)
            risk_factors.append(deficit_risk * 0.2)
        else:
            risk_factors.append(0)
        
        # 4. 体重下降程度
        weight_loss_ratio = (person.initial_weight - person.weight) / person.initial_weight
        if weight_loss_ratio > 0.1:  # 减重超过10%
            weight_risk = min(1.0, (weight_loss_ratio - 0.1) / 0.2)
            risk_factors.append(weight_risk * 0.3)
        else:
            risk_factors.append(0)
        
        total_risk = sum(risk_factors)
        
        logger.debug(f"平台期风险评估: 适应={adaptation_risk:.2f}, 时长={time_risk:.2f}, "
                    f"缺口={risk_factors[2]:.2f}, 减重={risk_factors[3]:.2f}, "
                    f"总风险={total_risk:.2f}")
        
        return total_risk


class AdvancedMetabolicModel(MetabolicModel):
    """高级代谢模型（包含更多生理细节）"""
    
    def __init__(self):
        super().__init__()
        
        # 激素影响参数
        self.LEPTIN_SENSITIVITY = 0.1  # 瘦素敏感度
        self.CORTISOL_FACTOR = 0.05  # 皮质醇影响因子
        self.THYROID_FACTOR = 0.1  # 甲状腺激素影响因子
        
    def calculate_hormonal_effects(self, person: PersonProfile, 
                                  solution, energy_deficit: float) -> Dict[str, float]:
        """计算激素对代谢的影响"""
        effects = {}
        
        # 瘦素水平（随体脂下降而下降）
        fat_loss_ratio = (person.initial_weight * person.body_fat_percentage / 100 - 
                         person.fat_mass) / (person.initial_weight * person.body_fat_percentage / 100)
        leptin_effect = 1 - (fat_loss_ratio * self.LEPTIN_SENSITIVITY)
        effects['leptin'] = max(0.8, leptin_effect)
        
        # 皮质醇水平（压力激素，缺口过大时升高）
        deficit_stress = energy_deficit / self.calculate_bmr(person)
        if deficit_stress > 0.3:  # 缺口超过30%
            cortisol_effect = 1 - (deficit_stress - 0.3) * self.CORTISOL_FACTOR
        else:
            cortisol_effect = 1.0
        effects['cortisol'] = max(0.9, cortisol_effect)
        
        # 甲状腺激素（长期限制热量会下降）
        if person.weeks_on_diet > 4:
            thyroid_effect = 1 - ((person.weeks_on_diet - 4) / 52) * self.THYROID_FACTOR
        else:
            thyroid_effect = 1.0
        effects['thyroid'] = max(0.85, thyroid_effect)
        
        # 睡眠不足对激素的影响
        if solution.sleep_hours < 7:
            sleep_hormone_effect = 0.95 + (solution.sleep_hours - 6) * 0.05
            for hormone in effects:
                effects[hormone] *= sleep_hormone_effect
        
        return effects
    
    def calculate_neat(self, person: PersonProfile, energy_deficit: float) -> float:
        """计算非运动性活动消耗（NEAT）的变化"""
        # NEAT会随着能量缺口增大而自适应性降低
        if energy_deficit > 0:
            deficit_ratio = energy_deficit / self.calculate_bmr(person)
            neat_reduction = min(0.2, deficit_ratio * 0.3)  # 最多降低20%
            neat_factor = 1 - neat_reduction
        else:
            neat_factor = 1.0
        
        return neat_factor
    
    def calculate_tdee(self, person: PersonProfile, solution) -> float:
        """计算TDEE（包含更多因素）"""
        base_tdee = super().calculate_tdee(person, solution)
        
        # 计算激素影响
        energy_deficit = base_tdee - solution.calories
        hormonal_effects = self.calculate_hormonal_effects(person, solution, energy_deficit)
        hormonal_factor = np.mean(list(hormonal_effects.values()))
        
        # 计算NEAT影响
        neat_factor = self.calculate_neat(person, energy_deficit)
        
        # 综合调整
        adjusted_tdee = base_tdee * hormonal_factor * neat_factor
        
        logger.debug(f"高级TDEE计算: 基础={base_tdee:.0f}, "
                    f"激素因子={hormonal_factor:.3f}, NEAT因子={neat_factor:.3f}, "
                    f"调整后={adjusted_tdee:.0f}")
        
        return adjusted_tdee