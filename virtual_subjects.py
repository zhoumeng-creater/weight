"""
虚拟实验对象生成器模块
用于生成各种类型的虚拟实验对象，支持标准人群、平台期人群和边界案例
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from metabolic_model import PersonProfile

logger = logging.getLogger(__name__)


class VirtualSubjectGenerator:
    """虚拟实验对象生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化生成器
        
        Args:
            seed: 随机种子，用于确保结果可重复
        """
        np.random.seed(seed)
        self.subjects_created = 0
        
        # 预定义的人群特征分布
        self.age_groups = {
            "young": (20, 35),
            "middle": (35, 50),
            "senior": (50, 65)
        }
        
        self.bmi_categories = {
            "normal": (18.5, 25.0),
            "overweight": (25.0, 30.0),
            "obese_1": (30.0, 35.0),
            "obese_2": (35.0, 40.0)
        }
        
        self.activity_levels = {
            "sedentary": 1.2,
            "lightly_active": 1.375,
            "moderately_active": 1.55,
            "very_active": 1.725,
            "extremely_active": 1.9
        }
        
    def generate_standard_subjects(self, n_subjects: int = 100) -> List[PersonProfile]:
        """
        生成标准虚拟人群体
        
        Args:
            n_subjects: 生成的虚拟人数量
            
        Returns:
            虚拟人列表
        """
        subjects = []
        
        for i in range(n_subjects):
            # 随机选择年龄组
            age_group = np.random.choice(list(self.age_groups.keys()))
            age = np.random.randint(*self.age_groups[age_group])
            
            # 随机性别
            gender = np.random.choice(['male', 'female'])
            
            # 根据性别设置身高
            if gender == 'male':
                height = np.random.normal(175, 7)  # 男性平均身高175cm
            else:
                height = np.random.normal(162, 6)  # 女性平均身高162cm
            
            # 随机选择BMI类别
            bmi_category = np.random.choice(list(self.bmi_categories.keys()), 
                                          p=[0.1, 0.35, 0.35, 0.2])  # 权重分布
            bmi = np.random.uniform(*self.bmi_categories[bmi_category])
            
            # 根据BMI计算体重
            weight = bmi * (height / 100) ** 2
            
            # 根据BMI和性别估算体脂率
            if gender == 'male':
                body_fat = 1.2 * bmi + 0.23 * age - 16.2
            else:
                body_fat = 1.2 * bmi + 0.23 * age - 5.4
            
            # 添加一些随机性
            body_fat = np.clip(body_fat + np.random.normal(0, 2), 5, 50)
            
            # 随机活动水平
            activity_level = np.random.choice(list(self.activity_levels.values()))
            
            # 随机减肥历史
            weeks_on_diet = np.random.choice([0, 4, 8, 12, 16], p=[0.3, 0.2, 0.2, 0.2, 0.1])
            
            # 创建虚拟人
            subject = PersonProfile(
                age=int(age),
                gender=gender,
                height=height,
                weight=weight,
                body_fat_percentage=body_fat,
                activity_level=activity_level,
                weeks_on_diet=weeks_on_diet
            )
            
            subjects.append(subject)
            self.subjects_created += 1
        
        logger.info(f"生成了 {n_subjects} 个标准虚拟人")
        return subjects
    
    def generate_diverse_population(self, n_subjects: int = 120) -> List[PersonProfile]:
        """
        生成多样化的虚拟人群体（用于全面测试）
        
        Args:
            n_subjects: 生成的虚拟人数量
            
        Returns:
            多样化的虚拟人列表
        """
        subjects = []
        
        # 确保覆盖各种组合
        age_values = [25, 35, 45, 55]
        gender_values = ['male', 'female']
        bmi_values = [24, 27, 32, 37]
        activity_values = [1.2, 1.375, 1.55, 1.725]
        diet_history_values = [0, 8, 16]
        
        # 生成所有可能的组合
        from itertools import product
        
        combinations = list(product(age_values, gender_values, bmi_values, 
                                   activity_values, diet_history_values))
        
        # 如果组合数超过需求，随机采样
        if len(combinations) > n_subjects:
            selected_indices = np.random.choice(len(combinations), n_subjects, replace=False)
            combinations = [combinations[i] for i in selected_indices]
        
        for age, gender, bmi, activity, weeks_on_diet in combinations[:n_subjects]:
            # 根据性别设置身高
            height = 175 if gender == 'male' else 162
            height += np.random.normal(0, 3)
            
            # 根据BMI计算体重
            weight = bmi * (height / 100) ** 2
            
            # 估算体脂率
            if gender == 'male':
                body_fat = 1.2 * bmi + 0.23 * age - 16.2
            else:
                body_fat = 1.2 * bmi + 0.23 * age - 5.4
            
            body_fat = np.clip(body_fat + np.random.normal(0, 1), 5, 50)
            
            subject = PersonProfile(
                age=int(age),
                gender=gender,
                height=height,
                weight=weight,
                body_fat_percentage=body_fat,
                activity_level=activity,
                weeks_on_diet=int(weeks_on_diet)
            )
            
            subjects.append(subject)
            self.subjects_created += 1
        
        # 如果还需要更多，用随机生成补充
        while len(subjects) < n_subjects:
            additional = self.generate_standard_subjects(1)
            subjects.extend(additional)
        
        logger.info(f"生成了 {len(subjects)} 个多样化虚拟人")
        return subjects[:n_subjects]
    
    def generate_plateau_subjects(self, n_subjects: int = 30) -> List[PersonProfile]:
        """
        生成已经处于平台期的虚拟人
        
        Args:
            n_subjects: 生成的虚拟人数量
            
        Returns:
            处于平台期的虚拟人列表
        """
        plateau_subjects = []
        
        for _ in range(n_subjects):
            # 这些人已经减肥一段时间
            weeks_on_diet = np.random.randint(8, 20)
            
            # 初始体重较高
            initial_weight = np.random.normal(95, 15)
            
            # 已经减掉一些体重（5-15%）
            weight_loss_percentage = np.random.uniform(0.05, 0.15)
            current_weight = initial_weight * (1 - weight_loss_percentage)
            
            # 随机年龄和性别
            age = np.random.randint(25, 55)
            gender = np.random.choice(['male', 'female'])
            
            # 身高
            if gender == 'male':
                height = np.random.normal(175, 7)
            else:
                height = np.random.normal(162, 6)
            
            # 当前BMI和体脂率
            current_bmi = current_weight / (height / 100) ** 2
            
            if gender == 'male':
                body_fat = 1.2 * current_bmi + 0.23 * age - 16.2
            else:
                body_fat = 1.2 * current_bmi + 0.23 * age - 5.4
            
            body_fat = np.clip(body_fat + np.random.normal(0, 2), 10, 45)
            
            # 活动水平（通常不高）
            activity_level = np.random.choice([1.2, 1.375, 1.55])
            
            # 创建虚拟人
            subject = PersonProfile(
                age=age,
                gender=gender,
                height=height,
                weight=current_weight,
                body_fat_percentage=body_fat,
                activity_level=activity_level,
                weeks_on_diet=weeks_on_diet
            )
            
            # 设置代谢适应（平台期的关键特征）
            subject.metabolic_adaptation_factor = np.random.uniform(0.75, 0.85)
            subject.initial_weight = initial_weight
            
            plateau_subjects.append(subject)
            self.subjects_created += 1
        
        logger.info(f"生成了 {n_subjects} 个处于平台期的虚拟人")
        return plateau_subjects
    
    def generate_edge_cases(self) -> List[Tuple[PersonProfile, Dict]]:
        """
        生成边界案例（用于测试极端情况）
        
        Returns:
            边界案例列表，每个元素是(虚拟人, 案例描述)的元组
        """
        edge_cases = []
        
        # 定义边界案例
        cases = [
            # 极低BMI
            {"name": "极瘦", "bmi": 17, "body_fat": 8, "gender": "female", "age": 25},
            
            # 极高BMI
            {"name": "重度肥胖", "bmi": 45, "body_fat": 45, "gender": "male", "age": 40},
            
            # 老年人
            {"name": "老年肥胖", "bmi": 32, "body_fat": 35, "gender": "female", "age": 65},
            
            # 年轻运动员
            {"name": "年轻运动员", "bmi": 24, "body_fat": 12, "gender": "male", "age": 20,
             "activity_level": 1.9},
            
            # 长期减肥者
            {"name": "长期减肥", "bmi": 28, "body_fat": 30, "gender": "female", "age": 35,
             "weeks_on_diet": 52},
            
            # 肌肉型
            {"name": "肌肉型", "bmi": 28, "body_fat": 15, "gender": "male", "age": 30,
             "activity_level": 1.725},
            
            # 久坐办公
            {"name": "久坐办公", "bmi": 30, "body_fat": 32, "gender": "female", "age": 38,
             "activity_level": 1.2},
            
            # 代谢受损
            {"name": "代谢受损", "bmi": 35, "body_fat": 38, "gender": "male", "age": 45,
             "weeks_on_diet": 26, "metabolic_factor": 0.7}
        ]
        
        for case_data in cases:
            case_type = case_data.copy()
            
            # 提取参数
            gender = case_type.pop('gender', 'male')
            age = case_type.pop('age', 30)
            bmi = case_type.pop('bmi', 25)
            body_fat = case_type.pop('body_fat', 25)
            activity_level = case_type.pop('activity_level', 1.4)
            weeks_on_diet = case_type.pop('weeks_on_diet', 0)
            metabolic_factor = case_type.pop('metabolic_factor', 1.0)
            name = case_type.pop('name', 'unknown')
            
            # 计算身高和体重
            height = 175 if gender == 'male' else 162
            weight = bmi * (height / 100) ** 2
            
            # 创建虚拟人
            subject = PersonProfile(
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                body_fat_percentage=body_fat,
                activity_level=activity_level,
                weeks_on_diet=weeks_on_diet
            )
            
            # 设置特殊属性
            if metabolic_factor != 1.0:
                subject.metabolic_adaptation_factor = metabolic_factor
            
            edge_cases.append((subject, {"name": name, "description": case_data}))
            self.subjects_created += 1
        
        logger.info(f"生成了 {len(edge_cases)} 个边界案例")
        return edge_cases
    
    def generate_by_criteria(self, 
                           n_subjects: int = 10,
                           age_range: Tuple[int, int] = (20, 60),
                           gender: Optional[str] = None,
                           bmi_range: Tuple[float, float] = (18.5, 40),
                           activity_range: Tuple[float, float] = (1.2, 1.9),
                           weeks_on_diet_range: Tuple[int, int] = (0, 52)) -> List[PersonProfile]:
        """
        根据特定条件生成虚拟人
        
        Args:
            n_subjects: 生成数量
            age_range: 年龄范围
            gender: 性别（None表示随机）
            bmi_range: BMI范围
            activity_range: 活动水平范围
            weeks_on_diet_range: 减肥周数范围
            
        Returns:
            符合条件的虚拟人列表
        """
        subjects = []
        
        for _ in range(n_subjects):
            # 年龄
            age = np.random.randint(age_range[0], age_range[1] + 1)
            
            # 性别
            if gender is None:
                subject_gender = np.random.choice(['male', 'female'])
            else:
                subject_gender = gender
            
            # 身高
            if subject_gender == 'male':
                height = np.random.normal(175, 7)
            else:
                height = np.random.normal(162, 6)
            
            # BMI和体重
            bmi = np.random.uniform(bmi_range[0], bmi_range[1])
            weight = bmi * (height / 100) ** 2
            
            # 体脂率
            if subject_gender == 'male':
                body_fat = 1.2 * bmi + 0.23 * age - 16.2
            else:
                body_fat = 1.2 * bmi + 0.23 * age - 5.4
            
            body_fat = np.clip(body_fat + np.random.normal(0, 2), 5, 50)
            
            # 活动水平
            activity_level = np.random.uniform(activity_range[0], activity_range[1])
            
            # 减肥历史
            weeks_on_diet = np.random.randint(weeks_on_diet_range[0], 
                                             weeks_on_diet_range[1] + 1)
            
            # 创建虚拟人
            subject = PersonProfile(
                age=age,
                gender=subject_gender,
                height=height,
                weight=weight,
                body_fat_percentage=body_fat,
                activity_level=activity_level,
                weeks_on_diet=weeks_on_diet
            )
            
            subjects.append(subject)
            self.subjects_created += 1
        
        logger.info(f"根据条件生成了 {n_subjects} 个虚拟人")
        return subjects
    
    def get_statistics(self) -> Dict:
        """
        获取生成器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_subjects_created": self.subjects_created,
            "age_groups": list(self.age_groups.keys()),
            "bmi_categories": list(self.bmi_categories.keys()),
            "activity_levels": list(self.activity_levels.keys())
        }
    
    def reset(self):
        """重置生成器"""
        self.subjects_created = 0
        logger.info("虚拟人生成器已重置")


# 便捷函数
def generate_test_population(size: str = 'small') -> List[PersonProfile]:
    """
    生成测试人群的便捷函数
    
    Args:
        size: 'small' (10人), 'medium' (50人), 'large' (100人)
        
    Returns:
        虚拟人列表
    """
    generator = VirtualSubjectGenerator()
    
    sizes = {
        'small': 10,
        'medium': 50,
        'large': 100
    }
    
    n_subjects = sizes.get(size, 10)
    return generator.generate_standard_subjects(n_subjects)


if __name__ == "__main__":
    # 测试模块
    generator = VirtualSubjectGenerator()
    
    # 生成标准人群
    standard_subjects = generator.generate_standard_subjects(10)
    print(f"生成标准人群: {len(standard_subjects)} 人")
    
    # 生成平台期人群
    plateau_subjects = generator.generate_plateau_subjects(5)
    print(f"生成平台期人群: {len(plateau_subjects)} 人")
    
    # 生成边界案例
    edge_cases = generator.generate_edge_cases()
    print(f"生成边界案例: {len(edge_cases)} 个")
    
    # 生成特定条件人群
    young_females = generator.generate_by_criteria(
        n_subjects=5,
        age_range=(20, 30),
        gender='female',
        bmi_range=(25, 30)
    )
    print(f"生成年轻女性群体: {len(young_females)} 人")
    
    # 显示统计信息
    stats = generator.get_statistics()
    print(f"\n总计生成虚拟人: {stats['total_subjects_created']} 人")