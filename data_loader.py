"""
外部数据加载模块
支持从CSV、JSON等格式加载真实实验数据
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from metabolic_model import PersonProfile
from solution_generator import Solution
from visualization import DataTracker

logger = logging.getLogger(__name__)


class ExperimentDataLoader:
    """实验数据加载器"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.xlsx']
        
    def load_experiment_data(self, filepath: str) -> Dict:
        """
        加载实验数据
        
        Args:
            filepath: 数据文件路径
            
        Returns:
            包含实验数据的字典
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        suffix = path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {suffix}")
        
        if suffix == '.csv':
            return self._load_csv(filepath)
        elif suffix == '.json':
            return self._load_json(filepath)
        elif suffix == '.xlsx':
            return self._load_excel(filepath)
    
    def _load_csv(self, filepath: str) -> Dict:
        """加载CSV格式的实验数据"""
        df = pd.read_csv(filepath)
        
        # 验证必需的列
        required_columns = ['week', 'weight', 'body_fat_percentage']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"缺少必需的列: {missing_columns}")
        
        # 转换为字典格式
        data = {
            'experiment_name': Path(filepath).stem,
            'data_type': 'real_experiment',
            'source_file': filepath,
            'participants': []
        }
        
        # 如果有多个参与者，按ID分组
        if 'participant_id' in df.columns:
            for pid, group in df.groupby('participant_id'):
                participant_data = self._extract_participant_data(group, pid)
                data['participants'].append(participant_data)
        else:
            # 只有一个参与者
            participant_data = self._extract_participant_data(df, 'P001')
            data['participants'].append(participant_data)
        
        return data
    
    def _extract_participant_data(self, df: pd.DataFrame, participant_id: str) -> Dict:
        """提取单个参与者的数据"""
        # 基本信息
        participant = {
            'id': participant_id,
            'profile': {},
            'weekly_data': [],
            'interventions': []
        }
        
        # 如果有初始档案信息
        profile_columns = ['age', 'gender', 'height', 'initial_weight', 
                          'initial_body_fat', 'activity_level']
        for col in profile_columns:
            if col in df.columns:
                participant['profile'][col] = df[col].iloc[0]
        
        # 提取每周数据
        for _, row in df.iterrows():
            week_data = {
                'week': int(row['week']) if 'week' in row else None,
                'weight': float(row['weight']) if 'weight' in row and pd.notna(row['weight']) else None,
                'body_fat_percentage': float(row['body_fat_percentage']) if 'body_fat_percentage' in row and pd.notna(row['body_fat_percentage']) else None,
            }
            
            # 添加其他可用数据
            optional_columns = [
                'muscle_mass', 'fat_mass', 'bmr', 'tdee',
                'calories_consumed', 'protein_grams', 'carb_grams', 'fat_grams',
                'cardio_minutes', 'strength_minutes', 'sleep_hours',
                'stress_level', 'energy_level', 'hunger_level'
            ]
            
            for col in optional_columns:
                if col in row and pd.notna(row[col]):
                    week_data[col] = float(row[col])
            
            participant['weekly_data'].append(week_data)
        
        # 提取干预措施（如果有）
        if 'intervention' in df.columns:
            interventions = df[['week', 'intervention']].dropna().to_dict('records')
            participant['interventions'] = interventions
        
        return participant
    
    def _load_json(self, filepath: str) -> Dict:
        """加载JSON格式的实验数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据格式
        if 'participants' not in data:
            # 尝试转换为标准格式
            data = {
                'experiment_name': Path(filepath).stem,
                'data_type': 'real_experiment',
                'source_file': filepath,
                'participants': [data]  # 假设整个文件是一个参与者的数据
            }
        
        return data
    
    def _load_excel(self, filepath: str) -> Dict:
        """加载Excel格式的实验数据"""
        # 读取所有工作表
        excel_file = pd.ExcelFile(filepath)
        
        data = {
            'experiment_name': Path(filepath).stem,
            'data_type': 'real_experiment',
            'source_file': filepath,
            'participants': []
        }
        
        # 每个工作表可能是一个参与者
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            participant_data = self._extract_participant_data(df, sheet_name)
            data['participants'].append(participant_data)
        
        return data
    
    def convert_to_tracker(self, experiment_data: Dict, 
                          participant_index: int = 0) -> DataTracker:
        """
        将实验数据转换为DataTracker对象
        
        Args:
            experiment_data: 实验数据字典
            participant_index: 参与者索引
            
        Returns:
            DataTracker对象
        """
        tracker = DataTracker()
        
        # 设置元数据
        tracker.metadata['experiment_name'] = experiment_data.get('experiment_name', 'Unknown')
        tracker.metadata['data_type'] = experiment_data.get('data_type', 'real_experiment')
        tracker.metadata['start_date'] = datetime.now()
        
        # 获取参与者数据
        if participant_index >= len(experiment_data['participants']):
            raise IndexError(f"参与者索引 {participant_index} 超出范围")
        
        participant = experiment_data['participants'][participant_index]
        
        # 设置参与者档案
        tracker.metadata['person_profile'] = participant['profile']
        
        # 添加每周数据
        for week_data in participant['weekly_data']:
            week = week_data.get('week', len(tracker.data['week']))
            tracker.add_record(week, week_data)
        
        return tracker
    
    def create_person_profile(self, participant_data: Dict) -> PersonProfile:
        """
        从参与者数据创建PersonProfile对象
        
        Args:
            participant_data: 参与者数据字典
            
        Returns:
            PersonProfile对象
        """
        profile = participant_data['profile']
        
        # 提取初始数据
        initial_data = participant_data['weekly_data'][0] if participant_data['weekly_data'] else {}
        
        return PersonProfile(
            age=profile.get('age', 30),
            gender=profile.get('gender', 'male'),
            height=profile.get('height', 170),
            weight=profile.get('initial_weight', initial_data.get('weight', 70)),
            body_fat_percentage=profile.get('initial_body_fat', 
                                          initial_data.get('body_fat_percentage', 25)),
            activity_level=profile.get('activity_level', 1.4),
            weeks_on_diet=0
        )


class ExperimentDesigner:
    """实验设计器"""
    
    def __init__(self):
        self.experiment_types = [
            'single_subject',      # 单被试实验
            'parallel_group',      # 平行组实验
            'crossover',          # 交叉实验
            'factorial',          # 因子设计
            'adaptive'            # 自适应实验
        ]
    
    def create_experiment_plan(self, 
                              experiment_type: str,
                              participants: List[PersonProfile],
                              duration_weeks: int,
                              conditions: List[Dict]) -> Dict:
        """
        创建实验计划
        
        Args:
            experiment_type: 实验类型
            participants: 参与者列表
            duration_weeks: 实验持续周数
            conditions: 实验条件列表
            
        Returns:
            实验计划字典
        """
        if experiment_type not in self.experiment_types:
            raise ValueError(f"不支持的实验类型: {experiment_type}")
        
        plan = {
            'type': experiment_type,
            'participants': participants,
            'duration': duration_weeks,
            'conditions': conditions,
            'schedule': [],
            'measurements': []
        }
        
        if experiment_type == 'single_subject':
            plan['schedule'] = self._create_single_subject_schedule(duration_weeks, conditions)
        elif experiment_type == 'parallel_group':
            plan['schedule'] = self._create_parallel_group_schedule(participants, conditions)
        elif experiment_type == 'crossover':
            plan['schedule'] = self._create_crossover_schedule(participants, duration_weeks, conditions)
        
        # 添加测量计划
        plan['measurements'] = self._create_measurement_schedule(duration_weeks)
        
        return plan
    
    def _create_single_subject_schedule(self, duration: int, conditions: List[Dict]) -> List[Dict]:
        """创建单被试实验计划"""
        schedule = []
        
        # 基线期（2周）
        for week in range(2):
            schedule.append({
                'week': week,
                'phase': 'baseline',
                'condition': 'control',
                'intervention': None
            })
        
        # 干预期
        condition_duration = (duration - 2) // len(conditions)
        current_week = 2
        
        for i, condition in enumerate(conditions):
            for week in range(condition_duration):
                schedule.append({
                    'week': current_week,
                    'phase': f'intervention_{i+1}',
                    'condition': condition['name'],
                    'intervention': condition
                })
                current_week += 1
        
        return schedule
    
    def _create_parallel_group_schedule(self, participants: List, conditions: List[Dict]) -> List[Dict]:
        """创建平行组实验计划"""
        schedule = []
        
        # 随机分配参与者到不同条件
        n_participants = len(participants)
        n_conditions = len(conditions)
        
        # 简单的轮流分配（实际应该随机）
        for i, participant in enumerate(participants):
            condition_index = i % n_conditions
            schedule.append({
                'participant_id': f'P{i+1:03d}',
                'condition': conditions[condition_index]['name'],
                'intervention': conditions[condition_index]
            })
        
        return schedule
    
    def _create_crossover_schedule(self, participants: List, duration: int, conditions: List[Dict]) -> List[Dict]:
        """创建交叉实验计划"""
        schedule = []
        n_periods = len(conditions)
        period_duration = duration // n_periods
        
        # 拉丁方设计
        for i, participant in enumerate(participants):
            participant_schedule = []
            # 循环移位条件顺序
            rotated_conditions = conditions[i % n_periods:] + conditions[:i % n_periods]
            
            for period, condition in enumerate(rotated_conditions):
                for week in range(period_duration):
                    participant_schedule.append({
                        'participant_id': f'P{i+1:03d}',
                        'week': period * period_duration + week,
                        'period': period + 1,
                        'condition': condition['name'],
                        'intervention': condition
                    })
            
            schedule.extend(participant_schedule)
        
        return schedule
    
    def _create_measurement_schedule(self, duration: int) -> List[Dict]:
        """创建测量计划"""
        measurements = []
        
        # 每周基础测量
        for week in range(duration):
            measurements.append({
                'week': week,
                'type': 'basic',
                'measures': ['weight', 'body_fat_percentage', 'waist_circumference']
            })
        
        # 每两周详细测量
        for week in range(0, duration, 2):
            measurements.append({
                'week': week,
                'type': 'detailed',
                'measures': ['muscle_mass', 'fat_mass', 'bmr', 'blood_pressure', 'resting_heart_rate']
            })
        
        # 每四周综合评估
        for week in range(0, duration, 4):
            measurements.append({
                'week': week,
                'type': 'comprehensive',
                'measures': ['blood_work', 'fitness_test', 'psychological_assessment']
            })
        
        return measurements


class SimulatedExperiment:
    """模拟实验类"""
    
    def __init__(self, metabolic_model=None):
        from metabolic_model import MetabolicModel
        self.metabolic_model = metabolic_model or MetabolicModel()
        
    def generate_synthetic_data(self, 
                               person: PersonProfile,
                               solution: Solution,
                               duration_weeks: int,
                               noise_level: float = 0.05) -> pd.DataFrame:
        """
        生成合成实验数据
        
        Args:
            person: 人物档案
            solution: 减肥方案
            duration_weeks: 持续周数
            noise_level: 噪声水平（0-1）
            
        Returns:
            包含合成数据的DataFrame
        """
        data = []
        current_person = person
        
        for week in range(duration_weeks):
            # 使用代谢模型模拟
            results = self.metabolic_model.simulate_week(current_person, solution, week)
            
            # 添加噪声
            noise_factor = 1 + np.random.normal(0, noise_level)
            
            week_data = {
                'week': week,
                'weight': results['final_weight'] * noise_factor,
                'body_fat_percentage': results['final_body_fat_percentage'] * noise_factor,
                'muscle_mass': current_person.lean_body_mass * noise_factor,
                'fat_mass': current_person.fat_mass * noise_factor,
                'bmr': self.metabolic_model.calculate_bmr(current_person) * noise_factor,
                'tdee': self.metabolic_model.calculate_tdee(current_person, solution) * noise_factor,
                'calories_consumed': solution.calories + np.random.normal(0, 50),
                'protein_grams': solution.calories * solution.protein_ratio / 4 + np.random.normal(0, 10),
                'cardio_minutes': solution.cardio_freq * solution.cardio_duration + np.random.normal(0, 10),
                'strength_minutes': solution.strength_freq * 60 + np.random.normal(0, 10),
                'sleep_hours': solution.sleep_hours + np.random.normal(0, 0.5),
                'hunger_level': np.random.uniform(3, 7),
                'energy_level': np.random.uniform(4, 8),
                'stress_level': np.random.uniform(2, 6)
            }
            
            data.append(week_data)
            
            # 更新人物状态
            current_person = self.metabolic_model.update_person_state(
                current_person, solution, week
            )
        
        return pd.DataFrame(data)
    
    def save_synthetic_dataset(self, 
                              n_participants: int,
                              duration_weeks: int,
                              output_file: str):
        """生成并保存合成数据集"""
        from solution_generator import SolutionGenerator
        
        all_data = []
        generator = SolutionGenerator()
        
        for i in range(n_participants):
            # 生成随机参与者
            person = PersonProfile(
                age=np.random.randint(20, 60),
                gender=np.random.choice(['male', 'female']),
                height=np.random.normal(170, 10),
                weight=np.random.normal(80, 15),
                body_fat_percentage=np.random.uniform(20, 35),
                activity_level=np.random.uniform(1.2, 1.6),
                weeks_on_diet=0
            )
            
            # 生成随机方案
            tdee = self.metabolic_model.calculate_bmr(person) * person.activity_level
            solution_vector = generator.generate_random_solution(tdee)
            solution = Solution(solution_vector)
            
            # 生成数据
            participant_data = self.generate_synthetic_data(person, solution, duration_weeks)
            participant_data['participant_id'] = f'P{i+1:03d}'
            
            # 添加初始信息
            participant_data['age'] = person.age
            participant_data['gender'] = person.gender
            participant_data['height'] = person.height
            participant_data['initial_weight'] = person.weight
            participant_data['initial_body_fat'] = person.body_fat_percentage
            participant_data['activity_level'] = person.activity_level
            
            all_data.append(participant_data)
        
        # 合并所有数据
        full_dataset = pd.concat(all_data, ignore_index=True)
        
        # 保存到文件
        if output_file.endswith('.csv'):
            full_dataset.to_csv(output_file, index=False)
        elif output_file.endswith('.xlsx'):
            full_dataset.to_excel(output_file, index=False)
        else:
            full_dataset.to_json(output_file, orient='records', indent=2)
        
        logger.info(f"合成数据集已保存到: {output_file}")
        return full_dataset


# 示例使用函数
def load_and_analyze_experiment(data_file: str):
    """加载并分析实验数据的示例"""
    # 创建数据加载器
    loader = ExperimentDataLoader()
    
    # 加载数据
    experiment_data = loader.load_experiment_data(data_file)
    
    # 转换为DataTracker
    tracker = loader.convert_to_tracker(experiment_data)
    
    # 创建可视化
    from visualization import WeightLossVisualizer
    viz = WeightLossVisualizer()
    viz.create_dashboard(tracker)
    
    return tracker


def create_synthetic_experiment():
    """创建合成实验数据的示例"""
    # 创建模拟实验
    sim = SimulatedExperiment()
    
    # 生成数据集
    sim.save_synthetic_dataset(
        n_participants=20,
        duration_weeks=12,
        output_file='synthetic_weight_loss_data.csv'
    )
    
    print("合成数据集已创建")


if __name__ == "__main__":
    # 测试代码
    print("数据加载模块")
    print("\n可用功能：")
    print("1. ExperimentDataLoader - 加载外部实验数据")
    print("2. ExperimentDesigner - 设计实验方案")
    print("3. SimulatedExperiment - 生成合成数据")
    
    # 生成示例数据
    create_synthetic_experiment()
