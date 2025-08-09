"""
数据追踪与可视化模块
提供全面的数据分析和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import json
import os

from font_manager import setup_chinese_font, get_chinese_font_prop
# 设置中文字体
setup_chinese_font()

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ThemeConfig:
    """主题配置"""
    name: str = "modern"
    
    # 颜色方案
    colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#1e88e5',      # 主色调 - 蓝色
        'secondary': '#00acc1',    # 次要色 - 青色
        'success': '#43a047',      # 成功 - 绿色
        'warning': '#fb8c00',      # 警告 - 橙色
        'danger': '#e53935',       # 危险 - 红色
        'info': '#5e35b1',         # 信息 - 紫色
        'light': '#f5f5f5',        # 浅色
        'dark': '#212121',         # 深色
        'muscle': '#4caf50',       # 肌肉 - 绿色
        'fat': '#ff5252',          # 脂肪 - 红色
        'weight': '#2196f3',       # 体重 - 蓝色
        'calorie': '#ff9800',      # 热量 - 橙色
        'exercise': '#9c27b0',     # 运动 - 紫色
        'sleep': '#607d8b'         # 睡眠 - 蓝灰色
    })
    
    # 字体配置
    fonts: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'title': {'size': 16, 'weight': 'bold'},
        'subtitle': {'size': 14, 'weight': 'semibold'},
        'label': {'size': 12, 'weight': 'normal'},
        'tick': {'size': 10, 'weight': 'normal'},
        'annotation': {'size': 9, 'weight': 'normal'}
    })
    
    # 样式配置
    styles: Dict[str, Any] = field(default_factory=lambda: {
        'line_width': 2.5,
        'marker_size': 8,
        'grid_alpha': 0.2,
        'fill_alpha': 0.3,
        'edge_width': 1.5,
        'dpi': 150,
        'figure_facecolor': '#ffffff',
        'axes_facecolor': '#fafafa'
    })


class DataTracker:
    """增强版数据追踪器"""
    
    def __init__(self):
        self.data = {
            # 时间数据
            'timestamp': [],
            'week': [],
            'date': [],
            
            # 身体指标
            'weight': [],
            'body_fat_percentage': [],
            'muscle_mass': [],
            'fat_mass': [],
            'water_weight': [],
            'bone_mass': [],
            'visceral_fat': [],
            'metabolic_age': [],
            
            # 代谢指标
            'bmr': [],
            'tdee': [],
            'metabolic_adaptation_factor': [],
            'neat': [],
            
            # 营养数据
            'calories_consumed': [],
            'protein_grams': [],
            'carb_grams': [],
            'fat_grams': [],
            'fiber_grams': [],
            'water_liters': [],
            
            # 运动数据
            'cardio_minutes': [],
            'strength_minutes': [],
            'steps': [],
            'active_calories': [],
            'exercise_types': [],
            
            # 生活方式
            'sleep_hours': [],
            'sleep_quality': [],
            'stress_level': [],
            'energy_level': [],
            
            # 方案参数
            'solution': [],
            
            # 评估指标
            'fitness_score': [],
            'muscle_retention_rate': [],
            'fat_loss_rate': [],
            'sustainability_score': [],
            
            # 预测数据
            'predicted_weight': [],
            'confidence_interval': []
        }
        
        # 元数据
        self.metadata = {
            'start_date': None,
            'person_profile': None,
            'goals': {},
            'milestones': []
        }
    
    def add_record(self, week: int, data_dict: Dict[str, Any]):
        """添加一条记录"""
        # 添加时间戳
        self.data['timestamp'].append(datetime.now())
        self.data['week'].append(week)
        
        if self.metadata['start_date']:
            current_date = self.metadata['start_date'] + timedelta(weeks=week)
            self.data['date'].append(current_date)
        
        # 添加各项数据
        for key, value in data_dict.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                logger.warning(f"未知的数据键: {key}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """获取数据框架"""
        # 确保所有列表长度一致
        max_len = max(len(v) for v in self.data.values() if isinstance(v, list))
        
        cleaned_data = {}
        for key, values in self.data.items():
            if isinstance(values, list):
                if len(values) < max_len:
                    # 用None填充缺失值
                    values.extend([None] * (max_len - len(values)))
                cleaned_data[key] = values[:max_len]
        
        return pd.DataFrame(cleaned_data)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取汇总统计"""
        df = self.get_dataframe()
        
        if len(df) == 0:
            return {}
        
        stats = {
            'total_weeks': len(df),
            'total_weight_loss': df['weight'].iloc[0] - df['weight'].iloc[-1] if 'weight' in df else 0,
            'avg_weekly_loss': (df['weight'].iloc[0] - df['weight'].iloc[-1]) / len(df) if 'weight' in df and len(df) > 0 else 0,
            'body_fat_change': df['body_fat_percentage'].iloc[0] - df['body_fat_percentage'].iloc[-1] if 'body_fat_percentage' in df else 0,
            'muscle_change': (df['muscle_mass'].iloc[-1] - df['muscle_mass'].iloc[0]) if ('muscle_mass' in df and not df['muscle_mass'].isna().all()) else 0,
            'best_week': df.loc[df['fitness_score'].idxmin(), 'week'] if 'fitness_score' in df and not df['fitness_score'].isna().all() else None
        }
        
        return stats
    
    def save_to_file(self, filepath: str):
        """保存数据到文件"""
        data_to_save = {
            'data': {k: v for k, v in self.data.items() if v},  # 只保存非空数据
            'metadata': self.metadata
        }
        
        # 处理datetime对象
        for key in ['timestamp', 'date']:
            if key in data_to_save['data']:
                data_to_save['data'][key] = [
                    d.isoformat() if isinstance(d, datetime) else d 
                    for d in data_to_save['data'][key]
                ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2, default=str)
    
    def load_from_file(self, filepath: str):
        """从文件加载数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.data.update(loaded_data.get('data', {}))
        self.metadata.update(loaded_data.get('metadata', {}))
        
        # 恢复datetime对象
        for key in ['timestamp', 'date']:
            if key in self.data:
                self.data[key] = [
                    datetime.fromisoformat(d) if isinstance(d, str) else d 
                    for d in self.data[key]
                ]


class WeightLossVisualizer:
    """减肥数据可视化主类"""
    
    def __init__(self, theme: Optional[ThemeConfig] = None):
        self.theme = theme or ThemeConfig()
        self._apply_theme()
        
    def _apply_theme(self):
        """应用主题设置"""
        # 先保存当前的字体设置
        current_font = plt.rcParams.get('font.sans-serif', [])
        current_family = plt.rcParams.get('font.family', 'sans-serif')
        
        # 应用样式（这会重置字体）
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 恢复字体设置
        plt.rcParams['font.sans-serif'] = current_font
        plt.rcParams['font.family'] = current_family
        plt.rcParams['axes.unicode_minus'] = False
        
        # 更新其他matplotlib参数
        plt.rcParams.update({
            'figure.facecolor': self.theme.styles['figure_facecolor'],
            'axes.facecolor': self.theme.styles['axes_facecolor'],
            'axes.grid': True,
            'grid.alpha': self.theme.styles['grid_alpha'],
            'axes.linewidth': self.theme.styles['edge_width'],
            'xtick.labelsize': self.theme.fonts['tick']['size'],
            'ytick.labelsize': self.theme.fonts['tick']['size'],
            'axes.labelsize': self.theme.fonts['label']['size'],
            'axes.titlesize': self.theme.fonts['subtitle']['size'],
            'figure.titlesize': self.theme.fonts['title']['size']
        })
    
    # 重新应用中文字体设置
    from font_manager import setup_chinese_font
    setup_chinese_font()
    
    def create_dashboard(self, tracker: DataTracker, 
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
        """创建综合仪表板"""
        # 创建图形
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 获取数据
        df = tracker.get_dataframe()
        stats = tracker.get_summary_stats()
        
        # 1. 标题和关键指标
        self._add_dashboard_header(fig, gs[0, :], stats, tracker.metadata)
        
        # 2. 体重变化主图
        ax_weight = fig.add_subplot(gs[1:3, :2])
        self._plot_weight_progress(ax_weight, df)
        
        # 3. 体成分分析
        ax_composition = fig.add_subplot(gs[1:3, 2:])
        self._plot_body_composition(ax_composition, df)
        
        # 4. 营养摄入分析
        ax_nutrition = fig.add_subplot(gs[3, :2])
        self._plot_nutrition_analysis(ax_nutrition, df)
        
        # 5. 运动模式分析
        ax_exercise = fig.add_subplot(gs[3, 2:])
        self._plot_exercise_patterns(ax_exercise, df)
        
        # 6. 代谢适应追踪
        ax_metabolism = fig.add_subplot(gs[4, :2])
        self._plot_metabolic_adaptation(ax_metabolism, df)
        
        # 7. 生活方式因素
        ax_lifestyle = fig.add_subplot(gs[4, 2:])
        self._plot_lifestyle_factors(ax_lifestyle, df)
        
        # 8. 预测与趋势
        ax_prediction = fig.add_subplot(gs[5, :])
        self._plot_predictions(ax_prediction, df)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存
        if save_path:
            fig.savefig(save_path, dpi=self.theme.styles['dpi'], 
                       bbox_inches='tight', facecolor=fig.get_facecolor())
        
        if show:
            plt.show()
        
        return fig
    
    def _add_dashboard_header(self, fig: plt.Figure, gs_section, 
                             stats: Dict, metadata: Dict):
        """添加仪表板头部"""
        ax = fig.add_subplot(gs_section)
        ax.axis('off')
        
        # 标题
        title_text = "减肥优化进度报告"
        ax.text(0.5, 0.85, title_text, ha='center', va='center',
               fontsize=24, fontweight='bold', transform=ax.transAxes)
        
        # 日期范围
        if metadata.get('start_date'):
            date_text = f"开始日期: {metadata['start_date'].strftime('%Y-%m-%d')}"
            ax.text(0.5, 0.7, date_text, ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
        
        # 关键指标卡片
        card_width = 0.22
        card_height = 0.4
        card_y = 0.15
        cards = [
            ('总减重', f"{stats.get('total_weight_loss', 0):.1f} kg", 
             self.theme.colors['success']),
            ('周均减重', f"{stats.get('avg_weekly_loss', 0):.2f} kg", 
             self.theme.colors['info']),
            ('体脂变化', f"{stats.get('body_fat_change', 0):.1f}%", 
             self.theme.colors['warning']),
            ('肌肉变化', f"{stats.get('muscle_change', 0):+.1f} kg", 
             self.theme.colors['muscle'])
        ]
        
        for i, (label, value, color) in enumerate(cards):
            x = i * 0.25 + 0.015
            
            # 创建卡片背景
            card = FancyBboxPatch((x, card_y), card_width, card_height,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, alpha=0.8,
                                 edgecolor='none',
                                 transform=ax.transAxes)
            ax.add_patch(card)
            
            # 添加文本
            ax.text(x + card_width/2, card_y + card_height*0.7, label,
                   ha='center', va='center', color='white',
                   fontsize=11, fontweight='bold',
                   transform=ax.transAxes)
            
            ax.text(x + card_width/2, card_y + card_height*0.3, value,
                   ha='center', va='center', color='white',
                   fontsize=16, fontweight='bold',
                   transform=ax.transAxes)
    
    def _plot_weight_progress(self, ax: plt.Axes, df: pd.DataFrame):
        """绘制体重进展"""
        if 'weight' not in df or df['weight'].isna().all():
            ax.text(0.5, 0.5, '暂无体重数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
            return
        
        weeks = df['week']
        weights = df['weight']
        
        # 主线图 - 过滤NaN值
        valid_mask = ~weights.isna()
        valid_weeks = weeks[valid_mask]
        valid_weights = weights[valid_mask]
        
        if len(valid_weeks) == 0:
            ax.text(0.5, 0.5, '暂无有效体重数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
            return
        
        line = ax.plot(valid_weeks, valid_weights, 'o-', 
                    color=self.theme.colors['primary'],
                    linewidth=self.theme.styles['line_width'],
                    markersize=self.theme.styles['marker_size'],
                    label='实际体重', zorder=5)[0]
        
        # 添加平滑趋势线
        if len(valid_weeks) > 3:
            z = np.polyfit(valid_weeks, valid_weights, 2)
            p = np.poly1d(z)
            trend_weeks = np.linspace(valid_weeks.min(), valid_weeks.max(), 100)
            ax.plot(trend_weeks, p(trend_weeks), '--',
                color=self.theme.colors['secondary'],
                linewidth=2, alpha=0.8, label='趋势线')
        
        # 标记平台期
        self._mark_plateaus(ax, valid_weeks, valid_weights)
        
        # 添加目标线（如果有）
        if 'goals' in df.columns and len(df['goals']) > 0 and 'target_weight' in df['goals'].iloc[0]:
            target = df['goals'].iloc[0]['target_weight']
            ax.axhline(y=target, color=self.theme.colors['success'],
                    linestyle=':', linewidth=2, label=f'目标: {target}kg')
        
        # 美化
        ax.set_xlabel('时间（周）', fontsize=self.theme.fonts['label']['size'])
        ax.set_ylabel('体重（kg）', fontsize=self.theme.fonts['label']['size'])
        ax.set_title('体重变化追踪', fontsize=self.theme.fonts['subtitle']['size'],
                    fontweight=self.theme.fonts['subtitle']['weight'], pad=15)
        
        # 添加网格
        ax.grid(True, alpha=self.theme.styles['grid_alpha'], linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # 添加数据标注
        for i in range(0, len(valid_weeks), max(1, len(valid_weeks)//8)):
            ax.annotate(f'{valid_weights.iloc[i]:.1f}',
                    xy=(valid_weeks.iloc[i], valid_weights.iloc[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', alpha=0.8))
        
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    def _plot_body_composition(self, ax: plt.Axes, df: pd.DataFrame):
        """绘制体成分分析"""
        required_cols = ['muscle_mass', 'fat_mass', 'body_fat_percentage']
        if not all(col in df for col in required_cols):
            ax.text(0.5, 0.5, '暂无体成分数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        weeks = df['week']
        
        # 创建双Y轴
        ax2 = ax.twinx()
        
        # 绘制质量数据（左Y轴）
        muscle_line = ax.plot(weeks, df['muscle_mass'], 'o-',
                            color=self.theme.colors['muscle'],
                            linewidth=self.theme.styles['line_width'],
                            markersize=self.theme.styles['marker_size'],
                            label='肌肉量')[0]
        
        fat_line = ax.plot(weeks, df['fat_mass'], 's-',
                          color=self.theme.colors['fat'],
                          linewidth=self.theme.styles['line_width'],
                          markersize=self.theme.styles['marker_size'],
                          label='脂肪量')[0]
        
        # 绘制体脂率（右Y轴）
        bf_line = ax2.plot(weeks, df['body_fat_percentage'], '^-',
                          color=self.theme.colors['warning'],
                          linewidth=self.theme.styles['line_width'],
                          markersize=self.theme.styles['marker_size'],
                          label='体脂率')[0]
        
        # 设置标签
        ax.set_xlabel('时间（周）', fontsize=self.theme.fonts['label']['size'])
        ax.set_ylabel('质量（kg）', fontsize=self.theme.fonts['label']['size'])
        ax2.set_ylabel('体脂率（%）', fontsize=self.theme.fonts['label']['size'])
        ax.set_title('体成分变化分析', fontsize=self.theme.fonts['subtitle']['size'],
                    fontweight=self.theme.fonts['subtitle']['weight'], pad=15)
        
        # 合并图例
        lines = [muscle_line, fat_line, bf_line]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', frameon=True, fancybox=True, shadow=True)
        
        # 颜色协调
        ax.tick_params(axis='y', colors=self.theme.colors['dark'])
        ax2.tick_params(axis='y', colors=self.theme.colors['warning'])
        
        # 网格
        ax.grid(True, alpha=self.theme.styles['grid_alpha'])
        ax.set_axisbelow(True)
    
    def _plot_nutrition_analysis(self, ax: plt.Axes, df: pd.DataFrame):
        """绘制营养分析"""
        required_cols = ['calories_consumed', 'protein_grams', 'carb_grams', 'fat_grams']
        if not any(col in df for col in required_cols):
            ax.text(0.5, 0.5, '暂无营养数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # 计算平均值
        avg_data = {
            '热量': df['calories_consumed'].mean() if 'calories_consumed' in df else 0,
            '蛋白质(×10)': df['protein_grams'].mean() * 10 if 'protein_grams' in df else 0,
            '碳水(×10)': df['carb_grams'].mean() * 10 if 'carb_grams' in df else 0,
            '脂肪(×10)': df['fat_grams'].mean() * 10 if 'fat_grams' in df else 0
        }
        
        # 创建条形图
        nutrients = list(avg_data.keys())
        values = list(avg_data.values())
        colors = [self.theme.colors['calorie'], self.theme.colors['muscle'],
                 self.theme.colors['warning'], self.theme.colors['danger']]
        
        bars = ax.bar(nutrients, values, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if '×10' in bar.get_label():
                actual_val = val / 10
                label = f'{actual_val:.1f}g'
            else:
                label = f'{val:.0f}kcal'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        ax.set_ylabel('平均摄入量', fontsize=self.theme.fonts['label']['size'])
        ax.set_title('营养摄入分析（日均）', fontsize=self.theme.fonts['subtitle']['size'],
                    fontweight=self.theme.fonts['subtitle']['weight'])
        ax.grid(True, alpha=self.theme.styles['grid_alpha'], axis='y')
        ax.set_axisbelow(True)
    
    def _plot_exercise_patterns(self, ax: plt.Axes, df: pd.DataFrame):
        """绘制运动模式"""
        if 'cardio_minutes' not in df or 'strength_minutes' not in df:
            ax.text(0.5, 0.5, '暂无运动数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
            return
        
        weeks = df['week']
        # 修复 FutureWarning
        cardio = pd.to_numeric(df['cardio_minutes'], errors='coerce').fillna(0)
        strength = pd.to_numeric(df['strength_minutes'], errors='coerce').fillna(0)
        
        # 堆叠面积图
        ax.fill_between(weeks, 0, cardio, 
                    color=self.theme.colors['info'],
                    alpha=0.6, label='有氧运动')
        ax.fill_between(weeks, cardio, cardio + strength,
                    color=self.theme.colors['secondary'],
                    alpha=0.6, label='力量训练')
        
        # 添加总运动时间线
        total = cardio + strength
        ax.plot(weeks, total, 'k-', linewidth=2, label='总运动时间')
        
        ax.set_xlabel('时间（周）', fontsize=self.theme.fonts['label']['size'])
        ax.set_ylabel('运动时间（分钟/周）', fontsize=self.theme.fonts['label']['size'])
        ax.set_title('运动模式分析', fontsize=self.theme.fonts['subtitle']['size'],
                    fontweight=self.theme.fonts['subtitle']['weight'])
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=self.theme.styles['grid_alpha'])
        ax.set_axisbelow(True)
    
    def _plot_metabolic_adaptation(self, ax: plt.Axes, df: pd.DataFrame):
        """绘制代谢适应"""
        if 'bmr' not in df or 'tdee' not in df:
            ax.text(0.5, 0.5, '暂无代谢数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        weeks = df['week']
        bmr = df['bmr']
        tdee = df['tdee']
        
        # 计算相对变化
        initial_bmr = bmr.iloc[0] if not bmr.empty else 1
        initial_tdee = tdee.iloc[0] if not tdee.empty else 1
        
        bmr_relative = (bmr / initial_bmr - 1) * 100
        tdee_relative = (tdee / initial_tdee - 1) * 100
        
        # 绘制
        ax.plot(weeks, bmr_relative, 'o-',
               color=self.theme.colors['primary'],
               linewidth=self.theme.styles['line_width'],
               markersize=self.theme.styles['marker_size'],
               label='BMR变化')
        
        ax.plot(weeks, tdee_relative, 's-',
               color=self.theme.colors['secondary'],
               linewidth=self.theme.styles['line_width'],
               markersize=self.theme.styles['marker_size'],
               label='TDEE变化')
        
        # 添加参考线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhline(y=-10, color=self.theme.colors['warning'], 
                  linestyle='--', linewidth=1, alpha=0.5, label='轻度适应')
        ax.axhline(y=-20, color=self.theme.colors['danger'], 
                  linestyle='--', linewidth=1, alpha=0.5, label='重度适应')
        
        # 填充危险区域
        ax.fill_between(weeks, -20, -100, color=self.theme.colors['danger'],
                       alpha=0.1)
        
        ax.set_xlabel('时间（周）', fontsize=self.theme.fonts['label']['size'])
        ax.set_ylabel('相对变化（%）', fontsize=self.theme.fonts['label']['size'])
        ax.set_title('代谢适应追踪', fontsize=self.theme.fonts['subtitle']['size'],
                    fontweight=self.theme.fonts['subtitle']['weight'])
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=self.theme.styles['grid_alpha'])
        ax.set_ylim(bottom=-30)
        ax.set_axisbelow(True)
    
    def _plot_lifestyle_factors(self, ax: plt.Axes, df: pd.DataFrame):
        """绘制生活方式因素"""
        lifestyle_cols = ['sleep_hours', 'stress_level', 'energy_level']
        available_cols = [col for col in lifestyle_cols if col in df]
        
        if not available_cols:
            ax.text(0.5, 0.5, '暂无生活方式数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # 创建雷达图
        categories = []
        values = []
        
        if 'sleep_hours' in df:
            avg_sleep = df['sleep_hours'].mean()
            sleep_score = min(100, (avg_sleep / 8) * 100)
            categories.append('睡眠')
            values.append(sleep_score)
        
        if 'stress_level' in df:
            avg_stress = df['stress_level'].mean()
            stress_score = 100 - (avg_stress * 20)  # 假设stress_level是1-5
            categories.append('压力管理')
            values.append(stress_score)
        
        if 'energy_level' in df:
            avg_energy = df['energy_level'].mean()
            energy_score = avg_energy * 20  # 假设energy_level是1-5
            categories.append('能量水平')
            values.append(energy_score)
        
        # 补充其他可能的指标
        if len(categories) < 3:
            categories.extend(['恢复', '心情', '动力'][:3-len(categories)])
            values.extend([75, 80, 85][:3-len(values)])
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_plot = values + values[:1]  # 闭合
        angles += angles[:1]
        
        ax.plot(angles, values_plot, 'o-', linewidth=2,
               color=self.theme.colors['success'], label='当前状态')
        ax.fill(angles, values_plot, alpha=0.25, color=self.theme.colors['success'])
        
        # 添加参考圈
        for level in [20, 40, 60, 80, 100]:
            ax.plot(angles, [level] * len(angles), 'k-', linewidth=0.5, alpha=0.3)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('生活方式因素评估', fontsize=self.theme.fonts['subtitle']['size'],
                    fontweight=self.theme.fonts['subtitle']['weight'])
        ax.grid(True)
    
    def _plot_predictions(self, ax: plt.Axes, df: pd.DataFrame):
        """绘制预测和趋势"""
        if 'weight' not in df or df['weight'].isna().all():
            ax.text(0.5, 0.5, '暂无预测数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
            return
        
        weeks = df['week']
        weights = df['weight']
        
        # 过滤掉 NaN 值（同时过滤 weeks 和 weights）
        valid_mask = ~weights.isna()
        valid_weeks = weeks[valid_mask]
        valid_weights = weights[valid_mask]
        
        if len(valid_weights) == 0:
            ax.text(0.5, 0.5, '暂无有效体重数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
            return
        
        # 历史数据
        ax.plot(valid_weeks, valid_weights, 'o-',
            color=self.theme.colors['primary'],
            linewidth=self.theme.styles['line_width'],
            markersize=self.theme.styles['marker_size'],
            label='实际数据')
        
        # 预测未来趋势
        if len(valid_weeks) > 3:
            # 使用有效数据进行多项式拟合
            z = np.polyfit(valid_weeks.values, valid_weights.values, 2)
            p = np.poly1d(z)
            
            # 预测未来8周
            future_weeks = np.arange(valid_weeks.max() + 1, valid_weeks.max() + 9)
            predicted_weights = p(future_weeks)
            
            # 绘制预测
            ax.plot(future_weeks, predicted_weights, 'o--',
                color=self.theme.colors['warning'],
                linewidth=self.theme.styles['line_width'],
                markersize=self.theme.styles['marker_size'],
                label='预测趋势', alpha=0.7)
            
            # 添加置信区间
            std_dev = valid_weights.std()
            confidence_upper = predicted_weights + 1.96 * std_dev
            confidence_lower = predicted_weights - 1.96 * std_dev
            
            ax.fill_between(future_weeks, confidence_lower, confidence_upper,
                        color=self.theme.colors['warning'], alpha=0.2,
                        label='95%置信区间')
            
            # 标记预测目标达成点（如果有目标体重）
            if 'goals' in df.columns and len(df['goals']) > 0:
                try:
                    target = df['goals'].iloc[0].get('target_weight')
                    if target:
                        crossing_points = np.where(np.diff(np.sign(predicted_weights - target)))[0]
                        if len(crossing_points) > 0:
                            target_week = future_weeks[crossing_points[0]]
                            target_weight = predicted_weights[crossing_points[0]]
                            ax.plot(target_week, target_weight, 'r*', markersize=15,
                                label=f'预计达标: 第{target_week}周')
                except:
                    pass  # 忽略目标相关的错误
        
        ax.set_xlabel('时间（周）', fontsize=self.theme.fonts['label']['size'])
        ax.set_ylabel('体重（kg）', fontsize=self.theme.fonts['label']['size'])
        ax.set_title('体重预测与趋势分析', fontsize=self.theme.fonts['subtitle']['size'],
                    fontweight=self.theme.fonts['subtitle']['weight'])
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=self.theme.styles['grid_alpha'])
        ax.set_axisbelow(True)
        
        # 添加当前日期垂直线
        current_week = valid_weeks.max()
        ax.axvline(x=current_week, color='black', linestyle=':', alpha=0.5)
        ax.text(current_week, ax.get_ylim()[1], '当前', ha='center', va='bottom')

    def _mark_plateaus(self, ax: plt.Axes, weeks, weights):
        """标记平台期"""
        if len(weights) < 3:
            return
        
        # 确保weeks和weights是Series类型
        if not isinstance(weeks, pd.Series):
            weeks = pd.Series(weeks.values if hasattr(weeks, 'values') else weeks)
        if not isinstance(weights, pd.Series):
            weights = pd.Series(weights.values if hasattr(weights, 'values') else weights)
        
        # 重置索引以确保对齐
        weeks = weeks.reset_index(drop=True)
        weights = weights.reset_index(drop=True)
        
        # 检测平台期（连续两周体重变化小于0.2kg）
        plateau_threshold = 0.2
        plateau_regions = []
        current_plateau_start = None
        
        for i in range(1, len(weights)):
            if pd.isna(weights.iloc[i]) or pd.isna(weights.iloc[i-1]):
                continue
                
            weight_change = abs(weights.iloc[i] - weights.iloc[i-1])
            
            if weight_change < plateau_threshold:
                if current_plateau_start is None:
                    current_plateau_start = i - 1
            else:
                if current_plateau_start is not None:
                    plateau_regions.append((current_plateau_start, i - 1))
                    current_plateau_start = None
        
        # 处理最后一个平台期
        if current_plateau_start is not None:
            plateau_regions.append((current_plateau_start, len(weights) - 1))
        
        # 绘制平台期区域
        for start, end in plateau_regions:
            if end - start >= 1:  # 至少持续2周
                ax.axvspan(weeks.iloc[start], weeks.iloc[end],
                          alpha=0.2, color=self.theme.colors['warning'],
                          label='平台期' if start == plateau_regions[0][0] else '')
    
    def create_comparison_report(self, trackers: Dict[str, DataTracker],
                               save_path: Optional[str] = None) -> plt.Figure:
        """创建多方案对比报告"""
        n_trackers = len(trackers)
        fig = plt.figure(figsize=(16, 4 * n_trackers))
        
        for idx, (name, tracker) in enumerate(trackers.items()):
            df = tracker.get_dataframe()
            
            # 为每个方案创建一行子图
            ax1 = plt.subplot(n_trackers, 3, idx * 3 + 1)
            ax2 = plt.subplot(n_trackers, 3, idx * 3 + 2)
            ax3 = plt.subplot(n_trackers, 3, idx * 3 + 3)
            
            # 体重变化
            if 'weight' in df:
                ax1.plot(df['week'], df['weight'], 'o-',
                        color=self.theme.colors['primary'],
                        linewidth=2, markersize=6)
                ax1.set_title(f'{name} - 体重变化')
                ax1.set_xlabel('周')
                ax1.set_ylabel('体重(kg)')
                ax1.grid(True, alpha=0.3)
            
            # 体脂率变化
            if 'body_fat_percentage' in df:
                ax2.plot(df['week'], df['body_fat_percentage'], 's-',
                        color=self.theme.colors['warning'],
                        linewidth=2, markersize=6)
                ax2.set_title(f'{name} - 体脂率变化')
                ax2.set_xlabel('周')
                ax2.set_ylabel('体脂率(%)')
                ax2.grid(True, alpha=0.3)
            
            # 适应度评分
            if 'fitness_score' in df:
                ax3.bar(df['week'], df['fitness_score'],
                       color=self.theme.colors['success'], alpha=0.7)
                ax3.set_title(f'{name} - 适应度评分')
                ax3.set_xlabel('周')
                ax3.set_ylabel('评分')
                ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('多方案对比分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        return fig
    
    def create_animation(self, tracker: DataTracker, 
                        save_path: Optional[str] = None) -> FuncAnimation:
        """创建动画展示减肥过程"""
        df = tracker.get_dataframe()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 初始化
        weeks = df['week'].values
        weights = df['weight'].values
        body_fat = df['body_fat_percentage'].values if 'body_fat_percentage' in df else None
        
        # 设置轴范围
        ax1.set_xlim(weeks.min() - 1, weeks.max() + 1)
        ax1.set_ylim(weights.min() - 2, weights.max() + 2)
        ax1.set_xlabel('周')
        ax1.set_ylabel('体重 (kg)')
        ax1.set_title('体重变化动画')
        ax1.grid(True, alpha=0.3)
        
        if body_fat is not None:
            ax2.set_xlim(weeks.min() - 1, weeks.max() + 1)
            ax2.set_ylim(body_fat.min() - 2, body_fat.max() + 2)
            ax2.set_xlabel('周')
            ax2.set_ylabel('体脂率 (%)')
            ax2.set_title('体脂率变化动画')
            ax2.grid(True, alpha=0.3)
        
        # 创建线对象
        line1, = ax1.plot([], [], 'o-', color=self.theme.colors['primary'],
                         linewidth=2, markersize=8)
        line2, = ax2.plot([], [], 's-', color=self.theme.colors['warning'],
                         linewidth=2, markersize=8)
        
        # 动画更新函数
        def update(frame):
            line1.set_data(weeks[:frame+1], weights[:frame+1])
            if body_fat is not None:
                line2.set_data(weeks[:frame+1], body_fat[:frame+1])
            
            # 添加当前值标注
            if frame < len(weeks):
                ax1.texts.clear()
                ax2.texts.clear()
                
                ax1.text(weeks[frame], weights[frame] + 0.5,
                        f'{weights[frame]:.1f}', ha='center', fontsize=10)
                
                if body_fat is not None:
                    ax2.text(weeks[frame], body_fat[frame] + 0.5,
                            f'{body_fat[frame]:.1f}', ha='center', fontsize=10)
            
            return line1, line2
        
        # 创建动画
        anim = FuncAnimation(fig, update, frames=len(weeks),
                           interval=500, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
        
        return anim


class OptimizationVisualizer(WeightLossVisualizer):
    """差分进化优化过程可视化"""
    
    def plot_optimization_results(self, results: Dict, save_path: Optional[str] = None):
        """绘制优化结果（整合自原de_weight_loss_optimizer.py）"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 适应度进化曲线
        iterations = range(1, len(results['best_fitness_history']) + 1)
        ax1.plot(iterations, results['best_fitness_history'], 'b-', label='最佳适应度', linewidth=2)
        ax1.plot(iterations, results['avg_fitness_history'], 'r--', label='平均适应度', linewidth=1)
        ax1.set_xlabel('迭代次数（周）')
        ax1.set_ylabel('适应度值')
        ax1.set_title('适应度进化曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 体重变化曲线
        if 'final_person_state' in results and 'initial_weight' in results:
            # 从结果中重建体重历史
            initial_weight = results['initial_weight']
            weights = [initial_weight]
            
            # 通过最优方案历史计算体重变化
            current_weight = initial_weight
            for i, solution in enumerate(results['best_solutions_history']):
                # 简化的体重计算（实际应该从模拟结果中获取）
                weight_loss = 0.5  # 假设每周减0.5kg作为示例
                current_weight -= weight_loss
                weights.append(current_weight)
            
            ax2.plot(range(len(weights)), weights, 'g-', linewidth=2, marker='o')
            ax2.set_xlabel('时间（周）')
            ax2.set_ylabel('体重 (kg)')
            ax2.set_title('体重变化曲线')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '暂无体重数据', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
        
        # 3. 热量摄入变化
        calories = [s.calories for s in results['best_solutions_history']]
        ax3.plot(iterations, calories, 'orange', linewidth=2, marker='s')
        ax3.set_xlabel('迭代次数（周）')
        ax3.set_ylabel('日均热量摄入 (kcal)')
        ax3.set_title('热量摄入调整曲线')
        ax3.grid(True, alpha=0.3)
        
        # 4. 营养素比例变化
        proteins = [s.protein_ratio * 100 for s in results['best_solutions_history']]
        carbs = [s.carb_ratio * 100 for s in results['best_solutions_history']]
        fats = [s.fat_ratio * 100 for s in results['best_solutions_history']]
        
        ax4.plot(iterations, proteins, 'b-', label='蛋白质', linewidth=2)
        ax4.plot(iterations, carbs, 'g-', label='碳水化合物', linewidth=2)
        ax4.plot(iterations, fats, 'r-', label='脂肪', linewidth=2)
        ax4.set_xlabel('迭代次数（周）')
        ax4.set_ylabel('营养素比例 (%)')
        ax4.set_title('营养素比例变化')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.theme.styles.get('show_plots', True):
            plt.show()
        
        return fig
    
    def plot_optimization_progress(self, 
                                 population_history: List[List],
                                 fitness_history: List[List],
                                 best_history: List,
                                 save_path: Optional[str] = None):
        """绘制优化进展"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        generations = range(len(fitness_history))
        
        # 1. 适应度进化
        best_fitness = [min(gen_fitness) for gen_fitness in fitness_history]
        avg_fitness = [np.mean(gen_fitness) for gen_fitness in fitness_history]
        worst_fitness = [max(gen_fitness) for gen_fitness in fitness_history]
        
        ax1.plot(generations, best_fitness, 'o-',
                color=self.theme.colors['success'], linewidth=2,
                markersize=8, label='最优')
        ax1.plot(generations, avg_fitness, 's-',
                color=self.theme.colors['info'], linewidth=2,
                markersize=6, label='平均')
        ax1.plot(generations, worst_fitness, '^-',
                color=self.theme.colors['danger'], linewidth=2,
                markersize=6, label='最差')
        
        ax1.fill_between(generations, best_fitness, worst_fitness,
                        alpha=0.2, color=self.theme.colors['primary'])
        
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.set_title('适应度进化曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 种群多样性
        diversity_scores = []
        for pop in population_history:
            # 计算种群多样性（标准差）
            pop_array = np.array([ind.to_vector() for ind in pop])
            diversity = np.mean(np.std(pop_array, axis=0))
            diversity_scores.append(diversity)
        
        ax2.plot(generations, diversity_scores, 'o-',
                color=self.theme.colors['secondary'], linewidth=2,
                markersize=8)
        ax2.fill_between(generations, 0, diversity_scores,
                        alpha=0.3, color=self.theme.colors['secondary'])
        
        ax2.set_xlabel('代数')
        ax2.set_ylabel('多样性指数')
        ax2.set_title('种群多样性变化')
        ax2.grid(True, alpha=0.3)
        
        # 3. 参数演化热图
        params = ['热量', '蛋白质', '碳水', '脂肪', '有氧', '力量', '睡眠']
        param_evolution = np.zeros((len(params), len(best_history)))
        
        for gen_idx, solution in enumerate(best_history):
            param_evolution[0, gen_idx] = solution.calories / 100
            param_evolution[1, gen_idx] = solution.protein_ratio * 100
            param_evolution[2, gen_idx] = solution.carb_ratio * 100
            param_evolution[3, gen_idx] = solution.fat_ratio * 100
            param_evolution[4, gen_idx] = solution.cardio_freq * 10
            param_evolution[5, gen_idx] = solution.strength_freq * 10
            param_evolution[6, gen_idx] = solution.sleep_hours * 10
        
        im = ax3.imshow(param_evolution, aspect='auto', cmap='YlOrRd')
        ax3.set_yticks(range(len(params)))
        ax3.set_yticklabels(params)
        ax3.set_xlabel('代数')
        ax3.set_title('最优方案参数演化')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('相对值')
        
        # 4. 收敛速度分析
        improvements = []
        for i in range(1, len(best_fitness)):
            improvement = (best_fitness[i-1] - best_fitness[i]) / best_fitness[i-1] * 100
            improvements.append(max(0, improvement))
        
        ax4.bar(range(1, len(improvements)+1), improvements,
               color=self.theme.colors['info'], alpha=0.7)
        ax4.set_xlabel('代数')
        ax4.set_ylabel('改善率 (%)')
        ax4.set_title('逐代改善率')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('差分进化优化过程分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        return fig
    
    def plot_solution_space(self, solutions: List, 
                          highlight_best: bool = True,
                          save_path: Optional[str] = None):
        """绘制解空间分布"""
        # 使用PCA降维到2D进行可视化
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # 提取解向量
        solution_vectors = np.array([s.to_vector() for s in solutions])
        
        # 标准化
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(solution_vectors)
        
        # PCA降维
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(scaled_vectors)
        
        # 获取适应度值（用于着色）
        fitness_values = [s.fitness if hasattr(s, 'fitness') else 0 for s in solutions]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制散点图
        scatter = ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1],
                           c=fitness_values, s=100, cmap='RdYlGn_r',
                           edgecolors='black', linewidth=1, alpha=0.7)
        
        # 高亮最优解
        if highlight_best and fitness_values:
            best_idx = np.argmin(fitness_values)
            ax.scatter(reduced_vectors[best_idx, 0], reduced_vectors[best_idx, 1],
                      s=500, marker='*', color='gold', edgecolors='black',
                      linewidth=2, label='最优解', zorder=10)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('适应度值')
        
        # 设置标签
        ax.set_xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.1%}方差)')
        ax.set_ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.1%}方差)')
        ax.set_title('解空间分布（PCA投影）', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加等密度线
        if len(reduced_vectors) > 20:
            from scipy.stats import gaussian_kde
            
            # 计算核密度估计
            kde = gaussian_kde(reduced_vectors.T)
            
            # 创建网格
            x_min, x_max = reduced_vectors[:, 0].min(), reduced_vectors[:, 0].max()
            y_min, y_max = reduced_vectors[:, 1].min(), reduced_vectors[:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # 计算密度
            density = np.reshape(kde(positions).T, xx.shape)
            
            # 添加等密度线
            contours = ax.contour(xx, yy, density, colors='black',
                                 linewidths=1, alpha=0.5)
            ax.clabel(contours, inline=True, fontsize=8)
        
        if save_path:
            fig.savefig(save_path, dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        return fig


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, visualizer: WeightLossVisualizer):
        self.visualizer = visualizer
        
    def generate_html_report(self, tracker: DataTracker,
                           best_solution,
                           optimization_history: Dict,
                           save_path: str = "report.html"):
        """生成HTML格式的综合报告"""
        # 获取统计数据
        df = tracker.get_dataframe()
        stats = tracker.get_summary_stats()
        
        # 生成各种图表
        import tempfile
        import base64
        from io import BytesIO
        
        # 创建临时目录存储图片
        temp_dir = tempfile.mkdtemp()
        
        # 生成主仪表板
        dashboard_path = f"{temp_dir}/dashboard.png"
        self.visualizer.create_dashboard(tracker, save_path=dashboard_path, show=False)
        
        # 生成优化过程图
        if optimization_history:
            opt_viz = OptimizationVisualizer(self.visualizer.theme)
            opt_path = f"{temp_dir}/optimization.png"
            opt_viz.plot_optimization_progress(
                optimization_history.get('population_history', []),
                optimization_history.get('fitness_history', []),
                optimization_history.get('best_history', []),
                save_path=opt_path
            )
        
        # HTML模板
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>减肥优化报告</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #1e88e5;
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .section {{
                    background-color: white;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-card {{
                    display: inline-block;
                    background-color: #f8f9fa;
                    padding: 20px;
                    margin: 10px;
                    border-radius: 5px;
                    text-align: center;
                    min-width: 200px;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #1e88e5;
                }}
                .metric-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #1e88e5;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .recommendation {{
                    background-color: #e3f2fd;
                    padding: 15px;
                    border-left: 4px solid #1e88e5;
                    margin: 15px 0;
                }}
                .warning {{
                    background-color: #fff3cd;
                    padding: 15px;
                    border-left: 4px solid #ffc107;
                    margin: 15px 0;
                }}
                .success {{
                    background-color: #d4edda;
                    padding: 15px;
                    border-left: 4px solid #28a745;
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>减肥优化综合报告</h1>
                <p>基于差分进化算法的个性化方案优化</p>
                <p>生成时间：{report_date}</p>
            </div>
            
            <div class="section">
                <h2>📊 关键成果</h2>
                <div style="text-align: center;">
                    <div class="metric-card">
                        <div class="metric-value">{total_weight_loss:.1f} kg</div>
                        <div class="metric-label">总减重</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{total_weeks} 周</div>
                        <div class="metric-label">执行时长</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{avg_weekly_loss:.2f} kg</div>
                        <div class="metric-label">周均减重</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{body_fat_change:.1f}%</div>
                        <div class="metric-label">体脂率变化</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 最优方案</h2>
                <table>
                    <tr>
                        <th>参数</th>
                        <th>数值</th>
                        <th>说明</th>
                    </tr>
                    <tr>
                        <td>每日热量</td>
                        <td>{calories:.0f} kcal</td>
                        <td>建议的每日热量摄入</td>
                    </tr>
                    <tr>
                        <td>营养素比例</td>
                        <td>蛋白质 {protein:.0%} / 碳水 {carb:.0%} / 脂肪 {fat:.0%}</td>
                        <td>宏量营养素分配</td>
                    </tr>
                    <tr>
                        <td>有氧运动</td>
                        <td>每周 {cardio_freq} 次，每次 {cardio_dur} 分钟</td>
                        <td>推荐的有氧运动安排</td>
                    </tr>
                    <tr>
                        <td>力量训练</td>
                        <td>每周 {strength_freq} 次</td>
                        <td>推荐的力量训练频率</td>
                    </tr>
                    <tr>
                        <td>睡眠时间</td>
                        <td>{sleep:.1f} 小时/晚</td>
                        <td>建议的睡眠时长</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📈 进度可视化</h2>
                <div class="chart-container">
                    <img src="{dashboard_img}" alt="综合仪表板">
                </div>
            </div>
            
            {optimization_section}
            
            <div class="section">
                <h2>💡 个性化建议</h2>
                {recommendations}
            </div>
            
            <div class="section">
                <h2>⚠️ 注意事项</h2>
                <div class="warning">
                    <strong>医疗声明：</strong>本报告仅供参考，不构成医疗建议。在开始任何减肥计划前，请咨询专业医生或营养师。
                </div>
                <div class="recommendation">
                    <strong>持续监测：</strong>建议每周定期测量体重和体脂率，并根据实际情况调整方案。
                </div>
            </div>
            
            <div class="section" style="text-align: center; color: #666;">
                <p>© 2024 差分进化减肥优化系统 | 技术支持：DE Weight Loss Optimizer</p>
            </div>
        </body>
        </html>
        """
        
        # 将图片转换为base64
        def img_to_base64(img_path):
            with open(img_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        
        dashboard_b64 = f"data:image/png;base64,{img_to_base64(dashboard_path)}"
        
        # 生成优化部分（如果有）
        optimization_section = ""
        if optimization_history and 'opt_path' in locals():
            opt_b64 = f"data:image/png;base64,{img_to_base64(opt_path)}"
            optimization_section = f"""
            <div class="section">
                <h2>🔄 优化过程</h2>
                <div class="chart-container">
                    <img src="{opt_b64}" alt="优化过程分析">
                </div>
            </div>
            """
        
        # 生成建议
        recommendations = self._generate_recommendations(stats, best_solution, df)
        
        # 填充模板
        html_content = html_template.format(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_weight_loss=stats.get('total_weight_loss', 0),
            total_weeks=stats.get('total_weeks', 0),
            avg_weekly_loss=stats.get('avg_weekly_loss', 0),
            body_fat_change=stats.get('body_fat_change', 0),
            calories=best_solution.calories,
            protein=best_solution.protein_ratio,
            carb=best_solution.carb_ratio,
            fat=best_solution.fat_ratio,
            cardio_freq=best_solution.cardio_freq,
            cardio_dur=best_solution.cardio_duration,
            strength_freq=best_solution.strength_freq,
            sleep=best_solution.sleep_hours,
            dashboard_img=dashboard_b64,
            optimization_section=optimization_section,
            recommendations=recommendations
        )
        
        # 保存HTML文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        
        logger.info(f"报告已生成：{save_path}")
        
    def _generate_recommendations(self, stats: Dict, solution, df: pd.DataFrame) -> str:
        """生成个性化建议"""
        recommendations = []
        
        # 基于减重速度的建议
        avg_loss = stats.get('avg_weekly_loss', 0)
        if avg_loss > 1.0:
            recommendations.append(
                '<div class="warning">减重速度过快，建议适当增加热量摄入，避免肌肉流失。</div>'
            )
        elif avg_loss < 0.25:
            recommendations.append(
                '<div class="recommendation">减重速度较慢，可考虑增加运动量或适当降低热量摄入。</div>'
            )
        else:
            recommendations.append(
                '<div class="success">当前减重速度理想，请继续保持！</div>'
            )
        
        # 基于营养分配的建议
        if solution.protein_ratio < 0.25:
            recommendations.append(
                '<div class="recommendation">蛋白质摄入偏低，建议增加瘦肉、鱼类、豆制品等高蛋白食物。</div>'
            )
        
        # 基于运动的建议
        total_exercise = solution.cardio_freq * solution.cardio_duration / 60 + solution.strength_freq
        if total_exercise < 3:
            recommendations.append(
                '<div class="recommendation">运动量偏少，建议逐步增加运动频率，提高代谢水平。</div>'
            )
        elif total_exercise > 10:
            recommendations.append(
                '<div class="warning">运动量较大，注意充分休息和恢复，避免过度训练。</div>'
            )
        
        # 检测平台期
        if len(df) > 4:
            recent_weights = df['weight'].tail(3).values
            if np.std(recent_weights) < 0.2:
                recommendations.append(
                    '<div class="warning">可能进入平台期，建议尝试热量循环或改变运动方式。</div>'
                )
        
        return '\n'.join(recommendations)


# 便捷函数
def quick_plot(tracker: DataTracker, plot_type: str = 'dashboard'):
    """快速绘图函数"""
    viz = WeightLossVisualizer()
    
    if plot_type == 'dashboard':
        viz.create_dashboard(tracker)
    elif plot_type == 'weight':
        fig, ax = plt.subplots(figsize=(10, 6))
        viz._plot_weight_progress(ax, tracker.get_dataframe())
        plt.show()
    elif plot_type == 'composition':
        fig, ax = plt.subplots(figsize=(10, 6))
        viz._plot_body_composition(ax, tracker.get_dataframe())
        plt.show()
    else:
        logger.warning(f"未知的绘图类型: {plot_type}")


if __name__ == "__main__":
    print("高级可视化模块已加载")
    print("\n可用的主要类：")
    print("- DataTracker: 增强版数据追踪")
    print("- WeightLossVisualizer: 主可视化类")
    print("- OptimizationVisualizer: 优化过程可视化")
    print("- ReportGenerator: 报告生成器")
    print("- ThemeConfig: 主题配置")
    print("\n使用 quick_plot() 函数可快速生成图表")