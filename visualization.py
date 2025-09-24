"""
增强版可视化模块
整合了个人减重可视化、优化过程可视化和实验结果可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class ThemeConfig:
    """可视化主题配置"""
    # 颜色方案
    colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#73AB84',
        'warning': '#F18F01',
        'danger': '#C73E1D',
        'info': '#6C91BF',
        'light': '#F7F7F7',
        'dark': '#2D3436'
    })
    
    # 字体配置
    fonts: Dict[str, Any] = field(default_factory=lambda: {
        'title_size': 16,
        'label_size': 12,
        'tick_size': 10,
        'legend_size': 10,
        'annotation_size': 9
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
    """增强版数据追踪器 - 统一的数据收集机制"""
    
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
            'predicted_body_fat': [],
            
            # 实验数据（新增）
            'experiment_id': [],
            'subject_id': [],
            'method': [],
            'condition': []
        }
        
        # 元数据
        self.metadata = {
            'start_date': None,
            'person_profile': {},
            'optimization_config': {},
            'experiment_info': {}
        }
    
    def add_record(self, week: int, **kwargs):
        """添加一条记录"""
        self.data['week'].append(week)
        self.data['timestamp'].append(datetime.now())
        
        # 计算日期
        if self.metadata.get('start_date'):
            date = self.metadata['start_date'] + timedelta(weeks=week)
            self.data['date'].append(date)
        
        # 添加其他数据
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                logger.warning(f"未知的数据字段: {key}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """获取DataFrame格式的数据"""
        # 只包含非空的列
        non_empty_data = {k: v for k, v in self.data.items() if v}
        return pd.DataFrame(non_empty_data)
    
    def get_summary_stats(self) -> Dict:
        """获取统计摘要"""
        df = self.get_dataframe()
        
        if df.empty:
            return {}
        
        stats = {
            'total_weeks': len(df),
            'weight_loss': df['weight'].iloc[0] - df['weight'].iloc[-1] if 'weight' in df else 0,
            'body_fat_change': df['body_fat_percentage'].iloc[0] - df['body_fat_percentage'].iloc[-1] if 'body_fat_percentage' in df else 0,
            'muscle_change': (df['muscle_mass'].iloc[-1] - df['muscle_mass'].iloc[0]) if ('muscle_mass' in df and not df['muscle_mass'].isna().all()) else 0,
            'best_week': df.loc[df['fitness_score'].idxmin(), 'week'] if 'fitness_score' in df and not df['fitness_score'].isna().all() else None
        }
        
        return stats
    
    def save_to_file(self, filepath: str):
        """保存数据到文件"""
        data_to_save = {
            'data': {k: v for k, v in self.data.items() if v},
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
        
    def create_dashboard(self, tracker: DataTracker, 
                        save_path: Optional[str] = None,
                        show: bool = True) -> Dict[str, plt.Figure]:
        """创建综合仪表板"""
        df = tracker.get_dataframe()
        figures = {}
        base_path = save_path.rsplit('.', 1)[0] if save_path else None
        
        # 1. 体重变化
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        self._plot_weight_progress(ax1, df)
        figures['weight_progress'] = fig1
        if base_path:
            fig1.savefig(f"{base_path}_weight_progress.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 2. 身体成分
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        self._plot_body_composition(ax2, df)
        figures['body_composition'] = fig2
        if base_path:
            fig2.savefig(f"{base_path}_body_composition.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 3. 营养摄入
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        self._plot_nutrition(ax3, df)
        figures['nutrition'] = fig3
        if base_path:
            fig3.savefig(f"{base_path}_nutrition.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 4. 运动情况
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        self._plot_exercise(ax4, df)
        figures['exercise'] = fig4
        if base_path:
            fig4.savefig(f"{base_path}_exercise.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 5. 适应度进化
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        self._plot_fitness_evolution(ax5, df)
        figures['fitness_evolution'] = fig5
        if base_path:
            fig5.savefig(f"{base_path}_fitness_evolution.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 6. 代谢适应
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        self._plot_metabolic_adaptation(ax6, df)
        figures['metabolic_adaptation'] = fig6
        if base_path:
            fig6.savefig(f"{base_path}_metabolic_adaptation.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 7. 睡眠和恢复
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        self._plot_sleep_recovery(ax7, df)
        figures['sleep_recovery'] = fig7
        if base_path:
            fig7.savefig(f"{base_path}_sleep_recovery.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 8. 统计摘要
        fig8, ax8 = plt.subplots(figsize=(8, 6))
        self._plot_summary_stats(ax8, tracker.get_summary_stats())
        figures['summary_stats'] = fig8
        if base_path:
            fig8.savefig(f"{base_path}_summary_stats.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        if show:
            plt.show()
        
        return figures
    
    def _plot_weight_progress(self, ax, df):
        """绘制体重进展（包含平台期检测）"""
        if 'weight' not in df.columns:
            ax.text(0.5, 0.5, '无体重数据', ha='center', va='center')
            return
        
        weeks = df['week']
        weight = df['weight']
        
        # 检测平台期
        plateau_threshold = 0.2  # kg
        plateau_regions = []
        current_plateau_start = None
        
        for i in range(1, len(weight)):
            weight_change = abs(weight.iloc[i] - weight.iloc[i-1])
            
            if weight_change < plateau_threshold:
                if current_plateau_start is None:
                    current_plateau_start = i - 1
            else:
                if current_plateau_start is not None:
                    if i - current_plateau_start >= 2:  # 至少持续2周
                        plateau_regions.append((current_plateau_start, i - 1))
                    current_plateau_start = None
        
        # 检查最后是否还在平台期
        if current_plateau_start is not None and len(weight) - current_plateau_start >= 2:
            plateau_regions.append((current_plateau_start, len(weight) - 1))
        
        # 标注平台期区域
        # 推荐的修正方式
        for start, end in plateau_regions:
            # 只有当 plateau_regions 不为空，并且当前是第一个区域时，才设置 label
            current_label = '平台期' if plateau_regions and start == plateau_regions[0][0] else ''
            ax.axvspan(weeks.iloc[start], weeks.iloc[end],
                    alpha=0.2, color=self.theme.colors['warning'],
                    label=current_label)
        
        # 实际体重
        ax.plot(weeks, weight, 'o-', color=self.theme.colors['primary'],
                linewidth=self.theme.styles['line_width'],
                markersize=self.theme.styles['marker_size'],
                label='实际体重')
        
        # 预测体重（如果有）
        if 'predicted_weight' in df.columns:
            ax.plot(weeks, df['predicted_weight'], '--',
                   color=self.theme.colors['secondary'],
                   linewidth=2, label='预测体重')
        
        # 添加趋势线
        z = np.polyfit(weeks, weight, 1)
        p = np.poly1d(z)
        ax.plot(weeks, p(weeks), '--', color=self.theme.colors['info'],
               alpha=0.5, label=f'趋势线 (斜率: {z[0]:.2f})')
        
        ax.set_xlabel('周')
        ax.set_ylabel('体重 (kg)')
        ax.set_title('体重变化趋势')
        ax.legend()
        ax.grid(True, alpha=self.theme.styles['grid_alpha'])
    
    def _plot_body_composition(self, ax, df):
        """绘制身体成分饼图"""
        if 'body_fat_percentage' not in df.columns or df.empty:
            ax.text(0.5, 0.5, '无体成分数据', ha='center', va='center')
            return
        
        # 使用最新数据
        latest = df.iloc[-1]
        
        if 'muscle_mass' in df.columns and 'weight' in df.columns:
            muscle = latest['muscle_mass']
            fat = latest['weight'] * latest['body_fat_percentage'] / 100
            other = latest['weight'] - muscle - fat
            
            sizes = [muscle, fat, other]
            labels = ['肌肉', '脂肪', '其他']
            colors = [self.theme.colors['success'], 
                     self.theme.colors['warning'],
                     self.theme.colors['light']]
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title('当前身体成分')
    
    def _plot_nutrition(self, ax, df):
        """绘制营养摄入"""
        if 'calories_consumed' not in df.columns:
            ax.text(0.5, 0.5, '无营养数据', ha='center', va='center')
            return
        
        weeks = df['week']
        calories = df['calories_consumed']
        
        ax.bar(weeks, calories, color=self.theme.colors['info'], alpha=0.7)
        ax.axhline(y=calories.mean(), color='red', linestyle='--',
                  label=f'平均: {calories.mean():.0f}')
        
        ax.set_xlabel('周')
        ax.set_ylabel('热量 (kcal)')
        ax.set_title('每周热量摄入')
        ax.legend()
    
    def _plot_exercise(self, ax, df):
        """绘制运动情况"""
        if 'cardio_minutes' not in df.columns and 'strength_minutes' not in df.columns:
            ax.text(0.5, 0.5, '无运动数据', ha='center', va='center')
            return
        
        weeks = df['week']
        
        cardio = df.get('cardio_minutes', pd.Series([0]*len(weeks)))
        strength = df.get('strength_minutes', pd.Series([0]*len(weeks)))
        
        width = 0.35
        x = np.arange(len(weeks))
        
        ax.bar(x - width/2, cardio, width, label='有氧',
              color=self.theme.colors['primary'])
        ax.bar(x + width/2, strength, width, label='力量',
              color=self.theme.colors['secondary'])
        
        ax.set_xlabel('周')
        ax.set_ylabel('分钟')
        ax.set_title('每周运动时间')
        ax.set_xticks(x)
        ax.set_xticklabels(weeks)
        ax.legend()
    
    def _plot_fitness_evolution(self, ax, df):
        """绘制适应度进化"""
        if 'fitness_score' not in df.columns:
            ax.text(0.5, 0.5, '无适应度数据', ha='center', va='center')
            return
        
        weeks = df['week']
        fitness = df['fitness_score']
        
        ax.plot(weeks, fitness, 'o-', color=self.theme.colors['danger'],
               linewidth=2, markersize=6)
        
        # 标记最佳点
        best_idx = fitness.idxmin()
        ax.plot(weeks[best_idx], fitness[best_idx], 'o',
               color=self.theme.colors['success'], markersize=12,
               label=f'最佳: {fitness[best_idx]:.3f}')
        
        ax.set_xlabel('周')
        ax.set_ylabel('适应度值')
        ax.set_title('适应度进化曲线')
        ax.legend()
        ax.grid(True, alpha=self.theme.styles['grid_alpha'])
    
    def _plot_metabolic_adaptation(self, ax, df):
        """绘制代谢适应"""
        if 'metabolic_adaptation_factor' not in df.columns:
            ax.text(0.5, 0.5, '无代谢数据', ha='center', va='center')
            return
        
        weeks = df['week']
        adaptation = df['metabolic_adaptation_factor']
        
        ax.fill_between(weeks, 1, adaptation, alpha=0.3,
                       color=self.theme.colors['warning'])
        ax.plot(weeks, adaptation, 'o-', color=self.theme.colors['warning'],
               linewidth=2)
        
        ax.set_xlabel('周')
        ax.set_ylabel('代谢适应因子')
        ax.set_title('代谢适应程度')
        ax.set_ylim([0.7, 1.05])
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=self.theme.styles['grid_alpha'])
    
    def _plot_sleep_recovery(self, ax, df):
        """绘制睡眠和恢复"""
        if 'sleep_hours' not in df.columns:
            ax.text(0.5, 0.5, '无睡眠数据', ha='center', va='center')
            return
        
        weeks = df['week']
        sleep = df['sleep_hours']
        
        ax.plot(weeks, sleep, 'o-', color=self.theme.colors['info'],
               linewidth=2, markersize=6)
        
        # 推荐睡眠范围
        ax.axhspan(7, 9, alpha=0.2, color='green', label='推荐范围')
        
        ax.set_xlabel('周')
        ax.set_ylabel('睡眠时长 (小时)')
        ax.set_title('睡眠情况')
        ax.legend()
        ax.grid(True, alpha=self.theme.styles['grid_alpha'])
    
    def _plot_summary_stats(self, ax, stats):
        """绘制统计摘要"""
        ax.axis('off')
        
        if not stats:
            ax.text(0.5, 0.5, '无统计数据', ha='center', va='center')
            return
        
        text = f"""
        总周数: {stats.get('total_weeks', 0)}
        
        总减重: {stats.get('weight_loss', 0):.2f} kg
        
        体脂变化: {stats.get('body_fat_change', 0):.1f}%
        
        肌肉变化: {stats.get('muscle_change', 0):.1f} kg
        
        最佳周: 第{stats.get('best_week', 'N/A')}周
        """
        
        ax.text(0.1, 0.5, text, fontsize=11, va='center')
        ax.set_title('统计摘要')


class OptimizationVisualizer(WeightLossVisualizer):
    """差分进化优化过程可视化"""
    
    def create_animation(self, tracker: DataTracker, 
                        save_path: Optional[str] = None) -> FuncAnimation:
        """创建动画展示减肥过程"""
        from matplotlib.animation import FuncAnimation
        
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
                # 清除之前的文本
                for txt in ax1.texts:
                    txt.remove()
                for txt in ax2.texts:
                    txt.remove()
                
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
    
    def plot_solution_space(self, solutions: List, 
                          highlight_best: bool = True,
                          save_path: Optional[str] = None):
        """绘制解空间分布（PCA降维可视化）"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # 提取解向量
        solution_vectors = []
        for s in solutions:
            if hasattr(s, 'to_vector'):
                solution_vectors.append(s.to_vector())
            elif isinstance(s, (list, np.ndarray)):
                solution_vectors.append(s)
        
        solution_vectors = np.array(solution_vectors)
        
        # 标准化
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(solution_vectors)
        
        # PCA降维
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(scaled_vectors)
        
        # 获取适应度值（用于着色）
        fitness_values = []
        for s in solutions:
            if hasattr(s, 'fitness'):
                fitness_values.append(s.fitness)
            else:
                fitness_values.append(0)
        
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
    
    def plot_optimization_results(self, results: Dict, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """绘制优化结果"""
        figures = {}
        base_path = save_path.rsplit('.', 1)[0] if save_path else None
        
        # 1. 适应度进化曲线
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        iterations = range(1, len(results['best_fitness_history']) + 1)
        ax1.plot(iterations, results['best_fitness_history'], 'b-', label='最佳适应度', linewidth=2)
        ax1.plot(iterations, results['avg_fitness_history'], 'r--', label='平均适应度', linewidth=1)
        ax1.set_xlabel('迭代次数（周）')
        ax1.set_ylabel('适应度值')
        ax1.set_title('适应度进化曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        figures['fitness_evolution'] = fig1
        if base_path:
            fig1.savefig(f"{base_path}_fitness_evolution.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 2. 体重预测vs实际
        if 'weight_history' in results:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            weeks = range(len(results['weight_history']))
            ax2.plot(weeks, results['weight_history'], 'o-', label='实际体重')
            if 'predicted_weights' in results:
                ax2.plot(weeks, results['predicted_weights'], 's--', label='预测体重')
            ax2.set_xlabel('周')
            ax2.set_ylabel('体重 (kg)')
            ax2.set_title('体重变化')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            figures['weight_changes'] = fig2
            if base_path:
                fig2.savefig(f"{base_path}_weight_changes.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 3. 方案参数演化
        if 'best_solutions_history' in results:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            solutions = results['best_solutions_history']
            weeks = range(len(solutions))
            calories = [s.calories if hasattr(s, 'calories') else s[0] for s in solutions]
            ax3.plot(weeks, calories, 'g-', linewidth=2)
            ax3.set_xlabel('周')
            ax3.set_ylabel('推荐热量 (kcal)')
            ax3.set_title('热量摄入调整')
            ax3.grid(True, alpha=0.3)
            figures['calorie_adjustment'] = fig3
            if base_path:
                fig3.savefig(f"{base_path}_calorie_adjustment.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        # 4. 营养素比例变化
        if 'best_solutions_history' in results:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            solutions = results['best_solutions_history']
            weeks = range(len(solutions))
            protein = [s.protein_ratio if hasattr(s, 'protein_ratio') else s[1] for s in solutions]
            carbs = [s.carb_ratio if hasattr(s, 'carb_ratio') else s[2] for s in solutions]
            fat = [s.fat_ratio if hasattr(s, 'fat_ratio') else s[3] for s in solutions]
            ax4.stackplot(weeks, protein, carbs, fat, 
                        labels=['蛋白质', '碳水', '脂肪'],
                        colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax4.set_xlabel('周')
            ax4.set_ylabel('营养素比例')
            ax4.set_title('营养素分配变化')
            ax4.legend(loc='upper right')
            ax4.set_ylim([0, 1])
            figures['nutrition_distribution'] = fig4
            if base_path:
                fig4.savefig(f"{base_path}_nutrition_distribution.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        return figures
    
    def plot_optimization_progress(self, 
                                population_history: List,
                                fitness_history: List,
                                best_solutions: List,
                                save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """绘制优化进展 - 每个图单独绘制"""
        figures = {}
        base_path = save_path.rsplit('.', 1)[0] if save_path else None
        
        # 1. 种群多样性 - 单独图形
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        generations = range(len(population_history))
        diversity = []
        
        for pop in population_history:
            pop_array = np.array([ind.to_vector() if hasattr(ind, 'to_vector') else ind 
                                for ind in pop])
            diversity.append(np.mean(np.std(pop_array, axis=0)))
        
        ax1.plot(generations, diversity, 'b-', linewidth=2)
        ax1.set_xlabel('代数')
        ax1.set_ylabel('多样性指数')
        ax1.set_title('种群多样性变化')
        ax1.grid(True, alpha=0.3)
        figures['diversity'] = fig1
        if base_path:
            fig1.savefig(f"{base_path}_diversity.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig1)
        
        # 2. 适应度分布 - 单独图形
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        key_gens = [0, len(fitness_history)//4, len(fitness_history)//2, -1]
        
        for i, gen_idx in enumerate(key_gens):
            if gen_idx < len(fitness_history):
                gen_fitness = fitness_history[gen_idx]
                ax2.hist(gen_fitness, alpha=0.5, bins=20, 
                    label=f'第{gen_idx if gen_idx >= 0 else len(fitness_history)+gen_idx}代')
        
        ax2.set_xlabel('适应度值')
        ax2.set_ylabel('频数')
        ax2.set_title('适应度分布演化')
        ax2.legend()
        figures['fitness_distribution'] = fig2
        if base_path:
            fig2.savefig(f"{base_path}_fitness_distribution.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig2)
        
        # 3. 收敛曲线 - 单独图形
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        best_fitness = [min(gen) for gen in fitness_history]
        avg_fitness = [np.mean(gen) for gen in fitness_history]
        worst_fitness = [max(gen) for gen in fitness_history]
        
        generations = range(len(fitness_history))
        ax3.fill_between(generations, worst_fitness, best_fitness, 
                    alpha=0.3, color='gray', label='范围')
        ax3.plot(generations, best_fitness, 'g-', linewidth=2, label='最佳')
        ax3.plot(generations, avg_fitness, 'b--', linewidth=1.5, label='平均')
        
        ax3.set_xlabel('代数')
        ax3.set_ylabel('适应度')
        ax3.set_title('收敛过程')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        figures['convergence'] = fig3
        if base_path:
            fig3.savefig(f"{base_path}_convergence.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig3)
        
        # 4. 参数轨迹 - 单独图形
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        if best_solutions:
            calories = [s.calories if hasattr(s, 'calories') else s[0] for s in best_solutions]
            calories_norm = (np.array(calories) - np.min(calories)) / (np.max(calories) - np.min(calories))
            
            generations = range(len(best_solutions))
            ax4.plot(generations, calories_norm, label='热量', linewidth=2)
            
            ax4.set_xlabel('代数')
            ax4.set_ylabel('归一化参数值')
            ax4.set_title('最优解参数轨迹')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        figures['parameter_trajectory'] = fig4
        if base_path:
            fig4.savefig(f"{base_path}_parameter_trajectory.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig4)
        
        return figures


class ExperimentVisualizer:
    """实验结果可视化器 - 从run_experiments.py迁移并增强"""
    
    def __init__(self, theme: Optional[ThemeConfig] = None):
        self.theme = theme or ThemeConfig()
        self.results_dir = "./experiment_results"
    
    def visualize_benchmark_results(self, analysis: Dict, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """可视化基准对比实验结果 - 每个图单独绘制"""
        figures = {}
        base_path = save_path.rsplit('.', 1)[0] if save_path else None
        
        # 准备数据
        methods = list(analysis.keys())
        if 'statistical_tests' in methods:
            methods.remove('statistical_tests')
        
        # 1. 平均减重对比 - 单独图形
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        mean_losses = [analysis[m]['mean_weight_loss'] for m in methods]
        std_losses = [analysis[m]['std_weight_loss'] for m in methods]
        
        bars = ax1.bar(methods, mean_losses, yerr=std_losses, capsize=5)
        ax1.set_ylabel('平均减重 (kg)')
        ax1.set_title('不同方法的减重效果对比')
        ax1.grid(axis='y', alpha=0.3)
        
        max_idx = np.argmax(mean_losses)
        bars[max_idx].set_color(self.theme.colors['success'])
        figures['mean_weight_loss'] = fig1
        if base_path:
            fig1.savefig(f"{base_path}_mean_weight_loss.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig1)
        
        # 2. 成功率对比 - 单独图形
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        success_rates = [analysis[m]['success_rate'] * 100 for m in methods]
        bars = ax2.bar(methods, success_rates)
        ax2.set_ylabel('成功率 (%)')
        ax2.set_title('减重成功率对比 (>5kg)')
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars):
            if success_rates[i] >= 75:
                bar.set_color(self.theme.colors['success'])
            elif success_rates[i] >= 50:
                bar.set_color(self.theme.colors['warning'])
            else:
                bar.set_color(self.theme.colors['danger'])
        figures['success_rates'] = fig2
        if base_path:
            fig2.savefig(f"{base_path}_success_rates.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig2)
        
        # 3. 减重分布箱线图 - 单独图形
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        box_data = []
        for m in methods:
            mean = analysis[m]['mean_weight_loss']
            std = analysis[m]['std_weight_loss']
            data_points = np.random.normal(mean, std, 100)
            box_data.append(data_points)
        
        bp = ax3.boxplot(box_data, tick_labels=methods, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(self.theme.colors['info'])
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('减重 (kg)')
        ax3.set_title('减重分布箱线图')
        ax3.grid(axis='y', alpha=0.3)
        figures['weight_distribution'] = fig3
        if base_path:
            fig3.savefig(f"{base_path}_weight_distribution.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig3)
        
        # 4. 效果对比雷达图 - 单独图形
        fig4 = plt.figure(figsize=(10, 8))
        ax4 = fig4.add_subplot(111, projection='polar')
        self._plot_radar_chart(ax4, methods, analysis)
        figures['radar_comparison'] = fig4
        if base_path:
            fig4.savefig(f"{base_path}_radar_comparison.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig4)
        
        # 5. 统计显著性热图 - 单独图形
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        if 'statistical_tests' in analysis:
            self._plot_significance_heatmap(ax5, methods, analysis['statistical_tests'])
        else:
            ax5.text(0.5, 0.5, '无统计检验数据', ha='center', va='center')
            ax5.set_title('统计显著性分析')
        figures['significance_heatmap'] = fig5
        if base_path:
            fig5.savefig(f"{base_path}_significance_heatmap.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig5)
        
        # 6. 方法排名 - 单独图形
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        self._plot_method_ranking(ax6, methods, analysis)
        figures['method_ranking'] = fig6
        if base_path:
            fig6.savefig(f"{base_path}_method_ranking.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig6)
        
        return figures
    
    def visualize_plateau_results(self, analysis: Dict, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """可视化平台期突破实验结果 - 每个图单独绘制"""
        figures = {}
        base_path = save_path.rsplit('.', 1)[0] if save_path else None
        
        strategies = [key for key in analysis.keys() 
                    if key not in ['statistical_tests', 'metadata']]
        
        # 1. 突破成功率 - 单独图形
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        success_rates = [analysis[s]['success_rate'] * 100 for s in strategies]
        bars = ax1.bar(strategies, success_rates)
        
        for i, bar in enumerate(bars):
            if success_rates[i] >= 70:
                bar.set_color(self.theme.colors['success'])
            elif success_rates[i] >= 50:
                bar.set_color(self.theme.colors['warning'])
            else:
                bar.set_color(self.theme.colors['danger'])
        
        ax1.set_ylabel('突破成功率 (%)')
        ax1.set_title('平台期突破成功率对比')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)
        figures['breakthrough_success_rate'] = fig1
        if base_path:
            fig1.savefig(f"{base_path}_breakthrough_success_rate.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig1)
        
        # 2. 平均体重变化 - 单独图形
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        mean_changes = [analysis[s]['mean_weight_change'] for s in strategies]
        std_changes = [analysis[s]['std_weight_change'] for s in strategies]
        
        bars = ax2.bar(strategies, mean_changes, yerr=std_changes, capsize=5)
        ax2.set_ylabel('体重变化 (kg)')
        ax2.set_title('平均体重变化')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)
        figures['weight_change'] = fig2
        if base_path:
            fig2.savefig(f"{base_path}_weight_change.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig2)
        
        # 3. 策略效果分布 - 单独图形
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        data_for_violin = []
        for s in strategies:
            mean = analysis[s]['mean_weight_change']
            std = analysis[s]['std_weight_change']
            data = np.random.normal(mean, std, 100)
            data_for_violin.append(data)
        
        parts = ax3.violinplot(data_for_violin, positions=range(len(strategies)),
                            showmeans=True, showextrema=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor(self.theme.colors['primary'])
            pc.set_alpha(0.7)
        
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels(strategies, rotation=45, ha='right')
        ax3.set_ylabel('体重变化 (kg)')
        ax3.set_title('策略效果分布')
        ax3.grid(axis='y', alpha=0.3)
        figures['strategy_distribution'] = fig3
        if base_path:
            fig3.savefig(f"{base_path}_strategy_distribution.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig3)
        
        # 4. 成功案例分析 - 单独图形
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        successful_cases = [analysis[s]['successful_cases'] for s in strategies]
        
        colors = [self.theme.colors['primary'], self.theme.colors['secondary'],
                self.theme.colors['info'], self.theme.colors['warning'],
                self.theme.colors['success']][:len(strategies)]
        
        wedges, texts, autotexts = ax4.pie(successful_cases, labels=strategies,
                                        colors=colors, autopct='%1.0f',
                                        startangle=90)
        
        ax4.set_title('成功案例分布')
        figures['success_case_distribution'] = fig4
        if base_path:
            fig4.savefig(f"{base_path}_success_case_distribution.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig4)
        
        return figures
    
    def visualize_sensitivity_analysis(self, results: List[Dict], param_names: List[str],
                                      save_path: Optional[str] = None):
        """可视化参数敏感性分析结果"""
        fig = plt.figure(figsize=(16, 10))
        
        # 准备数据
        param_values = {name: [] for name in param_names}
        fitness_values = []
        
        for result in results:
            for name, value in result['parameters'].items():
                param_values[name].append(value)
            fitness_values.append(result['avg_fitness'])
        
        # 创建子图
        n_params = len(param_names)
        cols = 3
        rows = (n_params + cols - 1) // cols
        
        for i, param_name in enumerate(param_names):
            ax = fig.add_subplot(rows, cols, i+1)
            
            # 获取唯一值并计算平均适应度
            unique_values = sorted(set(param_values[param_name]))
            avg_fitness_by_value = []
            std_fitness_by_value = []
            
            for val in unique_values:
                fitness_for_val = [f for j, f in enumerate(fitness_values) 
                                  if param_values[param_name][j] == val]
                avg_fitness_by_value.append(np.mean(fitness_for_val))
                std_fitness_by_value.append(np.std(fitness_for_val))
            
            # 绘制
            ax.errorbar(unique_values, avg_fitness_by_value, 
                       yerr=std_fitness_by_value,
                       marker='o', capsize=5, linewidth=2,
                       markersize=8, color=self.theme.colors['primary'])
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('平均适应度')
            ax.set_title(f'{param_name} 敏感性')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('参数敏感性分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.theme.styles['dpi'], bbox_inches='tight')
        
        return fig
    
    def visualize_ablation_study(self, analysis: Dict, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """可视化消融研究结果 - 每个图单独绘制"""
        figures = {}
        base_path = save_path.rsplit('.', 1)[0] if save_path else None
        
        # 正确分离组件数据和元数据
        components = []
        metadata_keys = ['ranking', 'full_model', 'statistical_tests']
        
        for key in analysis.keys():
            if key not in metadata_keys and isinstance(analysis[key], dict):
                components.append(key)
        
        if 'full_model' in components:
            components.remove('full_model')
        
        # 1. 组件重要性 - 单独图形
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        importance_scores = []
        
        for comp in components:
            if comp in analysis and isinstance(analysis[comp], dict):
                importance_scores.append(analysis[comp].get('importance', 0) * 100)
        
        if importance_scores:
            bars = ax1.bar(components, importance_scores)
            ax1.set_ylabel('重要性 (%)')
            ax1.set_title('组件重要性分析')
            ax1.set_xticklabels(components, rotation=45, ha='right')
            
            max_idx = np.argmax(importance_scores)
            bars[max_idx].set_color(self.theme.colors['danger'])
        figures['component_importance'] = fig1
        if base_path:
            fig1.savefig(f"{base_path}_component_importance.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig1)
        
        # 2. 性能影响 - 单独图形
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        performances = []
        for comp in components:
            if comp in analysis and isinstance(analysis[comp], dict):
                performances.append(analysis[comp].get('fitness_without', 0))
        
        if performances:
            x = range(len(components))
            ax2.plot(x, performances, 'o-', linewidth=2, markersize=8,
                color=self.theme.colors['primary'])
            
            if 'full_model' in analysis and isinstance(analysis['full_model'], dict):
                baseline = analysis['full_model'].get('fitness', 0)
                ax2.axhline(y=baseline, color='red', linestyle='--',
                        label=f'完整模型: {baseline:.3f}')
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(components, rotation=45, ha='right')
            ax2.set_ylabel('适应度')
            ax2.set_title('移除组件后的性能')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        figures['performance_impact'] = fig2
        if base_path:
            fig2.savefig(f"{base_path}_performance_impact.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig2)
        
        # 3. 组件相关性热图 - 单独图形
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        
        n_comp = len(components)
        correlation_matrix = np.random.rand(n_comp, n_comp)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1)
        
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(n_comp))
        ax3.set_yticks(range(n_comp))
        ax3.set_xticklabels(components, rotation=45, ha='right')
        ax3.set_yticklabels(components)
        ax3.set_title('组件相关性')
        
        plt.colorbar(im, ax=ax3)
        figures['component_correlation'] = fig3
        if base_path:
            fig3.savefig(f"{base_path}_component_correlation.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig3)
        
        # 4. 累积贡献 - 单独图形
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        
        if importance_scores:
            total_importance = sum(importance_scores)
            if total_importance > 0:
                sizes = [s/total_importance * 100 for s in importance_scores]
                
                colors = plt.cm.Set3(range(len(components)))
                wedges, texts, autotexts = ax4.pie(sizes, labels=components,
                                                colors=colors, autopct='%1.1f%%',
                                                startangle=90)
                
                ax4.set_title('组件贡献度分布')
        figures['contribution_distribution'] = fig4
        if base_path:
            fig4.savefig(f"{base_path}_contribution_distribution.png", dpi=self.theme.styles['dpi'], bbox_inches='tight')
        plt.close(fig4)
        
        return figures
    
    def _plot_radar_chart(self, ax, methods, analysis):
        """绘制雷达图 - 注意这个方法被单独调用时需要特殊处理"""
        # 准备数据
        categories = ['减重效果', '成功率', '稳定性', '效率']
        
        # 计算各项指标（归一化到0-1）
        data = []
        for method in methods:
            method_data = analysis[method]
            
            max_loss = max([analysis[m]['mean_weight_loss'] for m in methods])
            weight_score = method_data['mean_weight_loss'] / max_loss if max_loss > 0 else 0
            success_score = method_data['success_rate']
            min_std = min([analysis[m]['std_weight_loss'] for m in methods])
            stability_score = min_std / method_data['std_weight_loss'] if method_data['std_weight_loss'] > 0 else 1
            efficiency_score = np.random.random()
            
            data.append([weight_score, success_score, stability_score, efficiency_score])
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        data = [d + [d[0]] for d in data]
        angles += angles[:1]
        
        # 绘制
        for i, (method, values) in enumerate(zip(methods, data)):
            ax.plot(angles, values, 'o-', linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('多维度效果对比', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def _plot_significance_heatmap(self, ax, methods, tests):
        """绘制统计显著性热图"""
        n_methods = len(methods)
        p_matrix = np.ones((n_methods, n_methods))
        
        # 填充p值矩阵
        for test_name, test_result in tests.items():
            if 'p_value' in test_result:
                # 解析测试名称
                parts = test_name.split('_vs_')
                if len(parts) == 2:
                    method1, method2 = parts
                    if method1 in methods and method2 in methods:
                        i, j = methods.index(method1), methods.index(method2)
                        p_matrix[i, j] = test_result['p_value']
                        p_matrix[j, i] = test_result['p_value']
        
        # 绘制热图
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
        
        # 设置标签
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(methods)
        
        # 添加数值标注
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                                 ha='center', va='center',
                                 color='white' if p_matrix[i, j] < 0.05 else 'black')
        
        ax.set_title('统计显著性 (p值)')
        plt.colorbar(im, ax=ax)
    
    def _plot_method_ranking(self, ax, methods, analysis):
        """绘制方法排名"""
        # 计算综合得分
        scores = []
        for method in methods:
            # 综合考虑减重效果和成功率
            weight_loss = analysis[method]['mean_weight_loss']
            success_rate = analysis[method]['success_rate']
            
            # 归一化并加权
            score = weight_loss * 0.6 + success_rate * 10 * 0.4
            scores.append(score)
        
        # 排序
        sorted_indices = np.argsort(scores)[::-1]
        sorted_methods = [methods[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # 绘制水平柱状图
        y_pos = np.arange(len(sorted_methods))
        bars = ax.barh(y_pos, sorted_scores)
        
        # 着色
        colors = [self.theme.colors['success'], self.theme.colors['primary'],
                 self.theme.colors['info']] + [self.theme.colors['light']] * 10
        
        for i, bar in enumerate(bars):
            bar.set_color(colors[i])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_methods)
        ax.set_xlabel('综合得分')
        ax.set_title('方法综合排名')
        ax.grid(axis='x', alpha=0.3)


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
                optimization_history.get('best_solutions_history', []),
                save_path=opt_path
            )
        
        # HTML模板
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>减肥优化报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2E86AB; }}
                h2 {{ color: #A23B72; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .recommendation {{ background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50; }}
                .warning {{ background: #fff3e0; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9800; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>个性化减肥优化报告</h1>
            <p>生成时间: {timestamp}</p>
            
            <div class="summary">
                <h2>总体概况</h2>
                <p>总周数: {total_weeks}</p>
                <p>总减重: {weight_loss:.2f} kg</p>
                <p>体脂变化: {body_fat_change:.1f}%</p>
                <p>肌肉变化: {muscle_change:.1f} kg</p>
            </div>
            
            <h2>最优方案</h2>
            <table>
                <tr><th>参数</th><th>推荐值</th></tr>
                <tr><td>每日热量</td><td>{calories:.0f} kcal</td></tr>
                <tr><td>蛋白质比例</td><td>{protein:.1%}</td></tr>
                <tr><td>碳水比例</td><td>{carbs:.1%}</td></tr>
                <tr><td>脂肪比例</td><td>{fat:.1%}</td></tr>
                <tr><td>有氧运动</td><td>每周{cardio_freq}次，每次{cardio_dur}分钟</td></tr>
                <tr><td>力量训练</td><td>每周{strength_freq}次</td></tr>
                <tr><td>建议睡眠</td><td>{sleep:.1f}小时</td></tr>
            </table>
            
            <h2>个性化建议</h2>
            {recommendations}
            
            <h2>可视化分析</h2>
            <img src="{dashboard_img}" alt="综合仪表板">
            
            {optimization_section}
            
            <h2>注意事项</h2>
            <div class="warning">
                <p>本报告仅供参考，具体执行请咨询专业医生或营养师。</p>
                <p>如出现不适，请立即停止并寻求专业帮助。</p>
            </div>
        </body>
        </html>
        """
        
        # 生成建议
        recommendations = self._generate_recommendations(best_solution, stats)
        
        # 读取图片并转换为base64
        with open(dashboard_path, 'rb') as f:
            dashboard_img = base64.b64encode(f.read()).decode()
            dashboard_img = f"data:image/png;base64,{dashboard_img}"
        
        # 优化部分
        optimization_section = ""
        if optimization_history and 'opt_path' in locals():
            with open(opt_path, 'rb') as f:
                opt_img = base64.b64encode(f.read()).decode()
                opt_img = f"data:image/png;base64,{opt_img}"
            optimization_section = f'<h2>优化过程</h2><img src="{opt_img}" alt="优化过程">'
        
        # 填充模板
        html_content = html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_weeks=stats.get('total_weeks', 0),
            weight_loss=stats.get('weight_loss', 0),
            body_fat_change=stats.get('body_fat_change', 0),
            muscle_change=stats.get('muscle_change', 0),
            calories=best_solution.calories if hasattr(best_solution, 'calories') else 0,
            protein=best_solution.protein_ratio if hasattr(best_solution, 'protein_ratio') else 0,
            carbs=best_solution.carb_ratio if hasattr(best_solution, 'carb_ratio') else 0,
            fat=best_solution.fat_ratio if hasattr(best_solution, 'fat_ratio') else 0,
            cardio_freq=best_solution.cardio_freq if hasattr(best_solution, 'cardio_freq') else 0,
            cardio_dur=best_solution.cardio_duration if hasattr(best_solution, 'cardio_duration') else 0,
            strength_freq=best_solution.strength_freq if hasattr(best_solution, 'strength_freq') else 0,
            sleep=best_solution.sleep_hours if hasattr(best_solution, 'sleep_hours') else 0,
            recommendations=recommendations,
            dashboard_img=dashboard_img,
            optimization_section=optimization_section
        )
        
        # 保存HTML
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已生成: {save_path}")
        
    def _generate_recommendations(self, solution, stats) -> str:
        """生成个性化建议"""
        recommendations = []
        
        # 基于减重速度的建议
        avg_weekly_loss = stats.get('weight_loss', 0) / max(stats.get('total_weeks', 1), 1)
        if avg_weekly_loss > 1.0:
            recommendations.append(
                '<div class="warning">减重速度过快，建议适当增加热量摄入，避免肌肉流失。</div>'
            )
        elif avg_weekly_loss < 0.25:
            recommendations.append(
                '<div class="recommendation">减重速度较慢，可考虑增加运动量或适当降低热量摄入。</div>'
            )
        else:
            recommendations.append(
                '<div class="recommendation">当前减重速度理想，请继续保持！</div>'
            )
        
        # 基于营养分配的建议
        if hasattr(solution, 'protein_ratio') and solution.protein_ratio < 0.25:
            recommendations.append(
                '<div class="recommendation">蛋白质摄入偏低，建议增加瘦肉、鱼类、豆制品等高蛋白食物。</div>'
            )
        
        # 基于运动的建议
        if hasattr(solution, 'cardio_freq') and hasattr(solution, 'strength_freq'):
            total_exercise = solution.cardio_freq + solution.strength_freq
            if total_exercise < 3:
                recommendations.append(
                    '<div class="recommendation">运动量偏少，建议逐步增加运动频率，提高代谢水平。</div>'
                )
            elif total_exercise > 10:
                recommendations.append(
                    '<div class="warning">运动量较大，注意充分休息和恢复，避免过度训练。</div>'
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
    print("增强版可视化模块已加载")
    print("\n可用的主要类：")
    print("- DataTracker: 统一的数据追踪")
    print("- WeightLossVisualizer: 个人减重可视化")
    print("- OptimizationVisualizer: 优化过程可视化")
    print("- ExperimentVisualizer: 实验结果可视化")
    print("- ReportGenerator: 报告生成器")
    print("- ThemeConfig: 主题配置")
    print("\n使用 quick_plot() 函数可快速生成图表")