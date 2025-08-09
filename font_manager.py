"""
字体管理模块
解决中文显示问题
"""

import os
import sys
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import logging

logger = logging.getLogger(__name__)


class ChineseFontManager:
    """中文字体管理器"""
    
    def __init__(self):
        self.system = platform.system()
        self.font_found = False
        self.font_path = None
        self.font_prop = None
        
    def find_chinese_font(self):
        """查找可用的中文字体"""
        # 清除字体缓存
        try:
            fm._rebuild()
        except:
            pass
            
        # 获取所有可用字体
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        
        # 定义要查找的字体文件名（按优先级）
        if self.system == 'Windows':
            # Windows字体路径
            font_candidates = [
                'C:/Windows/Fonts/msyh.ttc',      # 微软雅黑
                'C:/Windows/Fonts/msyhbd.ttc',    # 微软雅黑粗体
                'C:/Windows/Fonts/simhei.ttf',    # 黑体
                'C:/Windows/Fonts/simsun.ttc',    # 宋体
                'C:/Windows/Fonts/simkai.ttf',    # 楷体
                'C:/Windows/Fonts/STKAITI.TTF',   # 华文楷体
                'C:/Windows/Fonts/STXINWEI.TTF',  # 华文新魏
                'C:/Windows/Fonts/STSONG.TTF',    # 华文宋体
            ]
        elif self.system == 'Darwin':  # macOS
            font_candidates = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/System/Library/Fonts/STHeiti Medium.ttc',
                '/Library/Fonts/Songti.ttc',
                '/System/Library/Fonts/Hiragino Sans GB.ttc',
                '/Library/Fonts/Arial Unicode.ttf',
            ]
        else:  # Linux
            font_candidates = [
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
            ]
            
            # 添加更多Linux字体路径
            for font_dir in ['/usr/share/fonts', '/usr/local/share/fonts', '~/.fonts']:
                if os.path.exists(os.path.expanduser(font_dir)):
                    for root, dirs, files in os.walk(os.path.expanduser(font_dir)):
                        for file in files:
                            if 'chinese' in file.lower() or 'cjk' in file.lower():
                                font_candidates.append(os.path.join(root, file))
        
        # 查找第一个存在的字体文件
        for font_path in font_candidates:
            if os.path.exists(font_path):
                self.font_path = font_path
                self.font_found = True
                logger.info(f"找到中文字体: {font_path}")
                return font_path
                
        # 如果预定义路径都没找到，搜索系统字体
        chinese_fonts = []
        for font in font_list:
            try:
                # 尝试加载字体并检查是否支持中文
                prop = fm.FontProperties(fname=font)
                font_name = prop.get_name()
                
                # 检查字体名称是否包含中文相关关键词
                chinese_keywords = ['chinese', 'cjk', 'cn', 'sc', 'tc', '中', '宋', '黑', '楷', '微软', 'noto', 'droid', 'pingfang', 'heiti', 'songti']
                if any(keyword in font_name.lower() or keyword in font.lower() for keyword in chinese_keywords):
                    chinese_fonts.append(font)
            except:
                continue
                
        if chinese_fonts:
            self.font_path = chinese_fonts[0]
            self.font_found = True
            logger.info(f"找到中文字体: {self.font_path}")
            return self.font_path
            
        logger.warning("未找到中文字体")
        return None
        
    def setup_matplotlib(self):
        """设置matplotlib使用中文字体"""
        if not self.font_found:
            self.find_chinese_font()
            
        if self.font_path:
            try:
                # 方法1：直接设置字体文件路径
                self.font_prop = FontProperties(fname=self.font_path)
                
                # 方法2：添加字体到matplotlib
                fm.fontManager.addfont(self.font_path)
                prop = fm.FontProperties(fname=self.font_path)
                font_name = prop.get_name()
                
                # 设置matplotlib参数
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                plt.rcParams['font.family'] = 'sans-serif'
                
                logger.info(f"Matplotlib字体设置为: {font_name}")
                
            except Exception as e:
                logger.error(f"设置字体失败: {e}")
                # 使用备用方案
                self._use_fallback()
        else:
            # 没找到字体，使用备用方案
            self._use_fallback()
            
        # 始终设置这个参数，避免负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
    def _use_fallback(self):
        """使用备用字体方案"""
        logger.warning("使用备用字体方案")
        
        # 尝试使用系统默认字体
        if self.system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'sans-serif']
        elif self.system == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'sans-serif']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
            
        plt.rcParams['font.family'] = 'sans-serif'
        
    def get_font_prop(self):
        """获取字体属性对象（用于单独设置文本）"""
        if not self.font_prop and self.font_path:
            self.font_prop = FontProperties(fname=self.font_path)
        return self.font_prop
        
    def test_chinese_display(self):
        """测试中文显示"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.7, '中文字体测试', fontsize=20, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.5, '微软雅黑 黑体 宋体 楷体', fontsize=16, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.3, '1234567890 ABCDEFG abcdefg', fontsize=14, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.1, '数学符号: ±×÷≈≠∑∏∫√', fontsize=14, ha='center', transform=ax.transAxes)
        ax.set_title('中文显示测试图')
        ax.set_xlabel('X轴标签')
        ax.set_ylabel('Y轴标签')
        
        plt.tight_layout()
        plt.savefig('chinese_font_test.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("测试图已保存为 chinese_font_test.png")
        print(f"当前使用字体: {self.font_path if self.font_path else '系统默认'}")
        

# 创建全局字体管理器实例
font_manager = ChineseFontManager()


def setup_chinese_font():
    """设置中文字体的便捷函数"""
    font_manager.setup_matplotlib()
    return font_manager.font_found


def get_chinese_font_prop():
    """获取中文字体属性"""
    return font_manager.get_font_prop()


def test_font():
    """测试字体显示"""
    font_manager.test_chinese_display()


# 自动初始化
setup_chinese_font()