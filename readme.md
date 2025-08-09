# 差分进化算法减肥优化系统 使用说明

## 项目结构

项目结构如下：

```
.
├── main.py                 # 主程序入口
├── de_algorithm.py         # 差分进化算法核心实现
├── solution_generator.py   # 方案生成和管理
├── metabolic_model.py      # 人体代谢模型
├── fitness_evaluator.py    # 适应度评估
├── visualization.py        # 数据可视化
├── config.py              # 配置管理
└── README.md              # 本文件
```

## 快速开始

### 1. 基本使用

```bash
# 交互式模式（推荐新用户）
python main.py

# 使用默认参数快速运行
python main.py --mode default

# 测试模式（快速验证）
python main.py --mode test
```

### 2. 命令行参数

```bash
python main.py [选项]

选项:
  --mode {interactive,default,test}  
        运行模式 (默认: interactive)
        - interactive: 交互式输入用户信息
        - default: 使用默认用户档案
        - test: 快速测试模式
        
  --preset {aggressive,balanced,conservative}  
        配置预设 (默认: balanced)
        - aggressive: 激进模式（快速减重）
        - balanced: 平衡模式（推荐）
        - conservative: 保守模式（稳健减重）
        
  --output OUTPUT              
        输出目录 (默认: ./results)
        
  --log-level {DEBUG,INFO,WARNING,ERROR}  
        日志级别 (默认: INFO)
        
  --advanced                   
        启用高级功能
```

### 3. 使用示例

```bash
# 示例1: 使用平衡模式进行优化
python main.py --mode default --preset balanced

# 示例2: 激进模式，输出到指定目录
python main.py --mode default --preset aggressive --output ./my_results

# 示例3: 启用高级功能的交互式模式
python main.py --mode interactive --advanced

# 示例4: 快速测试
python main.py --mode test --output ./test_results
```

## 输出文件说明

运行后会在输出目录（默认`./results`）生成以下文件：

1. **optimization_[时间戳].log** - 运行日志
2. **tracking_data_[时间戳].json** - 详细的追踪数据
3. **config_[时间戳].json** - 使用的配置文件
4. **weight_loss_report.html** - 综合HTML报告（推荐查看）
5. **图表文件**：
   - optimization_results.png - 优化结果
   - optimization_progress.png - 优化进展
   - solution_space.png - 解空间分布
   - weight_loss_dashboard.png - 综合仪表板
   - weight_loss_animation.gif - 动画（可选）

## 模块说明

### 核心模块职责

- **main.py**: 程序入口，协调所有模块
- **de_algorithm.py**: 只包含差分进化算法实现
- **solution_generator.py**: 管理所有方案相关操作
- **config.py**: 纯配置管理，无业务逻辑
- **其他模块**: 保持原有功能

### 为什么要重构？

1. **职责分离**: 每个模块只负责一项功能
2. **避免重复**: 消除了功能重叠
3. **易于维护**: 清晰的模块边界
4. **灵活扩展**: 便于添加新功能

## 进阶使用

### 自定义配置

```python
# 在代码中自定义配置
from config import ConfigManager

config = ConfigManager()
config.algorithm.population_size = 20
config.algorithm.max_iterations = 15
config.nutrition.calorie_deficit_range = (0.20, 0.35)
```

### 使用高级功能

启用 `--advanced` 参数后，系统会使用：
- 自适应适应度评估器
- 高级代谢模型（考虑激素影响）
- 更精确的NEAT计算

## 常见问题

**Q: 如何只运行算法测试？**
A: 使用 `python main.py --mode test` 进行快速测试。

**Q: 如何查看详细的优化过程？**
A: 使用 `--log-level DEBUG` 参数查看详细日志。

## 注意事项

1. 首次运行建议使用交互式模式
2. HTML报告包含最全面的结果展示
3. 图表使用中文字体，确保系统安装了支持的字体
