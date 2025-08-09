"""
测试中文字体显示
"""

from font_manager import test_font

if __name__ == "__main__":
    print("开始测试中文字体显示...")
    test_font()
    print("\n如果看到正常的中文，说明字体设置成功！")
    print("如果仍然是方框，请检查：")
    print("1. 系统是否安装了中文字体")
    print("2. 运行 pip install --upgrade matplotlib")
    print("3. 删除 matplotlib 缓存：~/.matplotlib/fontlist-*.json")