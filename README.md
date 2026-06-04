# Image2Func - 图像转函数工具

将手绘曲线图像自动转换为数学函数表达式，并生成可视化图表。

## 功能特性

- 图像轮廓检测与提取
- 曲线拟合为数学函数
- 函数可视化渲染
- Web 界面交互

## 技术栈

- **后端**: Django
- **图像处理**: OpenCV, NumPy
- **函数拟合**: SciPy
- **可视化**: Matplotlib

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

访问 http://127.0.0.1:8000 使用 Web 界面。

## 项目结构

```
backend/
├── backend/              # Django 配置
├── image2func/           # 核心应用
│   ├── utils/            # 工具模块
│   │   ├── contour_processor.py  # 轮廓处理
│   │   ├── frame_worker.py       # 帧处理
│   │   ├── function_fitter.py    # 函数拟合
│   │   └── function_plotter.py   # 函数绘图
│   ├── templates/        # 模板文件
│   └── views.py          # 视图
└── media/                # 媒体文件
```

## 许可证

MIT License