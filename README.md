# OpenAI API Debugger Tool

一个用于调试 OpenAI 标准接口服务的图形界面工具，支持实时流式响应、推理速度计算和对话管理。

## 功能特性

- **图形界面**: 基于 Tkinter 的跨平台 GUI，在 macOS 上完美运行
- **服务配置**: 可设置 API 地址、密钥、模型等参数
- **对话界面**: 实时显示对话内容，支持流式输出
- **速度监控**: 实时显示 tokens/秒 的推理速度
- **交互日志**: 详细的请求/响应日志，支持过滤和导出
- **预设模板**: 一键加载常用测试场景
- **参数编辑**: 可调整 temperature、max_tokens 等参数

## 安装依赖

```bash
pip install aiohttp tiktoken
```

Tkinter 通常随 Python 一起安装（macOS 已内置）。

## 使用方法

### 启动应用

```bash
python openai_debugger.py
```

### 主要界面区域

1. **左侧面板** - 配置区
   - API Configuration: 设置 Base URL、API Key、Model 等
   - Quick Actions: 预设测试模板
   - Performance: 实时显示生成速度和 token 数

2. **右侧面板** - 对话区
   - 显示完整的对话历史
   - 输入框支持多行编辑
   - Enter 发送，Shift+Enter 换行

3. **底部面板** - 日志区
   - 显示所有交互日志
   - 支持按级别过滤（INFO/SUCCESS/ERROR/REQUEST/RESPONSE）
   - 可导出日志到文件

### 预设测试模板

- Simple Hello: 简单问候测试
- System Instruction: 系统指令测试
- Code Review: 代码审查测试
- Translation Test: 翻译能力测试
- JSON Response: JSON 格式响应测试

## 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio

# 运行所有测试
pytest test_openai_debugger.py -v

# 运行特定测试类
pytest test_openai_debugger.py::TestTokenCounter -v
```

## 项目结构

```
/workspace/
├── openai_debugger.py      # 主程序（包含核心逻辑和 GUI）
├── test_openai_debugger.py # 单元测试（pytest）
└── README.md               # 说明文档
```

## 核心组件

- `APIConfig`: API 配置管理
- `ConversationHistory`: 对话历史管理
- `SpeedCalculator`: 实时速度计算
- `APIClient`: 异步 API 客户端
- `OpenAIDebuggerApp`: 主 GUI 应用
- `LogEntry`: 日志条目

## 兼容性

- macOS (原生支持)
- Linux
- Windows

## License

MIT
