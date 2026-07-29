# OpenAI Debug Tool

一个用于调试 OpenAI 兼容 API 端点的图形界面工具，支持 macOS、Windows 和 Linux。

## 功能特性

- **配置管理**: 设置 Base URL、API Key、模型名称
- **对话界面**: 显示用户和助手的对话历史
- **实时推理速度**: 显示 tokens/秒的实时计算速度
- **Token 计数**: 显示当前对话的 token 数量
- **API 通信日志**: 完整的请求/响应详情，方便 debug
- **流式响应**: 支持实时流式输出显示
- **配置持久化**: 自动保存配置到本地文件

## 安装依赖

```bash
pip install httpx
```

## 使用方法

在 macOS（或其他平台）上运行：

```bash
python openai_debug_tool.py
```

## 界面说明

### 配置区域 (顶部)
- **Base URL**: OpenAI 兼容 API 的基础 URL（默认：https://api.openai.com/v1）
- **API Key**: 你的 API 密钥
- **Model**: 要使用的模型名称（默认：gpt-3.5-turbo）
- **Save Config**: 保存当前配置

### 对话区域 (左侧)
- **Conversation**: 显示对话历史，不同角色使用不同颜色
  - 用户消息：蓝色
  - 助手消息：绿色
  - 系统消息：橙色
  - 错误消息：红色
- **Message 输入框**: 输入要发送的消息
  - `Enter`: 发送消息
  - `Shift+Enter`: 换行
  - `Control+Enter` / `Command+Enter` (Mac): 发送消息
- **按钮**:
  - **Send (Enter)**: 发送消息
  - **Stop**: 停止当前生成
  - **Clear Chat**: 清空对话历史
- **状态栏**: 显示推理速度和 token 数量

### 日志区域 (右侧)
- **API Communication Log**: 显示所有 API 通信详情
  - 请求信息（URL、方法、headers、body）
  - 响应信息（状态码、响应体）
  - 错误信息
- **按钮**:
  - **Clear Log**: 清空日志
  - **Export Log**: 导出日志为 JSON 文件

## 运行测试

```bash
pytest test_openai_debug_tool.py test_gui_components.py -v
```

## 配置文件

配置保存在 `~/.openai_debug_tool/config.json`

## 技术栈

- **GUI**: tkinter (跨平台)
- **HTTP 客户端**: httpx (异步)
- **测试**: pytest

## 系统要求

- Python 3.8+
- tkinter (通常随 Python 一起安装)
- httpx

## 注意事项

- 在 macOS 上，确保安装了 tkinter：`brew install python-tk`
- 在无头环境（headless）下无法运行 GUI，但核心模块可以正常导入和使用
