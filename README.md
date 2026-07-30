# OpenAI 标准服务接口调试工具

一个基于 PyQt6 的图形界面调试工具，用于测试和调试兼容 OpenAI 标准 API 的服务。

## 功能特性

- **配置管理**: 设置 Base URL、API Key、模型名称
- **模型列表获取**: 一键从服务端获取可用模型列表并选择
- **对话界面**: 支持流式响应，实时显示推理速度 (tokens/s)
- **调试日志**: 实时显示交互日志，方便排查问题
- **配置持久化**: 自动保存和加载配置

## 安装依赖

```bash
pip install pyqt6 requests
```

## 运行方式

### macOS / Linux (有图形界面)
```bash
python run.py
```

### Headless 环境 (Docker/WSL/远程服务器)
```bash
QT_QPA_PLATFORM=offscreen python run.py
```

### Windows
```bash
python run.py
```

## 使用说明

1. **配置连接参数**
   - Base URL: 服务地址 (如 `http://localhost:11434/v1` 或 `https://api.openai.com/v1`)
   - API Key: API 密钥 (可选，取决于服务)
   - 点击"获取模型列表"按钮自动填充模型名称

2. **选择模型**
   - 如果有多个模型，会弹出对话框供选择

3. **开始对话**
   - 在输入框中输入消息
   - 点击"发送"按钮或按 Enter 键
   - 查看实时响应和推理速度

4. **查看日志**
   - 底部日志面板显示详细的交互信息
   - 包括请求、响应、错误等

## 项目结构

```
/workspace
├── run.py                 # 应用入口
├── README.md              # 本文档
├── src/
│   ├── main.py           # PyQt6 GUI 主程序
│   ├── config_manager.py # 配置管理
│   ├── api_client.py     # API 客户端
│   ├── message_history.py# 对话历史管理
│   ├── speed_calculator.py# 推理速度计算
│   ├── logger.py         # 日志管理
│   └── log_entry.py      # 日志条目类
└── tests/
    ├── test_openai_debug_tool.py  # 核心模块测试
    └── test_gui.py       # GUI 组件测试
```

## 测试

运行所有测试：
```bash
pytest tests/ -v
```

## 兼容性

支持任何兼容 OpenAI 标准 API 的服务：
- OpenAI API (`https://api.openai.com/v1`)
- Ollama 本地服务
- LM Studio
- vLLM
- 其他兼容实现

## 注意事项

- 如果遇到 `qt.qpa.plugin` 错误，请尝试使用 `QT_QPA_PLATFORM=offscreen` 环境变量
- 首次运行时会在用户目录创建 `~/.openai_debug_tool/config.json` 保存配置
