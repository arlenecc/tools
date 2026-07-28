# OpenAI API 调试工具

一个用于调试 OpenAI 标准接口的图形化工具，支持 macOS 平台。

## 功能特性

- **API 配置管理**: 设置服务地址、API Key、模型名称等
- **请求参数编辑**: 可调整 temperature、max_tokens、stream 等参数
- **自定义 HTTP Headers**: 支持添加自定义请求头
- **对话界面**: 直观的聊天对话显示
- **实时速度显示**: 显示实时的推理输出速度 (tokens/s)
- **交互日志**: 完整的请求/响应日志记录，支持过滤和导出
- **预设动作**: 
  - 测试联通性
  - 获取模型信息
  - Say Hello
  - 测试正常返回
  - 自定义请求
- **双击执行**: 双击预设动作直接发送请求

## 文件结构

```
/workspace/
├── openai_debugger.py      # GUI 主程序
├── openai_debugger_core.py # 核心逻辑模块（可独立测试）
├── test_openai_debugger.py # pytest 测试用例
└── README.md               # 说明文档
```

## 安装依赖

```bash
pip install aiohttp pytest pytest-asyncio
```

## 运行方式

### 启动 GUI 应用

```bash
python openai_debugger.py
```

### 运行测试

```bash
pytest test_openai_debugger.py -v
```

## 使用说明

1. **配置 API**: 在"API 配置"区域填写 Base URL、API Key 和 Model
2. **选择预设动作**: 双击左侧预设动作列表中的项目快速测试
3. **自定义请求**: 在输入框中输入消息，点击"发送"按钮
4. **查看日志**: 底部日志区域显示所有请求和响应的详细信息
5. **监控速度**: 对话区域右下角实时显示 tokens/s

## 注意事项

- 本工具需要 tkinter 支持，macOS 系统通常已预装
- 确保目标 API 服务支持 OpenAI 标准接口格式
- 流式响应需要服务器支持 text/event-stream 内容类型

## 许可证

MIT License
