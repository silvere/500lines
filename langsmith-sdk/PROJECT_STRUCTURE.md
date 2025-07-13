# LangSmith Agent SDK - 项目结构

本文档描述了LangSmith Agent SDK的项目结构和各个文件的作用。

## 项目目录结构

```
langsmith-sdk/
├── langsmith_agent_sdk.py      # 核心SDK实现
├── __init__.py                 # 包初始化文件
├── cli.py                      # 命令行工具
├── requirements.txt            # 依赖文件
├── setup.py                    # 安装配置
├── quick_start.py             # 快速开始脚本
├── README.md                  # 主要文档
├── USAGE_GUIDE.md             # 详细使用指南
├── PROJECT_STRUCTURE.md       # 项目结构说明（本文件）
└── examples/                  # 示例代码目录
    ├── basic_usage.py         # 基本使用示例
    └── advanced_usage.py      # 高级使用示例
```

## 核心文件说明

### 1. `langsmith_agent_sdk.py`
**主要的SDK实现文件**

包含：
- `LangSmithAgentSDK` - 主要SDK类
- `AgentTrace` - Trace数据模型
- `EvaluationResult` - 评估结果模型
- 内置评估器函数
- 工具函数

核心功能：
- Trace创建和管理
- 步骤记录
- 评估系统
- 数据分析
- 数据导出

### 2. `__init__.py`
**包初始化文件**

作用：
- 导出公共API
- 设置包版本信息
- 定义 `__all__` 列表

### 3. `cli.py`
**命令行工具**

提供命令：
- `analytics` - 获取分析数据
- `export` - 导出trace数据
- `list` - 列出项目信息

使用示例：
```bash
python -m langsmith_agent_sdk.cli analytics --project my-project
python -m langsmith_agent_sdk.cli export output.json --format json
```

### 4. `requirements.txt`
**依赖文件**

包含必要的Python包：
- `langsmith` - LangSmith官方客户端
- `langchain` - LangChain核心库
- `pydantic` - 数据验证
- `rich` - 美化控制台输出
- 其他支持库

### 5. `setup.py`
**安装配置文件**

用于：
- 包的安装和分发
- 依赖管理
- 元数据定义
- 命令行工具注册

安装方式：
```bash
pip install -e .  # 开发模式
pip install .     # 正常安装
```

## 文档文件

### 1. `README.md`
**主要文档**

内容：
- 项目概述和特性
- 安装指南
- 快速开始教程
- API参考
- 使用示例

### 2. `USAGE_GUIDE.md`
**详细使用指南**

内容：
- 完整的使用教程
- 高级功能说明
- 最佳实践
- 故障排除

### 3. `PROJECT_STRUCTURE.md`
**项目结构说明**

内容：
- 目录结构
- 文件作用说明
- 使用建议

## 示例代码

### 1. `quick_start.py`
**快速开始脚本**

特点：
- 自动检查环境
- 演示核心功能
- 友好的用户界面
- 错误处理

运行：
```bash
python quick_start.py
```

### 2. `examples/basic_usage.py`
**基本使用示例**

包含：
- 手动trace管理
- 上下文管理器使用
- 装饰器使用
- 基本评估
- 数据分析

### 3. `examples/advanced_usage.py`
**高级使用示例**

包含：
- 复杂Agent类设计
- 自定义评估器
- 批量评估
- 实时监控
- 错误处理策略

## 使用建议

### 新用户入门路径

1. **阅读README.md** - 了解项目概述
2. **运行quick_start.py** - 体验核心功能
3. **查看basic_usage.py** - 学习基本用法
4. **阅读USAGE_GUIDE.md** - 深入了解功能

### 开发者路径

1. **阅读项目结构** - 理解代码组织
2. **查看langsmith_agent_sdk.py** - 了解核心实现
3. **运行advanced_usage.py** - 学习高级功能
4. **自定义和扩展** - 根据需求修改

### 部署建议

1. **开发环境**：
   ```bash
   pip install -e .
   ```

2. **生产环境**：
   ```bash
   pip install .
   ```

3. **Docker化**：
   ```dockerfile
   FROM python:3.9
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   RUN pip install .
   ```

## 扩展建议

### 添加新功能

1. **新的评估器**：
   - 在`langsmith_agent_sdk.py`中添加函数
   - 使用`@custom_evaluator`装饰器
   - 在`__init__.py`中导出

2. **新的CLI命令**：
   - 在`cli.py`中添加子命令
   - 实现对应的处理函数

3. **新的数据模型**：
   - 使用Pydantic定义模型
   - 添加到核心SDK中

### 测试建议

1. **单元测试**：
   ```bash
   pytest tests/
   ```

2. **集成测试**：
   ```bash
   python examples/basic_usage.py
   python examples/advanced_usage.py
   ```

3. **性能测试**：
   - 使用大数据集测试
   - 监控内存使用
   - 测试并发性能

## 维护建议

### 版本管理

1. **语义化版本**：
   - 主版本：不兼容的API更改
   - 次版本：向后兼容的新功能
   - 补丁版本：向后兼容的bug修复

2. **更新流程**：
   - 更新`setup.py`中的版本号
   - 更新`__init__.py`中的版本号
   - 更新文档中的版本信息

### 依赖管理

1. **定期更新**：
   - 检查依赖包的新版本
   - 测试兼容性
   - 更新`requirements.txt`

2. **安全检查**：
   - 使用`pip-audit`检查安全漏洞
   - 及时更新有漏洞的包

### 文档维护

1. **保持同步**：
   - 代码更改时同步更新文档
   - 确保示例代码可运行
   - 定期检查链接有效性

2. **用户反馈**：
   - 收集用户反馈
   - 改进文档结构
   - 添加常见问题解答

## 贡献指南

### 代码贡献

1. **Fork项目**
2. **创建功能分支**
3. **编写测试**
4. **提交PR**

### 文档贡献

1. **改进现有文档**
2. **添加新的示例**
3. **翻译文档**
4. **报告错误**

---

如果你有任何问题或建议，请创建Issue或提交Pull Request。