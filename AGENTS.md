# AGENTS.md — 项目协作约定

## 用户背景与说明方式

- 用户是前端工程师，精通 JS/TS，目前正在用 Python 学习 AI Agent；本项目所有示例代码都是学习材料。
- 解释 Python 概念、语法、库 API 时，**优先用 JS/TS 类比**，并给出"等价关系 + 关键差异"：
  - `class X(TypedDict)` ↔ TS `interface` / zod schema（运行时存在、可转 JSON Schema）
  - `Annotated[str, "说明"]` ↔ `z.string().describe("说明")`
  - mypy / pyright ↔ `tsc`；Python 类型注解 ↔ TS 类型标注
  - Python 缩进 ↔ JS 花括号；`#` ↔ `//`；`print()` ↔ `console.log()`；命名参数 ↔ options 对象
- 不要默认用户懂 Python 语法细节；术语第一次出现时用一行说明。

## 回复风格

- **简洁直接**：先给结论/结果，再给必要细节；不复述用户已知内容，不用"如您所知"式开头。
- 用中文回复；代码示例保持与项目现有风格一致（中文注释）。

## 代码交付规范（必须遵守）

- **写完或修改任何代码后，必须运行验证命令检测错误，修复后再交付**；不得只贴代码并声称"应该没问题"。
- 验证命令（在本项目根目录执行）：
  1. 运行脚本：`uv run python -u src/<脚本名>.py`（src/ 下的示例脚本均可直接运行）
  2. 类型检查：`uvx pyright src/`（uvx 临时运行，不装进项目依赖；报错必须修复）
- 报错处理流程：读完整报错 → 定位原因（必要时查官方文档/API）→ 修复 → 重新运行验证。
- 交付时报告：实跑结果（成功输出或报错摘要）+ 验证过哪些命令；不夸大"已验证"。

## 项目环境备忘

- Python 3.14，uv 管理依赖（pyproject.toml），`uv run` 执行脚本；.env 存放 API key。
- 模型 deepseek-v4-flash（OpenAI 兼容接口，base_url 指向 api.deepseek.com）：
  - `llm.with_structured_output()` **必须传 `method="function_calling"`**——DeepSeek 不支持默认的 `json_schema` response_format。
  - 该模型**默认开启思考模式**，思考模式下不允许强制 tool_choice，结构化输出时需要禁用：`extra_body={"thinking": {"type": "disabled"}}`。
- 学习示例按章节放在 `src/`，命名风格：`章节_知识点.py`；新示例遵循同样风格并保留知识速览注释。
