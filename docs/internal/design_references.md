# PhenoCV 设计参考清单

> 项目定位：面向植物表型的**视觉通用工具**（不限于时序分割，未来可扩展检测/分类/深度估计等）。
> 参考项目：SAM 2 (facebookresearch/sam2)、Autodistill (autodistill/autodistill)、FiftyOne (voxel51/fiftyone)。
> 目的：为 PhenoCV 仓库的脚手架、README、CI、文档与跨平台 skill 提供可落地约定。

---

## 1. 命名与标识

| 项 | 值 |
|---|---|
| 项目名 / 仓库 | **PhenoCV** |
| 包名（import） | `phenocv` |
| CLI 命令 | `phenocv` |
| GitHub 仓库 | `perseus-wy/PhenoCV` |
| 本地路径 | （仓库根目录，clone 目标自选） |
| 许可证 | MIT |
| 发布身份 | `perseus-wy` + GitHub no-reply 邮箱（**不**用 qq 邮箱） |

---

## 2. 仓库结构约定

采用“核心包 + 适配器 + 示例 + 文档 + 跨平台 skill”分离布局，便于植物表型研究者直接复用，也便于 CV 开发者扩展。

```
PhenoCV/
├── .github/workflows/ci.yml      # pytest，无需 GPU
├── src/phenocv/                  # 可 pip 安装的核心包
│   ├── __init__.py
│   ├── engine.py                 # 时序分割引擎（数据源无关）
│   ├── cli.py                    # `phenocv` 命令行入口
│   ├── config.py                 # 配置加载与预设
│   ├── adapters/                 # 数据适配器
│   │   ├── __init__.py
│   │   ├── base.py               # 适配器基类
│   │   ├── csv_manifest.py       # 通用 CSV/JSON manifest 适配器（默认）
│   │   └── plant_phenotyping.py  # 植物表型示例适配器（大豆）
│   └── utils/                    # 通用工具（ROI、mask IO、边界 F1 等）
├── configs/
│   ├── default.yaml
│   └── presets/
│       ├── plant_phenotyping.yaml
│       ├── high_recall.yaml
│       └── rigid_object.yaml
├── tests/                        # CPU-only 单元测试
├── docs/
│   ├── internal/design_references.md   # 本文件
│   ├── tuning.md
│   ├── export_formats.md
│   └── adapter_guide.md
├── samples/                      # 真实数据不提交，仅放 README
├── tools/make_demo_sample.py     # 合成示例生成器
├── notebooks/
│   └── quickstart.ipynb
├── SKILL.md                      # 主技能文件（Anthropic Agent Skills 规范，中英双语）
├── AGENTS.md                     # OpenAI Codex 入口（薄包装，指向 SKILL.md）
├── CLAUDE.md                     # Claude Code 项目指令（薄包装，指向 SKILL.md）
├── README.md                     # 英文主文档
├── README.zh-CN.md               # 中文文档
├── LICENSE                       # MIT
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml
└── .gitignore
```

---

## 3. 跨平台 Skill 设计（WorkBuddy / Claude Code / Codex）

**核心原则：单源正文，多平台薄包装。**

### 3.1 主技能文件 `SKILL.md`
- 遵循 **Anthropic Agent Skills 规范**：文件以 YAML frontmatter 开头，至少含 `name` 与 `description` 字段。
- 该格式当前被 **Claude Code、WorkBuddy、Cursor** 等直接识别为可安装 skill。
- 正文**中英双语**：`description` 同时给出中英文；正文以英文为主、关键段落附中文。
- 内容覆盖：何时触发（时序分割 / 植物表型 / SAM2 video propagation）、核心概念、CLI 与 API 用法、适配器契约、预设、QA（LOO IoU）。

### 3.2 `CLAUDE.md`（Claude Code 项目指令）
- 作为 Claude Code 进入仓库时的项目级引导。
- 内容：一句话项目简介 + “see `SKILL.md` for the agent skill” + 关键命令速查。
- 不重复 SKILL.md 全文，避免漂移。

### 3.3 `AGENTS.md`（OpenAI Codex 入口）
- Codex 读取 `AGENTS.md` 作为仓库指令。
- 内容：项目简介 + 指向 `SKILL.md` 的技能说明 + 贡献/测试约定。
- 同样不重复正文，仅引用。

### 3.4 同步维护规则
- 技能逻辑只在 `SKILL.md` 维护；`CLAUDE.md` / `AGENTS.md` 仅做引用与导航。
- 修改技能正文后，检查三份文件的一致性（引用路径、命令名）。

---

## 4. README 章节约定（中英双语）

README.md 以英文为主，README.zh-CN.md 为中文翻译。主 README 顶部包含：

1. **项目一句话定位** — 面向植物表型的开源视觉工具，时序分割为首个模块。
2. **Quicklinks** — 文档、Colab、PyPI、License、Skill。
3. **徽章（Badges）** — 参考 Autodistill：
   - PyPI version / Python versions / License (MIT) / Tests / Colab Quickstart。
4. **Features / 核心特性** — 时序一致性、双向传播、ROI 裁剪、LOO 验证、多预设、适配器机制。
5. **Installation** — `pip install phenocv` + `pip install -e ".[dev]"` 源码安装。
6. **Quickstart** — 合成数据三行代码跑通；指向 `notebooks/quickstart.ipynb`。
7. **CLI Usage** — `phenocv segment` / `phenocv validate` / `phenocv export`。
8. **Adapter Contract** — 用户如何写自己的适配器。
9. **Presets** — 植物表型、高召回、刚性物体等场景。
10. **Architecture** — engine + adapter 分离简述。
11. **Citation** — BibTeX。
12. **License & Contributing** — 链接 LICENSE / CONTRIBUTING.md。

---

## 5. 文档风格约定

- 标题使用 emoji（参考 FiftyOne / Autodistill），增强可读性：
  - `## 🚀 Quickstart` / `## 💿 Installation` / `## 📚 Adapter Guide` / `## 🔧 Tuning`
- 关键概念用表格呈现（模型矩阵、预设对比、adapter 方法）。
- 代码块优先使用 ```python / ```bash，并保证可粘贴运行。
- 中文 README 保持相同结构，术语保留英文（ROI、LOO、SAM2、adapter）。

---

## 6. CLI 与 API 设计约定

- **Python API 为主，CLI 为辅**（参考 SAM 2）。
- CLI 子命令简洁：`phenocv segment --config ...`、`phenocv validate --loo`、`phenocv export --format isat`。
- 配置驱动：用户用 yaml 覆盖默认 `configs/default.yaml`，无需改代码。
- 所有关键步骤产出 `pred_source` 溯源列（manual/propagated/lowthr/point_rescue/failed），便于审计。
