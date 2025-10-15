# Convert PDF (MathTextbook) to MarkDown

> 🚀 将数学类 PDF 教材自动转换为结构化 Markdown，并维持统一的章节层级与公式格式。

`ctt2md` 利用 `pdf2image` 将 PDF 页面渲染为图像，再通过 [OpenAI Python SDK](https://github.com/openai/openai-python) 调用 **Qwen 3 VL** 多模态模型识别文字、公式与排版。随后，内置的“两阶段异步精整”流水线会在窗口范围内推断章节标题与排版策略，并对识别结果做一致化处理，最终生成包含 `<titles>` 索引与逐页 `<page n>` 标记的干净 Markdown。

## ✨ 功能特性

- 📄 **高质量页面渲染**：使用 `pdf2image` + `Pillow` 将 PDF 转为 PNG，支持自定义 DPI。
- 🤖 **Qwen 3 VL 识别**：通过 OpenAI 兼容接口批量并发调用 `chat.completions`，精确捕捉公式（`$...$` / `$$...$$`）与表格。
- 🧠 **两阶段结构精整**：
  - Phase A：对滑动窗口内的页面推断全局标题与章节层级策略。
  - Phase B：依据策略重写 Markdown，只保留正文内容并保持标题一致性。
- 🧵 **异步并发优化**：图像识别、窗口推断与重写全程异步，支持自定义并发数、窗口大小与滑动步长。
- 🧩 **高度可定制**：可通过配置类或命令行指定模型、分辨率、额外提示词、标题作用域等参数。
- 📦 **库 & CLI 双用**：既可在脚本中调用，也提供 `ctt2md` 命令行工具批量处理多个 PDF。

## 📁 输出结果

生成的 Markdown 文件由两部分组成：

```text
<titles>
# 一级标题
## 二级标题
...
</titles>

<page 1>
...
</page 1>
<page 2>
...
</page 2>
...
```

每个 `<page n>` 区块内包含该页的标题与正文内容，方便二次处理或切页校对。

## 🛠 环境准备

1. **Python**：需要 Python 3.9 及以上版本。
2. **Poppler**：`pdf2image` 依赖 Poppler，请提前安装：
   - macOS（Homebrew）：`brew install poppler`
   - Ubuntu / Debian：`apt-get update && apt-get install -y poppler-utils`
   - Windows：下载 [Poppler for Windows](https://blog.alivate.com.au/poppler-windows/) 并将 `bin` 目录加入 `PATH`
3. **OpenAI / DashScope 凭证**：
   - 设置 `OPENAI_API_KEY` 为可访问 Qwen 3 VL 的密钥。
   - 默认使用 DashScope 兼容端点（`https://dashscope.aliyuncs.com/compatible-mode/v1`）。若需自建网关，可在初始化 `OpenAI` 客户端时调整 `base_url`。

## 📦 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -e .
```

项目依赖已在 `pyproject.toml` 中声明，包括 `pdf2image`、`openai`、`python-dotenv` 与 `tqdm`。

## 🚀 快速上手

### 命令行

```bash
export OPENAI_API_KEY="sk-..."
ctt2md my_textbook.pdf -o output.md
```

- 处理多个文件时，`-o/--output` 必须指向目录；若缺省，则输出到当前工作目录下的 `markdown/`：

  ```bash
  ctt2md chapter1.pdf chapter2.pdf --output converted_markdown/
  ```

- 常用参数：
  - `--model`：覆盖默认的 Qwen 模型（默认 `qwen3-vl-plus`）。
  - `--dpi`：调整 PDF 渲染分辨率（默认 100）。
  - `--max-output-tokens`：限制模型输出的最大 token 数。
  - `--extra-instruction`：追加额外提示词，可重复多次。

### 作为库调用

```python
from pathlib import Path
from ctt2md import ConversionConfig, PDFToMarkdownConverter

config = ConversionConfig(
    model="qwen3-vl-plus",
    dpi=150,
    extra_instructions=("请严格保留定理编号",),
    refine_window_size=40,
    refine_keep_tail=20,
)
converter = PDFToMarkdownConverter(config=config)
markdown_pages = converter.convert(Path("my_textbook.pdf"), output_path="my_textbook.md")
```

`convert` 返回一个列表，包含每页正文的 Markdown 字符串；若提供 `output_path`，还会将包含 `<titles>` 与 `<page n>` 的完整文档写入磁盘。

### 批量处理

```python
from ctt2md import PDFToMarkdownConverter

converter = PDFToMarkdownConverter()
converter.convert_many(["algebra.pdf", "calculus.pdf"], output_dir="markdown")
```

`convert_many` 会为每个 PDF 生成同名的 `.md` 文件。

## ⚙️ 关键配置

`ConversionConfig` 提供以下重要字段（括号内为默认值）：

| 字段 | 说明 |
| ---- | ---- |
| `model` (`"qwen3-vl-plus"`) | 图像转 Markdown 所使用的 Qwen VL 模型标识。 |
| `dpi` (`100`) | PDF 渲染 DPI，值越大越清晰但速度越慢。 |
| `image2md_concurrency` (`20`) | 图像转 Markdown 的最大并发数。 |
| `refine_enabled` (`True`) | 是否启用两阶段精整流水线。 |
| `refine_model` (`"qwen-plus"`) | Phase A/B 使用的文本重写模型。 |
| `refine_window_size` (`50`) | 窗口大小 `W`，每批处理的页数。 |
| `refine_step` (`50`) | 滑动窗口步长 `S`。 |
| `refine_keep_tail` (`59`) | Phase B 每个窗口写回的尾部页数 `K`（实际生效值为 `min(K, W)`）。 |
| `refine_titles_scope` (`"global"`) | Phase B 使用全局还是窗口级标题策略。 |
| `extra_instructions` (`()`) | 追加到基础提示词的额外说明。 |
| `show_progress` (`True`) | 是否显示 `tqdm` 进度条。 |

通过调整上述参数，可在准确度与性能之间取得平衡。

## 🔁 工作流程

1. **PDF → 图像**：`pdf2image.convert_from_path` 将每页渲染为 PNG，并存入内存缓冲区。
2. **异步识别**：并发调用 Qwen 3 VL，将图像作为 base64 data URL 发送到 `chat.completions` 接口，返回初版 Markdown。
3. **Phase A（标题推断）**：对窗口内的页面进行标题抽取与层级规则归纳，累计形成全局 `<titles>` 与 `<heading_policy>`。
4. **Phase B（结构重写）**：基于推断结果对 Markdown 进行清洗，只保留正文与标题，删除页眉、脚注、噪声；并对空白或非正文页输出占位。
5. **结果合并**：缺失的页将退回初版 Markdown；最终输出 `<titles>` 与逐页 `<page n>` 区块，写入目标文件或返回列表。

## 🧯 故障排查

- **401/403 错误**：确认 `OPENAI_API_KEY` 有权限访问目标模型，或检查 `base_url` 是否指向正确网关。
- **Poppler 未找到**：在终端执行 `which pdftoppm` 验证安装，必要时重启终端使 `PATH` 生效。
- **输出缺字/乱码**：提高 `--dpi` 或添加额外提示词增强版面识别；也可调低并发确保接口稳定。
- **标题层级异常**：调整 `refine_window_size` / `refine_step` 以便 Phase A 捕捉更多上下文，或切换 `refine_titles_scope`。

## 📜 许可证

本项目基于 MIT 许可证开源。欢迎提交 Issue 或 PR 改进工具体验。
