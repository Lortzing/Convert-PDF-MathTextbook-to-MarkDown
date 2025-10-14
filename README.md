# ctt2md

将数学类 PDF 课本批量转换为 Markdown，结合 `pdf2image` 将页面渲染成图片，再使用 [OpenAI Python SDK](https://github.com/openai/openai-python) 调用 **Qwen 3 VL** 多模态模型进行文本与公式识别。

## 功能特性

- 📄 使用 `pdf2image` 将 PDF 页面渲染为高分辨率图像。
- 🤖 通过 OpenAI `responses` 接口调用 Qwen 3 VL 识别文本、公式与排版。
- 🧮 针对数学教材进行了提示词优化，输出含 LaTeX 公式的 GitHub 风格 Markdown。
- 🗂️ 支持一次性处理多个 PDF，并自动将结果写入 Markdown 文件。

## 环境准备

1. **Python**：需要 Python 3.9 及以上版本。
2. **Poppler**：`pdf2image` 依赖 Poppler，请先安装：
   - macOS（Homebrew）：`brew install poppler`
   - Ubuntu / Debian：`apt-get update && apt-get install -y poppler-utils`
   - Windows：下载 [Poppler for Windows](https://blog.alivate.com.au/poppler-windows/) 并将 `bin` 目录加入 `PATH`。
3. **OpenAI 访问凭证**：确保环境变量 `OPENAI_API_KEY` 指向可访问 Qwen 3 VL 的密钥，如果服务部署在自建网关上，还需要设置 `OPENAI_BASE_URL` 等自定义参数，OpenAI Python SDK 会自动读取这些配置。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 下使用 .venv\\Scripts\\activate
pip install -e .
```

## 使用方法

### 命令行

```bash
export OPENAI_API_KEY="sk-..."
ctt2md my_textbook.pdf -o output.md
```

- 传入多个 PDF 时，`-o/--output` 需要指定一个目录，未指定则默认输出到当前目录下的 `markdown/` 文件夹：

```bash
ctt2md chapter1.pdf chapter2.pdf --output converted_markdown/
```

- 其他常用参数：
  - `--model`：切换为不同的 Qwen 3 VL 型号，例如 `qwen-3.5-vl`。
  - `--dpi`：调整 PDF 渲染分辨率，默认 300。
  - `--max-output-tokens`：限制模型输出令牌数。
  - `--extra-instruction`：为提示词追加额外说明，可重复多次。

### 作为库使用

```python
from ctt2md import PDFToMarkdownConverter, ConversionConfig

config = ConversionConfig(model="qwen-3-vl", dpi=300)
converter = PDFToMarkdownConverter(config=config)
markdown_pages = converter.convert("my_textbook.pdf", output_path="my_textbook.md")
```

`convert` 返回每个页面的 Markdown 字符串列表，同时可选地写入输出文件。

## 工作流程

1. `pdf2image` 将 PDF 每页渲染为 PNG。
2. 将图像编码为 Base64 后发送给 Qwen 3 VL。
3. 模型返回 Markdown，包含数学公式（`$...$` 或 `$$...$$`）。
4. 按页面顺序合并并输出结果。

## 故障排查

- **模型请求失败**：检查 `OPENAI_API_KEY` 是否有效，若使用自建服务需确认 `OPENAI_BASE_URL` 设置。
- **Poppler 未找到**：确保系统路径包含 Poppler，可通过 `which pdftoppm` 验证。
- **输出公式异常**：尝试提高 DPI 或在命令中追加 `--extra-instruction` 指导模型处理特定排版。

## 许可证

本项目使用 MIT 许可证发布。
