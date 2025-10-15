"""Utilities for converting PDF textbooks into Markdown using Qwen 3 VL (+ DeepSeek-R1 refinement)."""

from __future__ import annotations

import os
import re
import base64
import asyncio
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

from tqdm.auto import tqdm
from openai import OpenAI
from pdf2image import convert_from_path

# -------------------- PROMPTS --------------------

PROMPT_TEMPLATE = (
    # 识别规则（数学/Markdown/无代码围栏）
    "You are an OCR expert who rewrites textbook pages as Markdown. "
    "Recognise mathematics precisely and emit inline math between $ and block math "
    "between $$ fences using LaTeX. Preserve headings, numbered equations, "
    "and tables when possible. Return pure GitHub-flavoured Markdown only. "
    "Do NOT include code fences like ``` or ```markdown; reply plain text in Markdown.\n"

    # 目录页规则（去掉点线与页码）
    "If a page looks like a table of contents, extract only real headings/subheadings, "
    "and strictly remove dot leaders (e.g., '......', '···') and page numbers at the line ends or in brackets. "
    "Do not output page numbers anywhere.\n"

    # 配图占位规则（图像）
    "For figures/illustrations, output a one-line placeholder using Markdown image syntax with an empty URL, "
    "and a short description as alt text, e.g., '![figure: brief description]()'. "
    "Do not attempt to draw the image or include any binary data.\n"

    # 内容包裹规则（只把真正正文包进 content）
    "Only the actual body content (paragraphs, equations, tables, figure placeholders) must be wrapped with "
    "<|content|> ... </|content|>. Headings should be outside of this block as normal Markdown headings.\n"

    # 标题一致性规则 + 示例
    "Maintain STRICT heading-level consistency across pages:\n"
    "- If 'Chapter 1' used level-1 '#', then 'Chapter 2' MUST also use level-1 '#'.\n"
    "- If Arabic-numbered headings like '2.3 Gaussian Elimination' used level-3 '###', then "
    "subsequent headings like '3.1 ...' and '3.2 ...' MUST also use level-3 '###'.\n"
    "- Do NOT downgrade/upgrade heading levels arbitrarily between pages.\n"
    "- Never include page numbers, dot leaders, or trailing dots in headings.\n"

    # 其他
    "Ignore page numbers and running headers/footers. "
    "Use the main language in the picture to reply!!若带有中文则不要输出任何英文内容!!!（数学公式除外）"
    "If the current page is not main body content — such as a cover page, table of contents, preface, bibliographic metadata, epilogue, or a blank page — return an empty <|content|> NOT MAIN BODY CONTENT </|content|> block only."
)

# DeepSeek-R1 重整提示（必须删去非主体文本；保留页锚；返回 <|titles|> 分页标题）
DEEPSEEK_R1_REWRITE_PROMPT = """
你是教材排版与结构一致性专家。给你若干页由 OCR/VL 模型输出的 Markdown 文本（每页用 <|page i|> ... </|page i|> 包裹，i 为页码）。
请对这些页面进行“重整”：严格执行以下规则，并逐页输出，**必须保留并按原顺序输出每个页边界标记**。

【必须遵守的规则】
1) 仅保留“标题 + 正文内容”两部分。删除任何非正文内容：页眉、页脚、提示语、说明性文字、注释、无关噪声等。
2) 目录页处理：只保留真实标题/小节，**删除**点线（如“......”“···”）与任何页码；**标题末尾不得有页码或点线**。
3) 标题对齐：在整个文档范围内保持标题层级一致（同类标题使用相同 # 数量）。已知历史标题（可能为空）如下：
<|prior_titles|>
{prior_titles}
</|prior_titles|>
必须沿用相同层级策略，不得任意升级/降级。
“章”使用二级标题，“节”使用二级标题，中文数字带顿号如“一、”使用三级标题，阿拉伯数字带点如“1.”使用四级标题，“1.1”等使用五级标题！！！若给出的titles出现层级/重复问题，请在回复的<|titles|> ... </|titles|>中修复
4) 语言：若正文为中文，则**除数学公式外**不得出现任何英文词语或提示性句子。
5) 数学：行内数学使用 $...$；块级数学使用 $$...$$。不要使用代码围栏。
6) 输出结构约束（逐页）：
   - 标题必须在 <|content|> 块之外。
   - 正文（段落/表格/公式/图片占位）**必须且只能**出现在 <|content|> ... </|content|> 中。
   - 每页输出都必须按以下模板：
     <|page i|>
     # 或 ### 等标题（如有；严禁含页码/点线）
     <|content|>
     （本页正文，仅正文）
     </|content|>
     </|page i|>
7) 同时返回一个标题清单块，用于驱动后续页的层级对齐。**标题清单必须按页分组并保留页锚**，格式如下：
<|titles|>
<|page i|>
# 一级标题
## 二级标题
### 三级标题
</|page i|>
...
</|titles|>

请对输入各页做就地重整，严格遵守以上约束：
- 逐页输出并保留 <|page i|> ... </|page i|>；
- 最后附上 <|titles|> ... </|titles|>，其中每页子块按页锚分组，标题行只包含以 # 起始的 Markdown 标题。
不要添加任何其他注释/解释。
""".strip()

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$', re.M)
_CONTENT_RE = re.compile(r"<\|content\|>(.*?)</\|content\|>", re.S | re.I)
# 页块：起止标签都带页码（严格锚定）
_PAGE_SPLIT_RE = re.compile(r"<\|page\s+(\d+)\|>(.*?)</\|page\s+\1\|>", re.S | re.I)
_TITLES_BLOCK_RE = re.compile(r"<\|titles\|>(.*?)</\|titles\|>", re.S | re.I)

# -------------------- CONFIG --------------------

@dataclass(slots=True)
class ConversionConfig:
    """Configuration for PDF to Markdown conversion."""

    # Qwen VL
    model: str = "qwen3-vl-plus"
    dpi: int = 100
    prompt: str = PROMPT_TEMPLATE
    image_format: str = "PNG"
    extra_instructions: Sequence[str] = field(default_factory=tuple)
    show_progress: bool = True
    image2md_concurrency: int = 20  # 并发上限

    # 标题去重
    dedup_titles: bool = True
    dedup_strategy: str = "level_text"  # "level_text" | "text"

    # DeepSeek-R1 重整
    refine_enabled: bool = True
    refine_model: str = "deepseek-r1"
    refine_window_size: int = 5          # 固定 5
    refine_keep_tail: int = 3            # 仅采纳“后 3 页”
    refine_step: int = 3                 # 滑窗步长 3
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_extra_body: Dict[str, object] = field(default_factory=lambda: {"enable_thinking": True, "thinking_budget": 1500})

    def build_prompt(self) -> str:
        if not self.extra_instructions:
            return self.prompt
        return "\n".join([self.prompt, *self.extra_instructions])

# -------------------- CONVERTER --------------------

class PDFToMarkdownConverter:
    """Convert PDF pages into Markdown text using Qwen VL, then window-refine with DeepSeek-R1."""

    titles: List[str]

    def __init__(self, *, client: OpenAI | None = None, config: ConversionConfig | None = None,
                 deepseek_client: OpenAI | None = None) -> None:
        # Qwen（OpenAI 兼容）
        self.client = client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.config = config or ConversionConfig()

        # DeepSeek-R1（OpenAI 兼容端点）
        if deepseek_client is not None:
            self.deepseek = deepseek_client
        else:
            ds_key = os.getenv(self.config.deepseek_api_key_env)
            self.deepseek = OpenAI(api_key=ds_key, base_url=self.config.deepseek_base_url) if ds_key else None

        # 标题管理（仅在 R1 重整阶段使用）
        self._titles_lock = asyncio.Lock()
        self._titles_by_page: Dict[int, List[str]] = {}
        self.titles: List[str] = []
        self._seen_title_keys: set[str] = set()

    # ---------------- 外部接口 ----------------

    def convert(self, pdf_path: str | Path, *, output_path: str | Path | None = None) -> None:
        asyncio.run(self._convert_async(pdf_path, output_path=output_path))
        return None

    def convert_many(self, pdf_paths: Iterable[str | Path], *, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for pdf in pdf_paths:
            pdf_path = Path(pdf)
            target = output_dir / (pdf_path.stem + ".md")
            self.convert(pdf_path, output_path=target)

    # ---------------- 内部：主流程 ----------------

    async def _convert_async(self, pdf_path: str | Path, *, output_path: str | Path | None = None) -> List[str]:
        pdf_path = Path(pdf_path)
        if output_path is not None:
            output_path = Path(output_path)

        # 1) PDF → Images
        images = await asyncio.to_thread(self._pdf_to_images, pdf_path)

        # 2) Qwen：每页原始 Markdown（不使用 titles 作为先验）
        qwen_md_per_page = await self._images_to_qwen_markdown_bounded(
            images,
            limit=self.config.image2md_concurrency,
            show_progress=self.config.show_progress,
            pdf_name=pdf_path.name,
        )  # List[str]（每页）

        # 3) R1 重整（窗=5，步=3，仅采纳后3页；未覆盖到的页用 Qwen 结果兜底）
        refined_md_per_page = await self._refine_with_sliding_windows(qwen_md_per_page)

        # 4) 从 R1 批次输出中的 <|titles|> 增量更新全局 titles；对每页抽取标题兜底
        await self._rebuild_global_titles_from_pages(refined_md_per_page)

        # 5) 落盘：先输出 <|titles|>（按页分组），再按页输出 <|page i|> … </|page i|>
        if output_path is not None:
            titles_block = self._build_titles_block_grouped()
            pages_block = self._build_pages_with_markers(refined_md_per_page)
            final_text = (titles_block + "\n\n" + pages_block).strip()
            await asyncio.to_thread(output_path.write_text, final_text, encoding="utf-8")

        # 返回纯正文（供调用方兼容使用）
        contents_only = [self._extract_content(md) for md in refined_md_per_page]
        return contents_only

    # ---------------- Qwen 阶段：并发识别（不注入 titles） ----------------

    async def _images_to_qwen_markdown_bounded(
        self,
        images: List[BytesIO],
        *,
        limit: int,
        show_progress: bool,
        pdf_name: str,
    ) -> List[str]:
        sem = asyncio.Semaphore(max(1, limit))

        async def worker(img: BytesIO, idx: int) -> Tuple[int, str]:
            async with sem:
                # 不再注入 prior_titles（按你的第 4 条要求）
                md_full = await asyncio.to_thread(self._image_to_markdown, img, idx)
                return idx, md_full

        tasks = [asyncio.create_task(worker(img, i)) for i, img in enumerate(images, start=1)]
        results: List[Optional[str]] = [None] * len(images)
        pbar = tqdm(total=len(images), desc=f"Converting {pdf_name}", unit="page") if show_progress else None
        try:
            for coro in asyncio.as_completed(tasks):
                idx, md_full = await coro
                results[idx - 1] = md_full
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()
        return [r if r is not None else "" for r in results]

    # ---------------- R1 阶段：滑窗=5，仅保留后3页 ----------------

    async def _refine_with_sliding_windows(self, md_pages: List[str]) -> List[str]:
        """
        - 窗口大小固定 5（若不足，则平移起点使窗口仍为 5）
        - 每次只采纳窗口的“后 3 页”作为本次有效重整结果
        - 窗口步长 3（覆盖所有 3..N 页；1、2 页由 Qwen 结果兜底）
        - 始终保留 <|page i|>…</|page i|> 标记
        - 同步更新全局 titles：优先使用 <|titles|> 块的分页面标题；否则用页内标题兜底
        """
        N = len(md_pages)
        refined = md_pages[:]  # 默认先放 Qwen 结果（用于 1、2 页兜底）
        if not (self.config.refine_enabled and self.deepseek and N > 0):
            return refined

        W = self.config.refine_window_size  # = 5
        K = self.config.refine_keep_tail    # = 3
        S = self.config.refine_step         # = 3

        # 预先清空（R1 才维护 titles）
        async with self._titles_lock:
            self._titles_by_page.clear()
            self._seen_title_keys.clear()
            self.titles.clear()

        # 主循环：以“最后页 pos”为 5、8、11... 构造窗口
        pos = W
        while pos <= N:
            start = pos - W + 1
            end = pos
            # 打包 5 页
            packed_text = self._pack_pages_with_markers(md_pages[start-1:end], list(range(start, end+1)))

            # prior_titles：使用目前全局已知标题（来自之前窗口）
            prior_titles_text = await self._get_prior_titles_text(page_number=start)

            prompt = DEEPSEEK_R1_REWRITE_PROMPT.format(prior_titles=prior_titles_text)

            try:
                refined_chunk = await asyncio.to_thread(self._call_deepseek_refine, prompt, packed_text)
                # 解析出分页内容 & <|titles|>
                per_page_map = self._split_pages_by_marker(refined_chunk)      # Dict[int, str]
                titles_by_page = self._parse_titles_block_grouped(refined_chunk)  # Dict[int, List[str]]

                # 仅采纳“后 3 页”
                keep_pages = list(range(end - K + 1, end + 1))
                for p in keep_pages:
                    if p in per_page_map and per_page_map[p].strip():
                        refined[p-1] = per_page_map[p].strip()

                # titles 覆盖更新（仅更新本窗口涉及页）
                await self._set_titles_for_pages(titles_by_page)

            except Exception:
                # 出错：跳过本窗口；不影响下一个窗口
                pass

            pos += S

        # 处理尾部不足 5 页的情况：把窗口向左平移至恰好 5 页
        if N % S != 2:  # 若无法落到 N==pos 的形式，则再补一窗（例如 N=7，pos 走到 5 后跳到 8>7）
            if N >= 3:
                end = N
                start = max(1, end - W + 1)
                if end - start + 1 < W:
                    start = max(1, end - W + 1)  # 再次确保窗口=5
                if end - start + 1 == W:
                    packed_text = self._pack_pages_with_markers(md_pages[start-1:end], list(range(start, end+1)))
                    prior_titles_text = await self._get_prior_titles_text(page_number=start)
                    prompt = DEEPSEEK_R1_REWRITE_PROMPT.format(prior_titles=prior_titles_text)
                    try:
                        refined_chunk = await asyncio.to_thread(self._call_deepseek_refine, prompt, packed_text)
                        per_page_map = self._split_pages_by_marker(refined_chunk)
                        titles_by_page = self._parse_titles_block_grouped(refined_chunk)
                        keep_pages = list(range(end - K + 1, end + 1))
                        for p in keep_pages:
                            if p in per_page_map and per_page_map[p].strip():
                                refined[p-1] = per_page_map[p].strip()
                        await self._set_titles_for_pages(titles_by_page)
                    except Exception:
                        pass

        return refined

    def _pack_pages_with_markers(self, pages_md: List[str], pages_nums: List[int]) -> str:
        """把若干页打包，并加上严格闭合的 <|page i|> … </|page i|> 边界（输入 R1）。"""
        blocks = []
        for pno, md in zip(pages_nums, pages_md):
            md = md.strip()
            # 确保页内不含旧的页锚，避免嵌套
            md = _PAGE_SPLIT_RE.sub(lambda _: "", md)
            blocks.append(f"<|page {pno}|>\n{md}\n</|page {pno}|>")
        return "\n\n".join(blocks)

    def _build_pages_with_markers(self, refined_md_per_page: List[str]) -> str:
        """落盘：将每页内容用 <|page i|> … </|page i|> 包裹并串联。"""
        blocks = []
        for i, md in enumerate(refined_md_per_page, start=1):
            body = md.strip()
            # 若 R1 已经返回带页锚，我们只取内部内容避免双重包裹
            m = _PAGE_SPLIT_RE.search(body)
            if m and int(m.group(1)) == i:
                body = m.group(2).strip()
            blocks.append(f"<|page {i}|>\n{body}\n</|page {i}|>")
        return "\n\n".join(blocks)

    def _call_deepseek_refine(self, prompt: str, chunk_text: str) -> str:
        """同步调用 DeepSeek-R1（OpenAI 兼容接口）。"""
        resp = self.deepseek.chat.completions.create(
            model=self.config.refine_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": chunk_text},
            ]}],
            extra_body=self.config.deepseek_extra_body or None,
        )
        return resp.choices[0].message.content or ""

    def _split_pages_by_marker(self, text: str) -> Dict[int, str]:
        """从文本中按严格页锚提取每页块（含“标题 + <|content|>正文”）。"""
        out: Dict[int, str] = {}
        for m in _PAGE_SPLIT_RE.finditer(text):
            pno = int(m.group(1))
            body = m.group(2).strip()
            out[pno] = f"<|page {pno}|>\n{body}\n</|page {pno}|>"
        return out

    def _parse_titles_block_grouped(self, text: str) -> Dict[int, List[str]]:
        """
        解析 R1 返回的 <|titles|> 块（按页分组，每页仍用 <|page i|> … </|page i|>）。
        返回：{page_no: ["# A", "## B", ...]}
        """
        res: Dict[int, List[str]] = {}
        tb = _TITLES_BLOCK_RE.search(text)
        if not tb:
            return res
        inner = tb.group(1)
        for m in _PAGE_SPLIT_RE.finditer(inner):
            pno = int(m.group(1))
            body = m.group(2)
            titles: List[str] = []
            for t in _HEADING_RE.finditer(body):
                level = t.group(1)
                text = t.group(2).strip()
                if text:
                    titles.append(f"{level} {text}")
            if titles:
                res[pno] = self._dedup_titles_list(titles)
        return res

    async def _set_titles_for_pages(self, titles_by_page: Dict[int, List[str]]) -> None:
        """使用 R1 返回的 <|titles|> 覆盖更新指定页的标题集合（全局去重策略生效）。"""
        if not titles_by_page:
            return
        async with self._titles_lock:
            for pno in sorted(titles_by_page.keys()):
                page_titles = titles_by_page[pno]
                filtered: List[str] = []
                for line in page_titles:
                    m = _HEADING_RE.match(line)
                    if not m:
                        continue
                    level, text = m.group(1), m.group(2).strip()
                    key = self._title_key(level, text)
                    if self.config.dedup_titles and key in self._seen_title_keys:
                        continue
                    self._seen_title_keys.add(key)
                    filtered.append(line)
                self._titles_by_page[pno] = filtered
            # 重建全量 titles
            lines: List[str] = []
            for p in sorted(self._titles_by_page.keys()):
                lines.extend(self._titles_by_page[p])
            self.titles = lines

    async def _rebuild_global_titles_from_pages(self, refined_md_per_page: List[str]) -> None:
        """
        若某些页未被 <|titles|> 覆盖，则从该页内容再抽取标题兜底，保证最终 titles 完整。
        """
        async with self._titles_lock:
            covered = set(self._titles_by_page.keys())
        for page_idx, md in enumerate(refined_md_per_page, start=1):
            if page_idx in covered:
                continue
            page_titles = self._extract_titles(md)
            await self._update_titles(page_idx, page_titles)

    # ---------------- Qwen 单页识别 ----------------

    def _image_to_markdown(self, image_stream: BytesIO, page_number: int) -> str:
        encoded_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
        mime = f"image/{self.config.image_format.lower()}"

        base = self.config.build_prompt()
        # 第 4 条：不再注入 prior_titles
        page_hint = f"\nThis image corresponds to textbook page {page_number}. Do not output page numbers anywhere.\n"
        prompt = f"{base}\n{page_hint}"

        data_url = f"data:{mime};base64,{encoded_image}"

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "QwenVL Markdown"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            extra_body={'enable_thinking': True, "thinking_budget": 500},
        )
        return response.choices[0].message.content

    # ---------------- 通用：PDF 渲染/标题/内容 ----------------

    def _pdf_to_images(self, pdf_path: Path) -> List[BytesIO]:
        pil_images = convert_from_path(str(pdf_path), dpi=self.config.dpi, fmt=self.config.image_format.lower())
        streams: List[BytesIO] = []
        for page in pil_images:
            buffer = BytesIO()
            page.save(buffer, format=self.config.image_format)
            buffer.seek(0)
            streams.append(buffer)
        return streams

    def _extract_titles(self, md: str) -> List[str]:
        titles: List[str] = []
        for m in _HEADING_RE.finditer(md):
            level = m.group(1)
            text = m.group(2).strip()
            # 过滤目录点线和页码
            if text.replace('.', '').strip().isdigit():
                continue
            if re.search(r'(?:\.{3,}|·{2,})\s*\d+\s*$', text):
                continue
            if re.search(r'\(\s*\d+\s*\)\s*$', text):
                continue
            titles.append(f"{level} {text}")
        return self._dedup_titles_list(titles)

    @staticmethod
    def _extract_content(md: str) -> str:
        blocks = [b.strip() for b in _CONTENT_RE.findall(md)]
        return "\n\n".join(b for b in blocks if b)

    async def _get_prior_titles_text(self, page_number: int) -> str:
        """
        仅供 R1 阶段使用：已完成的 < page_number 的所有页标题，按页序拼接。
        """
        async with self._titles_lock:
            ordered_pages = sorted(k for k in self._titles_by_page.keys() if k < page_number)
            lines: List[str] = []
            for p in ordered_pages:
                lines.extend(self._titles_by_page[p])
            return "\n".join(lines)

    async def _update_titles(self, page_number: int, page_titles: List[str]) -> None:
        """
        兜底：当某页未被 <|titles|> 覆盖时，从页内解析并合并；保持全局去重策略。
        """
        async with self._titles_lock:
            if self.config.dedup_titles:
                filtered: List[str] = []
                for line in page_titles:
                    m = _HEADING_RE.match(line)
                    if not m:
                        continue
                    level, text = m.group(1), m.group(2).strip()
                    key = self._title_key(level, text)
                    if key in self._seen_title_keys:
                        continue
                    self._seen_title_keys.add(key)
                    filtered.append(line)
                self._titles_by_page[page_number] = filtered
            else:
                self._titles_by_page[page_number] = page_titles

            lines: List[str] = []
            for p in sorted(self._titles_by_page.keys()):
                lines.extend(self._titles_by_page[p])
            self.titles = lines

    # ---------- 标题规范化/去重 ----------

    def _normalize_title_text(self, text: str) -> str:
        t = unicodedata.normalize("NFKC", text)
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"(?:\.{3,}|·{2,})\s*\d+\s*$", "", t).strip()
        t = re.sub(r"\(\s*\d+\s*\)\s*$", "", t).strip()
        t = re.sub(r"[\s\.\u3002、，,：:；;—\-·…]+$", "", t).strip()
        t = t.lower()
        return t

    def _title_key(self, level: str, text: str) -> str:
        norm = self._normalize_title_text(text)
        if self.config.dedup_strategy == "text":
            return norm
        return f"{len(level)}|{norm}"

    def _dedup_titles_list(self, titles: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for line in titles:
            m = _HEADING_RE.match(line)
            if not m:
                continue
            level, text = m.group(1), m.group(2).strip()
            key = self._title_key(level, text)
            if key in seen:
                continue
            seen.add(key)
            out.append(line)
        return out

    # ---------- 构建 <|titles|> 与分页面输出 ----------

    def _build_titles_block_grouped(self) -> str:
        """
        将全局 titles 按页分组输出为：
        <|titles|>
        <|page i|>
        # ...
        ## ...
        </|page i|>
        ...
        </|titles|>
        """
        parts = ["<|titles|>"]
        with_pages = sorted(self._titles_by_page.items(), key=lambda kv: kv[0])
        for pno, lines in with_pages:
            if not lines:
                continue
            parts.append(f"<|page {pno}|>")
            parts.extend(lines)
            parts.append(f"</|page {pno}|>")
        parts.append("</|titles|>")
        return "\n".join(parts).strip()


__all__ = ["PDFToMarkdownConverter", "ConversionConfig"]
