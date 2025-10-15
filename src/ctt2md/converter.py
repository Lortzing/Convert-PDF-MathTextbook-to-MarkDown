"""Utilities for converting PDF textbooks into Markdown using Qwen 3 VL (+ Async Refine)."""

from __future__ import annotations

import os
import re
import base64
import asyncio
import contextlib
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
    "<content> ... </content>. Headings should be outside of this block as normal Markdown headings.\n"

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
    "If the current page is not main body content — such as a cover page, table of contents, preface, bibliographic metadata, epilogue, or a blank page — return an empty <content> NOT MAIN BODY CONTENT </content> block only."
)

# Refine 模型提示（删去非正文；严格页锚；返回 <titles>）
# -------------------- MODIFY: REFINE_REWRITE_PROMPT --------------------
REFINE_REWRITE_PROMPT = """
你是教材排版与结构一致性专家。给你若干页由 OCR/VL 模型输出的 Markdown 文本（每页用 <page i> ... </page i> 包裹，i 为页码）。
请对这些页面进行“重整”：严格执行以下规则，并逐页输出，**必须保留并按原顺序输出每个页边界标记**。

【必须遵守的规则】
1) 仅保留“标题 + 正文内容”两部分。删除任何非正文内容：页眉、页脚、提示语、说明性文字、注释、无关噪声等；清除诸如“NOT MAIN BODY CONTENT”“Content continues here but is truncated...” 等无意义文本。
2) 目录页（或前言/版权/致谢等非正文页）：正文应为空（如有内容也应清除），若某一页几乎全是标题行，即将其认定为目录页，直接划为空白。
3) 标题对齐：**在整个文档范围内保持标题层级一致（同类标题使用相同 # 数量）**。已知历史标题（可能为空）如下：
<prior_titles>
{prior_titles}
</prior_titles>

你**必须**在最终的 <titles> 中覆盖上述 prior_titles 中所有唯一、有意义的条目（若层级有误应予以纠正）。
同时做如下事：
    1. 若先前的标题中有重复或无意义内容，请剔除并重排，保持一致性；
    2. 若本批页面出现先前没有的新标题，请补充到 <titles>；
    3. 层级规则建议：中文“章”用一级标题；中文“节”用二级标题；“一、二、三、...”用三级标题；“1.”用五级；“1.1/1.1.1”用四/五级；“例”使用****包裹即可（若你判断更合理，可统一修正，但需全局一致）；
    4. 去掉前言、目录等无关内容。
    5. 章节一般不会单独成章，如遇"# 第一章/n ## 函数与极限"这些结构，应改为"# 第一章 函数与极限"，并以此调整后续章节标题等级

4) 数学：行内 $...$；块级以独占行 $$...$$；不要使用代码围栏。
5) 输出结构约束（逐页）：每页输出必须为如下结构；若正文应为空仍须给出空的 <page i>：
   <page i>
   标题（若有）与正文（若有）。**在 <page i> 内禁止出现 <content> 标签**；直接输出纯 Markdown。
   </page i>

6) 返回最后一个**全局标题清单**，格式：
<titles>
# 一级标题
## 二级标题
### 三级标题
...
</titles>
（注意：此处**不包含**任何 <page i>，不可重复，不应保留无用标题）

严格遵守以上约束：逐页输出并保留 <page i> ... </page i>，最后附上仅一次的 <titles> ... </titles>，从中总结所有上述的标题，保持规整与对齐。不要添加任何其它注释/解释。
""".strip()



_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$', re.M)
_CONTENT_RE = re.compile(r"<content>(.*?)</content>", re.S | re.I)
_PAGE_SPLIT_RE = re.compile(r"<page\s+(\d+)>(.*?)</page\s+\1>", re.S | re.I)
_TITLES_BLOCK_RE = re.compile(r"<titles>(.*?)</titles>", re.S | re.I)

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

    # Refine（异步并行窗口）
    refine_enabled: bool = True
    refine_model: str = "qwen-plus"
    refine_window_size: int = 50       # 窗口大小 W
    refine_keep_tail: int = 59         # 仅采纳“后 K 页”
    refine_step: int = 50              # 步长 S（滑动窗口）
    refine_concurrency: int = 6        # 重整并发上限

    def build_prompt(self) -> str:
        if not self.extra_instructions:
            return self.prompt
        return "\n".join([self.prompt, *self.extra_instructions])

# -------------------- CONVERTER --------------------

class PDFToMarkdownConverter:
    """Convert PDF pages into Markdown text using Qwen 3 VL, then async-window-refine with a secondary model."""

    def __init__(self, *, client: OpenAI | None = None, config: ConversionConfig | None = None,
                 refine_client: OpenAI | None = None) -> None:
        # Qwen（OpenAI 兼容）
        self.client = client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # Refine 模型（如用 DeepSeek 专用端点，在此替换 base_url/api_key）
        self.refine_client = refine_client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.config = config or ConversionConfig()

        # 全局标题（不做“手动去重/规范化”）
        self._titles_lock = asyncio.Lock()
        self._titles_global: List[str] = []  # 如 "# 章名", "## 节名", ...

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

        # 2&3) 流水线：并发 image2md；滑动窗口（W=5, S=3）就绪即 refine
        qwen_md_per_page, refined_md_per_page = await self._image2md_then_stream_refine(
            images,
            limit=self.config.image2md_concurrency,
            show_progress=self.config.show_progress,
            pdf_name=pdf_path.name,
        )

        # 5) 落盘
        if output_path is not None:
            titles_block = await self._build_titles_block()
            pages_block = self._build_pages_with_markers(refined_md_per_page)
            final_text = (titles_block + "\n\n" + pages_block).strip()
            await asyncio.to_thread(output_path.write_text, final_text, encoding="utf-8")

        contents_only = [self._extract_content(md) for md in refined_md_per_page]
        return contents_only

    # ---------------- Qwen 阶段：并发识别 ----------------

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

    # ---------------- Refine 阶段：滑动窗口（W=5, S=3）流水线 ----------------
    async def _image2md_then_stream_refine(
        self,
        images: List[BytesIO],
        *,
        limit: int,
        show_progress: bool,
        pdf_name: str,
    ) -> Tuple[List[str], List[str]]:
        """
        - 并发 image2md（受 limit 控制）。
        - 预先生成滑动窗口 windows（W=self.config.refine_window_size, S=self.config.refine_step）。
        - ★ 顺序化 refine：严格按 windows 顺序，一个窗口 refine 完并把 <titles> 拼接到全局后，才允许下一个窗口。
        - 每个窗口只回填“后 K 页”（K=self.config.refine_keep_tail）。
        返回：(qwen_md_per_page, final_pages)
        """
        N = len(images)
        if N == 0:
            return [], []
        W = max(1, self.config.refine_window_size)
        S = max(1, self.config.refine_step)
        K = max(1, min(self.config.refine_keep_tail, W))  # K 不超过 W

        # 1) 预计算滑动窗口（与原来一致：W=5, S=3）
        windows: List[Tuple[int, int]] = []
        if N >= W:
            pos = W
            while pos <= N:
                windows.append((pos - W + 1, pos))
                pos += S
            if windows and windows[-1][1] != N:
                windows.append((max(1, N - W + 1), N))
        else:
            windows = [(1, N)]

        # 进度条
        pbar_conv = tqdm(total=N, desc=f"Converting {pdf_name}", unit="page") if show_progress else None
        pbar_refine = tqdm(total=len(windows), desc=f"Refining (W={W}, S={S})", unit="window") if show_progress else None

        # 2) 并发 image2md
        qwen_md_per_page: List[str] = [""] * N
        refined_md_per_page: List[str] = [""] * N
        results: Dict[int, str] = {}  # idx(1-based) -> qwen md

        sem = asyncio.Semaphore(max(1, limit))

        async def page_worker(i: int, img: BytesIO) -> str:
            async with sem:
                md_full = await asyncio.to_thread(self._image_to_markdown, img, i)
            results[i] = md_full
            qwen_md_per_page[i - 1] = md_full
            if pbar_conv:
                pbar_conv.update(1)
            return md_full

        page_tasks: Dict[int, asyncio.Task] = {
            i: asyncio.create_task(page_worker(i, img))
            for i, img in enumerate(images, start=1)
        }

        # 3) 定义“单窗 refine”（同步等待，保证 titles 先写回再下一个窗口）
        async def run_refine_window(start: int, end: int) -> None:
            # 等待该窗口所需页全部完成 image2md
            await asyncio.gather(*(page_tasks[i] for i in range(start, end + 1)))
            md_slice = [results[i] for i in range(start, end + 1)]

            # 快照 prior_titles（此刻已包含之前所有窗口写入的 titles）
            async with self._titles_lock:
                prior_titles = "\n".join(self._titles_global)

            prompt = REFINE_REWRITE_PROMPT.format(prior_titles=prior_titles)
            text = await asyncio.to_thread(self._call_refine_model, prompt, self._pack_pages_with_markers(md_slice, list(range(start, end + 1))))

            # 解析并仅回填“后 K 页”
            per_page_map = self._split_pages_by_marker(text)
            titles_list = self._parse_titles_block_flat(text)

            keep_start = max(start, end - K + 1)
            for p in range(keep_start, end + 1):
                block = per_page_map.get(p, "").strip()
                if block:
                    refined_md_per_page[p - 1] = block

            # ★ 把本窗 titles 拼接进全局 —— 下一个窗口才允许开始
            if titles_list:
                async with self._titles_lock:
                    self._titles_global = titles_list

            if pbar_refine:
                pbar_refine.update(1)

        # 4) 串行跑所有窗口：严格保证“titles 先合并再下窗”
        try:
            for (s, e) in windows:
                await run_refine_window(s, e)
        finally:
            if pbar_conv:
                pbar_conv.close()
            if pbar_refine:
                pbar_refine.close()

        # 5) 合并结果：优先 refined，否则回退 qwen
        final_pages = [refined_md_per_page[i] or qwen_md_per_page[i] for i in range(N)]
        return qwen_md_per_page, final_pages



    # ---------------- 调用 + 解析 ----------------

    def _pack_pages_with_markers(self, pages_md: List[str], pages_nums: List[int]) -> str:
        """把若干页打包，并加上严格闭合的 <page i> … </page i> 边界（输入 refine）。"""
        blocks = []
        for pno, md in zip(pages_nums, pages_md):
            md = md.strip()
            # 确保页内不含旧的页锚，避免嵌套
            md = _PAGE_SPLIT_RE.sub(lambda _: "", md)
            blocks.append(f"<page {pno}>\n{md}\n</page {pno}>")
        return "\n\n".join(blocks)

    def _build_pages_with_markers(self, refined_md_per_page: List[str]) -> str:
        """落盘：将每页内容用 <page i> … </page i> 包裹并串联。"""
        blocks = []
        for i, md in enumerate(refined_md_per_page, start=1):
            body = md.strip()
            # 若 refine 已返回带页锚，只取内部内容避免双包裹
            m = _PAGE_SPLIT_RE.search(body)
            if m and int(m.group(1)) == i:
                body = m.group(2).strip()
            blocks.append(f"<page {i}>\n{body}\n</page {i}>")
        return "\n\n".join(blocks)

    def _call_refine_model(self, prompt: str, chunk_text: str) -> str:
        """同步调用 refine 模型（OpenAI 兼容接口）。"""
        # print("="*100)
        # print([{
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": prompt},
        #             {"type": "text", "text": chunk_text},
        #         ],
        #     }])
        resp = self.refine_client.chat.completions.create(
            model=self.config.refine_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": chunk_text},
                ],
            }],
        )
        # print("="*100)
        # print(resp.choices[0].message.content)
        return resp.choices[0].message.content or ""

    def _split_pages_by_marker(self, text: str) -> Dict[int, str]:
        """从文本中按严格页锚提取每页块（含“标题 + <content>正文或空内容”）。"""
        out: Dict[int, str] = {}
        for m in _PAGE_SPLIT_RE.finditer(text):
            pno = int(m.group(1))
            body = m.group(2).strip()
            out[pno] = f"<page {pno}>\n{body}\n</page {pno}>"
        return out

    def _parse_titles_block_flat(self, text: str) -> List[str]:
        """解析 <titles>…</titles> 为全局标题列表（不分页，不做去重/规范化）。"""
        tb = _TITLES_BLOCK_RE.search(text)
        if not tb:
            return []
        inner = tb.group(1)
        return [f"{m.group(1)} {m.group(2).strip()}" for m in _HEADING_RE.finditer(inner)]

    # ---------------- Qwen 单页识别 ----------------

    def _image_to_markdown(self, image_stream: BytesIO, page_number: int) -> str:
        encoded_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
        mime = f"image/{self.config.image_format.lower()}"

        base = self.config.build_prompt()
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

    @staticmethod
    def _extract_content(md: str) -> str:
        """
        优先抽取 <content>…</content>；若缺失，则：
        - 去掉页锚，剔除以 '#' 开头的标题行，余下即正文。
        """
        blocks = [b.strip() for b in _CONTENT_RE.findall(md)]
        if blocks:
            return "\n\n".join(b for b in blocks if b)

        inner = md
        m = _PAGE_SPLIT_RE.search(inner)
        if m:
            inner = m.group(2)
        lines = inner.splitlines()
        body_lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
        body = "\n".join(body_lines).strip()
        return body

    # ---------- 构建 <titles> 与分页面输出 ----------

    async def _build_titles_block(self) -> str:
        """
        输出：
        <titles>
        # ...
        ## ...
        ...
        </titles>
        """
        async with self._titles_lock:
            titles = list(self._titles_global)

        parts = ["<titles>"]
        parts.extend(titles)
        parts.append("</titles>")
        return "\n".join(parts).strip()


__all__ = ["PDFToMarkdownConverter", "ConversionConfig"]
