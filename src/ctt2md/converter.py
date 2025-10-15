"""Utilities for converting PDF textbooks into Markdown using Qwen 3 VL (+ Two‑Phase Async Refine).

Phase A (Title Inference):
  - From image2md outputs, infer ALL headings and a consistent heading policy.
  - Merge titles sequentially across sliding windows (W,S). Each window waits for previous
    windows to commit their <titles> and <heading_policy> before proceeding.

Phase B (Structure Refine):
  - Using the accumulated titles + inferred policy, rewrite pages (remove non‑body, enforce
    consistent heading levels, etc.). Windows run sequentially; only the last K pages in each
    window are committed back (streaming tail adoption) to reduce churn.

Notes
  - Both phases use the same refine model (e.g., "qwen-plus") via an OpenAI‑compatible endpoint.
  - Maintains original external API: PDFToMarkdownConverter.convert(...)
  - Adds config.refine_titles_scope to choose whether Phase B uses window‑local or global titles.
  - Keeps strict <page i> ... </page i> markers and a final <titles> ... </titles> block.
"""
from __future__ import annotations

import os
import re
import base64
import asyncio
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Literal

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
    "<content> ... </content>\n"

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

# -------------------- NEW: Phase A (Titles & Policy) Prompt --------------------
TITLE_INFER_PROMPT = (
    "你将收到若干页由 OCR/VL 模型输出的 Markdown 文本（每页用 <page i> ... </page i> 包裹，i 为页码）。\n"
    "你的任务是：从这些页面中**仅识别标题**，并基于全量标题**归纳一个统一的标题层级规则**。\n\n"
    "【必须输出的结构】\n"
    "<titles>\n"
    "# 一级标题\n"
    "## 二级标题\n"
    "### 三级标题\n"
    "...（只包含真正的章节/小节/条目标题；去掉页码、点线、无意义或重复项；保持规范）\n"
    "</titles>\n\n"
    "<heading_policy>\n"
    "- 一级标题：用于“章/Chapter/Chapter N/第一章 …”等\n"
    "- 二级标题：用于“节/Section/第1节 …”等\n"
    "- 三级标题：用于“一、二、三、…/Subsection”等\n"
    "- 四级标题：用于“1., 2., 3.” 等\n"
    "- 五级标题：用于“1.1, 1.2, …”，六级标题：用于“1.1.1, …”等\n"
    "（若你的判断需要做出更合理的一致性修正，请在此处给出简要而明确的规则描述）\n"
    "</heading_policy>\n\n"
    "上述<heading_policy>中的内容为示例描述，你应当仅输出其中有意义的部分、在书中体现的部分，并进行修改，使其真正反映书本的标题结构与内容栏目，而非照搬\n\n"
    "严格要求：\n"
    "1) 只输出上面两个区块（<titles> 与 <heading_policy>），**不要**输出其它内容；\n"
    "2) 严格去除目录/前言/版权等非正文标题；清除点线和页码；\n"
    "3) 依据既有的历史标题，保持对齐和去重。历史标题如下：\n"
    "<prior_titles>\n{prior_titles}\n</prior_titles>\n\n"
    "若历史标题为空，也要从本批页面推断出一致的层级规则。"
).strip()

# -------------------- Phase B (Rewrite with Titles & Policy) Prompt --------------------
REFINE_REWRITE_PROMPT = (
    "你是教材排版与结构一致性专家。给你若干页由 OCR/VL 模型输出的 Markdown 文本（每页用 <page i> ... </page i> 包裹，i 为页码）。\n"
    "请对这些页面进行“重整”：严格执行以下规则，并逐页输出，必须保留并按原顺序输出每个页边界标记。\n\n"
    "【必须遵守的规则】\n"
    "1) 仅保留“标题 + 正文内容”两部分。删除任何非正文内容：页眉、页脚、提示语、说明性文字、注释、无关噪声等；\n"
    "2) 目录页（或前言/版权/致谢等非正文页）：正文应为空（如有内容也应清除）；若一页中具有大量标题，且基本只具有标题结构，则应视为目录页，应当清除其内容，正文记为空\n"
    "3) 标题对齐：在整个文档范围内保持标题层级一致（同类标题使用相同 # 数量）。既定规则：\n"
    "<heading_policy>\n{heading_policy}\n</heading_policy>\n"
    "已知全局/窗口标题如下（据此对齐层级并去重）：\n"
    "<titles>\n{titles}\n</titles>\n\n"
    "一定仔细检查每一个标题是否符合相应规则，是否前后文互相对应，是否与<titles>对应!!!\n"
    "4) 数学：行内 $...$；块级以独占行 $$...$$；不要使用代码围栏；\n"
    "5) 输出结构约束（逐页）：每页输出必须为如下结构；若正文应为空仍须给出空的 <page i>：\n"
    "   <page i>\n"
    "   标题（若有）与正文（若有）。**在 <page i> 内禁止出现 <content> 标签**；只输出纯 Markdown。\n"
    "   </page i>\n\n"
    "6) 返回最后一个**全局标题清单**，格式：\n"
    "<titles>\n# 一级标题\n## 二级标题\n### 三级标题\n...</titles>\n"
    "（注意：此处不包含任何 <page i>，不可重复，不应保留无用标题）\n\n"
    "严格遵守以上约束：逐页输出并保留 <page i> ... </page i>，最后附上仅一次的 <titles> ... </titles>，不要添加其它注释/解释。"
).strip()

# -------------------- REGEX --------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
_CONTENT_RE = re.compile(r"<content>(.*?)</content>", re.S | re.I)
_PAGE_SPLIT_RE = re.compile(r"<page\s+(\d+)>\s*(.*?)\s*</page\s+\1>", re.S | re.I)
_TITLES_BLOCK_RE = re.compile(r"<titles>(.*?)</titles>", re.S | re.I)
_POLICY_BLOCK_RE = re.compile(r"<heading_policy>(.*?)</heading_policy>", re.S | re.I)

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
    refine_keep_tail: int = 50         # 仅采纳“后 K 页”
    refine_step: int = 50              # 步长 S（滑动窗口）
    refine_concurrency: int = 6        # 重整并发上限（用于 Phase B 的页内并发占位）

    # NEW: Titles scope for Phase B
    refine_titles_scope: Literal["global", "window"] = "global"

    def build_prompt(self) -> str:
        if not self.extra_instructions:
            return self.prompt
        return "\n".join([self.prompt, *self.extra_instructions])

# -------------------- CONVERTER --------------------
class PDFToMarkdownConverter:
    """Convert PDF pages into Markdown using Qwen 3 VL, then two‑phase async refine with a secondary model."""

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        config: ConversionConfig | None = None,
        refine_client: OpenAI | None = None,
    ) -> None:
        # Qwen（OpenAI 兼容）
        self.client = client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # Refine 模型（两阶段都用同一个）
        self.refine_client = refine_client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.config = config or ConversionConfig()

        # 全局标题 / 策略
        self._titles_lock = asyncio.Lock()
        self._titles_global: List[str] = []
        self._policy_global: str = ""

        # 保存每个窗口的 titles，供 scope="window" 使用
        self._titles_windows: Dict[Tuple[int, int], List[str]] = {}
        self._policy_windows: Dict[Tuple[int, int], str] = {}

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

        # 2) 两阶段流水线：并发 image2md；窗口化 Phase A (titles/policy)；再窗口化 Phase B (rewrite)
        qwen_md_per_page, final_md_per_page = await self._image2md_two_phase_refine(
            images,
            limit=self.config.image2md_concurrency,
            show_progress=self.config.show_progress,
            pdf_name=pdf_path.name,
        )

        # 3) 落盘
        if output_path is not None:
            titles_block = await self._build_titles_block()
            pages_block = self._build_pages_with_markers(final_md_per_page)
            final_text = (titles_block + "\n\n" + pages_block).strip()
            await asyncio.to_thread(output_path.write_text, final_text, encoding="utf-8")

        contents_only = [self._extract_content(md) for md in final_md_per_page]
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

    # ---------------- NEW: Two‑Phase Refinement Pipeline ----------------
    async def _image2md_two_phase_refine(
        self,
        images: List[BytesIO],
        *,
        limit: int,
        show_progress: bool,
        pdf_name: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Pipeline:
        - Concurrent image2md (limit controlled).
        - Precompute sliding windows (W, S). Windows cover [1..N].
        - Phase A: For each window in order, wait for needed pages, infer titles & policy, then
                   merge into global and record window‑local copies. Next window starts only when
                   current window has committed its titles/policy.
        - Phase B: For each window in order, wait for needed pages (already done), and rewrite
                   using either global or window titles/policy (configurable). Only commit the tail K pages.
        - Return (raw image2md pages, final refined pages).
        """
        N = len(images)
        if N == 0:
            return [], []
        W = max(1, self.config.refine_window_size)
        S = max(1, self.config.refine_step)
        K = max(1, min(self.config.refine_keep_tail, W))

        # 1) Build sliding windows
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

        # Progress bars
        pbar_conv = tqdm(total=N, desc=f"Converting {pdf_name}", unit="page") if show_progress else None
        pbar_titles = tqdm(total=len(windows), desc=f"Phase A: Titles/Policy (W={W}, S={S})", unit="window") if show_progress else None
        pbar_refine = tqdm(total=len(windows), desc=f"Phase B: Rewrite (W={W}, S={S})", unit="window") if show_progress else None

        # 2) Concurrent image2md workers (start immediately)
        qwen_md_per_page: List[str] = [""] * N
        refined_md_per_page: List[str] = [""] * N
        results: Dict[int, str] = {}

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
            i: asyncio.create_task(page_worker(i, img)) for i, img in enumerate(images, start=1)
        }

        # 3) Phase A — Titles & Policy per window (sequential)
        async def run_titles_window(start: int, end: int) -> None:
            # Wait pages of this window to finish image2md
            await asyncio.gather(*(page_tasks[i] for i in range(start, end + 1)))
            md_slice = [results[i] for i in range(start, end + 1)]

            # Snapshot prior titles
            async with self._titles_lock:
                prior_titles = "\n".join(self._titles_global)

            prompt = TITLE_INFER_PROMPT.format(prior_titles=prior_titles)
            chunk = self._pack_pages_with_markers(md_slice, list(range(start, end + 1)))
            out = await asyncio.to_thread(self._call_refine_model, prompt, chunk)

            titles_list = self._parse_titles_block_flat(out)
            policy_text = self._parse_policy_block(out)

            # Commit to global (replace with dedup/normalized union)
            if titles_list or policy_text:
                async with self._titles_lock:
                    if titles_list:
                        self._titles_global = self._merge_titles(self._titles_global, titles_list)
                    if policy_text:
                        # Prefer latest non-empty policy; could also merge heuristically
                        self._policy_global = policy_text.strip() or self._policy_global

            # Record window-local
            self._titles_windows[(start, end)] = titles_list
            self._policy_windows[(start, end)] = policy_text

            if pbar_titles:
                pbar_titles.update(1)

        # 4) Phase B — Rewrite per window (sequential; commit tail K pages)
        async def run_refine_window(start: int, end: int) -> None:
            # Pages are already available from Phase A wait, but ensure anyway
            await asyncio.gather(*(page_tasks[i] for i in range(start, end + 1)))
            md_slice = [results[i] for i in range(start, end + 1)]

            # Titles scope selection
            if self.config.refine_titles_scope == "window":
                titles = self._titles_windows.get((start, end), [])
                policy = self._policy_windows.get((start, end), "") or self._policy_global
            else:
                async with self._titles_lock:
                    titles = list(self._titles_global)
                    policy = self._policy_global

            prompt = REFINE_REWRITE_PROMPT.format(
                heading_policy=policy.strip(),
                titles="\n".join(titles).strip(),
            )
            chunk = self._pack_pages_with_markers(md_slice, list(range(start, end + 1)))
            out = await asyncio.to_thread(self._call_refine_model, prompt, chunk)

            per_page_map = self._split_pages_by_marker(out)
            titles_list = self._parse_titles_block_flat(out)

            # Commit tail K
            keep_start = max(start, end - K + 1)
            for p in range(keep_start, end + 1):
                block = per_page_map.get(p, "").strip()
                if block:
                    refined_md_per_page[p - 1] = block

            # Update global titles if refine produced better set
            if titles_list:
                async with self._titles_lock:
                    self._titles_global = self._merge_titles(self._titles_global, titles_list)

            if pbar_refine:
                pbar_refine.update(1)

        try:
            # Phase A — sequential
            for (s, e) in windows:
                await run_titles_window(s, e)
            # Phase B — sequential
            for (s, e) in windows:
                await run_refine_window(s, e)
        finally:
            if pbar_conv:
                pbar_conv.close()
            if pbar_titles:
                pbar_titles.close()
            if pbar_refine:
                pbar_refine.close()

        final_pages = [refined_md_per_page[i] or qwen_md_per_page[i] for i in range(N)]
        return qwen_md_per_page, final_pages

    # ---------------- 调用 + 解析 ----------------
    def _pack_pages_with_markers(self, pages_md: List[str], pages_nums: List[int]) -> str:
        """把若干页打包，并加上严格闭合的 <page i> … </page i> 边界（输入 refine）。"""
        blocks = []
        for pno, md in zip(pages_nums, pages_md):
            md = md.strip()
            md = _PAGE_SPLIT_RE.sub(lambda _: "", md)  # 去除旧页锚避免嵌套
            blocks.append(f"<page {pno}>\n{md}\n</page {pno}>")
        return "\n\n".join(blocks)

    def _build_pages_with_markers(self, refined_md_per_page: List[str]) -> str:
        """落盘：将每页内容用 <page i> … </page i> 包裹并串联。"""
        blocks = []
        for i, md in enumerate(refined_md_per_page, start=1):
            body = md.strip()
            m = _PAGE_SPLIT_RE.search(body)
            if m and int(m.group(1)) == i:
                body = m.group(2).strip()
            blocks.append(f"<page {i}>\n{body}\n</page {i}>")
        return "\n\n".join(blocks)

    def _call_refine_model(self, prompt: str, chunk_text: str) -> str:
        """同步调用 refine 模型（OpenAI 兼容接口）。"""
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
        
        return resp.choices[0].message.content or ""

    def _split_pages_by_marker(self, text: str) -> Dict[int, str]:
        """从文本中按严格页锚提取每页块（含“标题 + 正文或空内容”）。"""
        out: Dict[int, str] = {}
        for m in _PAGE_SPLIT_RE.finditer(text):
            pno = int(m.group(1))
            body = m.group(2).strip()
            out[pno] = f"<page {pno}>\n{body}\n</page {pno}>"
        return out

    def _parse_titles_block_flat(self, text: str) -> List[str]:
        tb = _TITLES_BLOCK_RE.search(text)
        if not tb:
            return []
        inner = tb.group(1)
        return [f"{m.group(1)} {m.group(2).strip()}" for m in _HEADING_RE.finditer(inner)]

    def _parse_policy_block(self, text: str) -> str:
        pb = _POLICY_BLOCK_RE.search(text)
        return (pb.group(1).strip() if pb else "").strip()

    def _merge_titles(self, old: List[str], new: List[str]) -> List[str]:
        """Simple stable merge with de‑duplication and normalization of whitespace."""
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip())
        seen = {norm(t): True for t in old}
        merged = list(old)
        for t in new:
            key = norm(t)
            if key and key not in seen:
                merged.append(t)
                seen[key] = True
        return merged

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
            extra_body={"enable_thinking": True, "thinking_budget": 500},
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
