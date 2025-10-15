"""Utilities for converting PDF textbooks into Markdown using Qwen 3 VL."""

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
)


_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$', re.M)  # 匹配 #/##/... 标题行
_CONTENT_RE = re.compile(r"<\|content\|>(.*?)</\|content\|>", re.S | re.I)  # 提取 content 块（允许多段）


@dataclass(slots=True)
class ConversionConfig:
    """Configuration for PDF to Markdown conversion."""

    model: str = "qwen3-vl-plus"
    dpi: int = 100
    prompt: str = PROMPT_TEMPLATE
    image_format: str = "PNG"
    extra_instructions: Sequence[str] = field(default_factory=tuple)
    show_progress: bool = True
    image2md_concurrency: int = 20  # 并发上限

    dedup_titles: bool = True
    # "level_text": 同层级同标题才判为重复（更保守，默认）
    # "text": 忽略层级，只要标题文本相同就认为重复（更激进）
    dedup_strategy: str = "level_text"

    def build_prompt(self) -> str:
        if not self.extra_instructions:
            return self.prompt
        return "\n".join([self.prompt, *self.extra_instructions])


class PDFToMarkdownConverter:
    """Convert PDF pages into Markdown text using the OpenAI-compatible API."""

    titles: List[str]  # 全量标题容器：按页序拼接，用 "\n" 间隔

    def __init__(self, *, client: OpenAI | None = None, config: ConversionConfig | None = None) -> None:
        self.client = client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.config = config or ConversionConfig()

        # 并发下的标题管理
        self._titles_lock = asyncio.Lock()
        self._titles_by_page: Dict[int, List[str]] = {}
        self.titles = []  # 以 "\n" 间隔拼接时使用："\n".join(self.titles)
        self._seen_title_keys: set[str] = set()

    # -----------------------------
    # 对外：同步入口（兼容你现有调用方式）
    # -----------------------------
    def convert(self, pdf_path: str | Path, *, output_path: str | Path | None = None) -> None:
        """Convert the given PDF into Markdown and optionally persist to disk."""
        asyncio.run(self._convert_async(pdf_path, output_path=output_path))
        return None

    # -----------------------------
    # 对外：批量（仍然同步入口）
    # -----------------------------
    def convert_many(self, pdf_paths: Iterable[str | Path], *, output_dir: str | Path) -> None:
        """Convert multiple PDFs, mirroring their names in the output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for pdf in pdf_paths:
            pdf_path = Path(pdf)
            target = output_dir / (pdf_path.stem + ".md")
            self.convert(pdf_path, output_path=target)

    # =============================
    # 内部：异步实现
    # =============================
    async def _convert_async(self, pdf_path: str | Path, *, output_path: str | Path | None = None) -> List[str]:
        pdf_path = Path(pdf_path)
        if output_path is not None:
            output_path = Path(output_path)

        # 1) PDF -> Images（线程池异步化）
        images = await asyncio.to_thread(self._pdf_to_images, pdf_path)

        # 2) Images -> Markdown（并发、限流、保序）
        #    注意：results 里只放 <|content|> 的正文，不含标题
        contents_only = await self._images_to_markdown_bounded(
            images,
            limit=self.config.image2md_concurrency,
            show_progress=self.config.show_progress,
            pdf_name=pdf_path.name,
        )

        # 3) 持久化（线程池异步化）
        if output_path is not None:
            # 先输出全量标题，再输出正文
            titles_block = "<|title_start|>\n" + "\n".join(self.titles).strip() + "\n<|title_end|>"
            content_block = "\n\n".join(c.strip() for c in contents_only if c and c.strip())
            final_text = (titles_block + "\n\n" + content_block).strip() if titles_block else content_block
            await asyncio.to_thread(output_path.write_text, final_text, encoding="utf-8")

        return contents_only

    async def _images_to_markdown_bounded(
        self,
        images: List[BytesIO],
        *,
        limit: int,
        show_progress: bool,
        pdf_name: str,
    ) -> List[str]:
        """
        并发把每页 image 转成 markdown，但保证最终列表按页码排序不变。
        通过 asyncio.Semaphore(limit) 限制同时处理的数量上限。
        """
        sem = asyncio.Semaphore(max(1, limit))

        async def worker(img: BytesIO, idx: int) -> Tuple[int, str]:
            # 取“到当前为止、已知的前序页标题”
            prior_titles_text = await self._get_prior_titles_text(idx)

            # 调用模型
            md_full = await asyncio.to_thread(self._image_to_markdown, img, idx, prior_titles_text)

            # 解析标题并写回（影响全局 titles）
            page_titles = self._extract_titles(md_full)
            await self._update_titles(idx, page_titles)

            # 只抽取 <|content|> 的正文作为结果
            page_content = self._extract_content(md_full)
            return idx, page_content

        tasks = [asyncio.create_task(worker(img, i)) for i, img in enumerate(images, start=1)]

        results: List[Optional[str]] = [None] * len(images)
        pbar = tqdm(total=len(images), desc=f"Converting {pdf_name}", unit="page") if show_progress else None

        try:
            for coro in asyncio.as_completed(tasks):
                idx, content_only = await coro
                results[idx - 1] = content_only
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        # 收敛为非 None 列表（保序）
        return [r if r is not None else "" for r in results]

    # =============================
    # 内部：同步底层实现（保持你的原始逻辑）
    # =============================
    def _pdf_to_images(self, pdf_path: Path) -> List[BytesIO]:
        """Render PDF pages into PNG streams."""
        pil_images = convert_from_path(str(pdf_path), dpi=self.config.dpi, fmt=self.config.image_format.lower())
        streams: List[BytesIO] = []
        for page in pil_images:
            buffer = BytesIO()
            page.save(buffer, format=self.config.image_format)
            buffer.seek(0)
            streams.append(buffer)
        return streams

    # ---- 标题/内容抽取与维护 ----
    def _extract_titles(self, md: str) -> List[str]:
        """
        从 Markdown 文本中抽取 '#/##/...' 标题行，保持出现顺序。
        只返回完整标题行（含 # 前缀），用于直接拼接。
        """
        titles: List[str] = []
        for m in _HEADING_RE.finditer(md):
            level = m.group(1)
            text = m.group(2).strip()
            # 过滤目录点线和页码
            # 末尾页码样式如 "...... 12" 或 " ... 34" 或 "(12)"
            if text.replace('.', '').strip().isdigit():
                continue
            if re.search(r'(?:\.{3,}|·{2,})\s*\d+\s*$', text):
                continue
            if re.search(r'\(\s*\d+\s*\)\s*$', text):
                continue
            titles.append(f"{level} {text}")
        deduped = self._dedup_titles_list(titles)
        return deduped

    @staticmethod
    def _extract_content(md: str) -> str:
        """
        抽取所有 <|content|>...</|content|> 段落，按出现顺序拼接。
        如果不存在，返回空字符串。
        """
        blocks = [b.strip() for b in _CONTENT_RE.findall(md)]
        return "\n\n".join(b for b in blocks if b)

    async def _get_prior_titles_text(self, page_number: int) -> str:
        """
        获取“已完成的 < page_number 的所有页”的标题，按页序拼接，用 '\\n' 间隔。
        并发下使用锁保证一致视图，但不会阻塞等待尚未完成的前序页。
        """
        async with self._titles_lock:
            ordered_pages = sorted(k for k in self._titles_by_page.keys() if k < page_number)
            lines: List[str] = []
            for p in ordered_pages:
                lines.extend(self._titles_by_page[p])
            return "\n".join(lines)

    async def _update_titles(self, page_number: int, page_titles: List[str]) -> None:
        """
        写回本页标题，并重建 self.titles（全量标题，按页序、用 '\\n' 间隔）。
        新增：全局去重（只保留首见的标题），配置可控。
        """
        async with self._titles_lock:
            # 1) 页内去重已在 _extract_titles 做过；这里进一步做 **全局去重**
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

            # 2) 重建全量 titles（保持页序）
            lines: List[str] = []
            for p in sorted(self._titles_by_page.keys()):
                lines.extend(self._titles_by_page[p])
            self.titles = lines  # 使用时通过 "\n".join(self.titles)


    def _image_to_markdown(self, image_stream: BytesIO, page_number: int, prior_titles_text: str) -> str:
        encoded_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
        mime = f"image/{self.config.image_format.lower()}"

        base = self.config.build_prompt()
        prior_block = (
            f"\nPreviously seen titles (keep levels consistent and avoid duplicates):\n{prior_titles_text}\n"
            if prior_titles_text else ""
        )

        schema_hint = "“章”使用二级标题，“节”使用二级标题，中文数字带顿号如“一、”使用三级标题，阿拉伯数字带点如“1.”使用四级标题，“1.1”等使用五级标题！！！"

        page_hint = (
            f"\nThis image corresponds to textbook page {page_number}. "
            f"Do not output page numbers anywhere.\n"
        )

        prompt = f"{base}\n{schema_hint}{prior_block}{page_hint}"

        data_url = f"data:{mime};base64,{encoded_image}"

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "QwenVL Markdown"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            extra_body={
                'enable_thinking': True,
                "thinking_budget": 500
            },
        )

        return response.choices[0].message.content
    
        # NEW ----------
    def _normalize_title_text(self, text: str) -> str:
        """
        规范化标题文本用于去重：
        - NFKC 统一宽/半角与兼容字符
        - 去掉首尾空白、压缩多空白
        - 去除末尾常见分隔/标点（.,，。:：；;、·…—- 等）
        - 再小写（英文有效，中文不受影响）
        - 额外兜底去掉可能残留的目录点线“……”“···”之类
        """
        t = unicodedata.normalize("NFKC", text)
        t = re.sub(r"\s+", " ", t).strip()

        # 去掉目录点线与可能残留的页码（兜底；主过滤在 _extract_titles 已做）
        t = re.sub(r"(?:\.{3,}|·{2,})\s*\d+\s*$", "", t).strip()
        t = re.sub(r"\(\s*\d+\s*\)\s*$", "", t).strip()

        # 去掉结尾的标点/连字符/省略号等
        t = re.sub(r"[\s\.\u3002、，,：:；;—\-·…]+$", "", t).strip()

        # 统一英文大小写（中文不受影响）
        t = t.lower()
        return t

    # NEW ----------
    def _title_key(self, level: str, text: str) -> str:
        """
        根据配置生成标题去重 key。
        - level 形如 '#', '##', ...
        - text 为原始文本（本函数内部会做规范化）
        """
        norm = self._normalize_title_text(text)
        if self.config.dedup_strategy == "text":
            return norm
        # 默认：包含层级信息，更保守，避免 "绪论" 在不同层级被误杀
        return f"{len(level)}|{norm}"

    # NEW ----------
    def _dedup_titles_list(self, titles: List[str]) -> List[str]:
        """
        对“单页标题列表”先做**页内去重**：
        输入 titles 形如 ["# A", "## B", "## B", "# A"]
        按配置算 key，保留首见，保持顺序。
        """
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



__all__ = ["PDFToMarkdownConverter", "ConversionConfig"]
