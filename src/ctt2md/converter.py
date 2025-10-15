"""Utilities for converting PDF textbooks into Markdown using Qwen 3 VL."""

from __future__ import annotations

import os
import re
import base64
import asyncio
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

    # 其他
    "Ignore page numbers and running headers/footers."
    "Use the main language in the picture to reply!!"
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
            titles_block = "\n".join(self.titles).strip()
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
    @staticmethod
    def _extract_titles(md: str) -> List[str]:
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
        return titles

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
        """
        async with self._titles_lock:
            self._titles_by_page[page_number] = page_titles
            lines: List[str] = []
            for p in sorted(self._titles_by_page.keys()):
                lines.extend(self._titles_by_page[p])
            self.titles = lines  # 使用时通过 "\n".join(self.titles)

    def _image_to_markdown(self, image_stream: BytesIO, page_number: int, prior_titles_text: str) -> str:
        """Send an image to the Qwen model and return Markdown output."""
        encoded_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
        mime = f"image/{self.config.image_format.lower()}"

        # Prompt 组合
        base = self.config.build_prompt()
        prior_block = (
            f"\nPreviously seen titles (keep levels consistent and avoid duplicates):\n{prior_titles_text}\n"
            if prior_titles_text else ""
        )
        page_hint = (
            f"\nThis image corresponds to textbook page {page_number}. "
            f"Do not output page numbers anywhere.\n"
        )
        prompt = f"{base}{prior_block}{page_hint}"

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


__all__ = ["PDFToMarkdownConverter", "ConversionConfig"]
