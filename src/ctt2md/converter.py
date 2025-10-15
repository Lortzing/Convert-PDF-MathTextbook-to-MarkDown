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
    "You are an OCR expert who rewrites textbook pages as Markdown. "
    "Recognise mathematics precisely and emit inline math between $ and block math "
    "between $$ fences using LaTeX. Preserve headings, numbered equations, "
    "and tables when possible. Return pure GitHub-flavoured Markdown only. "
    "Don't include fences like ``` or ```markdown; reply plain text in Markdown. "
    "Ignore page numbers and running headers/footers. "
    "Wrap the main extracted content with <|content|> ... <|content|>. "
    "Only output titles/subtitles that truly belong to the content (no dots leaders or page references)."
)

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$', re.M)  # 匹配 #/##/... 标题行


@dataclass(slots=True)
class ConversionConfig:
    """Configuration for PDF to Markdown conversion."""

    model: str = "qwen3-vl-plus"
    dpi: int = 300
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

    # 全量标题容器：按页序拼接，用 "\n" 间隔
    titles: List[str]

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
        markdown_pages = await self._images_to_markdown_bounded(
            images,
            limit=self.config.image2md_concurrency,
            show_progress=self.config.show_progress,
            pdf_name=pdf_path.name,
        )

        # 3) 持久化（线程池异步化）
        if output_path is not None:
            text = "\n\n".join(markdown_pages)
            await asyncio.to_thread(output_path.write_text, text, encoding="utf-8")

        return markdown_pages

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

            # 把同步网络调用放入线程池，避免阻塞事件循环
            md = await asyncio.to_thread(self._image_to_markdown, img, idx, prior_titles_text)

            # 解析并写回本页标题；然后重建全量 titles（按页序）
            page_titles = self._extract_titles(md)
            await self._update_titles(idx, page_titles)

            return idx, md

        tasks = [asyncio.create_task(worker(img, i)) for i, img in enumerate(images, start=1)]

        results: List[Optional[str]] = [None] * len(images)
        pbar = tqdm(total=len(images), desc=f"Converting {pdf_name}", unit="page") if show_progress else None

        try:
            for coro in asyncio.as_completed(tasks):
                idx, md = await coro
                results[idx - 1] = md
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

    # ---- 标题抽取与维护 ----
    @staticmethod
    def _extract_titles(md: str) -> List[str]:
        """
        从 Markdown 文本中抽取 '#/##/...' 标题行，保持出现顺序。
        只返回完整标题行（含 # 前缀），用于直接拼接。
        """
        titles: List[str] = []
        for m in _HEADING_RE.finditer(md):
            # 还原形如 "## Title"
            level = m.group(1)
            text = m.group(2).strip()
            # 过滤可能的点线/页码引导（常见目录点线）——尽量保守
            if text.replace('.', '').strip().isdigit():
                # 纯数字（疑似页码）跳过
                continue
            if '........' in text or '···' in text:
                # 明显目录引导点线跳过
                continue
            titles.append(f"{level} {text}")
        return titles

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
            # 重新拼接全量标题，保证顺序性
            lines: List[str] = []
            for p in sorted(self._titles_by_page.keys()):
                lines.extend(self._titles_by_page[p])
            self.titles = lines  # 保持为逐行列表；使用时通过 "\n".join(self.titles)

    def _image_to_markdown(self, image_stream: BytesIO, page_number: int, prior_titles_text: str) -> str:
        """Send an image to the Qwen model and return Markdown output."""
        encoded_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
        mime = f"image/{self.config.image_format.lower()}"

        # —— Prompt 组合（完善版）——
        # 1) 基础指令（数学/Markdown/不输出围栏/加 content 标签）
        # 2) 提供“已识别的历史标题”，要求保持层级一致
        # 3) 说明当前页码（仅作上下文，不要求输出页码）
        base = self.config.build_prompt()
        prior_block = f"\nPreviously seen titles (keep levels consistent and avoid duplicates):\n{prior_titles_text}\n" if prior_titles_text else ""
        page_hint = f"\nThis image corresponds to textbook page {page_number}. Do not output page numbers.\n"

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
        )

        # 返回 Markdown
        return response.choices[0].message.content


__all__ = ["PDFToMarkdownConverter", "ConversionConfig"]
