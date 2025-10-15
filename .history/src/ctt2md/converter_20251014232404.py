"""Utilities for converting PDF textbooks into Markdown using Qwen 3 VL."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence
from tqdm.auto import tqdm

from openai import OpenAI
from pdf2image import convert_from_path

PROMPT_TEMPLATE = (
    "You are an OCR expert who rewrites textbook pages as Markdown. "
    "Recognise mathematics precisely and emit inline math between $ and block math "
    "between $$ fences using LaTeX. Preserve headings, numbered equations, "
    "and tables when possible. Return pure GitHub-flavoured Markdown only."
)


@dataclass(slots=True)
class ConversionConfig:
    """Configuration for PDF to Markdown conversion."""

    model: str = "qwen-3-vl"
    dpi: int = 300
    max_output_tokens: int | None = None
    prompt: str = PROMPT_TEMPLATE
    image_format: str = "PNG"
    extra_instructions: Sequence[str] = field(default_factory=tuple)
    show_progress: bool = True

    def build_prompt(self) -> str:
        """Combine the base prompt with any extra instructions."""
        if not self.extra_instructions:
            return self.prompt
        return "\n".join([self.prompt, *self.extra_instructions])


class PDFToMarkdownConverter:
    """Convert PDF pages into Markdown text using the OpenAI Responses API."""

    def __init__(self, *, client: OpenAI | None = None, config: ConversionConfig | None = None) -> None:
        self.client = client or OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.config = config or ConversionConfig()

    def convert(self, pdf_path: str | Path, *, output_path: str | Path | None = None) -> List[str]:
        """Convert the given PDF into Markdown and optionally persist to disk."""

        pdf_path = Path(pdf_path)
        if output_path is not None:
            output_path = Path(output_path)

        images = self._pdf_to_images(pdf_path)
        iterable = tqdm(images, desc=f"Converting {pdf_path.name}", unit="page") if self.config.show_progress else images

        markdown_pages: List[str] = []
        for index, image in enumerate(iterable, start=1):
            markdown_pages.append(self._image_to_markdown(image, index))

        if output_path is not None:
            output_path.write_text("\n\n".join(markdown_pages), encoding="utf-8")

        return markdown_pages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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

    def _image_to_markdown(self, image_stream: BytesIO, page_number: int) -> str:
        """Send an image to the Qwen model and return Markdown output."""

        encoded_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
        prompt = self.config.build_prompt() + f"\nThe content comes from page {page_number}."

        response = self.client.responses.create(
            model=self.config.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image": {
                                "data": encoded_image,
                                "mime_type": f"image/{self.config.image_format.lower()}",
                            },
                        },
                    ],
                }
            ],
            max_output_tokens=self.config.max_output_tokens,
        )

        if hasattr(response, "output_text"):
            return response.output_text.strip()

        # Fallback for older response objects where content must be unpacked manually
        return "\n".join(
            item.get("text", "")
            for choice in getattr(response, "output", [])
            for item in getattr(choice, "content", [])
            if item.get("type") == "output_text"
        ).strip()

    # ------------------------------------------------------------------
    # Batch utilities
    # ------------------------------------------------------------------
    def convert_many(self, pdf_paths: Iterable[str | Path], *, output_dir: str | Path) -> None:
        """Convert multiple PDFs, mirroring their names in the output directory."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for pdf in pdf_paths:
            pdf_path = Path(pdf)
            target = output_dir / (pdf_path.stem + ".md")
            self.convert(pdf_path, output_path=target)


__all__ = ["PDFToMarkdownConverter", "ConversionConfig"]
