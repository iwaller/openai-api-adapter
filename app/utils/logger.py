import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import settings

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
}


def setup_logger() -> logging.Logger:
    """
    Setup application logger with file and console handlers.

    Log files are stored in the configured log_dir with rotation.
    """
    logger = logging.getLogger("openai-adapter")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create logs directory
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler with rotation (10MB per file, keep 5 files)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_dir / "adapter.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (no format, we'll handle it in log functions)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logger()


def _format_content(content, indent: int = 4) -> str:
    """Format content for display."""
    c = COLORS
    if content is None:
        return f"{c['dim']}(empty){c['reset']}"
    if isinstance(content, str):
        # Add indentation to multiline content
        lines = content.split("\n")
        if len(lines) > 1:
            indented = "\n" + "\n".join(" " * indent + line for line in lines)
            return indented
        return content
    if isinstance(content, list):
        # Format each content block
        parts = []
        for i, block in enumerate(content):
            if hasattr(block, "type"):
                # Pydantic model (ContentBlock)
                block_type = block.type
                if block_type == "text":
                    text = getattr(block, "text", "") or ""
                    parts.append(f"{c['dim']}[text]{c['reset']} {text}")
                elif block_type == "image":
                    source = getattr(block, "source", None)
                    if source:
                        parts.append(f"{c['dim']}[image:{source.type}]{c['reset']} {source.media_type}")
                    else:
                        parts.append(f"{c['dim']}[image]{c['reset']}")
                elif block_type == "tool_use":
                    tool = getattr(block, "tool_use", None)
                    if tool:
                        parts.append(f"{c['dim']}[tool_use:{tool.name}]{c['reset']} {tool.input}")
                    else:
                        parts.append(f"{c['dim']}[tool_use]{c['reset']}")
                elif block_type == "tool_result":
                    result = getattr(block, "tool_result", None)
                    if result:
                        parts.append(f"{c['dim']}[tool_result:{result.tool_use_id}]{c['reset']} {result.content}")
                    else:
                        parts.append(f"{c['dim']}[tool_result]{c['reset']}")
                else:
                    parts.append(f"{c['dim']}[{block_type}]{c['reset']} {block}")
            elif isinstance(block, dict):
                # Dict format
                block_type = block.get("type", "unknown")
                if block_type == "text":
                    parts.append(f"{c['dim']}[text]{c['reset']} {block.get('text', '')}")
                elif block_type == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        parts.append(f"{c['dim']}[image:base64]{c['reset']} {url[:50]}...")
                    else:
                        parts.append(f"{c['dim']}[image:url]{c['reset']} {url}")
                else:
                    parts.append(f"{c['dim']}[{block_type}]{c['reset']} {block}")
            else:
                parts.append(str(block))

        if len(parts) == 1:
            return parts[0]
        return "\n" + "\n".join(" " * indent + p for p in parts)
    return str(content)


def log_request(request_id: str, model: str, messages: list, **kwargs) -> None:
    """Log incoming chat request with beautiful formatting."""
    c = COLORS
    separator = f"{c['dim']}{'─' * 80}{c['reset']}"

    lines = [
        "",
        separator,
        f"{c['cyan']}{c['bold']}▶ REQUEST{c['reset']}  {c['dim']}[{request_id}]{c['reset']}",
        separator,
        f"  {c['bold']}Model:{c['reset']}       {c['green']}{model}{c['reset']}",
        f"  {c['bold']}Stream:{c['reset']}      {kwargs.get('stream', False)}",
        f"  {c['bold']}Max Tokens:{c['reset']}  {kwargs.get('max_tokens') or 'default'}",
        f"  {c['bold']}Temperature:{c['reset']} {kwargs.get('temperature') or 'default'}",
        "",
        f"  {c['bold']}Messages:{c['reset']}",
    ]

    for i, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)

        role_color = {
            "system": c['magenta'],
            "user": c['blue'],
            "assistant": c['green'],
        }.get(role, c['reset'])

        lines.append(f"  {c['dim']}[{i+1}]{c['reset']} {role_color}{c['bold']}{role}{c['reset']}")
        lines.append(f"      {_format_content(content, indent=6)}")
        lines.append("")

    logger.info("\n".join(lines))


def log_response(
    request_id: str,
    content: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    finish_reason: str = "stop",
    error: str | None = None,
) -> None:
    """Log chat response with beautiful formatting."""
    c = COLORS
    separator = f"{c['dim']}{'─' * 80}{c['reset']}"

    if error:
        lines = [
            "",
            separator,
            f"{c['red']}{c['bold']}✖ ERROR{c['reset']}  {c['dim']}[{request_id}]{c['reset']}",
            separator,
            f"  {c['red']}{error}{c['reset']}",
            separator,
        ]
        logger.error("\n".join(lines))
    else:
        lines = [
            "",
            separator,
            f"{c['green']}{c['bold']}◀ RESPONSE{c['reset']}  {c['dim']}[{request_id}]{c['reset']}",
            separator,
            f"  {c['bold']}Finish Reason:{c['reset']} {finish_reason}",
            f"  {c['bold']}Tokens:{c['reset']}        {c['yellow']}input={input_tokens} output={output_tokens} total={input_tokens + output_tokens}{c['reset']}",
            f"  {c['bold']}Content Length:{c['reset']} {len(content) if content else 0} chars",
            "",
            f"  {c['bold']}Content:{c['reset']}",
            f"      {_format_content(content, indent=6)}",
            "",
            separator,
        ]
        logger.info("\n".join(lines))


def log_stream_start(request_id: str, model: str) -> None:
    """Log stream start."""
    c = COLORS
    logger.debug(f"{c['cyan']}⟳ STREAM START{c['reset']} {c['dim']}[{request_id}]{c['reset']} model={c['green']}{model}{c['reset']}")


def log_stream_end(request_id: str) -> None:
    """Log stream end."""
    c = COLORS
    logger.debug(f"{c['cyan']}⟳ STREAM END{c['reset']} {c['dim']}[{request_id}]{c['reset']}")


def log_stream_chunk(request_id: str, content: str) -> None:
    """Log stream chunk (only in DEBUG level)."""
    c = COLORS
    if logger.level <= logging.DEBUG:
        # Show chunk content inline, escape newlines for readability
        display = content.replace("\n", "\\n")
        if len(display) > 100:
            display = display[:100] + "..."
        logger.debug(f"{c['dim']}  chunk [{request_id}]:{c['reset']} {display}")
