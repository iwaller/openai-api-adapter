import logging
import re
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.config import settings

# Constants
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5
LOG_SEPARATOR_WIDTH = 80
LOG_CHUNK_DISPLAY_MAX_LENGTH = 100

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

# Thread-safe logger initialization
_logger_lock = threading.Lock()
_logger_initialized = False


class StripAnsiFilter(logging.Filter):
    """Filter that strips ANSI escape codes from log messages."""

    ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.ANSI_ESCAPE.sub("", record.msg)
        return True


def setup_logger() -> logging.Logger:
    """
    Setup application logger with file and console handlers.

    Log files are stored in the configured log_dir with rotation.
    Thread-safe initialization to prevent duplicate handlers.
    """
    global _logger_initialized

    logger = logging.getLogger("openai-adapter")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    with _logger_lock:
        if _logger_initialized:
            return logger
        _logger_initialized = True

        # Try to create logs directory with error handling
        try:
            log_dir = Path(settings.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # File handler with rotation - strips ANSI codes
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = RotatingFileHandler(
                log_dir / "adapter.log",
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(StripAnsiFilter())  # Strip ANSI from file logs
            logger.addHandler(file_handler)

        except (PermissionError, OSError) as e:
            # Fall back to console-only logging
            print(
                f"Warning: Could not create log directory {settings.log_dir}: {e}",
                file=sys.stderr,
            )

        # Console handler - keep colors if TTY
        console_handler = logging.StreamHandler(sys.stdout)
        if sys.stdout.isatty():
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            # No TTY, strip colors
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            console_handler.addFilter(StripAnsiFilter())
        logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logger()


def _truncate_content(content: str | None, max_length: int = 0) -> str | None:
    """Truncate content if it exceeds max_length."""
    if content is None or max_length <= 0:
        return content
    if len(content) > max_length:
        return content[:max_length] + f"... [TRUNCATED {len(content) - max_length} chars]"
    return content


def _redact_content(content: Any) -> str:
    """Redact content for privacy, showing only length."""
    if content is None:
        return "(empty)"
    if isinstance(content, str):
        return f"[{len(content)} chars]"
    if isinstance(content, list):
        return f"[{len(content)} blocks]"
    return f"[{type(content).__name__}]"


def _format_content(content: Any, indent: int = 4) -> str:
    """Format content for display with optional redaction and truncation."""
    c = COLORS

    # Check if we should redact content
    if not settings.log_full_content:
        return f"{c['dim']}{_redact_content(content)}{c['reset']}"

    if content is None:
        return f"{c['dim']}(empty){c['reset']}"

    if isinstance(content, str):
        # Apply truncation if configured
        if settings.log_max_content_length > 0:
            content = _truncate_content(content, settings.log_max_content_length)

        # Add indentation to multiline content
        lines = content.split("\n")
        if len(lines) > 1:
            indented = "\n" + "\n".join(" " * indent + line for line in lines)
            return indented
        return content

    if isinstance(content, list):
        # Format each content block
        parts = []
        for block in content:
            if hasattr(block, "type"):
                # Pydantic model (ContentBlock)
                block_type = block.type
                if block_type == "text":
                    text = getattr(block, "text", "") or ""
                    if settings.log_max_content_length > 0:
                        text = _truncate_content(text, settings.log_max_content_length)
                    parts.append(f"{c['dim']}[text]{c['reset']} {text}")
                elif block_type == "image":
                    source = getattr(block, "source", None)
                    if source:
                        parts.append(
                            f"{c['dim']}[image:{source.type}]{c['reset']} {source.media_type}"
                        )
                    else:
                        parts.append(f"{c['dim']}[image]{c['reset']}")
                elif block_type == "tool_use":
                    tool = getattr(block, "tool_use", None)
                    if tool:
                        parts.append(
                            f"{c['dim']}[tool_use:{tool.name}]{c['reset']} {tool.input}"
                        )
                    else:
                        parts.append(f"{c['dim']}[tool_use]{c['reset']}")
                elif block_type == "tool_result":
                    result = getattr(block, "tool_result", None)
                    if result:
                        parts.append(
                            f"{c['dim']}[tool_result:{result.tool_use_id}]{c['reset']} {result.content}"
                        )
                    else:
                        parts.append(f"{c['dim']}[tool_result]{c['reset']}")
                else:
                    parts.append(f"{c['dim']}[{block_type}]{c['reset']} {block}")
            elif isinstance(block, dict):
                # Dict format
                block_type = block.get("type", "unknown")
                if block_type == "text":
                    text = block.get("text", "")
                    if settings.log_max_content_length > 0:
                        text = _truncate_content(text, settings.log_max_content_length)
                    parts.append(f"{c['dim']}[text]{c['reset']} {text}")
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


def log_request(
    request_id: str, model: str, messages: list[dict[str, Any]], **kwargs: Any
) -> None:
    """Log incoming chat request with beautiful formatting."""
    c = COLORS
    separator = f"{c['dim']}{'─' * LOG_SEPARATOR_WIDTH}{c['reset']}"

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
            "system": c["magenta"],
            "user": c["blue"],
            "assistant": c["green"],
        }.get(role, c["reset"])

        lines.append(
            f"  {c['dim']}[{i+1}]{c['reset']} {role_color}{c['bold']}{role}{c['reset']}"
        )
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
    separator = f"{c['dim']}{'─' * LOG_SEPARATOR_WIDTH}{c['reset']}"

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
        # Apply content formatting (with optional redaction/truncation)
        display_content = _format_content(content, indent=6)

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
            f"      {display_content}",
            "",
            separator,
        ]
        logger.info("\n".join(lines))


def log_stream_start(request_id: str, model: str) -> None:
    """Log stream start."""
    c = COLORS
    logger.debug(
        f"{c['cyan']}⟳ STREAM START{c['reset']} {c['dim']}[{request_id}]{c['reset']} model={c['green']}{model}{c['reset']}"
    )


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
        if len(display) > LOG_CHUNK_DISPLAY_MAX_LENGTH:
            display = display[:LOG_CHUNK_DISPLAY_MAX_LENGTH] + "..."
        logger.debug(f"{c['dim']}  chunk [{request_id}]:{c['reset']} {display}")
