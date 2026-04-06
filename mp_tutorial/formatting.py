"""Formatting utilities for model parallelism tutorial notebooks.

Provides helpers for styled output, comparison tables, highlighted code
blocks, and other notebook display enhancements.
"""

from IPython.display import HTML, display


def info_box(text, title=None):
    """Display a styled info box in the notebook.

    Args:
        text: Content to display.
        title: Optional title for the box.
    """
    title_html = ""
    if title:
        title_html = (
            f'<div style="font-weight:bold;margin-bottom:6px;'
            f'font-size:14px;">{title}</div>'
        )
    html = (
        f'<div style="border-left:4px solid #4A90D9;background:#EBF5FB;'
        f'padding:12px 16px;margin:8px 0;border-radius:4px;'
        f'font-family:sans-serif;font-size:13px;">'
        f'{title_html}{text}</div>'
    )
    display(HTML(html))


def gpu_required_banner():
    """Display a banner indicating the following cells require GPU.

    Reminds the user to run on a multi-GPU machine if no GPU
    is available locally.
    """
    html = (
        '<div style="border-left:4px solid #E67E22;background:#FDF2E9;'
        'padding:12px 16px;margin:8px 0;border-radius:4px;'
        'font-family:sans-serif;font-size:13px;">'
        '<div style="font-weight:bold;margin-bottom:6px;">'
        '⚠️ GPU Required</div>'
        'The following cells require CUDA GPUs. '
        'Run on a machine with CUDA GPUs (4+ recommended).<br>'
        'See README.md for remote Jupyter setup instructions.'
        '</div>'
    )
    display(HTML(html))


def comparison_table(headers, rows, title=None):
    """Display a styled comparison table.

    Args:
        headers: List of column header strings.
        rows: List of row lists.
        title: Optional table title.
    """
    title_html = ""
    if title:
        title_html = (
            f'<div style="font-weight:bold;font-size:14px;'
            f'margin-bottom:8px;">{title}</div>'
        )
    header_cells = "".join(
        f'<th style="padding:8px 12px;border-bottom:2px solid #ddd;'
        f'text-align:left;background:#f8f9fa;">{h}</th>'
        for h in headers
    )
    body_rows = ""
    for row in rows:
        cells = "".join(
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;">'
            f'{cell}</td>'
            for cell in row
        )
        body_rows += f"<tr>{cells}</tr>"
    html = (
        f'{title_html}'
        f'<table style="border-collapse:collapse;font-family:sans-serif;'
        f'font-size:13px;margin:8px 0;">'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{body_rows}</tbody></table>'
    )
    display(HTML(html))


def code_reference(code, source="Megatron-LM", filepath=None):
    """Display a highlighted code reference block.

    Args:
        code: Code string to display.
        source: Source framework/project name.
        filepath: Optional file path in the source project.
    """
    path_html = ""
    if filepath:
        path_html = f' — <code>{filepath}</code>'
    html = (
        f'<div style="margin:8px 0;border:1px solid #ddd;border-radius:4px;'
        f'overflow:hidden;">'
        f'<div style="background:#f8f9fa;padding:6px 12px;font-size:12px;'
        f'font-family:sans-serif;border-bottom:1px solid #ddd;color:#555;">'
        f'📖 {source}{path_html}</div>'
        f'<pre style="margin:0;padding:12px;background:#fff;'
        f'font-size:13px;overflow-x:auto;">'
        f'<code>{code}</code></pre></div>'
    )
    display(HTML(html))
