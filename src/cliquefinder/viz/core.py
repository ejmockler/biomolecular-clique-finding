"""
Core visualization primitives: Figure wrapper and FigureCollection.

This module provides unified abstractions for matplotlib and plotly figures,
enabling consistent saving, display, and report generation.

Design follows perceptual engineering principles:
- Figure: Single visualization unit with metadata
- FigureCollection: Batch operations and report generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union, Optional, Any
from datetime import datetime
import io
import base64

import matplotlib.figure
import matplotlib.pyplot as plt

# Type aliases
FigureType = Union[matplotlib.figure.Figure, Any]  # Any for plotly.graph_objects.Figure
OutputFormat = Literal["png", "pdf", "svg", "html", "json"]


@dataclass
class Figure:
    """
    Unified wrapper for matplotlib and plotly figures.

    Provides consistent interface for saving, displaying, and embedding
    figures regardless of underlying library.

    Attributes
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The underlying figure object
    title : str
        Human-readable title for the figure
    description : str
        Longer description explaining what the figure shows
    figure_type : {"matplotlib", "plotly"}
        Which library created this figure
    metadata : dict
        Additional metadata (creation time, parameters used, etc.)

    Examples
    --------
    >>> fig = Figure(
    ...     fig=plt.figure(),
    ...     title="Outlier Distribution",
    ...     description="MAD-Z scores before and after imputation",
    ...     figure_type="matplotlib"
    ... )
    >>> fig.save("outliers.pdf")
    """
    fig: FigureType
    title: str
    description: str
    figure_type: Literal["matplotlib", "plotly"]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()

    def save(
        self,
        path: Path | str,
        format: Optional[OutputFormat] = None,
        dpi: int = 300,
        **kwargs
    ) -> Path:
        """
        Save figure to file.

        Parameters
        ----------
        path : Path or str
            Output file path. Format inferred from extension if not specified.
        format : str, optional
            Output format. If None, inferred from path extension.
        dpi : int, default 300
            DPI for raster formats (png, jpg). Ignored for vector formats.
        **kwargs
            Additional arguments passed to underlying save function.

        Returns
        -------
        Path
            The path where the figure was saved.
        """
        path = Path(path)

        # Infer format from extension if not provided
        if format is None:
            format = path.suffix.lstrip(".").lower()
            if format not in ("png", "pdf", "svg", "html", "json"):
                format = "png"

        path.parent.mkdir(parents=True, exist_ok=True)

        if self.figure_type == "matplotlib":
            self._save_matplotlib(path, format, dpi, **kwargs)
        else:
            self._save_plotly(path, format, **kwargs)

        return path

    def _save_matplotlib(self, path: Path, format: str, dpi: int, **kwargs):
        """Save matplotlib figure."""
        save_kwargs = {
            "dpi": dpi,
            "bbox_inches": "tight",
            "facecolor": "white",
            **kwargs
        }

        if format == "html":
            # Convert to PNG and embed in minimal HTML
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", **save_kwargs)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            html = f"""<!DOCTYPE html>
<html><head><title>{self.title}</title></head>
<body style="margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;background:#f5f5f5;">
<img src="data:image/png;base64,{img_b64}" alt="{self.title}">
</body></html>"""
            path.write_text(html)
        else:
            self.fig.savefig(path, format=format, **save_kwargs)

    def _save_plotly(self, path: Path, format: str, **kwargs):
        """Save plotly figure."""
        if format == "html":
            self.fig.write_html(
                path,
                include_plotlyjs="cdn",  # Use CDN to reduce file size
                full_html=True,
                **kwargs
            )
        elif format == "json":
            self.fig.write_json(path, **kwargs)
        else:
            # Static image export (requires kaleido)
            try:
                self.fig.write_image(path, format=format, scale=2, **kwargs)
            except ValueError as e:
                if "kaleido" in str(e).lower():
                    raise RuntimeError(
                        "Static image export requires kaleido. "
                        "Install with: pip install kaleido"
                    ) from e
                raise

    def show(self):
        """
        Display figure interactively.

        Works in Jupyter notebooks and standalone Python scripts.
        """
        if self.figure_type == "matplotlib":
            plt.show()
        else:
            self.fig.show()

    def to_base64(self, format: str = "png", dpi: int = 150) -> str:
        """
        Convert figure to base64-encoded string.

        Useful for embedding in HTML reports without separate files.

        Parameters
        ----------
        format : str, default "png"
            Image format for matplotlib figures.
        dpi : int, default 150
            DPI for raster output.

        Returns
        -------
        str
            Base64-encoded image data.
        """
        buf = io.BytesIO()

        if self.figure_type == "matplotlib":
            self.fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
        else:
            # Plotly to PNG
            self.fig.write_image(buf, format=format, scale=2)

        return base64.b64encode(buf.getvalue()).decode()

    def close(self):
        """Close the figure to free memory."""
        if self.figure_type == "matplotlib":
            plt.close(self.fig)


class FigureCollection:
    """
    Collection of figures for batch operations and report generation.

    Provides unified interface for saving multiple figures, generating
    HTML reports, and managing figure lifecycle.

    Attributes
    ----------
    figures : dict[str, Figure]
        Named collection of figures.

    Examples
    --------
    >>> collection = FigureCollection()
    >>> collection.add("outliers", viz.plot_outlier_distribution(...))
    >>> collection.add("sample_corr", viz.plot_sample_correlation(...))
    >>>
    >>> # Save all as PDF
    >>> collection.save_all(Path("figures/"), format="pdf")
    >>>
    >>> # Generate HTML report
    >>> collection.to_html_report(Path("report.html"), title="QC Report")
    """

    def __init__(self):
        self.figures: dict[str, Figure] = {}
        self._creation_order: list[str] = []

    def add(self, key: str, fig: Figure) -> "FigureCollection":
        """
        Add a named figure to the collection.

        Parameters
        ----------
        key : str
            Unique identifier for this figure.
        fig : Figure
            The figure to add.

        Returns
        -------
        FigureCollection
            Self, for method chaining.
        """
        self.figures[key] = fig
        if key not in self._creation_order:
            self._creation_order.append(key)
        return self

    def get(self, key: str) -> Optional[Figure]:
        """Get figure by key, or None if not found."""
        return self.figures.get(key)

    def __getitem__(self, key: str) -> Figure:
        return self.figures[key]

    def __len__(self) -> int:
        return len(self.figures)

    def __iter__(self):
        """Iterate in creation order."""
        for key in self._creation_order:
            yield key, self.figures[key]

    def save_all(
        self,
        output_dir: Path | str,
        format: OutputFormat = "png",
        dpi: int = 300
    ) -> list[Path]:
        """
        Save all figures to a directory.

        Parameters
        ----------
        output_dir : Path or str
            Directory for output files.
        format : str, default "png"
            Output format for all figures.
        dpi : int, default 300
            DPI for raster formats.

        Returns
        -------
        list[Path]
            Paths to saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for key, fig in self:
            path = output_dir / f"{key}.{format}"
            fig.save(path, format=format, dpi=dpi)
            saved.append(path)

        return saved

    def to_html_report(
        self,
        output_path: Path | str,
        title: str = "Analysis Report",
        description: str = "",
        template: Optional[str] = None,
        **context
    ) -> Path:
        """
        Generate HTML report with all figures embedded.

        Parameters
        ----------
        output_path : Path or str
            Output HTML file path.
        title : str
            Report title.
        description : str
            Report description.
        template : str, optional
            Custom HTML template. If None, uses default template.
        **context
            Additional variables for template rendering.

        Returns
        -------
        Path
            Path to generated report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare figure content
        figure_html = []
        for key, fig in self:
            if fig.figure_type == "plotly":
                # Embed plotly as interactive div
                div_html = fig.fig.to_html(include_plotlyjs=False, full_html=False)
                figure_html.append({
                    "key": key,
                    "title": fig.title,
                    "description": fig.description,
                    "content": div_html,
                    "type": "plotly"
                })
            else:
                # Embed matplotlib as base64 image
                img_b64 = fig.to_base64(format="png", dpi=150)
                figure_html.append({
                    "key": key,
                    "title": fig.title,
                    "description": fig.description,
                    "content": f'<img src="data:image/png;base64,{img_b64}" alt="{fig.title}">',
                    "type": "matplotlib"
                })

        # Use template or default
        if template is None:
            template = self._default_template()

        # Simple template rendering (avoid jinja2 dependency for basic use)
        html = template.replace("{{title}}", title)
        html = html.replace("{{description}}", description)
        html = html.replace("{{timestamp}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Build figures section
        figures_section = []
        for fig_data in figure_html:
            figures_section.append(f"""
            <section class="figure-section" id="{fig_data['key']}">
                <h2>{fig_data['title']}</h2>
                <p class="description">{fig_data['description']}</p>
                <div class="figure-content">{fig_data['content']}</div>
            </section>
            """)

        html = html.replace("{{figures}}", "\n".join(figures_section))

        # Check if any plotly figures
        has_plotly = any(f["type"] == "plotly" for f in figure_html)
        plotly_script = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>' if has_plotly else ""
        html = html.replace("{{plotly_script}}", plotly_script)

        output_path.write_text(html)
        return output_path

    def _default_template(self) -> str:
        """Default HTML report template."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    {{plotly_script}}
    <style>
        :root {
            --bg: #fafafa;
            --fg: #1a1a1a;
            --accent: #2563eb;
            --border: #e5e7eb;
            --section-bg: #ffffff;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: var(--fg);
            background: var(--bg);
            margin: 0;
            padding: 2rem;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border);
        }
        h1 {
            font-size: 2rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
        }
        .timestamp {
            color: #6b7280;
            font-size: 0.875rem;
        }
        .figure-section {
            background: var(--section-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .figure-section h2 {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
            color: var(--fg);
        }
        .description {
            color: #6b7280;
            font-size: 0.875rem;
            margin: 0 0 1rem 0;
        }
        .figure-content {
            display: flex;
            justify-content: center;
            overflow-x: auto;
        }
        .figure-content img {
            max-width: 100%;
            height: auto;
        }
        nav {
            position: sticky;
            top: 1rem;
            background: var(--section-bg);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }
        nav a {
            display: block;
            color: var(--accent);
            text-decoration: none;
            padding: 0.25rem 0;
            font-size: 0.875rem;
        }
        nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{title}}</h1>
            <p class="timestamp">Generated: {{timestamp}}</p>
            <p>{{description}}</p>
        </header>
        <main>
            {{figures}}
        </main>
    </div>
</body>
</html>"""

    def close_all(self):
        """Close all figures to free memory."""
        for _, fig in self:
            fig.close()
        self.figures.clear()
        self._creation_order.clear()
