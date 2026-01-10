"""Common chart styling and helper functions for the dashboard."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


# Color palette (Catppuccin Mocha)
COLORS = {
    "red": "#f38ba8",
    "green": "#a6e3a1",
    "blue": "#89b4fa",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "mauve": "#cba6f7",
    "teal": "#94e2d5",
    "sky": "#89dceb",
}

# Common layout settings for dark theme
DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
}

GRID_STYLE = {"gridcolor": "#313244"}


def apply_dark_style(fig: go.Figure, height: int = 280, margin: dict | None = None) -> go.Figure:
    """Apply standard dark styling to a figure."""
    if margin is None:
        margin = {"l": 0, "r": 0, "t": 30, "b": 0}
    
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin=margin,
    )
    return fig


def make_metric_chart(
    df: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    title: str,
    height: int = 280,
    x_col: str | None = None,
) -> go.Figure:
    """
    Create a line chart for multiple metrics.
    
    Args:
        df: DataFrame with the data
        metrics: List of (column_name, display_name, color) tuples
        title: Chart title
        height: Chart height in pixels
        x_col: Column to use for x-axis (None for index)
    """
    fig = go.Figure()
    
    for col, name, color in metrics:
        if col in df.columns:
            x = df[x_col] if x_col and x_col in df.columns else df.index
            fig.add_trace(go.Scatter(
                x=x,
                y=df[col],
                name=name,
                line=dict(color=color, width=2),
                mode="lines",
            ))
    
    fig.update_layout(
        title=title,
        height=height,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    
    return fig


def make_heatmap(
    z_data: list[list[float]],
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    height: int = 250,
    colorscale: str = "Viridis",
    show_scale: bool = False,
) -> go.Figure:
    """Create a heatmap with standard styling."""
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        showscale=show_scale,
        hovertemplate="%{y}<br>%{x}<br>%{z:.1f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    
    return fig


def make_bar_chart(
    x: list,
    y: list,
    title: str,
    height: int = 250,
    color: str = COLORS["blue"],
    orientation: str = "v",
    x_title: str = "",
    y_title: str = "",
) -> go.Figure:
    """Create a bar chart with standard styling."""
    if orientation == "h":
        fig = go.Figure(data=go.Bar(x=y, y=x, orientation="h", marker_color=color))
    else:
        fig = go.Figure(data=go.Bar(x=x, y=y, marker_color=color))
    
    fig.update_layout(
        title=title,
        height=height,
        **DARK_LAYOUT,
        xaxis=dict(title=x_title, **GRID_STYLE),
        yaxis=dict(title=y_title, **GRID_STYLE),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    
    return fig


def make_dual_axis_chart(
    x: list[str],
    y1: list[float],
    y2: list[float],
    name1: str,
    name2: str,
    title: str,
    y1_title: str = "",
    y2_title: str = "",
    color1: str = COLORS["red"],
    color2: str = COLORS["green"],
    height: int = 250,
) -> go.Figure:
    """Create a dual Y-axis line chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=x, y=y1, name=name1,
        line=dict(color=color1, width=2),
        hovertemplate=f"{name1}: %{{y:.2f}}<extra></extra>",
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=x, y=y2, name=name2,
        line=dict(color=color2, width=2),
        hovertemplate=f"{name2}: %{{y:.1f}}%<extra></extra>",
    ), secondary_y=True)
    
    # Calculate ranges
    y1_max = max(y1) * 1.1 if y1 and max(y1) > 0 else 1
    y2_max = max(y2) * 1.1 if y2 and max(y2) > 0 else 100
    
    fig.update_layout(
        title=title,
        height=height,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    
    fig.update_yaxes(
        title_text=y1_title, secondary_y=False,
        gridcolor="#313244", range=[0, y1_max],
        title_font=dict(color=color1),
    )
    fig.update_yaxes(
        title_text=y2_title, secondary_y=True,
        gridcolor="#313244", range=[0, y2_max],
        title_font=dict(color=color2),
    )
    
    return fig


def make_box_plot(
    data_by_category: dict[str, list[float]],
    title: str,
    y_title: str = "",
    height: int = 300,
    color: str = COLORS["green"],
    line_color: str = COLORS["teal"],
) -> go.Figure:
    """Create a box plot for multiple categories."""
    fig = go.Figure()
    
    for category, values in data_by_category.items():
        if values:
            fig.add_trace(go.Box(
                y=values,
                name=category,
                boxpoints="outliers",
                marker_color=color,
                line_color=line_color,
            ))
    
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        height=height,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )
    
    return fig
