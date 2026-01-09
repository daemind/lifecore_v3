#!/usr/bin/env python3
"""
PIPELINE VISUALIZER - Dynamic Gradio Dashboard
===============================================

Visualisation temps réel d'un pipeline de production.
- Charge la config depuis YAML
- Affiche le Gantt dynamiquement
- Mise à jour des états en temps réel

Run:
    python examples/pipeline_visualizer.py
"""

import numpy as np
import sys
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, '.')

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    sys.exit(1)

from lifecore.pipeline import (
    Pipeline, Stage, Job, JobStatus,
    create_fulfillment_pipeline, create_manufacturing_pipeline
)


# === GANTT RENDERING ===

def render_gantt_svg(pipeline: Pipeline, width: int = 800, height: int = 400) -> str:
    """Génère un SVG du diagramme Gantt."""
    
    # Handle None pipeline
    if pipeline is None:
        return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#1a1a2e"/>
            <text x="50%" y="50%" fill="#888" text-anchor="middle" font-size="20">
                Create a pipeline to start
            </text>
        </svg>"""
    
    gantt = pipeline.export_gantt(max_jobs=30)
    if not gantt:
        return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#1a1a2e"/>
            <text x="50%" y="50%" fill="#888" text-anchor="middle" font-size="20">
                No jobs yet - click Step or Run
            </text>
        </svg>"""
    
    # Couleurs par stage
    stage_colors = {
        "picking": "#4CAF50",
        "transport": "#2196F3", 
        "packing": "#FF9800",
        "shipping": "#9C27B0",
        "raw_material": "#795548",
        "cutting": "#F44336",
        "assembly": "#00BCD4",
        "quality_check": "#FFEB3B",
        "packaging": "#E91E63"
    }
    default_color = "#607D8B"
    
    # Dimensions
    margin_left = 80
    margin_top = 50
    margin_right = 20
    margin_bottom = 30
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    
    # Calculer échelle temps
    max_time = max(entry["end"] for entry in gantt) if gantt else 100
    min_time = min(entry["start"] for entry in gantt) if gantt else 0
    time_range = max(1, max_time - min_time)
    
    # Grouper par job
    jobs_data = {}
    for entry in gantt:
        job_id = entry["job_id"]
        if job_id not in jobs_data:
            jobs_data[job_id] = []
        jobs_data[job_id].append(entry)
    
    # Trier les jobs
    job_ids = sorted(jobs_data.keys())[-20:]  # Last 20 jobs
    row_height = min(25, chart_height / max(1, len(job_ids)))
    
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '<defs>',
        '<linearGradient id="bgGrad" x1="0%" y1="0%" x2="0%" y2="100%">',
        '<stop offset="0%" style="stop-color:#1a1a2e"/>',
        '<stop offset="100%" style="stop-color:#16213e"/>',
        '</linearGradient>',
        '</defs>',
        '<rect width="100%" height="100%" fill="url(#bgGrad)"/>',
        
        # Title
        f'<text x="{width/2}" y="25" fill="white" text-anchor="middle" font-size="16" font-weight="bold">',
        f'Pipeline: {pipeline.name} | Time: {pipeline.current_time:.0f}',
        '</text>',
    ]
    
    # Grid lines
    num_lines = 10
    for i in range(num_lines + 1):
        x = margin_left + (i / num_lines) * chart_width
        time_val = min_time + (i / num_lines) * time_range
        svg_parts.append(
            f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + chart_height}" '
            f'stroke="#333" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{x}" y="{height - 10}" fill="#666" text-anchor="middle" font-size="10">'
            f'{time_val:.0f}</text>'
        )
    
    # Bars
    for row_idx, job_id in enumerate(job_ids):
        y = margin_top + row_idx * row_height
        
        # Job label
        svg_parts.append(
            f'<text x="{margin_left - 5}" y="{y + row_height/2 + 4}" '
            f'fill="#aaa" text-anchor="end" font-size="11">Job {job_id}</text>'
        )
        
        for entry in jobs_data[job_id]:
            start_x = margin_left + ((entry["start"] - min_time) / time_range) * chart_width
            end_x = margin_left + ((entry["end"] - min_time) / time_range) * chart_width
            bar_width = max(2, end_x - start_x)
            
            color = stage_colors.get(entry["stage"], default_color)
            opacity = "1.0" if entry["status"] == "completed" else "0.6"
            
            # Bar
            svg_parts.append(
                f'<rect x="{start_x}" y="{y + 2}" width="{bar_width}" height="{row_height - 4}" '
                f'fill="{color}" opacity="{opacity}" rx="3"/>'
            )
            
            # Stage label (if wide enough)
            if bar_width > 30:
                svg_parts.append(
                    f'<text x="{start_x + bar_width/2}" y="{y + row_height/2 + 3}" '
                    f'fill="white" text-anchor="middle" font-size="9" font-weight="bold">'
                    f'{entry["stage"][:4]}</text>'
                )
    
    # Legend
    legend_y = height - 25
    legend_x = margin_left
    for i, (stage, color) in enumerate(list(stage_colors.items())[:6]):
        x = legend_x + i * 100
        svg_parts.append(f'<rect x="{x}" y="{legend_y}" width="12" height="12" fill="{color}" rx="2"/>')
        svg_parts.append(
            f'<text x="{x + 16}" y="{legend_y + 10}" fill="#888" font-size="10">{stage}</text>'
        )
    
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def render_stats_html(pipeline: Pipeline) -> str:
    """Génère le HTML des statistiques."""
    stats = pipeline.get_stats()
    
    # Progress bar
    completion_rate = stats["completion_rate"] * 100
    
    # Resource utilization
    util_html = ""
    for res, util in stats["resource_utilization"].items():
        pct = util * 100
        color = "#4CAF50" if pct < 70 else "#FF9800" if pct < 90 else "#F44336"
        util_html += f"""
        <div style="margin: 5px 0;">
            <div style="display: flex; justify-content: space-between;">
                <span>{res}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background: #333; height: 8px; border-radius: 4px;">
                <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
        """
    
    return f"""
    <div style="color: white; font-family: sans-serif; padding: 10px;">
        <h3 style="margin: 0 0 15px 0; color: #4CAF50;">📊 Statistics</h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
            <div style="background: #2a2a4e; padding: 10px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #4CAF50;">{stats['jobs_completed']}</div>
                <div style="font-size: 12px; color: #888;">Completed</div>
            </div>
            <div style="background: #2a2a4e; padding: 10px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #F44336;">{stats['jobs_failed']}</div>
                <div style="font-size: 12px; color: #888;">Failed</div>
            </div>
            <div style="background: #2a2a4e; padding: 10px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #2196F3;">{stats['jobs_in_progress']}</div>
                <div style="font-size: 12px; color: #888;">In Progress</div>
            </div>
            <div style="background: #2a2a4e; padding: 10px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #FF9800;">{stats['avg_time_in_system']:.1f}</div>
                <div style="font-size: 12px; color: #888;">Avg Time</div>
            </div>
        </div>
        
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Completion Rate</span>
                <span>{completion_rate:.1f}%</span>
            </div>
            <div style="background: #333; height: 12px; border-radius: 6px;">
                <div style="background: linear-gradient(90deg, #4CAF50, #8BC34A); width: {completion_rate}%; height: 100%; border-radius: 6px;"></div>
            </div>
        </div>
        
        <h4 style="margin: 15px 0 10px 0; color: #888;">Resource Utilization</h4>
        {util_html}
        
        <div style="margin-top: 15px; padding: 10px; background: #2a2a4e; border-radius: 8px;">
            <div style="font-size: 12px; color: #888;">Time: {stats['current_time']:.0f}</div>
            <div style="font-size: 12px; color: #888;">Total Jobs: {stats['jobs_created']}</div>
        </div>
    </div>
    """


# === GLOBAL STATE ===

class PipelineState:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_running = False
    
    def create_pipeline(self, pipeline_type: str, robots: int, stations: int, docks: int) -> str:
        if pipeline_type == "fulfillment":
            self.pipeline = create_fulfillment_pipeline(
                robots=robots,
                packing_stations=stations, 
                shipping_docks=docks
            )
        else:
            self.pipeline = create_manufacturing_pipeline(
                machines=robots,
                workers=stations,
                testers=docks
            )
        return f"✅ Pipeline '{pipeline_type}' created"
    
    def add_jobs(self, count: int, priority: int, deadline: int) -> str:
        if not self.pipeline:
            return "❌ Create a pipeline first"
        for _ in range(count):
            self.pipeline.create_job(priority=priority, deadline=deadline)
        return f"✅ Added {count} jobs (priority={priority}, deadline={deadline})"
    
    def step(self, steps: int = 1) -> tuple:
        if not self.pipeline:
            return "No pipeline", ""
        for _ in range(steps):
            self.pipeline.step()
        return render_gantt_svg(self.pipeline), render_stats_html(self.pipeline)
    
    def reset(self) -> str:
        self.pipeline = None
        return "🔄 Pipeline reset"


state = PipelineState()


# === GRADIO INTERFACE ===

def create_interface():
    with gr.Blocks(title="Pipeline Visualizer") as demo:
        
        gr.Markdown("""
        # 🏭 Pipeline Visualizer
        **Dynamic Gantt chart for production scheduling**
        """)
        
        with gr.Row():
            # Left: Controls
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                pipeline_type = gr.Dropdown(
                    choices=["fulfillment", "manufacturing"],
                    value="fulfillment",
                    label="Pipeline Type"
                )
                
                with gr.Row():
                    robots = gr.Slider(5, 100, value=30, step=5, label="Robots/Machines")
                    stations = gr.Slider(5, 50, value=15, step=5, label="Stations/Workers")
                    docks = gr.Slider(1, 20, value=5, step=1, label="Docks/Testers")
                
                create_btn = gr.Button("🔧 Create Pipeline", variant="primary")
                create_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### 📦 Add Jobs")
                
                with gr.Row():
                    job_count = gr.Slider(1, 50, value=10, step=1, label="Count")
                    job_priority = gr.Slider(1, 10, value=5, step=1, label="Priority")
                    job_deadline = gr.Slider(20, 200, value=60, step=10, label="Deadline")
                
                add_jobs_btn = gr.Button("➕ Add Jobs")
                jobs_status = gr.Textbox(label="Jobs Status", interactive=False)
                
                gr.Markdown("### ▶️ Simulation")
                
                with gr.Row():
                    step_btn = gr.Button("Step +1")
                    step10_btn = gr.Button("Step +10")
                    step50_btn = gr.Button("Step +50")
                
                reset_btn = gr.Button("🔄 Reset", variant="secondary")
            
            # Right: Visualization
            with gr.Column(scale=2):
                gantt_output = gr.HTML(
                    value=render_gantt_svg(None) if not state.pipeline else render_gantt_svg(state.pipeline),
                    label="Gantt Chart"
                )
                stats_output = gr.HTML(
                    value="<div style='color:#888; text-align:center; padding:20px;'>Create a pipeline to see stats</div>",
                    label="Statistics"
                )
        
        # Event handlers
        def on_create(ptype, r, s, d):
            status = state.create_pipeline(ptype, int(r), int(s), int(d))
            gantt, stats = state.step(0)
            return status, gantt, stats if stats else "<div style='color:#888;'>No stats yet</div>"
        
        def on_add_jobs(count, priority, deadline):
            status = state.add_jobs(int(count), int(priority), int(deadline))
            gantt, stats = state.step(0)
            return status, gantt, stats
        
        def on_step(steps):
            gantt, stats = state.step(steps)
            return gantt, stats
        
        def on_reset():
            status = state.reset()
            return status, render_gantt_svg(None), "<div style='color:#888;'>Pipeline reset</div>"
        
        # Wire up
        create_btn.click(
            on_create,
            inputs=[pipeline_type, robots, stations, docks],
            outputs=[create_status, gantt_output, stats_output]
        )
        
        add_jobs_btn.click(
            on_add_jobs,
            inputs=[job_count, job_priority, job_deadline],
            outputs=[jobs_status, gantt_output, stats_output]
        )
        
        step_btn.click(lambda: on_step(1), outputs=[gantt_output, stats_output])
        step10_btn.click(lambda: on_step(10), outputs=[gantt_output, stats_output])
        step50_btn.click(lambda: on_step(50), outputs=[gantt_output, stats_output])
        
        reset_btn.click(
            on_reset,
            outputs=[create_status, gantt_output, stats_output]
        )
    
    return demo


if __name__ == "__main__":
    print("🏭 Pipeline Visualizer")
    print("=" * 50)
    print("Starting Gradio server...")
    print()
    
    demo = create_interface()
    demo.launch(share=False)
