"""PaperBanana Dash web interface for academic figure generation."""

import asyncio
import logging
import os
import uuid

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Example content (mirrors the original PaperBanana Gradio demo)
# ---------------------------------------------------------------------------

EXAMPLE_METHOD = r"""## Methodology: The PaperBanana Framework

In this section, we present the architecture of PaperBanana, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperBanana orchestrates a collaborative team of five specialized agents---Retriever, Planner, Stylist, Visualizer, and Critic---to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

The Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$

### Visualizer Agent

The Visualizer Agent leverages an image generation model:
$$
I_t = \text{Image-Gen}(P_t)
$$

### Critic Agent

The Critic provides targeted feedback and produces a refined description:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
The Visualizer-Critic loop iterates for $T=3$ rounds."""

EXAMPLE_CAPTION = (
    "Figure 1: Overview of our PaperBanana framework. Given the source context "
    "and communicative intent, we first apply a Linear Planning Phase to retrieve "
    "relevant reference examples and synthesize a stylistically optimized "
    "description. We then use an Iterative Refinement Loop (consisting of "
    "Visualizer and Critic agents) to transform the description into visual "
    "output and conduct multi-round refinements to produce the final academic "
    "illustration."
)

PIPELINE_DESCRIPTIONS = {
    "vanilla": "Direct image generation from content (no agents)",
    "dev_planner": "Retriever -> Planner -> Visualizer",
    "dev_planner_critic": "Retriever -> Planner -> Visualizer -> Critic -> Visualizer",
    "demo_full": "Retriever -> Planner -> Stylist -> Visualizer -> Critic -> Visualizer",
}

PIPELINE_OPTIONS = [
    {"label": f"{mode} -- {desc}", "value": mode}
    for mode, desc in PIPELINE_DESCRIPTIONS.items()
]

ASPECT_RATIO_OPTIONS = [
    {"label": "16:9 (widescreen)", "value": "16:9"},
    {"label": "21:9 (ultrawide)", "value": "21:9"},
    {"label": "3:2 (standard)", "value": "3:2"},
]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="PaperBanana",
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

server = app.server  # for deployment (gunicorn, etc.)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _build_header():
    return html.Div(
        className="pb-header",
        children=[
            html.Img(
                src=app.get_asset_url("logo.jpg"),
                alt="PaperBanana logo",
                height=56,
                width=56,
            ),
            html.Div([
                html.P("PaperBanana", className="pb-header-title"),
                html.Div(
                    className="pb-header-badges",
                    children=[
                        html.Span("Multi-Agent", className="pb-badge"),
                        html.Span("Scientific Diagrams", className="pb-badge"),
                    ],
                ),
            ]),
        ],
    )


def _build_sidebar():
    default_api_key = os.environ.get("GOOGLE_API_KEY", "")

    return html.Div(
        className="pb-sidebar",
        children=[
            html.P("Settings", className="pb-section-label"),

            # API key
            dbc.Label(
                "Google API Key",
                html_for="api-key-input",
                style={"marginTop": "8px"},
            ),
            dbc.Input(
                id="api-key-input",
                type="password",
                value=default_api_key,
                placeholder="AIza...",
                autocomplete="off",
                spellCheck=False,
            ),

            html.Hr(style={"borderColor": "#444"}),

            # Pipeline mode
            dbc.Label("Pipeline Mode", html_for="pipeline-mode"),
            dcc.Dropdown(
                id="pipeline-mode",
                options=PIPELINE_OPTIONS,
                value="dev_planner_critic",
                clearable=False,
                style={"backgroundColor": "#333", "color": "#e0e0e0"},
            ),

            # Task type
            dbc.Label(
                "Task Type",
                html_for="task-type",
                style={"marginTop": "12px"},
            ),
            dcc.Dropdown(
                id="task-type",
                options=[
                    {"label": "Diagram", "value": "diagram"},
                    {"label": "Plot", "value": "plot"},
                ],
                value="diagram",
                clearable=False,
                style={"backgroundColor": "#333", "color": "#e0e0e0"},
            ),

            # Aspect ratio
            dbc.Label(
                "Aspect Ratio",
                html_for="aspect-ratio",
                style={"marginTop": "12px"},
            ),
            dcc.Dropdown(
                id="aspect-ratio",
                options=ASPECT_RATIO_OPTIONS,
                value="16:9",
                clearable=False,
                style={"backgroundColor": "#333", "color": "#e0e0e0"},
            ),

            # Number of candidates
            dbc.Label(
                "Number of Candidates",
                html_for="num-candidates",
                style={"marginTop": "12px"},
            ),
            dbc.Input(
                id="num-candidates",
                type="number",
                value=1,
                min=1,
                max=10,
                step=1,
                autocomplete="off",
            ),

            # Max critic rounds
            dbc.Label(
                "Max Critic Rounds",
                html_for="max-critic-rounds",
                style={"marginTop": "12px"},
            ),
            dcc.Slider(
                id="max-critic-rounds",
                min=0,
                max=5,
                step=1,
                value=3,
                marks={i: {"label": str(i), "style": {"color": "#ccc"}} for i in range(6)},
            ),
        ],
    )


def _build_main_area():
    return html.Div(
        className="pb-main",
        children=[
            html.P("Input", className="pb-section-label"),

            # Method content
            dbc.Label("Method Content", html_for="method-content"),
            dbc.Textarea(
                id="method-content",
                value=EXAMPLE_METHOD,
                placeholder="Paste your paper's methodology section here...",
                style={
                    "height": "220px",
                    "backgroundColor": "#333",
                    "color": "#e0e0e0",
                    "borderColor": "#555",
                },
            ),

            # Figure caption
            dbc.Label(
                "Figure Caption",
                html_for="figure-caption",
                style={"marginTop": "12px"},
            ),
            dbc.Textarea(
                id="figure-caption",
                value=EXAMPLE_CAPTION,
                placeholder="Describe the desired figure content and intent...",
                style={
                    "height": "100px",
                    "backgroundColor": "#333",
                    "color": "#e0e0e0",
                    "borderColor": "#555",
                },
            ),

            # Generate button
            html.Div(
                className="pb-generate-btn",
                style={"marginTop": "16px"},
                children=[
                    dbc.Button(
                        "Generate Candidates",
                        id="generate-btn",
                        color="warning",
                        size="lg",
                        style={"width": "100%"},
                    ),
                ],
            ),

            # Status
            html.Div(
                id="status-text",
                className="pb-status",
                style={"marginTop": "16px"},
                children="Ready. Configure settings and press Generate Candidates.",
            ),

            html.Hr(style={"borderColor": "#444"}),

            # Results section
            html.P("Results", className="pb-section-label"),
            html.Div(
                id="results-gallery",
                children=html.Div(
                    className="pb-empty-state",
                    children="No results yet. Generate candidates to see them here.",
                ),
            ),

            # Hidden stores
            dcc.Store(id="results-store", data=[]),
            dcc.Store(id="run-id-store", data=""),
            # Interval for polling long-running tasks
            dcc.Interval(
                id="poll-interval",
                interval=2000,
                n_intervals=0,
                disabled=True,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Full layout
# ---------------------------------------------------------------------------

app.layout = dbc.Container(
    fluid=True,
    style={"maxWidth": "1400px", "paddingTop": "20px", "paddingBottom": "40px"},
    children=[
        _build_header(),
        dbc.Row(
            [
                dbc.Col(
                    _build_sidebar(),
                    xs=12, md=4, lg=3,
                    style={"marginBottom": "16px"},
                ),
                dbc.Col(
                    _build_main_area(),
                    xs=12, md=8, lg=9,
                ),
            ],
        ),
        html.Div(
            className="pb-footer",
            children=[
                html.Span("PaperBanana "),
                html.A(
                    "Paper",
                    href="https://arxiv.org/abs/2601.23265",
                    target="_blank",
                ),
                html.Span(" | "),
                html.A(
                    "GitHub",
                    href="https://github.com/dwzhu-pku/PaperBanana",
                    target="_blank",
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Pipeline execution (runs async pipeline in a thread-safe way)
# ---------------------------------------------------------------------------

# In-memory store for background task results keyed by run_id.
_background_results: dict[str, dict] = {}


def _extract_final_image(result: dict, task_name: str, mode: str) -> str | None:
    """Extract the best final image (base64 JPEG) from a pipeline result dict."""
    if mode == "vanilla":
        return result.get(f"vanilla_{task_name}_base64_jpg")

    # Try critic rounds in reverse order (best refinement first)
    for r in range(5, -1, -1):
        key = f"target_{task_name}_critic_desc{r}_base64_jpg"
        if key in result and result[key]:
            return result[key]

    # Fall back to stylist or planner image
    if mode == "demo_full":
        key = f"target_{task_name}_stylist_desc0_base64_jpg"
    else:
        key = f"target_{task_name}_desc0_base64_jpg"
    return result.get(key)


def _run_generation(
    api_key: str,
    mode: str,
    task_type: str,
    aspect_ratio: str,
    num_candidates: int,
    max_critic_rounds: int,
    method_content: str,
    figure_caption: str,
    run_id: str,
):
    """Execute pipeline in a background thread, storing results when done."""
    from pipeline import run_batch

    data_list = []
    for i in range(num_candidates):
        data_list.append({
            "task_name": task_type,
            "content": method_content,
            "visual_intent": figure_caption,
            "filename": f"candidate_{i}",
            "additional_info": {"rounded_ratio": aspect_ratio},
        })

    async def _async_run():
        images = []
        async for result in run_batch(
            data_list,
            mode=mode,
            api_key=api_key,
            max_critic_rounds=max_critic_rounds,
        ):
            img_b64 = _extract_final_image(result, task_type, mode)
            if img_b64:
                images.append(img_b64)
        return images

    try:
        loop = asyncio.new_event_loop()
        images = loop.run_until_complete(_async_run())
        loop.close()
        _background_results[run_id] = {"status": "done", "images": images}
    except Exception as exc:
        logger.exception("Pipeline failed for run %s", run_id)
        _background_results[run_id] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "images": [],
        }


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("status-text", "children"),
    Output("generate-btn", "disabled"),
    Output("run-id-store", "data"),
    Output("poll-interval", "disabled"),
    Input("generate-btn", "n_clicks"),
    State("api-key-input", "value"),
    State("pipeline-mode", "value"),
    State("task-type", "value"),
    State("aspect-ratio", "value"),
    State("num-candidates", "value"),
    State("max-critic-rounds", "value"),
    State("method-content", "value"),
    State("figure-caption", "value"),
    prevent_initial_call=True,
)
def start_generation(
    n_clicks,
    api_key,
    mode,
    task_type,
    aspect_ratio,
    num_candidates,
    max_critic_rounds,
    method_content,
    figure_caption,
):
    """Validate inputs, launch background pipeline, enable polling."""
    if not api_key or not api_key.strip():
        return (
            "Error: Please provide a Google API Key in the settings sidebar.",
            False,
            no_update,
            True,
        )

    if not method_content or not method_content.strip():
        return (
            "Error: Method Content is required. Paste your methodology text.",
            False,
            no_update,
            True,
        )

    if not figure_caption or not figure_caption.strip():
        return (
            "Error: Figure Caption is required. Describe the desired figure.",
            False,
            no_update,
            True,
        )

    num_candidates = max(1, min(10, int(num_candidates or 1)))
    max_critic_rounds = max(0, min(5, int(max_critic_rounds or 3)))

    run_id = uuid.uuid4().hex
    _background_results[run_id] = {"status": "running", "images": []}

    import threading
    thread = threading.Thread(
        target=_run_generation,
        args=(
            api_key.strip(),
            mode,
            task_type,
            aspect_ratio,
            num_candidates,
            max_critic_rounds,
            method_content.strip(),
            figure_caption.strip(),
            run_id,
        ),
        daemon=True,
    )
    thread.start()

    status_content = html.Span([
        html.Span(className="pb-spinner"),
        f"Generating {num_candidates} candidate(s) with {mode} pipeline...",
    ])

    return status_content, True, run_id, False


@callback(
    Output("results-gallery", "children"),
    Output("results-store", "data"),
    Output("status-text", "children", allow_duplicate=True),
    Output("generate-btn", "disabled", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    State("run-id-store", "data"),
    prevent_initial_call=True,
)
def poll_results(n_intervals, run_id):
    """Poll background task and update gallery when done."""
    if not run_id or run_id not in _background_results:
        return no_update, no_update, no_update, no_update, no_update

    result = _background_results[run_id]

    if result["status"] == "running":
        return no_update, no_update, no_update, no_update, no_update

    # Task finished -- stop polling and clean up
    _background_results.pop(run_id, None)

    if result["status"] == "error":
        error_msg = result.get("error", "Unknown error occurred.")
        return (
            html.Div(
                className="pb-empty-state",
                children=f"Generation failed: {error_msg}. Check your API key and try again.",
            ),
            [],
            f"Error: {error_msg}. Check your API key and settings, then try again.",
            False,
            True,
        )

    # Success
    images = result.get("images", [])

    if not images:
        return (
            html.Div(
                className="pb-empty-state",
                children="Generation completed but no images were produced. Try a different pipeline mode or input.",
            ),
            [],
            "Completed, but no images were generated. Try different settings.",
            False,
            True,
        )

    # Build gallery cards
    cards = []
    for idx, img_b64 in enumerate(images):
        data_uri = f"data:image/jpeg;base64,{img_b64}"
        cards.append(
            html.Div(
                className="pb-candidate-card",
                children=[
                    html.P(
                        f"Candidate {idx + 1}",
                        className="pb-candidate-label",
                    ),
                    html.Img(
                        src=data_uri,
                        alt=f"Generated candidate {idx + 1}",
                        style={"maxWidth": "100%", "height": "auto"},
                        width=400,
                    ),
                    html.A(
                        "Download",
                        href=data_uri,
                        download=f"paperbanana_candidate_{idx + 1}.jpg",
                        className="pb-download-btn",
                        role="button",
                        **{"aria-label": f"Download candidate {idx + 1}"},
                    ),
                ],
            )
        )

    gallery = html.Div(className="pb-gallery", children=cards)
    count = len(images)
    status_msg = f"Done! Generated {count} candidate(s) successfully."

    return gallery, [True] * count, status_msg, False, True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
