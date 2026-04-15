"""PaperBananaLight — academic diagram generation, editorial interface."""

import asyncio
import logging
import os
import uuid

import dash
from dash import Input, Output, State, callback, dcc, html, no_update
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_KEY_FROM_ENV = os.environ.get("GOOGLE_API_KEY", "").strip()
_HAS_ENV_KEY = bool(_API_KEY_FROM_ENV)

EXAMPLE_METHOD = r"""## Methodology: The PaperBanana Framework

In this section, we present the architecture of PaperBanana, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperBanana orchestrates a collaborative team of five specialized agents---Retriever, Planner, Stylist, Visualizer, and Critic---to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents.
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$

### Planner Agent

The Planner Agent serves as the cognitive core of the system:
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

ASPECT_OPTIONS = [
    {"label": "16:9", "value": "16:9"},
    {"label": "21:9", "value": "21:9"},
    {"label": "3:2", "value": "3:2"},
]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css",
    ],
    title="PaperBananaLight",
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "color-scheme", "content": "light"},
        {"name": "theme-color", "content": "#FBF8F1"},
    ],
)

server = app.server

app.index_string = """<!DOCTYPE html>
<html data-theme="light" lang="en">
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _header():
    return html.Header(
        className="pb-header",
        children=[
            html.Div(
                className="pb-title-group",
                children=[
                    html.H1("PaperBananaLight", className="pb-header-title"),
                    html.P(
                        "Multi-agent academic diagram generation",
                        className="pb-header-sub",
                    ),
                ],
            ),
        ],
    )


def _api_key_section():
    if _HAS_ENV_KEY:
        return html.Div(className="pb-api-configured", children=[
            html.Label("API Key", htmlFor="api-key-input"),
            html.Div(className="pb-api-status", children=[
                html.Span("Configured via .env", className="pb-api-badge-ok"),
            ]),
            dcc.Input(
                id="api-key-input",
                type="hidden",
                value=_API_KEY_FROM_ENV,
            ),
        ])

    return html.Div(children=[
        html.Label("Google API Key", htmlFor="api-key-input"),
        dcc.Input(
            id="api-key-input",
            type="password",
            value="",
            placeholder="AIza...",
            autoComplete="off",
            className="pb-input",
        ),
        html.Small(
            "Required. Set GOOGLE_API_KEY in .env to skip this.",
            className="pb-helper-text",
        ),
    ])


def _sidebar():
    return html.Article(children=[
        html.H2("Configuration", className="pb-section-label"),

        _api_key_section(),

        html.Hr(),

        # Aspect ratio
        html.Label("Aspect Ratio", htmlFor="aspect-ratio"),
        dcc.Dropdown(
            id="aspect-ratio",
            value="16:9",
            options=ASPECT_OPTIONS,
            clearable=False,
            className="pb-dropdown",
        ),

        # Candidates
        html.Label("Candidates", htmlFor="num-candidates"),
        dcc.Slider(
            id="num-candidates",
            min=1,
            max=10,
            step=1,
            value=1,
            marks={i: str(i) for i in range(1, 11)},
            className="pb-slider",
        ),

        # Critic rounds
        html.Label(
            ["Critic Rounds: ", html.Output(id="critic-rounds-value", children="3")],
            htmlFor="max-critic-rounds",
        ),
        dcc.Slider(
            id="max-critic-rounds",
            min=0,
            max=5,
            step=1,
            value=3,
            marks={i: str(i) for i in range(6)},
            className="pb-slider",
        ),
    ])


def _main():
    return html.Article(
        className="pb-main-card",
        children=[
            html.H2("Input", className="pb-section-label"),

            html.Label("Method Content", htmlFor="method-content"),
            dcc.Textarea(
                id="method-content",
                value=EXAMPLE_METHOD,
                placeholder="Paste your methodology section here...",
                rows=12,
                className="pb-textarea",
            ),

            html.Label("Figure Caption", htmlFor="figure-caption"),
            dcc.Textarea(
                id="figure-caption",
                value=EXAMPLE_CAPTION,
                placeholder="Describe the diagram you want to generate...",
                rows=4,
                className="pb-textarea",
            ),

            html.Button(
                "Generate Candidates",
                id="generate-btn",
                className="pb-generate-btn",
            ),

            html.P(
                id="status-text",
                className="pb-status",
                children="Ready.",
                **{"aria-busy": "false"},
            ),

            html.Hr(),

            html.H2("Results", className="pb-section-label"),
            html.Div(
                id="results-gallery",
                children=html.P(
                    "Your generated diagrams will appear here.",
                    className="pb-empty-state",
                ),
            ),

            # Stores
            dcc.Store(id="results-store", data=[]),
            dcc.Store(id="run-id-store", data=""),
            dcc.Interval(id="poll-interval", interval=2000, n_intervals=0, disabled=True),
        ],
    )


app.layout = html.Div(children=[
    _header(),
    html.Div(
        className="pb-layout",
        children=[
            html.Aside(className="pb-sidebar", children=[_sidebar()]),
            html.Section(children=[_main()]),
        ],
    ),
    html.Footer(
        className="pb-footer",
        children=html.Small([
            "PaperBananaLight  ",
            html.A("Paper", href="https://arxiv.org/abs/2601.23265", target="_blank"),
            "  |  ",
            html.A("GitHub", href="https://github.com/dwzhu-pku/PaperBanana", target="_blank"),
        ]),
    ),
])


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

_background_results: dict[str, dict] = {}

PIPELINE_MODE = "demo_full"
TASK_TYPE = "diagram"


def _extract_final_image(result: dict) -> str | None:
    for r in range(5, -1, -1):
        key = f"target_diagram_critic_desc{r}_base64_jpg"
        if key in result and result[key]:
            return result[key]
    return result.get("target_diagram_stylist_desc0_base64_jpg")


def _run_generation(api_key, aspect_ratio, num_candidates, max_critic_rounds,
                    method_content, figure_caption, run_id):
    from pipeline import run_batch

    data_list = [
        {
            "task_name": TASK_TYPE,
            "content": method_content,
            "visual_intent": figure_caption,
            "filename": f"candidate_{i}",
            "additional_info": {"rounded_ratio": aspect_ratio},
        }
        for i in range(num_candidates)
    ]

    async def _go():
        images = []
        async for result in run_batch(data_list, mode=PIPELINE_MODE, api_key=api_key,
                                      max_critic_rounds=max_critic_rounds):
            img = _extract_final_image(result)
            if img:
                images.append(img)
        return images

    try:
        loop = asyncio.new_event_loop()
        images = loop.run_until_complete(_go())
        loop.close()
        _background_results[run_id] = {"status": "done", "images": images}
    except Exception as exc:
        logger.exception("Pipeline failed: %s", run_id)
        _background_results[run_id] = {"status": "error", "error": str(exc), "images": []}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(Output("critic-rounds-value", "children"), Input("max-critic-rounds", "value"))
def update_slider(value):
    return str(value)


@callback(
    Output("status-text", "children"),
    Output("status-text", "className"),
    Output("status-text", "aria-busy"),
    Output("generate-btn", "disabled"),
    Output("run-id-store", "data"),
    Output("poll-interval", "disabled"),
    Input("generate-btn", "n_clicks"),
    State("api-key-input", "value"),
    State("aspect-ratio", "value"),
    State("num-candidates", "value"),
    State("max-critic-rounds", "value"),
    State("method-content", "value"),
    State("figure-caption", "value"),
    prevent_initial_call=True,
)
def start_generation(n_clicks, api_key, aspect_ratio, num_candidates,
                     max_critic_rounds, method_content, figure_caption):
    err = lambda msg: (msg, "pb-status pb-status-error", "false", False, no_update, True)

    if not api_key or not api_key.strip():
        return err("Enter your Google API Key in the sidebar.")
    if not method_content or not method_content.strip():
        return err("Paste your methodology text above.")
    if not figure_caption or not figure_caption.strip():
        return err("Describe the diagram you want to generate.")

    num_candidates = max(1, min(10, int(num_candidates or 1)))
    max_critic_rounds = max(0, min(5, int(max_critic_rounds or 3)))
    run_id = uuid.uuid4().hex

    _background_results[run_id] = {"status": "running", "images": []}

    import threading
    threading.Thread(
        target=_run_generation,
        args=(api_key.strip(), aspect_ratio, num_candidates,
              max_critic_rounds, method_content.strip(), figure_caption.strip(), run_id),
        daemon=True,
    ).start()

    return (
        f"Generating {num_candidates} candidate(s)...",
        "pb-status pb-status-busy",
        "true",
        True,
        run_id,
        False,
    )


@callback(
    Output("results-gallery", "children"),
    Output("results-store", "data"),
    Output("status-text", "children", allow_duplicate=True),
    Output("status-text", "className", allow_duplicate=True),
    Output("status-text", "aria-busy", allow_duplicate=True),
    Output("generate-btn", "disabled", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    State("run-id-store", "data"),
    prevent_initial_call=True,
)
def poll_results(n_intervals, run_id):
    noop = (no_update,) * 7

    if not run_id or run_id not in _background_results:
        return noop
    result = _background_results[run_id]
    if result["status"] == "running":
        return noop

    _background_results.pop(run_id, None)

    if result["status"] == "error":
        msg = result.get("error", "Unknown error")
        return (
            html.P(f"Failed: {msg}. Check API key and retry.", className="pb-empty-state"),
            [], f"Error: {msg}", "pb-status pb-status-error", "false", False, True,
        )

    images = result.get("images", [])
    if not images:
        return (
            html.P("No diagrams produced. Try different input.", className="pb-empty-state"),
            [], "Done, but no diagrams generated.", "pb-status", "false", False, True,
        )

    cards = []
    for i, img_b64 in enumerate(images):
        uri = f"data:image/jpeg;base64,{img_b64}"
        cards.append(
            html.Article(
                className="pb-candidate-card",
                children=[
                    html.H3(f"Candidate {i + 1}", className="pb-candidate-label"),
                    html.Img(src=uri, alt=f"Candidate {i + 1}", width=400, height=225,
                             style={"width": "100%", "height": "auto"}),
                    html.A("Download", href=uri, download=f"paperbanana_{i + 1}.jpg",
                           className="pb-download-btn",
                           **{"aria-label": f"Download candidate {i + 1}"}),
                ],
            )
        )

    n = len(images)
    return (
        html.Div(className="pb-gallery", children=cards),
        [True] * n,
        f"Done! {n} candidate(s) generated.",
        "pb-status",
        "false",
        False,
        True,
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
