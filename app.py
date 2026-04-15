"""PaperBanana Light — academic diagram generation, editorial interface."""

import asyncio
import logging
import os
import threading
import uuid

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, _dash_renderer, callback, dcc, html, no_update
from dash_iconify import DashIconify
from dotenv import load_dotenv

_dash_renderer._set_react_version("18.2.0")
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_KEY_FROM_ENV = os.environ.get("GOOGLE_API_KEY", "").strip()
_HAS_ENV_KEY = bool(_API_KEY_FROM_ENV)

EXAMPLE_METHOD = r"""## How To Use This Input

Use this field to provide the scientific content that should ground the figure generation process. In most cases, a paper abstract, an introduction-style summary, or a concise methodology section is sufficient to communicate the core components, workflow, and scientific intent of the desired illustration.

### Recommended Input Scope

For best results, we recommend providing a focused description of the method rather than the entire paper. A shorter and more structured input usually leads to clearer diagrams because the system can more easily infer the relevant entities, stages, and relationships. You may also paste a complete paper in `.tex` format if you are willing to spend more on tokens, but this can introduce irrelevant details, ambiguity, or unintended interpretations. This is currently a practical hypothesis rather than a formally validated limitation.

You can also include LaTeX equations and notation directly, for example:
$$
z = f_\theta(x), \qquad \hat{y} = g_\phi(z)
$$
This may help communicate the mathematical structure of the method when such notation is central to the figure.

### Practical Guidance

When possible, restrict the content to the main pipeline, model components, data flow, or algorithmic stages that should appear in the figure. If the paper contains multiple contributions, narrowing the scope is often preferable to asking for a single diagram that explains everything at once.
"""

EXAMPLE_CAPTION = (
    "Figure 1: Overview of our extended PaperBanana pipeline. Compared with the "
    "original PaperBanana framework, our system replaces long-context example "
    "conditioning with a retrieval-augmented generation (RAG) stage for reference "
    "selection and injection. This design retrieves only the most relevant examples "
    "for the target paper, reducing token usage and overall cost while preserving "
    "high-quality diagram planning, styling, visualization, and iterative critique."
)

ASPECT_OPTIONS = [
    {"label": "16:9", "value": "16:9"},
    {"label": "21:9", "value": "21:9"},
    {"label": "3:2", "value": "3:2"},
]

CANDIDATES_MARKS = [{"value": i, "label": str(i)} for i in range(1, 11)]
CRITIC_MARKS = [{"value": i, "label": str(i)} for i in range(6)]

# ---------------------------------------------------------------------------
# Theme — warm editorial amber on paper
# ---------------------------------------------------------------------------

THEME = {
    "primaryColor": "amber",
    "primaryShade": 6,
    "defaultRadius": "md",
    "fontFamily": '"IBM Plex Sans", system-ui, -apple-system, sans-serif',
    "fontFamilyMonospace": '"IBM Plex Mono", monospace',
    "headings": {
        "fontFamily": '"Fraunces", Georgia, serif',
        "fontWeight": "700",
    },
    "colors": {
        "amber": [
            "#FDF6E9",
            "#F5EDD8",
            "#E8D9B4",
            "#D4B97A",
            "#C89938",
            "#C47A0A",
            "#B8720A",
            "#A86608",
            "#8B6914",
            "#5C4410",
        ],
    },
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    title="PaperBanana Light",
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "color-scheme", "content": "light"},
        {"name": "theme-color", "content": "#FBF8F1"},
    ],
)

server = app.server

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _header():
    return html.Header(
        className="pb-header",
        children=dmc.Stack(gap=2, children=[
            dmc.Title(
                ["PaperBanana ", html.Em("Light")],
                order=1,
                className="pb-header-title",
            ),
            dmc.Text(
                "Multi-agent academic diagram generation",
                className="pb-header-sub",
            ),
        ]),
    )


def _api_key_section():
    if _HAS_ENV_KEY:
        return dmc.Stack(gap=6, children=[
            dmc.Text("API Key", size="sm", fw=500),
            dmc.Badge(
                "Configured via .env",
                color="green",
                variant="light",
                radius="sm",
                size="md",
            ),
            dcc.Input(id="api-key-input", type="hidden", value=_API_KEY_FROM_ENV),
        ])
    return dmc.PasswordInput(
        id="api-key-input",
        label=dmc.Group(gap=6, align="center", children=[
            dmc.Text("Google API Key", size="sm", fw=500, lh=1),
            dmc.Tooltip(
                label="Set GOOGLE_API_KEY in .env to skip this.",
                position="right",
                withArrow=True,
                multiline=True,
                w=220,
                children=html.Span(
                    DashIconify(icon="tabler:info-circle", width=14, height=14),
                    style={
                        "cursor": "help",
                        "color": "#A69E90",
                        "display": "flex",
                        "alignItems": "center",
                        "lineHeight": 0,
                    },
                ),
            ),
        ]),
        placeholder="AIza...",
        description="Required.",
        autoComplete="off",
        value="",
    )


def _sidebar():
    return dmc.Paper(
        withBorder=True,
        shadow="sm",
        radius="md",
        p=24,
        className="pb-sidebar-card",
        children=dmc.Stack(gap="lg", children=[
            dmc.Text("Configuration", className="pb-section-label"),

            _api_key_section(),

            dmc.Divider(),

            dmc.Select(
                id="aspect-ratio",
                label="Aspect Ratio",
                value="16:9",
                data=ASPECT_OPTIONS,
                allowDeselect=False,
                clearable=False,
                styles={"label": {"marginBottom": 8}},
            ),

            dmc.Box(children=[
                dmc.Group(justify="space-between", mb=6, children=[
                    dmc.Text("Candidates", size="sm", fw=500),
                    dmc.Text(id="num-candidates-value", size="sm", fw=600, c="amber.7"),
                ]),
                dmc.Slider(
                    id="num-candidates",
                    min=1,
                    max=10,
                    step=1,
                    value=1,
                    marks=CANDIDATES_MARKS,
                    color="amber",
                    mb="xl",
                ),
            ]),

            dmc.Box(children=[
                dmc.Group(justify="space-between", mb=6, children=[
                    dmc.Text("Critic Rounds", size="sm", fw=500),
                    dmc.Text(id="critic-rounds-value", size="sm", fw=600, c="amber.7"),
                ]),
                dmc.Slider(
                    id="max-critic-rounds",
                    min=0,
                    max=5,
                    step=1,
                    value=3,
                    marks=CRITIC_MARKS,
                    color="amber",
                    mb="xl",
                ),
            ]),
        ]),
    )


def _status_alert(message, state="default"):
    colors = {"default": "gray", "busy": "amber", "error": "red"}
    return dmc.Alert(
        message,
        color=colors.get(state, "gray"),
        radius="md",
    )


def _empty_state(message):
    return dmc.Text(
        message,
        c="dimmed",
        ta="center",
        fs="italic",
        py=56,
    )


def _main():
    return dmc.Paper(
        withBorder=True,
        shadow="sm",
        radius="md",
        p=24,
        className="pb-main-card",
        children=dmc.Stack(gap="lg", children=[
            dmc.Text("Input", className="pb-section-label"),

            dmc.Textarea(
                id="method-content",
                label="Method Content",
                placeholder="Paste your methodology section here...",
                value=EXAMPLE_METHOD,
                styles={
                    "label": {"marginBottom": 8},
                    "input": {"height": 340, "resize": "vertical"},
                },
            ),

            html.Div(children=[
                dmc.Textarea(
                    id="figure-caption",
                    label=dmc.Group(
                        justify="space-between",
                        w="100%",
                        align="center",
                        children=[
                            dmc.Text("Figure Caption", size="sm", fw=500, lh=1),
                            dmc.Switch(
                                id="auto-suggest-switch",
                                label="Auto-suggest",
                                size="xs",
                                color="amber",
                                checked=False,
                                styles={"label": {"fontWeight": 400}},
                            ),
                        ],
                    ),
                    placeholder="Describe the diagram you want to generate...",
                    value=EXAMPLE_CAPTION,
                    minRows=4,
                    autosize=True,
                    maxRows=10,
                    styles={"label": {"width": "100%", "marginBottom": 8}},
                ),
                html.Div(
                    id="suggest-btn-container",
                    style={"display": "none", "marginTop": 8},
                    children=dmc.Group(gap="xs", grow=True, children=[
                        dmc.Button(
                            "Suggest caption",
                            id="suggest-caption-btn",
                            leftSection=DashIconify(icon="tabler:sparkles", width=16),
                            variant="light",
                            color="amber",
                            size="xs",
                        ),
                        dmc.Button(
                            "Accept",
                            id="accept-caption-btn",
                            leftSection=DashIconify(icon="tabler:check", width=16),
                            variant="filled",
                            color="amber",
                            size="xs",
                            disabled=True,
                        ),
                    ]),
                ),
            ]),

            dmc.Button(
                "Generate Candidates",
                id="generate-btn",
                fullWidth=True,
                size="md",
                variant="gradient",
                gradient={"from": "#D4860B", "to": "#B8720A", "deg": 135},
                mt="xs",
            ),

            html.Div(id="status-container"),

            dmc.Text("Results", className="pb-section-label"),

            html.Div(
                id="results-gallery",
                children=_empty_state("Your generated diagrams will appear here."),
            ),

            dcc.Store(id="results-store", data=[]),
            dcc.Store(id="run-id-store", data=""),
            dcc.Store(id="suggest-task-store", data=""),
            dcc.Interval(id="poll-interval", interval=2000, n_intervals=0, disabled=True),
            dcc.Interval(id="suggest-interval", interval=1000, n_intervals=0, disabled=True),
        ]),
    )


def _footer():
    return html.Footer(
        className="pb-footer",
        children=dmc.Text(size="xs", c="dimmed", children=[
            "PaperBanana ",
            html.Em("Light"),
            "  |  ",
            dmc.Anchor("GitHub", href="https://github.com/damarals/paperbanana-light", target="_blank"),
        ]),
    )


app.layout = dmc.MantineProvider(
    theme=THEME,
    forceColorScheme="light",
    children=html.Div([
        _header(),
        html.Div(
            className="pb-layout",
            children=[
                html.Aside(className="pb-sidebar", children=[_sidebar()]),
                html.Section(children=[_main()]),
            ],
        ),
        _footer(),
    ]),
)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

_background_results: dict[str, dict] = {}
_background_suggestions: dict[str, dict] = {}


def _run_suggest_bg(api_key: str, content: str, task_id: str) -> None:
    from pipeline import suggest_caption

    try:
        loop = asyncio.new_event_loop()
        caption = loop.run_until_complete(suggest_caption(content, api_key))
        loop.close()
        _background_suggestions[task_id] = {"status": "done", "caption": caption}
    except Exception as exc:
        logger.exception("Suggest caption failed: %s", task_id)
        _background_suggestions[task_id] = {"status": "error", "error": str(exc)}


def _extract_final_image(result: dict) -> str | None:
    for r in range(5, -1, -1):
        key = f"critic_image_{r}"
        if key in result and result[key]:
            return result[key]
    return result.get("stylist_image")


def _run_generation(api_key, aspect_ratio, num_candidates, max_critic_rounds,
                    method_content, figure_caption, run_id):
    from pipeline import run_batch

    data_list = [
        {
            "content": method_content,
            "visual_intent": figure_caption,
            "filename": f"candidate_{i}",
            "additional_info": {"rounded_ratio": aspect_ratio},
        }
        for i in range(num_candidates)
    ]

    async def _go():
        images = []
        async for result in run_batch(data_list, api_key=api_key,
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
def update_critic_rounds(value):
    return str(value)


@callback(Output("num-candidates-value", "children"), Input("num-candidates", "value"))
def update_num_candidates(value):
    return str(value)


@callback(
    Output("suggest-btn-container", "style"),
    Output("figure-caption", "placeholder"),
    Output("generate-btn", "disabled", allow_duplicate=True),
    Output("accept-caption-btn", "disabled", allow_duplicate=True),
    Output("suggest-caption-btn", "children", allow_duplicate=True),
    Input("auto-suggest-switch", "checked"),
    prevent_initial_call=True,
)
def toggle_auto_suggest(checked):
    if checked:
        return (
            {"display": "block", "marginTop": 8},
            "Click 'Suggest caption' to generate one from the methodology...",
            True,
            True,
            "Suggest caption",
        )
    return (
        {"display": "none", "marginTop": 8},
        "Describe the diagram you want to generate...",
        False,
        True,
        "Suggest caption",
    )


@callback(
    Output("suggest-caption-btn", "loading"),
    Output("suggest-task-store", "data"),
    Output("suggest-interval", "disabled"),
    Output("status-container", "children", allow_duplicate=True),
    Output("accept-caption-btn", "disabled", allow_duplicate=True),
    Input("suggest-caption-btn", "n_clicks"),
    State("api-key-input", "value"),
    State("method-content", "value"),
    prevent_initial_call=True,
)
def start_suggest(n_clicks, api_key, method_content):
    if not api_key or not api_key.strip():
        return False, no_update, True, _status_alert(
            "Enter your Google API Key in the sidebar.", "error"
        ), no_update
    if not method_content or not method_content.strip():
        return False, no_update, True, _status_alert(
            "Paste your methodology text first.", "error"
        ), no_update

    task_id = uuid.uuid4().hex
    _background_suggestions[task_id] = {"status": "running"}

    threading.Thread(
        target=_run_suggest_bg,
        args=(api_key.strip(), method_content.strip(), task_id),
        daemon=True,
    ).start()

    return True, task_id, False, None, True


@callback(
    Output("figure-caption", "value"),
    Output("suggest-caption-btn", "loading", allow_duplicate=True),
    Output("suggest-caption-btn", "children", allow_duplicate=True),
    Output("suggest-interval", "disabled", allow_duplicate=True),
    Output("status-container", "children", allow_duplicate=True),
    Output("accept-caption-btn", "disabled", allow_duplicate=True),
    Input("suggest-interval", "n_intervals"),
    State("suggest-task-store", "data"),
    prevent_initial_call=True,
)
def poll_suggest(n_intervals, task_id):
    noop = (no_update,) * 6
    if not task_id or task_id not in _background_suggestions:
        return noop
    result = _background_suggestions[task_id]
    if result["status"] == "running":
        return noop

    _background_suggestions.pop(task_id, None)

    if result["status"] == "error":
        msg = result.get("error", "Unknown error")
        return (
            no_update,
            False,
            "Suggest caption",
            True,
            _status_alert(f"Suggestion failed: {msg}", "error"),
            True,
        )

    return result["caption"], False, "Regenerate", True, None, False


@callback(
    Output("generate-btn", "disabled", allow_duplicate=True),
    Output("suggest-btn-container", "style", allow_duplicate=True),
    Input("accept-caption-btn", "n_clicks"),
    prevent_initial_call=True,
)
def accept_caption(n_clicks):
    if not n_clicks:
        return no_update, no_update
    return False, {"display": "none", "marginTop": 8}


@callback(
    Output("status-container", "children"),
    Output("generate-btn", "disabled"),
    Output("generate-btn", "loading"),
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
    def err(msg):
        return _status_alert(msg, "error"), False, False, no_update, True

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

    threading.Thread(
        target=_run_generation,
        args=(api_key.strip(), aspect_ratio, num_candidates,
              max_critic_rounds, method_content.strip(), figure_caption.strip(), run_id),
        daemon=True,
    ).start()

    return (
        None,
        True,
        True,
        run_id,
        False,
    )


@callback(
    Output("results-gallery", "children"),
    Output("results-store", "data"),
    Output("status-container", "children", allow_duplicate=True),
    Output("generate-btn", "disabled", allow_duplicate=True),
    Output("generate-btn", "loading", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    State("run-id-store", "data"),
    prevent_initial_call=True,
)
def poll_results(n_intervals, run_id):
    noop = (no_update,) * 6

    if not run_id or run_id not in _background_results:
        return noop
    result = _background_results[run_id]
    if result["status"] == "running":
        return noop

    _background_results.pop(run_id, None)

    if result["status"] == "error":
        msg = result.get("error", "Unknown error")
        return (
            _empty_state("Failed. Check API key and retry."),
            [],
            _status_alert(f"Error: {msg}", "error"),
            False, False, True,
        )

    images = result.get("images", [])
    if not images:
        return (
            _empty_state("No diagrams produced. Try different input."),
            [],
            _status_alert("Done, but no diagrams generated.", "default"),
            False, False, True,
        )

    cards = []
    for i, img_b64 in enumerate(images):
        uri = f"data:image/jpeg;base64,{img_b64}"
        cards.append(
            dmc.Card(
                withBorder=True,
                radius="md",
                shadow="sm",
                padding="md",
                children=dmc.Stack(gap="sm", align="center", children=[
                    dmc.Image(src=uri, alt=f"Candidate {i + 1}", radius="sm"),
                    dmc.Title(f"Candidate {i + 1}", order=3, className="pb-candidate-label"),
                    html.A(
                        "Download",
                        href=uri,
                        download=f"paperbanana_{i + 1}.jpg",
                        className="pb-download-btn",
                        **{"aria-label": f"Download candidate {i + 1}"},
                    ),
                ]),
            )
        )

    n = len(images)
    return (
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 2, "lg": 3},
            spacing="lg",
            children=cards,
        ),
        [True] * n,
        _status_alert(f"Done! {n} candidate(s) generated.", "default"),
        False, False, True,
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
