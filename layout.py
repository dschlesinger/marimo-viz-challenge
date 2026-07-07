# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.2.2",
#     "huggingface-hub==1.22.0",
#     "marimo>=0.23.13",
#     "numpy==2.5.1",
#     "pandas==3.0.3",
#     "plotly==6.8.0",
#     "torch==2.12.1",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import math, time, os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    pass
    return F, device, go, make_subplots, math, mo, nn, np, os, pd, time, torch


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # TRMs Practice the Socratic Method better with Randomness

    To preface this notebook I must define some terms and explain the frame of thinking required for PTRMs.

    Vocabulary

    | Phrase | Meaning |
    |---|---|
    | Answer Space/Token Space | Text in most cases, the modality the answer is in |
    | Latent Space | A space of vectors which encode meaning |
    | LLM | A large transformer model with many unique layers |
    | TRM | A smaller models that repitivley updates an answer |
    | Deterministic | Maps one input to one output, no randomness involved |

    **Reasoning** - The repitive update of ones beliefs through challenging assumptions. As per Socrates: define, challenge with counter examples, edit and repeat!
    """)
    return


@app.cell(hide_code=True)
def explaining_latent_answer_space(mo):
    _answering_in_latent_space = mo.md(r"""
    ### A quick note on how machines reason

    LLMs have many methods of reasoning. From chain of thought to internal circuts dedicated to proposing ideas$^1$. But these methods have many flaws. Chain of thought is expensive and has no way of correcting previous assumptions, a token writen cannot be unwritten. Additionally internal reasoning circuts do not scale to problems with a large number of steps. 

    TRMs are built specifically to reason. To do this they ditch one shot token outputs and switch to iteritivly updating a latent reasonning and answer space, reduce the model size drastically, and control for problem difficulty.

    Before we explore these improvements lets get a grasp on what progressing through a answer latent space looks like.

    1. [Towards a Mechanistic Understanding of Propositional
    Logical Reasoning in Large Language Models](https://arxiv.org/pdf/2601.04260)
    """)

    _answer_space_math_1 = mo.md(r"""2 + 2 = 4""")
    _latent_space_math_1 = None # 3D display

    _answering_in_latent_space
    return


@app.cell(hide_code=True)
def _(DEEP_SUP_MATH, Xev_math, amp, math_trm, np, torch):
    # --- Shared PCA basis for the math-TRM's latent space, fit once on the eval set (every problem) ---
    # x,y in every reasoning plot below are the SAME 2 principal components, so trajectories from
    # different equations/operators sit in one comparable coordinate system. The 3D "landscape" for
    # any one problem (below) is instead built from PTRM rollouts of that SPECIFIC problem, then
    # just projected through this shared basis.
    @torch.no_grad()
    def _collect_latents(m, X, steps):
        m.eval(); zH, zL = m.init_carry(X.shape[0])
        lat = []
        for _ in range(steps):
            with amp():
                zH, zL, _, _, _ = m.recur(zH, zL, X)
            lat.append(zH.float().mean(1))
        return torch.stack(lat, 1)                                   # [B,T,h]

    _lat = _collect_latents(math_trm, Xev_math, DEEP_SUP_MATH).reshape(-1, math_trm.h).cpu().numpy()

    MATH_PCA_MEAN = _lat.mean(0)
    _u, _s, _vt = np.linalg.svd(_lat - MATH_PCA_MEAN, full_matrices=False)
    MATH_PCA_COMPONENTS = _vt[:2]                                    # [2, h] top-2 principal directions

    def project_latent(z):
        """[..., h] pooled latent -> [..., 2] coords in the shared math-TRM PCA basis."""
        return (np.asarray(z) - MATH_PCA_MEAN) @ MATH_PCA_COMPONENTS.T

    def kde_surface(xy, z, xrange, yrange, grid_n=40, bw_frac=0.12):
        """Gaussian-kernel-weighted (Nadaraya-Watson) smoothing of scattered (xy, z) onto a grid:
        height = local weighted-average z, surfacecolor = local point density (a 'KDE-style' surface)."""
        gx = np.linspace(xrange[0], xrange[1], grid_n)
        gy = np.linspace(yrange[0], yrange[1], grid_n)
        gxx, gyy = np.meshgrid(gx, gy)
        bw_x = bw_frac * max(xrange[1] - xrange[0], 1e-9)
        bw_y = bw_frac * max(yrange[1] - yrange[0], 1e-9)
        dx = (gxx.reshape(-1, 1) - xy[None, :, 0]) / bw_x
        dy = (gyy.reshape(-1, 1) - xy[None, :, 1]) / bw_y
        w = np.exp(-0.5 * (dx ** 2 + dy ** 2))
        wsum = w.sum(1)
        density = (wsum / len(xy)).reshape(grid_n, grid_n)
        zsurf = (w @ z / np.clip(wsum, 1e-6, None)).reshape(grid_n, grid_n)
        return gxx, gyy, zsurf, density

    return kde_surface, project_latent


@app.cell(hide_code=True)
def _(
    amp,
    decode_eq,
    device,
    encode_eq,
    go,
    kde_surface,
    math_trm,
    np,
    project_latent,
    torch,
):
    # --- Run the math TRM on one equation, one reasoning step at a time ---
    def equation_to_tensor(a, b, op):
        return torch.tensor(encode_eq(a, b, op, None).astype(np.int64), device=device).unsqueeze(0)

    @torch.no_grad()
    def trm_trace(a, b, op, true_c, steps):
        """One equation -> list of per-step dicts: predicted eq string/answer, PCA xy, |pred - true_c|.
        One row per H-cycle sub-step (not just one row per outer supervision step), starting from
        the TRM's first H-cycle -- i.e. as soon as it has actually processed the problem."""
        X = equation_to_tensor(a, b, op)
        math_trm.eval()
        zH, zL = math_trm.init_carry(1)
        rows = []

        for _ in range(steps):
            with amp():
                zH, zL, logits, qh, qc, zH_steps, logits_steps = math_trm.recur_traced(zH, zL, X)
            for zh_i, lg_i in zip(zH_steps, logits_steps):
                pred = lg_i.argmax(-1)[0].cpu().numpy()
                pred_c = int((pred[6] - 2) * 10 + (pred[7] - 2))
                xy = project_latent(zh_i.float().mean(1).cpu().numpy())[0]
                rows.append({
                    "eq": decode_eq(pred),
                    "pred_c": pred_c,
                    "x": float(xy[0]), "y": float(xy[1]),
                    "dist": abs(pred_c - true_c),
                })
        return rows

    @torch.no_grad()
    def ptrm_rollout_landscape(a, b, op, true_c, steps, K=256, sigma=0.15, pad_frac=0.1):
        """PTRM: K noisy rollouts of ONE specific equation -> a local KDE landscape around where
        THIS problem's reasoning actually lives (not a global, every-problem landscape). The PCA
        axes are the shared basis, but the plotted range is clipped to this rollout cloud's extent
        (+ pad_frac padding) so the plot zooms into the area this problem actually explores."""
        X = equation_to_tensor(a, b, op).repeat(K, 1)
        math_trm.eval()
        zH, zL = math_trm.init_carry(K)
        xy_steps, dist_steps = [], []
        for _ in range(steps):
            if sigma > 0:
                zL = zL + sigma * torch.randn_like(zL)          # the PTRM line: noise into z_L each step
            with amp():
                zH, zL, logits, _, _ = math_trm.recur(zH, zL, X)
            pred = logits.argmax(-1)
            pred_c = (pred[:, 6] - 2) * 10 + (pred[:, 7] - 2)
            xy_steps.append(project_latent(zH.float().mean(1).cpu().numpy()))       # [K,2]
            dist_steps.append((pred_c - true_c).abs().float().cpu().numpy())        # [K]
        xy_all = np.concatenate(xy_steps, 0)                     # [K*steps, 2]
        dist_all = np.concatenate(dist_steps, 0)                 # [K*steps]

        pad_x = pad_frac * max(np.ptp(xy_all[:, 0]), 1e-6)
        pad_y = pad_frac * max(np.ptp(xy_all[:, 1]), 1e-6)
        xy_range = (
            (float(xy_all[:, 0].min() - pad_x), float(xy_all[:, 0].max() + pad_x)),
            (float(xy_all[:, 1].min() - pad_y), float(xy_all[:, 1].max() + pad_y)),
        )
        z_range = (0.0, float(max(dist_all.max(), 1e-6)))
        gxx, gyy, zsurf, density = kde_surface(xy_all, dist_all, xy_range[0], xy_range[1])
        return xy_range, z_range, gxx, gyy, zsurf, density

    # --- Animated 3D figure: x,y = PCA(latent), height = |predicted answer - true answer| ---
    # A "ball" descends into the reasoning landscape's valley as its guess converges on the
    # correct answer; Play/Pause + a slider scrub through steps.
    def make_reasoning_figure(rows, title, xy_range, z_range, surface_x, surface_y, surface_z, surface_density,
                               z_label="|prediction - answer|"):
        xs = [r["x"] for r in rows]; ys = [r["y"] for r in rows]; zs = [r["dist"] for r in rows]
        background = go.Surface(
            x=surface_x, y=surface_y, z=surface_z, surfacecolor=surface_density,
            colorscale="Blues", opacity=0.75, showscale=False,
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
        )
        path = go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]], mode="lines",
                             line=dict(color="#e8590c", width=5), showlegend=False, hoverinfo="skip")
        ball = go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]], mode="markers",
                             marker=dict(size=7, color="#e8590c"), showlegend=False,
                             text=[rows[0]["eq"]], hoverinfo="text")
        frames = [
            go.Frame(
                data=[go.Scatter3d(x=xs[:t + 1], y=ys[:t + 1], z=zs[:t + 1]),
                      go.Scatter3d(x=[xs[t]], y=[ys[t]], z=[zs[t]], text=[rows[t]["eq"]])],
                traces=[1, 2], name=str(t),
                layout=go.Layout(annotations=[dict(text=rows[t]["eq"], showarrow=False, x=0.5, y=1.1,
                                                    xref="paper", yref="paper", font=dict(size=16))]),
            )
            for t in range(len(rows))
        ]
        return go.Figure(
            data=[background, path, ball],
            frames=frames,
            layout=go.Layout(
                title=title, height=420,
                scene=dict(
                    xaxis=dict(title="PC1", range=list(xy_range[0])),
                    yaxis=dict(title="PC2", range=list(xy_range[1])),
                    zaxis=dict(title=z_label, range=list(z_range)),
                ),
                margin=dict(l=0, r=0, t=60, b=0),
                annotations=[dict(text=rows[0]["eq"], showarrow=False, x=0.5, y=1.1,
                                  xref="paper", yref="paper", font=dict(size=16))],
                updatemenus=[dict(
                    type="buttons", showactive=False, x=0.05, y=0.02, xanchor="left", yanchor="bottom",
                    buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, dict(frame=dict(duration=700, redraw=True),
                                              transition=dict(duration=300, easing="cubic-in-out"),
                                              fromcurrent=True)]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                    ],
                )],
                sliders=[dict(
                    x=0.15, y=0.02, len=0.8,
                    steps=[dict(method="animate", label=f"step {t}",
                                args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                           for t in range(len(rows))],
                )],
            ),
        )

    return make_reasoning_figure, ptrm_rollout_landscape, trm_trace


@app.cell(hide_code=True)
def _(
    DEEP_SUP_MATH,
    Xev_math,
    Xtr_math,
    Yev_math,
    Ytr_math,
    amp,
    math_trm,
    torch,
):
    # --- Pick one example per operator that actually shows a "wrong -> right" journey ---
    # Hand-picked round-number equations are often already correct after the first supervision
    # step (8 inner H/L-cycle refinements) -- this tiny task converges fast. Instead, search the
    # full equation pool (train+eval; this is just picking a demo, not evaluating generalization)
    # for the example per operator with the most total movement that still lands correct.
    @torch.no_grad()
    def _op_journey_examples():
        X_all = torch.cat([Xtr_math, Xev_math], 0)
        Y_all = torch.cat([Ytr_math, Yev_math], 0)
        true_c = (Y_all[:, 6] - 2) * 10 + (Y_all[:, 7] - 2)
        op_col = X_all[:, 2]
        math_trm.eval()
        zH, zL = math_trm.init_carry(X_all.shape[0])
        dist_steps = []
        for _ in range(DEEP_SUP_MATH):
            with amp():
                zH, zL, logits, _, _ = math_trm.recur(zH, zL, X_all)
            pred = logits.argmax(-1)
            pred_c = (pred[:, 6] - 2) * 10 + (pred[:, 7] - 2)
            dist_steps.append((pred_c - true_c).abs())
        dist_steps = torch.stack(dist_steps, 1)              # [N, T]

        picks = {}
        for tok, sym in {12: "+", 13: "-", 14: "*", 15: "/"}.items():
            idx_pool = (op_col == tok).nonzero(as_tuple=True)[0]
            d = dist_steps[idx_pool]                          # [n_op, T]
            finishes_correct = d[:, -1] == 0
            if finishes_correct.any():
                pool, score = idx_pool[finishes_correct], d[finishes_correct].sum(1)
            else:                                             # fallback: closest to correct at the end
                pool, score = idx_pool, -d[:, -1]
            best = pool[score.argmax()].item()
            a = int((X_all[best, 0] - 2) * 10 + (X_all[best, 1] - 2))
            b = int((X_all[best, 3] - 2) * 10 + (X_all[best, 4] - 2))
            picks[sym] = (a, b, int(true_c[best].item()))
        return picks

    MATH_JOURNEY_EXAMPLES = _op_journey_examples()
    return (MATH_JOURNEY_EXAMPLES,)


@app.cell(hide_code=True)
def latent_reasoning_4_examples(
    DEEP_SUP_MATH,
    MATH_JOURNEY_EXAMPLES,
    make_reasoning_figure,
    mo,
    ptrm_rollout_landscape,
    trm_trace,
):
    # --- One reasoning-in-latent-space demo per operator, laid out as a 2x2 grid ---
    # The 3D landscape under each ball is that ONE equation's own 256-rollout PTRM cloud, not a
    # global landscape shared across all problems.
    _cards = []
    for _op, (_a, _b, _c) in MATH_JOURNEY_EXAMPLES.items():
        _trace = trm_trace(_a, _b, _op, _c, DEEP_SUP_MATH)
        _xy_range, _z_range, _sx, _sy, _sz, _sd = ptrm_rollout_landscape(_a, _b, _op, _c, DEEP_SUP_MATH)
        _fig = make_reasoning_figure(
            _trace, "", _xy_range, _z_range, _sx, _sy, _sz, _sd,
        )
        _correct = "✅ correct" if _trace[-1]["pred_c"] == _c else f"❌ model said {_trace[-1]['pred_c']}"
        _panel = mo.md(
            f"{_a} {_op} {_b} = {_c}"
        )
        _cards.append(mo.vstack([_panel, _fig], align="center", gap=0))
    mo.vstack([
        mo.hstack([_cards[0], _cards[1]], gap=0, align="start", justify="center"),
        mo.hstack([_cards[2], _cards[3]], gap=0, align="start", justify="center"),
    ], gap=2)
    return


@app.cell(hide_code=True)
def custom_latent_example(mo):
    math_custom_a = mo.ui.number(10, 99, step=1, value=23, label="A")
    math_custom_b = mo.ui.number(2, 99, step=1, value=4, label="B")
    math_custom_op = mo.ui.dropdown(options=["+", "-", "*", "/"], value="*", label="Operator")
    math_custom_run = mo.ui.run_button(label="Visualize reasoning")
    mo.vstack([
        mo.md("**Try your own equation** (rules are still enforced — see message below if it's rejected)"),
        mo.hstack([math_custom_a, math_custom_op, math_custom_b, math_custom_run], gap=1, justify="start"),
    ])
    return math_custom_a, math_custom_b, math_custom_op, math_custom_run


@app.cell(hide_code=True)
def _(
    DEEP_SUP_MATH,
    make_reasoning_figure,
    math_custom_a,
    math_custom_b,
    math_custom_op,
    math_custom_run,
    mo,
    ptrm_rollout_landscape,
    trm_trace,
    validate_equation,
):
    mo.stop(not math_custom_run.value, mo.md("_Enter an equation above and click **Visualize reasoning**._"))

    _a, _b, _op = math_custom_a.value, math_custom_b.value, math_custom_op.value
    _ok, _msg, _c = validate_equation(_a, _b, _op)
    mo.stop(not _ok, mo.md(
        f"**Invalid equation — {_msg}**\n\n"
        f"Rules: two-digit operands (10-99) for `+`/`-`; a two-digit first operand with a "
        f"single-digit (2-9) second operand for `*`/`/`; the answer must land in [1, 99]."
    ))

    _trace = trm_trace(_a, _b, _op, _c, DEEP_SUP_MATH)
    _xy_range, _z_range, _sx, _sy, _sz, _sd = ptrm_rollout_landscape(_a, _b, _op, _c, DEEP_SUP_MATH)
    _fig = make_reasoning_figure(
        _trace, f"TRM reasoning: {_a} {_op} {_b} = ?", _xy_range, _z_range, _sx, _sy, _sz, _sd,
    )
    _correct = "✅ correct" if _trace[-1]["pred_c"] == _c else f"❌ model said {_trace[-1]['pred_c']}"
    _panel = mo.md(
        f"**Answer Space**\n\nQuestion — `{_a} {_op} {_b} = ?`\n\nTrue answer — `{_c}`\n\n{_correct}"
    )
    mo.hstack([_panel, _fig], widths=[1, 3], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Equations to try | Why |
    |---|---|
    | 23 * 4 | Perfect example of the TRM starting in a suboptimal location and improving |
    | 23 * 2 | Interesting example where the TRM is initially correct and then goes to the wrong answer  |
    | 20 - 10 | For easy examples the model quickly converges on the correct answer |
    | 25 / 5 | The model converges in a local minima 11 and cannot find a correct solution |

    Figure 1 - A TRM directs a latent answer vector which starts at the encoded problem and which can be decoded back to the answer. By training a TRM on basic math ( + , - , * , / ) we can visualize this process. The x and y axis are the priniciple compoments of the 256 PTRM roll outs (explained later). The z axis a KDE of the difference between the true answer and the decoded answer at each step in the PTRM roll outs. Giving us a pseudo loss landscape. The orange ball is the path the TRM takes through this landscape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does a TRM look like and how does it differ from an LLM?
    """)
    return


@app.cell(hide_code=True)
def trm_vs_transformer_viz(mo):
    with open("trm_vs_transformer.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # Render inside an isolated iframe
    mo.iframe(html=html_content, width="100%", height="100px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can think of each layer as a set of questions being asked. The the LLM early layers ask questions like "How many digits does each number have?", in later stages "What is a reasonable order of magnitude?". In TRMs there are less layers so fewer questions are asked. But they are asked repetitively allowing the model to refine it answer over time. For example a TRM can ask "Is digit 1 reasonable given digit 2?" and "Is digit 2 reasonable given digit 1?" over and over again. (This is reasoning!)

    Below is the code for a single TRM training step.

    ```python
    with torch.no_grad():
        for _H_step in range(self.config.H_cycles - 1):
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)

    for _L_step in range(self.config.L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
    z_H = self.L_level(z_H, z_L, **seq_info)

    answer = lm_head(z_H)
    lm_loss   = CategoricalCrossentropy(answer, ground_truth)

    q_logit   = q_head(z_H)
    q_loss    = BinaryCrossEntropy(q_logit, isCorrect(answer))

    loss = lm_loss + 0.5 * q_loss
    ```

    <small>Adapted from [TRM Code](https://github.com/samsungsailmontreal/tinyrecursivemodels) models/recursive_reasoning/trm.py lines 208-222</small>

    The model updates both the reasoning and answer latents a predefined number of times with out a gradient and then calculates the loss based on the next update. L_level is the body of TRM and can be Transformer of MLP-t$^2$ blocks

    The lm_head decodes the answer latent into the text/answer space. The q_head is important. For TRMs its used for Adaptive Computation Time. Problems like 20 - 10 vs 23 * 4 are different complexities and require different amounts of reasoning. To allocate a problem the required compute the q_head predicts whether the answer latent is correct. Not only does this save compute time but it can prevent a correct answer from being overwritten (prevents the model from over thinking).

    2. MLP-t inspired by [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601)
    """)
    return


@app.cell(hide_code=True)
def table_proof(mo):
    mo.md(r"""
    ## Proof by Results
    | Method                                       |  # Params | Sudoku (%) | Maze (%) |
    | :------------------------------------------- | --------: | ---------: | -------: |
    | **Chain-of-thought, pretrained**             |           |            |          |
    | DeepSeek R1                                  |      671B |        0.0 |      0.0 |
    | Claude 3.7 (8K)                              |         ? |        0.0 |      0.0 |
    | o3-mini-high                                 |         ? |        0.0 |      0.0 |
    | **Direct prediction, small-sample training** |           |            |          |
    | Direct Prediction                            |       27M |        0.0 |      0.0 |
    | **TRM-Att (Ours)**                           |        7M |       74.7 | **85.3** |
    | **TRM-MLP (Ours)**                           | 5M / 19M¹ |   **87.4** |      0.0$^3$ |

    Table 4. Test accuracy (%) on puzzle benchmarks (Sudoku-Extreme and Maze-Hard). Adapted from Figure 4/Table 4 of Less is More: Recursive Reasoning with Tiny Networks. Paper (arXiv:2510.04871)

    3. MLP-t fails on Maze due to its large grid size, 30x30 vs Sudoko's 9x9, which scales parameters quadratically and the fact Maze only has 1000 examples. Both of which contribute to overcapacity. Additionally Maze requires dynamic routing while MLP-t cannot weight based on a positions input.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Latent Traps Trap Deterministic TRMs

    Our TRM can get stuck similar to how optimizers can get stuck in local minima. Click through the examples below to see a visualization!
    """)
    return


@app.cell(hide_code=True)
def _(DEEP_SUP, TRMSolver, Xhd, np, trm):
    # --- Shared PCA basis for the Sudoku TRM's latent space, fit on the harder ("stumbles") slice ---
    _det = TRMSolver(trm, steps=DEEP_SUP).predict(Xhd)
    _lat = _det.reasoning_trace.reshape(-1, trm.h).cpu().numpy()      # [B*T, h]

    SUDOKU_PCA_MEAN = _lat.mean(0)
    _u, _s, _vt = np.linalg.svd(_lat - SUDOKU_PCA_MEAN, full_matrices=False)
    SUDOKU_PCA_COMPONENTS = _vt[:2]

    def project_latent_sudoku(z):
        """[..., h] pooled latent -> [..., 2] coords in the shared Sudoku-TRM PCA basis."""
        return (np.asarray(z) - SUDOKU_PCA_MEAN) @ SUDOKU_PCA_COMPONENTS.T

    return (project_latent_sudoku,)


@app.cell(hide_code=True)
def _(DEEP_SUP, TRMSolver, Xhd, Yhd, trm):
    # --- Find puzzles where the deterministic TRM gets trapped in a wrong basin ---
    # Prefer failures that are CLOSE to solved (high correct-cell count) -- the classic "stuck near
    # the right answer" basin, not just noisy garbage. bf16 autocast makes the model's output
    # slightly batch-size-dependent, and these borderline puzzles are exactly the ones close enough
    # to flip between "wrong" and "correct" depending on batch size -- so re-verify each candidate
    # with the SAME batch-of-1 call the visualization cell below will actually use, and only keep
    # ones that are still wrong under that call.
    _det = TRMSolver(trm, steps=DEEP_SUP).predict(Xhd)
    _n_correct = (_det.answer == Yhd).sum(-1)              # [B]
    _wrong_idx = (_n_correct < 81).nonzero(as_tuple=True)[0]
    _order = _wrong_idx[_n_correct[_wrong_idx].argsort(descending=True)].tolist()

    _top3 = []
    for _cand in _order:
        _single = TRMSolver(trm, steps=DEEP_SUP).predict(Xhd[_cand:_cand + 1])
        _nc = int((_single.answer[0] == Yhd[_cand]).sum().item())
        if _nc < 81:
            _top3.append((_cand, _nc))
            if len(_top3) == 3:
                break

    SUDOKU_TRAP_EXAMPLES = {
        f"Puzzle {_i + 1} — stuck at {_nc}/81 correct": _idx
        for _i, (_idx, _nc) in enumerate(_top3)
    }
    return (SUDOKU_TRAP_EXAMPLES,)


@app.cell(hide_code=True)
def _(SUDOKU_TRAP_EXAMPLES, mo):
    sudoku_trap_choice = mo.ui.dropdown(
        options=list(SUDOKU_TRAP_EXAMPLES.keys()),
        value=next(iter(SUDOKU_TRAP_EXAMPLES)),
        label="Choose a stuck puzzle",
    )
    sudoku_trap_choice
    return (sudoku_trap_choice,)


@app.cell(hide_code=True)
def _(go, make_subplots):
    # --- One figure, two synced subplots (grid heatmap | 3D landscape), ONE shared Play/Pause + ---
    # slider, so the grid and the ball animate through the same reasoning steps together.
    def make_trap_figure(grid_zs, grid_texts, box_lines, rows, title, xy_range, z_range,
                          surface_x, surface_y, surface_z, surface_density, z_label):
        xs = [r["x"] for r in rows]; ys = [r["y"] for r in rows]; zs = [r["dist"] for r in rows]

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]],
                             column_widths=[0.42, 0.58])

        fig.add_trace(go.Heatmap(
            z=grid_zs[0], text=grid_texts[0], texttemplate="%{text}", textfont=dict(size=14),
            colorscale=[[0, "#e03131"], [1, "#2f9e44"]], zmin=0, zmax=1,
            showscale=False, xgap=2, ygap=2,
        ), row=1, col=1)
        fig.add_trace(go.Surface(
            x=surface_x, y=surface_y, z=surface_z, surfacecolor=surface_density,
            colorscale="Blues", opacity=0.75, showscale=False,
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
        ), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]], mode="lines",
                                    line=dict(color="#e8590c", width=6), showlegend=False,
                                    hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]], mode="markers",
                                    marker=dict(size=7, color="#e8590c"), showlegend=False,
                                    text=[rows[0]["eq"]], hoverinfo="text"), row=1, col=2)

        frames = []
        for t in range(len(rows)):
            frames.append(go.Frame(
                data=[
                    go.Heatmap(z=grid_zs[t], text=grid_texts[t]),
                    go.Scatter3d(x=xs[:t + 1], y=ys[:t + 1], z=zs[:t + 1]),
                    go.Scatter3d(x=[xs[t]], y=[ys[t]], z=[zs[t]], text=[rows[t]["eq"]]),
                ],
                traces=[0, 2, 3], name=str(t),
                layout=go.Layout(annotations=[dict(text=rows[t]["eq"], showarrow=False, x=0.78, y=1.12,
                                                    xref="paper", yref="paper", font=dict(size=16))]),
            ))
        fig.frames = frames

        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, autorange="reversed", row=1, col=1)
        fig.update_layout(
            title=title, height=460,
            shapes=box_lines,
            scene=dict(
                xaxis=dict(title="PC1", range=list(xy_range[0])),
                yaxis=dict(title="PC2", range=list(xy_range[1])),
                zaxis=dict(title=z_label, range=list(z_range)),
            ),
            margin=dict(l=0, r=0, t=60, b=80),
            annotations=[dict(text=rows[0]["eq"], showarrow=False, x=0.78, y=1.12,
                              xref="paper", yref="paper", font=dict(size=16))],
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.05, y=-0.12, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=700, redraw=True),
                                          transition=dict(duration=300, easing="cubic-in-out"),
                                          fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                x=0.2, y=-0.12, len=0.75,
                steps=[dict(method="animate", label=f"step {t + 1}",
                            args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                       for t in range(len(rows))],
            )],
        )
        return fig

    return (make_trap_figure,)


@app.cell(hide_code=True)
def latent_trap_display(
    DEEP_SUP,
    PTRMSolver,
    SUDOKU_TRAP_EXAMPLES,
    TRMSolver,
    Xhd,
    Yhd,
    decode_tokens,
    kde_surface,
    make_trap_figure,
    np,
    project_latent_sudoku,
    sudoku_trap_choice,
    trm,
):
    # --- Deterministic trace: what the TRM's grid guess looks like at every H-cycle sub-step ---
    # (not just each of the 6 outer supervision steps), starting from its first H-cycle.
    _idx = SUDOKU_TRAP_EXAMPLES[sudoku_trap_choice.value]
    _Xone = Xhd[_idx:_idx + 1]
    _Yone = Yhd[_idx]                                                                  # [81]
    _true_grid = decode_tokens(_Yone).reshape(9, 9).cpu().numpy()

    _det = TRMSolver(trm, steps=DEEP_SUP).predict_traced(_Xone)
    _det_grids = decode_tokens(_det.answer_trace[0]).cpu().numpy().reshape(-1, 9, 9)   # [T,9,9]
    _det_lat = project_latent_sudoku(_det.reasoning_trace[0].cpu().numpy())            # [T,2]
    _det_correct = (_det.answer_trace[0] == _Yone).sum(-1).cpu().numpy()               # [T]

    # --- PTRM: 256 noisy rollouts of this SAME puzzle -> local landscape, height = #correct cells ---
    _ptrm = PTRMSolver(trm, K=256, steps=DEEP_SUP, sigma=0.15).predict(_Xone, full_rollouts=True)
    _roll_lat = _ptrm.extra["rollout_reasoning"][:, 0].cpu().numpy()                   # [K,T,h]
    _roll_ans = _ptrm.extra["rollout_answer_trace"][:, 0]                              # [K,T,81]
    _roll_correct = (_roll_ans == _Yone).sum(-1).cpu().numpy().astype(np.float64)      # [K,T]
    _roll_xy = project_latent_sudoku(_roll_lat.reshape(-1, trm.h))                     # [K*T,2]
    _roll_correct_flat = _roll_correct.reshape(-1)                                     # [K*T]

    _pad_x = 0.1 * max(np.ptp(_roll_xy[:, 0]), 1e-6)
    _pad_y = 0.1 * max(np.ptp(_roll_xy[:, 1]), 1e-6)
    _xy_range = (
        (float(_roll_xy[:, 0].min() - _pad_x), float(_roll_xy[:, 0].max() + _pad_x)),
        (float(_roll_xy[:, 1].min() - _pad_y), float(_roll_xy[:, 1].max() + _pad_y)),
    )
    _z_range = (0.0, 81.0)
    _gxx, _gyy, _zsurf, _density = kde_surface(_roll_xy, 81 - _roll_correct_flat, _xy_range[0], _xy_range[1])

    _rows = []
    for _t in range(_det_lat.shape[0]):
        _nc = int(_det_correct[_t])
        _d, _h = divmod(_t, trm.H_cycles)
        _rows.append({
            "x": float(_det_lat[_t, 0]), "y": float(_det_lat[_t, 1]),
            "dist": 81 - float(_nc), "eq": f"step {_d + 1}.{_h + 1}: {_nc}/81 correct",
        })

    # --- Correctness per step: green = matches solution, red = doesn't; digits overlaid ---
    def _grid_frame(g):
        z = (g == _true_grid).astype(float)
        text = np.where(g == 0, "", g.astype(str))
        return z, text

    _grid_zs, _grid_texts = zip(*(_grid_frame(g) for g in _det_grids))
    _box_lines = [
        dict(type="line", x0=_b - 0.5, x1=_b - 0.5, y0=-0.5, y1=8.5, line=dict(color="black", width=3))
        for _b in (3, 6)
    ] + [
        dict(type="line", x0=-0.5, x1=8.5, y0=_b - 0.5, y1=_b - 0.5, line=dict(color="black", width=3))
        for _b in (3, 6)
    ]

    make_trap_figure(
        _grid_zs, _grid_texts, _box_lines, _rows,
        f"{sudoku_trap_choice.value} — deterministic reasoning", _xy_range, _z_range,
        _gxx, _gyy, _zsurf, _density, "# correct cells (of 81)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Figure 2 - Similar to Figure 1 the landscape is generated by PTRM roll outs and the z axis is 81 - # of correct cells. The TRMs get close to the answer, in example 1 the TRM is only missing 1 cell which had previously been solved. But the TRMs are in latent traps, no PTRM succeeded without first escaping them.
    """)
    return


@app.cell
def _(mo):
    _ptrm_intro = mo.md(r"""## How does a PTRM escape latent traps?

    PTRMs adds gaussian noise to the reasoning latent z_L at every step.

    """).text

    _top_fade = mo.md("""
    ```python
    with torch.no_grad():
        for _H_step in range(self.config.H_cycles - 1):
    ```
    """).text

    _highlight = mo.md("""
    ```python
            z_L = z_L + torch.randn_like(z_L) * sigma # <-- Only change!
    ```
    """).text

    _bottom_fade = mo.md("""
    ```python
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
        
    for _L_step in range(self.config.L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
    z_H = self.L_level(z_H, z_L, **seq_info)

    answer = lm_head(z_H)
    lm_loss   = CategoricalCrossentropy(answer, ground_truth)

    q_logit   = q_head(z_H)
    q_loss    = BinaryCrossEntropy(q_logit, isCorrect(answer))

    loss = lm_loss + 0.5 * q_loss
    ```
    """).text

    _style = '<style>.ptrms-block pre { margin: 0 !important; } .ptrms-block p { margin: 0 !important; }</style>'

    _ptrms_solve_this = mo.Html(
        _style + '<div class="ptrms-block">'
        + _ptrm_intro
        + '<div style="opacity: 0.7;">' + _top_fade + '</div>'
        + _highlight
        + '<div style="opacity: 0.7;">' + _bottom_fade + '</div>'
        + '</div>'
    )
    _ptrms_solve_this
    return


@app.cell
def _(mo):
    _what_ptrm_param = mo.md(r"""Since the PTRM is non-deterministic. Meaning the same output will give different results. We can run the PTRM any number of times and choose the best results, this is the number of roll outs previously mentioned.""")

    _what_ptrm_param
    return


@app.cell(hide_code=True)
def _(device, mo, np, os, pd, time, torch):
    # --- sudoku-extreme: EASY training slice + matched eval + a HARDER demo slice ---
    # Shared kernel: public names are distinct (Xtr/Ytr/Xev/Yev/Xhd/Yhd) to avoid sibling-notebook clashes.
    from huggingface_hub import hf_hub_download

    VOCAB, SEQ_LEN = 11, 81                        # token = digit+1: blank(0)->1, 1..9->2..10, token 0 = PAD (unused)
    EASY_MAX = 3                                   # train/eval on the easy end of sudoku-extreme (model can master it)
    HARD_MIN, HARD_MAX = 4, 6                      # harder demo slice: model stumbles -> recoverable bad-basin errors
    N_TR, N_EV, N_HD = 40_000, 2_000, 1_024

    def encode_grid(s):
        """81-char puzzle/solution string ('.'/'0' = blank) -> int8 token array (digit+1)."""
        b = np.frombuffer(str(s).zfill(81).encode("ascii"), dtype=np.uint8).astype(np.int16) - 48
        b[(b < 1) | (b > 9)] = 0                    # '.' (46-48) and '0' -> blank
        return (b + 1).astype(np.int8)

    def decode_tokens(t):
        """token array/tensor -> digits 0-9 (0 = blank). Inverse of encode_grid's +1."""
        if isinstance(t, torch.Tensor):
            return (t - 1).clamp(0, 9)
        return (np.asarray(t) - 1).clip(0, 9)

    def _load_band(fname, rmin, rmax, n, seed):
        path = hf_hub_download("sapientinc/sudoku-extreme", fname, repo_type="dataset")
        df = pd.read_csv(path, usecols=["question", "answer", "rating"],
                         dtype={"question": str, "answer": str, "rating": int})
        df = df[(df["rating"] >= rmin) & (df["rating"] <= rmax)]
        df = df.sample(n=min(n, len(df)), random_state=seed)
        P = np.stack([encode_grid(q) for q in df["question"].to_numpy()])
        S = np.stack([encode_grid(a) for a in df["answer"].to_numpy()])
        return P, S, df["rating"].to_numpy()

    _cache = "sudoku_extreme_easyhard_v2.npz"
    if os.path.exists(_cache):
        _d = np.load(_cache)
        trP, trS, evP, evS, hdP, hdS = _d["trP"], _d["trS"], _d["evP"], _d["evS"], _d["hdP"], _d["hdS"]
        tr_rating, ev_rating, hd_rating = _d["tr_rating"], _d["ev_rating"], _d["hd_rating"]
    else:
        _t0 = time.time()
        trP, trS, tr_rating = _load_band("train.csv", 0, EASY_MAX, N_TR, 0)        # easy train
        evP, evS, ev_rating = _load_band("test.csv", 0, EASY_MAX, N_EV, 1)         # easy eval (in-distribution)
        hdP, hdS, hd_rating = _load_band("test.csv", HARD_MIN, HARD_MAX, N_HD, 2)  # harder demo slice
        np.savez(_cache, trP=trP, trS=trS, evP=evP, evS=evS, hdP=hdP, hdS=hdS,
                 tr_rating=tr_rating, ev_rating=ev_rating, hd_rating=hd_rating)
        print(f"built easy/hard subset in {time.time() - _t0:.1f}s")

    Xtr = torch.tensor(trP.astype(np.int64), device=device)
    Ytr = torch.tensor(trS.astype(np.int64), device=device)
    Xev = torch.tensor(evP.astype(np.int64), device=device)
    Yev = torch.tensor(evS.astype(np.int64), device=device)
    Xhd = torch.tensor(hdP.astype(np.int64), device=device)      # harder puzzles for the basin demo
    Yhd = torch.tensor(hdS.astype(np.int64), device=device)

    _ge = float((trP != 1).sum(1).mean()); _gh = float((hdP != 1).sum(1).mean())
    mo.md(
        f"**Dataset — `sapientinc/sudoku-extreme`.** Easy train/eval (rating ≤ {EASY_MAX}, ~{_ge:.0f} givens): "
        f"{len(Xtr):,} / {len(Xev):,}. Harder demo slice (rating {HARD_MIN}–{HARD_MAX}, ~{_gh:.0f} givens): "
        f"{len(Xhd):,}. Tokens `digit+1`, vocab={VOCAB}, seq_len={SEQ_LEN}."
    )
    mo.output.clear()  # hidden for now
    return Xev, Xhd, Xtr, Yev, Yhd, Ytr, decode_tokens


@app.cell(hide_code=True)
def _(F, device, math, nn, torch):
    # --- Architectures: a recursive TRM (MLP-T) and a one-pass attention Transformer ---
    # Helper blocks are private (leading underscore) so they stay cell-local and don't collide
    # with the sibling notebook's graph. Only TRMNet, SudokuTransformer, and amp are public.
    def _mult256(a, b=256):
        return (-(a // -b)) * b

    def _rms_norm(x, eps=1e-5):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x.to(d)

    def amp():
        """bf16 autocast on GPU (no-op on CPU) — shared by training and inference."""
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")

    class _SwiGLU(nn.Module):
        def __init__(self, dim, expansion):
            super().__init__()
            inter = _mult256(round(expansion * dim * 2 / 3))
            self.gate_up = nn.Linear(dim, inter * 2, bias=False)
            self.down = nn.Linear(inter, dim, bias=False)
        def forward(self, x):
            g, u = self.gate_up(x).chunk(2, dim=-1)
            return self.down(F.silu(g) * u)

    # ---- TRM: depth via repetition (MLP-T mixing block) ----
    class _BlockMLPT(nn.Module):
        """Token-mixing SwiGLU across the 81 cells, then channel SwiGLU, post-norm."""
        def __init__(self, seq_len, h, expansion):
            super().__init__()
            self.tok = _SwiGLU(seq_len, expansion)
            self.mlp = _SwiGLU(h, expansion)
        def forward(self, x):
            x = x.transpose(1, 2)
            x = _rms_norm(x + self.tok(x))
            x = x.transpose(1, 2)
            return _rms_norm(x + self.mlp(x))

    class _ReasoningModule(nn.Module):
        def __init__(self, n_layers, seq_len, h, expansion):
            super().__init__()
            self.layers = nn.ModuleList([_BlockMLPT(seq_len, h, expansion) for _ in range(n_layers)])
        def forward(self, hidden, injection):
            hidden = hidden + injection
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

    class TRMNet(nn.Module):
        """Tiny Recursive Model: refine answer-latent z_H and reasoning-latent z_L by deep recursion."""
        def __init__(self, vocab=11, seq_len=81, h=256, expansion=4.0, L_layers=2, H_cycles=3, L_cycles=6):
            super().__init__()
            self.seq_len, self.h = seq_len, h
            self.H_cycles, self.L_cycles = H_cycles, L_cycles
            self.embed_scale = math.sqrt(h)
            self.embed = nn.Embedding(vocab, h)
            nn.init.trunc_normal_(self.embed.weight, std=1.0 / self.embed_scale)
            self.net = _ReasoningModule(L_layers, seq_len, h, expansion)
            self.lm_head = nn.Linear(h, vocab, bias=False)
            self.q_head = nn.Linear(h, 2, bias=True)            # (halt, continue) logits
            nn.init.zeros_(self.q_head.weight)
            nn.init.constant_(self.q_head.bias, -5.0)            # Q starts near 0 for stable bootstrapping
            self.H_init = nn.Parameter(torch.randn(h), requires_grad=False)
            self.L_init = nn.Parameter(torch.randn(h), requires_grad=False)

        def init_carry(self, B):
            zH = self.H_init.view(1, 1, -1).expand(B, self.seq_len, self.h).contiguous()
            zL = self.L_init.view(1, 1, -1).expand(B, self.seq_len, self.h).contiguous()
            return zH, zL

        def recur(self, zH, zL, x_tokens):
            """One deep-recursion step: H_cycles*L_cycles refinements, only the last carries grad."""
            emb = self.embed_scale * self.embed(x_tokens)
            with torch.no_grad():
                for _ in range(self.H_cycles - 1):
                    for _ in range(self.L_cycles):
                        zL = self.net(zL, zH + emb)
                    zH = self.net(zH, zL)
            for _ in range(self.L_cycles):
                zL = self.net(zL, zH + emb)
            zH = self.net(zH, zL)
            logits = self.lm_head(zH)
            q = self.q_head(zH[:, 0]).float()
            return zH, zL, logits, q[:, 0], q[:, 1]

        @torch.no_grad()
        def recur_traced(self, zH, zL, x_tokens):
            """Inference-only variant of recur(): identical computation, but also returns z_H --
            and its logit-lens readout -- after EVERY H-cycle, not just the final one. Lets the
            latent-trajectory visualizations show full H_cycles*L_cycles granularity instead of
            one point per supervision step."""
            emb = self.embed_scale * self.embed(x_tokens)
            zH_steps, logits_steps = [], []
            for _ in range(self.H_cycles):
                for _ in range(self.L_cycles):
                    zL = self.net(zL, zH + emb)
                zH = self.net(zH, zL)
                zH_steps.append(zH)
                logits_steps.append(self.lm_head(zH))
            q = self.q_head(zH[:, 0]).float()
            return zH, zL, logits_steps[-1], q[:, 0], q[:, 1], zH_steps, logits_steps

    # ---- Transformer: a standard bidirectional encoder (one-pass baseline) ----
    class _TFBlock(nn.Module):
        def __init__(self, h, n_heads, expansion):
            super().__init__()
            self.attn = nn.MultiheadAttention(h, n_heads, batch_first=True)
            self.mlp = _SwiGLU(h, expansion)
        def forward(self, x):
            a, _ = self.attn(x, x, x, need_weights=False)
            x = _rms_norm(x + a)
            return _rms_norm(x + self.mlp(x))

    class SudokuTransformer(nn.Module):
        """Plain bidirectional Transformer encoder: puzzle tokens -> solution logits in one pass."""
        def __init__(self, vocab=11, seq_len=81, h=256, n_layers=6, n_heads=8, expansion=4.0):
            super().__init__()
            self.h = h
            self.embed_scale = math.sqrt(h)
            self.embed = nn.Embedding(vocab, h)
            nn.init.trunc_normal_(self.embed.weight, std=1.0 / self.embed_scale)
            self.pos = nn.Parameter(torch.randn(1, seq_len, h) * 0.02)
            self.blocks = nn.ModuleList([_TFBlock(h, n_heads, expansion) for _ in range(n_layers)])
            self.lm_head = nn.Linear(h, vocab, bias=False)
            self.q_head = nn.Linear(h, 2, bias=True)
            nn.init.zeros_(self.q_head.weight)
            nn.init.constant_(self.q_head.bias, -5.0)

        def forward(self, x_tokens, collect=False):
            x = self.embed_scale * self.embed(x_tokens) + self.pos
            hs = []
            for blk in self.blocks:
                x = blk(x)
                if collect:
                    hs.append(x)
            logits = self.lm_head(x)
            q = self.q_head(x.mean(1)).float()
            return logits, q[:, 0], q[:, 1], hs


    return TRMNet, amp


@app.cell(hide_code=True)
def _(Xtr, Ytr, torch):
    # --- Label-preserving Sudoku augmentation (digit relabel + band/row/col/stack perms + transpose) ---
    def _axis_perm(B, dev):
        """Random valid 9-element row/col permutation: shuffle the 3 bands and the 3 lines within each."""
        band_order = torch.argsort(torch.rand(B, 3, device=dev), dim=1)            # [B,3] order of the 3 bands
        within = torch.argsort(torch.rand(B, 3, 3, device=dev), dim=2)             # [B,3,3] order within each band
        chosen = torch.gather(within, 1, band_order[:, :, None].expand(B, 3, 3))   # within-orders for the chosen bands
        return (band_order[:, :, None] * 3 + chosen).reshape(B, 9)                 # [B,9] source indices

    def augment_batch(Xtok, Ytok):
        """Apply one random Sudoku symmetry jointly to puzzle+solution token grids [B,81] -> [B,81]."""
        B, dev = Xtok.shape[0], Xtok.device
        Xg, Yg = Xtok.view(B, 9, 9), Ytok.view(B, 9, 9)

        # 1) digit relabel: permute digit tokens 2..10; blank token 1 (and PAD 0) stay fixed
        perm = torch.argsort(torch.rand(B, 9, device=dev), dim=1) + 2              # [B,9] permutation of tokens 2..10
        lut = torch.empty(B, 11, dtype=torch.long, device=dev)
        lut[:, 0] = 0
        lut[:, 1] = 1
        lut.scatter_(1, torch.arange(2, 11, device=dev).expand(B, 9), perm)
        Xg = torch.gather(lut, 1, Xg.reshape(B, 81)).view(B, 9, 9)
        Yg = torch.gather(lut, 1, Yg.reshape(B, 81)).view(B, 9, 9)

        # 2) permute rows (within/among bands) and columns (within/among stacks), same map for puzzle+solution
        rows = _axis_perm(B, dev)
        cols = _axis_perm(B, dev)
        Xg = torch.gather(Xg, 1, rows[:, :, None].expand(B, 9, 9))
        Yg = torch.gather(Yg, 1, rows[:, :, None].expand(B, 9, 9))
        Xg = torch.gather(Xg, 2, cols[:, None, :].expand(B, 9, 9))
        Yg = torch.gather(Yg, 2, cols[:, None, :].expand(B, 9, 9))

        # 3) random transpose (per sample)
        t = (torch.rand(B, device=dev) < 0.5)[:, None, None]
        Xg = torch.where(t, Xg.transpose(1, 2), Xg)
        Yg = torch.where(t, Yg.transpose(1, 2), Yg)
        return Xg.reshape(B, 81), Yg.reshape(B, 81)

    # sanity: augmentation preserves the puzzle/solution relationship and digit multiset
    _xa, _ya = augment_batch(Xtr[:4].clone(), Ytr[:4].clone())
    _ok = ((_xa == 1) | (_xa == _ya)).all().item()    # every given still matches the (relabeled) solution
    return (augment_batch,)


@app.cell(hide_code=True)
def _(amp, decode_tokens, torch):
    # --- Unified solver interface: predict() -> SolveResult with reasoning/answer/Q traces ---
    from dataclasses import dataclass, field

    @dataclass
    class SolveResult:
        """Unified solver output. Tokens are digit+1 (blank->1, 1..9->2..10);
        use decode_tokens() / .digits() for human-readable 0-9 grids."""
        answer: object                       # [B, 81]    final predicted tokens
        answer_trace: object                 # [B, T, 81] predicted tokens at each reasoning step
        reasoning_trace: object              # [B, T, h]  pooled latent (z_H / hidden) at each step
        q_trace: object = None               # [B, T]     Q halt logit per step, or None
        step_kind: str = "step"              # what one step means: "supervision" | "layer"
        kind: str = "?"                      # solver name
        extra: dict = field(default_factory=dict)

        def correct(self, Y):                # Y in token space [B,81] -> [B] bool exact-match
            return (self.answer == Y).all(-1)
        def digits(self):                    # final answer as 0-9 grid(s)
            return decode_tokens(self.answer)
        @property
        def n_steps(self):
            return self.answer_trace.shape[1]

    class TRMSolver:
        """Deterministic TRM: D supervision steps, no noise."""
        kind = "trm"
        def __init__(self, model, steps=6):
            self.model, self.steps = model, steps
        @torch.no_grad()
        def predict(self, X, steps=None):
            m = self.model; m.eval()
            D = steps or self.steps
            zH, zL = m.init_carry(X.shape[0])
            ans, lat, qs = [], [], []
            for _ in range(D):
                with amp():
                    zH, zL, logits, qh, qc = m.recur(zH, zL, X)
                ans.append(logits.argmax(-1)); lat.append(zH.float().mean(1)); qs.append(qh)
            return SolveResult(answer=ans[-1], answer_trace=torch.stack(ans, 1),
                               reasoning_trace=torch.stack(lat, 1), q_trace=torch.stack(qs, 1),
                               step_kind="supervision", kind="trm")

        @torch.no_grad()
        def predict_traced(self, X, steps=None):
            """Like predict(), but exposes every internal H-cycle (not just one point per outer
            supervision step) -- starting from the TRM's first H-cycle, i.e. as soon as it has
            actually processed the puzzle."""
            m = self.model; m.eval()
            D = steps or self.steps
            zH, zL = m.init_carry(X.shape[0])

            ans, lat = [], []
            for _ in range(D):
                with amp():
                    zH, zL, _, _, _, zH_steps, logits_steps = m.recur_traced(zH, zL, X)
                for zh_i, lg_i in zip(zH_steps, logits_steps):
                    ans.append(lg_i.argmax(-1))
                    lat.append(zh_i.float().mean(1))
            return SolveResult(answer=ans[-1], answer_trace=torch.stack(ans, 1),
                               reasoning_trace=torch.stack(lat, 1),
                               step_kind="H-cycle", kind="trm")

    class PTRMSolver:
        """PTRM: K parallel rollouts of the same TRM, Gaussian noise sigma into z_L each step,
        select the rollout with the highest final Q. Inference-time only — no retraining."""
        kind = "ptrm"
        def __init__(self, model, K=64, steps=6, sigma=0.15, max_par=8192):
            self.model, self.K, self.steps, self.sigma, self.max_par = model, K, steps, sigma, max_par
        @torch.no_grad()
        def predict(self, X, K=None, steps=None, sigma=None, full_rollouts=False):
            m = self.model; m.eval()
            K = K or self.K; D = steps or self.steps
            sigma = self.sigma if sigma is None else sigma
            B = X.shape[0]
            # chunk over puzzles so the K*chunk effective batch stays within memory
            chunk = max(1, self.max_par // K)
            if B > chunk:
                parts = [self.predict(X[i:i + chunk], K=K, steps=D, sigma=sigma,
                                      full_rollouts=full_rollouts) for i in range(0, B, chunk)]
                extra = dict(K=K, sigma=sigma,
                             best_k=torch.cat([p.extra["best_k"] for p in parts]),
                             rollout_answers=torch.cat([p.extra["rollout_answers"] for p in parts], 1),
                             rollout_q=torch.cat([p.extra["rollout_q"] for p in parts], 1))
                if full_rollouts:
                    extra["rollout_reasoning"] = torch.cat([p.extra["rollout_reasoning"] for p in parts], 1)
                    extra["rollout_q_step"] = torch.cat([p.extra["rollout_q_step"] for p in parts], 1)
                    extra["rollout_answer_trace"] = torch.cat([p.extra["rollout_answer_trace"] for p in parts], 1)
                return SolveResult(
                    answer=torch.cat([p.answer for p in parts]),
                    answer_trace=torch.cat([p.answer_trace for p in parts]),
                    reasoning_trace=torch.cat([p.reasoning_trace for p in parts]),
                    q_trace=torch.cat([p.q_trace for p in parts]),
                    step_kind="supervision", kind="ptrm", extra=extra)
            Xr = X.repeat(K, 1)
            zH, zL = m.init_carry(K * B)
            ans, lat, qs = [], [], []
            for _ in range(D):
                if sigma > 0:
                    zL = zL + sigma * torch.randn_like(zL)        # the one line that makes TRM probabilistic
                with amp():
                    zH, zL, logits, qh, qc = m.recur(zH, zL, Xr)
                ans.append(logits.argmax(-1).view(K, B, -1))      # [K,B,81]
                lat.append(zH.float().mean(1).view(K, B, -1))     # [K,B,h]
                qs.append(qh.view(K, B))                          # [K,B]
            A = torch.stack(ans, 2)            # [K,B,T,81]
            Lt = torch.stack(lat, 2)           # [K,B,T,h]
            Q = torch.stack(qs, 2)             # [K,B,T]
            best_k = Q[..., -1].argmax(0)      # [B] choose rollout by final Q (the PTRM rule)
            bidx = torch.arange(B, device=X.device)
            sel_ans, sel_lat, sel_q = A[best_k, bidx], Lt[best_k, bidx], Q[best_k, bidx]
            extra = dict(K=K, sigma=sigma, best_k=best_k,
                         rollout_answers=A[:, :, -1],   # [K,B,81] each rollout's final answer
                         rollout_q=Q[..., -1])          # [K,B] each rollout's final Q
            if full_rollouts:                           # full per-rollout latent path + Q-per-step (for the basin demo)
                extra["rollout_reasoning"] = Lt         # [K,B,T,h]
                extra["rollout_q_step"] = Q             # [K,B,T]
                extra["rollout_answer_trace"] = A       # [K,B,T,81] full per-step predicted tokens
            return SolveResult(answer=sel_ans[:, -1], answer_trace=sel_ans, reasoning_trace=sel_lat,
                               q_trace=sel_q, step_kind="supervision", kind="ptrm", extra=extra)

    class TransformerSolver:
        """One-pass attention encoder. Trace is a logit-lens read-out: one step = one layer."""
        kind = "transformer"
        def __init__(self, model):
            self.model = model
        @torch.no_grad()
        def predict(self, X):
            m = self.model; m.eval()
            with amp():
                logits, qh, qc, hs = m(X, collect=True)               # hs: list[n_layers] of [B,81,h]
                ans = torch.stack([m.lm_head(h).argmax(-1) for h in hs], 1)              # [B,L,81]
                lat = torch.stack([h.float().mean(1) for h in hs], 1)                   # [B,L,h]
                q = torch.stack([m.q_head(h.mean(1)).float()[:, 0] for h in hs], 1)     # [B,L]
                answer = logits.argmax(-1)
            return SolveResult(answer=answer, answer_trace=ans, reasoning_trace=lat,
                               q_trace=q, step_kind="layer", kind="transformer")

    def make_solver(kind, model, **kw):
        """Factory: make_solver('trm'|'ptrm'|'transformer', model, **kw) -> solver."""
        return {"trm": TRMSolver, "ptrm": PTRMSolver, "transformer": TransformerSolver}[kind](model, **kw)

    return PTRMSolver, TRMSolver


@app.cell(hide_code=True)
def _(
    F,
    TRMNet,
    Xev,
    Xtr,
    Yev,
    Ytr,
    amp,
    augment_batch,
    device,
    os,
    time,
    torch,
):
    # --- Train (quick-demo budget) or load cached checkpoints ---
    TRM_CFG = dict(vocab=11, seq_len=81, h=256, expansion=4.0, L_layers=2, H_cycles=3, L_cycles=6)
    DEEP_SUP = 6                                   # supervision steps for TRM (train + default inference depth)

    @torch.no_grad()
    def _eval_exact_trm(m, D):
        m.eval(); zH, zL = m.init_carry(Xev.shape[0])
        for _ in range(D):
            with amp():
                zH, zL, lg, _, _ = m.recur(zH, zL, Xev)
        return (lg.argmax(-1) == Yev).all(-1).float().mean().item()

    def fit_trm(m, steps=400, bs=512, nsup=DEEP_SUP, lr=1e-3, augment=True):
        """Train a TRMNet with deep supervision (Q-head learns correctness). Returns history dict."""
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
        hist = {"step": [], "loss": [], "eval_exact": []}
        m.train()
        for step in range(1, steps + 1):
            idx = torch.randint(0, Xtr.shape[0], (bs,), device=device)
            X, Y = Xtr[idx], Ytr[idx]
            if augment:
                X, Y = augment_batch(X, Y)
            zH, zL = m.init_carry(bs); opt.zero_grad(); sl = 0.0
            for _ in range(nsup):                                  # deep supervision
                zH, zL = zH.detach(), zL.detach()
                with amp():
                    zH, zL, logits, qh, qc = m.recur(zH, zL, X)
                    loss = F.cross_entropy(logits.float().reshape(-1, 11), Y.reshape(-1))
                    seqc = (logits.argmax(-1) == Y).all(-1).float()
                    qloss = F.binary_cross_entropy_with_logits(qh, seqc)
                (loss + 0.5 * qloss).backward(); sl += loss.item()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
            if step % 25 == 0:
                hist["step"].append(step); hist["loss"].append(sl / nsup)
                hist["eval_exact"].append(_eval_exact_trm(m, nsup)); m.train()
        return hist

    torch.manual_seed(0)
    trm = TRMNet(**TRM_CFG).to(device)

    _trm_ckpt = "trm_easy.pt"
    if os.path.exists(_trm_ckpt):
        _c = torch.load(_trm_ckpt, map_location=device, weights_only=False)
        trm.load_state_dict(_c["sd"]); trm_history = _c["hist"]
    else:
        _t0 = time.time(); trm_history = fit_trm(trm, steps=1200)
        print(f"trained TRM in {time.time() - _t0:.1f}s")
        torch.save({"sd": trm.state_dict(), "hist": trm_history}, _trm_ckpt)

    # trm.eval()
    # _np_trm = sum(p.numel() for p in trm.parameters()) / 1e6
    # mo.md(f"**Trained.** TRM ({_np_trm:.2f}M) best exact **{max(trm_history['eval_exact']):.1%}** · ")
    return DEEP_SUP, trm


@app.cell(hide_code=True)
def _(device, np, torch):
    VOCAB_MATH, SEQ_LEN_MATH = 17, 8
    OP_PLUS, OP_MINUS, OP_MUL, OP_DIV, EQ_TOK, BLANK_TOK = (
        12,
        13,
        14,
        15,
        16,
        1,
    )
    _OP_TOK = {"+": OP_PLUS, "-": OP_MINUS, "*": OP_MUL, "/": OP_DIV}
    _TOK_OP = {v: k for k, v in _OP_TOK.items()}

    def encode_eq(a, b, op, c):
        """a: two-digit int. b: two-digit int (+/-) or single-digit 2-9 (*//).
        op: '+'/'-'/'*'/'/' . c: two-digit answer, or None to blank it. -> token row [8]."""
        c_tens, c_ones = (
            (BLANK_TOK, BLANK_TOK) if c is None else (c // 10 + 2, c % 10 + 2)
        )
        return np.array(
            [
                a // 10 + 2,
                a % 10 + 2,
                _OP_TOK[op],
                b // 10 + 2,
                b % 10 + 2,
                EQ_TOK,
                c_tens,
                c_ones,
            ],
            dtype=np.int8,
        )

    def decode_eq(row):
        """Inverse of encode_eq: token row [8] -> 'A op B = C' string."""
        row = np.asarray(row)
        a = (row[0] - 2) * 10 + (row[1] - 2)
        b = (row[3] - 2) * 10 + (row[4] - 2)
        c = (row[6] - 2) * 10 + (row[7] - 2)
        return f"{a} {_TOK_OP[int(row[2])]} {b} = {c}"

    def validate_equation(a, b, op):
        """Enforce the same rules used to build the dataset. -> (ok, error_message, answer_or_None)."""
        if not (10 <= a <= 99):
            return False, "A must be a two-digit number (10-99).", None
        if op in ("+", "-"):
            if not (10 <= b <= 99):
                return False, f"B must be a two-digit number (10-99) for '{op}'.", None
            c = a + b if op == "+" else a - b
            if not (10 <= c <= 99):
                return False, f"{a} {op} {b} = {c}, which isn't a two-digit positive answer (10-99).", None
            return True, "", c
        if not (2 <= b <= 9):
            return False, (f"B must be a single digit (2-9) for '{op}' — two two-digit operands can't "
                            f"give a two-digit product/quotient."), None
        if op == "*":
            c = a * b
            if not (10 <= c <= 99):
                return False, f"{a} * {b} = {c}, which isn't a two-digit positive answer (10-99).", None
            return True, "", c
        if a % b != 0:
            return False, f"{a} / {b} isn't a whole number.", None
        c = a // b
        if not (1 <= c <= 99):
            return False, f"{a} / {b} = {c}, which is out of the allowed answer range (1-99).", None
        return True, "", c

    def _all_equations():
        """Every valid equation per op, grouped so the train/eval split can be stratified —
        * and / are inherently rare (single-digit second operand) next to +/-'s much larger space."""
        by_op = {"+": [], "-": [], "*": [], "/": []}
        for a in range(10, 100):
            for b in range(10, 100):
                c = a + b
                if 10 <= c <= 99:
                    by_op["+"].append((a, b, c))
                c = a - b
                if 10 <= c <= 99:
                    by_op["-"].append((a, b, c))
            for b in range(2, 10):
                c = a * b
                if 10 <= c <= 99:
                    by_op["*"].append((a, b, c))
                    by_op["*"].append((b, a, c))
                if a % b == 0:
                    c = a // b
                    if 1 <= c <= 99:
                        by_op["/"].append((a, b, c))
        return by_op

    _by_op = _all_equations()
    _rng = np.random.default_rng(0)
    _eval_frac = 0.10
    _tr_rows, _ev_rows = [], []
    for _op, _triples in _by_op.items():
        _idx = _rng.permutation(len(_triples))
        _n_ev = max(1, round(len(_triples) * _eval_frac))
        for _i in _idx[_n_ev:]:
            a, b, c = _triples[_i]
            _tr_rows.append(
                (encode_eq(a, b, _op, None), encode_eq(a, b, _op, c))
            )
        for _i in _idx[:_n_ev]:
            a, b, c = _triples[_i]
            _ev_rows.append(
                (encode_eq(a, b, _op, None), encode_eq(a, b, _op, c))
            )
    _rng.shuffle(_tr_rows)
    _rng.shuffle(_ev_rows)

    Xtr_math = torch.tensor(
        np.stack([r[0] for r in _tr_rows]).astype(np.int64), device=device
    )
    Ytr_math = torch.tensor(
        np.stack([r[1] for r in _tr_rows]).astype(np.int64), device=device
    )
    Xev_math = torch.tensor(
        np.stack([r[0] for r in _ev_rows]).astype(np.int64), device=device
    )
    Yev_math = torch.tensor(
        np.stack([r[1] for r in _ev_rows]).astype(np.int64), device=device
    )

    # mo.md(
    #     f"**Dataset — two-digit `+ - * /` with a two-digit positive answer.** "
    #     f"{len(Xtr_math):,} train / {len(Xev_math):,} eval equations "
    #     + ", ".join(f"{op}:{len(t)}" for op, t in _by_op.items())
    #     + f". Tokens: digit+2, vocab={VOCAB_MATH}, seq_len={SEQ_LEN_MATH}."
    # )
    return (
        Xev_math,
        Xtr_math,
        Yev_math,
        Ytr_math,
        decode_eq,
        encode_eq,
        validate_equation,
    )


@app.cell(hide_code=True)
def _(
    F,
    TRMNet,
    Xev_math,
    Xtr_math,
    Yev_math,
    Ytr_math,
    amp,
    device,
    os,
    time,
    torch,
):
    # --- Train the arithmetic TRM: same TRMNet class as Sudoku, sized for an 8-token sequence ---
    # so this fits comfortably in an RTX 3050 Ti's 4GB and trains in well under 2 minutes.
    MATH_CFG = dict(vocab=17, seq_len=8, h=64, expansion=4.0, L_layers=2, H_cycles=2, L_cycles=4)
    DEEP_SUP_MATH = 4                              # supervision steps for the math TRM
    _OP_SYMS = {12: "+", 13: "-", 14: "*", 15: "/"}

    @torch.no_grad()
    def _eval_exact_math(m, D, per_op=False):
        m.eval(); zH, zL = m.init_carry(Xev_math.shape[0])
        for _ in range(D):
            with amp():
                zH, zL, lg, _, _ = m.recur(zH, zL, Xev_math)
        correct = (lg.argmax(-1) == Yev_math).all(-1)
        if not per_op:
            return correct.float().mean().item()
        op_col = Xev_math[:, 2]
        return {sym: correct[op_col == tok].float().mean().item() for tok, sym in _OP_SYMS.items()
                if (op_col == tok).any()}

    def fit_trm_math(m, steps=400, bs=512, nsup=DEEP_SUP_MATH, lr=1e-3):
        """Train a TRMNet on the arithmetic task with deep supervision. Returns history dict."""
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
        hist = {"step": [], "loss": [], "eval_exact": []}
        m.train()
        for step in range(1, steps + 1):
            idx = torch.randint(0, Xtr_math.shape[0], (bs,), device=device)
            X, Y = Xtr_math[idx], Ytr_math[idx]
            zH, zL = m.init_carry(bs); opt.zero_grad(); sl = 0.0
            for _ in range(nsup):
                zH, zL = zH.detach(), zL.detach()
                with amp():
                    zH, zL, logits, qh, qc = m.recur(zH, zL, X)
                    loss = F.cross_entropy(logits.float().reshape(-1, MATH_CFG["vocab"]), Y.reshape(-1))
                    seqc = (logits.argmax(-1) == Y).all(-1).float()
                    qloss = F.binary_cross_entropy_with_logits(qh, seqc)
                (loss + 0.5 * qloss).backward(); sl += loss.item()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
            if step % 25 == 0:
                hist["step"].append(step); hist["loss"].append(sl / nsup)
                hist["eval_exact"].append(_eval_exact_math(m, nsup)); m.train()
        return hist

    torch.manual_seed(0)
    math_trm = TRMNet(**MATH_CFG).to(device)

    _math_ckpt = "trm_math_v2.pt"                 # v2: vocab grew (15->17) to add '*' and '/' tokens
    if os.path.exists(_math_ckpt):
        _c = torch.load(_math_ckpt, map_location=device, weights_only=False)
        math_trm.load_state_dict(_c["sd"]); math_history = _c["hist"]
    else:
        _t0 = time.time(); math_history = fit_trm_math(math_trm, steps=400)
        print(f"trained math TRM in {time.time() - _t0:.1f}s")
        torch.save({"sd": math_trm.state_dict(), "hist": math_history}, _math_ckpt)

    math_trm.eval()
    _np_math = sum(p.numel() for p in math_trm.parameters()) / 1e6
    _per_op = _eval_exact_math(math_trm, DEEP_SUP_MATH, per_op=True)
    # mo.md(
    #     f"**Trained.** Arithmetic TRM ({_np_math:.3f}M params) best exact **{max(math_history['eval_exact']):.1%}** overall — "
    #     + ", ".join(f"`{op}` {acc:.1%}" for op, acc in _per_op.items())
    # )
    return DEEP_SUP_MATH, math_trm


if __name__ == "__main__":
    app.run()
