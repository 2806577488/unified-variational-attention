"""
UVA 认知对话 — Web 调参台（Gradio）。

用法（在项目根目录）::

    pip install -r requirements-web.txt
    python dialogue_web_ui.py
    python dialogue_web_ui.py --port 7860 --share

在浏览器中：填写分词器 / 可选对话 JSON 路径 →「加载模型」→ 调参 →
「单轮对话」或「内心一拍」查看输出；可对上一轮用 good / bad / meh 做偏好反馈。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

import gradio as gr

from uva_model.dialogue import (
    CognitiveDialogueAgent,
    DialogueTurn,
    InternalMonologue,
)
from uva_model.tokenizer import PrecisionTokenizer


ROOT = Path(__file__).resolve().parent


class SessionState(TypedDict, total=False):
    agent: CognitiveDialogueAgent
    log: str
    last_fb: Union[DialogueTurn, InternalMonologue, None]


def _default_tokenizer_path() -> str:
    p = ROOT / "tokenizer_zh_from_chunks_v2.json"
    if p.is_file():
        return str(p)
    p2 = ROOT / "tokenizer_zh_from_chunks_v2.small.json"
    if p2.is_file():
        return str(p2)
    return str(ROOT / "tokenizer_zh_from_chunks_v2.json")


def _feedback_reward(cmd: str) -> Optional[float]:
    low = cmd.strip().lower()
    if low == "good":
        return 1.0
    if low == "bad":
        return -1.0
    if low == "meh":
        return 0.0
    return None


def _format_turn(turn: DialogueTurn) -> str:
    it = turn.intent
    lines = [
        "### 对外轮",
        f"- **用户**: {turn.user_text!r}",
        f"- **模型**: {turn.reply}",
        f"- **分支**: `{turn.output_branch}`",
        f"- **best_efe**: {turn.best_efe:.4f}",
        f"- **ε_social_in / ε_secondary**: {turn.epsilon_social_in:.4f} / {turn.epsilon_secondary:.4f}",
        f"- **ε_imprint_max / 焦点**: {turn.semantic_surprise_max:.4f} / `{turn.conflict_focus_token or '—'}`",
        f"- **意图**: {it.speech_act}, anchor=`{it.anchor_slot}`, tension=`{it.tension_slot}`",
        f"- **u_c / u_t / π_s**: {turn.u_curiosity:.4f} / {turn.u_task:.4f} / {turn.pi_statement:.4f}",
        f"- **资源**: {turn.resource_snapshot}",
    ]
    return "\n".join(lines)


def _format_mono(mono: InternalMonologue) -> str:
    lines = [
        "### 内心一拍",
        f"- **reply**: {mono.reply}",
        f"- **trigger**: `{mono.trigger_token}`",
        f"- **tension / best_efe**: {mono.tension:.4f} / {mono.best_efe:.4f}",
        f"- **分支**: `{mono.output_branch}`",
        f"- **ε_secondary / ε_imprint_max**: {mono.epsilon_secondary:.4f} / {mono.semantic_surprise_max:.4f}",
    ]
    return "\n".join(lines)


def _append_log(state: SessionState, block: str) -> SessionState:
    prev = state.get("log", "")
    sep = "\n\n---\n\n" if prev.strip() else ""
    state["log"] = prev + sep + block
    return state


def load_model(
    tokenizer_path: str,
    dialogue_path: str,
    learn_tokenizer: bool,
    arbitration_ticket_scale: float,
    lambda_explore_0: float,
    conflict_surprise_threshold: float,
    conflict_surprise_gain: float,
    internal_tension_threshold: float,
    internal_pending_increment: float,
    internal_global_scan_k: float,
    internal_spontaneous_jitter: float,
    internal_rng_seed: str,
    branch_bias_weight: float,
    cognitive_debt_surprise_gamma: float,
    curiosity_pool_max: float,
    curiosity_per_event_cap: float,
    curiosity_refill_external: float,
) -> tuple[SessionState, str]:
    tok_path = tokenizer_path.strip()
    if not tok_path:
        return cast(SessionState, {}), "请填写分词器 JSON 路径。"
    p = Path(tok_path)
    if not p.is_file():
        return cast(SessionState, {}), f"分词器文件不存在: {p}"

    tok = PrecisionTokenizer.load_model(str(p.resolve()))
    seed: Optional[int] = None
    s = internal_rng_seed.strip()
    if s != "":
        try:
            seed = int(s)
        except ValueError:
            return cast(SessionState, {}), "内心随机种子须为整数或留空。"

    agent = CognitiveDialogueAgent(
        tok,
        learn_tokenizer_from_user=bool(learn_tokenizer),
        lambda_explore_0=float(lambda_explore_0),
        conflict_surprise_threshold=float(conflict_surprise_threshold),
        conflict_surprise_gain=float(conflict_surprise_gain),
        internal_tension_threshold=float(internal_tension_threshold),
        internal_pending_increment=float(internal_pending_increment),
        internal_global_scan_k=int(internal_global_scan_k),
        internal_spontaneous_jitter=float(internal_spontaneous_jitter),
        internal_rng_seed=seed,
        branch_bias_weight=float(branch_bias_weight),
        cognitive_debt_surprise_gamma=float(cognitive_debt_surprise_gamma),
        curiosity_injection_pool_max=float(curiosity_pool_max),
        curiosity_injection_per_event_cap=float(curiosity_per_event_cap),
        curiosity_injection_pool_refill_external=float(curiosity_refill_external),
    )
    agent.ARBITRATION_SOCIAL_TICKET_SCALE = float(arbitration_ticket_scale)

    dlg = dialogue_path.strip()
    if dlg:
        dp = Path(dlg)
        if not dp.is_file():
            return cast(SessionState, {}), f"对话状态文件不存在: {dp}"
        agent.load_dialogue_model(str(dp.resolve()))

    st: SessionState = {"agent": agent, "log": "", "last_fb": None}
    msg = f"已加载分词器: `{p}`"
    if dlg:
        msg += f"\n已加载对话状态: `{Path(dlg).resolve()}`"
    msg += f"\n分词器 fitted={tok.fitted}"
    return st, msg


def run_single_turn(state: SessionState, user_text: str) -> tuple[SessionState, str, str]:
    if not state or "agent" not in state:
        return cast(SessionState, {}), "", "请先点击「加载模型」。"
    text = (user_text or "").strip()
    if not text:
        return state, state.get("log", ""), "（用户输入为空）"
    agent = state["agent"]
    turn = agent.turn(text)
    state["last_fb"] = turn
    block = _format_turn(turn)
    _append_log(state, block)
    return state, state["log"], "本轮已完成。"


def run_internal_tick(state: SessionState) -> tuple[SessionState, str, str]:
    if not state or "agent" not in state:
        return cast(SessionState, {}), "", "请先点击「加载模型」。"
    agent = state["agent"]
    mono = agent.internal_tick()
    if mono is None:
        block = "### 内心一拍\n- （本拍无输出：未过阈或无候选）"
        state["last_fb"] = None
    else:
        state["last_fb"] = mono
        block = _format_mono(mono)
    _append_log(state, block)
    return state, state["log"], "内心一拍已完成。"


def apply_feedback_cmd(state: SessionState, cmd: str) -> tuple[SessionState, str, str]:
    if not state or "agent" not in state:
        return cast(SessionState, {}), "", "请先加载模型。"
    rew = _feedback_reward(cmd)
    if rew is None:
        return state, state.get("log", ""), "请输入 good、bad 或 meh。"
    fb = state.get("last_fb")
    if fb is None:
        return state, state.get("log", ""), "尚无上一轮可对齐的模型输出（先跑单轮对话或内心一拍）。"
    agent = state["agent"]
    if isinstance(fb, DialogueTurn):
        agent.apply_dialogue_feedback(rew, fb)
    else:
        agent.apply_internal_monologue_feedback(rew, fb)
    note = f"### 偏好反馈\n- 命令: `{cmd.strip()}` → reward={rew:+.0f}\n- 类型: `{'对外轮' if isinstance(fb, DialogueTurn) else '内心'}`"
    state["last_fb"] = None
    _append_log(state, note)
    return state, state["log"], "已记录反馈。"


def clear_log(state: SessionState) -> tuple[SessionState, str, str]:
    if state:
        state["log"] = ""
        state["last_fb"] = None
    return state or cast(SessionState, {}), "", "日志已清空。"


def build_ui() -> gr.Blocks:
    state: gr.State = gr.State(cast(SessionState, {}))

    with gr.Blocks(title="UVA 对话调参台") as demo:
        gr.Markdown(
            "# UVA 认知对话 · Web 调参台\n"
            "加载分词器（及可选 `.dialogue.json`）后，可调整下方超参并重新加载；"
            "用「单轮对话」模拟 `--text`，「内心一拍」模拟 `internal_tick()`。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                tok_path = gr.Textbox(
                    label="分词器 JSON 路径",
                    value=_default_tokenizer_path(),
                    placeholder="tokenizer_zh_from_chunks_v2.json",
                )
                dlg_path = gr.Textbox(
                    label="对话状态 JSON（可选）",
                    value="",
                    placeholder="与分词器同名的 .dialogue.json，可留空",
                )
                learn_tok = gr.Checkbox(label="允许从用户句在线学分词器", value=False)

            with gr.Column(scale=1):
                arb_scale = gr.Slider(
                    0.1,
                    3.0,
                    value=1.0,
                    step=0.05,
                    label="仲裁社会入场券 scale（ARBITRATION_SOCIAL_TICKET_SCALE）",
                )
                lam_exp = gr.Slider(0.05, 1.2, value=0.55, step=0.01, label="λ_explore_0")
                c_thr = gr.Slider(0.0, 0.5, value=0.10, step=0.01, label="conflict_surprise_threshold")
                c_gain = gr.Slider(0.0, 1.5, value=0.45, step=0.01, label="conflict_surprise_gain")

        with gr.Accordion("内心 / pending / 好奇心注入", open=False):
            with gr.Row():
                int_thr = gr.Slider(0.02, 0.5, value=0.12, step=0.01, label="internal_tension_threshold")
                int_inc = gr.Slider(0.01, 0.3, value=0.09, step=0.01, label="internal_pending_increment")
                int_k = gr.Slider(0, 32, value=8, step=1, label="internal_global_scan_k")
            with gr.Row():
                int_jit = gr.Slider(0.0, 0.08, value=0.015, step=0.001, label="internal_spontaneous_jitter")
                int_seed = gr.Textbox(label="内心 RNG 种子（留空=随机）", value="")
            with gr.Row():
                bb_w = gr.Slider(0.0, 0.2, value=0.06, step=0.005, label="branch_bias_weight")
            with gr.Row():
                debt_g = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="cognitive_debt_surprise_gamma")
                c_pool = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="curiosity_injection_pool_max")
                c_cap = gr.Slider(0.0, 1.0, value=0.42, step=0.01, label="curiosity_injection_per_event_cap")
                c_ref = gr.Slider(0.0, 0.2, value=0.06, step=0.005, label="curiosity_pool_refill_external")

        load_inputs: List[Any] = [
            tok_path,
            dlg_path,
            learn_tok,
            arb_scale,
            lam_exp,
            c_thr,
            c_gain,
            int_thr,
            int_inc,
            int_k,
            int_jit,
            int_seed,
            bb_w,
            debt_g,
            c_pool,
            c_cap,
            c_ref,
        ]

        load_btn = gr.Button("加载模型", variant="primary")
        load_status = gr.Markdown("尚未加载。")

        user_in = gr.Textbox(label="用户输入（单轮）", lines=2, placeholder="输入一句话…")
        with gr.Row():
            run_btn = gr.Button("单轮对话", variant="primary")
            tick_btn = gr.Button("内心一拍")
        run_status = gr.Markdown()

        with gr.Row():
            fb_in = gr.Textbox(label="对上一轮输出反馈", placeholder="good / bad / meh", scale=2)
            fb_btn = gr.Button("提交反馈")
            clr_btn = gr.Button("清空日志")

        log_out = gr.Markdown(label="运行日志")

        load_btn.click(
            fn=load_model,
            inputs=load_inputs,
            outputs=[state, load_status],
        )
        run_btn.click(
            fn=run_single_turn,
            inputs=[state, user_in],
            outputs=[state, log_out, run_status],
        )
        tick_btn.click(
            fn=run_internal_tick,
            inputs=[state],
            outputs=[state, log_out, run_status],
        )
        fb_btn.click(
            fn=apply_feedback_cmd,
            inputs=[state, fb_in],
            outputs=[state, log_out, run_status],
        )
        clr_btn.click(
            fn=clear_log,
            inputs=[state],
            outputs=[state, log_out, run_status],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="UVA 对话 Web 调参台")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="端口")
    parser.add_argument("--share", action="store_true", help="创建 Gradio 临时公网链接")
    args = parser.parse_args()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=int(args.port),
        share=bool(args.share),
    )


if __name__ == "__main__":
    main()
