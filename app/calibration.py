"""Split conformal 下一字符校准：样本批次管理 + 共形分位数 + 后台计算任务。

用途：文本生成研究员录入一批人工核实过的「提示词 → 真实下一字符」样本，
用固定目标模型的真值概率计算非一致性分数，按有限样本修正的 split
conformal 分位数，为待判提示词输出「可打印 ASCII 下一字符集合」。

数学约定（split conformal 分类形式）：
  - 非一致性分数 s_i = -ln P_target(y_i | x_i)，y_i 为人工核实的真实下一字符。
    该分数与经典的 1 - P 分数严格单调等价（二者给出的预测集合在精确算术下
    完全相同：入集条件都是 P(c|x) ≥ 校准真值概率的第 k 大值）。选用对数域
    是因为本目标模型置信度极高（真值概率常低至 1e-30 以下），1 - P 在
    float32/float64 下都会饱和成精确的 1.0，丢失全部区分度；
  - 有限样本修正分位数：k = ⌈(n+1)(1-α)⌉，q̂ 为第 k 小的校准分数；
    k > n 时 q̂ = +∞（任何分数都入集，预测集合为全部可打印字符；
    本约束下 n ≥ 10、α ≥ 0.1 保证 k ≤ n，该分支不会经 API 触发）；
  - 预测集合 C(x) = { c ∈ 可打印ASCII : -ln P(c|x) ≤ q̂ }；
    分数相等（并列）的字符全部纳入，不做随机拆分；
    集合可能为空（模型对该提示词所有可打印字符的把握都低于阈值），如实返回。

并发与版本约定：
  - 样本批次带修订号，提交必须基于所见修订号，冲突（409）不覆盖；
  - 计算在后台线程进行，计算期间仍可提交新修订；
  - 计算结果标记其所基于的修订号；样本随后被修订时，旧结果保留但标记为
    过时（stale），绝不冒充当前样本的结论；
  - 计算失败时保留标有修订号的上次成功结果。

隔离约定：校准只读使用固定目标模型——前向走 use_cache=False（不触碰 KV
缓存），且通过 forward_count_suspended 暂停本线程前向计数，模型权重、
KV 缓存与原解码的前向计数均不受影响。
"""

from __future__ import annotations

import math
import threading
import uuid
from datetime import UTC, datetime

import torch

from . import tokenizer
from .model import CausalTransformer, get_models

# ---- 录入约束 ----
MIN_SAMPLES = 10
MAX_SAMPLES = 30
PROMPT_MIN_LEN = 1
PROMPT_MAX_LEN = 32
MAX_TEST_PROMPTS = 20
# 允许的误判率 alpha；k 的计算用分数精确整数运算，避免浮点取整误差
# （例如 20*0.8 在浮点下为 16.000000000000004，ceil 会错得 17）
_ALPHA_FRAC = {0.1: (9, 10), 0.2: (4, 5)}  # alpha -> (1-alpha) 的 (分子, 分母)
ALPHAS = tuple(_ALPHA_FRAC.keys())

# 复核启发式（纯展示用途，不属于共形覆盖率保证）：
# 空集或集合覆盖超过半数可打印字符 → 建议人工复核
_REVIEW_SIZE_THRESHOLD = len(tokenizer.PRINTABLE_CHARS) // 2  # 47


class ValidationError(Exception):
    """录入数据不合法（400）。details 为逐条中文错误信息。"""

    def __init__(self, details: list[str]):
        super().__init__("; ".join(details))
        self.details = details


class ConflictError(Exception):
    """修订号冲突（409）：提交基于的修订号已过期，不覆盖当前数据。"""

    def __init__(self, current_revision: int):
        super().__init__(f"修订冲突：当前修订为 {current_revision}")
        self.current_revision = current_revision


class BusyError(Exception):
    """已有计算任务在进行中（409）。"""


def _is_printable_ascii(s: str) -> bool:
    return all(0x20 <= ord(c) <= 0x7E for c in s)


def validate_samples(samples: list[dict]) -> None:
    """校验整批样本；不合法时抛 ValidationError（带逐条明细）。"""
    errors: list[str] = []
    if not (MIN_SAMPLES <= len(samples) <= MAX_SAMPLES):
        errors.append(f"样本数量须为 {MIN_SAMPLES}～{MAX_SAMPLES} 条，当前 {len(samples)} 条。")
    for i, s in enumerate(samples):
        prompt, nxt = s.get("prompt", ""), s.get("next_char", "")
        if not isinstance(prompt, str) or not isinstance(nxt, str):
            errors.append(f"第 {i + 1} 条：prompt 与 next_char 均须为字符串。")
            continue
        if not (PROMPT_MIN_LEN <= len(prompt) <= PROMPT_MAX_LEN):
            errors.append(
                f"第 {i + 1} 条：提示词长度须为 {PROMPT_MIN_LEN}～{PROMPT_MAX_LEN} 字符，"
                f"当前 {len(prompt)}。"
            )
        elif not _is_printable_ascii(prompt):
            errors.append(f"第 {i + 1} 条：提示词含非可打印 ASCII 字符。")
        if len(nxt) != 1:
            errors.append(f"第 {i + 1} 条：下一字符须恰好 1 个字符，当前 {len(nxt)} 个。")
        elif not _is_printable_ascii(nxt):
            errors.append(f"第 {i + 1} 条：下一字符须为可打印 ASCII。")
    if errors:
        raise ValidationError(errors)


def validate_alpha(alpha) -> float:
    """校验并返回规范化的 alpha（0.1 或 0.2）；不合法抛 ValidationError。"""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValidationError([f"错误率 alpha 只支持 {list(ALPHAS)}，收到 {alpha!r}。"])
    for a in ALPHAS:
        if math.isclose(float(alpha), a, rel_tol=0, abs_tol=1e-12):
            return a
    raise ValidationError([f"错误率 alpha 只支持 {list(ALPHAS)}，收到 {alpha!r}。"])


def validate_test_prompts(prompts: list[str]) -> None:
    errors: list[str] = []
    if not (1 <= len(prompts) <= MAX_TEST_PROMPTS):
        errors.append(f"待判提示词须为 1～{MAX_TEST_PROMPTS} 条，当前 {len(prompts)} 条。")
    for i, p in enumerate(prompts):
        if not isinstance(p, str):
            errors.append(f"第 {i + 1} 条待判提示词须为字符串。")
        elif not (PROMPT_MIN_LEN <= len(p) <= PROMPT_MAX_LEN):
            errors.append(
                f"第 {i + 1} 条待判提示词长度须为 {PROMPT_MIN_LEN}～{PROMPT_MAX_LEN} 字符，"
                f"当前 {len(p)}。"
            )
        elif not _is_printable_ascii(p):
            errors.append(f"第 {i + 1} 条待判提示词含非可打印 ASCII 字符。")
    if errors:
        raise ValidationError(errors)


# ---- 共形计算（纯函数，便于单测） ----

def conformal_k(n: int, alpha: float) -> int:
    """有限样本修正的分位数序号 k = ⌈(n+1)(1-α)⌉（精确整数运算）。"""
    num, den = _ALPHA_FRAC[float(alpha)]
    return ((n + 1) * num + den - 1) // den


def conformal_quantile(scores: list[float], alpha: float) -> tuple[float, int]:
    """返回 (q̂, k)。k > n 时 q̂ = +∞（任何分数都入集，即全集）。"""
    n = len(scores)
    k = conformal_k(n, alpha)
    if k > n:
        return math.inf, k
    return sorted(scores)[k - 1], k


def prediction_set(logprobs: list[float], q_hat: float) -> list[str]:
    """可打印 ASCII 预测集合：{ c : -ln P(c) ≤ q̂ }。

    分数相等的字符全部纳入（≤ 比较，不做随机拆分）；可能为空集，如实返回。
    """
    out: list[str] = []
    for ch in tokenizer.PRINTABLE_CHARS:
        if -logprobs[tokenizer.char_id(ch)] <= q_hat:
            out.append(ch)
    return out


def next_char_logprobs(target: CausalTransformer, prompt_ids: list[int]) -> list[float]:
    """目标模型对下一字符的真值对数概率（全词表 log-softmax，float64）。

    旁路前向：use_cache=False 不触碰 KV 缓存；forward_count_suspended
    暂停本线程前向计数；只读权重。原解码的缓存、计数与权重均不受影响。
    对数域取值可避免该高置信度模型下 softmax 概率的浮点饱和/下溢。
    """
    with torch.no_grad(), target.forward_count_suspended():
        logits = target.forward(prompt_ids, use_cache=False)
    return torch.log_softmax(logits[-1].double(), dim=-1).tolist()


def _needs_review(set_size: int) -> tuple[bool, str]:
    """复核启发式（展示用途）：空集或集合过大 → 建议人工复核。"""
    if set_size == 0:
        return True, "需复核：空集（所有可打印字符的模型概率均低于阈值）"
    if set_size > _REVIEW_SIZE_THRESHOLD:
        return True, f"需复核：集合过大（覆盖 {set_size}/95 个可打印字符，模型无把握）"
    return False, "集合可用"


class Job:
    """一次后台校准计算。结果始终标记其基于的样本修订号。"""

    def __init__(self, revision: int, alpha: float):
        self.id = uuid.uuid4().hex[:12]
        self.revision = revision
        self.alpha = alpha
        self.status = "running"  # running | done | failed
        self.started_at = _now()
        self.finished_at: str | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "revision": self.revision,
            "alpha": self.alpha,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CalibrationStore:
    """样本批次 + 修订号 + 后台计算任务的进程内状态。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: list[dict] = []
        self._revision = 0
        self._job: Job | None = None
        self._last_success: dict | None = None

    # ---- 样本批次（乐观并发：提交须基于所见修订号，冲突不覆盖）----
    def submit_samples(self, base_revision: int, samples: list[dict]) -> int:
        validate_samples(samples)
        cleaned = [{"prompt": s["prompt"], "next_char": s["next_char"]} for s in samples]
        with self._lock:
            if base_revision != self._revision:
                raise ConflictError(self._revision)
            self._samples = cleaned
            self._revision += 1
            return self._revision

    # ---- 计算任务 ----
    def start_compute(self, alpha: float, test_prompts: list[str]) -> Job:
        alpha = validate_alpha(alpha)
        validate_test_prompts(test_prompts)
        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise BusyError("已有校准计算在进行中，请等待其完成后再发起。")
            if not self._samples:
                raise ValidationError(["样本批次为空，请先录入样本。"])
            # 快照当前修订的样本：计算期间新提交的修订不影响本次计算
            samples = [dict(s) for s in self._samples]
            job = Job(revision=self._revision, alpha=alpha)
            self._job = job
        job.thread = threading.Thread(
            target=self._run_job,
            args=(job, samples, list(test_prompts)),
            name=f"calibration-{job.id}",
            daemon=True,
        )
        job.thread.start()
        return job

    def _run_job(self, job: Job, samples: list[dict], test_prompts: list[str]) -> None:
        try:
            result = self._compute(job, samples, test_prompts)
        # 后台任务必须兜住一切异常：标记任务失败，保留标有修订号的上次成功结果
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = _now()
            return
        with self._lock:
            job.status = "done"
            job.finished_at = result["finished_at"]
            self._last_success = result

    def _compute(self, job: Job, samples: list[dict], test_prompts: list[str]) -> dict:
        _, target = get_models()
        # 1) 校准分数：s_i = -ln P(真实下一字符 | 提示词)
        score_rows = []
        for s in samples:
            logprobs = next_char_logprobs(target, tokenizer.encode(s["prompt"]))
            logp_true = logprobs[tokenizer.char_id(s["next_char"])]
            score_rows.append({
                "prompt": s["prompt"],
                "next_char": s["next_char"],
                "p_true": math.exp(logp_true),
                "score": -logp_true,
            })
        scores = [r["score"] for r in score_rows]
        q_hat, k = conformal_quantile(scores, job.alpha)

        # 2) 待判提示词的预测集合
        results = []
        for prompt in test_prompts:
            logprobs = next_char_logprobs(target, tokenizer.encode(prompt))
            chars = prediction_set(logprobs, q_hat)
            top1_id = max(range(len(logprobs)), key=lambda i: logprobs[i])
            needs_review, reason = _needs_review(len(chars))
            results.append({
                "prompt": prompt,
                "set": chars,
                "set_size": len(chars),
                "empty": len(chars) == 0,
                "needs_review": needs_review,
                "review_reason": reason,
                "top1": {"char": tokenizer.render_token(top1_id),
                         "prob": math.exp(logprobs[top1_id])},
            })

        return {
            "revision": job.revision,   # 结果永远标记其基于的样本修订号
            "alpha": job.alpha,
            "n": len(samples),
            "k": k,
            "q_hat": q_hat,
            "prob_threshold": math.exp(-q_hat),
            "scores": score_rows,
            "results": results,
            "finished_at": _now(),
        }

    # ---- 状态快照 ----
    def state(self) -> dict:
        with self._lock:
            last = self._last_success
            return {
                "revision": self._revision,
                "samples": [dict(s) for s in self._samples],
                "constraints": {
                    "min_samples": MIN_SAMPLES,
                    "max_samples": MAX_SAMPLES,
                    "prompt_min_len": PROMPT_MIN_LEN,
                    "prompt_max_len": PROMPT_MAX_LEN,
                    "alphas": list(ALPHAS),
                    "max_test_prompts": MAX_TEST_PROMPTS,
                    "review_size_threshold": _REVIEW_SIZE_THRESHOLD,
                },
                "job": self._job.public() if self._job else None,
                "last_success": last,
                # 旧版计算不得冒充新结论：修订号不一致即标记过时
                "stale": bool(last is not None and last["revision"] != self._revision),
            }


STORE = CalibrationStore()
