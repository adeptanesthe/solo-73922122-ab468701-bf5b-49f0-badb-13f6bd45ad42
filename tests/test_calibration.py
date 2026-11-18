"""下一字符 split conformal 校准的关键功能测试：

  1. 有限样本修正的分位数序号 k = ⌈(n+1)(1-α)⌉ 精确计算（整数运算，无浮点取整误差）；
  2. 预测集合：分数并列的字符全部纳入、空集如实返回、非可打印字符不入集；
  3. 样本校验：数量 10～30、提示词 1～32 可打印 ASCII、下一字符恰 1 个可打印 ASCII；
  4. 修订号乐观并发：提交须基于所见修订号，冲突拒绝且不覆盖；
  5. 计算结果标记修订号：计算期间提交新修订 → 结果标记旧修订并判为过时；
  6. 计算失败保留标有修订号的上次成功结果；
  7. 隔离性：校准不触碰目标模型权重、KV 缓存与前向计数，草稿模型完全不受影响；
  8. 校准计算进行中，原解码（基线+推测）仍可用且结果、计数完全正确；
  9. API 端到端：页面可达、提交/冲突/计算/轮询/解码全流程。
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import calibration, tokenizer
from app.calibration import (
    BusyError,
    CalibrationStore,
    ConflictError,
    ValidationError,
    conformal_k,
    conformal_quantile,
    next_char_logprobs,
    prediction_set,
    validate_alpha,
    validate_samples,
    validate_test_prompts,
)
from app.decoding import compare, greedy_baseline, speculative_decode
from app.model import get_models

torch.set_num_threads(1)
DRAFT, TARGET = get_models()

SAMPLES_12 = [
    {"prompt": "the quick brown ", "next_char": "f"},
    {"prompt": "hello, my name is ", "next_char": "j"},
    {"prompt": "1 + 1 = ", "next_char": "2"},
    {"prompt": "to be or not to ", "next_char": "b"},
    {"prompt": "a b c d e f ", "next_char": "g"},
    {"prompt": "the capital of france is ", "next_char": "p"},
    {"prompt": "once upon a ", "next_char": "t"},
    {"prompt": "https://www.", "next_char": "e"},
    {"prompt": "dear sir or ", "next_char": "m"},
    {"prompt": "happy new ", "next_char": "y"},
    {"prompt": "rock paper ", "next_char": "s"},
    {"prompt": "the answer is ", "next_char": "4"},
]


def _join(job, timeout: float = 60.0) -> None:
    job.thread.join(timeout)
    assert job.status != "running", "计算任务超时未完成"


# ---- 1. 分位数数学 ----

def test_conformal_k_exact() -> None:
    # k = ⌈(n+1)(1-α)⌉，用分数精确计算
    assert conformal_k(10, 0.1) == 10   # ⌈11*0.9⌉ = ⌈9.9⌉
    assert conformal_k(10, 0.2) == 9    # ⌈11*0.8⌉ = ⌈8.8⌉
    assert conformal_k(19, 0.2) == 16   # 20*0.8 = 16 整（浮点会得 16.000000000000004）
    assert conformal_k(30, 0.1) == 28   # ⌈31*0.9⌉ = ⌈27.9⌉
    assert conformal_k(30, 0.2) == 25   # ⌈31*0.8⌉ = ⌈24.8⌉
    assert conformal_k(12, 0.1) == 12   # ⌈13*0.9⌉ = ⌈11.7⌉
    print("ok: 有限样本修正分位序号 k 精确计算（含浮点陷阱 n=19, α=0.2）")


def test_conformal_quantile_value() -> None:
    scores = [0.91, 0.50, 0.72, 0.33, 0.88, 0.64, 0.97, 0.41, 0.55, 0.80]
    q, k = conformal_quantile(scores, 0.1)
    assert k == 10 and abs(q - 0.97) < 1e-12          # 第 10 小 = 最大值
    q, k = conformal_quantile(scores, 0.2)
    assert k == 9 and abs(q - 0.91) < 1e-12           # 第 9 小
    # k > n：q̂ = +∞，任何分数都入集（全集）
    q, k = conformal_quantile([0.4, 0.6], 0.2)
    assert k == 3 and q == math.inf
    print("ok: q̂ 取第 k 小校准分数；k > n 时 q̂=+∞（全集）")


# ---- 2. 预测集合 ----

def test_prediction_set_ties_empty_and_printable_only() -> None:
    logprobs = [math.log(1e-9)] * tokenizer.VOCAB_SIZE
    logprobs[tokenizer.char_id("a")] = math.log(0.30)
    logprobs[tokenizer.char_id("b")] = math.log(0.20)
    logprobs[tokenizer.char_id("c")] = math.log(0.20)   # 与 'b' 并列
    logprobs[tokenizer.EOS_ID] = math.log(0.90)         # 非可打印字符，概率再高也不入集
    # 阈值 0.2：a/b/c 全入（并列全纳入），EOS 不入
    assert prediction_set(logprobs, -math.log(0.2)) == ["a", "b", "c"]
    # 阈值 0.25：仅 a
    assert prediction_set(logprobs, -math.log(0.25)) == ["a"]
    # 阈值 0.5：空集，如实返回
    assert prediction_set(logprobs, -math.log(0.5)) == []
    print("ok: 并列分数全纳入、空集如实返回、非可打印字符不入集")


def test_log_score_equivalent_to_one_minus_p() -> None:
    """-lnP 分数与经典 1-P 分数在精确算术下给出完全相同的预测集合。"""
    rng = torch.Generator().manual_seed(7)
    raw = torch.rand(98, generator=rng) + 0.01
    p = (raw / raw.sum()).tolist()
    logp = [math.log(x) for x in p]
    cal_p = [0.42, 0.17, 0.63, 0.05, 0.29, 0.51, 0.38, 0.71, 0.11, 0.55]
    for alpha in (0.1, 0.2):
        k = conformal_k(len(cal_p), alpha)
        # 1-P 域：q̂ = 第 k 小的 1-p_i ⟺ 入集阈值 = 第 k 大的 p_i
        thr = sorted(cal_p, reverse=True)[k - 1]
        set_1mp = [c for c in tokenizer.PRINTABLE_CHARS
                   if p[tokenizer.char_id(c)] >= thr]
        # 对数域：q̂ = 第 k 小的 -ln p_i
        q_log = sorted(-math.log(x) for x in cal_p)[k - 1]
        set_log = prediction_set(logp, q_log)
        assert set_log == set_1mp
    print("ok: -lnP 分数与 1-P 分数的预测集合在精确算术下完全一致")


# ---- 3. 录入校验 ----

def test_validation() -> None:
    ok = [{"prompt": "abc", "next_char": "d"}] * 10
    validate_samples(ok)  # 不抛异常

    def expect_err(samples, frag):
        try:
            validate_samples(samples)
        except ValidationError as e:
            assert any(frag in d for d in e.details), (frag, e.details)
            return
        raise AssertionError(f"应拒绝: {frag}")

    expect_err(ok[:9], "样本数量")                       # 太少
    expect_err(ok * 4, "样本数量")                       # 40 条太多
    bad = [dict(s) for s in ok]
    bad[3] = {"prompt": "", "next_char": "d"}
    expect_err(bad, "第 4 条：提示词长度")
    bad = [dict(s) for s in ok]
    bad[3] = {"prompt": "x" * 33, "next_char": "d"}
    expect_err(bad, "第 4 条：提示词长度")
    bad = [dict(s) for s in ok]
    bad[5] = {"prompt": "tab\there", "next_char": "d"}
    expect_err(bad, "第 6 条：提示词含非可打印")
    bad = [dict(s) for s in ok]
    bad[0] = {"prompt": "abc", "next_char": "dd"}
    expect_err(bad, "第 1 条：下一字符须恰好 1 个")
    bad = [dict(s) for s in ok]
    bad[0] = {"prompt": "abc", "next_char": "é"}
    expect_err(bad, "第 1 条：下一字符须为可打印")

    validate_alpha(0.1); validate_alpha(0.2)
    assert validate_alpha(0.1 + 1e-13) == 0.1   # 近似值规范化为合法 alpha
    for bad_alpha in (0.3, 0, "0.1", None, True):
        try:
            validate_alpha(bad_alpha)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"应拒绝 alpha={bad_alpha!r}")

    validate_test_prompts(["a"] * 20)
    for bad, frag in (([], "1～20"), (["a"] * 21, "1～20"),
                      ([""], "长度"), (["x" * 33], "长度"), (["a\nb"], "非可打印")):
        try:
            validate_test_prompts(bad)
        except ValidationError as e:
            assert any(frag in d for d in e.details), (bad, e.details)
        else:
            raise AssertionError(f"应拒绝 test_prompts={bad!r}")
    print("ok: 样本/α/待判提示词校验（数量、长度、可打印 ASCII）")


# ---- 4. 修订号并发 ----

def test_revision_conflict_no_overwrite() -> None:
    store = CalibrationStore()
    assert store.state()["revision"] == 0
    rev = store.submit_samples(0, SAMPLES_12)
    assert rev == 1
    # 基于过期修订号提交 → 冲突，不覆盖
    try:
        store.submit_samples(0, SAMPLES_12[:10])
        raise AssertionError("应抛 ConflictError")
    except ConflictError as e:
        assert e.current_revision == 1
    st = store.state()
    assert st["revision"] == 1 and len(st["samples"]) == 12  # 原数据未被覆盖
    # 基于当前修订号提交 → 接受
    rev = store.submit_samples(1, SAMPLES_12[:10])
    assert rev == 2 and len(store.state()["samples"]) == 10
    print("ok: 提交须基于所见修订号，冲突拒绝且不覆盖已有数据")


# ---- 5/6. 计算任务的修订号标记、过时判定、失败保留 ----

def _blocking_probs(original, entered, release):
    def wrapper(target, ids):
        entered.set()
        release.wait(30)
        return original(target, ids)
    return wrapper


def test_edit_during_compute_marks_stale() -> None:
    """计算进行中提交新修订：结果标记其基于的旧修订号，且被判为过时。"""
    store = CalibrationStore()
    store.submit_samples(0, SAMPLES_12)
    entered, release = threading.Event(), threading.Event()
    original = calibration.next_char_logprobs
    calibration.next_char_logprobs = _blocking_probs(original, entered, release)
    try:
        job = store.start_compute(0.1, ["test prompt "])
        assert entered.wait(10), "计算线程未启动"
        # 计算进行中：另一窗口提交新修订（必须成功）
        new_rev = store.submit_samples(1, SAMPLES_12[:10])
        assert new_rev == 2
        release.set()
        _join(job)
    finally:
        release.set()
        calibration.next_char_logprobs = original
    assert job.status == "done"
    st = store.state()
    # 结果标记旧修订 #1，当前修订 #2 → 判为过时，不冒充新结论
    assert st["last_success"]["revision"] == 1
    assert st["revision"] == 2
    assert st["stale"] is True
    print("ok: 计算期间可提交修订；旧版结果标记修订号并判为过时")


def test_failure_keeps_last_success() -> None:
    store = CalibrationStore()
    store.submit_samples(0, SAMPLES_12)
    job1 = store.start_compute(0.1, ["hello "])
    _join(job1)
    assert job1.status == "done"
    first = store.state()["last_success"]
    assert first["revision"] == 1

    store.submit_samples(1, SAMPLES_12[:10])
    original = calibration.next_char_logprobs
    def boom(target, ids):
        raise RuntimeError("模拟计算故障")
    calibration.next_char_logprobs = boom
    try:
        job2 = store.start_compute(0.2, ["hello "])
        _join(job2)
    finally:
        calibration.next_char_logprobs = original
    assert job2.status == "failed" and "模拟计算故障" in job2.error
    st = store.state()
    # 上次成功结果原样保留（含其修订号），并因样本已修订而标记过时
    assert st["last_success"] is first
    assert st["last_success"]["revision"] == 1
    assert st["stale"] is True
    print("ok: 计算失败保留标有修订号的上次成功结果")


def test_busy_while_running() -> None:
    store = CalibrationStore()
    store.submit_samples(0, SAMPLES_12)
    entered, release = threading.Event(), threading.Event()
    original = calibration.next_char_logprobs
    calibration.next_char_logprobs = _blocking_probs(original, entered, release)
    try:
        job = store.start_compute(0.1, ["a "])
        assert entered.wait(10)
        try:
            store.start_compute(0.1, ["b "])
            raise AssertionError("应抛 BusyError")
        except BusyError:
            pass
    finally:
        release.set()
        calibration.next_char_logprobs = original
    _join(job)
    print("ok: 计算进行中拒绝并发计算（样本修订不受此限）")


# ---- 7/8. 隔离性 ----

def test_isolation_from_decode() -> None:
    """校准不触碰目标模型权重、KV 缓存与前向计数；草稿模型完全不受影响。"""
    ids = tokenizer.encode("the quick brown fox jumps over ")
    spec = speculative_decode(DRAFT, TARGET, ids, 3, 16)
    t_count = TARGET.forward_count
    t_cache_len = TARGET.cache_len
    t_cache = [(k.clone(), v.clone()) for k, v in TARGET._cache]
    d_count, d_cache_len = DRAFT.forward_count, DRAFT.cache_len
    d_cache = [(k.clone(), v.clone()) for k, v in DRAFT._cache]

    store = CalibrationStore()
    store.submit_samples(0, SAMPLES_12)
    job = store.start_compute(0.1, ["test prompt a ", "another prompt "])
    _join(job)
    assert job.status == "done"

    assert TARGET.forward_count == t_count, "校准改动了目标模型前向计数"
    assert TARGET.cache_len == t_cache_len
    for (k0, v0), (k1, v1) in zip(t_cache, TARGET._cache):
        assert torch.equal(k0, k1) and torch.equal(v0, v1), "校准改动了目标 KV 缓存"
    assert DRAFT.forward_count == d_count and DRAFT.cache_len == d_cache_len
    for (k0, v0), (k1, v1) in zip(d_cache, DRAFT._cache):
        assert torch.equal(k0, k1) and torch.equal(v0, v1), "校准触碰了草稿模型"

    # 权重未被改动：校准后再解码与校准前逐 token 一致
    before = greedy_baseline(TARGET, ids, 16)
    job2 = store.start_compute(0.2, ["zzz "])
    _join(job2)
    after = greedy_baseline(TARGET, ids, 16)
    assert before.new_tokens == after.new_tokens == spec.new_tokens
    print("ok: 校准后目标/草稿的权重、KV 缓存、前向计数均不受影响")


def test_decode_usable_during_calibration() -> None:
    """校准计算（后台线程）进行中，原解码仍可用且计数完全正确。"""
    ids = tokenizer.encode("hello world, this is a demo of ")
    ref = compare(DRAFT, TARGET, ids, 3, 24)   # 校准前的参照结果

    entered, release = threading.Event(), threading.Event()
    original = calibration.next_char_logprobs
    calibration.next_char_logprobs = _blocking_probs(original, entered, release)
    store = CalibrationStore()
    store.submit_samples(0, SAMPLES_12)
    try:
        job = store.start_compute(0.1, ["judge me "])
        assert entered.wait(10)
        # 校准进行中：原解码照常工作
        during = compare(DRAFT, TARGET, ids, 3, 24)
        assert during.match
        assert during.baseline.new_tokens == ref.baseline.new_tokens
        assert during.speculative.new_tokens == ref.speculative.new_tokens
        # 前向计数不受并行的校准线程影响
        assert during.baseline.target_forward_count == ref.baseline.target_forward_count
        assert during.speculative.target_forward_count == ref.speculative.target_forward_count
        assert during.speculative.draft_forward_count == ref.speculative.draft_forward_count
        release.set()
        _join(job)
    finally:
        release.set()
        calibration.next_char_logprobs = original
    assert job.status == "done"
    print("ok: 校准计算进行中，原解码可用且结果/计数完全正确")


# ---- 9. 端到端结果正确性（真实模型） ----

def test_compute_matches_direct_model() -> None:
    store = CalibrationStore()
    store.submit_samples(0, SAMPLES_12)
    job = store.start_compute(0.1, ["test ", "the "])
    _join(job)
    st = store.state()
    last = st["last_success"]
    assert last["revision"] == 1 and last["n"] == 12 and st["stale"] is False

    # 独立重算校准分数与分位数（对数域）
    scores = []
    for s in SAMPLES_12:
        logprobs = next_char_logprobs(TARGET, tokenizer.encode(s["prompt"]))
        scores.append(-logprobs[tokenizer.char_id(s["next_char"])])
    q_hat, k = conformal_quantile(scores, 0.1)
    assert k == last["k"] == 12
    assert abs(q_hat - last["q_hat"]) < 1e-9
    for row, s in zip(last["scores"], scores):
        assert abs(row["score"] - s) < 1e-9

    # 独立重算预测集合（含并列纳入规则）
    for res, prompt in zip(last["results"], ["test ", "the "]):
        logprobs = next_char_logprobs(TARGET, tokenizer.encode(prompt))
        expect = prediction_set(logprobs, q_hat)
        assert res["set"] == expect
        assert res["set_size"] == len(expect)
        assert res["empty"] == (len(expect) == 0)
        assert all(ch in tokenizer.PRINTABLE_CHARS for ch in res["set"])
    print(f"ok: 端到端计算与独立重算一致（q̂={last['q_hat']:.4f}，"
          f"集合大小 {[r['set_size'] for r in last['results']]}）")


# ---- 10. API 端到端 ----

def test_api_end_to_end() -> None:
    from fastapi.testclient import TestClient

    from app.main import app
    client = TestClient(app)

    # 页面可达
    r = client.get("/calibration")
    assert r.status_code == 200 and "校准" in r.text
    r = client.get("/")
    assert r.status_code == 200

    st = client.get("/api/calibration/state").json()
    rev0 = st["revision"]

    # 空批次不能计算
    r = client.post("/api/calibration/compute",
                    json={"alpha": 0.1, "test_prompts": ["abc"]})
    assert r.status_code == 400

    # 样本不合法 → 400（含逐条明细）
    r = client.put("/api/calibration/samples",
                   json={"base_revision": rev0,
                         "samples": [{"prompt": "x", "next_char": "y"}]})
    assert r.status_code == 400 and r.json()["details"]

    # 合法提交 → 修订号 +1
    r = client.put("/api/calibration/samples",
                   json={"base_revision": rev0, "samples": SAMPLES_12})
    assert r.status_code == 200 and r.json()["revision"] == rev0 + 1

    # 基于过期修订号提交 → 409，不覆盖
    r = client.put("/api/calibration/samples",
                   json={"base_revision": rev0, "samples": SAMPLES_12[:10]})
    assert r.status_code == 409
    assert r.json()["current_revision"] == rev0 + 1
    st = client.get("/api/calibration/state").json()
    assert len(st["samples"]) == 12

    # α 不合法 → 400
    r = client.post("/api/calibration/compute",
                    json={"alpha": 0.3, "test_prompts": ["abc"]})
    assert r.status_code == 400

    # 发起计算 → 202，轮询至完成
    r = client.post("/api/calibration/compute",
                    json={"alpha": 0.1, "test_prompts": ["judge this ", "and this "]})
    assert r.status_code == 202
    job_id = r.json()["job"]["id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        st = client.get("/api/calibration/state").json()
        if st["job"] and st["job"]["status"] != "running":
            break
        time.sleep(0.1)
    assert st["job"]["id"] == job_id and st["job"]["status"] == "done"
    assert st["last_success"]["revision"] == rev0 + 1
    assert st["stale"] is False
    assert len(st["last_success"]["results"]) == 2
    for res in st["last_success"]["results"]:
        assert set(res["set"]) <= set(tokenizer.PRINTABLE_CHARS)

    # 原解码接口在校准后照常工作
    r = client.post("/api/decode",
                    json={"prompt": "hello world, this is a demo of ", "k": 3, "max_new": 16})
    assert r.status_code == 200 and r.json()["status"] == "PASS"
    print("ok: API 端到端（页面/提交/冲突/计算/轮询/解码）")


if __name__ == "__main__":
    test_conformal_k_exact()
    test_conformal_quantile_value()
    test_prediction_set_ties_empty_and_printable_only()
    test_log_score_equivalent_to_one_minus_p()
    test_validation()
    test_revision_conflict_no_overwrite()
    test_edit_during_compute_marks_stale()
    test_failure_keeps_last_success()
    test_busy_while_running()
    test_isolation_from_decode()
    test_decode_usable_during_calibration()
    test_compute_matches_direct_model()
    test_api_end_to_end()
    print("\n校准测试全部通过 ✔")
