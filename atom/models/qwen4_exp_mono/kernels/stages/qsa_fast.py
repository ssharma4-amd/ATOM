"""Q3 and Q4 of the QSA layers: the index blocks' scores and the 512 best
blocks, for contexts up to :data:`MAX_CONTEXT` tokens.

- Q3 runs on every CTA (``GRID / rows`` tasks per request row). Waves 0-3
  first build index query head ``w`` in LDS (zero-centered norm, RoPE at row
  ``L``). Then each wave takes its blocks :data:`U` at a time, the batch's
  pages, index keys and RoPE rows loaded before any of its math: it pools a
  block's 4 index key rows (row ``L`` from the mailbox), normalizes and
  rotates the pooled key at row ``4 b``, takes its 4 dot products in f64
  (rounded to f32) and publishes the block's score (``isc``). The wave of the
  block row ``L`` completes also stores its rotated key in the compressed key
  cache, which only ATOM's own steps read. The 64-bit wave sums swap the
  upper halves of the wave in VALU (``permlane``). Each task ends with its
  hint (``fisch``).
- Q4 runs in two levels per request row: :data:`CHUNKS` chunk tasks, each
  selecting the 512 best of its :data:`FAST_CHUNK` blocks, and a final task
  selecting the 512 best of their candidates and ranking them, ties to the
  lower block. The 512 best of all blocks are among their chunk's 512 best.
  Up to :data:`FAST_CHUNK` blocks the final task selects from the scores
  itself, and up to 512 takes them all.

Selection (:func:`_select`): each thread holds its keys in registers --
folded score bits and ``0xFFFF - block``, unique -- and the radix passes
(:data:`RADIX`) count the keys matching the prefix so far in one of three
LDS histograms in turn, so that a pass needs one barrier: every wave scans
the histogram it just filled while the one two passes on is cleared. The
candidates go to LDS as 64-bit keys, and each one's rank is the number of
keys above it (:func:`_rank`).

The scores and this module's mailboxes (:data:`FSC`) sit in the scratch
buffer's second half (``layout.scratch_bytes``), :data:`SCORE_SHIFT` bytes
past the first: each row's scores at its ``isc`` offset, up to
:data:`MAX_BLOCKS` of them, and the mailboxes before them.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.qwen4_exp_mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp_mono.kernels import mailbox as mbx
from atom.models.qwen4_exp_mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp_mono.kernels.config import GRID, THREADS
from atom.models.qwen4_exp_mono.kernels.device_ops import rsrc, wave_umax_dpp, xshfl
from atom.models.qwen4_exp_mono.kernels.layout import (
    BASE,
    MAX_ROWS,
    PAIR,
    SC,
    SCRATCH_BYTES,
    STRIDE,
    TASKS,
    WAVES,
)
from atom.models.qwen4_exp_mono.kernels.stages.common import bf16r, mb_ix, task_index
from atom.models.qwen4_exp_mono.kernels.stages.hc import _f32
from atom.models.qwen4_exp_mono.kernels.stages.qsa import (
    INDEX_WORDS,
    P_IK,
    Q1_INDEX_TASKS,
    R128,
    ROPE_WORDS,
    SEL,
    SIGN,
    SQRT128,
    _i32,
    _lds_i4,
    _request,
    norm,
    page_block,
    qsa_row,
    row_len,
    unpack,
    word,
)

# Index blocks of a request row (16 bits of the radix key hold the block), and
# the contexts they hold.
MAX_BLOCKS = 1 << 16
MAX_CONTEXT = MAX_BLOCKS * C.index_compress  # 262144
# The scores' region sits this many bytes past ``isc``, in the scratch's second
# half (wide builds add their rows past ``MAX_ROWS``, ``kernel._stage_mailboxes``).
SCORE_SHIFT = MAX_ROWS * SCRATCH_BYTES
assert (
    SC["isc"] + MAX_BLOCKS * PAIR <= SCRATCH_BYTES
)  # a row's scores end within its row

U = 4  # Q3: blocks per wave per batch
FAST_CHUNK = 8192  # Q4: blocks per chunk task
CHUNKS = MAX_BLOCKS // FAST_CHUNK  # 8
Q4_TASKS = CHUNKS + 1  # per request row: the chunk tasks, then the final one
Q4_BATCH = 8  # scores polled per thread and retry
KEYS_CHUNK = FAST_CHUNK // THREADS  # 16 keys per thread
KEYS_FINAL = CHUNKS * SEL // THREADS  # 8 candidates per thread
assert (
    KEYS_FINAL == CHUNKS and SEL == THREADS
)  # thread ``tid`` holds slot ``tid`` of each chunk
assert FAST_CHUNK % (THREADS * Q4_BATCH) == 0 and FAST_CHUNK > SEL
assert TASKS["q3"] == GRID and TASKS["q4"] == Q4_TASKS
assert BASE["q4"] == BASE["q7"] + TASKS["q7"]  # Q4 on the CTAs after Q7's

# Q4's radix passes over the key (score bits, 0xFFFF - block): (key word,
# shift of the 8-bit digit).
RADIX = ((0, 24), (0, 16), (0, 8), (0, 0), (1, 8), (1, 0))

# Mailboxes, bytes into each row's second half: Q3's hints (one per task, at
# most ``GRID`` per row), the chunk tasks' hints, then their candidates (two
# pairs each: folded score bits, block).
_FAST_REGIONS = {"fisch": GRID, "fch": CHUNKS, "fcand": CHUNKS * SEL * 2}
FSC = {}
_off = 0
for _name, _pairs in _FAST_REGIONS.items():
    FSC[_name] = _off
    _off += (_pairs * PAIR + 255) // 256 * 256
assert _off <= SC["isc"]  # before the row's scores

# LDS words from ``q4_keys``: the three radix histograms, the waves' counts,
# then the candidates' 64-bit keys (two words each) and blocks.
H_WORDS = 256
CNT = 3 * H_WORDS
CK = CNT + 256
CB = CK + 2 * SEL
LDS_WORDS = CB + SEL
assert LDS_WORDS <= STRIDE["q4_keys"]


def _mailboxes(c):
    """Move this stage's ``isc`` to the second half and add :data:`FSC`'s
    (``_run`` sets every mailbox afresh before the next stage)."""
    c.mb_isc = c.mb_isc + SCORE_SHIFT
    half = c.mb_isc - SC["isc"]
    for name, off in FSC.items():
        setattr(c, "mb_" + name, half + off)


# ------------------------------------------------------------------ Q3


def _swap(v, off):
    """(the lower lane's ``v``, the upper lane's) of lanes ``l`` and ``l ^
    off`` (32 or 16), in VALU, as ``device_ops.xred``."""
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pair = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(v)),
        as_ir_value(fx.Int32(v)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pair, [k])) for k in range(2))


def _wave_sum64(v):
    """Butterfly sum of an f64 over the wave; every lane gets the same bits.
    The 32 and 16 steps swap in VALU (:func:`_swap`)."""
    for off in (32, 16, 8, 4, 2, 1):
        w = fx.Vector.from_elements([v], fx.Float64).bitcast(fx.Int32)
        if off >= 16:
            halves = [_swap(w[0], off), _swap(w[1], off)]
            a = fx.Vector.from_elements([halves[0][0], halves[1][0]], fx.Int32)
            b = fx.Vector.from_elements([halves[0][1], halves[1][1]], fx.Int32)
            v = a.bitcast(fx.Float64)[0] + b.bitcast(fx.Float64)[0]
        else:
            o = fx.Vector.from_elements([xshfl(w[0], off), xshfl(w[1], off)], fx.Int32)
            v = v + o.bitcast(fx.Float64)[0]
    return v


def _rope16(y, cs, sn, lane, active):
    """``qsa.rope`` with ``off`` 16, the partner from a VALU swap."""
    neg = ((lane & 16) == 0).select(fx.Int32(SIGN), fx.Int32(0))
    out = []
    for v, c, s in zip(y, cs, sn):
        lo, hi = _swap(v.bitcast(fx.Int32), 16)
        partner = ((lane & 16) == 0).select(hi, lo)
        partner = (partner ^ neg).bitcast(fx.Float32)
        out.append(active.select(bf16r(bf16r(v * c) + bf16r(partner * s)), v))
    return out


def q3(c, p):
    _mailboxes(c)
    lane, wave = c.lane, c.wave
    t = task_index(c, "q3")
    iqn = c.lds_f + c.L.lds["iqn"]
    per = c.L.tasks["q3"] // c.rows
    rq, tq = _request(c, t, per)
    L = row_len(p, rq)
    a_pages = qsa_row(p, rq, "pages")
    a_index_key = qsa_row(p, rq, "index_key")
    a_index_cmp = qsa_row(p, rq, "index_cmp")
    a_cos = qsa_row(p, rq, "cos")
    a_sin = qsa_row(p, rq, "sin")
    cb = (L + 1) // 4

    # Tasks with no block skip the stage; Q4 waits on the others' hints, and
    # on task 0's at any context (a row at positions 0-2 has no block).
    if (t < c.L.tasks["q3"]) & (tq * WAVES < fx.max(cb, 1)):
        r_pg = rsrc(a_pages)
        r_ik = rsrc(a_index_key)
        r_cmp = rsrc(a_index_cmp)
        r_cos = rsrc(a_cos)
        r_sin = rsrc(a_sin)
        rope_word = fx.min(lane, ROPE_WORDS - 1)
        wq = unpack([word(rsrc(p.a_iq_norm), lane)])
        wk = unpack([word(rsrc(p.a_ik_norm), lane)])
        cs = unpack([word(r_cos, L * ROPE_WORDS + rope_word)])
        sn = unpack([word(r_sin, L * ROPE_WORDS + rope_word)])
        mbx.pre_poll(lambda m: (c.mb_qp, mb_ix(32 * m, rq)), Q1_INDEX_TASKS, c.tag)

        if wave < C.index_heads:
            x = mbx.poll([(c.mb_qp, mb_ix(INDEX_WORDS * wave + lane, rq), 1)], c.tag)[0]
            y = norm(unpack(x), wq, R128, True)
            y = _rope16(y, cs, sn, lane, lane < ROPE_WORDS)
            for k in range_constexpr(2):
                fx.ptr_store(y[k], iqn + (wave * C.index_dim + 2 * lane + k))
        if const_expr(c.seq):
            got = mbx.poll(
                [(c.mb_qp, mb_ix(P_IK + lane, q), 1) for q in range(c.rows)], c.tag
            )
            iks = [v[0] for v in got]
        else:
            ik = mbx.poll([(c.mb_qp, mb_ix(P_IK + lane, rq), 1)], c.tag)[0][0]
        gpu.barrier()
        q = [
            fx.ptr_load(iqn + (h * C.index_dim + 2 * lane + e))
            for h in range(4)
            for e in range(2)
        ]

        # Wave ``w`` of task ``tq`` takes blocks ``8 tq + w + i nw``, ``U`` of
        # them per batch: the batch's loads first.
        nw = per * WAVES
        for b0 in range(tq * WAVES + wave, cb, U * nw):
            bs = [b0 + u * nw for u in range(U)]
            bc = [fx.min(b, cb - 1) for b in bs]
            gs = [page_block(r_pg, b, True) for b in bc]
            kws = [
                [word(r_ik, (4 * gs[u] + r) * INDEX_WORDS + lane) for r in range(4)]
                for u in range(U)
            ]
            ats = [4 * b * ROPE_WORDS + rope_word for b in bc]
            kcs = [word(r_cos, a) for a in ats]
            kss = [word(r_sin, a) for a in ats]
            for u in range_constexpr(U):
                b = bs[u]
                if b < cb:
                    raw = []
                    for r in range_constexpr(4):
                        row = 4 * b + r
                        kw = kws[u][r]
                        if const_expr(c.seq):
                            for s in range_constexpr(c.rows):
                                kw = (row == p.Ls[s]).select(iks[s], kw)
                            raw.append(unpack([fx.Int32(kw)]))
                        else:
                            raw.append(unpack([fx.Int32((row == L).select(ik, kw))]))
                    pooled = [
                        bf16r(
                            ((raw[0][e] + raw[1][e]) + (raw[2][e] + raw[3][e]))
                            * fx.Float32(0.25)
                        )
                        for e in range(2)
                    ]
                    kn = _rope16(
                        norm(pooled, wk, R128, True),
                        unpack([kcs[u]]),
                        unpack([kss[u]]),
                        lane,
                        lane < ROPE_WORDS,
                    )
                    relu = []
                    for h in range_constexpr(C.index_heads):
                        part = (q[2 * h] * kn[0]).to(fx.Float64) + (
                            q[2 * h + 1] * kn[1]
                        ).to(fx.Float64)
                        relu.append(
                            fx.max(_wave_sum64(part).to(fx.Float32), fx.Float32(0.0))
                        )
                    score = ((relu[0] + relu[1]) + (relu[2] + relu[3])) / _f32(SQRT128)
                    if lane == 0:
                        mbx.put(c.mb_isc, mb_ix(b, rq), score, c.tag)
                    # Row L completes block b: its key for ATOM's compressed cache.
                    if 4 * b + 3 == L:
                        bo.buffer_store(
                            mbx.bf16_pair(kn[0], kn[1]),
                            r_cmp,
                            gs[u] * INDEX_WORDS + lane,
                        )

        gpu.barrier()
        if c.tid == 0:
            mbx.put(c.mb_fisch, mb_ix(tq, rq), 1, c.tag)


q3 = ASTRewriter.transform(q3)


# ------------------------------------------------------------------ Q4


def _clear(c, k0):
    """Histograms 0 and 1 to zero (``THREADS`` words); a barrier must follow
    before the first pass."""
    fx.ptr_store(fx.Int32(0), k0 + fx.Int32(c.tid))


def _digit(c, h, which, shift, prefix, remain):
    """From the counts in histogram ``h``, the digit at ``shift`` of the
    ``remain``-th largest key among those matching ``prefix``: the extended
    prefix and the count still to take. In every wave: lane ``l`` scans digits
    ``255 - 4 l ..``; the one lane holding the digit packs it with the count."""
    lane = c.lane
    v = _lds_i4(h, 252 - 4 * lane)
    cnt = [v[3], v[2], v[1], v[0]]  # digits 255 - 4 l, 254 - 4 l, ...
    tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
    above = fx.coop.warp_inclusive_scan(tot, fx.ReductionOp.ADD, width=64) - tot
    pick = fx.Int32(0)
    for k in range_constexpr(4):
        hit = (above < remain) & (above + cnt[k] >= remain)
        pick = hit.select(((255 - 4 * lane - k) << 16) | (remain - above), pick)
        above = above + cnt[k]
    got = wave_umax_dpp(pick)
    return prefix | ((got >> 16) << shift), got & 0xFFFF


def _select(c, k0, hi, lo):
    """The 512th largest of the keys ``(hi, lo)`` this CTA's threads hold,
    as ``[hi, lo]``; a key ``hi`` -1 holds no block, and no pass counts it
    (the first tests it, the others' prefixes clear its top bit).
    Histograms 0 and 1 cleared."""
    tid = fx.Int32(c.tid)
    prefix = [fx.Int32(0), fx.Int32(0)]
    mask = [0, 0]
    remain = fx.Int32(SEL)
    for k in range_constexpr(len(RADIX)):
        which, shift = RADIX[k]
        h = k0 + H_WORDS * (k % 3)
        for e in range_constexpr(len(hi)):
            m = hi[e] >= 0
            if const_expr(k > 0):
                m = ((hi[e] & _i32(mask[0])) == prefix[0]) & (
                    (lo[e] & _i32(mask[1])) == prefix[1]
                )
            if m:
                d = ((hi[e], lo[e])[which] >> shift) & 255
                fx.atomic_add(h + d, fx.Int32(1), syncscope="workgroup")
        gpu.barrier()
        # The histogram of pass k + 2, read by every wave in pass k - 1.
        if const_expr(k + 2 < len(RADIX)):
            if tid < H_WORDS:
                fx.ptr_store(fx.Int32(0), k0 + (H_WORDS * ((k + 2) % 3) + tid))
        prefix[which], remain = _digit(c, h, which, shift, prefix[which], remain)
        mask[which] |= 0xFF << shift
    return prefix


_select = ASTRewriter.transform(_select)


def _slots(c, k0, keep):
    """This thread's first slot among the kept keys of the CTA, in thread
    order (a barrier)."""
    lane, wave = c.lane, c.wave
    n = fx.Int32(0)
    for kk in keep:
        n = n + kk.select(fx.Int32(1), fx.Int32(0))
    incl = fx.coop.warp_inclusive_scan(n, fx.ReductionOp.ADD, width=64)
    if lane == 63:
        fx.ptr_store(incl, k0 + (CNT + wave))
    gpu.barrier()
    before = fx.Int32(0)
    for w in range_constexpr(WAVES):
        before = before + (wave > w).select(fx.ptr_load(k0 + (CNT + w)), fx.Int32(0))
    return before + incl - n


_slots = ASTRewriter.transform(_slots)


def _kept(hi, lo, thr, every=None):
    """Which keys are among the 512 best: at or above ``thr``, or all
    (``every``; none with ``hi`` -1, below every threshold)."""
    if every is not None:
        thr = [every.select(fx.Int32(0), t) for t in thr]
    return [(h > thr[0]) | ((h == thr[0]) & (lo_ >= thr[1])) for h, lo_ in zip(hi, lo)]


def _put_key(k0, i, hi, lo, blk):
    """Candidate slot ``i`` in LDS: its 64-bit key ``hi << 16 | lo``, its block."""
    fx.ptr_store((hi << 16) | lo, k0 + (CK + 2 * i))
    fx.ptr_store(hi >> 16, k0 + (CK + 2 * i + 1))
    fx.ptr_store(blk, k0 + (CB + i))


def _scores(c, base, n, rq):
    """Thread ``tid``'s blocks ``base + tid + 512 e`` (``e`` <
    :data:`KEYS_CHUNK`; ``hi`` -1 from ``base + n``): keys and blocks."""
    tid = fx.Int32(c.tid)
    hi, lo, blk = [], [], []
    for e0 in range_constexpr(0, KEYS_CHUNK, Q4_BATCH):
        at = [fx.min(tid + THREADS * (e0 + k), n - 1) for k in range(Q4_BATCH)]
        got = mbx.poll([(c.mb_isc, mb_ix(base + a, rq), 1) for a in at], c.tag)
        for k in range_constexpr(Q4_BATCH):
            j = tid + THREADS * (e0 + k)
            hi.append((j < n).select(got[k][0] & 0x7FFFFFFF, fx.Int32(-1)))
            lo.append(0xFFFF - (base + j))
            blk.append(base + j)
    return hi, lo, blk


def _candidates(c, cb, rq):
    """Thread ``tid``'s candidates: slot ``tid`` of each chunk (``hi`` -1
    where the chunk has none; those poll chunk 0's, which always has it)."""
    tid = fx.Int32(c.tid)
    hi, lo, blk = [], [], []
    for e0 in range_constexpr(0, CHUNKS, 4):
        specs, have = [], []
        for e in range(e0, e0 + 4):
            have.append(tid < fx.min(cb - e * FAST_CHUNK, SEL))
            specs.append(
                (c.mb_fcand, mb_ix(2 * have[-1].select(e * SEL + tid, tid), rq), 2)
            )
        got = mbx.poll(specs, c.tag)
        for k in range_constexpr(4):
            hi.append(have[k].select(got[k][0], fx.Int32(-1)))
            blk.append(got[k][1])
            lo.append(0xFFFF - got[k][1])
    return hi, lo, blk


def _keep(c, k0, hi, lo, blk):
    """The 512 best of the held keys to LDS slots (:func:`_put_key`)."""
    thr = _select(c, k0, hi, lo)
    keep = _kept(hi, lo, thr)
    slot = _slots(c, k0, keep)
    for e in range_constexpr(len(hi)):
        if keep[e]:
            _put_key(k0, slot, hi[e], lo[e], blk[e])
        slot = slot + keep[e].select(fx.Int32(1), fx.Int32(0))


_keep = ASTRewriter.transform(_keep)


def _chunk(c, j, cb, busy, per, k0, rq):
    """Chunk task ``j``: the 512 best of blocks ``j FAST_CHUNK ..`` (all of
    them up to 512) to ``fcand``, then its hint."""
    tid = fx.Int32(c.tid)
    base = j * FAST_CHUNK
    n = fx.min(cb - base, FAST_CHUNK)
    _clear(c, k0)
    mbx.pre_poll(lambda m: (c.mb_fisch, mb_ix(fx.min(m, busy - 1), rq)), per, c.tag)
    hi, lo, blk = _scores(c, base, n, rq)
    thr = _select(c, k0, hi, lo)
    keep = _kept(hi, lo, thr, n <= SEL)
    slot = _slots(c, k0, keep)
    for e in range_constexpr(KEYS_CHUNK):
        if keep[e]:
            at = mb_ix(2 * (j * SEL + slot), rq)
            mbx.put2(c.mb_fcand, at, hi[e], blk[e], c.tag)
        slot = slot + keep[e].select(fx.Int32(1), fx.Int32(0))
    gpu.barrier()
    if tid == 0:
        mbx.put(c.mb_fch, mb_ix(j, rq), 1, c.tag)


_chunk = ASTRewriter.transform(_chunk)


def _rank(c, k0, n, rq):
    """Each of the ``n`` candidates in LDS to ``sel`` at its rank (the keys
    above it), then the hint."""
    tid = fx.Int32(c.tid)
    n32 = (n + 31) // 32 * 32
    if (tid >= n) & (tid < n32):
        fx.ptr_store(fx.Int32(-1), k0 + (CK + 2 * tid))  # below every key
        fx.ptr_store(fx.Int32(-1), k0 + (CK + 2 * tid + 1))
    gpu.barrier()
    if tid < n:
        mine = fx.Vector.from_elements(
            [fx.ptr_load(k0 + (CK + 2 * tid)), fx.ptr_load(k0 + (CK + 2 * tid + 1))],
            fx.Int32,
        ).bitcast(fx.Int64)[0]
        mb = fx.ptr_load(k0 + (CB + tid))
        rank = fx.Int32(0)
        for j in range(0, n32, 32):
            for q in range_constexpr(16):
                v = _lds_i4(k0, CK + 2 * j + 4 * q)
                two = fx.Vector.from_elements(v, fx.Int32).bitcast(fx.Int64)
                for e in range_constexpr(2):
                    rank = rank + (two[e] > mine).select(fx.Int32(1), fx.Int32(0))
        mbx.put(c.mb_sel, mb_ix(rank, rq), mb, c.tag)
    gpu.barrier()
    if tid == 0:
        mbx.put(c.mb_selh, mb_ix(0, rq), 1, c.tag)


_rank = ASTRewriter.transform(_rank)


def _final(c, cb, busy, per, nch, k0, rq):
    """The final task: the 512 best of the chunks' candidates (past
    :data:`FAST_CHUNK` blocks) or of the scores, ranked to ``sel``."""
    tid = fx.Int32(c.tid)
    _clear(c, k0)
    if cb > FAST_CHUNK:
        mbx.pre_poll(lambda m: (c.mb_fch, mb_ix(fx.min(m, nch - 1), rq)), CHUNKS, c.tag)
        hi, lo, blk = _candidates(c, cb, rq)
        _keep(c, k0, hi, lo, blk)
    else:
        mbx.pre_poll(lambda m: (c.mb_fisch, mb_ix(fx.min(m, busy - 1), rq)), per, c.tag)
        if cb > SEL:
            hi, lo, blk = _scores(c, 0, cb, rq)
            _keep(c, k0, hi, lo, blk)
        else:
            if tid < cb:
                sv = mbx.poll([(c.mb_isc, mb_ix(tid, rq), 1)], c.tag)[0][0]
                _put_key(k0, tid, sv & 0x7FFFFFFF, 0xFFFF - tid, tid)
    _rank(c, k0, fx.min(cb, SEL), rq)


_final = ASTRewriter.transform(_final)


def q4(c, p):
    _mailboxes(c)
    t = task_index(c, "q4")
    k0 = c.lds_i + c.L.lds["q4_keys"]
    per = c.L.tasks["q3"] // c.rows  # Q3 tasks of a request row
    rq, j = _request(c, t, Q4_TASKS)
    L = row_len(p, rq)

    if t < c.L.tasks["q4"]:
        cb = (L + 1) // 4
        busy = fx.max(fx.min((cb + WAVES - 1) // WAVES, per), 1)  # Q3 tasks with blocks
        nch = (cb + FAST_CHUNK - 1) // FAST_CHUNK
        if j < CHUNKS:
            if (cb > FAST_CHUNK) & (j < nch):
                _chunk(c, j, cb, busy, per, k0, rq)
        else:
            _final(c, cb, busy, per, nch, k0, rq)


q4 = ASTRewriter.transform(q4)
