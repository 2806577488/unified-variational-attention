# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp, log


cdef inline double _max_double(double a, double b):
    return a if a >= b else b


cdef inline double _min_double(double a, double b):
    return a if a <= b else b


cdef inline double _novelty_signal(double surprise, double sigma0):
    return (surprise * surprise) / (sigma0 * sigma0 + surprise * surprise)


cdef inline double _sigmoid(double x):
    return 1.0 / (1.0 + exp(-x))


def trace_tokenize(
    object tokenizer,
    str text,
    *,
    bint collect_precision=True,
    bint collect_surprise=True,
    bint collect_boundary_indices=True,
    bint collect_resource=True,
    bint collect_mind=True,
):
    """Cython version of PrecisionTokenizer._trace_python."""
    from uva_model.tokenizer import TokenizationTrace

    cdef list tokens = []
    cdef str current = ""
    cdef list precision = [] if collect_precision else []
    cdef list surprise_list = [] if collect_surprise else []
    cdef list boundary_indices = [] if collect_boundary_indices else []
    cdef list resource_trace = [] if collect_resource else []
    cdef list mind_trace = [] if collect_mind else []

    cdef double prev_surprise = 0.0
    cdef double prev_free_energy = 0.0
    cdef double prev_m = <double>tokenizer.m
    cdef int cooldown = 0
    cdef int auto_rest_count = 0
    cdef double f_sum = 0.0
    cdef int f_count = 0
    cdef double surprise_sum = 0.0

    cdef double pi_min = <double>tokenizer.pi_min
    cdef double pi = pi_min
    cdef double prev_pi = pi
    cdef str prev_ch = "^"
    cdef dict unigram = tokenizer.unigram
    cdef dict bigram = tokenizer.bigram
    cdef double alpha = <double>tokenizer.alpha
    cdef double beta = <double>tokenizer.beta
    cdef double dt = <double>tokenizer.dt
    cdef double decay = <double>tokenizer.decay
    cdef double sigma0 = <double>tokenizer.sigma0
    cdef double surprise_threshold = <double>tokenizer.surprise_threshold
    cdef double auto_rest_threshold = <double>tokenizer.auto_rest_threshold
    cdef double auto_resume_threshold = <double>tokenizer.auto_resume_threshold
    cdef int auto_rest_steps = <int>tokenizer.auto_rest_steps
    cdef double R = <double>tokenizer.R
    cdef double m = <double>tokenizer.m
    cdef double R_max = <double>tokenizer.R_max
    cdef double F_ema = <double>tokenizer.F_ema
    cdef double rho = <double>tokenizer.rho
    cdef double lambda_deplete = <double>tokenizer.lambda_deplete
    cdef double tau_m = <double>tokenizer.tau_m
    cdef double theta_F = <double>tokenizer.theta_F
    cdef double R_crit = <double>tokenizer.R_crit
    cdef double R_base = <double>tokenizer.R_base
    cdef double R_max_cap = <double>tokenizer.R_max_cap
    cdef double tau_grow = <double>tokenizer.tau_grow
    cdef double eta_learn = <double>tokenizer.eta_learn
    cdef double lambda_grow = <double>tokenizer.lambda_grow
    cdef double F_ema_beta = <double>tokenizer.F_ema_beta

    cdef Py_ssize_t idx
    cdef str ch
    cdef object row
    cdef long prev_count
    cdef long vocab
    cdef long next_count
    cdef double prob
    cdef double surprise
    cdef double novelty
    cdef double task
    cdef double gain
    cdef double d_pi
    cdef double free_energy
    cdef double usage
    cdef double target
    cdef double dm
    cdef double dR
    cdef int idle_idx
    cdef double surprise_jump
    cdef double free_energy_jump
    cdef double mind_jump
    cdef double event_score
    cdef bint is_boundary

    for idx, ch in enumerate(text):
        while R <= auto_rest_threshold:
            for idle_idx in range(auto_rest_steps if auto_rest_steps > 0 else 0):
                target = _sigmoid(-theta_F) * _max_double(0.0, R_crit - R)
                dm = (target - m) / _max_double(1e-6, tau_m)
                m = _min_double(1.0, _max_double(0.0, m + dt * dm))
                dR = rho * m * (R_max - R)
                R = _min_double(R_max, _max_double(0.0, R + dt * dR))
            auto_rest_count += 1
            if R >= auto_resume_threshold:
                break

        if ch.isspace():
            if current:
                tokens.append(current)
                current = ""
            prev_ch = "^"
            pi = pi_min
            prev_pi = pi
            prev_surprise = 0.0
            prev_free_energy = 0.0
            prev_m = m
            cooldown = 0
            continue

        prev_count = <long>unigram.get(prev_ch, 0)
        vocab = <long>len(unigram)
        if vocab < 1:
            vocab = 1
        row = bigram.get(prev_ch)
        next_count = 0 if row is None else <long>row.get(ch, 0)
        prob = (next_count + 1.0) / (prev_count + vocab)
        surprise = -log(_max_double(prob, 1e-9))
        novelty = _novelty_signal(surprise, sigma0)
        task = 0.2 if prev_ch.isalnum() and ch.isalnum() else 0.05
        gain = _max_double(0.0, 1.0 - m)
        d_pi = gain * (alpha * novelty + beta * task) - decay * (pi - pi_min)
        pi = _max_double(pi_min, pi + dt * d_pi)
        free_energy = surprise * (0.2 + pi)
        usage = _max_double(0.0, pi - pi_min)
        target = _sigmoid(free_energy - theta_F) * _max_double(0.0, R_crit - R)
        dm = (target - m) / _max_double(1e-6, tau_m)
        m = _min_double(1.0, _max_double(0.0, m + dt * dm))
        dR = rho * m * (R_max - R) - lambda_deplete * _max_double(0.0, usage)
        R = _min_double(R_max, _max_double(0.0, R + dt * dR))
        f_sum += free_energy
        f_count += 1
        surprise_sum += surprise

        if collect_resource:
            resource_trace.append(R)
        if collect_mind:
            mind_trace.append(m)

        surprise_jump = surprise - prev_surprise
        free_energy_jump = free_energy - prev_free_energy
        mind_jump = m - prev_m
        event_score = (
            0.45 * _max_double(0.0, surprise_jump)
            + 0.25 * _max_double(0.0, free_energy_jump)
            + 0.20 * _max_double(0.0, pi - prev_pi)
            + 0.10 * _max_double(0.0, mind_jump)
        )
        is_boundary = (
            current != ""
            and len(current) >= 3
            and cooldown <= 0
            and surprise > surprise_threshold
            and event_score > 0.36
        )
        if is_boundary:
            tokens.append(current)
            current = ch
            if collect_boundary_indices:
                boundary_indices.append(idx)
            cooldown = 2
        else:
            current += ch
            cooldown = 0 if cooldown <= 0 else cooldown - 1

        if collect_precision:
            precision.append(pi)
        if collect_surprise:
            surprise_list.append(surprise)
        prev_pi = pi
        prev_ch = ch
        prev_surprise = surprise
        prev_free_energy = free_energy
        prev_m = m

    if current:
        tokens.append(current)

    if f_count > 0:
        F_ema = (1.0 - F_ema_beta) * F_ema + F_ema_beta * _max_double(f_sum / f_count, 1e-6)
        dR = (1.0 / _max_double(1e-6, tau_grow)) * (
            eta_learn / _max_double(F_ema, 0.05) - lambda_grow * (R_max - R_base)
        )
        R_max = _min_double(R_max_cap, _max_double(R_base, R_max + dR))
        R = _min_double(R, R_max)

    tokenizer.R = R
    tokenizer.m = m
    tokenizer.R_max = R_max
    tokenizer.F_ema = F_ema

    return TokenizationTrace(
        tokens=tokens,
        precision=precision,
        surprise=surprise_list,
        boundary_indices=boundary_indices,
        resource=resource_trace,
        mind_wander=mind_trace,
        auto_rest_count=auto_rest_count,
        mean_surprise=(surprise_sum / f_count) if f_count > 0 else 0.0,
    )


def trace_mean_surprise_batch(object tokenizer, list texts):
    cdef list out = []
    cdef str text
    cdef object trace
    for text in texts:
        trace = trace_tokenize(
            tokenizer,
            text,
            collect_precision=False,
            collect_surprise=False,
            collect_boundary_indices=False,
            collect_resource=False,
            collect_mind=False,
        )
        out.append(float(trace.mean_surprise))
    return out
