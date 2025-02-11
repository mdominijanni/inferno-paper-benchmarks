import brian2 as b2
import json
import sys
from tqdm import tqdm


def pdl(
    n_runs: int,
    n_steps: int,
    n_neurons: int,
    device: str,
    seed: int,
    f_max_spike: float,
    n_warmup: int,
):
    # set device
    b2.set_device(device, build_on_run=False)

    # set preferences
    b2.prefs.core.default_float_dtype = b2.float32
    b2.prefs.logging.console_log_level = "WARNING"
    b2.utils.logger.get_logger(__name__).log_level_warn()

    # general test constants
    step_time = 1.0 * b2.ms

    # connection-specific constants
    winit_min = 0.0  # noqa: F841;
    winit_max = 1.0  # noqa: F841;

    # encoder constants
    f_max_spike = f_max_spike * b2.Hz  # noqa: F841;

    # neuron-specific constants
    rest_v = -60.0 * b2.mV
    reset_v = -65.0 * b2.mV  # noqa: F841;
    thresh_v = -50.0 * b2.mV  # noqa: F841;
    refrac_t = 3.0 * b2.ms
    time_constant = 20.0 * b2.ms  # noqa: F841;
    resistance = 1.0 * b2.Mohm  # noqa: F841;

    # set global random seed since it can't be passed in
    b2.seed(seed)

    # build device
    b2.device.build(clean=True)

    # perform the benchmark
    results = []
    for n in range(n_runs + n_warmup):
        # set default clock
        b2.defaultclock.dt = step_time

        # create components
        encd = b2.PoissonGroup(n_neurons, "rand() * f_max_spike")

        eqs = "dv/dt = (rest_v - v) / time_constant : volt (unless refractory)"
        neur = b2.NeuronGroup(
            n_neurons,
            model=eqs,
            reset="v = reset_v",
            threshold="v > thresh_v",
            refractory=refrac_t,
            method="exact",
        )
        neur.v = rest_v
        conn = b2.Synapses(encd, neur, "w : amp", on_pre="v += w * resistance")
        conn.connect()
        conn.w = "(rand() * (winit_max - winit_min) + winit_min) * nA"

        # process inputs
        b2.run(step_time * float(n_steps), profile=True)

        # add to results if not warming up (seconds to milliseconds)
        if n >= n_warmup:
            results.append(
                sum(float(t) for _, t in b2.magic_network.profiling_info) * 1e3
            )

        # cleanup
        b2.device.reinit()
        b2.device.delete(force=True)
        b2.device.activate()

    return {"mean": sum(results) / len(results), "total": sum(results)}


def sweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
    n_steps = 1000
    f_max_spike = 250
    seed = 271828
    n_neurons = (
        [*range(100, 500, 100)]
        + [*range(500, 5000, 500)]
        + [*range(5000, 10000 + 1, 1000)]
    )

    res = {}
    res["info"] = {}
    res["info"]["bench"] = "pdl"
    res["info"]["device"] = device
    res["info"]["n_trials"] = n_runs
    res["info"]["n_warmup"] = n_warmup
    res["results"] = {}

    print(f"PDL Sweep ({device}), {n_runs}+{n_warmup} Runs")
    for nneur in tqdm(n_neurons):
        res["results"][nneur] = pdl(
            n_runs, n_steps, nneur, device, seed, f_max_spike, n_warmup
        )
    return res


def lsweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
    n_steps = 1000
    f_max_spike = 250
    seed = 271828
    n_neurons = [15000]

    res = {}
    res["info"] = {}
    res["info"]["bench"] = "pdl"
    res["info"]["device"] = device
    res["info"]["n_trials"] = n_runs
    res["info"]["n_warmup"] = n_warmup
    res["results"] = {}

    print(f"PDL Sweep ({device}), {n_runs}+{n_warmup} Runs")
    for nneur in tqdm(n_neurons):
        res["results"][nneur] = pdl(
            n_runs, n_steps, nneur, device, seed, f_max_spike, n_warmup
        )
    return res


def xsweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
    n_steps = 1000
    f_max_spike = 250
    seed = 271828
    n_neurons = [20000]

    res = {}
    res["info"] = {}
    res["info"]["bench"] = "pdl"
    res["info"]["device"] = device
    res["info"]["n_trials"] = n_runs
    res["info"]["n_warmup"] = n_warmup
    res["results"] = {}

    print(f"PDL Sweep ({device}), {n_runs}+{n_warmup} Runs")
    for nneur in tqdm(n_neurons):
        res["results"][nneur] = pdl(
            n_runs, n_steps, nneur, device, seed, f_max_spike, n_warmup
        )
    return res


if __name__ == "__main__":
    dd = sweep(10, "runtime", 5)
    with open("b2-pdlres-r-cpu.json", "w") as f:
        json.dump(dd, f)

    dd = lsweep(10, "runtime", 5)
    with open("b2-pdlres-lg-cpu.json", "w") as f:
        json.dump(dd, f)

    if len(sys.argv) == 1 or sys.argv[1] != "nogpu":
        import brian2cuda  # noqa: F401;

        dd = sweep(10, "cuda_standalone", 5)
        with open("b2-pdlres-r-cuda.json", "w") as f:
            json.dump(dd, f)

        dd = lsweep(10, "cuda_standalone", 5)
        with open("b2-pdlres-lg-cuda.json", "w") as f:
            json.dump(dd, f)

        dd = xsweep(10, "cuda_standalone", 5)
        with open("b2-pdlres-xl-cuda.json", "w") as f:
            json.dump(dd, f)
