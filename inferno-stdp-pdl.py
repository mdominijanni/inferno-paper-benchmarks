import inferno
from inferno.learn import STDP
import inferno.neural as snn
import json
import sys
import torch
import time
from tqdm import tqdm


def pdl(
    n_runs: int,
    n_steps: int,
    n_neurons: int,
    batch_sz: int,
    device: str,
    seed: int,
    f_max_spike: float,
    n_warmup: int,
):
    # general test constants
    step_time = 1.0

    # connection-specific constants
    wmin = 0.0
    wmax = 1.0

    # neuron-specific constants
    rest_v = -60.0
    reset_v = -65.0
    thresh_v = -50.0
    refrac_t = 3.0
    time_constant = 20.0

    # updater-specific constants
    lrpre = 0.001
    lrpost = 0.001
    tcpre = 20.0
    tcpost = 20.0

    # create random generator and dependent values
    rng = torch.Generator().manual_seed(seed)
    eoff = 1
    xoff = 2

    # create components
    encd = snn.HomogeneousPoissonEncoder(
        n_steps,
        step_time,
        f_max_spike,
        refrac=0.0,
        compensate=False,
        generator=torch.Generator(device=device).manual_seed(seed + eoff),
    )

    neur = snn.LIF(
        (n_neurons,),
        step_time,
        rest_v=rest_v,
        reset_v=reset_v,
        thresh_v=thresh_v,
        refrac_t=refrac_t,
        time_constant=time_constant,
        batch_size=batch_sz,
    )

    conn = snn.LinearDense(
        (n_neurons,),
        (n_neurons,),
        step_time,
        synapse=snn.DeltaCurrent.partialconstructor(time_constant),
        batch_size=batch_sz,
        weight_init=lambda w, wmin=wmin, wmax=wmax, r=rng: inferno.rescale(
            inferno.uniform(w, generator=r), wmin, wmax, srcmin=0.0, srcmax=1.0
        ),
    )
    conn.updater = conn.defaultupdater()

    layer = snn.Serial(conn, neur)

    clamp = snn.Clamping(conn.updater, "parent.weight", min=wmin, max=wmax)
    trainer = STDP(
        step_time, lr_post=lrpost, lr_pre=-lrpre, tc_post=tcpost, tc_pre=tcpre
    )

    # enable training mode and register
    layer.train()
    trainer.train()

    _ = trainer.register_cell("ff", layer.cell)

    # move to the designated device
    rng = torch.Generator(device=device).manual_seed(seed + xoff)
    layer = layer.to(device=device)
    clamp = clamp.to(device=device)
    trainer = trainer.to(device=device)

    # create synchronization function
    match device.lower().partition(":")[0]:
        case "cpu":

            def sync(device):
                pass

        case "cuda":

            def sync(device):
                torch.cuda.synchronize(device=device)

        case "mps":

            def sync(device):
                torch.mps.synchronize()

        case _:
            raise RuntimeError(f"bad device '{device}' specified")

    # perform the benchmark
    results = []
    for n in range(n_runs + n_warmup):
        # reset model state
        layer.neuron.clear()

        # generate inputs
        inputs = torch.rand(batch_sz, n_neurons, device=device, generator=rng)

        # start timing
        sync(device)
        ti = time.perf_counter()

        # process inputs
        with torch.no_grad():
            for spikes in encd(inputs):
                _ = layer(spikes)
                trainer()
                layer.cell.connection.update()

        # end timing
        sync(device)
        tf = time.perf_counter()

        # add to results if not warming up (seconds to milliseconds)
        if n >= n_warmup:
            results.append((tf - ti) * 1e3)

    # return results
    return {
        "mean": sum(results) / len(results),
        "total": sum(results),
    }


def minisweep(n_runs: int, device: str, n_warmup: int):
    n_steps = 1000
    batch_sz = 1
    f_max_spike = 250
    seed = 271828
    n_neurons = [*range(100, 500 + 1, 100)]

    res = {}
    res["info"] = {}
    res["info"]["bench"] = "stdp-pdl"
    res["info"]["device"] = device
    res["info"]["n_trials"] = n_runs
    res["info"]["n_warmup"] = n_warmup
    res["results"] = {}

    print(f"STDP-PDL Sweep ({device}), {n_runs}+{n_warmup} Runs")
    for nneur in tqdm(n_neurons):
        res["results"][nneur] = pdl(
            n_runs,
            n_steps,
            nneur,
            batch_sz,
            device,
            seed,
            f_max_spike,
            n_warmup,
        )
    return res


def ssweep(n_runs: int, device: str, n_warmup: int):
    n_steps = 1000
    batch_sz = 1
    f_max_spike = 250
    seed = 271828
    n_neurons = [*range(100, 500, 100)] + [*range(500, 4000 + 1, 500)]

    res = {}
    res["info"] = {}
    res["info"]["bench"] = "stdp-pdl"
    res["info"]["device"] = device
    res["info"]["n_trials"] = n_runs
    res["info"]["n_warmup"] = n_warmup
    res["results"] = {}

    print(f"STDP-PDL Sweep ({device}), {n_runs}+{n_warmup} Runs")
    for nneur in tqdm(n_neurons):
        res["results"][nneur] = pdl(
            n_runs,
            n_steps,
            nneur,
            batch_sz,
            device,
            seed,
            f_max_spike,
            n_warmup,
        )
    return res


def msweep(n_runs: int, device: str, n_warmup: int):
    n_steps = 1000
    batch_sz = 1
    f_max_spike = 250
    seed = 271828
    n_neurons = [*range(5000, 10000 + 1, 1000)]

    res = {}
    res["info"] = {}
    res["info"]["bench"] = "stdp-pdl"
    res["info"]["device"] = device
    res["info"]["n_trials"] = n_runs
    res["info"]["n_warmup"] = n_warmup
    res["results"] = {}

    print(f"STDP-PDL Sweep ({device}), {n_runs}+{n_warmup} Runs")
    for nneur in tqdm(n_neurons):
        res["results"][nneur] = pdl(
            n_runs,
            n_steps,
            nneur,
            batch_sz,
            device,
            seed,
            f_max_spike,
            n_warmup,
        )
    return res


def testrun():
    dd = minisweep(10, "cpu", 5)
    with open("inf-stdpres-mini-cpu.json", "w") as f:
        json.dump(dd, f)


if __name__ == "__main__":
    dd = ssweep(10, "cpu", 5)
    with open("inf-stdpres-s-cpu.json", "w") as f:
        json.dump(dd, f)

    if len(sys.argv) == 1 or sys.argv[1] != "nogpu":
        dd = ssweep(10, "cuda:0", 5)
        with open("inf-stdpres-s-cuda.json", "w") as f:
            json.dump(dd, f)

        dd = msweep(10, "cuda:0", 5)
        with open("inf-stdpres-m-cuda.json", "w") as f:
            json.dump(dd, f)
