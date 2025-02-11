from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.network.topology import Connection
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

    # create random generator
    rng = torch.Generator().manual_seed(seed)
    eoff = 1
    xoff = 2

    # create components
    encd = PoissonEncoder(
        n_steps,
        step_time,
        approx=False,
        device=device,
    )

    inpt = Input(
        shape=(n_neurons,),
        traces=True,
        traces_additive=True,
        tc_trace=tcpre,
    )

    neur = LIFNodes(
        shape=(n_neurons,),
        traces=True,
        traces_additive=True,
        tc_trace=tcpost,
        thresh=thresh_v,
        rest=rest_v,
        reset=reset_v,
        refrac=(refrac_t // step_time),
        tc_decay=time_constant,
    )

    conn = Connection(inpt, neur, update_rule=PostPre, nu=(lrpre, lrpost))
    conn.w.data = (
        torch.rand(
            conn.w.data.shape,
            generator=rng,
            dtype=conn.w.data.dtype,
            layout=conn.w.data.layout,
            device=conn.w.data.device,
            requires_grad=conn.w.data.requires_grad,
        )
        * (wmax - wmin)
    ) + wmin

    network = Network(dt=step_time, batch_size=batch_sz, learning=True)
    network.add_layer(inpt, name="X")
    network.add_layer(neur, name="Y")
    network.add_connection(conn, "X", "Y")

    # move to the designated device
    network = network.to(device=device)
    rng = torch.Generator(device=device).manual_seed(seed + xoff)

    # set global random seed since a generator can't be passed in
    torch.manual_seed(seed + eoff)

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
        network.reset_state_variables()

        # generate inputs
        inputs = (
            torch.rand(batch_sz, n_neurons, device=device, generator=rng) * f_max_spike
        )

        # start timing
        sync(device)
        ti = time.perf_counter()

        # process inputs
        with torch.no_grad():
            _ = network.run({"X": encd(inputs)}, time=n_steps)

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


def minisweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
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
            n_runs, n_steps, nneur, batch_sz, device, seed, f_max_spike, n_warmup
        )
    return res


def ssweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
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
            n_runs, n_steps, nneur, batch_sz, device, seed, f_max_spike, n_warmup
        )
    return res


def msweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
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
            n_runs, n_steps, nneur, batch_sz, device, seed, f_max_spike, n_warmup
        )
    return res


def testrun():
    dd = minisweep(10, "cpu", 5)
    with open("bn-stdpres-mini-cpu.json", "w") as f:
        json.dump(dd, f)


if __name__ == "__main__":
    dd = ssweep(10, "cpu", 5)
    with open("bn-stdpres-s-cpu.json", "w") as f:
        json.dump(dd, f)

    if len(sys.argv) == 1 or sys.argv[1] != "nogpu":
        dd = ssweep(10, "cuda:0", 5)
        with open("bn-stdpres-s-cuda.json", "w") as f:
            json.dump(dd, f)

        dd = msweep(10, "cuda:0", 5)
        with open("bn-stdpres-m-cuda.json", "w") as f:
            json.dump(dd, f)
