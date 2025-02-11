from bindsnet.encoding import PoissonEncoder
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
    winit_min = 0.0
    winit_max = 1.0

    # neuron-specific constants
    rest_v = -60.0
    reset_v = -65.0
    thresh_v = -50.0
    refrac_t = 3.0
    time_constant = 20.0

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
        traces=False,
    )

    neur = LIFNodes(
        shape=(n_neurons,),
        traces=False,
        thresh=thresh_v,
        rest=rest_v,
        reset=reset_v,
        refrac=(refrac_t // step_time),
        tc_decay=time_constant,
    )

    conn = Connection(inpt, neur)
    conn.w.data = (
        torch.rand(
            conn.w.data.shape,
            generator=rng,
            dtype=conn.w.data.dtype,
            layout=conn.w.data.layout,
            device=conn.w.data.device,
            requires_grad=conn.w.data.requires_grad,
        )
        * (winit_max - winit_min)
    ) + winit_min

    network = Network(dt=step_time, batch_size=batch_sz, learning=False)
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


def sweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
    n_steps = 1000
    batch_sz = 1
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
            n_runs, n_steps, nneur, batch_sz, device, seed, f_max_spike, n_warmup
        )
    return res


def lsweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
    n_steps = 1000
    batch_sz = 1
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
            n_runs, n_steps, nneur, batch_sz, device, seed, f_max_spike, n_warmup
        )
    return res


def xsweep(
    n_runs: int,
    device: str,
    n_warmup: int,
):
    n_steps = 1000
    batch_sz = 1
    f_max_spike = 250
    seed = 271828
    n_neurons = [*range(20000, 30000 + 1, 5000)]

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
            n_runs, n_steps, nneur, batch_sz, device, seed, f_max_spike, n_warmup
        )
    return res


if __name__ == "__main__":
    dd = sweep(10, "cpu", 5)
    with open("bn-pdlres-r-cpu.json", "w") as f:
        json.dump(dd, f)

    dd = lsweep(10, "cpu", 5)
    with open("bn-pdlres-lg-cpu.json", "w") as f:
        json.dump(dd, f)

    if len(sys.argv) == 1 or sys.argv[1] != "nogpu":
        dd = sweep(10, "cuda:0", 5)
        with open("bn-pdlres-r-cuda.json", "w") as f:
            json.dump(dd, f)

        dd = lsweep(10, "cuda:0", 5)
        with open("bn-pdlres-lg-cuda.json", "w") as f:
            json.dump(dd, f)

        dd = xsweep(10, "cuda:0", 5)
        with open("bn-pdlres-xl-cuda.json", "w") as f:
            json.dump(dd, f)
