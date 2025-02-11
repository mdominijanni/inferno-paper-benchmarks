import json
import numpy as np
import matplotlib.pyplot as plt


def get_axes(
    filename: str,
    outer: list[str],
    inner: list[str],
    exclude: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    # read-in file
    with open(filename, "r") as f:
        dd = json.load(f)

    # unwrap results
    for key in outer:
        dd = dd[key]

    # unwrap each result
    X, Y = [], []
    for k, v in dd.items():
        x, y = k, v
        for key in inner:
            y = y[key]
        if x not in exclude:
            X.append(int(x))
            Y.append(float(y))

    # create numpy arrays and return
    return np.array(X), np.array(Y)


def joint_axes(
    filenames: list[str] | str, outer: list[str], inner: list[str], exclude: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(filenames, str):
        return get_axes(filenames, outer, inner, exclude)
    else:
        xy = [get_axes(fn, outer, inner, exclude) for fn in filenames]
        return np.concatenate([x for x, _ in xy]), np.concatenate([y for _, y in xy])


def plot_pdl():
    prefix = "results/"
    outer, inner = ["results"], ["mean"]
    exclude = ["12500", "40000", "50000"]
    colors = ["darkviolet", "royalblue", "teal"]
    cpu_fnames = [
        [
            "inf-pdlres-r-cpu.json",
            "inf-pdlres-lg-cpu.json",
        ],
        [
            "bn-pdlres-r-cpu.json",
            "bn-pdlres-lg-cpu.json",
        ],
        [
            "b2-pdlres-r-cpu.json",
            "b2-pdlres-lg-cpu.json",
        ],
    ]
    cpu_labels = ["Inferno", "BindsNET", "Brian2"]
    gpu_fnames = [
        [
            "inf-pdlres-r-cuda.json",
            "inf-pdlres-lg-cuda.json",
            "inf-pdlres-xl-cuda.json",
        ],
        [
            "bn-pdlres-r-cuda.json",
            "bn-pdlres-lg-cuda.json",
            "bn-pdlres-xl-cuda.json",
        ],
        [
            "b2-pdlres-r-cuda.json",
            "b2-pdlres-lg-cuda.json",
            "b2-pdlres-xl-cuda.json",
        ],
    ]
    gpu_labels = ["Inferno", "BindsNET", "Brian2CUDA"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    cax, gax = axes

    for fns, label, color in zip(gpu_fnames, gpu_labels, colors):
        X, Y = joint_axes([prefix + fn for fn in fns], outer, inner, exclude)
        gax.plot(X, Y, color=color, linestyle="-", marker="|", label=label)

    gax.set_xticks([100, 5000, 10000, 20000, 30000])
    gax.set_title("PDL Benchmark (CUDA)")
    gax.legend(loc=2)

    for fns, label, color in zip(cpu_fnames, cpu_labels, colors):
        X, Y = joint_axes([prefix + fn for fn in fns], outer, inner, exclude)
        cax.plot(X, Y, color=color, linestyle="-", marker="|", label=label)

    cax.set_xticks([100, 2000, 5000, 10000, 15000])
    cax.set_title("PDL Benchmark (CPU)")
    cax.legend(loc=2)

    fig.supxlabel("Simulated Neurons (count)")
    fig.supylabel("Execution Time (ms)")

    fig.tight_layout()
    fig.savefig("pdl.png", dpi=600)


def plot_stdp():
    prefix = "results/"
    outer, inner = ["results"], ["mean"]
    exclude = []
    colors = ["darkviolet", "royalblue", "teal"]
    cpu_fnames = [
        [
            "inf-stdpres-s-cpu.json",
        ],
        [
            "bn-stdpres-s-cpu.json",
        ],
        [
            "b2-stdpres-s-cpu.json",
        ],
    ]
    cpu_labels = ["Inferno", "BindsNET", "Brian2"]
    gpu_fnames = [
        [
            "inf-stdpres-s-cuda.json",
            "inf-stdpres-m-cuda.json",
        ],
        [
            "bn-stdpres-s-cuda.json",
            "bn-stdpres-m-cuda.json",
        ],
        [
            "b2-stdpres-s-cuda.json",
            "b2-stdpres-m-cuda.json",
        ],
    ]
    gpu_labels = ["Inferno", "BindsNET", "Brian2CUDA"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    cax, gax = axes

    for fns, label, color in zip(gpu_fnames, gpu_labels, colors):
        X, Y = joint_axes([prefix + fn for fn in fns], outer, inner, exclude)
        gax.plot(X, Y, color=color, linestyle="-", marker="|", label=label)

    gax.set_title("STDP + PDL Benchmark (CUDA)")
    gax.legend(loc=2)

    for fns, label, color in zip(cpu_fnames, cpu_labels, colors):
        X, Y = joint_axes([prefix + fn for fn in fns], outer, inner, exclude)
        cax.plot(X, Y, color=color, linestyle="-", marker="|", label=label)

    cax.set_title("STDP + PDL Benchmark (CPU)")
    cax.legend(loc=2)

    fig.supxlabel("Simulated Neurons (count)")
    fig.supylabel("Execution Time (ms)")

    fig.tight_layout()
    fig.savefig("stdp.png", dpi=600)


if __name__ == "__main__":
    plot_pdl()
    plot_stdp()
