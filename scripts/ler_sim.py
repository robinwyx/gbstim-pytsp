import argparse
import pickle as pkl
import parse
import sinter

from gbstim.bposd import BPOSD
from gbstim.gb import CustomUnpickler
from time import time


def generate_tasks(code, p_range, idle, w, t1, t2):
    tasks = []
    for p in p_range:
        tasks.append(
            sinter.Task(
                circuit=code.stim_circ(
                    p, p, p, t1, t2, dec_type="Z", idle=idle, num_rounds=1
                ),
                json_metadata={
                    "p": p,
                    "Code": f"[{code.n}, {code.k}, {code.d}] Weight {w}",
                },
            )
        )
    return tasks


def main():
    start_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("code")
    parser.add_argument("-p", nargs="+", type=float)
    parser.add_argument("-i", "--idle", action="count", default=0)
    parser.add_argument("-t1", type=float, default=1e7)
    parser.add_argument("-t2", type=float, default=1e7)
    args = parser.parse_args()
    n, k, d, w = parse.parse("codes/{}-{}-{}-w-{}.pkl", args.code)
    code = pkl.load(open(args.code, "rb"))
    print(f"Simulating code: [{code.n}, {code.k}, {code.d}] Weight {w}")

    tasks = generate_tasks(code, args.p, args.idle, w, args.t1, args.t2)
    samples = sinter.collect(
        num_workers=48,
        max_shots=1_000_000,
        max_errors=50,
        tasks=tasks,
        count_observable_error_combos=True,
        decoders=["bposd"],
        custom_decoders={
            "bposd": BPOSD(
                max_iter=10_000, bp_method="ms", osd_order=10, osd_method="osd_cs"
            )
        },
    )
    # print the results to results/n-k-d-w-samples.txt
    with open(f"results/{n}-{k}-{d}-w-{w}-samples.txt", "w") as f:
        f.write(str(samples))
    pkl.dump(samples, open(f"results/{n}-{k}-{d}-w-{w}-samples.pkl", "wb"))
    print(f"Time taken: {time() - start_time} seconds")


if __name__ == "__main__":
    main()
