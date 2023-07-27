"""Conlloovia application. The data were measured in AWS. We'll have this
configuration:

- One region: us-east-1
- Instances from the c familiy
- Two apps: yolov5s and yolov5l
- The traces are from Alibaba (GPU), processed to represent two apps
"""

from pathlib import Path
import pickle
from typing import Tuple
from dataclasses import dataclass
from functools import cache

from frozendict import frozendict
import pandas as pd
import pint
from rich import print
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress
from pulp import PULP_CBC_CMD  # type: ignore

from cloudmodel.unified.units import (
    ComputationalUnits,
    Currency,
    CurrencyPerTime,
    Time,
    Requests,
    RequestsPerTime,
    Storage,
)

from conlloovia import (
    App,
    InstanceClass,
    ContainerClass,
    System,
    Workload,
    Problem,
    ConllooviaAllocator,
    Status,
    Solution,
)
from conlloovia.first_fit import FirstFitAllocator, FirstFitIcOrdering

DIR_TRACES = Path("traces/alibaba")


@dataclass(frozen=True)
class SummaryStats:
    workload: tuple[float, float]
    cost: pint.Quantity
    num_vms: int
    num_containers: int
    sol: Solution


def get_alibaba_workloads(resample_period: str) -> Tuple[list, list]:
    """Receives a resample period ("1s", "1T", "5T", "15T" or "60T") and reads
    the corresponding workload. It returns a tuple with two lists, one for
    yolo_s (app 0 in the trace) and the other for yolo_l (app 1).
    """
    trace_file_name = f"alibaba_2apps_{resample_period}_p95"
    extension = ".csv" if not resample_period == "1s" else ".csv.gz"
    df = pd.read_csv(DIR_TRACES / (trace_file_name + extension))

    wl_s = list(df.app0)
    wl_l = list(df.app1)

    return wl_s, wl_l


@cache
def solve_conlloovia(
    app_list,
    ic_list,
    cc_list,
    perfs,
    workloads,
    sched_time_size,
    frac_gap: float,
    max_seconds: int,
) -> Solution:
    """Solve the problem using Conlloovia."""
    system = System(apps=app_list, ics=ic_list, ccs=cc_list, perfs=perfs)
    problem = Problem(
        system=system, workloads=workloads, sched_time_size=sched_time_size
    )
    alloc = ConllooviaAllocator(problem)

    solver = PULP_CBC_CMD(gapRel=frac_gap, threads=7, timeLimit=max_seconds, msg=False)
    # solver = PULP_CBC_CMD(gapRel=frac_gap, threads=7)
    sol = alloc.solve(solver)
    return sol


@cache
def solve_first_fit(
    app_list,
    ic_list,
    cc_list,
    perfs,
    workloads,
    sched_time_size,
    ordering: FirstFitIcOrdering,
) -> Solution:
    """Solve the problem using First Fit."""
    system = System(apps=app_list, ics=ic_list, ccs=cc_list, perfs=perfs)
    problem = Problem(
        system=system, workloads=workloads, sched_time_size=sched_time_size
    )
    alloc = FirstFitAllocator(problem, ordering=ordering)

    sol = alloc.solve()
    return sol


def get_formatted_status(stats: SummaryStats):
    """Returns a formatted string with the status of the solution."""
    if stats.sol.solving_stats.status == Status.OPTIMAL:
        status = "Optimal"
    elif stats.sol.solving_stats.status == Status.INTEGER_FEASIBLE:
        bound = Currency(f"{stats.sol.solving_stats.lower_bound} usd")
        if bound > 0:
            gap = 100 * ((stats.sol.cost - bound) / bound).magnitude
        else:
            gap = 0
        status = f"Feasible (bound: {bound}. gap: {gap:.2f}%)"
    else:
        status = str(stats.sol.solving_stats.status)
    return status


def print_sol(
    sol: Solution, perfs, window_size, allocator_name: str, verbose: bool = True
) -> Tuple[int, int]:
    """Prints the solution and returns the number of VMs and containers."""
    num_vms = 0
    num_cores_vms = 0.0
    mem_vms = 0.0
    num_containers = 0
    num_cores_containers = 0.0
    mem_containers = 0.0

    if sol.solving_stats.status == Status.INFEASIBLE:
        print(f"[bold red]{sol.solving_stats.status}")
        return 0, 0

    print(Markdown(f"# Allocation {allocator_name}"))
    for vm, vm_is_used in sol.alloc.vms.items():
        if not vm_is_used:
            continue

        if verbose:
            print(
                f"VM {vm.ic.name}-{vm.id_} ({vm.ic.cores} {vm.ic.mem}. "
                f"Price: {vm.ic.price}. In window: {(window_size*vm.ic.price).to_reduced_units()})"
            )
        else:
            print(f"VM {vm.ic.name}-{vm.id_}")
        num_vms += 1
        num_cores_vms += vm.ic.cores
        mem_vms += vm.ic.mem

        for cc, num_replicas in sol.alloc.containers.items():
            if num_replicas > 0 and cc.vm == vm:
                perf = perfs[(vm.ic, cc.cc)]
                if verbose:
                    print(
                        f"    {cc.cc.name} ({cc.cc.cores} {cc.cc.mem}) {num_replicas} replicas. "
                        f"perf: {perf}. In window: {(window_size*perf).to_reduced_units()}"
                    )
                else:
                    print(f"    {cc.cc.name} {num_replicas} replicas.")
                num_containers += num_replicas
                num_cores_containers += num_replicas * cc.cc.cores
                mem_containers += num_replicas * cc.cc.mem

    print(Markdown(f"# Summary {allocator_name}"))
    print(f"Total number of VMs: {num_vms} ({num_cores_vms}, {mem_vms})")
    print(
        f"Total number of containers: {num_containers} ({num_cores_containers}, {mem_containers})"
    )
    print(f"Cost: {sol.cost}")
    print(f"Stats: {sol.solving_stats}")

    return num_vms, num_containers


def main() -> None:
    """Main function."""
    # Parameters for the solve
    frac_gap = None
    max_seconds = 120

    # System data
    app_list = (
        App(name="yolov5s"),
        App(name="yolov5l"),
    )

    apps = {a.name: a for a in app_list}

    # Conlloovia requires limits for InstanceClasses and ContainerClasses.
    # This is just a big enough number as if there were no practical limits.
    limit = 20

    # Notice we are using the number of physical cores, not the number of vCPUs.
    ic_tuple = (
        # c5
        InstanceClass(
            name="c5.large",
            price=CurrencyPerTime("0.085 usd/hour"),
            cores=ComputationalUnits("1 core"),
            mem=Storage("4 gibibytes"),
            limit=limit,
        ),
        InstanceClass(
            name="c5.xlarge",
            price=CurrencyPerTime("0.17 usd/hour"),
            cores=ComputationalUnits("2 cores"),
            mem=Storage("8 gibibytes"),
            limit=limit,
        ),
        InstanceClass(
            name="c5.2xlarge",
            price=CurrencyPerTime("0.34 usd/hour"),
            cores=ComputationalUnits("4 cores"),
            mem=Storage("16 gibibytes"),
            limit=limit,
        ),
        # 6i
        InstanceClass(
            name="c6i.large",
            price=CurrencyPerTime("0.085 usd/hour"),
            cores=ComputationalUnits("1 core"),
            mem=Storage("4 gibibytes"),
            limit=limit,
        ),
        InstanceClass(
            name="c6i.xlarge",
            price=CurrencyPerTime("0.17 usd/hour"),
            cores=ComputationalUnits("2 cores"),
            mem=Storage("8 gibibytes"),
            limit=limit,
        ),
        InstanceClass(
            name="c6i.2xlarge",
            price=CurrencyPerTime("0.34 usd/hour"),
            cores=ComputationalUnits("4 cores"),
            mem=Storage("16 gibibytes"),
            limit=limit,
        ),
    )

    ics = {i.name: i for i in ic_tuple}

    cc_tuple = (
        # Data from yolov5s
        ContainerClass(
            name="cc0app0",
            cores=ComputationalUnits("0.5 cores"),
            mem=Storage("379 mebibytes"),
            app=apps["yolov5s"],
        ),
        ContainerClass(
            name="cc1app0",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("379 mebibytes"),
            app=apps["yolov5s"],
        ),
        ContainerClass(
            name="cc2app0",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("379 mebibytes"),
            app=apps["yolov5s"],
        ),
        # Data from yolov5l
        ContainerClass(
            name="cc0app1",
            cores=ComputationalUnits("0.5 cores"),
            mem=Storage("562 mebibytes"),
            app=apps["yolov5l"],
        ),
        ContainerClass(
            name="cc1app1",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("562 mebibytes"),
            app=apps["yolov5l"],
        ),
        ContainerClass(
            name="cc2app1",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("562 mebibytes"),
            app=apps["yolov5l"],
        ),
    )

    ccs = {c.name: c for c in cc_tuple}

    perfs = {
        # ------------
        # --- app0 ---
        # ------------
        # cc0
        # *** c5 ***
        (ics["c5.large"], ccs["cc0app0"]): RequestsPerTime("2.1 req/s"),
        (ics["c5.xlarge"], ccs["cc0app0"]): RequestsPerTime("2.1 req/s"),
        (ics["c5.2xlarge"], ccs["cc0app0"]): RequestsPerTime("2.1 req/s"),
        # *** c6i ***
        (ics["c6i.large"], ccs["cc0app0"]): RequestsPerTime("2.29 req/s"),
        (ics["c6i.xlarge"], ccs["cc0app0"]): RequestsPerTime("2.29 req/s"),
        (ics["c6i.2xlarge"], ccs["cc0app0"]): RequestsPerTime("2.29 req/s"),
        # cc1
        # *** c5 ***
        (ics["c5.large"], ccs["cc1app0"]): RequestsPerTime("4.3 req/s"),
        (ics["c5.xlarge"], ccs["cc1app0"]): RequestsPerTime("4.3 req/s"),
        (ics["c5.2xlarge"], ccs["cc1app0"]): RequestsPerTime("4.3 req/s"),
        # *** c6i ***
        (ics["c6i.large"], ccs["cc1app0"]): RequestsPerTime("4.71 req/s"),
        (ics["c6i.xlarge"], ccs["cc1app0"]): RequestsPerTime("4.71 req/s"),
        (ics["c6i.2xlarge"], ccs["cc1app0"]): RequestsPerTime("4.71 req/s"),
        # cc2
        # *** c5 ***
        (ics["c5.large"], ccs["cc2app0"]): RequestsPerTime("6.35 req/s"),
        (ics["c5.xlarge"], ccs["cc2app0"]): RequestsPerTime("6.35 req/s"),
        (ics["c5.2xlarge"], ccs["cc2app0"]): RequestsPerTime("6.35 req/s"),
        # *** c6i ***
        (ics["c6i.large"], ccs["cc2app0"]): RequestsPerTime("6.82 req/s"),
        (ics["c6i.xlarge"], ccs["cc2app0"]): RequestsPerTime("6.82 req/s"),
        (ics["c6i.2xlarge"], ccs["cc2app0"]): RequestsPerTime("6.82 req/s"),
        # ------------
        # --- app1 ---
        # ------------
        # cc0
        # *** c5 ***
        (ics["c5.large"], ccs["cc0app1"]): RequestsPerTime("0.46 req/s"),
        (ics["c5.xlarge"], ccs["cc0app1"]): RequestsPerTime("0.46 req/s"),
        (ics["c5.2xlarge"], ccs["cc0app1"]): RequestsPerTime("0.46 req/s"),
        # *** c6i ***
        (ics["c6i.large"], ccs["cc0app1"]): RequestsPerTime("0.5 req/s"),
        (ics["c6i.xlarge"], ccs["cc0app1"]): RequestsPerTime("0.5 req/s"),
        (ics["c6i.2xlarge"], ccs["cc0app1"]): RequestsPerTime("0.5 req/s"),
        # cc1
        # *** c5 ***
        (ics["c5.large"], ccs["cc1app1"]): RequestsPerTime("0.96 req/s"),
        (ics["c5.xlarge"], ccs["cc1app1"]): RequestsPerTime("0.96 req/s"),
        (ics["c5.2xlarge"], ccs["cc1app1"]): RequestsPerTime("0.96 req/s"),
        # *** c6i ***
        (ics["c6i.large"], ccs["cc1app1"]): RequestsPerTime("1.02 req/s"),
        (ics["c6i.xlarge"], ccs["cc1app1"]): RequestsPerTime("1.02 req/s"),
        (ics["c6i.2xlarge"], ccs["cc1app1"]): RequestsPerTime("1.02 req/s"),
        # cc2
        # *** c5 ***
        (ics["c5.large"], ccs["cc2app1"]): RequestsPerTime("1.63 req/s"),
        (ics["c5.xlarge"], ccs["cc2app1"]): RequestsPerTime("1.63 req/s"),
        (ics["c5.2xlarge"], ccs["cc2app1"]): RequestsPerTime("1.63 req/s"),
        # *** c6i ***
        (ics["c6i.large"], ccs["cc2app1"]): RequestsPerTime("1.76 req/s"),
        (ics["c6i.xlarge"], ccs["cc2app1"]): RequestsPerTime("1.76 req/s"),
        (ics["c6i.2xlarge"], ccs["cc2app1"]): RequestsPerTime("1.76 req/s"),
    }
    perfs = frozendict(perfs)

    summary: dict[pint.Quantity, list[dict[str, SummaryStats]]] = {}
    window_size_to_resample_period = {
        Time("1 min"): "1T",
        Time("5 min"): "5T",
        Time("15 min"): "15T",
        Time("1 hour"): "60T",
    }

    allocators = ["conlloovia", "FFC", "FFP"]

    for window_size in [Time("5 min"), Time("15 min"), Time("1 hour")]:
        print(Markdown(f"# Using window size {window_size}"))
        wl_s, wl_l = get_alibaba_workloads(window_size_to_resample_period[window_size])

        # Each item is a dict of SummaryStats where the key is the allocator
        summary[window_size] = []

        with Progress() as progress:
            slot_count = len(wl_s)
            slot_index = 0
            task = progress.add_task("Processing...", total=slot_count)

            for req_s, req_l in zip(wl_s, wl_l):
                print()
                print(Markdown("---"))
                progress_pct = slot_index * 100 / slot_count
                print(
                    f"Solving for {req_s} reqs for yolo_s, {req_l} reqs for yolo_l (window_size: {window_size})"
                    f" {slot_index}/{slot_count} ({progress_pct:.2f}%)"
                )
                print(Markdown("---"))

                wl_ts_s = Workload(
                    num_reqs=Requests(f"{req_s} req"),
                    time_slot_size=window_size,
                    app=apps["yolov5s"],
                )
                wl_ts_l = Workload(
                    num_reqs=Requests(f"{req_l} req"),
                    time_slot_size=window_size,
                    app=apps["yolov5l"],
                )
                workloads = {apps["yolov5s"]: wl_ts_s, apps["yolov5l"]: wl_ts_l}
                workloads = frozendict(workloads)

                sols = {}
                sols["conlloovia"] = solve_conlloovia(
                    app_list=app_list,
                    ic_list=ic_tuple,
                    cc_list=cc_tuple,
                    perfs=perfs,
                    workloads=workloads,
                    sched_time_size=window_size,
                    frac_gap=frac_gap,
                    max_seconds=max_seconds,
                )

                sols["FFC"] = solve_first_fit(
                    app_list=app_list,
                    ic_list=ic_tuple,
                    cc_list=cc_tuple,
                    perfs=perfs,
                    workloads=workloads,
                    sched_time_size=window_size,
                    ordering=FirstFitIcOrdering.CORE_DESCENDING,
                )

                sols["FFP"] = solve_first_fit(
                    app_list=app_list,
                    ic_list=ic_tuple,
                    cc_list=cc_tuple,
                    perfs=perfs,
                    workloads=workloads,
                    sched_time_size=window_size,
                    ordering=FirstFitIcOrdering.PRICE_ASCENDING,
                )

                stats_allocators = {}  # A SummaryStats for each allocator
                for allocator_name, sol in sols.items():
                    if sol.solving_stats.status in [
                        Status.OPTIMAL,
                        Status.INTEGER_FEASIBLE,
                    ]:
                        # print(yaml.dump(sol))
                        num_vms, num_containers = print_sol(
                            sol, perfs, window_size, allocator_name, verbose=False
                        )
                        stats = SummaryStats(
                            (
                                wl_ts_s.num_reqs.to("reqs").magnitude,
                                wl_ts_l.num_reqs.to("reqs").magnitude,
                            ),
                            sol.cost,
                            num_vms,
                            num_containers,
                            sol,
                        )
                        stats_allocators[allocator_name] = stats
                    else:
                        print(f"[bold red]{sol.solving_stats.status}")
                        # print(alloc.lp_problem)

                summary[window_size].append(stats_allocators)
                slot_index += 1
                progress.update(task, advance=1)

    table_windows = Table(
        "Window size",
        *[f"{allocator} total cost" for allocator in allocators],
        *[f"{allocator} avg. VMs" for allocator in allocators],
        *[f"{allocator} avg. containers" for allocator in allocators],
    )

    for window_size, summaries in summary.items():
        table = Table(
            "Workload",
            title=f"Window size: {window_size}",
        )
        for param in ["cost", "num_vms", "num_containers", "status"]:
            for allocator in allocators:
                table.add_column(f"{allocator} {param}")

        params = ["cost", "num_vms", "num_containers"]
        totals = {param: {allocator: 0 for allocator in allocators} for param in params}

        for alloc_stats in summaries:
            workload_info = tuple(wl for wl in alloc_stats["conlloovia"].workload)
            row = [
                str(workload_info),
                *[str(alloc_stats[allocator].cost) for allocator in allocators],
                *[str(alloc_stats[allocator].num_vms) for allocator in allocators],
                *[
                    str(alloc_stats[allocator].num_containers)
                    for allocator in allocators
                ],
                *[
                    get_formatted_status(alloc_stats[allocator])
                    for allocator in allocators
                ],
            ]

            for allocator in allocators:
                totals["cost"][allocator] += alloc_stats[allocator].cost
                totals["num_vms"][allocator] += alloc_stats[allocator].num_vms
                totals["num_containers"][allocator] += alloc_stats[
                    allocator
                ].num_containers

            table.add_row(*row)

        table.add_section()
        total_row = ["Total"]
        for param in params:
            for allocator in allocators:
                total_row.append(str(totals[param][allocator]))

        table.add_row(*total_row)
        print(table)

        table_windows.add_row(
            str(window_size),
            *[f"{totals['cost'][allocator]:.4f}" for allocator in allocators],
            *[
                f"{totals['num_vms'][allocator] / len(summaries):.2f}"
                for allocator in allocators
            ],
            *[
                f"{totals['num_containers'][allocator] / len(summaries):.2f}"
                for allocator in allocators
            ],
        )

    print(table_windows)

    if not frac_gap:
        frac_gap = 0
    if not max_seconds:
        max_seconds = 0
    with open(
        f"summary_sol_gap_{frac_gap}_max_sec_{max_seconds}_limit_{limit}_alibaba_gpu_p95.p",
        "wb",
    ) as pickle_file:
        pickle.dump(summary, pickle_file)


if __name__ == "__main__":
    main()
