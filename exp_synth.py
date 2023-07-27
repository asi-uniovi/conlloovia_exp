"""Conlloovia using synthetic traces.

- One region: us-east-1
- Instances from the c familiy
"""

from dataclasses import dataclass
from datetime import datetime
import itertools
from pathlib import Path
import pickle
from typing import Optional, TypeAlias

import pandas as pd  # type: ignore
import pint
from rich import print  # pylint: disable=redefined-builtin
from rich.table import Table
from rich.markdown import Markdown
from pulp import PULP_CBC_CMD  # type: ignore

from cloudmodel.unified.units import (
    ComputationalUnits,
    CurrencyPerTime,
    Time,
    RequestsPerTime,
    Storage,
    Currency,
)

from conlloovia import (
    App,
    InstanceClass,
    ContainerClass,
    Requests,
    System,
    Workload,
    Problem,
    ConllooviaAllocator,
    Solution,
    Status,
)
from conlloovia.visualization import SolutionPrettyPrinter, ProblemPrettyPrinter
from conlloovia.first_fit import FirstFitAllocator, FirstFitIcOrdering

# SummaryType is a dict where the first key is the number of apps and the second
# key is the name of the traces. The value is a ScenarioSummaryType, i.e., a
# dictionary where the key is an algorithm name (conlloovia, FFC or FFP)
# and the key a window size and the value, a list of SummaryStats for each time
# slot.
NumApps: TypeAlias = int
TraceName: TypeAlias = str
AllocatorName: TypeAlias = str
ScenarioSummaryType: TypeAlias = dict[Time, dict[AllocatorName, list["SummaryStats"]]]
SummaryType: TypeAlias = dict[NumApps, dict[TraceName, ScenarioSummaryType]]

ALLOCATORS = ["conlloovia", "FFC", "FFP"]

DIR_TRACES = Path("traces/synth")


@dataclass(frozen=True)
class SummaryStats:
    """Summary statistics for a solution."""

    workload: tuple[float, ...]
    cost: pint.Quantity
    num_vms: int
    num_containers: int
    sol: Solution


def get_workload(filename: str, window_size_s: int) -> list[float]:
    """Reads the workload from a filename that has it per second and aggregates
    it in the window with window_size_s seconds. Returns the value as a list."""
    df_wl_sec = pd.read_csv(filename, header=None, names=["reqs"])

    df_wl_sec.index = pd.to_datetime(df_wl_sec.index, unit="s")
    return list(
        df_wl_sec.resample(f"{window_size_s}s").quantile(q=0.95).reqs * window_size_s
    )


def create_ics(limit: int) -> tuple[InstanceClass, ...]:
    """Creates the instance classes with real data from AWS."""
    ic_tuple = (
        # c5
        InstanceClass(
            name="c5.large",
            price=CurrencyPerTime("0.085 usd/hour"),
            cores=ComputationalUnits("1 cores"),
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
            cores=ComputationalUnits("1 cores"),
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

    return ic_tuple


def create_system(num_apps: int) -> System:
    """Creates the system. It is prepared to work with 2 apps at most. It first
    creates the data for 2 apps, but then it returns only the data for the
    number of apps specified in the parameter."""
    assert num_apps in [1, 2]

    app_tuple = (App(name=f"app{0}"), App(name=f"app{1}"))

    apps = {a.name: a for a in app_tuple}

    # Conlloovia requires limits for InstanceClasses and ContainerClasses.
    # This is just a big enough number as if there were no practical limits.
    limit_ic = 30

    ic_tuple = create_ics(limit_ic)

    ics = {i.name: i for i in ic_tuple}

    cc_tuple = (
        # Data from yolov5s
        ContainerClass(
            name="cc0app0",
            cores=ComputationalUnits("0.5 cores"),
            mem=Storage("379 mebibytes"),
            app=apps["app0"],
        ),
        ContainerClass(
            name="cc1app0",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("379 mebibytes"),
            app=apps["app0"],
        ),
        ContainerClass(
            name="cc2app0",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("379 mebibytes"),
            app=apps["app0"],
        ),
        # Data from yolov5l
        ContainerClass(
            name="cc0app1",
            cores=ComputationalUnits("0.5 cores"),
            mem=Storage("562 mebibytes"),
            app=apps["app1"],
        ),
        ContainerClass(
            name="cc1app1",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("562 mebibytes"),
            app=apps["app1"],
        ),
        ContainerClass(
            name="cc2app1",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("562 mebibytes"),
            app=apps["app1"],
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

    limited_app_tuple = tuple(app_tuple[i] for i in range(num_apps))
    limited_cc_tuple = tuple(cc for cc in cc_tuple if cc.app in limited_app_tuple)
    limited_perfs = {
        (ic, cc): perfs[(ic, cc)]
        for ic in ic_tuple
        for cc in limited_cc_tuple
        if (ic, cc) in perfs
    }
    system = System(
        apps=limited_app_tuple, ics=ic_tuple, ccs=limited_cc_tuple, perfs=limited_perfs
    )
    return system


def save_sol(
    allocator_name: str,
    sol: Solution,
    time_slot: int,
    traces: tuple[str, ...],
    sol_dir: Optional[Path],
) -> None:
    """Save the solution to the given directory in a pickle file.

    Args:
        allocator_name: The name of the allocator.
        sol: The solution to save.
        time_slot: The time slot of the solution.
        traces: The names of the traces for each app.
        sol_dir: The directory where to save the solution. If None, the solution
            is not saved.
    """
    if sol_dir is None:
        return

    sol_dir.mkdir(parents=True, exist_ok=True)
    w_size = str(sol.problem.sched_time_size).replace(" ", "_")
    num_apps = len(sol.problem.system.apps)
    trace_name = "_".join(traces)
    f_name = f"sol_{allocator_name}_{w_size}_a_{num_apps}_{trace_name}_ts_{time_slot}.p"
    sol_path = sol_dir / f_name
    with sol_path.open("wb") as f:
        pickle.dump(sol, f)


def compute_num_vms_containers(sol: Solution) -> tuple[int, int]:
    """Computes the number of VMs and containers in the solution."""
    if sol.solving_stats.status not in [Status.OPTIMAL, Status.INTEGER_FEASIBLE]:
        return 0, 0

    # vms is list of booleans, True if the VM is used, False otherwise
    vms = sol.alloc.vms.values()
    num_vms = sum(vms)

    # containers is list of ints, the number of containers in each VM
    containers = sol.alloc.containers.values()
    num_containers = sum(containers)
    return num_vms, num_containers


def run_time_slot_conlloovia(
    system: System,
    window_size: Time,
    time_slot: int,
    reqs: tuple[float, ...],
    traces: tuple[str, ...],
    sol_dir: Optional[Path] = None,
) -> Optional[SummaryStats]:
    """Create, solve and print the solution of a time slot using the conlloovia
    allocator.

    Args:
        system: The system.
        window_size: The size of the time slot.
        time_slot: The index of the time slot.
        reqs: The number of requests in the time slot for each app.
        traces: The name of the traces for each app.
        sol_dir: The directory where to save the solution. If it is None, the
            solution is not saved.
    """
    # Create the workload for each app
    workloads = compose_workloads(system, window_size, reqs)

    problem = Problem(system=system, workloads=workloads, sched_time_size=window_size)

    if time_slot == 0:
        ProblemPrettyPrinter(problem).print()

    alloc_conlloovia = ConllooviaAllocator(problem)

    frac_gap = None
    max_seconds = 20
    solver = PULP_CBC_CMD(gapRel=frac_gap, threads=8, timeLimit=max_seconds, msg=False)
    sol_conlloovia = alloc_conlloovia.solve(solver)

    SolutionPrettyPrinter(sol_conlloovia).print()

    save_sol("conlloovia", sol_conlloovia, time_slot, traces, sol_dir)

    if sol_conlloovia.solving_stats.status in [
        Status.OPTIMAL,
        Status.INTEGER_FEASIBLE,
    ]:
        num_vms, num_containers = compute_num_vms_containers(sol_conlloovia)
        stats = SummaryStats(
            reqs,
            sol_conlloovia.cost,
            num_vms,
            num_containers,
            sol_conlloovia,
        )
    else:
        stats = SummaryStats(
            reqs,
            0,
            0,
            0,
            sol_conlloovia,
        )

    return stats


def run_time_slot_ffc(
    system: System,
    window_size: Time,
    time_slot: int,
    reqs: tuple[float, ...],
    traces: tuple[str, ...],
    sol_dir: Optional[Path] = None,
) -> Optional[SummaryStats]:
    """Create, solve and print the solution of a time slot using the FFC allocator.

    Args:
        system: The system.
        window_size: The size of the time slot.
        time_slot: The index of the time slot.
        reqs: The number of requests in the time slot for each app.
        traces: The name of the trace for each app.
        sol_dir: The directory where to save the solution. If it is None, the
            solution is not saved.
    """
    # Create the workload for each app
    workloads = compose_workloads(system, window_size, reqs)
    problem = Problem(system=system, workloads=workloads, sched_time_size=window_size)
    alloc_ffc = FirstFitAllocator(problem, ordering=FirstFitIcOrdering.CORE_DESCENDING)
    sol_ffc = alloc_ffc.solve()

    SolutionPrettyPrinter(sol_ffc).print()

    save_sol("ffc", sol_ffc, time_slot, traces, sol_dir)

    if sol_ffc.solving_stats.status in [
        Status.OPTIMAL,
        Status.INTEGER_FEASIBLE,
    ]:
        num_vms, num_containers = compute_num_vms_containers(sol_ffc)
        stats = SummaryStats(
            reqs,
            sol_ffc.cost,
            num_vms,
            num_containers,
            sol_ffc,
        )
        return stats

    return None


def run_time_slot_ffp(
    system: System,
    window_size: Time,
    time_slot: int,
    reqs: tuple[float, ...],
    traces: tuple[str, ...],
    sol_dir: Optional[Path] = None,
) -> Optional[SummaryStats]:
    """Create, solve and print the solution of a time slot using the FFP allocator.

    Args:
        system: The system.
        window_size: The size of the time slot.
        time_slot: The index of the time slot.
        reqs: The number of requests in the time slot for each app.
        traces: The name of the trace for each app.
        sol_dir: The directory where to save the solution. If it is None, the
            solution is not saved.
    """
    # Create the workload for each app
    workloads = compose_workloads(system, window_size, reqs)
    problem = Problem(system=system, workloads=workloads, sched_time_size=window_size)
    alloc_ffp = FirstFitAllocator(problem, ordering=FirstFitIcOrdering.PRICE_ASCENDING)
    sol_ffp = alloc_ffp.solve()

    SolutionPrettyPrinter(sol_ffp).print()

    save_sol("ffp", sol_ffp, time_slot, traces, sol_dir)

    if sol_ffp.solving_stats.status in [
        Status.OPTIMAL,
        Status.INTEGER_FEASIBLE,
    ]:
        num_vms, num_containers = compute_num_vms_containers(sol_ffp)
        stats = SummaryStats(
            reqs,
            sol_ffp.cost,
            num_vms,
            num_containers,
            sol_ffp,
        )
        return stats

    return None


def compose_workloads(
    system: System, window_size, reqs: tuple[float, ...]
) -> dict[App, Workload]:
    """Compose the workload for each app.

    Args:
        system: The system.
        window_size: The size of the time slot.
        reqs: The number of requests in the time slot for each app.

    Returns:
        A dictionary that maps each app to its workload.
    """
    workloads = {}
    for app, req in zip(system.apps, reqs):
        wl_ts = Workload(
            num_reqs=Requests(f"{req} reqs"),
            time_slot_size=window_size,
            app=app,
        )
        workloads[app] = wl_ts
    return workloads


def print_summary_scenario_per_window(
    scenario_summary: ScenarioSummaryType, traces: tuple[str, ...]
) -> None:
    """Print the summary of a scenario per window size. This prints several
    tables: one per window size and allocator with all the time slots, and one
    with the comparison of all allocators for each window size."""
    window_sizes = list(scenario_summary.keys())
    allocators = scenario_summary[window_sizes[0]].keys()

    # This table shows a comparison of the cost and number of VMs for each
    # allocator and window size
    table_windows = Table(
        "Window size",
        title=f"Window size comparison - Trace: {traces}",
    )

    # These are the parameters we are going to show in the table for each
    # allocator
    params = ["Cost", "VMs"]
    for param in params:
        for allocator in allocators:
            table_windows.add_column(f"{param} {allocator}")

    for window_size in window_sizes:
        totals = {col: {} for col in params}
        for allocator in allocators:
            time_slot_summaries = scenario_summary[window_size][allocator]
            table = Table(
                "Workload",
                "Cost",
                "VMs",
                "Containers",
                "Status",
                title=f"{allocator} - Window size: {window_size}. Trace: {traces}",
            )
            totals["Cost"][allocator] = 0
            totals["VMs"][allocator] = 0

            for ts_summary in time_slot_summaries:
                if ts_summary is None:
                    print(f"WARNING: No solution for {allocator} in {window_size}")
                    continue

                if ts_summary.sol.solving_stats.status == Status.OPTIMAL:
                    status = "Optimal"
                elif ts_summary.sol.solving_stats.status == Status.INTEGER_FEASIBLE:
                    bound = Currency(f"{ts_summary.sol.solving_stats.lower_bound} usd")
                    if bound == 0:
                        gap = 0
                    else:
                        gap = 100 * ((ts_summary.sol.cost - bound) / bound).magnitude
                    status = f"Feasible (bound: {bound}. gap: {gap:.2f}%)"
                else:
                    status = str(ts_summary.sol.solving_stats.status)

                table.add_row(
                    str(ts_summary.workload),
                    str(ts_summary.cost),
                    str(ts_summary.num_vms),
                    str(ts_summary.num_containers),
                    status,
                )
                totals["Cost"][allocator] += ts_summary.cost
                totals["VMs"][allocator] += ts_summary.num_vms

            table.add_section()
            table.add_row(
                "Total", str(totals["Cost"][allocator]), str(totals["VMs"][allocator])
            )
            print(table)

        extra_info = []
        for param in ["Cost", "VMs"]:
            for allocator in allocators:
                extra_info.append(str(totals[param][allocator]))

        table_windows.add_row(
            str(window_size),
            *extra_info,
        )

    print(table_windows)


def run_scenario(
    system: System,
    scenario_summary: ScenarioSummaryType,
    traces: tuple[str, ...],
    window_size: Time,
    prog_scenario: str,
    sol_dir: Optional[Path] = None,
) -> None:
    """Run a scenario, i.e., a combination of a number of apps, traces and window_size."""
    print(
        Markdown(
            f"# {prog_scenario} Using {len(traces)} apps with traces {traces} "
            f"and window size {window_size}"
        )
    )

    window_size_s = window_size.to("s").magnitude
    workloads = tuple(
        get_workload(f"{DIR_TRACES}/wl_{trace}_1s.csv", window_size_s)
        for trace in traces
    )

    # Check that all workloads have the same length
    assert all(len(wl) == len(workloads[0]) for wl in workloads)

    workload_len = len(workloads[0])

    # Each item is a SummaryStats
    scenario_summary[window_size] = {name: [] for name in ALLOCATORS}

    for timeslot in range(workload_len):
        print()
        print(Markdown("---"))
        prog_ts = f"[{timeslot+1}/{workload_len} time slots]"
        reqs = tuple(wl[timeslot] for wl in workloads)
        print(
            f"{prog_scenario} {prog_ts} Solving for {reqs} reqs "
            f"(window_size: {window_size}, trace: {traces})"
        )
        print(Markdown("---"))

        stats = {}
        stats["conlloovia"] = run_time_slot_conlloovia(
            system, window_size, timeslot, reqs, traces, sol_dir
        )
        stats["FFC"] = run_time_slot_ffc(
            system, window_size, timeslot, reqs, traces, sol_dir
        )
        stats["FFP"] = run_time_slot_ffp(
            system, window_size, timeslot, reqs, traces, sol_dir
        )

        for allocator, alloc_stats in stats.items():
            scenario_summary[window_size][allocator].append(alloc_stats)

    print_summary_scenario_per_window(scenario_summary, traces)


def print_per_scenario_summary_table(summary: SummaryType) -> None:
    """Print a table with the summary of the scenarios."""
    table = Table(
        "Apps",
        "Traces",
        "Window size",
        title="Summary",
    )
    params = ["Cost", "VMs"]
    for param in params:
        for allocator in ALLOCATORS:
            table.add_column(f"{param} {allocator}")

    for num_apps, app_summary in summary.items():
        for trace_name, trace_summary in app_summary.items():
            for window_size, summaries in trace_summary.items():
                totals: dict[str, dict[str, float]] = {}
                for param in params:
                    totals[param] = {allocator: 0 for allocator in ALLOCATORS}
                for allocator in ALLOCATORS:
                    for stats in summaries[allocator]:
                        if not stats:
                            # This means that the algorithm failed to find a solution
                            for total_param in totals.values():
                                total_param[allocator] = -1
                            break
                        totals["Cost"][allocator] += stats.cost
                        totals["VMs"][allocator] += stats.num_vms

                # We do two loops to print the information in the same order as the
                # columns: first the cost for all allocators, then their number of VMs
                allocators_info = []
                for allocator in ALLOCATORS:
                    allocators_info.append(f"{totals['Cost'][allocator]:.4f}")

                for allocator in ALLOCATORS:
                    vms_mean = totals["VMs"][allocator] / len(summaries[allocator])
                    allocators_info.append(f"{vms_mean:.2f}")

                table.add_row(
                    str(num_apps),
                    trace_name,
                    str(window_size),
                    *allocators_info,
                )
            table.add_section()

    print(table)


def compute_num_scenarios(
    num_apps_scenarios: list[int], trace_names: list[str], window_sizes: list[Time]
) -> int:
    """Compute the number of scenarios.

    Args:
        num_apps_scenarios: List of number of apps in each scenario.
        trace_names: List of trace names.
        window_sizes: List of window sizes.
    """
    num_scenarios = 0
    for num_apps in num_apps_scenarios:
        num_traces = len(list(itertools.combinations(trace_names, num_apps)))
        num_scenarios += num_traces * len(window_sizes)
    return num_scenarios


def summary_to_df(summary: SummaryType) -> pd.DataFrame:
    """Convert the summary to a pandas DataFrame."""
    df = pd.DataFrame(
        columns=["num_apps", "trace", "window_size_sec", "allocator", "cost_usd", "vms"]
    )
    for num_apps, app_summary in summary.items():
        for trace_name, trace_summary in app_summary.items():
            for window_size, summaries in trace_summary.items():
                for allocator, alloc_stats in summaries.items():
                    for alloc_stat in alloc_stats:
                        if alloc_stat:
                            try:
                                cost = alloc_stat.cost.to("usd").magnitude
                            except Exception as ex:
                                print(
                                    f"Error converting cost to usd: {alloc_stat.cost}"
                                    f" status: {alloc_stat.sol.solving_stats.status}"
                                    f" allocator: {allocator}"
                                    f" exception: {ex}"
                                )
                                print(
                                    f" (num_apps: {num_apps}, trace: {trace_name}, "
                                    f"window_size: {window_size}, allocator: {allocator})"
                                )
                                cost = alloc_stat.cost
                            df = pd.concat(
                                [
                                    df,
                                    pd.DataFrame(
                                        [
                                            [
                                                num_apps,
                                                trace_name,
                                                window_size.to("s").magnitude,
                                                allocator,
                                                cost,
                                                alloc_stat.num_vms,
                                            ]
                                        ],
                                        columns=df.columns,
                                    ),
                                ]
                            )
    return df


def main() -> None:
    """Run the main function."""
    num_apps_scenarios = [1, 2]

    trace_names = [
        "static",
        "increasing",
        "decreasing",
        "periodic",
        "unpredictable",
        "once",
        "everything",
    ]

    window_sizes = [Time("5 min"), Time("15 min"), Time("1 hour")]

    num_scenarios = compute_num_scenarios(num_apps_scenarios, trace_names, window_sizes)

    num_scenario = 0
    summary: SummaryType = {}

    # Create sol_dir name using the date and time. The idea is saving the solutions and
    # the summary in a way that they are not overwritten.
    sol_dir = Path(f"sols/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Saving solutions to {sol_dir}")

    for num_apps in num_apps_scenarios:
        print(Markdown(f"# Scenarios with {num_apps} apps"))
        system = create_system(num_apps)
        summary[num_apps] = {}

        traces_app_scenario = itertools.combinations(trace_names, num_apps)

        for traces in traces_app_scenario:
            trace_name = "/".join(traces)
            print(Markdown(f"# Trace {trace_name}"))
            summary[num_apps][trace_name] = {}
            for window_size in window_sizes:
                num_scenario += 1
                prog_scenario = f"[{num_scenario}/{num_scenarios} scenarios]"
                run_scenario(
                    system=system,
                    scenario_summary=summary[num_apps][trace_name],
                    traces=traces,
                    window_size=window_size,
                    prog_scenario=prog_scenario,
                    sol_dir=sol_dir,
                )

    print_per_scenario_summary_table(summary)

    pickle.dump(summary, open(sol_dir / "summary.p", "wb"))

    df = summary_to_df(summary)
    df.to_csv(sol_dir / "summary.csv", index=False)

    # Copy it also to the root directory so that it is used by the notebook
    df.to_csv("summary.csv", index=False)


if __name__ == "__main__":
    main()
