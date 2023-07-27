# conlloovia_exp
Experiments for the paper presenting
[Conlloovia](https://github.com/asi-uniovi/conlloovia). All the result files from the
experiments to generate the figures and the figures themselves are stored here. In
addition, the code to regenerate the results and the figures is also included.

Install the conlloovia package following the instructions in [its
repository](https://github.com/asi-uniovi/conlloovia).

There are three sets of experiments: about problem **sizes**, about the influence of the
window size with **synthetic** traces and about the influence of the window size with
real traces coming from **alibaba**. They are described below.

## Problem sizes

The experiments and the figures are contained in the [`ExpSize.ipynb`](ExpSize.ipynb)
notebook.

Running this notebook will generate figures in the [`figs`](figs) directory. In addition,
it will generate a summary of the unrestricted experiments in the
`exp_size_unrestricted.csv` file and a summary of the restricted experiments in the
`exp_size_restricted.csv` file.

## Synthetic traces

First, run the experiments:

```bash
python exp_synth.py
```

The solutions of all experiments will be stored in the `sols` directory, inside a
subdirectory with the date of the execution. A `summary.csv` file will be generated in the
root directory. As the solutions are not needed for plotting the figures, they are not
included in the repository.

After running the experiments, run the notebook [`ExpSynth.ipynb`](ExpSynth.ipynb) to
generate the figures in the `figs` directory using the `summary.csv` file.

## Alibaba traces

First, run the experiments:

```bash
python exp_alibaba.py
```

This will be generate the pickle file
`summary_sol_gap_0_max_sec_120_limit_20_alibaba_gpu_p95.p`, which can be processed with
the notebook [`ExpAlibaba.ipynb`](ExpAlibaba.ipynb) to generate the figures in the `figs` directory.

Additionally, the process to generate the traces for two applications from the original
Alibaba traces can be found in the notebook
[`AlibabaTraceProcessing.ipynb`](traces/alibaba/AlibabaTraceProcessing.ipynb) inside the
[`traces/alibaba`](traces/alibaba) directory.

