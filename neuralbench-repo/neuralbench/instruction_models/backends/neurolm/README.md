# NeuroLM backend

This integration keeps the official NeuroLM source in a separate checkout and
uses NeuralBench for dataset loading, subject splits, targets, and metrics.
It supports single-task generation evaluation and joint instruction tuning
over the seven EEG tasks used by the local BLPM study:

- PhysioNet motor imagery (motor_imagery/schalk2004bci2000)
- TUAB (pathology)
- TUEV (clinical_event, converted to the paper-style single label)
- HMC (sleep_stage/alvarez2022haaglanden)
- FACED (emotion)
- MentalArithmetic (mental_arithmetic)
- COG-BCI (mental_workload)

Evaluation follows official NeuroLM's closed-set generative protocol rather
than a linear or candidate-scoring head. The model greedily generates up to
five answer tokens (`top_k=1`), then the generated `(A)`, `Yes`, or `No` is
mapped back to the NeuralBench class. TUAB, TUEV, and HMC use the official
prompt/answer conventions. Tasks not present in the official NeuroLM suite use
the same convention: natural-language answers for binary questions and
`Answer: (` choice continuations for multiclass questions.

The model-specific EEG path resamples to 200 Hz, applies 0.1-75 Hz filtering
and a 50 Hz notch, converts volts to microvolts, and applies NeuroLM's official
division by 100 before 200-sample tokenization. Dataset-specific channel picks,
splits, windows, and labels remain owned by NeuralBench.

Install the optional dependencies from neuralbench-repo:

```bash
pip install -e ".[neurolm]"
```

The common D:/myeongju/NeuroLM workspace layout is auto-detected. For another
layout, pass --official-source-root, --checkpoint, and --text-data-dir, or set
NEUROLM_SOURCE_ROOT, NEUROLM_CHECKPOINT, and NEUROLM_TEXT_DATA_DIR.

Prepare all seven NeuralBench caches:

```bash
neuralbench eeg all --model neurolm --multi-task --prepare
```

Run a short one-epoch/two-batches-per-task integration check:

```bash
neuralbench eeg all --model neurolm --multi-task --debug --gpus 0,1
```

Run full joint instruction tuning:

```bash
neuralbench eeg all --model neurolm --multi-task --gpus 0,1,2,3
```

The full run defaults to five epochs and writes an auto-resumable checkpoint
to outputs/checkpoints/neurolm-7task/ckpt.pt. --force preserves the prior
checkpoint with a timestamp before starting over. By default training includes
the auxiliary OpenWebText loss used by official NeuroLM instruction tuning;
--no-text-loss is available only for an explicitly ablated run.

Evaluate one task from an instruction-tuned checkpoint:

```bash
neuralbench eeg pathology --model neurolm \
  --checkpoint outputs/checkpoints/neurolm-7task/ckpt.pt
```

Multi-GPU execution uses one complete model replica per GPU (DDP/data
parallelism), as in the official training script. It is not model parallelism,
so the selected checkpoint must fit on every individual GPU.
