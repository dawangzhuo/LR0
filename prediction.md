# LR0: Predictions Before the Formal Experiments

## Prior exploratory work

Before defining the formal experiment series, I ran preliminary
training and hyperparameter-search programs with a different
experimental setup.

I also checked the prescribed baseline implementation on the
validation split. With training seed 0, its validation accuracy
was 95%. Repeated seed-0 runs produced identical training metrics
and predictions, while seed 1 produced different results.

These development runs are not included in the formal result tables.
The predictions below are informed by this preliminary work.

## Fixed experimental conditions

- Dataset: scikit-learn load_digits.
- Split seed: 0.
- Split sizes: train 1077, validation 360, test 360.
- Pixel preprocessing: divide by 16.
- Training epochs: 30.
- Device: CPU.
- Number of CPU threads: 1.
- Data split indices and SHA-256 are stored in data/split.json
  and data/manifest.json.

## Experiment 1: Repeatability and variation

I expect the same training seed to produce identical predictions
and training metrics, excluding measured execution times.

For the baseline with training seeds 1 through 5, I expect
validation accuracy to be approximately 94% to 97%, with a
maximum-minus-minimum range of about 0.5 to 2 percentage points.

## Experiment 2: Hyperparameter search

I expect the best completed trial among 40 trials to improve
validation accuracy by approximately 1 to 3 percentage points
relative to the baseline.

I expect pruning to reduce the number of executed epochs.
Its measured time savings may differ from its epoch savings,
because configurations have different computational costs.

## Experiment 3: Comparison with new seeds

For training seeds 101 through 105, I expect the selected
configuration to outperform the baseline in most paired runs.

I expect its average validation accuracy with new seeds to be
lower than the best search-trial accuracy because of selection
bias and training randomness.

I do not assume that every test difference will be positive.
The conclusion will depend on all five paired differences
and their relationship to the baseline variation.

## Test-set policy

The prescribed test split will not be evaluated during development
or hyperparameter selection. It will be evaluated only in the
predefined final series after configurations are fixed.

Earlier exploratory test results used a different split and
are not treated as the final evaluation for this protocol.