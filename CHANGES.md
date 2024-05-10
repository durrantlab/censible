# Changes

## Version 1.0 (2023-10-21)

The initial version.

## Version 1.1 (2024-02-09)

- When training a new CENsible model, the code now saves the Pearson correlation
  coefficients (per epoch) on the training-set data, in addition to the
  testing-set data as before. This change aims to help users diagnose
  overfitting and other issues during training.

## Version 1.2 (2024-05-10)

- Added parameter to use CPU for inference.
