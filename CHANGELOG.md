# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2024-05-10

### Added

- Added parameter to use CPU for inference.

## [1.1.0] - 2024-02-09

### Changed

- When training a new CENsible model, the code now saves the Pearson correlation
  coefficients (per epoch) on the training-set data, in addition to the
  testing-set data as before. This change aims to help users diagnose
  overfitting and other issues during training.

## [1.0.0] - 2023-10-21

Initial release!
