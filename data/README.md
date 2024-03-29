# Data Directory

This directory contains all data files used by the models.

## Format

Proposed format:

- data/
  - undistorted/\<train / val / test\>
    - \<distortion parameters\>_\<image ID\>_raw.jpeg
  - distorted/\<train / val / test\>
    - \<distortion parameters\>_\<image ID\>_distorted.jpeg
  - point_maps/\<train / val / test\>
    - \<distortion parameters\>.gz
  - hash_to_params.json

Where \<distortion parameters\> is a hash of the tuple `(K1, K2, K3, P1, P2)` using `hash(params)` and \<image ID\> is a unique ID of the image (need to determine how to create ID).

## Train Val Test Split computation

The train/val/test split is computed using a pseudorandom split and can be obtained from querying the `hash_to_params.json` file or the `get_param_split()` function in `utils` module.
## Point maps

Mapping data for each set of distortion parameters in the following format (each variable as a float):

```text
<x_distorted> <y_distorted> <x_undistorted> <y_undistorted>
...
```

## TODOs

TODO:
- [ ] Choose image file format
- [ ] Add a script to generate the data files