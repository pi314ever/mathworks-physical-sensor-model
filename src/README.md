# Source code

This directory contains the source code.

## Format

Proposed format:

- src/
  - \<model name\>/
    - \<model implementation files\>
    - Model specific
  - data/
    - \<data collection files\>
    - Files that operate on the project-specific data
  - util/
    - \<common utility files\>
    - Generalized utility modules


## Scripts

- `data/generate_point_maps.py`: Generates point map data for a range of distortion parameters (specified in the script file)

## Models

### Point Map Neural Network

Found in `point_map_nn/`.