# Troubleshooting

Python Store alias fails:

- Install Python 3.10+ from python.org or disable the Microsoft Store alias and expose a real interpreter.

Git unavailable:

- Install Git for Windows or add it to PATH before running project tracking commands.

YAML config fails:

- Install `requirements-base.txt` so PyYAML is available.

Profile validation fails:

- Check that probability is between 0 and 1, severity is `none`, `light`, `medium`, or `heavy`, and crop/readability thresholds are within allowed ranges.

