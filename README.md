# MaskRcnnTreeHealth
Repository for the paper about tree health classification in Mongolia.

## Configuration setup

This repository includes a public-safe sample configuration at [config_sample.ini](config_sample.ini). To use the project locally:

1. Copy the sample file to a local file named `config.ini`.
2. Replace the placeholder values with your local dataset paths and output directories.
3. Keep `config.ini` local and do not upload it to GitHub.

Example:

```bash
cp config_sample.ini config.ini
```

Then edit `config.ini` and set your actual paths, such as:

- dataset directories
- annotation shapefiles
- output folders
- checkpoint locations

The sample file is designed so you can safely publish the repository without exposing your personal machine paths.
