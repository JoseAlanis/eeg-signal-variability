"""General utility functions that are re-used in different scripts."""
from os import path
from pathlib import Path

import json
import click
from mne.utils import logger
from mne import sys_info

# get path to current file
parent = Path(__file__).parent.resolve()


# -----------------------------------------------------------------------------
@click.command()
@click.option(
    "--rawdata",
    help="Path to `sourcedata` directory",
    default=None,
)
@click.option(
    "--sourcedata",
    help="Path to `sourcedata` directory",
    default=None,
)
@click.option(
    "--bids",
    help="Path to directory contain data in BIDS format",
    default=None,
)
@click.option(
    "--mkdirs",
    help="Should missing directories be created?",
    default=False,
    type=bool,
)
@click.option(
    "--overwrite",
    help="Should existing files be overwritten?",
    default=False,
    type=bool,
)
def set_paths(
    rawdata,
    sourcedata,
    bids,
    mkdirs,
    overwrite,
):
    """Parse inputs in case script is run from command line."""
    if rawdata is None:
        rawdata = path.join(parent, "data", "rawdata")
    if sourcedata is None:
        sourcedata = path.join(parent, "data", "sourcedata")
    if bids is None:
        bids = path.join(parent, "data", "bids")

    # collect all in dict
    path_vars = dict(
        rawdata=rawdata,
        sourcedata=sourcedata,
        bids=bids,
        overwrite=overwrite,
        mkdirs=mkdirs,
    )

    return path_vars


# -----------------------------------------------------------------------------
print("\nSystem information useful when triaging bugs:\n")
print(sys_info())

# -----------------------------------------------------------------------------
# write .json file containing basic set of paths needed for the study
paths = set_paths.main(standalone_mode=False)
fname = path.join(parent, "paths.json")
print("\n")
for key, val in paths.items():
    if not path.exists(val) and not paths["mkdirs"]:
        raise RuntimeError(
            f"\n    > Could not find '{key}' under {val}.\n"
            f"    > Be sure to check if the provided path is "
            f"correct or consider setting 'mkdirs to True'."
        )
    elif not path.exists(val) and paths["mkdirs"]:
        Path(val).mkdir(parents=True, exist_ok=True)

    logger.info(f"    > Setting the path for '{key}': to -> {val}")
if path.exists(fname):
    if paths["overwrite"]:
        with open(fname, "w") as file:
            json.dump(paths, file, indent=2)
    else:
        raise RuntimeError("\n%s already exists.\n" % fname)
else:
    with open(fname, "w") as file:
        json.dump(paths, file, indent=2)
