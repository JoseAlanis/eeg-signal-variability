"""General utility functions that are re-used in different scripts."""

import click
from mne.utils import logger

from tqdm import tqdm
from joblib import Parallel


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


@click.command()
@click.option(
    "--subject",
    type=int,
    help="Subject number.")
@click.option(
    "--session",
    type=int,
    help="Session number.")
@click.option(
    "--task",
    type=str,
    default="oddeven",
    help="Data from which task should be processed.",
)
@click.option(
    "--overwrite",
    default=False,
    type=bool,
    help="Should existing files be overwritten?",
)
@click.option(
    "--interactive",
    default=False,
    type=bool,
    help="Interactive?")
@click.option(
    "--report",
    default=False,
    type=bool,
    help="Do you want to generate HTML-reports?"
)
@click.option(
    "--jobs",
    default=1,
    type=int,
    help="The number of jobs to run in parallel."
)
def get_inputs(subject, session, task, overwrite, interactive, report, jobs):
    """Parse inputs in case script is run from command line.
    See Also
    --------
    parse_overwrite
    """
    # collect all in dict
    inputs = dict(
        subject=subject,
        session=session,
        task=task,
        overwrite=overwrite,
        interactive=interactive,
        report=report,
        jobs=jobs,
    )

    return inputs


def parse_overwrite(defaults):
    """Parse which variables to overwrite."""
    logger.info(f"\n    > Parsing command line options ...\n")

    # invoke `get_inputs()` as command line application
    inputs = get_inputs.main(standalone_mode=False, default_map=defaults)

    # check if any defaults should be overwritten
    for key, val in defaults.items():
        if val != inputs[key]:
            logger.info(
                f"    > Overwriting default '{key}': {val} -> {inputs[key]}"
            )
            defaults[key] = inputs[key]

    return defaults
