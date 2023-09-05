"""
Utility Functions for Data Processing and Command-Line Parsing

This script contains utility functions that are reused in different
data processing scripts. Specifically, it provides methods to parse
command-line inputs and handle the overwriting of default values.

Functions:
-----------
- `get_inputs`: Parses command-line options related to subject
  identifiers, data tasks, stimuli, and various processing options.
  Returns them as a dictionary.

- `parse_overwrite`: Compares user-provided command-line inputs
  with default values to determine which defaults should be
  overwritten. Updates the defaults as needed.

Note:
-----
The script relies on the 'click' library to handle command-line
arguments and options.

Examples:
---------
To see examples of how these functions can be used, please refer
to the individual function docstrings.
"""

import click
from mne.utils import logger

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
    "--stimulus",
    type=str,
    default="cue",
    help="The stimulus should be processed (e.g., the 'cue').",
)
@click.option(
    "--window",
    type=str,
    default="pre",
    help="The time window that should be processed "
         "(e.g., 'pre' for the 'pre cue timewindow').",
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
def get_inputs(subject, session, task, stimulus, window,
               overwrite, interactive, report, jobs):
    """
    Parse command-line inputs for data processing options.

    Parameters
    ----------
    subject : int
        Subject number to be processed.

    session : int
        Session number to be processed.

    task : str, optional
        Specifies the task data to be processed. Defaults to "oddeven".

    stimulus : str, optional
        Specifies the stimulus to be processed, e.g., "cue". Defaults to "cue".

    window : str, optional
        Specifies the time window to be processed. Defaults to "pre".

    overwrite : bool, optional
        Indicates if existing files should be overwritten. Defaults to False.

    interactive : bool, optional
        Indicates if the processing should be interactive. Defaults to False.

    report : bool, optional
        Indicates if HTML-reports should be generated. Defaults to False.

    jobs : int, optional
        The number of jobs to run in parallel. Defaults to 1.

    Returns
    -------
    dict
        A dictionary containing all the parsed input parameters.

    See Also
    --------
    parse_overwrite : Further function to parse the overwrite option.
    """
    inputs = dict(
        subject=subject,
        session=session,
        task=task,
        stimulus=stimulus,
        window=window,
        overwrite=overwrite,
        interactive=interactive,
        report=report,
        jobs=jobs,
    )

    return inputs


def parse_overwrite(defaults):
    """
   Parse command-line inputs to update default variables, if needed.

   Invokes `get_inputs` to fetch user-defined inputs from the command line.
   Compares these against the provided defaults and updates them if needed.

   Parameters
   ----------
   defaults : dict
       Dictionary of default parameters for data processing.

   Returns
   -------
   dict
       Updated dictionary with either default or overwritten parameters.

   See Also
   --------
   get_inputs : Function used to fetch command-line inputs.

   Examples
   --------
   >>> default_params = {'subject': 1, 'session': 1, 'task': 'oddeven'}
   >>> parse_overwrite(default_params)
   {
       'subject': 1,
       'session': 1,
       'task': 'new_task'  # if 'new_task' was input at command line
   }

   Notes
   -----
   Logs overwritten variables and their new values using a logger.
   """
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
