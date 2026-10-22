import os
import subprocess

RABINIZER_PATH = 'rabinizer4/bin/ltl2ldba'


def run_rabinizer(formula: str) -> str:
    """Convert an LTL formula to a LDBA in the HOA format.

    On Unix systems, prefers the provided shell script at `rabinizer4/bin/ltl2ldba`.
    On Windows, falls back to invoking Java directly using the bundled `owl.jar`.
    """
    # -p: parallel processing
    # -d: construct a non-generalised Buechi automaton
    # -e: keep generated epsilon transitions
    if os.name == 'nt':
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        jar_path = os.path.join(repo_root, 'rabinizer4', 'lib', 'owl.jar')
        command = ['java', '-classpath', jar_path, 'owl.translations.ltl2ldba.LTL2LDBACliParser',
                   '-i', formula, '-p', '-d', '-e']
    else:
        command = [RABINIZER_PATH, '-i', formula, '-p', '-d', '-e']

    run = subprocess.run(command, capture_output=True, text=True)
    # if run.stderr != '':
    #     raise RuntimeError(f'Rabinizer call `{" ".join(command)}` resulted in an error.\nError: {run.stderr}.')
    return run.stdout


if __name__ == '__main__':
    f = 'FG a'
    ldba = run_rabinizer(f)
    print(ldba)
