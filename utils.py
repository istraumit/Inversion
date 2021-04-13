import subprocess as sp

def run(cmd):
    try:
        o = sp.check_output(cmd, shell=True)
    except sp.CalledProcessError as err:
        print(err.output)

