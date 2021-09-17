#!/usr/bin/env python3

import os
import prode

bashrc = os.environ['HOME']+"/.bashrc"

with open(bashrc,'r') as f:
    lines = f.readlines()

remove = False
with open(bashrc,'w') as f:
    for l in lines:
        if l.startswith('# >>> prode'):
            remove = True
        if not remove:
            f.write(l)
        if l.startswith('# <<< prode'):
            remove = False

    f.write(
        "\n# >>> prode >>>\n"
        "export PATH={}/algos/:$PATH\n"
        "# <<< prode <<<".format(os.path.dirname(prode.__file__))
    )
    f.close()
