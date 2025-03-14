# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import sys
import argparse
import numpy as np

from oututils import selections, reporting, os_utils, print_outs


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, default=r'./outputs/sweep_outs')
    parser.add_argument("--result_dir", type=str, default=r'./outputs/results')
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--selections", nargs='+', type=str, default=['IID'], help='IID, OneOut, Oracle')
    args = parser.parse_args()

    SELECTION_METHODS = selections.get_methods(args.selections)

    # random grid
    records = reporting.load_records(args.input_dir)

    results_file = "0_sweep_results.tex" if args.latex else "0_sweep_results.txt"
    os.makedirs(args.result_dir, exist_ok=True)
    sys.stdout = os_utils.Tee(os.path.join(args.result_dir, results_file), "w")

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_outs.print_final_search_results(records, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")
