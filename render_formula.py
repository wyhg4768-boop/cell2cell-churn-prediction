"""Render the report’s scoring rule from LaTeX-style math notation."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc_context
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image

FORMULA = r'$x_i = (x_{i1},\ldots,x_{ip})\qquad s_i = f(x_i)\in[0,1]\qquad \hat{y}_i = 1[s_i\geq t]$'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/report/scoring-equation.png'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({'mathtext.fontset': 'stix'}):
        math_to_image(FORMULA, args.output, prop=FontProperties(size=12), dpi=600)
