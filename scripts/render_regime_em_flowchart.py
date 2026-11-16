"""Render the Gaussian regime-switching EM initialization chart for the report."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(8.5, 6.4))
ax.set(xlim=(0, 10), ylim=(0, 10))
ax.axis('off')
boxes = [
    (9.15, '1. E-step: filter and smooth',
     'Fix all parameters; compute smoothed state and transition probabilities.'),
    (7.30, '2. Update transition probabilities',
     'Use expected transition counts; keep the E-step probabilities fixed.'),
    (5.45, '3. Constrained weighted regression + residual variances',
     'Fit switching and shared coefficients jointly; then update variances.'),
    (3.60, '4. Assemble the new parameters and repeat',
     'New transitions, mean coefficients, and variances.'),
    (1.30, 'Final joint MLE optimization',
     'Optimize transitions, mean coefficients, and variances together.'),
]
for y, title, detail in boxes:
    patch = FancyBboxPatch((1.18, y-.49), 8.08, .98,
                           boxstyle='round,pad=0.12', linewidth=1.1,
                           edgecolor='#334155', facecolor='#f1f5f9')
    ax.add_patch(patch)
    ax.text(5.22, y+.15, title, ha='center', va='center', fontsize=11.5, weight='bold')
    ax.text(5.22, y-.18, detail, ha='center', va='center', fontsize=9.3)
for (ya,_,_), (yb,_,_) in zip(boxes, boxes[1:]):
    ax.add_patch(FancyArrowPatch((5.22,ya-.64),(5.22,yb+.64),
                                arrowstyle='-|>',mutation_scale=14,color='#334155',lw=1.2))
ax.text(5.38,2.45,'After initialization stops',va='center',fontsize=9.5)
# Repeat using all newly updated parameters; no E-step within the M-step blocks.
ax.plot([1.05,.43,.43],[3.60,3.60,9.15],color='#334155',lw=1.2)
ax.add_patch(FancyArrowPatch((.43,9.15),(1.05,9.15),arrowstyle='-|>',
                            mutation_scale=14,color='#334155',lw=1.2))
ax.text(.16,6.05,'Repeat with new parameters',rotation=90,ha='center',va='center',fontsize=9.5)
fig.tight_layout(pad=.35)
for ext in ('svg','png'):
    fig.savefig(ROOT / 'RegimeSwitching' / f'em_iteration_chart.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
