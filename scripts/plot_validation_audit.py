"""Render reproducible validation figures from the saved audit, without test data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save_formats(fig, output, stem):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'{stem}.{ext}', dpi=180)
    svg = output / f'{stem}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'reports/ufm_validation_audit_v2/report.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/figures/validation_20261003')
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    assert report['protocol']['split'] == 'validation' and report['test_evaluated'] is False
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    names = ['zero_shot', 'extreme_cold', 'cold', 'warm']
    colors = ['#156b59', '#dc9350']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(names))
    for offset, model, label, color in [(-.19,'ufm','UFM (seed 42)',colors[0]),
                                        (.19,'tfidf','TF-IDF',colors[1])]:
        values = [report['by_regime_and_history'][n]['all'][model+'_ndcg10'] for n in names]
        bars = ax.bar(positions+offset, values, .36, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set(xticks=positions, xticklabels=['Zero-shot', 'Extreme cold\n1–5 interactions',
           'Cold\n6–20 interactions', 'Warm\n>20 interactions'], ylabel='NDCG@10', ylim=(0,.75),
           title='Toys & Games: validation ranking by item frequency')
    ax.legend(loc='upper left', frameon=False)
    ax.set_axisbelow(True); ax.grid(axis='y', alpha=.18)
    fig.text(.08,.025,'40,000 cases · 100 sampled candidates/case · One UFM seed · Test not evaluated', fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1))
    save_formats(fig, args.output, 'ranking_by_regime')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10,4.6))
    for ax, phase, label, color in zip(axes, ['before','after'], ['Before calibration','After calibration'],colors):
        metric=report['calibration'][phase]['overall']
        bins=[b for b in metric['bins'] if b['n']]
        ax.plot([0,1],[0,1],linestyle='--',color='#77818c',linewidth=1)
        ax.scatter([b['confidence'] for b in bins],[b['accuracy'] for b in bins],
                   s=[20+180*b['n']/metric['n'] for b in bins],color=color)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence in bin',ylabel='Top-1 accuracy',
               title=f'{label} · ECE {metric["ece"]:.4f}')
        ax.grid(alpha=.18)
    fig.suptitle('Temperature calibration: exploratory late-validation audit', fontsize=12)
    fig.text(.08,.02,'Fit 20,000 early cases; audit 20,000 late cases. Checkpoint selection used all validation.\n'
             'Confidence is conditional on sampled candidates; bubble area indicates bin support.', fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.96))
    save_formats(fig, args.output, 'calibration_reliability')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
