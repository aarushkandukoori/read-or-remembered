"""Make the paper/README figures from the analysis outputs (no new analysis here, just plotting).

Writes PDFs to paper/figures/ and PNGs to docs/figures/. Usage: python3 scripts/make_figures.py
Colors: Okabe-Ito palette (readable for color-blind readers).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_PDF, OUT_PNG = ROOT / 'paper/figures', ROOT / 'docs/figures'
OI = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00', 'sky': '#56B4E9',
      'purple': '#CC79A7', 'yellow': '#F0E442', 'grey': '#999999', 'black': '#000000'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 8.5, 'legend.fontsize': 7, 'legend.frameon': False, 'pdf.fonttype': 42})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_PDF / f'{name}.pdf')
    fig.savefig(OUT_PNG / f'{name}.png', dpi=200)
    plt.close(fig)
    print('wrote', name)


def rows(path):
    return list(csv.DictReader(open(ROOT / path)))


# ---------- Figure: repo facts named before seen (RQ1, deterministic) ----------
def fig_facts():
    r = rows('outputs/rq1/fig_nbs_rate_accuracy.csv')
    pick = {(x['bench'], x['group_type'], x['group']): x for x in r if x['fact_type'] == 'all'}
    groups = [('DeepSeek\nV3.2', ('swebv', 'model', 'deepseek-v3.2')),
              ('Kimi\nK2.5', ('swebv', 'model', 'kimi-k2.5')),
              ('MiniMax\nM2.5', ('swebv', 'model', 'minimax-m2.5')),
              ('DeepSWE\n39 models', ('deepswe', 'all', 'all'))]
    w = rows('outputs/rq1/fig_wrong_breakdown.csv')
    wpick = {(x['bench'], x['group_type'], x['group']): x for x in w if x['fact_type'] == 'all'}
    s = rows('outputs/rq1/fig_self_correction.csv')
    spick = {(x['bench'], x['group_type'], x['group']): x for x in s if x['fact_type'] == 'all'}

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    xs = range(len(groups))
    ax[0].bar(xs, [100 * float(pick[k]['nbs_rate']) for _, k in groups], color=OI['blue'])
    ax[0].set_xticks(list(xs), [g for g, _ in groups], fontsize=5.6)
    ax[0].set_ylabel('% of first mentions')
    ax[0].set_title('(a) Repo facts stated before seen')
    ax[0].text(1, -0.30, 'SWE-bench Verified', transform=ax[0].get_xaxis_transform(), ha='center', fontsize=6, color=OI['grey'])
    for i, (_, k) in enumerate(groups):
        acc, lo, hi = (100 * float(pick[k][c]) for c in ('nbs_accuracy', 'nbs_accuracy_lo', 'nbs_accuracy_hi'))
        ax[1].errorbar(i, acc, yerr=[[acc - lo], [hi - acc]], fmt='o', color=OI['orange'], ms=4,
                       label='stated before seen' if i == 0 else None)
        ax[1].plot(i, 100 * float(pick[k]['seen_accuracy']), 's', color=OI['grey'], ms=4,
                   label='stated after seen' if i == 0 else None)
    ax[1].set_xticks(list(xs), [g for g, _ in groups], fontsize=5.6)
    ax[1].set_ylabel('% true at base commit')
    ax[1].set_ylim(60, 100)
    ax[1].set_title('(b) Accuracy')
    ax[1].text(1, -0.30, 'SWE-bench Verified', transform=ax[1].get_xaxis_transform(), ha='center', fontsize=6, color=OI['grey'])
    ax[1].legend(loc='lower left')
    benches = [('SWE-bench V.', ('swebv', 'all', 'all')), ('DeepSWE', ('deepswe', 'all', 'all'))]
    cats = [('never existed', 'never', OI['red']), ('other version', None, OI['purple']),
            ('signature', None, OI['sky'])]
    for i, (lab, k) in enumerate(benches):
        x = wpick[k]
        n = int(x['n_wrong_nbs'])
        parts = [int(x['never']), int(x['stale']) + int(x['future']),
                 int(x['names_differ']) + int(x['order_wrong']) + int(x['invalid'])]
        left = 0
        for (cl, _, col), v in zip(cats, parts):
            ax[2].barh(i, 100 * v / n, left=left, color=col, label=cl if i == 0 else None)
            left += 100 * v / n
        nc = float(spick[k]['contradicted_share'])
        ax[2].text(101, i, f'n={n}\n{100 * nc:.0f}% ever\ncontradicted', va='center', fontsize=6)
    ax[2].set_yticks([0, 1], [b for b, _ in benches])
    ax[2].set_xlim(0, 135)
    ax[2].set_xticks([0, 50, 100])
    ax[2].set_xlabel('% of wrong pre-seen facts')
    ax[2].set_title('(c) What wrong ones are')
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.45, -0.32), ncol=2, fontsize=6)
    save(fig, 'rq1_facts')


# ---------- Figure: was the code behind a faulty claim read first? (RQ1, You Wang et al. claims) ----------
def fig_claims():
    d = json.load(open(ROOT / 'outputs/rq1/claims_summary.json'))
    v = d['verdict_x_evidence_seen']
    ids = d['identifiers_verdict']
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={'width_ratios': [1.6, 1]})
    bars = [('DeepSeek refuted', v['deepseek_no_network (main)']['refuted']),
            ('DeepSeek supported', v['deepseek_no_network (main)']['supported']),
            ('Qwen refuted', v['qwen_network_on']['refuted']),
            ('Qwen supported', v['qwen_network_on']['supported'])]
    cols = {'all': OI['green'], 'some': OI['yellow'], 'none': OI['red']}
    for i, (lab, x) in enumerate(bars):
        left = 0
        for c in ('all', 'some', 'none'):
            p = 100 * x['anchored']['share_of_with_evidence'][c]['p']
            ax[0].barh(i, p, left=left, color=cols[c], label={'all': 'all cited lines shown before', 'some': 'some shown', 'none': 'none shown'}[c] if i == 0 else None)
            if p > 6:
                ax[0].text(left + p / 2, i, f'{p:.0f}', ha='center', va='center', fontsize=6)
            left += p
    ax[0].set_yticks(range(len(bars)), [b for b, _ in bars])
    ax[0].invert_yaxis()
    ax[0].set_xlabel('% of claims with repository evidence')
    ax[0].set_title('(a) Was the evidence shown before the claim?')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.45, -0.3), ncol=3, fontsize=6)
    lab = ['other-version name', 'name missing at base']
    for j, key in enumerate(['p_claim_has_other_version', 'p_claim_has_unexplained_missing']):
        for k, (verd, col) in enumerate([('refuted', OI['red']), ('supported', OI['blue'])]):
            x = ids['deepseek_no_network (main)'][verd][key]
            p, lo, hi = 100 * x['p'], 100 * x['ci95'][0], 100 * x['ci95'][1]
            ax[1].bar(j + (k - 0.5) * 0.36, p, 0.34, color=col, yerr=[[p - lo], [hi - p]], capsize=2,
                      label=verd if j == 0 else None)
    ax[1].set_xticks([0, 1], ['name only in\nother versions', 'name missing\nat base commit'], fontsize=6.5)
    ax[1].set_ylabel('% of claims')
    ax[1].set_title('(b) Names in claims (DeepSeek)')
    ax[1].legend()
    save(fig, 'rq1_claims')


# ---------- Figure: explicit memory talk, hidden vs visible (RQ3) ----------
def fig_disclosure():
    r = rows('outputs/rq3/fig_recall_rates.csv')
    pick = {(x['dataset'], x['model'], x['channel']): x for x in r if x['measure'] == 'A'}
    items = [('SWE-bench Verified (3 models)', 'swebv', 'ALL'), ('DeepSWE (33 models)', 'deepswe', 'ALL'),
             ('SWE-bench Pro (4 models)', 'faultyclaims', 'ALL'), ('   DeepSeek-V3.2', 'faultyclaims', 'DeepSeek-V3.2'),
             ('   DeepSeek-V4.1-Flash', 'faultyclaims', 'DeepSeek-V4.1-Flash'), ('   Qwen3.6-27B', 'faultyclaims', 'Qwen3.6-27B'),
             ('   Qwen3.8-27B', 'faultyclaims', 'Qwen3.8-27B')]
    floor = 0.004   # zero rates are drawn as an open marker at this x position and labeled 0
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for i, (lab, ds, m) in enumerate(items):
        for ch, col, off, mk in (('hidden reasoning', OI['red'], -0.16, 'o'), ('visible message', OI['blue'], 0.16, 's')):
            x = pick[(ds, m, ch)]
            p, lo, hi = (100 * float(x[c]) for c in ('mean_traj_rate', 'ci_lo', 'ci_hi'))
            if p <= 0:
                ax.plot(floor, i + off, mk, mfc='white', color=col, ms=3.5)
                hits = int(x['n_steps_with_hit'])
                ax.text(floor * 1.4, i + off, f'~0 ({hits} hits)' if hits else '0', va='center', fontsize=5.5, color=col)
                continue
            lo = max(lo, floor)
            ax.errorbar(p, i + off, xerr=[[p - lo], [max(hi, p) - p]], fmt=mk, color=col, ms=3.5,
                        label=ch if i == 0 else None)
    ax.set_xscale('log')
    ax.set_xlim(floor * 0.7, 80)
    ax.set_xticks([0.01, 0.1, 1, 10], ['0.01%', '0.1%', '1%', '10%'])
    ax.minorticks_off()
    ax.set_yticks(range(len(items)), [x[0] for x in items])
    ax.invert_yaxis()
    ax.set_xlabel('% of steps with explicit memory statements')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=OI['red'], marker='o', ls='', ms=3.5, label='hidden reasoning'),
          Line2D([], [], color=OI['blue'], marker='s', ls='', ms=3.5, label='visible message')]
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.4, 1.16), ncol=2)
    save(fig, 'rq3_disclosure')


# ---------- Figure: local-model memorization probes ----------
def fig_probes():
    r = rows('outputs/probes/fig_likelihood_by_model.csv')
    gen = [x for x in r if x['family'] == 'general']
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for grp, col, lab in (('agent', OI['orange'], 'facts agents relied on (n=63)'), ('control', OI['grey'], 'random controls (n=74)'),
                          ('pilot', OI['red'], 'pilot trap facts (n=2)')):
        xs = sorted((float(x['size_b']), float(x['mean_delta_trap']), float(x['ci_lo']), float(x['ci_hi'])) for x in gen if x['group'] == grp)
        ax.errorbar([a for a, *_ in xs], [b for _, b, _, _ in xs], yerr=[[b - lo for _, b, lo, _ in xs], [hi - b for _, b, _, hi in xs]],
                    fmt='-o', color=col, ms=3, capsize=2, label=lab)
    ax.axhline(0, color=OI['black'], lw=0.5)
    ax.set_xscale('log')
    ax.set_xticks([0.5, 1.5, 3, 7], ['0.5B', '1.5B', '3B', '7B'])
    ax.minorticks_off()
    ax.set_xlabel('Qwen2.5-Instruct size')
    ax.set_ylabel('log p(real) - log p(trapped), nats')
    ax.set_title('Preference for the real line (local probes)')
    ax.legend(fontsize=6)
    save(fig, 'probes')


# ---------- Figure: LLM vs human error-form labels (appendix) ----------
def fig_annotation():
    a = json.load(open(ROOT / 'outputs/annotation/agreement.json'))['calibration']
    forms = ['Misreference', 'Misstatement', 'Omission', 'Overgeneralization']
    m = [[a['confusion_rows_human_cols_llm'][h][l] for l in forms] for h in forms]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.imshow(m, cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, m[i][j], ha='center', va='center', fontsize=7, color='white' if m[i][j] > 100 else 'black')
    short = ['Misref.', 'Misstat.', 'Omiss.', 'Overgen.']
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel('LLM annotator')
    ax.set_ylabel('human (Wang et al.)')
    ax.set_title(f"n={a['n']}, agreement {100 * a['agreement']:.0f}%, kappa {a['kappa']:.2f}")
    save(fig, 'annotation_confusion')


if __name__ == '__main__':
    fig_facts()
    fig_claims()
    fig_disclosure()
    fig_probes()
    fig_annotation()


# ---------- Figure: trap and placebo runs (RQ2) ----------
def fig_traps():
    """Resolved k/n per arm for every trap experiment run so far, plus how many trap runs' final patches used the
    remembered (now wrong) fact. Sources: outputs/agent_runs, outputs/runs/sympy__sympy-13615*, outputs/targeted."""
    def kn(grades, arm):
        g = [x for x in grades if x['run'].startswith(arm)]
        return sum(1 for x in g if x['resolved']), len(g)
    tg = json.load(open(ROOT / 'outputs/targeted/targeted_summary.json'))
    exps = [
        ('django-11149  get_permission_codename\narg order, network on (pilot)', json.load(open(ROOT / 'outputs/agent_runs/grades.json')), 0),
        ('sympy-13615  sift arg order\nnetwork on (pilot)', json.load(open(ROOT / 'outputs/runs/sympy__sympy-13615/grades.json')), '0/2 (unused)'),
        ('sympy-13615  sift arg order\nnetwork off (pilot)', json.load(open(ROOT / 'outputs/runs/sympy__sympy-13615-nonet/grades.json')), '0/2 (unused)'),
        ('django-15563  add_filter arg order\nnetwork off (agent-targeted)', tg['django__django-15563'],
         sum(1 for x in tg['django__django-15563'] if x['run'].startswith('trap') and any('original' in f for f in x['final_patch']))),
        ('django-16100  atomic(using=) -> alias=\nnetwork off (agent-targeted)', tg['django__django-16100'],
         sum(1 for x in tg['django__django-16100'] if x['run'].startswith('trap') and x['final_patch'] == 'using=')),
        ('django-12741  atomic(using=) -> alias=,\nnew keyword on the edited line (targeted)',
         json.load(open(ROOT / 'outputs/targeted/runs/django__django-12741/grades.json')), 1),
    ]
    arms = ['original', 'placebo', 'trap']
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    cmap = plt.get_cmap('Greens')
    for i, (lab, grades, mem) in enumerate(exps):
        for j, arm in enumerate(arms):
            k, n = kn(grades, arm)
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=cmap(0.25 + 0.6 * k / n)))
            ax.text(j, i, f'{k}/{n}', ha='center', va='center', fontsize=7.5)
        nt = kn(grades, 'trap')[1]
        ax.text(3, i, mem if isinstance(mem, str) else f'{mem}/{nt}', ha='center', va='center', fontsize=7.5, color=OI['red'])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(len(exps) - 0.5, -0.5)
    ax.set_xticks(range(4), ['original\nresolved', 'placebo\nresolved', 'trap\nresolved', 'trap patch used\nremembered fact'], fontsize=6.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(exps)), [e[0] for e in exps], fontsize=6.3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    save(fig, 'fig_trap_effect')


if __name__ == '__main__':
    fig_traps()
