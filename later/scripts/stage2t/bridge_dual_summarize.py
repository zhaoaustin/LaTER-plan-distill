"""Collect runA/runB evals into summary_ab.json + markdown table (BRIDGE_AB_COMPARISON.md section 2)."""
import json, random, statistics as st, sys
R = sys.argv[1]
LEAK = {'s1val-00309', 's1val-00731', 's1val-00645', 's1val-00938'}
def boot(v, n=10000):
    rng = random.Random(0); m = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(n)); return [st.mean(v), m[int(.025*n)], m[int(.975*n)]]
out, md = {}, []
for g in 'AB':
    tr = {json.loads(l)['step']: json.loads(l) for l in open(f'{R}/run{g}/train.jsonl')}
    for s in [0, 25, 50, 100, 150]:
        d = json.load(open(f'{R}/run{g}/evaluations/eval_step{s}.json')); r = json.load(open(f'{R}/run{g}/evaluations/rows_step{s}.json'))
        h, gh, gp, nm = d['scores_heldout'], d['gen_heldout'], d['gen_para'], d['normal_mode']
        dm = boot([x['full']['delta_match'] for x in r['scores']['heldout']])
        G = [x for x in r['gens'] if x['latent'] in ('Z_A', 'Z_B')]
        nl = [x for x in G if x['id'] not in LEAK]
        row = dict(group=g, step=s, train_ce=tr[s]['train_ce'] if s else d['scores_train']['matched_ce'], heldout_ce=h['matched_ce'],
                   delta_match=dm, both_pos=h['full']['both_pos'], pair_follow=gh['pair_follow_auto'], A_adh=gh['A_adherence_auto'], B_adh=gh['B_adherence_auto'],
                   acc=gh['correct_A'] + gh['correct_B'], acc_noleak=sum(x['correct'] for x in nl), n_noleak=len(nl), fmt=gh['format_A'] + gh['format_B'],
                   para_pair=gp['pair_follow_auto'], para_adh=gp['A_adherence_auto'] + gp['B_adherence_auto'], para_dm=[d['scores_para']['full']['mean_delta_A'], d['scores_para']['full']['mean_delta_B']],
                   train_pair=d['gen_train']['pair_follow_auto'], normal_acc=nm['correct'], normal_fmt=nm['format_ok'],
                   train_s=tr[s]['elapsed_s'] if s else 0.0, tokens=tr[s]['target_tokens_seen'] if s else 0, eval_s=d['eval_seconds'])
        out[f'{g}{s}'] = row
        md.append(f"| {g} | {s} | {row['train_ce']:.2e} | {row['heldout_ce']:.3f} | {dm[0]:+.4f} [{dm[1]:+.4f},{dm[2]:+.4f}] | {row['pair_follow']}/16 (A {row['A_adh']}, B {row['B_adh']}) | "
                  f"{row['acc']}/32 (去泄漏 {row['acc_noleak']}/{row['n_noleak']}) | {row['fmt']}/32 | {row['para_pair']}/12 (adh {row['para_adh']}/24) | {row['train_s']/60:.1f} 分 + 评估 {row['eval_s']/60:.1f} 分 |")
json.dump(out, open(f'{R}/summary_ab.json', 'w'), indent=1)
print('\n'.join(md))
