"""Figures from real full-history arrays, with all seeds retained in tables."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .material import COMPONENTS

COLORS = {'reference':'#222222', 'ideal':'#0072B2', 'noisy':'#D55E00'}
STYLES = {'reference':'-', 'ideal':'--', 'noisy':':'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.labelsize':9, 'axes.titlesize':10, 'legend.fontsize':8,
        'svg.fonttype':'none', 'savefig.dpi':180, 'lines.linewidth':1.3})


def save(fig, directory, name):
    fig.savefig(directory/f'{name}.png', dpi=180)
    fig.savefig(directory/f'{name}.svg')
    plt.close(fig)


def reproduction_figures(out, compact, records, selected, predictions):
    style()
    truth = np.load(out/'reference.npz')
    obs = np.load(out/'observations.npz')
    target, time = truth['evaluation_stress'], obs['evaluation_time']
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.7), layout='constrained')
    for row, condition in enumerate(('ideal','noisy')):
        seed = selected[condition]['seed']
        pred = predictions[f'{condition}-seed-{seed:02d}']['stress']
        for k, component in enumerate(COMPONENTS):
            ax = axes[row,k]
            lo = min(np.nanmin(target[:,k]), np.nanmin(pred[:,k]))
            hi = max(np.nanmax(target[:,k]), np.nanmax(pred[:,k]))
            ax.plot([lo,hi],[lo,hi], color='#999999', lw=1)
            ax.scatter(target[1:,k], pred[1:,k], s=6, color=COLORS[condition], alpha=.65,
                       marker='o' if condition=='ideal' else '^')
            ax.set_title(f'{condition}, $\\sigma_{{{component}}}$')
            ax.set_xlabel('Reference (MPa)')
            if k == 0:
                ax.set_ylabel('Prediction (MPa)')
            ax.set_aspect('equal', adjustable='box')
    fig.suptitle('TM02 Figure 8: selected RNN method, independent 250-step free prediction')
    save(fig, compact, 'fig08-rnn-correlation')
    fig, axes = plt.subplots(3, 2, figsize=(9, 7.5), sharex=True, layout='constrained')
    for k, ax in enumerate(axes.flat):
        ax.plot(time, target[:,k], color=COLORS['reference'], label='Reference')
        for condition in ('ideal','noisy'):
            seed = selected[condition]['seed']
            pred = predictions[f'{condition}-seed-{seed:02d}']['stress']
            ax.plot(time, pred[:,k], color=COLORS[condition], ls=STYLES[condition],
                    label=f'{condition}, seed {seed}')
        ax.set_ylabel(f'$\\sigma_{{{COMPONENTS[k]}}}$ (MPa)')
        ax.grid(alpha=.2)
    axes[0,0].legend(loc='upper left', bbox_to_anchor=(0,1.30), ncol=3)
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Time (s)')
    fig.suptitle('TM02 Figure 9: complete stress histories (RNN discarded)')
    save(fig, compact, 'fig09-full-stress-path')
    q = np.load(out/'auxiliary-diagnostic.npz')
    fig, axes = plt.subplots(2, 1, figsize=(8,4.8), sharex=True, layout='constrained')
    for ax,k in zip(axes,(0,3)):
        ax.plot(q['time'], q['reference_q'][:,k], color=COLORS['reference'], label='Reference internal strain')
        ax.plot(q['time'], q['auxiliary_q'][:,k], '--', color=COLORS['ideal'], label='Training auxiliary RNN')
        ax.set_ylabel(f'$q_{{{COMPONENTS[k]}}}$')
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0,1.30), ncol=2)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('TM02 Figure 10: ideal-training auxiliary-state diagnostic')
    save(fig, compact, 'fig10-auxiliary-state')
    fig, axes = plt.subplots(2, 2, figsize=(9,6), sharex=True, layout='constrained')
    for col,condition in enumerate(('ideal','noisy')):
        for row,term in enumerate(('stress_MAE','biot_MAE')):
            ax = axes[row,col]
            for record in (r for r in records if r['condition']==condition):
                z = np.load(out/'final'/f"{condition}-seed-{record['seed']:02d}-loss.npz")
                chosen = record['seed']==selected[condition]['seed']
                ax.semilogy(z['iteration'][::10], z[term][::10],
                            color=COLORS[condition] if chosen else '#b6b6b6',
                            alpha=1 if chosen else .45, lw=1.5 if chosen else .5,
                            zorder=3 if chosen else 1)
            ax.set_title(f'{condition}: {term.replace("_", " ")}')
            ax.set_ylabel('MAE (MPa)')
            ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('Optimizer updates')
    fig.suptitle('All 25 initializations per condition; selected final-loss model highlighted')
    save(fig, compact, 'training-curves')
    fig, axes = plt.subplots(1, 2, figsize=(8,3.8), layout='constrained')
    for ax, metric in zip(axes,('MAE_MPa','RMSE_MPa')):
        for i,condition in enumerate(('ideal','noisy')):
            rows = [r for r in records if r['condition']==condition and r['integration_complete']]
            values = [r[metric] for r in rows]
            ax.boxplot([values], positions=[i], widths=.32, showfliers=False,
                       medianprops={'color':'black'}, whiskerprops={'color':'#777777'})
            ax.scatter(i+np.linspace(-.13,.13,len(rows)), values, s=16,
                       color=COLORS[condition], marker='o' if i==0 else '^', alpha=.8)
        ax.set_xticks([0,1],['ideal','noisy'])
        ax.set_ylabel(metric.replace('_',' '))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Independent full-path errors; 25 final models per condition\nDots: individual models; box: median/IQR; whiskers: 1.5 IQR')
    save(fig, compact, 'all-initialization-errors')


def revisit_figures(config, compact, rollouts, pairs):
    style()
    for preload in config['preloading_histories']:
        for delay in config['recovery_delays_s']:
            fig, axes = plt.subplots(2, 2, figsize=(10,5.8), sharex='col', layout='constrained')
            for col,candidate in enumerate(config['current_action_candidates']):
                hid = f"{preload['id']}-delay-{delay:g}-{candidate['id']}"
                result = rollouts[hid,'reference']
                t = result['time']
                axes[0,col].plot(t, result['stress'][:,0], color='#222222')
                axes[0,col].set_title(f"{candidate['id']} candidate")
                axes[0,col].set_ylabel('Prescribed $\\sigma_{11}$ (MPa)')
                for model in ('reference','ideal','noisy'):
                    path = rollouts[hid,model]
                    axes[1,col].plot(t, -100*path['strain'][:,0], color=COLORS[model],
                                     ls=STYLES[model], label=model)
                axes[1,col].set_ylabel('Compression $-\\epsilon_{11}$ (%)')
                axes[1,col].set_xlabel('Time (s)')
                for row in range(2):
                    axes[row,col].axvspan(2+delay,3+delay, color='#999999',alpha=.10)
                    axes[row,col].grid(alpha=.15)
            axes[1,0].legend(ncol=3, loc='upper left', bbox_to_anchor=(0,1.22))
            fig.suptitle(f"{preload['id']}, recovery delay {delay:g} s; shaded: common reload probe")
            save(fig, compact, f"history-{preload['id']}-delay-{delay:g}")
    fig, axes = plt.subplots(1,2, figsize=(9,3.8), sharey=True, layout='constrained')
    for ax,preload in zip(axes,config['preloading_histories']):
        rows = [p for p in pairs if p['preload']==preload['id']]
        for model in ('reference','ideal','noisy'):
            x = [p['delay_s'] for p in rows]
            y = [p[f'{model}_difference'] for p in rows]
            ax.plot(x,y, STYLES[model], color=COLORS[model], marker='o', label=model)
        ax.set_xscale('log')
        ax.axhspan(-config['metrics']['ranking_tie_tolerance_absolute_strain'],
                    config['metrics']['ranking_tie_tolerance_absolute_strain'], color='#999999',alpha=.2)
        ax.axhline(0,color='#555555',lw=.7)
        ax.set_title(preload['id'])
        ax.set_xlabel('Recovery interval (s)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Peak compression: early minus late')
    axes[0].legend()
    fig.suptitle('Six candidate pairs: negative favors early; gray band denotes ties')
    save(fig, compact, 'candidate-consequences')
    fig, axes = plt.subplots(1,2,figsize=(7.2,3.5),sharey=True,layout='constrained')
    for ax,preload in zip(axes,config['preloading_histories']):
        pair = next(p for p in pairs if p['preload']==preload['id'] and p['delay_s']==5)
        for x,model in enumerate(('reference','ideal','noisy')):
            value = pair[f'{model}_difference']*1e6
            ax.scatter(x,value,color=COLORS[model],marker=('o','s','^')[x],s=40,zorder=3)
            ax.annotate(f'{value:.3f}',(x,value),xytext=(0,-17),textcoords='offset points',ha='center')
        ax.axhspan(-1,1,color='#999999',alpha=.18)
        ax.axhline(0,color='#777777',lw=.7)
        ax.set_xticks([0,1,2],['reference','ideal','noisy'])
        ax.set_xlim(-.5,2.5)
        ax.set_ylim(-3.9,1.3)
        ax.set_title(preload['id'])
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Early minus late peak compression ($10^{-6}$)')
    fig.suptitle('5 s recovery: noisy model predicts a tie (gray band)')
    save(fig,compact,'long-recovery-ranking')
