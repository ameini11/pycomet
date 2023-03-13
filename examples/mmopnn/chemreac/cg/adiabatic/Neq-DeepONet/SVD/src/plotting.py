import numpy as np
import matplotlib.pyplot as plt


def cumvar(eigenvalues, xlim, eps=0.001, figname=None):
    fig, ax = plt.subplots()
    # X axis
    ax.set_xlabel('Component Index')
    ax.set_xlim(xlim)
    # Y axis
    ax.set_ylabel('Cumulative Variance [%]')
    ax.set_ylim([50,105])
    # Plot
    cumvar = np.cumsum(eigenvalues)
    ax.plot(
        cumvar*100,
        markeredgewidth=1,
        lw=1.0,
        markersize=3.0,
        marker='o'
    )
    pc = np.where(cumvar > 1.0-eps)[0][0]+1
    ax.text(
        0.5, 0.05,
        f'Component {pc} for var > {(1.0-eps)*100} %',
        transform=ax.transAxes
    )
    # Save figure
    plt.savefig(figname)
    # plt.show()
    plt.close()

def eigen(eigenvalues, xlim, figname=None):
    fig = plt.figure()
    # X axis
    plt.xlabel('Component Index')
    plt.xlim(xlim)
    # Y axis
    plt.ylabel('Eigenvalues')
    plt.yscale('log')
    # Plot
    plt.plot(eigenvalues[:xlim[1]])
    # Save figure
    plt.savefig(figname)
    # plt.show()
    plt.close()

def pca_modes(t, y, pca, fig_path):
    # Calculate the principal components
    pc = pca.transform(y)
    # Plot principal components
    for p, phi in enumerate(pc.T):
        fig = plt.figure()
        # Y axis
        plt.ylabel(r'$\phi_{{%s}}$' % str(p+1))
        # X axis
        plt.xlabel('t [s]')
        plt.xscale('log')
        # Plot
        plt.plot(t, phi)
        # Save figure
        plt.savefig(fig_path+'/phi'+str(p+1).zfill(3)+'.pdf')
        # plt.show()
        plt.close()

def rec_error(
    err,
    nb_pc_min,
    nb_pc_max,
    err_min,
    errors=['mare', 'mse'],
    reductions=['max', 'mean'],
    figname=None
):
    # Plot
    n, m = len(errors), len(reductions)
    fig, axs = plt.subplots(n, m, sharex=True, figsize=(10,10))
    axs = axs.reshape((n,m))
    for n_ in range(n):
        for m_ in range(m):
            if n_ == n-1:
                err_, nb_pc_min_ = err['mse'], nb_pc_min['mse']
                axs[n_,0].set_ylabel('MSE')
            else:
                err_, nb_pc_min_ = err['mrae'], nb_pc_min['mrae']
                axs[n_,0].set_ylabel('MRAE')
            # Text
            axs[n_,m_].text(
                0.5, 0.05,
                f'Nb. PC > {nb_pc_min_[reductions[m_]]} for error < {err_min}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=axs[n_,m_].transAxes
            )
            # X axis
            axs[n_,m_].set_xlim([0,nb_pc_max-1])
            interval = int(np.ceil(nb_pc_max/10))
            axs[n_,m_].set_xticks(np.arange(0,nb_pc_max,interval))
            axs[n_,m_].set_xticklabels(np.arange(1,nb_pc_max+1,interval))
            # Y axis
            axs[n_,m_].set_yscale('log')
            if n_ == n-1:
                axs[n_,m_].set_xlabel('# of Preserved Components')
            # Plot
            axs[n_,m_].plot(err_[reductions[m_]])
    axs[0,0].set_title('Max')
    axs[0,1].set_title('Mean')
    plt.tight_layout()
    # Save figure
    plt.savefig(figname)
    plt.close()

def rec_solution(
    t,
    y,
    y_rec,
    figname,
    scales=['log', 'linear'],
    labels=[r'$t$ [s]', r'$Y$']
):
    fig = plt.figure()
    # X axis
    plt.xlabel(labels[0])
    plt.xscale(scales[0])
    # Y axis
    plt.ylabel(labels[1])
    plt.yscale(scales[1])
    # Plot
    plt.plot(
        t,
        y,
        label='True'
    )
    plt.plot(
        t,
        y_rec,
        marker='^',
        fillstyle='none',
        markersize=5,
        markevery=y_rec.shape[0]//20,
        ls='--',
        label='Reconstructed'
    )
    plt.legend()
    plt.tight_layout()
    # Save figure
    plt.savefig(figname)
    # plt.show()
    plt.close()