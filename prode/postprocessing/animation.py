import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# Animation function.  This is called sequentially
def animate(i, t, lines, path, delta_epoch):

    # Get predicted data
    file = path + str(i*delta_epoch)
    y_pred = np.loadtxt(file)

    # Plot predicted data
    for d in range(y_pred.shape[1]):
        lines[d].set_data(t, y_pred[:,d])

    return lines


def animation(InputData, iTestCase, tData, xData, PredFile, current_epoch=float('inf'), y_scale='linear'):
    ### Requires: sudo apt-get install ffmpeg (Ubuntu) / brew install ffmpeg (MacOS)

    fig = plt.figure()
    plt.grid()

    # Preprocess figure -------------------------------------------------------
    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
    if isinstance(x, (list,tuple)):
        x = x[time_idx] # Time input
    else:
        x = x[:,time_idx]
    if x_label is not None:
        plt.xlabel(x_label)
    if x_scale is not None:
        plt.xscale(x_scale)
    if x_scale != 'log':
        plt.xlim([min(x), max(x)])

    # Y axis
    if y_label is not None:
        plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y_true)*5.e-1, np.amax(y_true)*5.e+0])
    else:
        delta = np.amax(y_true)*0.1
        plt.ylim([np.amin(y_true)-delta, np.amax(y_true)+delta])

    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, 1)

    # Plot figure -------------------------------------------------------------
    # Define styles
    true_style = dict(
        lw=1.
    )
    pred_style = dict(
        marker='^',
        fillstyle='none',
        markersize=5,
        markevery=y_true.shape[0]//20,
        ls='--',
        lw=1.
    )

    # Define parameters
    pred_lines = []
    colors = []
    n = y_true.shape[1]

    # Loop over features
    for d in range(n):

        # >> Plot true values
        plt.plot(
            x,
            y_true[:,d],
            **true_style
        )
        colors.append(plt.gca().lines[-1].get_color())

        # >> Plot predicted values
        pred_lines.append(
            plt.plot(
                [],
                [],
                **pred_style,
                label=var_names[d],
                c=colors[-1]
            )
        )

    # Plot legends
    fontsize = 'x-small'
    if n > 6:
        fontsize = 'xx-small'
    legend_col = plt.legend(
        [plt.plot([], [], c=colors[i])[0] for i in range(len(var_names))],
        var_names,
        fontsize=fontsize,
        loc=locs['col'],
        ncol=int(np.ceil(n/6))
    )
    legend_type = plt.legend(
        [
            plt.plot([], [], c="k", **true_style)[0],
            plt.plot([], [], c="k", **pred_style)[0]
        ],
        ["True", "Pred"],
        fontsize=fontsize,
        loc=locs['type']
    )
    plt.gca().add_artist(legend_col)





    # Animation function.  This is called sequentially
    def animate(i, path, delta_epoch):

        # Get predicted data
        file = path + str(i*delta_epoch)
        y_pred = np.loadtxt(path + str(i*delta_epoch))

        # Plot predicted data
        for d in range(n):
            lines[d].set_data(t, y_pred[:,d])

        return lines


    # Animation function.  This is called sequentially
    def animate(i, xDim, tData, xData, PredFile, read_data, NEpochTest):

        for d in range(n):

            # >> Plot true values
            plt.plot(
                x,
                y_true[:,d],
                **true_style
            )
            colors.append(plt.gca().lines[-1].get_color())

            # >> Plot predicted values
            pred_lines.append(
                plt.plot(
                    x,
                    y_pred[:,d],
                    **pred_style,
                    label=var_names[d],
                    c=colors[-1]
                )
            )


        for ixDim in range(xDim):
            lines[(ixDim)*2].set_data(tData, xData[:,ixDim])

        iRun         = i*NEpochTest
        PredFile     = PredFile + str(iRun)
        tPred, xPred = read_data(PredFile, xDim)

        for ixDim in range(xDim):
            lines[(ixDim)*2+1].set_data(tPred, xPred[:,ixDim])

        return lines









    # Plotting
    lin   = ['-', '--']
    col   = ['k', 'r', 'g', 'b', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    lines = []
    for ixDim in range(InputData.xDim):
        DataName = 'ODE, '                      + InputData.xNames[ixDim]
        PredName = InputData.ApproxModel + ', ' + InputData.xNames[ixDim]
        lines    = lines + list(map(lambda x,y: plt.plot([], [], c=col[ixDim], ls=lin[x], lw=1, label=y)[0], [0,1], [DataName, PredName]))
    plt.legend(fontsize='x-small')


    # Animation function.  This is called sequentially
    def animate(i, xDim, tData, xData, PredFile, read_data, NEpochTest):

        for ixDim in range(xDim):
            lines[(ixDim)*2].set_data(tData, xData[:,ixDim])

        iRun         = i*NEpochTest
        PredFile     = PredFile + str(iRun)
        tPred, xPred = read_data(PredFile, xDim)

        for ixDim in range(xDim):
            lines[(ixDim)*2+1].set_data(tPred, xPred[:,ixDim])

        return lines


    # call the animator.  blit=True means only re-draw the parts that have changed.
    #anim = animation.FuncAnimation(fig, animate, init_func=init, frames=40, interval=1, blit=True)
    frames = 

    ani = animation.FuncAnimation(fig, animate, fargs=(loss, val_loss, title), repeat=True, interval=1000, repeat_delay=1000)

    anim = animation.FuncAnimation(fig, animate, frames=NFrames, fargs=(i, t, lines, path, delta_epoch), interval=1, blit=True), repeat=True, interval=1000, repeat_delay=1000)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    VideoFileName = InputData.PathToFigFld + '/TestCase' + str(iTestCase+1) + '_' + y_scale + '.mp4'
    #anim.save(VideoFileName, progress_callback=lambda i, n: print(f'[ProPDE]:   Saving frame {i+1} of {n}'), dpi=600, fps=10, extra_args=['-vcodec', 'libx264'])
    anim.save(VideoFileName, dpi=600, fps=10, extra_args=['-vcodec', 'libx264'])

    plt.close()





import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,





# Animation function.  This is called sequentially
def animate(i, path, delta_epoch):

    # Get predicted data
    file = path + str(i*delta_epoch)
    y_pred = np.loadtxt(path + str(i*delta_epoch))

    # Plot predicted data
    for d in range(n):
        lines[d].set_data(t, y_pred[:,d])

    return lines












# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()




def animate(i, loss, val_loss, title):
    data = loss.iloc[:int(i+1)]
    data2 = val_loss.iloc[:int(i+1)]

    plt.plot(np.array(data.index), data[title], color='b', label='Train Loss')    
    plt.plot(np.array(data2.index), data2[title], color='r', label='Test Loss')

    global flag
    if flag:
        plt.legend(loc='center right', fontsize='xx-large')
        flag = False








def plot_test(
    path,
    x,
    y_true,
    y_pred,
    var_names,
    time_idx=-1,
    fig_name=None,
    labels=[None, None],
    scales=['log', 'linear'],
    locs={'col': 3, 'type': 1}
):
    """Variable plotting."""

    fig = plt.figure()
    plt.grid()

    # Preprocess figure -------------------------------------------------------
    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
    if isinstance(x, (list,tuple)):
        x = x[time_idx] # Time input
    else:
        x = x[:,time_idx]
    if x_label is not None:
        plt.xlabel(x_label)
    if x_scale is not None:
        plt.xscale(x_scale)
    if x_scale != 'log':
        plt.xlim([min(x), max(x)])

    # Y axis
    if y_label is not None:
        plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y_true)*5.e-1, np.amax(y_true)*5.e+0])
    else:
        delta = np.amax(y_true)*0.1
        plt.ylim([np.amin(y_true)-delta, np.amax(y_true)+delta])

    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, 1)

    # Plot figure -------------------------------------------------------------
    # Define styles
    true_style = dict(
        lw=1.
    )
    pred_style = dict(
        marker='^',
        fillstyle='none',
        markersize=5,
        markevery=y_true.shape[0]//20,
        ls='--',
        lw=1.
    )

    # Define parameters
    colors = []
    n = y_true.shape[1]

    # Loop over features
    for d in range(n):

        # >> Plot true values
        plt.plot(
            x,
            y_true[:,d],
            **true_style
        )
        colors.append(plt.gca().lines[-1].get_color())

        # >> Plot predicted values
        plt.plot(
            x,
            y_pred[:,d],
            **pred_style,
            label=var_names[d],
            c=colors[-1]
        )

    # Plot legends
    fontsize = 'x-small'
    if n > 6:
        fontsize = 'xx-small'
    legend_col = plt.legend(
        [plt.plot([], [], c=colors[i])[0] for i in range(len(var_names))],
        var_names,
        fontsize=fontsize,
        loc=locs['col'],
        ncol=int(np.ceil(n/6))
    )
    legend_type = plt.legend(
        [
            plt.plot([], [], c="k", **true_style)[0],
            plt.plot([], [], c="k", **pred_style)[0]
        ],
        ["True", "Pred"],
        fontsize=fontsize,
        loc=locs['type']
    )
    plt.gca().add_artist(legend_col)

    # Save figure
    name = y_scale if fig_name == None else fig_name + '_' + y_scale
    fig_path = path + '/' + name + '.pdf'
    fig.savefig(fig_path)
    plt.close()
