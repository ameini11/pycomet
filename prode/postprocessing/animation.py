import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def animation(InputData, iTestCase, tData, xData, PredFile, current_epoch=float('inf'), y_scale='linear'):
    ### Requires: sudo apt-get install ffmpeg (Ubuntu) / brew install ffmpeg (MacOS)

    NEpochTemp = int(np.minimum(current_epoch, int(InputData.NEpoch)))

    fig = plt.figure()

    plt.grid()
    plt.title(InputData.PDEName)

    # X axis
    plt.xlabel(r'$t\quad[s]$')
    if (InputData.xLogPlotFlg):
        plt.xscale('log')
        plt.xlim([InputData.xMinValue, max(tData)])
    else:
        plt.xlim([min(tData), max(tData)])

    # Y axis
    plt.ylabel(r'$\rho\quad[kg/m^3]$')
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(xData[-1,:])*5.e-1, np.amax(xData[-1,:])*5.e+0])
    else:
        delta = np.amax(xData)*0.1
        plt.ylim([np.amin(xData)-delta, np.amax(xData)+delta])

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
    NFrames = 0
    for iEpoch in range(NEpochTemp):
        if (iEpoch % InputData.NEpochTest == 0):
            NFrames+=1
    anim = animation.FuncAnimation(fig, animate, frames=NFrames, fargs=(InputData.xDim, tData, xData, PredFile, read_data, InputData.NEpochTest, ), interval=1, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    VideoFileName = InputData.PathToFigFld + '/TestCase' + str(iTestCase+1) + '_' + y_scale + '.mp4'
    #anim.save(VideoFileName, progress_callback=lambda i, n: print(f'[ProPDE]:   Saving frame {i+1} of {n}'), dpi=600, fps=10, extra_args=['-vcodec', 'libx264'])
    anim.save(VideoFileName, dpi=600, fps=10, extra_args=['-vcodec', 'libx264'])

    plt.close()
