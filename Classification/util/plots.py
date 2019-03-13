# This is utility functions used to generate plots.
import matplotlib.pyplot as mplot
import numpy


def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# plot_history plots accuracy and loss of a Neural Network model.
# history = model.fit_generator(....)
# plot_history(history)
def plot_history(history, save=False, path='', name='history', mean_N=0, ylim_acc=(-1,-1), ylim_loss=(-1,-1)):
    #print(history.history.keys())
    fig = mplot.figure()
    if (ylim_acc != (-1, -1)):
        mplot.ylim(ylim_acc)
    if (mean_N == 0):
        mplot.plot(history.history['acc'])
        mplot.plot(history.history['val_acc'])
    else:
        mplot.plot(running_mean(history.history['acc'],mean_N))
        mplot.plot(running_mean(history.history['val_acc'],mean_N))
    mplot.title('model accuracy')
    mplot.ylabel('accuracy')
    mplot.xlabel('epoch')
    mplot.legend(['train', 'val'], loc='upper left')
    mplot.show()
    # summarize history for loss
    fig_loss = mplot.figure()
    if (ylim_loss != (-1,-1)):
        mplot.ylim(ylim_loss)
    if (mean_N == 0):
        mplot.plot(history.history['loss'])
        mplot.plot(history.history['val_loss'])
    else:
        mplot.plot(running_mean(history.history['loss'],mean_N))
        mplot.plot(running_mean(history.history['val_loss'],mean_N))
    mplot.title('model loss')
    mplot.ylabel('loss')
    mplot.xlabel('epoch')
    mplot.legend(['train', 'val'], loc='upper left')
    mplot.show()
    if (save == True):
        fig.savefig(path+name+'_acc.png', dpi=fig.dpi)
        fig_loss.savefig(path+name+'_loss.png', dpi=fig.dpi)