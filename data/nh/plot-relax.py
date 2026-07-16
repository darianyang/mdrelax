"""
Plot the T4L NH NMR relaxation data at multiple field strengths.
"""

import numpy as np
import matplotlib.pyplot as plt

class RelaxPlot:
    """
    A class to plot NH NMR relaxation data at multiple field strengths.
    """

    def __init__(self, filename, ax=None):
        """
        Initialize the RelaxPlot object with a list of filenames.

        Parameters
        ----------
        filenames : str
            Path and filename containing the relaxation data.
        ax : matplotlib.axes.Axes, optional
            The axis on which to plot the data.
            If None, a new figure and axis will be created.
        """
        self.filename = filename
        if ax is None:
            self.fig, self.ax = plt.subplots(nrows=3, figsize=(7, 5))
        else:
            self.fig = plt.gcf()
            self.ax = ax
        
    def data_extract(self):
        """
        Extract the processed NH relaxation data from the filenames.
        """
        self.data = np.loadtxt(self.filename, delimiter='\t')
        return self.data

    def plot_data(self, ax_index, data_index):
        """
        Plot one set of relaxation data.

        Parameters
        ----------
        ax_index : int
            The index of the axis on which to plot the data.
        data_index : int
            The index of the data in the data array.
        """
        self.ax[ax_index].errorbar(self.data[:,0], self.data[:,data_index], 
                                   yerr=self.data[:,data_index + 1], 
                                   fmt='o', label=self.filename[:6],
                                   markersize=3)
        
    def plot_all_data(self):
        """
        Plot all relaxation data.
        """
        # plot R1, R2, and NOE data
        for ax_i, data_i in enumerate([1, 3, 5]):
            self.plot_data(ax_i, data_i)
    
    def plot_format(self):
        """
        Format the plot.
        """
        # set the x-axis label
        self.ax[2].set_xlabel('T4L Residue Number')
        # set the y-axis labels
        self.ax[0].set_ylabel('$R_1$ ($s^{-1}$)')
        self.ax[1].set_ylabel('$R_2$ ($s^{-1}$)')
        self.ax[2].set_ylabel('$^{15}N$-{$^1H$}-NOE')
        # set the title
        #self.ax[0].set_title('T4 Lysozyme NH Relaxation')
        # # set the y-axis limits
        # self.ax[0].set_ylim(0, 2)
        # self.ax[1].set_ylim(0, 20)
        self.ax[2].set_ylim(0, 1)
        # remove the top and right spines
        for ax in self.ax:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        # add a grid
        # for ax in self.ax:
        #     ax.grid(True)
        # add a legend
        self.ax[0].legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

    def run(self):
        """
        Main public class method for running the plotting code.
        """
        # extract the data
        self.data_extract()
        # plot r1, r2, and noe data
        self.plot_all_data()
        # format the plot
        self.plot_format()

if __name__ == '__main__':
    # create main fig and axes objects
    fig, ax = plt.subplots(nrows=3, figsize=(7, 5))

    # loop over the field strengths
    for hz in [500, 600, 800]:
        # create a RelaxPlot object
        relax = RelaxPlot(f'{hz}MHz-R1R2NOE.dat', ax=ax)
        # run the plotting code
        relax.run()

    # format and show the plot
    plt.tight_layout()
    plt.savefig("exp-relax.pdf")
    plt.show()