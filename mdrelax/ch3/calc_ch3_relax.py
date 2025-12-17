import MDAnalysis as mda
from MDAnalysis.analysis import align

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tqdm.auto import tqdm

# missing elements warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis.topology.PDBParser")

# TODO: eventually inherit from a Relaxation base class? then children for each spin system?
class NH_Relaxation:
    """
    Backbone Amide Relxation Rate Calculations from MD Simulations.
    """
    # Constants
    mu_0 = 4 * np.pi * 1e-7     # Permeability of free space (N·A^-2)
    hbar = 1.0545718e-34        # Reduced Planck's constant (J·s) (h/2pi)
    gamma_H = 267.513e6         # Gyromagnetic ratio of 1H (rad·s^-1·T^-1)
    gamma_N = -27.116e6         # Gyromagnetic ratio of 15N (rad·s^-1·T^-1)
    r_NH = 1.02e-10             # N-H bond length (meters)
    Delta_sigma = -170 * 1e-6   # CSA value (ppm) -170 ppm --> dimensionless units
    #Delta_sigma = 0             # CSA value (ppm)

    # Derived parameters
    d_oo = (1 / 20) * (mu_0 / (4 * np.pi))**2 * hbar**2 * gamma_H**2 * gamma_N**2
    d_oo *= r_NH**-6  # Scale by bond length to the power of -6
    c_oo = (1 / 15) * Delta_sigma**2

    def __init__(self, pdb, traj, traj_start=None, traj_stop=None, traj_step=1, traj_align=False,
                 max_lag=None, n_exps=5, acf_plot=False, tau_c=None, b0=600):
        """
        Initialize the RelaxationCalculator with simulation and analysis parameters.

        Parameters
        ----------
        pdb : str
            Path to the PDB or topology file.
        traj : str
            Path to the trajectory file.
        traj_start : int, optional
            The starting frame index for the trajectory (default is None).
        traj_stop : int, optional
            The stopping frame index for the trajectory (default is None).
        traj_step : int, optional
            Step interval for loading the trajectory (default is 10).
        traj_align : bool, optional
            Whether to align the trajectory to the reference pdb frame (default is False).
        max_lag : int, optional
            Maximum lag time for ACF computation (default is None, uses entire traj).
        n_exps : int, optional
            Number of exponential functions for ACF fitting (default is 5).
        acf_plot : bool, optional
            Whether to plot the ACF and its fit (default is False).
        tau_c : float, optional
            Overall tumbling time in seconds (default is None).
            Can input a value or otherwise will calculate it from simulation.
        b0 : float, optional
            Magnetic field strength in MHz (1H) (default is 600).
        """
        self.pdb = pdb
        self.traj = traj
        self.traj_start = traj_start
        self.traj_stop = traj_stop
        self.traj_step = traj_step
        self.traj_align = traj_align
        self.max_lag = max_lag
        self.n_exps = n_exps
        self.acf_plot = acf_plot
        self.tau_c = tau_c

        # Nuclei frequencies
        self.omega_H = b0 * 2 * np.pi * 1e6          # Proton frequency (rad/s)
        self.omega_N = self.omega_H / 10.0           # ~Nitrogen frequency (rad/s)

        self.u = self.load_align_traj()

        # initialize max_lag if not provided
        if self.max_lag is None:
            # Use the entire trajectory length for the ACF (n_frames)
            # limit by default to first 50% of the trajectory? (TODO)
            self.max_lag = int((len(self.u.trajectory) * 0.5))
            #self.max_lag = int(unit_vectors.shape[0] * 0.5)
            #print(f"max_lag not provided, setting to {self.max_lag} frames.")

    def load_align_traj(self):
        """
        Load and align input trajectory.

        Returns
        -------
        u : MDAnalysis.Universe
            The MDAnalysis Universe object containing the trajectory.
        """
        # Load the alanine dipeptide trajectory
        u = mda.Universe(self.pdb, self.traj, in_memory=True, in_memory_step=self.traj_step)

        # Align trajectory to the reference frame / pdb
        # removes overall translation and rotation
        if self.traj_align:
            ref = mda.Universe(self.pdb, self.pdb)
            align.AlignTraj(u, ref, select='name CA', in_memory=True).run()

        return u

    def compute_nh_vectors(self, start=None, stop=None, step=None):
        """
        Calculate NH bond vectors for each frame in the trajectory.

        Parameters
        ----------
        start : int, optional
            The starting frame index.
        stop : int, optional
            The stopping frame index.
        step : int, optional
            The step size between frames.

        Returns
        -------
        nh_vectors: numpy.ndarray
            An array of NH bond vectors with shape (n_frames, n_pairs, 3).
            Each entry corresponds to a bond vector for a specific frame and pair.
        """
        # Select the atoms involved in NH bonds
        # no prolines or the first N-terminal residue nitrogen
        selection = self.u.select_atoms('(name N or name H) and not resname PRO and not resnum 1')

        # Determine the number of frames and NH pairs
        n_frames = len(self.u.trajectory[start:stop:step])
        n_pairs = len(selection) // 2

        # Pre-cast a numpy array to store NH bond vectors
        nh_vectors = np.zeros((n_frames, n_pairs, 3))

        # Iterate over the trajectory frames and calculate NH bond vectors
        for i, _ in enumerate(self.u.trajectory[start:stop:step]):
            nh_vectors[i] = selection.positions[1::2] - selection.positions[::2]

        # list of the residue index for each NH pair
        self.residue_indices = np.array([atom.resid for atom in selection.atoms if atom.name == 'H'])
        # print("Residue Indices: ", self.residue_indices.shape, self.residue_indices, [i for i in selection.atoms])
        # print("NH Vectors Shape: ", nh_vectors.shape)

        # n_nitrogen = len([atom for atom in selection if atom.name == 'N'])
        # n_hydrogen = len([atom for atom in selection if atom.name == 'H'])
        # print(f"Number of Nitrogen atoms: {n_nitrogen}")
        # print(f"Number of Hydrogen atoms: {n_hydrogen}")

        #print(f"NH vectors: {nh_vectors[0,:,0]}")

        return nh_vectors

    # Compute ACF for the NH bond vectors => C_I(t)
    def calculate_acf(self, vectors):
        """
        Calculate the autocorrelation function (ACF) for NH bond vectors using the 
        second-Legendre polynomial.

        Parameters
        ----------
        vectors : numpy.ndarray
            A 3D array of shape (n_frames, n_bonds, 3), where each entry represents
            an NH bond vector at a specific time frame.

        Returns
        -------
        numpy.ndarray
            A 1D array of size `max_lag` containing the normalized autocorrelation
            function for each lag time.
        """
        # Normalize the NH bond vectors to unit vectors
        # TODO: here, we normalize over the norm or length of the NH bond vectors
        #       but note that the bond lengths are fixed in the simulation (SHAKE)
        #       so the norm is not fully accurate, need some adjustment to correct this
        unit_vectors = vectors / np.linalg.norm(vectors, axis=2, keepdims=True)
        #unit_vectors = vectors / (np.linalg.norm(vectors, axis=2, keepdims=True) + 0.02) # testing bond length correction factor
        # print("Vectors: ", vectors)
        # print("Bond Length: ", np.linalg.norm(vectors, axis=2, keepdims=True))
        # print("Unit Vectors: ", unit_vectors)
        #print("unit vector shape", unit_vectors.shape)

        # Initialize the array to store the ACF for each lag
        correlations = np.zeros((self.max_lag, unit_vectors.shape[1]), dtype=np.float64)

        # Loop over lag times
        for lag in tqdm(range(self.max_lag), desc="Calculating ACF"):
            # Compute dot products for all vectors separated by 'lag' frames
            dot_products = np.einsum(
                'ijk,ijk->ij', 
                unit_vectors[:-lag or None],    # Frames from 0 to len(vectors) - lag
                unit_vectors[lag:]              # Frames from lag to the end
            )
            
            # Apply the second-Legendre polynomial P2(x) = 0.5 * (3x^2 - 1)
            p2_values = 0.5 * (3 * dot_products**2 - 1)
            #print("P2 Shape: ", p2_values.shape)

            # Compute the mean over all time points for each NH bond vector
            correlations[lag, :] = np.nanmean(p2_values, axis=0)

        #print("Correlations Shape: ", correlations.shape)
        return correlations

    def calculate_acf_fft(self, vectors):
        """
        Compute the NH bond vector autocorrelation function using an FFT-based method.

        Computes:
            C(t) = < P2( u(0) · u(t) ) >
        where:
            P2(x) = 0.5 * (3x^2 - 1)

        Parameters
        ----------
        vectors : np.ndarray
            Array of shape (n_frames, n_bonds, 3)

        Returns
        -------
        np.ndarray
            ACF array of shape (max_lag, n_bonds)
        """
        # Normalize NH bond vectors
        u = vectors / np.linalg.norm(vectors, axis=2, keepdims=True)

        n_frames, n_bonds, _ = u.shape
        n_fft = 2 * n_frames  # zero-padding prevents circular convolution

        # Build quadratic Cartesian components
        # Q = [xx, yy, zz, xy, xz, yz]
        Q = np.empty((n_frames, n_bonds, 6), dtype=np.float64)
        Q[..., 0] = u[..., 0] * u[..., 0]  # xx
        Q[..., 1] = u[..., 1] * u[..., 1]  # yy
        Q[..., 2] = u[..., 2] * u[..., 2]  # zz
        Q[..., 3] = u[..., 0] * u[..., 1]  # xy
        Q[..., 4] = u[..., 0] * u[..., 2]  # xz
        Q[..., 5] = u[..., 1] * u[..., 2]  # yz

        # FFT autocorrelation of each component
        F = np.fft.fft(Q, n=n_fft, axis=0)
        acf_Q = np.fft.ifft(F * np.conjugate(F), axis=0).real

        # Keep only physical lags
        acf_Q = acf_Q[:n_frames]

        # Finite-length normalization
        norm = np.arange(n_frames, 0, -1, dtype=np.float64)[:, None, None]
        acf_Q /= norm

        # Reconstruct < (u(0) · u(t))^2 >
        dot2 = (
            acf_Q[..., 0] +
            acf_Q[..., 1] +
            acf_Q[..., 2] +
            2.0 * (
                acf_Q[..., 3] +
                acf_Q[..., 4] +
                acf_Q[..., 5]
            )
        )

        # Apply second-Legendre polynomial
        acf = 0.5 * (3.0 * dot2 - 1.0)

        # Truncate to requested lag time
        return acf[:self.max_lag]

    
    # Method to estimate tau_c from the ACF
    # TODO: could update to give better initial guess, and check units
    #       there is prob also a more accurate or correct way to do this (maybe with MF)
    def estimate_tau_c(self, acf_values):
        """
        Estimate the rotational correlation time (tau_c) from the ACF by fitting it to a 
        single exponential decay function.

        Parameters
        ----------
        acf_values : np.ndarray
            The ACF values to fit, with shape (max_lag, n_bonds).
        
        Returns
        -------
        float
            Estimated rotational correlation time (tau_c).
        """
        # Define the exponential decay function
        # Global tumbling: C_O(t) = exp(-t/tau_c)
        def exp_decay(t, tau_c):
            return np.exp(-t / tau_c)
        
        # Generate time lags
        time_lags = np.arange(acf_values.shape[0])
        
        # Flatten the ACF values and repeat the time lags for global fitting
        flattened_acf_values = acf_values.flatten()
        repeated_time_lags = np.tile(time_lags, acf_values.shape[1])
        
        # Initial guess for the tau_c parameter (None)
        initial_tau_c = self.tau_c
        
        # Perform the global fit using curve_fit
        popt, _ = curve_fit(exp_decay, repeated_time_lags, flattened_acf_values, p0=initial_tau_c)
        
        # Extract the optimized tau_c
        tau_c_estimate = popt[0]

        # Optionally plot the single exponential fit to the ACF background
        if self.acf_plot:
            plt.figure()
            # for i in range(acf_values.shape[1]):
            #     plt.plot(time_lags, acf_values[:, i], label=f'ACF {i}')
            plt.plot(time_lags, acf_values)
            plt.plot(time_lags, exp_decay(time_lags, tau_c_estimate), linestyle="--", color='black', linewidth=2)
            plt.title("tau_c Estimate from ACF")
            #plt.legend()
            plt.savefig("acf_tau_c.png", dpi=300)

        return tau_c_estimate
    
if __name__ == "__main__":
    # Example usage
    pdb_file = "test_ff15ipq/initial.pdb"
    traj_file = "test_ff15ipq/sim1-10ns_nopbc.xtc"
    #traj_file = "test_ff15ipq/sim1-10ns_nopbc_rot_trans.xtc"

    relax_calc = NH_Relaxation(
        pdb=pdb_file,
        traj=traj_file,
        traj_start=0,
        traj_stop=None,
        traj_step=1,
        max_lag=None,
        n_exps=5,
        acf_plot=True,
        tau_c=None,
        b0=600
    )

    nh_vectors = relax_calc.compute_nh_vectors()
    #acf_values = relax_calc.calculate_acf(nh_vectors)
    acf_values = relax_calc.calculate_acf_fft(nh_vectors)
    
    #np.testing.assert_allclose(acf_values, acf_values_fft, rtol=1e-5, atol=1e-8)

    # Estimate tau_c
    estimated_tau_c = relax_calc.estimate_tau_c(acf_values)
    print(f"Estimated tau_c: {estimated_tau_c} time units")

    print(acf_values.shape)