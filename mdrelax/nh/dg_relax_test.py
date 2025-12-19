import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def ACFfft(q, corrdim, corrstride):
    
    f = q[::corrstride]
    N = f.shape[0]
    
    # FFT-based autocorrelation
    fvi = np.fft.fft(f, n=2*N)
    acf = np.fft.ifft(fvi * np.conjugate(fvi))[:N].real

    # Normalize by number of points
    acf /= (N - np.arange(N))
    
    # Subtract square of mean
    acf -= np.mean(f)**2

    return acf[:corrdim]

def compute_all_acfs(chem_shifts, corrdim, corrstride, dt):
    """Compute ACF for each residue in parallel."""
    def compute_one_acf(row):
        centered = row - np.mean(row)
        return ACFfft(centered, corrdim, corrstride)

    # list append version
    # ACF_all = []
    # for i in tqdm(range(chem_shifts.shape[0])):
    #     ACF_all.append(compute_one_acf(chem_shifts[i]))

    # precast the ACF array (adds small overhead but uses less memory)
    ACF_all = np.empty((chem_shifts.shape[0], corrdim), dtype=np.float32)
    for i in tqdm(range(chem_shifts.shape[0])):
        ACF_all[i] = compute_one_acf(chem_shifts[i])
    
    tauaxis = np.arange(corrdim) * dt * corrstride
    
    return ACF_all, tauaxis

def compute_all_acfs_fft(chem_shifts, corrdim, corrstride, dt):
    """
    FFT-based ACF for all residues at once.
    """
    # Subsample
    f = chem_shifts[:, ::corrstride]

    # Remove mean per residue
    f = f - f.mean(axis=1, keepdims=True)

    n_res, N = f.shape
    n_fft = 2 * N

    # FFT along time axis
    F = np.fft.fft(f, n=n_fft, axis=1)

    # Autocorrelation via Wiener–Khinchin
    acf = np.fft.ifft(F * np.conjugate(F), axis=1).real[:, :N]

    # Finite-length normalization
    norm = (N - np.arange(N))[None, :]
    acf /= norm

    # Truncate
    acf = acf[:, :corrdim]

    tauaxis = np.arange(corrdim) * dt * corrstride
    return acf.astype(np.float32), tauaxis


if __name__ == "__main__":
    import time

    # Example usage
    # normal: (164, 1000000000)
    #chem_shifts = np.random.rand(100, 10000)  # 100 residues, 10000 time points
    # set constant seed for reproducibility
    rng = np.random.default_rng(42)
    chem_shifts = rng.random(size=(164, 10_000_000), dtype=np.float32)
    corrdim = 2**19  # Length of ACF to compute
    #print(f"Chem shifts: {chem_shifts[0]}")

    # Compute ACFs residue-wise and time it
    start_time = time.time()
    ACF_all, tauaxis = compute_all_acfs(
        chem_shifts, 
        corrdim=corrdim, 
        corrstride=1, 
        dt=0.03,  
    )
    end_time = time.time()
    print(f"Residue-wise FFT ACF computation time: {end_time - start_time:.2f} seconds")
    #print("ACF shape:", ACF_all.shape)

    # Compute ACFs vectorized and time it
    start_time = time.time()
    ACF_all_fft, tauaxis_fft = compute_all_acfs_fft(
        chem_shifts, 
        corrdim=corrdim, 
        corrstride=1, 
        dt=0.03,  
    )
    end_time = time.time()
    print(f"Vectorized FFT ACF computation time: {end_time - start_time:.2f} seconds")
    #print("ACF FFT shape:", ACF_all_fft.shape)

    # Check difference
    diff = np.abs(ACF_all - ACF_all_fft)
    print("Max difference between methods:", np.max(diff))
    # print("ACF first 10 values residue-wise:", ACF_all[0][:10])
    # print("ACF first 10 values vectorized:", ACF_all_fft[0][:10])
    assert np.allclose(ACF_all, ACF_all_fft, atol=1e-6), "ACF results do not match!"

    # # Plot ACF for the first residue
    plt.plot(tauaxis, ACF_all[0])
    plt.plot(tauaxis_fft, ACF_all_fft[0], linestyle='dashed')
    plt.xlabel('Time Lags')
    #plt.xscale('log')
    plt.ylabel('ACF')
    #plt.yscale('log')
    plt.savefig('acf_plot.png', dpi=300)
