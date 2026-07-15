import MDAnalysis as mda
import numpy as np

def pdb_inertia(input_pdb, output_pdb, results_file):
    # Load the PDB file
    u = mda.Universe(input_pdb)

    # Select all atoms
    selection = u.atoms

    # Number of atoms
    num_atoms = len(selection)
    num_atoms_skipped = 0

    # Calculate the total mass of the molecule
    total_mass = selection.total_mass()

    # Calculate the center of mass (COM)
    com = selection.center_of_mass()

    # Translate the structure so the COM is at the origin
    selection.translate(-com)

    # Compute the inertia tensor
    inertia_tensor = selection.moment_of_inertia()

    # Diagonalize the inertia tensor to get principal moments and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)

    # Sort principal moments (largest to smallest) and reorder eigenvectors
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Calculate relative moments
    relative_moments = eigvals / eigvals.max()

    # Ensure the rotation matrix has a determinant of 1 (right-handed)
    # Flip the signs of the first or third rows of the rotation matrix if needed
    rotation_matrix = eigvecs.T
    if np.linalg.det(rotation_matrix) < 0:
    #    rotation_matrix[0] = -rotation_matrix[0]
        rotation_matrix[2] = -rotation_matrix[2]

    # Rotate the molecule to align the principal axes with the Cartesian axes
    selection.rotate(rotation_matrix)

    # Write the rotated structure to a new PDB file
    selection.write(output_pdb)

    # Write the results to the output file
    with open(results_file, "w") as f:
        f.write(f"Input pdb file:  {input_pdb}\n")
        f.write(f"Output pdb file: {output_pdb}\n")
        f.write(f" # atoms read {num_atoms:>10d}      # atoms skipped {num_atoms_skipped:>10d}\n\n")
        f.write(f" mass {total_mass:>30.4f}\n")
        f.write(f" center of mass {com[0]:>15.4f} {com[1]:>15.4f} {com[2]:>15.4f}\n")
        f.write(f" principle moments {eigvals[0]:>15.4f} {eigvals[1]:>15.4f} {eigvals[2]:>15.4f}\n")
        f.write(f" relative moments {relative_moments[0]:>15.4f} {relative_moments[1]:>15.4f} {relative_moments[2]:>15.4f}\n\n")
        f.write("                           rotation matrix\n")
        for row in rotation_matrix:
            f.write(f"{row[0]:>15.4f} {row[1]:>15.4f} {row[2]:>15.4f}\n")
    
    print("Results written to:", results_file)
    print("Rotated structure written to:", output_pdb)

# Example usage
pdb_inertia("1ubq.prot.pdb", "test-1ubq.inertia.pdb", "test-results.txt")

