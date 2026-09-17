import os
import numpy as np
import torch
import pickle
import argparse

from tqdm import tqdm
import random


from modules.SO3lib import from_index_to_beta, bDelta, vectorized_wigner_d_matrix,\
        random_euler_zyz

if __name__ == '__main__':
    """
    This file is used to precompute Wigner d-matrices and other related quantities.
    They are saved in the precomp/ folder with the following naming scheme:
    - FULL_WIGNER_<n_beta>_<l_max>.list: full Wigner d-matrices for all orders up to l_max at the 
                specified latitude angle resolution n_beta.
    - SLICE_WIGNER_N0_<n_beta>_<l_max>.list: sliced Wigner d-matrices for n=0 (scalar fields) for
                all orders up to l_max at the specified latitude angle resolution n_beta.
    - SLICE_WIGNER_N1_<n_beta>_<l_max>.list: sliced Wigner d-matrices for n=1 (vector fields) for
                all orders up to l_max at the specified latitude angle resolution n_beta.
    - BDELTAS_<n_beta>_<l_max>.list: the b-deltas for all orders up to l_max at the specified
                latitude angle resolution n_beta. These are the Wigner d-matrices at beta=pi/2
                with the black delta convention
    - SINES_<n_beta>_<l_max>.torch: the sine of the latitude angles, used for the conj_prod
    - CONJ_PROD_<n_beta>_<l_max>.list: the precomputed torch.einsum() for the conj_prod
                of the b-deltas, i.e. the product of the b-deltas with their complex conjugate.
    """

    parser = argparse.ArgumentParser(description='Precomputation job')
    parser.add_argument('--n_beta', default="179", type=str, help='What is the latitude angle resolution?')
    """
    The latitude angle resolution is the number of points in the latitude direction, excluding the
    poles. Including the poles, the number of points would be n_beta+2 with the poles at -90° and
    +90° being the endpoints. The points are evenly spaced.
    """
    parser.add_argument('--l_max', default="179", type=str, help='What is the maximum order of the Wigner d-matrix?')
    """
    The maximum order of the Wigner d-matrices to be computed, at the specified latitude angle
    resolution. Most times this number is the same as the latitude resolution, as it provides the
    best approximation due to Nyquist theorem. However, sometimes it is useful to have a latitude
    resolution that is higher than the maximum order of the Wigner d-matrices, for example when
    applying rotations in the spectral domain.
    """
    parser.add_argument('--verbose', action='store_true', help='Print more information during the precomputation')
    parser.add_argument('--compute_wigner', action='store_true', help='Precompute Wigner d-matrices')
    parser.add_argument('--rotations', action='store_true', help='Precompute rotations for SO3Rotator')
    parser.add_argument('--n_rotations', default="1000", type=str, help='Number of random rotations to precompute')
    
    args = parser.parse_args()
    
    compute_wigner = bool(args.compute_wigner)
    l_max = int(args.l_max)
    n_beta = int(args.n_beta)
    verbose = bool(args.verbose)
    rotations = bool(args.rotations)
    n_rotations = int(args.n_rotations)

    if rotations:
        if verbose:
            print("Precomputing random rotations...")
            print("Precomputing validation rotations")
        list_of_rotations = []
        for i in tqdm(range(n_rotations), desc="Precomputing rotations"):
            rot = random_euler_zyz()
            list_of_rotations.append(rot)
        pickle.dump(list_of_rotations, open("precomp/FULL_VALIDATION_ROTATIONS.list","wb"))

        list_of_rotations = []
        for i in tqdm(range(n_rotations), desc="Precomputing rotations"):
            beta = random.randint(0,359)
            list_of_rotations.append(beta)
        pickle.dump(list_of_rotations, open("precomp/BETA_VALIDATION_ROTATIONS.list","wb"))

        if verbose:
            print("Precomputing test rotations")

        list_of_rotations = []
        for i in tqdm(range(n_rotations), desc="Precomputing rotations"):
            rot = random_euler_zyz(radians=True)
            list_of_rotations.append(rot)
        pickle.dump(list_of_rotations, open("precomp/FULL_TEST_ROTATIONS.list","wb"))

        list_of_rotations = []
        for i in tqdm(range(n_rotations), desc="Precomputing rotations"):
            beta = random.randint(0,359)
            list_of_rotations.append(beta)
        pickle.dump(list_of_rotations, open("precomp/BETA_TEST_ROTATIONS.list","wb"))
    
    if compute_wigner:

        if verbose:
            print(f"Precomputing Wigner d-matrices for n_beta={n_beta} and l_max={l_max}")

        COMPUTE_FULL_WIGNER = True
        COMPUTE_SLICE_WIGNER = True
        COMPUTE_BDELTAS = True
        # sin(from_index_to_beta(0,n_beta))
        COMPUTE_SINES = True
        # einsum(bdelta, conj(bdelta))
        COMPUTE_CONJ_PROD = True

        # Full wigner d-matrices
        full_wigner_d_list = []

        # Slice of wigner d-matrices corresponding to n=0 and n=1
        slice_wigner_0_list = []
        slice_wigner_1_list = []

        # Full wigner d-matrices corresponding to beta=pi/2 with the black delta convention
        bdeltas = []

        # Precomputation of torch.einsum()
        conj_prod = []

        #Check if the files have already been created, and skip them if they exist
        if os.path.exists("precomp/FULL_WIGNER_"+str(n_beta)+"_"+str(l_max)+".list"):
            COMPUTE_FULL_WIGNER = False
            if verbose:
                print("Full d-matrices found! Skipping computations of full Wigner d-matrices.")
        if os.path.exists("precomp/SLICE_WIGNER_N0_"+str(n_beta)+"_"+str(l_max)+".list") \
            and os.path.exists("precomp/SLICE_WIGNER_N1_"+str(n_beta)+"_"+str(l_max)+".list"):
            COMPUTE_SLICE_WIGNER = False
            if verbose:
                print("All slices found! Skipping computations of sliced Wigner d-matrices.")
        if os.path.exists("precomp/BDELTAS_"+str(n_beta)+"_"+str(l_max)+".list"):
            COMPUTE_BDELTAS = False
            if verbose:
                print("b-deltas found! Skipping computations of b-deltas.")
        if os.path.exists("precomp/SINES_"+str(n_beta)+"_"+str(l_max)+".torch"):
            COMPUTE_SINES = False
            if verbose:
                print("Sines found! Skipping computations of sines.")

        sines = None

        os.makedirs(os.path.dirname("precomp/"), exist_ok=True)

        if COMPUTE_SINES:
            sines = torch.sin(from_index_to_beta(torch.arange(0,n_beta), n_beta=n_beta)).to(torch.complex64)
            pickle.dump(sines, open("precomp/SINES_"+str(n_beta)+"_"+str(l_max)+".torch","wb"))
            if verbose:
                print("Saved angles/sines in precomp/SINES_"+str(n_beta)+"_"+str(l_max)+".torch")
        else:
            if verbose:
                print("Skipping computations of angles/sines")
        #elif COMPUTE_CONJ_PROD:
            # We have not computed the sines, but we need them to compute the conj_prod, so we load them from memory
            #sines = pickle.load(open("precomp/SINES.torch","rb"))

        if COMPUTE_BDELTAS or COMPUTE_FULL_WIGNER or COMPUTE_SLICE_WIGNER:
            if verbose:
                print("Computing Wigner d-matrices and storing them in precomp/")
            for l in tqdm(range(0,l_max+1)):
                angle = from_index_to_beta(np.arange(0,n_beta), n_beta=n_beta)
                aa = vectorized_wigner_d_matrix(l,angle)
                full_wigner_d_list.append(aa.requires_grad_(False))
                # `.contiguous()` matters: pickling a view writes out the whole
                # backing tensor, which made each slice file as large as the full one.
                slice_wigner_0_list.append(aa[:,:,l+0].contiguous().requires_grad_(False))
                if l>0: # The sliced version for n=1 cannot be computed for l=0 
                    slice_wigner_1_list.append(aa[:,:,l+1].contiguous().requires_grad_(False))
                bdeltas.append(bDelta(l, aa[n_beta//2]).requires_grad_(False)) # n_beta//2 = pi/2
            pickle.dump(full_wigner_d_list, open("precomp/FULL_WIGNER_"+str(n_beta)+"_"+str(l_max)+".list","wb"))
            if verbose:
                print("Saved full Wigner d-matrices in precomp/FULL_WIGNER_"+str(n_beta)+"_"+str(l_max)+".list")
            pickle.dump(slice_wigner_0_list, open("precomp/SLICE_WIGNER_N0_"+str(n_beta)+"_"+str(l_max)+".list","wb"))
            pickle.dump(slice_wigner_1_list, open("precomp/SLICE_WIGNER_N1_"+str(n_beta)+"_"+str(l_max)+".list","wb"))
            if verbose:
                print("Saved full Wigner d-matrices in precomp/SLICE_WIGNER_"+str(n_beta)+"_"+str(l_max)+".list")
            pickle.dump(bdeltas, open("precomp/BDELTAS_"+str(n_beta)+"_"+str(l_max)+".list","wb"))
            if verbose:
                print("Saved full Wigner d-matrices in precomp/BDELTAS_"+str(n_beta)+"_"+str(l_max)+".list")
        else:
            if verbose:
                print("Skipping computations of wigner-d matrices")

        if COMPUTE_CONJ_PROD:
            if verbose:
                print("Loading bdeltas from file in order to compute conj_prod")
            bdeltas = pickle.load(open("precomp/BDELTAS_"+str(n_beta)+"_"+str(l_max)+".list","rb"))
            if verbose:
                print("Computing conj_prod")
            for l in tqdm(range(0,l_max+1)):
                conj_prod.append(torch.einsum("sm,sn -> msn",
                                            bdeltas[l],
                                            torch.conj(bdeltas[l]),
                                ).requires_grad_(False))
            pickle.dump(conj_prod, open("precomp/CONJ_PROD_"+str(n_beta)+"_"+str(l_max)+".list","wb"))
            if verbose:
                print("Saved conjugate products in precomp/CONJ_PROD_"+str(n_beta)+"_"+str(l_max)+".list")
