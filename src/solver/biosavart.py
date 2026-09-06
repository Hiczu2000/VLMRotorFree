import numpy as np
from preprocessor.GeometrySets import Mesh2D

# ---------------------------------------------------------------------------
# Vortex core models
# ---------------------------------------------------------------------------

def leishman_delta(gamma:np.array,
                   kinematic_viscosity:float,
                   a1:float) -> np.array:
    """
    Compute the delta parameters required
    for vortex core diffusion based on the 
    relation of Squire's hypothesis [3].

    Parameters
    ----------
    gamma : np.array
        Filament vortex strength
    kinematic_viscosity : float
        Kinematic viscosity [m^2/s]
    a1 : float
        Modelling constant [-],
        with suggested order of magnitude 
        of 10^-4 [1,2]
        
    Returns
    -------
    np.array
        Delta parameters used for the
        vortex core diffusion

    References
    -------
        1. S. Ananthan and J. G. Leishman. “Role 
        of Filament Strain in the Free-Vortex 
        Modeling of Rotor Wakes”. In: Journal of 
        the American Helicopter Society 49.2 (2004), 
        pp. 176–91. DOI: 10.4050/jahs.49.176.

        2. M. J. Bhagwat and J. G. Leishman. 
        “Generalized viscous vortex model for 
        application free-vortex wake and 
        aeroacoustic calculation”. In: (2002). 
        DOI: 10.1017/s0001925900003516.

        3. H. B. Squire. “The Growth of a 
        Vortex in Turbulent Flow”. In: Aeronautical 
        Quarterly 16.3 (1965),pp. 302–06.

    """
    Rev = np.abs(gamma) / kinematic_viscosity
    return 1 + a1 * Rev


class VortexCoreModels:
    """Base interface for vortex core / cutoff models used in the
    Biot-Savart induced velocity calculation.

    Subclasses implement __call__ to return the induction kernel factor K,
    and cutoff_stats to report diagnostic information about applied cutoffs.
    """
    label: str = "base"

    def __call__(self, nr_cross, nr1, nr2, dot_r01, dot_r02, L) -> np.ndarray: 
        """Return the induction kernel factor K for this core model.

        Subclasses override this with their specific formulation.
        """
        raise NotImplementedError

    def cutoff_stats(self) -> dict:
        """Return diagnostic statistics about the applied vortex core cutoff.

        Base implementation returns only the model label; subclasses may
        extend this with model-specific diagnostics.
        """
        return {"model": self.label}


class LaminarVortexCoreModels(VortexCoreModels):
    """
    Laminar vortex core model defined
    by Vatistas et al [1].

    References
    ----------
        1. G. H. Vatistas, V. Kozel, and W. C. Mih. 
        “A simpler model for concentrated vortices”. 
        In: Experiments in Fluids 11.1 (1991), 
        pp. 73–76. DOI: 10.1007/bf00198434.
    """
    label: str = "base_laminar"
    def __init__(self,
                 initial_core_size:float, 
                 model_order:int,
                 vortex_diffusion_times:np.array = None,
                 lambs_constant:float = None,
                 kinematic_viscosity:float = None,
                 a1:float = None,
                 epsilon:float = 1e-12):
        """
        Initiate the vortex core laminar model

        Parameters
        ----------
        initial_core_size : float
            Initial radius of the vortex core
        model_order : int
            Order of Vatistas model defining the
            shape between the transition of
            rigid body movement below the vortex
            core radius and free vortex movement
            above it
        vortex_diffusion_times : np.array, optional
            Vortex age times used for the vortex
            diffusion core model, 
            by default None
        lambs_constant : float, optional
            Lamb's constant by definition [1]
            equals to 1.25643, 
            by default None
        kinematic_viscosity : float, optional
            Kinematic viscosity used for the 
            vortex core diffusion [m^2/s], 
            by default None
        a1 : float
            Modelling constant [-],
            with suggested order of magnitude 
            of 10^-4 [1,2]
        epsilon : float, optional
            Threshold of finding to avoid the
            evaluation on the point too close
            to the vortex filament, 
            by default 1e-12

        Raises
        ------
        ValueError
            Inputs: a1, lambs_constant and
            kinematic viscosity cannot be None

        References
        -------
            1. S. Ananthan and J. G. Leishman. “Role 
            of Filament Strain in the Free-Vortex Modeling 
            of Rotor Wakes”. In: Journal of the American 
            Helicopter Society 49.2 (2004), pp. 176–91. 
            DOI: 10.4050/jahs.49.176.

            2. M. J. Bhagwat and J. G. Leishman. 
            “Generalized viscous vortex model for 
            application free-vortex wake and 
            aeroacoustic calculation”. In: (2002). 
            DOI: 10.1017/s0001925900003516.
        """
        
        self.lambs_constant = lambs_constant
        self.kinematic_viscosity = kinematic_viscosity
        self.initial_core_size = initial_core_size
        self.n = model_order        
        self.epsilon = epsilon
        self.a1 = a1
        
        if vortex_diffusion_times is None:
            self.update = False
        else:
            self.update = True
            
        if isinstance(initial_core_size,float):
            initial_core_size = initial_core_size
        else: 
            initial_core_size = initial_core_size.ravel()

        if vortex_diffusion_times is None:
            self.rc = initial_core_size 
        else:
            if a1 is None:
                raise ValueError("a1 is not defined, vortex diffusion cannot be implemented in the solver")
            if lambs_constant is None:
                raise ValueError("Lamb's constant is not defined, vortex diffusion cannot be implemented in the solver")
            if kinematic_viscosity is None:
                raise ValueError("Kinematic viscosity is not defined, vortex diffusion cannot be implemented in the solver")
            
            times = vortex_diffusion_times.ravel()
            self.times = times
            self.initial_core_size = initial_core_size
            

    def initiate_diffusion(self,
                           gamma:np.array):
        """
        Initiate the vortex diffusion
        model

        Parameters
        ----------
        gamma : np.array
            Vortex filament strengths
        """
        gamma = np.asarray(gamma).ravel()

        delta = leishman_delta(gamma = gamma,
                               kinematic_viscosity = self.kinematic_viscosity,
                               a1 = self.a1)
        
        res = self.initial_core_size ** 2 + (4 * self.lambs_constant * delta * self.kinematic_viscosity * self.times)
        self.rc = np.sqrt(res)


    def __call__(self, nr_cross, nr1, nr2, dot_r01, dot_r02, L) -> np.ndarray: 
        """
        Compute the Vatistas laminar core induction kernel factor K.

        Parameters
        ----------
        nr_cross : np.ndarray
            Norm of the cross product r1 x r2
        nr1, nr2 : np.ndarray
            Norms of vectors from filament endpoints to collocation points
        dot_r01, dot_r02 : np.ndarray
            Dot products of filament vector with r1, r2
        L : np.ndarray
            Vortex filament length

        Returns
        -------
        np.ndarray
            Factor K
        """
        n, rc = self.n, self.rc
        eps = self.epsilon
        
        term1 = np.zeros_like(nr_cross)
        term2 = np.zeros_like(nr_cross)
        np.divide(dot_r01, nr1, out=term1, where=nr1 >= eps)
        np.divide(dot_r02, nr2, out=term2, where=nr2 >= eps)
        denom = 4 * np.pi * ( (rc**(2*n)) * (L**(2*n)) + (nr_cross**(2*n)) ) ** (1/n)
        
        return (term1 - term2)/ denom 

    
    def cutoff_stats(self) -> dict:
        """
        Returns
        -------
        dict
            Return diagnostic statistics 
            about the applied vortex core cutoff.
        """
        return {
            "model":           self.label,
            "core radius":         self.rc
        }

    
    def update_model_in_time(self,
                             timestep:float,
                             n_spanwise:int = None,
                             gamma:np.array = None):
        """
        Update the diffusion model by the 
        time interval.

        Parameters
        ----------
        timestep : float
            Time step of the diffusion update
        n_spanwise : int, optional
            Number of spanwise vortex rings, 
            by default None
        gamma : np.array, optional
            Vortex filament strengths, 
            by default None
        
        Notes
        ----------
        Time update can be only used with the constant 
        timestep through the analysis        
        """
        if hasattr(self, 'times'):
            new_time_line = self.times[-n_spanwise:] + timestep
            self.times = np.hstack([self.times, new_time_line])

            if not np.isscalar(self.initial_core_size): 
                self.initial_core_size = np.hstack([self.initial_core_size,self.initial_core_size[-n_spanwise:]]) 
           
            self.initiate_diffusion(gamma = gamma)

        else:
            if not np.isscalar(self.rc):
                self.rc = np.hstack([self.rc,self.rc[-n_spanwise:]])


class Scully(LaminarVortexCoreModels):
    """
    First order Vatistas laminar 
    core model [1]

    References
    -------
        1. G. H. Vatistas, G. D. Panagiotakakos, 
        and F. I. Manikis. “Extension of the 
        n-Vortex Model to Approximate the Effects 
        of Turbulence”. In: Journal of Aircraft 
        52.5 (2015), pp. 1721–25. DOI:
        10.2514/1.c033238.
    """
    label:str = "scully"
    def __init__(self,
                 initial_core_size:float,
                 vortex_diffusion_times:np.array = None,
                 lambs_constant:float = None,
                 a1:float = None,
                 kinematic_viscosity:float = None,
                 epsilon:float = 1e-12):
        super().__init__(initial_core_size, 
                         model_order = 1,
                         vortex_diffusion_times = vortex_diffusion_times,
                         lambs_constant = lambs_constant,
                         a1 = a1,
                         kinematic_viscosity = kinematic_viscosity,
                         epsilon = epsilon)


class LambOseen(LaminarVortexCoreModels):
    """
        Second order Vatistas laminar 
        core model
    
    References
    -------
        1. G. H. Vatistas, G. D. Panagiotakakos, 
        and F. I. Manikis. “Extension of the 
        n-Vortex Model to Approximate the Effects 
        of Turbulence”. In: Journal of Aircraft 
        52.5 (2015), pp. 1721–25. DOI:
        10.2514/1.c033238.
    """
    label:str = "lamb_oseen"
    def __init__(self,
                 initial_core_size:float,
                 vortex_diffusion_times:np.array = None,
                 lambs_constant:float = None,
                 a1:float = None,
                 kinematic_viscosity:float = None):
        super().__init__(initial_core_size, 
                         model_order = 2,
                         vortex_diffusion_times = vortex_diffusion_times,
                         lambs_constant = lambs_constant,
                         a1 = a1,
                         kinematic_viscosity = kinematic_viscosity)


class KatzPlotkin(VortexCoreModels):
    """
    Cut off procedure defined in [1]
    
    References
    -------
        1. Joseph Katz and Allen Plotkin. 
        Low–speed aerodynamics. 10th printing. 
        New York, USA: Cambridge University Press, 
        2010, Chapter 10
    """
    label = "katz_plotkin"

    def __init__(self, epsilon:float = 0.0025):
        """
        Initiate the Katz & Plotkin cut off
        vortex model.

        Parameters
        ----------
        epsilon : float, optional
            Computation threshold, 
            by default 0.0025
        """
        self.epsilon = epsilon

    def __call__(self, nr_cross, nr1, nr2, dot_r01, dot_r02, **kwargs) -> np.ndarray:
        """
        Compute the Katz & Plotkin core induction kernel factor K.

        Parameters
        ----------
        nr_cross : np.ndarray
            Norm of the cross product r1 x r2
        nr1, nr2 : np.ndarray
            Norms of vectors from filament endpoints to collocation points
        dot_r01, dot_r02 : np.ndarray
            Dot products of filament vector with r1, r2
        
        Returns
        -------
        np.ndarray
            Factor K
        """

        eps = self.epsilon

        term1 = np.zeros_like(nr_cross)
        term2 = np.zeros_like(nr_cross)
        K     = np.zeros_like(nr_cross)

        np.divide(dot_r01, nr1, out=term1, where=nr1 >= eps)        
        np.divide(dot_r02, nr2, out=term2, where=nr2 >= eps)        
        np.divide(term1 - term2, 4 * np.pi * nr_cross**2,
                  out=K, where=nr_cross >= eps)
        
        # store masks for diagnostics
        self._mask_nr1   = nr1      >= eps
        self._mask_nr2   = nr2      >= eps
        self._mask_cross = nr_cross >= eps

        return K
    
    def cutoff_stats(self) -> dict:
        """
        Returns
        -------
        dict
            Return diagnostic statistics 
            about the applied vortex core cutoff.
        """
        total = self._mask_cross.size
        return {
            "model":           self.label,
            "epsilon":         self.epsilon,
            "total_elements":  total,
            "cutoff_nr1":      int(np.count_nonzero(~self._mask_nr1)),
            "cutoff_nr2":      int(np.count_nonzero(~self._mask_nr2)),
            "cutoff_nr_cross": int(np.count_nonzero(~self._mask_cross)),
            "active_pct":      f"{np.count_nonzero(self._mask_cross) / total:.2%}",
        }

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def biosavart_vectorized(collocation_points:Mesh2D,
                         filament_start:Mesh2D,
                         filament_end:Mesh2D,
                         sym:bool = False,
                         core_model: VortexCoreModels = None,
                         verbose: bool = False) -> np.ndarray:
    """
    Vectorized biosavart computation consistent 
    with the procedure described in [1]

    Parameters
    ----------
    collocation_points : Mesh2D
        Collocation point locations
    filament_start : Mesh2D
        Vortex filament starts location
    filament_end : Mesh2D
        Vortex filament ends location
    sym : bool, optional
        Calculate the mirror symmetric vortex grid,
        option for the wing case, 
        by default False
    core_model : VortexCoreModels, optional
        Vortex core model used for the fixed vortex rings, 
        by default None
    verbose : bool, optional
        Print the information about the used core models, 
        by default False

    Returns
    -------
    np.ndarray
        Induced velocity in X direction
    np.ndarray
        Induced velocity in Y direction
    np.ndarray
        Induced velocity in Z direction

    References
    -------
        1. Joseph Katz and Allen Plotkin. 
        Low–speed aerodynamics. 10th printing. 
        New York, USA: Cambridge University Press, 
        2010, Chapter 10
    """
    
    if core_model is None:
        core_model = KatzPlotkin()

    cp_x = collocation_points.X.ravel() 
    cp_y = collocation_points.Y.ravel() 
    cp_z = collocation_points.Z.ravel()
    
    x1 = filament_start.X.ravel()
    y1 = filament_start.Y.ravel()
    z1 = filament_start.Z.ravel()

    x2 = filament_end.X.ravel()
    y2 = filament_end.Y.ravel()
    z2 = filament_end.Z.ravel()

    if sym is True:
        y1 = -y1
        y2 = -y2
    
    cp_x = cp_x[:,np.newaxis] # (N, 1)
    cp_y = cp_y[:,np.newaxis]
    cp_z = cp_z[:,np.newaxis]

    x1 = x1[np.newaxis,:] # (1, M)
    y1 = y1[np.newaxis,:]
    z1 = z1[np.newaxis,:]

    x2 = x2[np.newaxis,:] # (1, M)
    y2 = y2[np.newaxis,:]
    z2 = z2[np.newaxis,:]

    r0x = x2 - x1 # (1, M)
    r0y = y2 - y1 
    r0z = z2 - z1

    r1x = cp_x - x1 # (N, M)
    r1y = cp_y - y1 
    r1z = cp_z - z1 
    
    r2x = cp_x - x2 # (N, M)
    r2y = cp_y - y2 
    r2z = cp_z - z2 
    
    # r1 and r2 cross product
    cr_x = r1y * r2z - r2y * r1z # (N, M)
    cr_y = r2x * r1z - r1x * r2z
    cr_z = r1x * r2y - r1y * r2x

    # norms
    nr_cross = np.sqrt(cr_x ** 2 + cr_y ** 2 + cr_z ** 2)
    nr1 = np.sqrt(r1x ** 2 + r1y ** 2 + r1z ** 2)
    nr2 = np.sqrt(r2x ** 2 + r2y ** 2 + r2z ** 2)
    
    # r0, r1 and r0, r2 dot products
    dot_r01 = r0x * r1x + r0y * r1y + r0z * r1z
    dot_r02 = r0x * r2x + r0y * r2y + r0z * r2z

    # vortex filament length
    L  = np.sqrt(r0x ** 2 + r0y ** 2 + r0z ** 2)
    K = core_model(nr_cross, nr1, nr2, dot_r01, dot_r02, L = L)
        
    U = K * cr_x # (N, M)
    V = K * cr_y
    W = K * cr_z

    if verbose:
        stats = core_model.cutoff_stats()
        print("\n=== Biot-Savart diagnostics ===")
        for k, v in stats.items():
            print(f"  {k:<20}: {v}")
 
    return U, V, W