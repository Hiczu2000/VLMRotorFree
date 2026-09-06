import numpy as np
from preprocessor import VectorSet, Mesh2D
from dataclasses import dataclass
import matplotlib.pyplot as plt
from utilities import log, apply_thesis_style, thesis_plot, save_figure

apply_thesis_style()

@dataclass
class Airfoil:
    """
    Aerodynamic polar data for a single airfoil, used for viscous
    drag interpolation.

    Attributes
    ----------
    name : str
        Airfoil identifier, matches the source polar filename
        (cl_cd_<name>.dat)
    relative_thickness : float
        Relative thickness of the airfoil [%]
    cl_cd_values : np.ndarray
        Lift to drag ratio (Cl/Cd) [-] at each cl_values point
    cl_values : np.ndarray
        Lift coefficient values [-] corresponding to cl_cd_values,
        used as the interpolation axis
    """
    name: str
    relative_thickness: float
    cl_cd_values: np.ndarray
    cl_values: np.ndarray


class DragCoefficientInterpolator():
    """
    Imports and contains airfoil Cl/Cd and Cl data
    and interpolates Cd for target relative thickness and lift coefficient
    """
    def __init__(self,
                 airfoil_names: list,
                 cylinder_drag_coefficient:float = 0.6):
        """
        Initiate drag coefficient interpolator

        Parameters
        ----------
        airfoil_names : list
            List of the airfoil polars used for
            the viscous drag interpolation.
        cylinder_drag_coefficient : float, optional
            Drag coefficient used for the interpolation 
            of profiles with relative thickness above
            the highest defined one in the airfoil set, 
            by default 0.6
        """
        
        self.airfoils: list[Airfoil] = []
        self.cylinder_cd = cylinder_drag_coefficient

        for name in airfoil_names:
            airfoil = np.loadtxt('blade_input/cl_cd_' + name + ".dat")
            thickness = float(name[-3:]) / 10
            cl_cd = airfoil[:,1]
            cl = airfoil[:,0]
            self.airfoils.append(Airfoil(name, thickness, cl_cd, cl))

    def interpolate(self,
                    target_thickness:float,
                    target_lift_coefficient:float,
                    extrapolate:bool = False,
                    tol:float = 0.01,
                    verbose:bool = False,
                    plot_interpolation:bool = False) -> float:
        """Gives the drag coefficient for the target relative thickness 
        and target lift coefficient

        Parameters
        ----------
        target_thickness : float
            Relative thickness of the target profile [%]
        target_lift_coefficient : float
            Lift coefficient of the target profile [-]
        extrapolate : bool, optional
            Option of an extrapolation the Cl/Cd out of
            the defined Cl data by the linear scipy.interp1d
            extrapolation, when false, the data is clamped at 
            the border value, by default (uses numpy.interp)
        tol : float, optional
            Tolerance in finding an airfoil with
            the closest relative thickness to the 
            target one [%], by default 0.01
        verbose : bool, optional
            Option of showing the logs about the 
            drag coefficient approximation, by default False
        plot_interpolation : bool, optional
            Option of showing the plots of the
            drag coefficient approximation, by default False

        Returns
        -------
        float
            Approximated drag coefficient [-]

        Raises
        ------
        ValueError
            Input target lift coefficient must be above 0,
            polar inputs are defined only for the positve values
        ValueError
            Input target thickness must be above 0
        ValueError
            Input target thickness must be below 100
        """
        
        if target_lift_coefficient <= 0:
            raise ValueError(f"CL must be > 0, got {target_lift_coefficient}")
        
        if target_thickness <= 0:
            raise ValueError(f"Target thickness must be > 0, got {target_thickness}")
        
        if target_thickness > 100:
            raise ValueError(f"Target thickness must be <= 100%, got {target_thickness}")
        
        thicknesses = np.array([a.relative_thickness for a in self.airfoils])

        ## thickness above the defined airfoil with the highest one
        if target_thickness > np.max(thicknesses) + tol: 
            idx_max = np.argmax(thicknesses)
            airfoil_max = self.airfoils[idx_max]
            if target_lift_coefficient > np.max(airfoil_max.cl_values) and extrapolate is True:
                from scipy.interpolate import interp1d
                cl_cd_vs_cl= interp1d(airfoil_max.cl_values, airfoil_max.cl_cd_values, fill_value="extrapolate")
                cl_cd_max = cl_cd_vs_cl(target_lift_coefficient)
            else: 
                cl_cd_max = np.interp(target_lift_coefficient, airfoil_max.cl_values, airfoil_max.cl_cd_values)
            
            if cl_cd_max < 0:
                log(f"Obtained lift to drag coefficient is below 0, the final values has been clamped at the cylinder drag coefficient {self.cylinder_cd} to avoid unphysical results","WARN",color = True)
                cd_max = self.cylinder_cd
            else:
                cd_max = target_lift_coefficient / cl_cd_max

            thickness_points = np.array([airfoil_max.relative_thickness, 100]) #cylinder has always 100% thickness
            drag_coefficients = np.array([cd_max, self.cylinder_cd])

            cd = np.interp(target_thickness, thickness_points, drag_coefficients)

            if verbose:
                log(f"Target thickness {target_thickness} above the highest defined airfoil thickness {np.max(thicknesses)}", "WARN", color = True)
                log(f"Values are interpolated between airfoil with thickness {np.max(thicknesses)} and cylinder with drag coefficient {self.cylinder_cd}", "WARN", color = True)
                log(f"Interpolated Cd for target thickness {target_thickness} and lift coefficient {target_lift_coefficient} is {cd:.3f}", color = True)
                if target_lift_coefficient > np.max(airfoil_max.cl_values):
                    if extrapolate:
                        log(f"The target lift coefficient {target_lift_coefficient:3f} is above the defined data in {airfoil_max.name}, values have been extrapolated","WARN", color = True)
                    else:     
                        log(f"The target lift coefficient {target_lift_coefficient:3f} is above the defined data in {airfoil_max.name}, the boundary value has been used","WARN", color = True)
                        
            if plot_interpolation:
                fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
                thesis_plot(axs[0],
                            x = airfoil_max.cl_values,
                            y = airfoil_max.cl_cd_values,
                            label = airfoil_max.name,
                            xlabel = r"$c_l$ " + '[-]',
                            ylabel = r"$c_l/c_d$ " + '[-]')
                
                thesis_plot(axs[0],
                    x = target_lift_coefficient,
                    y = cl_cd_max,
                    style = "point",
                    label = "interpolated value")

                thesis_plot(axs[1],
                            x = thickness_points,
                            y = drag_coefficients,
                            label = "interpolation data",
                            xlabel = "relative thickness [%]",
                            ylabel = r"$c_d$ " + '[-]')

                thesis_plot(axs[1],
                            x = target_thickness,
                            y = cd,
                            style = "point",
                            label = "interpolated value")
                plt.show()
                save_figure(fig,
                            name = 'drag_interp1')

        ## thickness is inside the range of the lowest and the highest defined in the airfoils
        else:    
            idx_exact = np.where(np.abs(thicknesses - target_thickness) < tol)[0]
            ### thickness is enough close to one of the defined airfoils
            if len(idx_exact) > 0:
                a = self.airfoils[idx_exact[0]]
                if target_lift_coefficient > np.max(a.cl_values) and extrapolate is True:
                    from scipy.interpolate import interp1d
                    cl_cd_vs_cl = interp1d(a.cl_values, a.cl_cd_values, fill_value="extrapolate")
                    cl_cd = cl_cd_vs_cl(target_lift_coefficient)
                else:
                    cl_cd = np.interp(target_lift_coefficient, a.cl_values, a.cl_cd_values)   

                if cl_cd < 0:
                    log(f"Obtained lift to drag coefficient is below 0, the final values has been clamped at the cylinder drag coefficient {self.cylinder_cd} to avoid unphysical results","WARN",color = True)
                    cd = self.cylinder_cd
                else:
                    cd = target_lift_coefficient / cl_cd

                if verbose:
                    log(f"Target thickness {target_thickness} is in the tolerance distance {tol} from airfoil {a.name}", "WARN", color = True)
                    log(f"Interpolated Cd for target thickness {target_thickness} and lift coefficient {target_lift_coefficient} is {cd:.3f}", color = True)
                    if target_lift_coefficient > np.max(a.cl_values):
                        if extrapolate:
                            log(f"The target lift coefficient {target_lift_coefficient:3f} is above the defined data in {a.name}, values have been extrapolated","WARN", color = True)
                        else:     
                            log(f"The target lift coefficient {target_lift_coefficient:3f} is above the defined data in {a.name}, the boundary value has been used","WARN", color = True)

                if plot_interpolation:

                    fig, ax = plt.subplots(figsize=(6,4))
                    
                    thesis_plot(ax,
                            x = a.cl_values,
                            y = a.cl_cd_values,
                            label = a.name,
                            xlabel = r"$c_l$ " + '[-]',
                            ylabel = r"$c_l/c_d$ " + '[-]')

                    thesis_plot(ax,
                        x = target_lift_coefficient,
                        y = cl_cd,
                        style = "point",
                        label = "interpolated value")

                    plt.show()
                    save_figure(fig,
                            name = 'drag_interp2')
            ### thickness is between two defined airfoils 
            else:
                idx_upper = np.searchsorted(thicknesses, target_thickness)
                idx_lower = idx_upper - 1

                idx_list = [idx_lower, idx_upper]
                tc_list = []
                cl_cd_list = []

                for i in idx_list:
                    a = self.airfoils[i]
                    if target_lift_coefficient > np.max(a.cl_values) and extrapolate is True:
                        from scipy.interpolate import interp1d
                        cl_cd_vs_cl = interp1d(a.cl_values, a.cl_cd_values, fill_value="extrapolate")
                        cl_cd = cl_cd_vs_cl(target_lift_coefficient)
                    else:
                        cl_cd = np.interp(target_lift_coefficient, a.cl_values, a.cl_cd_values)       
                    cl_cd_list.append(cl_cd)
                    tc_list.append(a.relative_thickness)

                cl_cd_new = np.interp(target_thickness, tc_list, cl_cd_list)
                
                if cl_cd_new < 0:
                    log(f"Obtained lift to drag coefficient is below 0, the final values has been clamped at the cylinder drag coefficient {self.cylinder_cd} to avoid unphysical results","WARN",color = True)
                    cd = self.cylinder_cd
                else:
                    cd = target_lift_coefficient / cl_cd_new

                if verbose:
                    log(f"Target thickness {target_thickness} is between airfoils {[self.airfoils[i].name for i in idx_list]}", "WARN", color = True)
                    log(f"Interpolated Cd for target thickness {target_thickness} and lift coefficient {target_lift_coefficient} is {cd:.3f}", color = True)
                    for i in idx_list:
                        a = self.airfoils[i]
                        if target_lift_coefficient > np.max(a.cl_values):
                            if extrapolate:
                                log(f"The target lift coefficient {target_lift_coefficient:3f} is above the defined data in {a.name}, values have been extrapolated","WARN", color = True)
                            else:     
                                log(f"The target lift coefficient {target_lift_coefficient:3f} is above the defined data in {a.name}, the boundary value has been used","WARN", color = True)


                if plot_interpolation:

                    fig, ax = plt.subplots(figsize=(9,6))
                    for i in idx_list:
                        a = self.airfoils[i]
                        thesis_plot(ax,
                            x = a.cl_values,
                            y = a.cl_cd_values,
                            label = a.name,
                            xlabel = r"$c_l$ " + '[-]',
                            ylabel = r"$c_l/c_d$ " + '[-]')

                    thesis_plot(ax,
                        x = target_lift_coefficient,
                        y = cl_cd_list[0],
                        style = "point",
                        label = 'interpolated value on ' + self.airfoils[idx_list[0]].name)
                    
                    thesis_plot(ax,
                        x = target_lift_coefficient,
                        y = cl_cd_list[1],
                        style = "point",
                        label = 'interpolated value on ' + self.airfoils[idx_list[1]].name)
                    
                    thesis_plot(ax,
                        x = target_lift_coefficient,
                        y = cl_cd_new,
                        style = "point",
                        label = 'interpolated value')
                    
                    plt.show()
                    save_figure(fig,
                            name = 'drag_interp3')

        if cd>self.cylinder_cd:
            cd = self.cylinder_cd
            log(f"Obtained drag coefficient is above the cylinder one {self.cylinder_cd}, the final values has been clamped at the cylinder value to avoid unphysical results","WARN",color = True)
        if cd<0:
            cd = 0
            log(f"Obtained drag coefficient is below 0, the final values has been clamped at the 0 to avoid unphysical results","WARN",color = True)
        return cd
        
        
def viscous_drag(lift_magnitudes:Mesh2D,
                 cl_per_section:np.array,
                 cd_per_section:np.array,
                 drag_directions:VectorSet,
                 averaged_drag_chordwise:bool = False,
                 dynamic_pressure_times_area:np.array = None,
                 panel_widths:np.array = None
                 ):
    """
    Compute viscous drag vectors.

    Parameters
    ----------
    lift_magnitudes : Mesh2D
        Lift magnitudes in spanwise and chordwise direction
    cl_per_section : np.array
        Lift coefficient per spanwise section
    cd_per_section : np.array
        Drag coefficient per spanwise section
    drag_directions : VectorSet
        Drag direction vectors
    averaged_drag_chordwise : bool
        If True, drag is computed from Cd * q * A and distributed
        equally across chordwise filaments. If False, drag is scaled
        per panel from lift using Cd/Cl ratio.
        Preferred to use for high pitch angles.
    dynamic_pressure_times_area : np.ndarray (N,), optional
        0.5 * rho * V^2 * chord * dy, required when averaged_drag_chordwise=True,
        by default None
    panel_widths : np.array, optional
        Width of each spanwise section [m], required when
        averaged_drag_chordwise=True, by default None

    Returns
    -------
    VectorSet
        Vectors of drag forces
    """

    if averaged_drag_chordwise:
        chordwise_size = drag_directions.X.shape[0]
        drag_magnitudes = (panel_widths * cd_per_section * dynamic_pressure_times_area) / chordwise_size  # (N,)
    else:
        cd_vs_cl_per_section = cd_per_section / cl_per_section  # (N,)
        drag_magnitudes = lift_magnitudes * cd_vs_cl_per_section[np.newaxis, :]  # (M, N)

    drag_forces = VectorSet(x_set=drag_directions.X * drag_magnitudes,
                            y_set=drag_directions.Y * drag_magnitudes,
                            z_set=drag_directions.Z * drag_magnitudes)

    return drag_forces