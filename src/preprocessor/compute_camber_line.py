import numpy as np
import pandas as pd
from utilities import log
from dataclasses import dataclass
from scipy.interpolate import interp1d

@dataclass
class Airfoil:
    """
    Camber line geometry and metadata for a single airfoil section.

    Attributes
    ----------
    name : str
        Airfoil identifier, matches the source .dat filename (without extension)
    relative_thickness : float
        Relative thickness of the airfoil [%]
    x_coords : np.ndarray
        Camber line x coordinates (chordwise), shape (n_points, 1)
    z_coords : np.ndarray
        Camber line z coordinates (perpendicular to chord), shape (n_points, 1)
    """
    name: str
    relative_thickness: float
    x_coords: np.ndarray
    z_coords: np.ndarray


class CamberLineInterpolator:
    """
    Interpolates airfoil camber line geometry for arbitrary relative
    thicknesses, based on a discrete set of input airfoil profiles.

    Used to generate the camber surface of a lifting surface by blending
    between the closest defined airfoil sections.
    """
    def __init__(self,
                 airfoil_types: pd.Series):
        """
        Initialize the camber line interpolator

        Parameters
        ----------
        airfoil_types : pd.Series
            airfoil names related to the .dat file names
        """

        log("Camber line computation...", "STEP", shift_number = 1)
        self.airfoils: list[Airfoil] = [] 

        for name in airfoil_types:
            thickness = float(name[-3:]) / 10
            x,z = self.convert_airfoil_txt_to_camberline(name)
            self.airfoils.append(Airfoil(name, thickness, x, z))

        # adding cylinder shape, it's unnecessary when all of the considered sections are located between defined airfoils
        self.airfoils.append(Airfoil(name = 'cylinder',
                                     relative_thickness = 100,
                                     x_coords = self.airfoils[-1].x_coords,
                                     z_coords = np.zeros_like(self.airfoils[-1].x_coords)))


    @staticmethod
    def convert_airfoil_txt_to_camberline(airfoil_name:str) -> np.ndarray:
        """Convert the airfoil data file into camber line geometry

        Parameters
        ----------
        airfoil_name : str
            Name of the airfoil which should be represented in 
            the related data file in format: airfoil_name.dat

        Returns
        -------
        np.ndarray
            x: camberline geometry in chordwise direction
        np.ndarray
            z: camberline geometry in direction perpendiculat to the chord
            
        """
        airfoil = np.loadtxt('blade_input/' + airfoil_name + ".dat")
        sorted_indices = np.argsort(airfoil[:,0])
        airfoil = airfoil[sorted_indices]
        first_row = airfoil[0,:]

        even_rows = airfoil[1::2,:]
        odd_rows = airfoil[2::2,:]

        mean = (even_rows + odd_rows) / 2            
        res = np.vstack((first_row, mean))

        x = res[:,0].reshape(-1,1)
        z = res[:,1].reshape(-1,1)
        
        log(f"Camber line for the airfoil {airfoil_name} has been calculated",
            "INFO",
            color= True,
            shift_number = 1)

        return x, z


    def interpolate_camber_line(self,
                                target_thickness:float,
                                tol: float = 1e-3):
        """
        Interpolates the camberline based on the provided target relative thickess

        Parameters
        ----------
        target_thickness : float
            Relative thickness of the considered section
        tol : float, optional
            Tolerance with wich the relative thickness search the closest one defined in the 
            airfoil dataset, 
            by default 1e-3

        Returns
        -------
        np.ndarray
            Interpolated x coordinates of the camber line 
        np.ndarray
            Interpolated z coordinates of the camber line 
        str
            Name of the first airfoil used for the interpolation 
        str
            Name of the second airfoil used for the interpolation                  
        """

        # get the thickness from the provided airfoil set
        thicknesses = np.array([a.relative_thickness for a in self.airfoils])
        # find the closest thickness to the defined in the airfoil set
        idx_exact = np.where(np.abs(thicknesses - target_thickness) < tol)[0]

        # target relative thickness is within the defined tolerance to the defined airfoil dataset
        if len(idx_exact) > 0:
            a = self.airfoils[idx_exact[0]]
            return a.x_coords, a.z_coords, a.name, a.name

        # target relative thickness is between the two defined in the airfoil dataset
        else:
            idx_upper = np.searchsorted(thicknesses, target_thickness)
            idx_lower = idx_upper - 1

            a_low = self.airfoils[idx_lower]
            a_high = self.airfoils[idx_upper]
            
            x_matrix = np.hstack([a_low.x_coords, a_high.x_coords])
            z_matrix = np.hstack([a_low.z_coords, a_high.z_coords])
            
            interp_x = interp1d([a_low.relative_thickness, a_high.relative_thickness], x_matrix, axis=1)
            interp_z = interp1d([a_low.relative_thickness, a_high.relative_thickness], z_matrix, axis=1)
            
            return interp_x(target_thickness).reshape(-1, 1), interp_z(target_thickness).reshape(-1, 1), a_low.name, a_high.name
        

    def compute_camber_lines(self,
                             relative_thicknesses:pd.Series,
                             cut_off_thickness: float = 100,
                             show_details = False) -> np.ndarray:
        """
        Computes the camberline surface geometry

        Parameters
        ----------
        relative_thicknesses : pd.Series
            Target thicknesses of the interpolation [%]
        cut_off_thickness : float, optional
            Thickest thickness representing the upper modelling treshold [%]
        show_details : bool, optional
            Flag that allows to print the interpolation detail logs, by default False

        Returns
        -------
        np.ndarray
            The x coordinates of the camberline surface [points from one section, number of sections]
    
        np.ndarray
            The z coordinates of the camberline surface [points from one section, number of sections]
        """
        
        log(f' The cutoff thickness set up as {cut_off_thickness}',"WARN",color=True, shift_number = 1)

        # length of the points defining the camber line curve
        M = self.airfoils[0].x_coords.shape[0]
        # length of the considered spanwise secction
        N = len(relative_thicknesses)

        # initiating the camberline geometry
        camber_lines_x = np.zeros((M,N))
        camber_lines_z = np.zeros((M,N))

        # find the thickness of the thinnest defined airfoil
        min_airfoil_thickness = min(a.relative_thickness for a in self.airfoils)

        valid_cols = []
        non_valid_cols = []

        for section, relative_thickness_value in enumerate(relative_thicknesses):

            # sections with a relative thickness above the defined cut off thickness are omitted in the surface definition
            if relative_thickness_value > cut_off_thickness:
                if show_details:
                    log(f"The section number {section} with a relative thickness of {relative_thickness_value:.3f} has been skipped in modelling.",
                        "INFO", shift_number = 1)
                non_valid_cols.append(section)

            # sections with the thickness between the cut off thickness and the thinnest defined airfoil are used in the surface definition
            elif relative_thickness_value >= min_airfoil_thickness:
                camber_line_x, camber_line_z, low_name, high_name = self.interpolate_camber_line(
                    target_thickness = relative_thickness_value)   
                camber_lines_x[:, section], camber_lines_z[:, section] = camber_line_x.ravel(), camber_line_z.ravel()
                valid_cols.append(section)
                
                if show_details:
                    if high_name == low_name:
                        log(f"The section number {section} matches with thickness {relative_thickness_value:.3f} -> {high_name}", "INFO", shift_number = 1)
                    else:
                        log(f"The section number {section} with a relative thickness of {relative_thickness_value:.3f} has been interpolated based on airfoils {low_name} and {high_name}.",
                        "INFO", shift_number = 1)

            # sections with a relative thickness below the thinnest defined airfoil cannot be use in the surface definition
            else:
                raise ValueError(f'The input section number {section} has a relative thickness {relative_thickness_value} below the minimum, impossible to model.')    

        camber_lines_x = camber_lines_x[:, valid_cols]
        camber_lines_z = camber_lines_z[:, valid_cols]

        log(f" Sections: {non_valid_cols} have been skipped in modelling", "WARN", color=True, shift_number = 1)

        return camber_lines_x, camber_lines_z