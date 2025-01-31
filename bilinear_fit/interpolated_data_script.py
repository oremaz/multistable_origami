# This script contains the code to interpolate a dataset with a bilinear model.
# Copyright (C) 2025 Orel Mazor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from .analysis import *


def bilinear_fit(
    angle: Union[float, int],
    k_min: Union[float, int],
    k_max: Union[float, int],
    break_angle: Union[float, int],
    break_torque: Union[float, int],
):
    """
    Calcule une fonction bilinéaire où deux pentes (k_min et k_max) se croisent à (break_angle, break_torque).

    Paramètres:
    angle : Angle pour lequel calculer le couple (torque)
    k_min : Pente avant le point de rupture (basse raideur)
    k_max : Pente après le point de rupture (haute raideur)
    break_angle : L'angle de rupture où les deux pentes se croisent
    break_torque : Le couple à l'angle de rupture

    Retourne:
    Le couple correspondant à l'angle donné selon le modèle bilinéaire.
    """

    # Avant l'angle de rupture, on utilise k_min
    torque_before = k_min * (angle - break_angle) + break_torque

    # Après l'angle de rupture, on utilise k_max
    torque_after = k_max * (angle - break_angle) + break_torque

    # Utiliser np.where pour basculer entre les deux pentes
    torque = np.where(angle <= break_angle, torque_before, torque_after)

    return torque


def interpolated_fit_general(
    dL: Union[List[Union[float, int]], np.ndarray],
    offset: Union[List[Union[float, int]], np.ndarray],
    width: Union[List[Union[float, int]], np.ndarray],
    thickness: Union[List[Union[float, int]], np.ndarray],
    angles=False,
    method="Optimisation",
):
    thickness_ref = 0.127
    offset_ref = 0
    width_ref = 8
    df_ref_dL = analysis_2D(
        dL=np.round(np.arange(0.43, 1.005, 0.005), decimals=3),
        thickness=[thickness_ref],
        offset=[offset_ref],
        width=[width_ref],
        method=method,
    )
    k_min = fit_2D(df_ref_dL, "k_min", model=True)
    k_max = fit_2D(df_ref_dL, "k_max", model=True)
    break_angle = fit_2D(
        df_ref_dL, "break_angle", polynomial=True, model=True, degrees=[3]
    )
    break_torque = fit_2D(df_ref_dL, "break_torque", model=True, degrees=[3])
    dico_f = {}
    dico_rest_angle = {}
    dico_break_angle = {}
    for dL_value in dL:
        for thickness_value in thickness:
            for offset_value in offset:
                for width_value in width:
                    k_min_value = (
                        float(k_min.predict(np.array([[dL_value]]))[0])
                        * (thickness_value / thickness_ref) ** 3
                        * (width_value / width_ref)
                    )
                    k_max_value = (
                        float(k_max.predict(np.array([[dL_value]]))[0])
                        * (thickness_value / thickness_ref) ** 3
                        * (width_value / width_ref)
                    )
                    break_angle_value = float(
                        break_angle.predict(np.array([[dL_value]]))[0]
                    )
                    break_torque_value = (
                        float(break_torque.predict(np.array([[dL_value]]))[0])
                        * (thickness_value / thickness_ref) ** 3
                        * (width_value / width_ref)
                    )
                    dico_f[(dL_value, offset_value, width_value, thickness_value)] = (
                        lambda angle: bilinear_fit(
                            angle,
                            k_min_value,
                            k_max_value,
                            break_angle_value,
                            break_torque_value,
                        )
                    )
                    if angles:
                        dico_rest_angle[
                            (dL_value, offset_value, width_value, thickness_value)
                        ] = find_zeros_of_bilinear(
                            bilinear_func=dico_f[
                                (
                                    dL_value,
                                    offset_value,
                                    width_value,
                                    thickness_value,
                                )
                            ]
                        )
                        dico_break_angle[
                            (dL_value, offset_value, width_value, thickness_value)
                        ] = break_angle_value
    if angles:
        return dico_f, dico_rest_angle, dico_break_angle
    else:
        return dico_f


def interpolated_fit(
    dL: Union[float, int],
    offset: Union[float, int],
    width: Union[float, int],
    thickness: Union[float, int],
    angles=False,
    method="Optimisation",
):
    data = interpolated_fit_general(
        [dL], [offset], [width], [thickness], angles=angles, method=method
    )
    if angles:
        return (
            data[0][(dL, offset, width, thickness)],
            data[1][(dL, offset, width, thickness)],
            data[2][(dL, offset, width, thickness)],
        )
    else:
        return data[(dL, offset, width, thickness)]
