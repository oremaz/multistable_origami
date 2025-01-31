# This script contains functions to synthesize the stiffness, rest angle, and snap angle for different multistable origami with varying lenghts or thicknesses.
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


from .interpolated_data_script import *


def dataframe_complete_dL(
    dL_list: Union[List[Union[float, int]], np.ndarray] = np.round(
        np.arange(0.43, 1.005, 0.005), decimals=3
    ),
    fit=False,
    thickness: Union[float, int] = 0.127,
    methods: List[str] = ["Optimisation"],
):
    df = analysis_2D(dL=dL_list, offset=[0], width=[8], thickness=[thickness], fit=fit)
    for method in methods:
        rest_angle = []
        for dL in dL_list:
            df1 = load_df(dL, offset=0, width=8, thickness=0.127)
            rest_angle_fit = find_zeros_of_bilinear(df=df1, method=method, fit=fit)
            rest_angle.append(rest_angle_fit)
        df[f"rest_angle_{method}"] = rest_angle
    _, rest_angle_dico, break_angle_dico = interpolated_fit_general(
        dL=dL_list, thickness=[0.127], offset=[0], width=[8], angles=True
    )
    df["rest_angle_interpolated"] = df.index.map(
        lambda idx: rest_angle_dico.get((idx, 0, 8, 0.127), np.nan)
    )
    df["break_angle_interpolated"] = df.index.map(
        lambda idx: break_angle_dico.get((idx, 0, 8, 0.127), np.nan)
    )
    df["methods"] = [methods] * len(df)
    return df


def dataframe_complete_thickness(
    thickness_list: Union[List[Union[float, int]], np.ndarray] = np.round(
        np.arange(0.03, 0.32, 0.005), decimals=3
    ),
    fit=False,
    dL: Union[float, int] = 0.88,
    methods: List[str] = ["Optimisation"],
):
    df = analysis_2D(thickness=thickness_list, offset=[0], width=[8], dL=[dL], fit=fit)
    for method in methods:
        rest_angle = []
        for thickness in thickness_list:
            df1 = load_df(dL=dL, offset=0, width=8, thickness=thickness)
            rest_angle_fit = find_zeros_of_bilinear(df=df1, method=method, fit=fit)
            rest_angle.append(rest_angle_fit)
        df[f"rest_angle_{method}"] = rest_angle
    _, rest_angle_dico, break_angle_dico = interpolated_fit_general(
        thickness=thickness_list, dL=[dL], offset=[0], width=[8], angles=True
    )
    df["rest_angle_interpolated"] = df.index.map(
        lambda idx: rest_angle_dico.get((0.88, 0, 8, idx), np.nan)
    )
    df["break_angle_interpolated"] = df.index.map(
        lambda idx: break_angle_dico.get((0.88, 0, 8, idx), np.nan)
    )
    df["methods"] = methods
    return df
