import numpy as np
import pandas as pd

from import_auxiliary import array_persev_computation
from integrate_equations import integrate_model


# global variables
n_comp = 14                                                                                 # number of compartments
age_groups = ['0-4', '5-17', '18-29', '30-39', '40-49', '50-59', '60+']                     # list of strings with age groups 
age_groups_bins = [0, 5, 18, 30, 40, 50, 60, np.inf]                                        # list of int with age groups extremes 
n_age = len(age_groups)                                                                     # number of age groups
n_sev = 5                                                                                   # number of perceived severity groups

def matrix_distance(deaths_dict, peak_high_dict, peak_date_dict, function_types, contribution = False):

    """
    This function returns the normalised canberra distance for each pair of functions for each of the three metrics
        :param deaths_dict (dict): dictionary with the five functions as keys and 900 elements-list of number of deaths as values
        :param peak_high_dict (dict): dictionary with the five functions as keys and 900 elements-list of ICU peak height as values
        :param peak_date_dict (dict): dictionary with the five functions as keys and 900 elements-list of ICU peak date as values
        :param function_types (list): list of strings with the name of the five functions
        :param contribution (bool): if False the function returns normalised Canberra distance, if False the maximum contribution 
        :return: returns three dataframe (one for each metric) each one containing the normalised canberra distance for each pair of functions
    """
    # generate an empty array with dimensions (number of functions, number of functions, number of metrics)
    matrix = np.zeros((5,5,3))

    # fill each element of the empty array with the normalised canberra distance (or max contribution) obtained for each pair of functions and each metric
    for i, fun_type in zip(range(5), function_types):
        for j, fun_type2 in zip(range(5), function_types):
            for k, metric_dict in zip(range(3), [deaths_dict, peak_high_dict, peak_date_dict]):

                if contribution: 
                    matrix[i,j,k] = round(max([abs(i-j)/(i+j) for i, j in zip(metric_dict[fun_type], metric_dict[fun_type2])]),2)
                else:
                    matrix[i,j,k] = round(np.sum([abs(i-j)/(i+j) for i, j in zip(metric_dict[fun_type], metric_dict[fun_type2])])/len(metric_dict[fun_type]),4)

    # store the data of the array in three different dataframe (one for each metric) with appropriate indexes and return them
    return pd.DataFrame(matrix[:,:,0], index = function_types, columns = function_types), pd.DataFrame(matrix[:,:,1], index = function_types, columns = function_types), pd.DataFrame(matrix[:,:,2], index = function_types, columns = function_types)


def metrics_computation(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, alpha, gamma, a0, b0, t_step, maxsigma, grow, function_types):

    """
    This function performs simulations for a 30x30 grid of values of mean value of the midpoint and variance returning the values used and the metrics.
        :param country_dict (dict): dictionary with contact matrices and demographic data
        :param i0 (float): initial fraction of people in the infected compartments: L, P, I, A
        :param icu0 (float): initial fraction of people in the ICU compartment
        :param r0 (float): initial fraction of people in the recovered compartment
        :param d0 (float): initial fraction of people in the death compartment
        :param initial_date (datetime): starting date of the simulation
        :param t_max (float): max time step
        :param tV0 (float): time at which the vaccination campaign starts
        :param beta (float): attack rate
        :param epsilon (float): inverse of latent period (together with omega gives the incubation rate)
        :param omega (float): inverse of pre-symptomatic period (together with epsilon gives the incubation rate)
        :param mu (float): recovery rate
        :param delta (int): mean number of days of ICU bed occupancy
        :param chi (float): reduction in transmission for asymptomatic and pre-symptomatic individuals
        :param f (array): age-stratified fraction of asymptomatics
        :param IFR (array): age-stratified infection fatality rate
        :param IICUR (array): age-stratified infection ICU rate
        :param PICUD (array): age-stratified probability of dying if in ICU
        :param ICU_max (int): max number of ICU beds
        :param r_v (float): vaccination rate
        :param type_vaccination (string): vaccination strategy, either homogeneous or in reverse order of age
        :param VE_S (float): vaccine efficacy (on susceptibility)
        :param VE_Symp (float): vaccine efficacy (on symptomaticity)
        :param VE_D (float): vaccine efficacy (on severe outcomes)
        :param alpha (array): slope logisitic rate for transition S -> S_NC for each perceived severity group
        :param gamma (array): slope logisitic rate for transition S_NC -> S for each perceived severity group
        :param a0 (array): either 30 elements-array (for each mean value to explore) or a 5 elements-array (for each perceived severity group) of the midpoint of logisitic rate for transition S -> S_NC
        :param b0 (array): either 30 elements-array (for each mean value to explore) or a 5 elements-array (for each perceived severity group) of the midpoint of logisitic rate for transition S_NC -> S
        :param t_step (float): time step
        :param max_sigma (float): maximum value of the variance used
        :param grow (bool): boolean value that declares if we are performing the simulation for a0 (True) or b0 (False)
        :param function_types (list): list of strings with the name of the five functions
        :return: returns six dictionaries, with the five functions as keys and as values the 900 elements-list of, respectively,
                 mean value of the midpoint, variance of the midpoint, number of cases, number of deaths, ICU peak height and ICU peak date
    """

    # initiate the dictionary for each variable we are going to store
    var_dict = {}
    sigma_var_dict = {}
    cases_dict = {}
    deaths_dict = {}
    peak_high_dict = {}
    peak_date_dict = {}

    # iterate over the functions which are going to be keys of the dictionary 
    for fun_type in function_types:

        # intiate a list for each variable we are going to store
        var_plot = []
        sigma_var_plot = []
        cases_plot = []
        deaths_plot = []
        peak_high_plot = []
        peak_date_plot = []

        # which transition we are exploring and over which midpoint we are going to iterate
        if grow:
            vars = a0
        else:
            vars = b0

        counter_1 = 0

        # for each mean value of the midpoint 
        for var in vars:

            # generate a 30 elements-array of values of variance over which iterate
            sigma_vars = np.linspace(0, maxsigma, num=30)

            # for each variance 
            for sigma_var in sigma_vars:

                # compute the value of the midpoint for each perceived severity group
                var_vec = array_persev_computation(country_dict['Nij'], var, sigma_var, fun_type, grow)

                # run the model based on which transition we are exploring
                if grow:
                    y, dates = integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, alpha, gamma, var_vec, b0, t_step)
                else:
                    y, dates = integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, alpha, gamma, a0, var_vec, t_step)
                
                # compute the total final number of cases and deaths, summing over age groups and perceived severity groups
                vect_1 = 0
                vect_2 = 0
                for age in range(n_age):
                    for sev in range(n_sev):
                        vect_1 += y[age,sev,12,-1] + y[age,sev,13,-1]
                        vect_2 += y[age,sev,13,-1]

                # compute the fraction of ICU occupancy day by day
                ICU_every_day = np.sum(y[:,:,11,:], axis = (0,1))/ICU_max

                # append mean value of midpoint, variance of the midpoint, number of cases, number of deaths, maximum of ICU occupacy and its date to respective lists
                var_plot.append(var)
                sigma_var_plot.append(sigma_var)
                cases_plot.append(vect_1)
                deaths_plot.append(vect_2)
                peak_high_plot.append(max(ICU_every_day))
                peak_date_plot.append(np.argmax(ICU_every_day) + 1)
                
            counter_1 += 1

        # append each list to the respective dictionary
        var_dict[fun_type] = var_plot
        sigma_var_dict[fun_type] = sigma_var_plot
        cases_dict[fun_type] = cases_plot
        deaths_dict[fun_type] = deaths_plot
        peak_high_dict[fun_type] = peak_high_plot
        peak_date_dict[fun_type] = peak_date_plot

    return var_dict, sigma_var_dict, cases_dict, deaths_dict, peak_high_dict, peak_date_dict