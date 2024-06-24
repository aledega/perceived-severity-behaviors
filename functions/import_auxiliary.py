import numpy as np
import pandas as pd
import json

# global variables
n_comp = 14                                                                                 # number of compartments
age_groups = ['0-4', '5-17', '18-29', '30-39', '40-49', '50-59', '60+']                     # list of strings with age groups 
age_groups_bins = [0, 5, 18, 30, 40, 50, 60, np.inf]                                        # list of int with age groups extremes 
n_age = len(age_groups)                                                                     # number of age groups
n_sev = 5                                                                                   # number of perceived severity groups




def import_data(path_CM_red, path_CM_yellow, path_Nij):

    """
    This function returns a dictionary with all data needed to run the model
        :param path_CM_red (string): path to file json of compliant contact matrix
        :param path_CM_yellow (string): path to file json of non-compliant contact matrix
        :param path_Nij (string): path to file with distribution of population
        :return dict of data (contact matrices, Nij)
    """

    # Load the compliant matrix as a numoy array
    with open(path_CM_red, 'r') as f:
        CM_red = np.array(json.load(f))

    # Load the non compliant matrix as a numoy array
    with open(path_CM_yellow, 'r') as f:
        CM_yellow = np.array(json.load(f))

    #Load the number of individuals in each age and perceived severity group as a numpy array
    with open(path_Nij, 'r') as f:
        Nij = np.array(json.load(f))

    # create dict of data 
    country_dict = {'CM_red'    : CM_red,
                    'CM_yellow' : CM_yellow,
                    'Nij'       : Nij}

    return country_dict


def get_R0(age_group, beta, omega, mu, chi, f, CMC, CMNC, Nij, alpha, gamma, a0, b0, ICU_0):

    """
    This function returns R0 for our SEIR model with age and percevied severity structure
        :param age_group (int): number of age group for which R0 is computed
        :param beta (float): attack rate
        :param omega (float): inverse of pre-symptomatic period (together with epsilon gives the incubation rate)
        :param mu (float): recovery rate
        :param chi (float): reduction in transmission for asymptomatic and pre-symptomatic individuals
        :param f (array): age-stratified fraction of asymptomatics
        :param CMC (matrix): contacts matrix for compliant people
        :param CMNC (matrix): contacts matrix for non compliant people
        :param Nij (array): n. of individuals in different age groups and perceived severity groups
        :param alpha (array): slope for transition S -> S_NC (SV -> SV_NC)
        :param gamma (array): slope for transition S_NC -> S (SV_NC -> SV)
        :param a_0 (array): midpoint for transition S -> S_NC (SV -> SV_NC)
        :param b_0 (array): midpoint for transition S_NC -> S (SV_NC -> SV)
        :param ICU_0 (float): intial filling fraction of ICU (init number ICU / ICU_max)
        :return: returns the reproduction number R0 for age group age_group
    """

    # create empty matrix with the size of the two matrices used
    CM_tilde = np.zeros((CMC.shape[0], CMC.shape[1]))

    # assign to each element a weighted sum of the the corresponding elements of the two matrices
    for age in range(n_age) :
        for age2 in range(n_age) :
            for sev in range(n_sev):

                #compute the ratio between compliant and non compliant individuals at time 0
                ratio = (1 + np.exp(alpha[sev] * a0[sev])) / (1 + np.exp(- gamma[sev] * (ICU_0 - b0[sev])))

                for sev2 in range(n_sev):
                    # CM~_iji'j' = (N_ij / N_i'j') * [(ratio/(1+ratio)) * CMC_iji'j'+ (1/(1+ratio)) * CMNC_iji'j']
                    CM_tilde[n_sev*age+sev, n_sev*age2+sev2] = (Nij[age,sev]/Nij[age2,sev2]) * ((ratio/(1 + ratio)) * CMC[n_sev*age+sev, n_sev*age2+sev2] + (1/(1 + ratio)) * CMNC[n_sev*age+sev, n_sev*age2+sev2])

    #highest eigenvalue of the matrix times a combination of epi parameters gives R0
    return beta * ((chi/omega) + ((1-f[age_group])/mu) + (chi*f[age_group]/mu)) * (np.max([eigen.real for eigen in np.linalg.eig(CM_tilde)[0]]))




def initialize_model(Nij, i0, icu0, r0, d0, epsilon, omega, mu, f, alpha, gamma, a0, b0, ICU_max):

    """
    This function returns the initial conditions of our model
        :param Nij (float): n. of individuals in different age and perceived severity groups
        :param i0 (float): initial fraction of people in the infected compartments: L, P, I, A
        :param icu0 (float): initial fraction of people in the ICU compartment
        :param r0 (float): initial fraction of people in the recovered compartment
        :param d0 (float): initial fraction of people in the death compartment
        :param epsilon (float): inverse of latent period (together with omega gives the incubation rate)
        :param omega (float): inverse of pre-symptomatic period (together with epsilon gives the incubation rate)
        :param mu (float): recovery rate
        :param f (array): age-stratified fraction of asymptomatics
        :param alpha (array): slope for transition S -> S_NC (SV -> SV_NC)
        :param gamma (array): slope for transition S_NC -> S (SV_NC -> SV)
        :param a_0 (array): midpoint for transition S -> S_NC (SV -> SV_NC)
        :param b_0 (array): midpoint for transition S_NC -> S (SV_NC -> SV)
        :param ICU_max (int): max number of ICU beds
        :return: returns the vector of initial conditions of the model
    """

    # initialise array of initial conditions
    y0 = np.zeros((n_age, n_sev, n_comp))


    for age in range(n_age):
        for sev in range(n_sev):

            # place infected individuals in compartment L,P,I and A based on the time of permanence
            y0[age, sev, 4] = epsilon**(-1) / (mu**(-1) + omega**(-1) + epsilon**(-1)) * i0 * Nij[age, sev]
            y0[age, sev, 5] = omega**(-1) / (mu**(-1) + omega**(-1) + epsilon**(-1)) * i0 * Nij[age, sev]
            y0[age, sev, 6] = (1 - f[age]) * mu**(-1) / (mu**(-1) + omega**(-1) + epsilon**(-1)) * i0 * Nij[age, sev]
            y0[age, sev, 10] = f[age] * mu**(-1) / (mu**(-1) + omega**(-1) + epsilon**(-1)) * i0 * Nij[age, sev]

            # place individuals in ICU, R and D compartments
            y0[age, sev, 11] = icu0 * Nij[age, sev]
            y0[age, sev, 12] = r0 * Nij[age, sev]
            y0[age, sev, 13] = d0 * Nij[age, sev]

            # place the remaing individuals in susceptibles compliant and non compliant compartments
            ratio = ((1 + np.exp(alpha[sev] * a0[sev])) / (1 + np.exp(-gamma[sev] * ((icu0 * np.sum(Nij) / ICU_max) - b0[sev]))))
            y0[age,sev,0] = (Nij[age,sev] - y0[age,sev,4] - y0[age,sev,5] - y0[age,sev,6] - y0[age,sev,10] - y0[age,sev,11] - y0[age,sev,12] - y0[age,sev,13]) * (ratio / (1 + ratio))
            y0[age,sev,1] = (Nij[age,sev] - y0[age,sev,4] - y0[age,sev,5] - y0[age,sev,6] - y0[age,sev,10] - y0[age,sev,11] - y0[age,sev,12] - y0[age,sev,13]) * (1 / (1 + ratio))


    #check on the intial conditions
    if np.any(y0[:,:,:] < 0):
        print(y0[:,:,:])
        return 'problem with the initial conditions'

    return y0




def coef_computation(Nij, var, fun_type, grow = True):

    """
    This function returns the slope of the linear part of the function we use to model the dependency between parameters and perceived severity
        :param Nij (float): n. of individuals in different age and perceived severity groups
        :param var (float): variance for our parameter
        :param fun_type (string): type of function to model the dependency between a0 (b0) and percevied severity
        :param grow (bool): boolean value that declares if we want a positive coefficient (a0) or negative coefficient (b0)
        :return: returns the slope of the linear part of the function
    """

    # number of individuals in each perceived severity group
    N_per = Nij.sum(axis=0)

    # define some useful variables to shorten the notation afterwards
    N = Nij.sum()
    M = 0
    D = 0
    for i in range(len(N_per)):
        M += N_per[i] * i
        D += N_per[i] * i**2

    # compute the slope for each function type
    if fun_type == 'lin':
        if grow:
            return np.sqrt((N**2 * var)/(N*D-M**2))
        else:
            return - np.sqrt((N**2 * var)/(N*D-M**2))
    elif fun_type == 'centerlin':
        if grow:
            return np.sqrt((N**2 * var)/((D + N_per[0] - 7 * N_per[4]) * N - (M + N_per[0] - N_per[4])**2))
        else:
            return - np.sqrt((N**2 * var)/((D + N_per[0] - 7 * N_per[4]) * N - (M + N_per[0] - N_per[4])**2))
    elif fun_type == 'startendlin':
        if grow:
            return np.sqrt((N**2 * var)/((N - N_per[0] + 3 * N_per[4]) * N - (N - N_per[0] + N_per[4])**2))
        else:
            return - np.sqrt((N**2 * var)/((N - N_per[0] + 3 * N_per[4]) * N - (N - N_per[0] + N_per[4])**2))
    elif fun_type == 'startlin':
        if grow:
            return np.sqrt((N**2 * var)/((D - 5 * N_per[3] - 12 * N_per[4]) * N - (M - N_per[3] - 2 * N_per[4])**2))
        else:
            return - np.sqrt((N**2 * var)/((D - 5 * N_per[3] - 12 * N_per[4]) * N - (M - N_per[3] - 2 * N_per[4])**2))
    elif fun_type == 'endlin':
        if grow:
            return np.sqrt((N**2 * var)/((D + 4 * N_per[0] + 3 * N_per[1]) * N - (M + 2 * N_per[0] + N_per[1])**2))
        else:
            return - np.sqrt((N**2 * var)/((D + 4 * N_per[0] + 3 * N_per[1]) * N - (M + 2 * N_per[0] + N_per[1])**2))
    else:
        return 'ERROR'




def array_persev_computation(Nij, media, var, fun_type, grow = True):

    """
    This function returns the array with the values of the midpoint for each percevied severity group
        :param Nij (float): n. of individuals in different age and perceived severity groups
        :param media (float): mean value for our parameter
        :param var (float): variance for our parameter
        :param fun_type (string): type of function to model the dependency between a0 (b0) and percevied severity
        :param grow (bool): boolean value that declares if we want a positive coefficient (a0) or negative coefficient (b0)
        :return: returns the array of a0 (b0) with a value for each percevied severity group
    """

    # number of individuals in each perceived severity group
    N_per = Nij.sum(axis=0)

    # define some useful variables to shorten the notation afterwards
    N = Nij.sum()
    M = 0
    for i in range(len(N_per)):
        M += N_per[i] * i

    # comopute the slope of the linear of the function given the variance chosen
    m = coef_computation(Nij, var, fun_type, grow)

    # initialise list of values for perceived severity group
    vect = [0,0,0,0,0]

    # assign midpoint to each percevied severity group based on type of function
    if fun_type == 'lin':
        for i in range(len(N_per)):
            vect[i] = media + m*(i - M/N)
    elif fun_type == 'centerlin':
        for i in [1,2,3]:
            vect[i] = media + m*(i - (M + N_per[0] - N_per[-1])/N)
        vect[0] = vect[1]
        vect[4] = vect[3]
    elif fun_type == 'startendlin':
        for i in [0,1]:
            vect[i] = media + m*(i - (N - N_per[0] + N_per[-1])/N)
        vect[2] = vect[1]
        vect[3] = vect[2]
        vect[4] = media + m*(2 - (N - N_per[0] + N_per[-1])/N)
    elif fun_type == 'startlin':
        for i in [0,1,2]:
            vect[i] = media + m*(i - (M - N_per[3] - 2*N_per[4])/N)
        vect[3] = vect[2]
        vect[4] = vect[3]
    elif fun_type == 'endlin':
        for i in [2,3,4]:
            vect[i] = media + m*(i - (M + 2*N_per[0] + N_per[1])/N)
        vect[1] = vect[2]
        vect[0] = vect[1]
    else:
        return 'ERROR'
    return vect
