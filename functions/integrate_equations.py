from scipy.integrate import odeint
import numpy as np
import pandas as pd
from numba import jit
from datetime import datetime, timedelta

from import_auxiliary import initialize_model

# global variables
n_comp = 14                                                                                 # number of compartments
age_groups = ['0-4', '5-17', '18-29', '30-39', '40-49', '50-59', '60+']                     # list of strings with age groups 
age_groups_bins = [0, 5, 18, 30, 40, 50, 60, np.inf]                                        # list of int with age groups extremes 
n_age = len(age_groups)                                                                     # number of age groups
n_sev = 5                                                                                   # number of perceived severity groups


@jit
def get_vaccinated(y, t, tV0, Nij, r_v, type_vaccination):

    """
    This functions compute the n. of S individuals that will receive a vaccine in the next step
        :param y (array): compartment values at time t
        :param t (float): time of the simulation
        :param tV0 (float): time at which the vaccination campaign starts
        :param Nij (array): number of individuals in different age and perceived severity groups
        :param r_v (float): vaccination rate
        :param type_vaccination (string): homogeneous, age
        :return: returns the two arrays of n. of vaccinated in different age and perceived severity groups for S and S_NC in the next step
    """

    # list of n. of vaccinated in each age and perceived severity group for S and S_NC in this step
    new_V, new_V_NC = np.zeros((n_age, n_sev)), np.zeros((n_age, n_sev))

    # check if the vaccination campaign has started
    if t < tV0 or type_vaccination == None:
        return new_V, new_V_NC

    # tot n. of vaccine available this step
    tot_V = r_v * np.sum(Nij)
    
    # homogeneous vaccination
    if type_vaccination == 'homogeneous':
        
        # tot people that can receive the vaccines
        den = 0
        for age in range(n_age):
            for sev in range(n_sev):
                den += (y[n_comp * ((n_sev * age) + sev) + 0] + y[n_comp * ((n_sev * age) + sev) + 1])

        # all vaccinated
        if den <= 1: 
            return np.zeros((n_age, n_sev)), np.zeros((n_age, n_sev))
        
        # distribute vaccine homogeneously
        for age in range(n_age):
            for sev in range(n_sev):
                new_V[age,sev] = y[n_comp * ((n_sev * age) + sev) + 0] * tot_V / den
                new_V_NC[age,sev] = y[n_comp * ((n_sev * age) + sev) + 1] * tot_V / den

                # check we are not exceeding the tot n. of susceptibles left
                if new_V[age,sev] > y[n_comp * ((n_sev * age) + sev) + 0]:
                    new_V[age,sev] = y[n_comp * ((n_sev * age) + sev) + 0]
                if new_V_NC[age,sev] > y[n_comp * ((n_sev * age) + sev) + 1]:
                    new_V_NC[age,sev] = y[n_comp * ((n_sev * age) + sev) + 1]
        
        return new_V, new_V_NC

    # prioritize elderly in vaccination
    if type_vaccination == 'age':

        left_V = tot_V
        for age in range(n_age-1, -1, -1):
            if left_V < 1:
                return new_V, new_V_NC
            else:
                den_age = 0
                for sev in range(n_sev):
                    den_age += (y[n_comp * ((n_sev * age) + sev) + 0] + y[n_comp * ((n_sev * age) + sev) + 1])
                if den_age < left_V:
                    for sev in range(n_sev):
                        if y[n_comp * ((n_sev * age) + sev) + 0] > 0:
                            new_V[age,sev] = y[n_comp * ((n_sev * age) + sev) + 0]
                            left_V -= new_V[age,sev]
                        if y[n_comp * ((n_sev * age) + sev) + 1] > 0:
                            new_V_NC[age,sev] = y[n_comp * ((n_sev * age) + sev) + 1]
                            left_V -= new_V_NC[age,sev]
                    tot_V = left_V
                else:
                    for sev in range(n_sev):
                        new_V[age,sev] = tot_V * y[n_comp * ((n_sev * age) + sev) + 0] / den_age
                        new_V_NC[age,sev] = tot_V * y[n_comp * ((n_sev * age) + sev) + 1] / den_age
                        left_V -= (new_V[age,sev] + new_V_NC[age,sev])
        return new_V, new_V_NC 
    
@jit
def model(y, t, CM, CM2, Nij, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, VE_S, VE_Symp, VE_D, alpha, gamma, a0, b0, new_V, new_V_NC, v_t, ICU_t):

    """
    This function defines the system of differential equations
        :param y (array): compartment values at time t
        :param t (float): current time step
        :param CM (matrix): contacts matrix for compliant people
        :param CM2 (matrix): contacts matrix for non compliant people
        :param Nij (array): number of individuals in different age and perceived severity groups
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
        :param VE_S (float): vaccine efficacy (on susceptibility)
        :param VE_Symp (float): vaccine efficacy (on symptomaticity)
        :param VE_D (float): vaccine efficacy (on severe outcomes)
        :param alpha (float): slope logisitic rate for transition S -> S_NC
        :param gamma (float): slope logisitic rate for transition S_NC -> S
        :param a0 (float): midpoint logisitic rate for transition S -> S_NC
        :param b0 (float): midpoint logisitic rate for transition S_NC -> S
        :new_V (array): n. of new vaccinated in different age and perceived severity groups for S
        :new_V_NC (array): n. of new vaccinated in different age and perceived severity groups for S_NC
        :param v_t (float): fraction of people already vaccinated
        :param ICU_t (float): fraction of ICU occupancy
        :return: returns the system dydt of differential equations
    Note:
        - age groups must be ordered sequentially (from youngest to oldest)
        - for each age group compartments must be ordered like this:
              S:0, S_NC:1, V:2, V_NC:3, L:4, P:5, I:6, LV:7, PV:8, IV:9, A:10, ICU:11, R:12, D:13
    """


    # logisitic function for transition between compliant and non-compliant compartments
    def rate(x, k, x0):
        return 1 / (1 + np.exp(-k * (x - x0)))

    dydt = []

    # iterate over age group and perceived severity group and define for each one the respective differential equations
    for age1 in range(n_age):

        # compute rate of infected individuals which dies
        lambda_deaths = IFR[age1] - PICUD[age1] * IICUR[age1]
        #lambda_deaths_V = ((IFR[age1] * (1-VE_D)) - (PICUD[age1] * IICUR[age1] * (1-VE_D))) 

        for sev in range(n_sev):
            
            # compute the force of infection
            S_to_L = 0
            SNC_to_L = 0

            for age2 in range(n_age):
                for sev2 in range(n_sev):

                    # \sum_i'j' CM_iji'j' ((I_i'j' + I^V_i'j' + chi (P_i'j' + P^V_i'j' + A_i'j')) / N_i'j')
                    S_to_L += CM[n_sev*age1+sev, n_sev*age2+sev2] * (y[n_comp * ((n_sev * age2) + sev2) + 6] + y[n_comp * ((n_sev * age2) + sev2) + 9] + chi * (y[n_comp * ((n_sev * age2) + sev2) + 5] + y[n_comp * ((n_sev * age2) + sev2) + 8]+ y[n_comp * ((n_sev * age2) + sev2) + 10])) / Nij[age2, sev2]   

                    # \sum_i'j' CM2_iji'j' ((I_i'j' + I^V_i'j' + chi (P_i'j' + P^V_i'j' + A_i'j')) / N_i'j')
                    SNC_to_L += CM2[n_sev*age1+sev, n_sev*age2+sev2] * (y[n_comp * ((n_sev * age2) + sev2) + 6] + y[n_comp * ((n_sev * age2) + sev2) + 9] + chi * (y[n_comp * ((n_sev * age2) + sev2) + 5] + y[n_comp * ((n_sev * age2) + sev2) + 8]+ y[n_comp * ((n_sev * age2) + sev2) + 10])) / Nij[age2, sev2]

            #Susceptibles compartment (S)
            dydt.append(-rate(v_t, alpha[sev], a0[sev]) * y[n_comp * ((n_sev * age1) + sev) + 0]                                # S->SNC
                        -new_V[age1, sev]                                                                                       # S->V
                        -beta * S_to_L * y[n_comp * ((n_sev * age1) + sev) + 0]                                                 # S->L
                        +rate(ICU_t, gamma[sev], b0[sev]) * y[n_comp * ((n_sev * age1) + sev) + 1])                             # SNC->S

            #Susceptible non-compliant compartment (SNC)
            dydt.append(-rate(ICU_t, gamma[sev], b0[sev]) * y[n_comp * ((n_sev * age1) + sev) + 1]                              # SNC->S
                        -new_V_NC[age1, sev]                                                                                    # SNC->VNC
                        -beta * SNC_to_L * y[n_comp * ((n_sev * age1) + sev) + 1]                                               # SNC->L
                        +rate(v_t, alpha[sev], a0[sev]) * y[n_comp * ((n_sev * age1) + sev) + 0])                               # S->SNC

            #Vaccinated compartment (V)
            dydt.append(-rate(v_t, alpha[sev], a0[sev]) * y[n_comp * ((n_sev * age1) + sev) + 2]                                # V->VNC
                        -beta * (1-VE_S) * S_to_L * y[n_comp * ((n_sev * age1) + sev) + 2]                                      # V->LV  
                        +new_V[age1, sev]                                                                                       # S->V 
                        +rate(ICU_t, gamma[sev], b0[sev]) * y[n_comp * ((n_sev * age1) + sev) + 3])                             # VNC->V

            #Vaccinated non-compliant compartment (VNC)
            dydt.append(-rate(ICU_t, gamma[sev], b0[sev]) * y[n_comp * ((n_sev * age1) + sev) + 3]                              # VNC->V
                        -beta * (1-VE_S) * SNC_to_L * y[n_comp * ((n_sev * age1) + sev) + 3]                                    # VNC->LV  
                        +new_V_NC[age1, sev]                                                                                    # SNC->VNC 
                        +rate(v_t, alpha[sev], a0[sev]) * y[n_comp * ((n_sev * age1) + sev) + 2])                               # V->VNC

            #Latent compartment (L)
            dydt.append(-epsilon * y[n_comp * ((n_sev * age1) + sev) + 4]                                                       # L->P
                        +beta * S_to_L * y[n_comp * ((n_sev * age1) + sev) + 0]                                                 # S->L
                        +beta * SNC_to_L * y[n_comp * ((n_sev * age1) + sev) + 1])                                              # SNC->L 

            #Pre-symptomatic compartment (P)
            dydt.append(-omega * (1-f[age1]) * y[n_comp * ((n_sev * age1) + sev) + 5]                                           # P->I
                        -omega * f[age1] * y[n_comp * ((n_sev * age1) + sev) + 5]                                               # P->A
                        +epsilon * y[n_comp * ((n_sev * age1) + sev) + 4])                                                      # L->P    

            #Infected compartment (I)
            dydt.append(-mu * IICUR[age1] * y[n_comp * ((n_sev * age1) + sev) + 6]                                              # I->ICU
                        -mu * (1 - IICUR[age1] - lambda_deaths) * y[n_comp * ((n_sev * age1) + sev) + 6]                        # I->R
                        -mu * lambda_deaths * y[n_comp * ((n_sev * age1) + sev) + 6]                                            # I->D
                        +omega * (1-f[age1]) * y[n_comp * ((n_sev * age1) + sev) + 5])                                          # P->I
            
            #Latent Vaccinated compartment (LV)
            dydt.append(-epsilon * y[n_comp * ((n_sev * age1) + sev) + 7]                                                       # LV->PV
                        +beta * (1-VE_S) * S_to_L * y[n_comp * ((n_sev * age1) + sev) + 2]                                      # V->LV  
                        +beta * (1-VE_S) * SNC_to_L * y[n_comp * ((n_sev * age1) + sev) + 3])                                   # VNC->LV   

            #Pre-symptomatic Vaccinated compartment (PV)
            dydt.append(-omega * (1-f[age1]) * (1-VE_Symp) * y[n_comp * ((n_sev * age1) + sev) + 8]                             # PV->IV
                        -omega * (1 - ((1-f[age1]) * (1-VE_Symp))) * y[n_comp * ((n_sev * age1) + sev) + 8]                     # PV->A
                        +epsilon * y[n_comp * ((n_sev * age1) + sev) + 7])                                                      # LV->PV        
            
            #Infected Vaccinated compartment (IV)
            dydt.append(-mu * IICUR[age1] * (1-VE_D) * y[n_comp * ((n_sev * age1) + sev) + 9]                                   # IV->ICU
                        -mu * (1 - ((IICUR[age1] + lambda_deaths) * (1-VE_D))) * y[n_comp * ((n_sev * age1) + sev) + 9]         # IV->R
                        -mu * lambda_deaths * (1-VE_D) * y[n_comp * ((n_sev * age1) + sev) + 9]                                 # IV->D
                        +omega * (1-f[age1]) * (1-VE_Symp) * y[n_comp * ((n_sev * age1) + sev) + 8])                            # LV->IV

            #Asymptomatic compartment (A)
            dydt.append(-mu * y[n_comp * ((n_sev * age1) + sev) + 10]                                                           # A->R
                        +omega * f[age1] * y[n_comp * ((n_sev * age1) + sev) + 5]                                               # P->A
                        +omega * (1 - ((1-f[age1]) * (1-VE_Symp))) * y[n_comp * ((n_sev * age1) + sev) + 8])                    # PV->A

            #Intensive Care Units compartment (ICU)
            dydt.append(-(1/delta) * (1 - PICUD[age1]) * y[n_comp * ((n_sev * age1) + sev) + 11]                                # ICU->R
                        -(1/delta) * PICUD[age1] * y[n_comp * ((n_sev * age1) + sev) + 11]                                      # ICU->D
                        +mu * IICUR[age1] * y[n_comp * ((n_sev * age1) + sev) + 6]                                              # I->ICU
                        +mu * IICUR[age1] * (1-VE_D) * y[n_comp * ((n_sev * age1) + sev) + 9])                                  # IV->ICU

            #Recovered compartment (R)
            dydt.append(+mu * (1 - IICUR[age1] - lambda_deaths) * y[n_comp * ((n_sev * age1) + sev) + 6]                        # I->R
                        +mu * (1 - ((IICUR[age1] + lambda_deaths) * (1-VE_D))) * y[n_comp * ((n_sev * age1) + sev) + 9]         # IV->R
                        +mu * y[n_comp * ((n_sev * age1) + sev) + 10]                                                           # A->R
                        +(1/delta) * (1 - PICUD[age1]) * y[n_comp * ((n_sev * age1) + sev) + 11])                               # ICU->R

            #Deads compartment (D)
            dydt.append(+mu * lambda_deaths * y[n_comp * ((n_sev * age1) + sev) + 6]                                            # I->D
                        +mu * lambda_deaths * (1-VE_D) * y[n_comp * ((n_sev * age1) + sev) + 9]                                 # IV->D
                        +(1/delta) * PICUD[age1] * y[n_comp * ((n_sev * age1) + sev) + 11])                                     # ICU->D

    return dydt



def integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, alpha, gamma, a0, b0, t_step=1):

    """
    This function integrates step by step the system defined previously.
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
        :param a0 (array): midpoint logisitic rate for transition S -> S_NC for each perceived severity group
        :param b0 (array): midpoint logisitic rate for transition S_NC -> S for each perceived severity group
        :param t_step (float, default=1): time step
        :return: returns the solution to the system dydt of differential equations
    Note:
        - age groups must be ordered sequentially (from youngest to oldest)
        - for each age group compartments must be ordered like this: 
                      S:0, S_NC:1, V:2, V_NC:3, L:4, P:5, I:6, LV:7, PV:8, IV:9, A:10, ICU:11, R:12, D:13
        - the timestep by timestep trick is needed to properly update vt and dt
    """

    # assign useful variables
    Nij = country_dict['Nij']
    CM = country_dict['CM_red']
    CM2 = country_dict['CM_yellow']

    # create solution array
    sol  = np.zeros((n_age, n_sev, n_comp, t_max))

    # set initial conditions
    y0 = initialize_model(Nij, i0, icu0, r0, d0, epsilon, omega, mu, f, alpha, gamma, a0, b0, ICU_max)
    sol[:,:,:,0] = y0

    V = 0                         # total number of vaccinated
    t = 0                         # current time step
    v_list = []                   # time series of new vaccinated
    dates = [initial_date]        # array of dates

    # integrate
    for i in np.arange(1,t_max,1):
        
        # update dates
        dates.append(dates[-1] + timedelta(days=t_step))
        
        # update fraction of vaccinated and dead
        v_t = V * t_step / np.sum(Nij)
        ICU_t = np.sum(sol[:,:,11,i-1]) / ICU_max

        # number of people to vaccinate in this time step    
        new_V, new_V_NC = get_vaccinated(sol[:,:,:,i-1].ravel(), t, tV0, Nij, r_v, type_v)
        
        # integrate one step ahead
        y = odeint(model, sol[:,:,:,i-1].ravel(), [t, t+t_step], args=(CM, CM2, Nij, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, VE_S, VE_Symp, VE_D, alpha, gamma, a0, b0, new_V, new_V_NC, v_t, ICU_t))
        
        # update the solution (note y[0] is the system at time t, we want t+step)
        sol[:,:,:,i] = np.array(y[1]).reshape((n_age, n_sev, n_comp))

        # update number of vaccinated
        v_list.append(np.sum(new_V) + np.sum(new_V_NC))
        V += v_list[-1]

        # advance time step
        t += t_step
        
    return sol, dates