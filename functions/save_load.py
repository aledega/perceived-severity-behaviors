import pickle



def load_data_a0(folder, alpha, gamma, b0):

    """
    This function loads the data related to transition S -> S_NC in pickle format for the three metrics for specific values of the parameters
        :param folder (string): folder where the data are located
        :param alpha (float): value of the slope S -> S_NC used to generate the data
        :param gamma (array): value of the slope S_NC -> S used to generate the data
        :param b_0 (array): value of the midpoint S_NC -> S used to generate the data
        :return: returns six dictionaries, with the five functions as keys and the values are 900 elements-list of, respectively,
                 mean value of the midpoint a0, variance of the midpoint sigma^2_a0, number of cases, number of deaths, ICU peak height and ICU peak date 
    """

    a0_dict = pickle.load(open(folder + f'3metrics_heatmap_a0dict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'rb'))
    sigma_a0_dict = pickle.load(open(folder + f'3metrics_heatmap_sigmaa0dict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'rb'))
    cases_dict = pickle.load(open(folder + f'3metrics_heatmap_casesdict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'rb'))
    deaths_dict = pickle.load(open(folder + f'3metrics_heatmap_deathsdict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'rb'))
    peak_high_dict = pickle.load(open(folder + f'3metrics_heatmap_peakheightdict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'rb'))
    peak_date_dict = pickle.load(open(folder + f'3metrics_heatmap_peakdatedict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'rb'))
    return a0_dict, sigma_a0_dict, cases_dict, deaths_dict, peak_high_dict, peak_date_dict

def save_data_a0(folder, alpha, gamma, b0, a0_dict, sigma_a0_dict, cases_dict, deaths_dict, peak_high_dict, peak_date_dict):

    """
    This function saves the data related to transition S -> S_NC in pickle format for the three metrics for specific values of the parameters
        :param folder (string): folder where the data are going to be saved
        :param alpha (float): value of the slope S -> S_NC used to generate the data
        :param gamma (array): value of the slope S_NC -> S used to generate the data
        :param b_0 (array): value of the midpoint S_NC -> S used to generate the data
        :param a0_dict (dict): dictionary with the five functions as keys and 900 elements-list of mean values of the midpoint a0 as values
        :param sigma_a0_dict (dict): dictionary with the five functions as keys and 900 elements-list of variance of the midpoint sigma^2_a0 as values
        :param cases_dict (dict): dictionary with the five functions as keys and 900 elements-list of number of cases as values
        :param deaths_dict (dict): dictionary with the five functions as keys and 900 elements-list of number of deaths as values
        :param peak_high_dict (dict): dictionary with the five functions as keys and 900 elements-list of ICU peak height as values
        :param peak_date_dict (dict): dictionary with the five functions as keys and 900 elements-list of ICU peak date as values
        :return: none 
    """

    pickle.dump(a0_dict, open(folder + f'3metrics_heatmap_a0dict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(sigma_a0_dict, open(folder + f'3metrics_heatmap_sigmaa0dict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(cases_dict, open(folder + f'3metrics_heatmap_casesdict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(deaths_dict, open(folder + f'3metrics_heatmap_deathsdict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(peak_high_dict, open(folder + f'3metrics_heatmap_peakheightdict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(peak_date_dict, open(folder + f'3metrics_heatmap_peakdatedict_gamma{gamma}_b0{b0}_alpha{alpha}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def load_data_b0(folder, alpha, gamma, a0):

    """
    This function loads the data related to transition S_NC -> S_C in pickle format for the three metrics for specific values of the parameters
        :param folder (string): folder where the data are located
        :param alpha (float): value of the slope S -> S_NC used to generate the data
        :param gamma (array): value of the slope S_NC -> S used to generate the data
        :param a_0 (array): value of the midpoint S -> S_NC used to generate the data
        :return: returns six dictionaries, with the five functions as keys and the values are 900 elements-list of, respectively,
                 mean value of the midpoint b0, variance of the midpoint sigma^2_b0, number of cases, number of deaths, ICU peak height and ICU peak date 
    """

    b0_dict = pickle.load(open(folder + f'3metrics_heatmap_b0dict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'rb'))
    sigma_b0_dict = pickle.load(open(folder + f'3metrics_heatmap_sigmab0dict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'rb'))
    cases_dict = pickle.load(open(folder + f'3metrics_heatmap_casesdict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'rb'))
    deaths_dict = pickle.load(open(folder + f'3metrics_heatmap_deathsdict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'rb'))
    peak_high_dict = pickle.load(open(folder + f'3metrics_heatmap_peakheightdict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'rb'))
    peak_date_dict = pickle.load(open(folder + f'3metrics_heatmap_peakdatedict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'rb'))
    return b0_dict, sigma_b0_dict, cases_dict, deaths_dict, peak_high_dict, peak_date_dict

def save_data_b0(folder, alpha, gamma, a0, b0_dict, sigma_b0_dict, cases_dict, deaths_dict, peak_high_dict, peak_date_dict):

    """
    This function saves the data related to transition S_NC -> S in pickle format for the three metrics for specific values of the parameters
        :param folder (string): folder where the data are going to be saved
        :param alpha (float): value of the slope S -> S_NC used to generate the data
        :param gamma (array): value of the slope S_NC -> S used to generate the data
        :param a_0 (array): value of the midpoint S -> S_NC used to generate the data
        :param b0_dict (dict): dictionary with the five functions as keys and 900 elements-list of mean values of the midpoint b0 as values
        :param sigma_b0_dict (dict): dictionary with the five functions as keys and 900 elements-list of variance of the midpoint sigma^2_b0 as values
        :param cases_dict (dict): dictionary with the five functions as keys and 900 elements-list of number of cases as values
        :param deaths_dict (dict): dictionary with the five functions as keys and 900 elements-list of number of deaths as values
        :param peak_high_dict (dict): dictionary with the five functions as keys and 900 elements-list of ICU peak height as values
        :param peak_date_dict (dict): dictionary with the five functions as keys and 900 elements-list of ICU peak date as values
        :return: none 
    """

    pickle.dump(b0_dict, open(folder + f'3metrics_heatmap_b0dict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(sigma_b0_dict, open(folder + f'3metrics_heatmap_sigmab0dict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(cases_dict, open(folder + f'3metrics_heatmap_casesdict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(deaths_dict, open(folder + f'3metrics_heatmap_deathsdict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(peak_high_dict, open(folder + f'3metrics_heatmap_peakheightdict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(peak_date_dict, open(folder + f'3metrics_heatmap_peakdatedict_alpha{alpha}_a0{a0}_gamma{gamma}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)