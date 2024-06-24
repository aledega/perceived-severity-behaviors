import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.tri as tri
from matplotlib.ticker import MultipleLocator

from import_auxiliary import array_persev_computation
from integrate_equations import integrate_model


# global variables
n_comp = 14                                                                                 # number of compartments
age_groups = ['0-4', '5-17', '18-29', '30-39', '40-49', '50-59', '60+']                     # list of strings with age groups 
age_groups_bins = [0, 5, 18, 30, 40, 50, 60, np.inf]                                        # list of int with age groups extremes 
n_age = len(age_groups)                                                                     # number of age groups
n_sev = 5                                                                                   # number of perceived severity groups        


def metrics_plot(var_dict, sigma_var_dict, deaths_dict, peak_high_dict, peak_date_dict, grow, function_types, savefig = None):

    min_list = [0,0,0]
    max_list = [0,0,0]
    for i, metric in zip(range(3), [deaths_dict, peak_high_dict, peak_date_dict]):
        for j, fun_type in zip(range(5), function_types):
            if j==0:
                min_list[i] = min(metric[fun_type])
                max_list[i] = max(metric[fun_type])
            if min_list[i] > min(metric[fun_type]):
                min_list[i] = min(metric[fun_type])
            if max_list[i] < max(metric[fun_type]):
                max_list[i] = max(metric[fun_type])

    fig, axes = plt.subplots(3, 5, figsize=(25,15), sharey = 'row', sharex = 'col', constrained_layout=True)

    if grow: 
        function_types_labels = ['Growth', 'Central Growth', 'Start End Growth', 'Start Growth', 'End Growth']
    else: 
        function_types_labels = ['Decrease', 'Central Decrease', 'Start End Decrease', 'Start Decrease', 'End Decrease']

    cmap_list = [cm.coolwarm, cm.viridis, cm.plasma]

    for i, metric in zip(range(3), [deaths_dict, peak_high_dict, peak_date_dict]):
        for j, fun_type in zip(range(5), function_types):
            x = sigma_var_dict[fun_type]
            y = var_dict[fun_type]
            triang = tri.Triangulation(x,y)
            
            cntrf = axes[i,j].tricontourf(triang, metric[fun_type], levels = 14, cmap = cmap_list[i], norm = colors.Normalize(min_list[i], max_list[i]))
            if i == 0:
                axes[i, j].set_title(function_types_labels[j], fontsize = 30, pad = 15)
            if j == 0:
                if grow:
                    axes[i,j].set_ylabel(r"$\overline{a_0}$", fontsize = 24)
                else:
                    axes[i,j].set_ylabel(r"$\overline{b_0}$", fontsize = 24)
                axes[i,j].yaxis.set_tick_params(labelsize=20)
            if i == len(axes[:,0])-1:
                if grow:
                    axes[i,j].set_xlabel(r"$\sigma²_{a_0}$", fontsize = 24)
                else:
                    axes[i,j].set_xlabel(r"$\sigma²_{b_0}$", fontsize = 24)
                
                axes[i,j].xaxis.set_major_locator(plt.MaxNLocator(5))
                axes[i,j].xaxis.set_tick_params(labelsize=20)
                axes[i,j].xaxis.set_ticks([0.0, 0.1, 0.2, 0.3])
                
        if i == 0:
            cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(round(min_list[i]-1000,-3), round(max_list[i]+10000,-4)), cmap=cmap_list[i]), ax=axes[i,:])
            cbar.ax.tick_params(labelsize=20)
            cbar.set_ticks(np.linspace(round(min_list[i]-1000,-3), round(max_list[i]+10000,-4), 6))
            cbar.set_ticklabels([f'{tick:.0f}' for tick in np.linspace(round(min_list[i]-1000,-3), round(max_list[i]+10000,-4), 6)])
        elif i == 1:
            cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(round(min_list[i]-0.1,1), round(max_list[i]+0.1,1)), cmap=cmap_list[i]), ax=axes[i,:])
            cbar.ax.tick_params(labelsize=20)
            cbar.set_ticks(np.linspace(round(min_list[i]-0.1,1), round(max_list[i]+0.1,1), 6))
            cbar.set_ticklabels([f'{tick:.1f}' for tick in np.linspace(round(min_list[i]-0.1,1), round(max_list[i]+0.1,1), 6)])
        else:
            cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(min_list[i], max_list[i]), cmap=cmap_list[i]), ax=axes[i,:])
            cbar.ax.tick_params(labelsize=20)
            cbar.set_ticks(np.linspace(min_list[i], max_list[i], 6))
            cbar.set_ticklabels([f'{tick:.0f}' for tick in np.linspace(min_list[i], max_list[i], 6)])

    pad = 20 # in points
    rows = ['N of deaths', 'ICU peak height', 'ICU peak date'] 
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad,0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='right', va='center', rotation = 90, fontsize = 30)

    plt.tight_layout
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()


def epi_curves(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, alpha, gamma, a0, b0, t_step, maxsigma, grow, fun_type, savefig = None):
    
    comp_to_plot = ['V', 'I', 'Cases', 'D', 'ICU']

    fig, axes = plt.subplots(len(comp_to_plot), 7, figsize=(6*5, (len(comp_to_plot)+1)*5), sharey='row', sharex = 'col')

    label_dict = {0: 'Per. Sev. 1', 1: 'Per. Sev. 2', 2: 'Per. Sev. 3', 3: 'Per. Sev. 4', 4: 'Per. Sev. 5'}

    sigma_var_list = np.linspace(0, maxsigma, 7)

    for i, comp in zip(range(len(comp_to_plot)), comp_to_plot):
        for j, sigma_var in zip(range(len(sigma_var_list)), sigma_var_list):
            if grow:
                var_vec = array_persev_computation(country_dict['Nij'], a0, sigma_var, fun_type, grow=True)
                y, dates = integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, [alpha]*5, [gamma]*5, var_vec, [b0]*5, t_step)
            else:
                var_vec = array_persev_computation(country_dict['Nij'], b0, sigma_var, fun_type, grow=False)
                y, dates = integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, [alpha]*5, [gamma]*5, [a0]*5, var_vec, t_step)
            axes[i,j].grid(True, color = '0.8', which='major')
            axes[i,j].grid(True, color = '0.8', alpha = 0.5, which='minor', linestyle='--')
            axes[i,j].xaxis.set_major_locator(MultipleLocator(100))
            axes[i,j].xaxis.set_minor_locator(MultipleLocator(25))
            axes[i,j].yaxis.set_tick_params(labelsize=20)
            if comp == 'I':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), (y[:, k, 6, :].sum(axis=0) + y[:, k, 9, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                max = round(np.max((y[:, 0, 6, :].sum(axis=0) + y[:, 0, 9, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[0]), 3)
                if max <= 0.006:
                    axes[i,j].set_yticks([0.000, 0.002, 0.004, 0.006])
                elif (max > 0.006 and max <= 0.008):
                    axes[i,j].set_yticks(np.linspace(0, 0.008, 5))
                elif (max > 0.008 and max <= 0.009):
                    axes[i,j].set_yticks([0.000, 0.003, 0.006, 0.009])
                elif (max > 0.009 and max <= 0.012):
                    axes[i,j].set_yticks(np.linspace(0, 0.012, 5))
                elif (max > 0.012):
                    axes[i,j].set_yticks(np.linspace(0, 0.016, 5))
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-3,-3))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'Cases':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), (y[:, k, 12, :].sum(axis=0)+y[:, k, 13, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                max = round(np.max((y[:, 0, 12, :].sum(axis=0)+y[:, 0, 13, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[0]), 1)
                if max <= 0.4:
                    axes[i,j].set_yticks([0.1, 0.2, 0.3, 0.4])
                elif max > 0.4:
                    axes[i,j].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
            elif comp == 'D':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), y[:, k, 13, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                max = round(np.max(y[:, -1, 13, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[-1]), 4)
                if max <= 0.0006:
                    axes[i,j].set_yticks([0.0000, 0.0002, 0.0004, 0.0006])
                elif (max > 0.0006 and max <= 0.0008):
                    axes[i,j].set_yticks(np.linspace(0, 0.0008, 5))
                elif (max > 0.0008 and max <= 0.0009):
                    axes[i,j].set_yticks([0.0000, 0.0003, 0.0006, 0.0009])
                elif (max > 0.0009 and max <= 0.0012):
                    axes[i,j].set_yticks(np.linspace(0, 0.0012, 5))
                elif (max > 0.0012):
                    axes[i,j].set_yticks(np.linspace(0, 0.0016, 5))
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-4,-4))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'ICU':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), y[:, k, 11, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                max = round(np.max(y[:, -1, 11, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[-1]), 5)
                if max <= 0.00006:
                    axes[i,j].set_yticks([0.00000, 0.00002, 0.00004, 0.00006])
                elif (max > 0.00006 and max <= 0.00008):
                    axes[i,j].set_yticks(np.linspace(0, 0.00008, 5))
                elif (max > 0.00008 and max <= 0.00009):
                    axes[i,j].set_yticks([0.00000, 0.00003, 0.00006, 0.00009])
                elif (max > 0.00009 and max <= 0.00012):
                    axes[i,j].set_yticks(np.linspace(0, 0.00012, 5))
                elif (max > 0.00012 and max <= 0.00016):
                    axes[i,j].set_yticks(np.linspace(0, 0.00016, 5))
                elif (max > 0.00016):
                    axes[i,j].set_yticks([0.00000, 0.00006, 0.00012, 0.00018])
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-5,-5))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'V':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), (y[:, k, 2, :].sum(axis=0) + y[:, k, 3, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                axes[i,j].plot(list(range(len(dates))), (y[:, :, 2, :].sum(axis=(0,1)) + y[:, :, 3, :].sum(axis=(0,1)))/country_dict['Nij'].sum(axis=(0,1)), linestyle = '--', color = 'black')
                axes[i,j].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
            elif comp == 'V_aggregated':
                axes[i,j].plot(list(range(len(dates))), (y[:, :, 2, :].sum(axis=(0,1)) + y[:, :, 3, :].sum(axis=(0,1)))/country_dict['Nij'].sum(axis=(0,1)), linestyle = '--', color = 'black')
            if i == len(axes[:,0])-1:
                axes[i,j].set_xlabel('t', fontsize = 24)
            axes[i,j].tick_params(axis='x', rotation=45, labelsize = 20)

    if grow:
        cols = [r'$\sigma²_{a_0}$ = ' + str(round(i,3)) for i in sigma_var_list]
    else:
        cols = [r'$\sigma²_{b_0}$ = ' + str(round(i,3)) for i in sigma_var_list]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize = 24, pad = 30)


    bool_max1 = round(np.max((y[:, 0, 6, :].sum(axis=0) + y[:, 0, 9, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[0]), 3) >= 0.010
    bool_max2 = round(np.max(y[:, -1, 13, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[-1]), 4) >= 0.0010
    bool_max3 = round(np.max(y[:, -1, 11, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[-1]), 5) >= 0.00010


    pad = 30 # in points
    rows = ['Vaccinated fraction', 'Infected fraction', 'Cases fraction', 'Deaths fraction', 'ICU fraction']
    for ax, row in zip(axes[:,0], rows):
        if row in ['Vaccinated fraction', 'Cases fraction']:
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad,0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        ha='center', va='center', rotation = 90, fontsize = 24)
        elif row == 'Infected fraction':
            if bool_max1:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-5,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
            else:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-17,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
        elif row == 'Deaths fraction':
            if bool_max2:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-5,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
            else:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-17,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
        elif row == 'ICU fraction':
            if bool_max3:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-5,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
            else:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-17,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.3)  # create some space below the plots by increasing the bottom-value
    axes.flatten()[-4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, fontsize = 24)

    plt.tight_layout
    fig.subplots_adjust(left=0.15, top=0.90)
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()


def epi_comparison(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, alpha, gamma, a0, b0, t_step, maxsigma, grow, comp, function_types, savefig = None):

    fig, axes = plt.subplots(len(function_types), 7, figsize=(6*5, (len(function_types)+1)*5), sharey='all', sharex = 'col')

    label_dict = {0: 'Per. Sev. 1', 1: 'Per. Sev. 2', 2: 'Per. Sev. 3', 3: 'Per. Sev. 4', 4: 'Per. Sev. 5'}

    sigma_var_list = np.linspace(0, maxsigma, 7)

    for i, fun_type in zip(range(len(function_types)), function_types):
        for j, sigma_var in zip(range(len(sigma_var_list)), sigma_var_list):
            if grow:
                var_vec = array_persev_computation(country_dict['Nij'], a0, sigma_var, fun_type, grow=True)
                y, dates = integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, [alpha]*5, [gamma]*5, var_vec, [b0]*5, t_step)
            else:
                var_vec = array_persev_computation(country_dict['Nij'], b0, sigma_var, fun_type, grow=False)
                y, dates = integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, [alpha]*5, [gamma]*5, [a0]*5, var_vec, t_step)
            axes[i,j].grid(True, color = '0.8', which='major')
            axes[i,j].grid(True, color = '0.8', alpha = 0.5, which='minor', linestyle='--')
            axes[i,j].xaxis.set_major_locator(MultipleLocator(100))
            axes[i,j].xaxis.set_minor_locator(MultipleLocator(25))
            axes[i,j].yaxis.set_tick_params(labelsize=20)
            if comp == 'I':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), (y[:, k, 6, :].sum(axis=0) + y[:, k, 9, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                max = round(np.max((y[:, 0, 6, :].sum(axis=0) + y[:, 0, 9, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[0]), 3)
                if max <= 0.006:
                    axes[i,j].set_yticks([0.000, 0.002, 0.004, 0.006])
                elif (max > 0.006 and max <= 0.008):
                    axes[i,j].set_yticks(np.linspace(0, 0.008, 5))
                elif (max > 0.008 and max <= 0.009):
                    axes[i,j].set_yticks([0.000, 0.003, 0.006, 0.009])
                elif (max > 0.009 and max <= 0.012):
                    axes[i,j].set_yticks(np.linspace(0, 0.012, 5))
                elif (max > 0.012):
                    axes[i,j].set_yticks(np.linspace(0, 0.016, 5))
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-3,-3))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'Cases':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), (y[:, k, 12, :].sum(axis=0)+y[:, k, 13, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                max = round(np.max((y[:, 0, 12, :].sum(axis=0)+y[:, 0, 13, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[0]), 1)
                if max <= 0.4:
                    axes[i,j].set_yticks([0.1, 0.2, 0.3, 0.4])
                elif max > 0.4:
                    axes[i,j].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
            elif comp == 'D':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), y[:, k, 13, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                max = round(np.max(y[:, -1, 13, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[-1]), 4)
                if max <= 0.0006:
                    axes[i,j].set_yticks([0.0000, 0.0002, 0.0004, 0.0006])
                elif (max > 0.0006 and max <= 0.0008):
                    axes[i,j].set_yticks(np.linspace(0, 0.0008, 5))
                elif (max > 0.0008 and max <= 0.0009):
                    axes[i,j].set_yticks([0.0000, 0.0003, 0.0006, 0.0009])
                elif (max > 0.0009 and max <= 0.0012):
                    axes[i,j].set_yticks(np.linspace(0, 0.0012, 5))
                elif (max > 0.0012):
                    axes[i,j].set_yticks(np.linspace(0, 0.0016, 5))
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-4,-4))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'ICU':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), y[:, k, 11, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                max = round(np.max(y[:, -1, 11, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[-1]), 5)
                if max <= 0.00006:
                    axes[i,j].set_yticks([0.00000, 0.00002, 0.00004, 0.00006])
                elif (max > 0.00006 and max <= 0.00008):
                    axes[i,j].set_yticks(np.linspace(0, 0.00008, 5))
                elif (max > 0.00008 and max <= 0.00009):
                    axes[i,j].set_yticks([0.00000, 0.00003, 0.00006, 0.00009])
                elif (max > 0.00009 and max <= 0.00012):
                    axes[i,j].set_yticks(np.linspace(0, 0.00012, 5))
                elif (max > 0.00012 and max <= 0.00016):
                    axes[i,j].set_yticks(np.linspace(0, 0.00016, 5))
                elif (max > 0.00016):
                    axes[i,j].set_yticks([0.00000, 0.00006, 0.00012, 0.00018])
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-5,-5))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'V':
                for k in range(n_sev):
                    axes[i,j].plot(list(range(len(dates))), (y[:, k, 2, :].sum(axis=0) + y[:, k, 3, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[k], label = label_dict[k])
                axes[i,j].plot(list(range(len(dates))), (y[:, :, 2, :].sum(axis=(0,1)) + y[:, :, 3, :].sum(axis=(0,1)))/country_dict['Nij'].sum(axis=(0,1)), linestyle = '--', color = 'black')
            elif comp == 'V_aggregated':
                axes[i,j].plot(list(range(len(dates))), (y[:, :, 2, :].sum(axis=(0,1)) + y[:, :, 3, :].sum(axis=(0,1)))/country_dict['Nij'].sum(axis=(0,1)), linestyle = '--', color = 'black')
            if i == len(axes[:,0])-1:
                axes[i,j].set_xlabel('t', fontsize = 24)
            axes[i,j].tick_params(axis='x', rotation=45, labelsize = 20)

    if grow:
        cols = [r'$\sigma²_{a_0}$ = ' + str(round(i,3)) for i in sigma_var_list]
    else:
        cols = [r'$\sigma²_{b_0}$ = ' + str(round(i,3)) for i in sigma_var_list]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize = 24, pad = 30)

    pad = 20 # in points
    if grow: 
        rows = ['Growth', 'Central Growth', 'Start End Growth', 'Start Growth', 'End Growth']
    else: 
        rows = ['Decrease', 'Central Decrease', 'Start End Decrease', 'Start Decrease', 'End Decrease']
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad,0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='center', va='center', rotation = 90, fontsize = 24)

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.3)  # create some space below the plots by increasing the bottom-value
    axes.flatten()[-4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, fontsize = 24)

    plt.tight_layout
    fig.subplots_adjust(left=0.15, top=0.90)
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()




def epi_curves_age(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, alpha, gamma, a0, b0, t_step, maxsigma, grow, fun_type, savefig = None):
    
    comp_to_plot = ['V', 'I', 'Cases', 'D', 'ICU']

    fig, axes = plt.subplots(len(comp_to_plot), 7, figsize=(6*5, (len(comp_to_plot)+1)*5), sharey='row', sharex = 'col')

    label_dict = {0: '0-4', 1: '5-17', 2: '18-29', 3: '30-39', 4: '40-49', 5: '50-59', 6: '60+'}

    sigma_var_list = np.linspace(0, maxsigma, 7)

    for i, comp in zip(range(len(comp_to_plot)), comp_to_plot):
        for j, sigma_var in zip(range(len(sigma_var_list)), sigma_var_list):
            if grow:
                var_vec = array_persev_computation(country_dict['Nij'], a0, sigma_var, fun_type, grow=True)
                y, dates = integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, [alpha]*5, [gamma]*5, var_vec, [b0]*5, t_step)
            else:
                var_vec = array_persev_computation(country_dict['Nij'], b0, sigma_var, fun_type, grow=False)
                y, dates = integrate_model(country_dict, i0, icu0, r0, d0, initial_date, t_max, tV0, beta, epsilon, omega, mu, delta, chi, f, IFR, IICUR, PICUD, ICU_max, r_v, type_v, VE_S, VE_Symp, VE_D, [alpha]*5, [gamma]*5, [a0]*5, var_vec, t_step)
            axes[i,j].grid(True, color = '0.8', which='major')
            axes[i,j].grid(True, color = '0.8', alpha = 0.5, which='minor', linestyle='--')
            axes[i,j].xaxis.set_major_locator(MultipleLocator(100))
            axes[i,j].xaxis.set_minor_locator(MultipleLocator(25))
            axes[i,j].yaxis.set_tick_params(labelsize=20)
            if comp == 'I':
                for k in range(n_age):
                    axes[i,j].plot(list(range(len(dates))), (y[k, :, 6, :].sum(axis=0) + y[k, :, 9, :].sum(axis=0))/country_dict['Nij'].sum(axis=1)[k], label = label_dict[k])
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-3,-3))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'Cases':
                for k in range(n_age):
                    axes[i,j].plot(list(range(len(dates))), (y[k, :, 12, :].sum(axis=0)+y[k, :, 13, :].sum(axis=0))/country_dict['Nij'].sum(axis=1)[k], label = label_dict[k])
            elif comp == 'D':
                for k in range(n_age):
                    axes[i,j].plot(list(range(len(dates))), y[k, :, 13, :].sum(axis=0)/country_dict['Nij'].sum(axis=1)[k], label = label_dict[k])
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-4,-4))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'ICU':
                for k in range(n_age):
                    axes[i,j].plot(list(range(len(dates))), y[k, :, 11, :].sum(axis=0)/country_dict['Nij'].sum(axis=1)[k], label = label_dict[k])
                axes[i,j].ticklabel_format(style = 'sci', axis='y', scilimits=(-5,-5))
                axes[i,j].yaxis.get_offset_text().set_fontsize(20)
            elif comp == 'V':
                for k in range(n_age):
                    axes[i,j].plot(list(range(len(dates))), (y[k, :, 2, :].sum(axis=0) + y[k, :, 3, :].sum(axis=0))/country_dict['Nij'].sum(axis=1)[k], label = label_dict[k])
                axes[i,j].plot(list(range(len(dates))), (y[:, :, 2, :].sum(axis=(0,1)) + y[:, :, 3, :].sum(axis=(0,1)))/country_dict['Nij'].sum(axis=(0,1)), linestyle = '--', color = 'black')
                axes[i,j].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
            elif comp == 'V_aggregated':
                axes[i,j].plot(list(range(len(dates))), (y[:, :, 2, :].sum(axis=(0,1)) + y[:, :, 3, :].sum(axis=(0,1)))/country_dict['Nij'].sum(axis=(0,1)), linestyle = '--', color = 'black')
            if i == len(axes[:,0])-1:
                axes[i,j].set_xlabel('t', fontsize = 24)
            axes[i,j].tick_params(axis='x', rotation=45, labelsize = 20)

    if grow:
        cols = [r'$\sigma²_{a_0}$ = ' + str(round(i,3)) for i in sigma_var_list]
    else:
        cols = [r'$\sigma²_{b_0}$ = ' + str(round(i,3)) for i in sigma_var_list]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize = 24, pad = 30)


    bool_max1 = round(np.max((y[:, 0, 6, :].sum(axis=0) + y[:, 0, 9, :].sum(axis=0))/country_dict['Nij'].sum(axis=0)[0]), 3) >= 0.010
    bool_max2 = round(np.max(y[:, -1, 13, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[-1]), 4) >= 0.0010
    bool_max3 = round(np.max(y[:, -1, 11, :].sum(axis=0)/country_dict['Nij'].sum(axis=0)[-1]), 5) >= 0.00010


    pad = 30 # in points
    rows = ['Vaccinated fraction', 'Infected fraction', 'Cases fraction', 'Deaths fraction', 'ICU fraction']
    for ax, row in zip(axes[:,0], rows):
        if row in ['Vaccinated fraction', 'Cases fraction']:
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad,0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        ha='center', va='center', rotation = 90, fontsize = 24)
        elif row == 'Infected fraction':
            if bool_max1:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-5,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
            else:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-5,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
        elif row == 'Deaths fraction':
            if bool_max2:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-5,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
            else:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-17,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
        elif row == 'ICU fraction':
            if bool_max3:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-5,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)
            else:
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad-17,0), xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation = 90, fontsize = 24)

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.3)  # create some space below the plots by increasing the bottom-value
    axes.flatten()[-4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=7, fontsize = 24)

    plt.tight_layout
    fig.subplots_adjust(left=0.15, top=0.90)
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()