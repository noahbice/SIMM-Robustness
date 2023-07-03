import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f_oneway, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib import rcParams
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import make_interp_spline

plt.rc('axes', axisbelow=True)
plt.rc('axes', facecolor='whitesmoke')
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = "serif"
rcParams['font.sans-serif'] = "Computer Modern"

statistic = ' Coverage (95%)'
do_direction = True # make sensitivity-by-direction plot
abs_value = False
do_rotation = True # make sensitivity-by-rotation plot
do_delta_GI = False
do_delta_CI = False
do_R = False
do_num_mets = True

data = pd.read_csv('output.csv')
data = data[data[' Plan ID'] != ' SIMM_Full']
colors = ['mediumorchid', 'deepskyblue', 'goldenrod', 'forestgreen',
          'violet', 'chocolate', 'salmon',  'gold', 'darkgrey', 'blue']

# make magnitude and direction columns
magnitudes = []
directions = []
targetvols = []
for i in range(len(data)):
    planid = data[' Plan ID'].iloc[i]
    if 'mm' in planid:
        directions.append(planid[-1])
        magnitudes.append(planid.split('mm')[0][-4:])
    elif data[' Course ID'].iloc[i] == ' JT':
        magnitudes.append(0.0)
        directions.append('XYZR')
    elif 'deg' in planid:
        magnitudes.append(planid.split('deg')[0][-4:])
        directions.append('R')
    else:
        print('error with plan: ', planid)
data['Magnitude'] = magnitudes
data['Magnitude'] = pd.to_numeric(data['Magnitude'])
data['Direction'] = directions
data[statistic] = pd.to_numeric(data[statistic])

targetvolumes = np.array(pd.to_numeric(data[' Target Volume (cc)']))
targetvolumes = targetvolumes[targetvolumes < 8]
print(np.median(targetvolumes), np.mean(targetvolumes))
plt.hist(targetvolumes, bins=20)
plt.show()

individual_targets = data[data[' Target ID'].str.contains('ptv', case=True)]
total_targets = data[data[' Target ID'].str.contains('PTV', case=True)]

X = individual_targets[individual_targets['Direction'].str.contains('X')]
Y = individual_targets[individual_targets['Direction'].str.contains('Y')]
Z = individual_targets[individual_targets['Direction'].str.contains('Z')]
rot = individual_targets[individual_targets['Direction'].str.contains('R')]
dir_dict = {'X': X, 'Y': Y, 'Z': Z}
dir_tags = [' (L/R)', ' (A/P)', ' (S/I)']

if do_direction:
    i = 0
    residuals = []
    names = []
    for string, d in dir_dict.items():
        unique_mags = d['Magnitude'].unique()
        unique_mags.sort()
        meds = []
        bars = []
        means = []
        coverages = []
        for mag in unique_mags:
            if abs_value:
                dmag = d[d['Magnitude'].abs() == mag]
            else:
                dmag = d[d['Magnitude'] == mag]
            covs = dmag[statistic]
            med = covs.quantile(0.5)
            meds.append(med)
            bars.append([med - covs.quantile(0.25), covs.quantile(0.75) - med])
            means.append(covs.mean())
        residuals.append(means)
        meds = np.array(meds)
        bars = np.array(bars).T
        plt.errorbar(unique_mags, meds, yerr=bars, capsize=8, ls='', label=string + ' shift' + dir_tags[i],
                     elinewidth=2, color=colors[i])
        plt.plot(unique_mags, gaussian_filter1d(meds, sigma=0.6), color=colors[i], linewidth=0.8)
        names.append(string)
        # plt.plot(unique_mags, means, label=string)
        i += 1
    plt.grid(ls='--')
    # plt.title('Translation sensitivity by direction', fontsize=14)
    plt.ylabel(r'$V_{98\%}$', fontsize=14)
    # plt.ylabel(statistic[1:])
    plt.xlabel('$x$ [mm]', fontsize=14)
    legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.5, 0.065), fontsize=12)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    plt.savefig('./newfigs/v98byshift.png', dpi=300)
    plt.show()
    plt.plot()

    # one way anova -- H0: mean(stat - mean(stat)) = same (0) for all directions
    residuals = np.array(residuals)
    mean_correct = np.mean(residuals, axis=0)
    for i in range(residuals.shape[0]):
        residuals[i] -= mean_correct
    #     plt.scatter(unique_mags, residuals[i])
    # plt.show()

    print('')
    print('Directions comparison: ')
    print(f_oneway(*residuals))
    total_resid = []
    total_direction = []
    print(residuals.shape, names)
    for i in range(residuals.shape[0]):
        total_resid += list(residuals[i])
        total_direction += [names[i] for _ in range(residuals.shape[1])]
    stat_df = pd.DataFrame({'resid': total_resid, 'direction': total_direction})
    print(pairwise_tukeyhsd(stat_df['resid'], stat_df['direction']))

if do_rotation:
    i = 0
    for [string, d] in [['R', rot]]:
        unique_mags = d['Magnitude'].unique()
        unique_mags.sort()
        meds = []
        bars = []
        means = []
        for mag in unique_mags:
            dmag = d[d['Magnitude'] == mag]
            covs = dmag[statistic]
            med = covs.quantile(0.5)
            meds.append(med)
            bars.append([med - covs.quantile(0.25), covs.quantile(0.75) - med])
            means.append(covs.mean())
        meds = np.array(meds)
        bars = np.array(bars).T
        plt.errorbar(unique_mags, meds, ls='', capsize=8, yerr=bars, color=colors[i])
        # plt.plot(unique_mags, means, label=string)
        i += 1
    plt.grid(ls='--')
    # plt.title('Rotation sensitivity by plan')
    plt.ylabel(r'$V_{98\%}$', fontsize=14)
    # plt.ylabel(statistic[1:])
    plt.xlabel('Rotation [degrees]')
    plt.show()
    plt.plot()

if do_delta_GI:
    non_rots = individual_targets[np.invert(individual_targets['Direction'].str.contains('R'))]
    non_rots = non_rots[np.invert(non_rots[' Course ID'] == ' JT')]
    unique_mags = non_rots['Magnitude'].unique()
    unique_mags.sort()
    covs = []
    for mag in unique_mags[int(len(unique_mags) / 2):]:
        if abs_value:
            dmag = non_rots[non_rots['Magnitude'].abs() == mag]
        else:
            dmag = non_rots[non_rots['Magnitude'] == mag]
        init_covs = dmag[' Paddick GI']
        cov = []
        non_shift_plans = individual_targets[individual_targets[' Course ID'] == ' JT']
        non_shift_plan_ids = non_shift_plans[' Plan ID'].unique()
        non_shift_targets = non_shift_plans[' Target ID'].unique()
        for i in range(len(dmag)):
            course = dmag[' Course ID'].iloc[i]
            target = dmag[' Target ID'].iloc[i]
            lookup_plan = ''
            for possible_id in non_shift_plan_ids:
                if possible_id.split('-')[0] in course:
                    lookup_plan = possible_id
            correct_plan = non_shift_plans[non_shift_plans[' Plan ID'] == lookup_plan]
            correct_target = correct_plan[correct_plan[' Target ID'] == target]
            no_shift_cov = pd.to_numeric(correct_target[' Paddick GI'])
            cov.append(float(pd.to_numeric(init_covs.iloc[i]) - no_shift_cov))
        covs.append(cov)
    labels = [str(s) + ' mm' for s in unique_mags[int(len(unique_mags) / 2):]]
    plt.grid(ls='--')
    plt.boxplot(covs, labels=labels)
    # plt.title('Gradient index sensitivity')
    plt.ylabel(r'$\Delta GI}$')
    # plt.ylabel(statistic[1:])
    plt.xlabel('Translation magnitude')
    plt.ylim([-1.5, 1.5])
    legend = plt.legend(facecolor='white', framealpha=1, frameon=1)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    plt.show()
    plt.plot()

if do_delta_CI:
    non_rots = individual_targets[np.invert(individual_targets['Direction'].str.contains('R'))]
    non_rots = non_rots[np.invert(non_rots[' Course ID'] == ' JT')]
    unique_mags = non_rots['Magnitude'].unique()
    unique_mags.sort()
    covs = []
    for mag in unique_mags[int(len(unique_mags) / 2):]:
        if abs_value:
            dmag = non_rots[non_rots['Magnitude'].abs() == mag]
        else:
            dmag = non_rots[non_rots['Magnitude'] == mag]
        init_covs = dmag[' RTOG CI']
        cov = []
        non_shift_plans = individual_targets[individual_targets[' Course ID'] == ' JT']
        non_shift_plan_ids = non_shift_plans[' Plan ID'].unique()
        non_shift_targets = non_shift_plans[' Target ID'].unique()
        for i in range(len(dmag)):
            course = dmag[' Course ID'].iloc[i]
            target = dmag[' Target ID'].iloc[i]
            lookup_plan = ''
            for possible_id in non_shift_plan_ids:
                if possible_id.split('-')[0] in course:
                    lookup_plan = possible_id
            correct_plan = non_shift_plans[non_shift_plans[' Plan ID'] == lookup_plan]
            correct_target = correct_plan[correct_plan[' Target ID'] == target]
            no_shift_cov = pd.to_numeric(correct_target[' RTOG CI'])
            cov.append(float(pd.to_numeric(init_covs.iloc[i]) - no_shift_cov))
        covs.append(cov)
    labels = [str(s) + ' mm' for s in unique_mags[int(len(unique_mags) / 2):]]
    plt.grid(ls='--')
    plt.boxplot(covs, labels=labels)
    # plt.title('RTOG conformity index sensitivity')
    plt.ylabel(r'$\Delta CI}$')
    # plt.ylabel(statistic[1:])
    plt.xlabel('Translation magnitude')
    plt.ylim([-1.5, 1.5])
    legend = plt.legend(facecolor='white', framealpha=1, frameon=1)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    plt.show()
    plt.plot()



if do_R:
    covs = []
    target_vol_cutoff = 1.
    rot = individual_targets[individual_targets['Direction'].str.contains('R')]
    rot = rot[np.invert(rot[' Course ID'] == ' JT')]
    rot = rot[rot[' Target Volume (cc)'] > target_vol_cutoff]
    unique_distances = rot[' Target to iso (cm)'].unique()
    non_shift_plans = individual_targets[individual_targets[' Course ID'] == ' JT']
    non_shift_plan_ids = non_shift_plans[' Plan ID'].unique()
    non_shift_targets = non_shift_plans[' Target ID'].unique()
    means = []
    corr_distances = []
    meds = []
    bounds = []
    for distance in unique_distances:
        distdata = rot[rot[' Target to iso (cm)'] == distance]
        init_covs = distdata[statistic]
        cov = []
        for i in range(len(distdata)):
            course = distdata[' Course ID'].iloc[i]
            target = distdata[' Target ID'].iloc[i]
            lookup_plan = ''
            for possible_id in non_shift_plan_ids:
                if possible_id.split('-')[0] in course:
                    lookup_plan = possible_id
            correct_plan = non_shift_plans[non_shift_plans[' Plan ID'] == lookup_plan]
            correct_target = correct_plan[correct_plan[' Target ID'] == target]
            no_shift_cov = pd.to_numeric(correct_target[statistic])
            delta = pd.to_numeric(init_covs.iloc[i]) - no_shift_cov
            cov.append(float(delta))
        corr_distances.append([float(distance)] * len(distdata))
        covs.append(cov)
        means.append(np.mean(cov))
        med = np.percentile(cov, 50)
        meds.append(med)
        bounds.append([np.percentile(cov, 25) - med, med - np.percentile(cov, 75)])
    plt.errorbar(np.array(unique_distances, dtype='float32'), meds, yerr=np.array(np.abs(bounds)).T,
                 linestyle='', capsize=3, color=colors[5])
    plt.scatter(np.array(unique_distances, dtype='float32'), meds, color=colors[5])
    plt.grid(ls='--')
    plt.title('Cov by target-to-isocenter distance, > {} cc'.format(target_vol_cutoff), fontsize=14)
    plt.ylabel(r'$\Delta V_{98\%}$', fontsize=14)
    # plt.ylabel('$\Delta$' + statistic[1:], fontsize=14)
    plt.xlabel('Target-to-isocenter distance [cm]', fontsize=14)
    plt.show()
    plt.plot()

    corr_distances = np.array(corr_distances)
    covs = np.array(covs)
    print(corr_distances.shape, covs.shape)
    print(pearsonr(corr_distances.flatten(), covs.flatten()))

num_mets_dict = {' JT1': '3 mets', ' JT5': '5 mets', ' JT6': '6 mets', ' JT8': '8 mets', ' JT15': '15 mets'}
if do_num_mets:
    i = 0
    unique_courses = list(data[' Course ID'].unique())
    unique_courses.remove(' JT')
    names = []
    residuals = []
    for course in unique_courses:
        if course.split(' ')[1] in ' '.join(list(num_mets_dict.keys())):
            d = data[np.invert(data[' Plan ID'].str.contains('deg'))]
            d = d[d[' Course ID'] == course]
            unique_mags = d['Magnitude'].unique()
            unique_mags.sort()
            meds = []
            bars = []
            means = []
            for mag in unique_mags:
                if abs_value:
                    dmag = d[d['Magnitude'].abs() == mag]
                else:
                    dmag = d[d['Magnitude'] == mag]
                dmag = dmag[np.invert(dmag[' Target ID'].str.contains('C'))]
                covs = dmag[statistic]
                med = covs.quantile(0.5)
                meds.append(med)
                bars.append([med - covs.quantile(0.25), covs.quantile(0.75) - med])
                means.append(covs.median())
            unique_mags = list(unique_mags)
            zero_shift = data[data[' Course ID'] == ' JT']
            zero_shift = zero_shift[zero_shift[' Plan ID'].str.contains(course.split(' ')[1])]
            unique_mags.append(0)
            zero_shift_covs = zero_shift[statistic]
            zero_shift_med = zero_shift[statistic].quantile(0.5)
            meds.append(zero_shift_med)
            bars.append([zero_shift_med - zero_shift_covs.quantile(0.25), zero_shift_covs.quantile(0.75) - zero_shift_med])

            sorted_idx = np.argsort(unique_mags)
            unique_mags = np.array(unique_mags)[sorted_idx]
            meds = np.array(meds)[sorted_idx]
            bars = np.array(bars).T[:, sorted_idx]
            plt.errorbar(unique_mags, meds, yerr=bars, ls='', capsize=5, label=num_mets_dict[' ' + course.split(' ')[1]], color=colors[i])
            # plt.plot(unique_mags, means, label=string)
            plt.plot(unique_mags, gaussian_filter1d(meds, sigma=0.6), color=colors[i], linewidth=0.8)
            names.append(string)
            residuals.append(meds)
            names.append(num_mets_dict[' ' + course.split(' ')[1]])
            i += 1
    plt.grid(ls='--')
    # plt.title('Translation sensitivity by plan', fontsize=14)
    plt.ylabel(r'$V_{98\%}$', fontsize=14)
    # plt.ylabel(statistic[1:])
    plt.xlabel('$x$ [mm]', fontsize=14)
    legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.53, 0.065), fontsize=11)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    plt.savefig('./newfigs/v98bynummets.png', dpi=300)
    plt.show()

    # one way anova -- H0: mean(stat - mean(stat)) = same (0) for all plans
    residuals = np.array(residuals)
    mean_correct = np.mean(residuals, axis=0)
    for i in range(residuals.shape[0]):
        residuals[i] = residuals[i] - mean_correct

    print('')
    print('Number of mets / plan comparison: ')
    print(f_oneway(*residuals))
    total_resid = []
    total_plan = []
    for i in range(residuals.shape[0]):
        total_resid += list(residuals[i])
        total_plan += [names[i] for _ in range(residuals.shape[1])]
    stat_df = pd.DataFrame({'resid': total_resid, 'plan': total_plan})
    print(pairwise_tukeyhsd(stat_df['resid'], stat_df['plan']))

    i = 0
    unique_courses = list(data[' Course ID'].unique())
    unique_courses.remove(' JT')
    names = []
    residuals = []
    for course in unique_courses:
        if course.split(' ')[1] in ' '.join(list(num_mets_dict.keys())):
            d = data[data[' Plan ID'].str.contains('deg')]
            d = d[d[' Course ID'] == course]
            unique_mags = d['Magnitude'].unique()
            unique_mags.sort()
            meds = []
            bars = []
            means = []
            for mag in unique_mags:
                if abs_value:
                    dmag = d[d['Magnitude'].abs() == mag]
                else:
                    dmag = d[d['Magnitude'] == mag]
                dmag = dmag[np.invert(dmag[' Target ID'].str.contains('C'))]
                covs = dmag[statistic]
                med = covs.quantile(0.5)
                meds.append(med)
                bars.append([med - covs.quantile(0.25), covs.quantile(0.75) - med])
                means.append(covs.median())
            unique_mags = list(unique_mags)
            zero_shift = data[data[' Course ID'] == ' JT']
            zero_shift = zero_shift[zero_shift[' Plan ID'].str.contains(course.split(' ')[1])]
            unique_mags.append(0)
            zero_shift_covs = zero_shift[statistic]
            zero_shift_med = zero_shift[statistic].quantile(0.5)
            meds.append(zero_shift_med)
            bars.append(
                [zero_shift_med - zero_shift_covs.quantile(0.25), zero_shift_covs.quantile(0.75) - zero_shift_med])

            sorted_idx = np.argsort(unique_mags)
            unique_mags = np.array(unique_mags)[sorted_idx]
            meds = np.array(meds)[sorted_idx]
            bars = np.array(bars).T[:, sorted_idx]
            plt.errorbar(unique_mags, meds, yerr=bars, capsize=5, label=num_mets_dict[' ' + course.split(' ')[1]],
                         color=colors[i], ls='')
            plt.plot(unique_mags, meds, color=colors[i], linewidth=0.8)
            # xnew = np.linspace(np.amin(unique_mags), np.amax(unique_mags), 100)
            # spl = make_interp_spline(unique_mags, meds, k=2)  # type: BSpline
            # power_smooth = spl(xnew)
            # plt.plot(xnew, power_smooth, color=colors[i], linewidth=0.8)


            # plt.plot(unique_mags, means, label=string)
            residuals.append(meds)
            names.append(num_mets_dict[' ' + course.split(' ')[1]])
            i += 1
    plt.grid(ls='--')
    # plt.title('Rotation sensitivity by plan', fontsize=14)
    plt.ylabel(r'$V_{95\%}$', fontsize=14)
    # plt.ylabel(statistic[1:])
    plt.xlabel('Rotation [degrees]', fontsize=14)
    legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.1, 0.1), fontsize=12)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    plt.show()
    plt.plot()

    # one way anova -- H0: mean(stat - mean(stat)) = same (0) for all plans
    residuals = np.array(residuals)
    mean_correct = np.mean(residuals, axis=0)
    for i in range(residuals.shape[0]):
        residuals[i] = residuals[i] - mean_correct

    print('')
    print('Number of mets / plan comparison: ')
    print(f_oneway(*residuals))
    total_resid = []
    total_plan = []
    for i in range(residuals.shape[0]):
        total_resid += list(residuals[i])
        total_plan += [names[i] for _ in range(residuals.shape[1])]
    stat_df = pd.DataFrame({'resid': total_resid, 'plan': total_plan})
    print(pairwise_tukeyhsd(stat_df['resid'], stat_df['plan']))

