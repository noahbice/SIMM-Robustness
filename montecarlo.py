import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tqdm import tqdm
from matplotlib import rcParams
plt.rc('axes', axisbelow=True)
plt.rc('axes', facecolor='whitesmoke')
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = "serif"
rcParams['font.sans-serif'] = "Computer Modern"

statistic = ' Coverage (98%)'
statistic_objective = 90
percent_adequate_objective = 90
num_models = 500
num_samples_per_point = 5000
plot_error_model = False
plot_error_model2 = True
plot_coverage_by_error_model = True
plot_fraction_of_adequate_treatment = True
mode = 'median'

data = pd.read_csv('output.csv')
data = data[data[' Plan ID'] != ' SIMM_Full']
colors = ['mediumorchid', 'deepskyblue', 'goldenrod', 'forestgreen',
          'violet', 'chocolate', 'salmon',  'gold', 'darkgrey', 'blue']
statnum = statistic.split('(')[1].split('%')[0]

# make magnitude and direction columns
magnitudes = []
directions = []
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

individual_targets = data[data[' Target ID'].str.contains('ptv', case=True)]
total_targets = data[data[' Target ID'].str.contains('PTV', case=True)]

X = individual_targets[individual_targets['Direction'].str.contains('X')]
Y = individual_targets[individual_targets['Direction'].str.contains('Y')]
Z = individual_targets[individual_targets['Direction'].str.contains('Z')]
rot = individual_targets[individual_targets['Direction'].str.contains('R')]

def gauss(x, s=1):
    return (1 / (s * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x / s) ** 2))

if plot_error_model:
    x = np.linspace(-5, 5, 1000)
    g = gauss(x, s=1)
    plt.plot(x, g, color=colors[-1])
    # plt.title('Example error model $\mathcal{N}(0, 1 mm)$', fontsize=14)
    plt.xlabel(r'$x$ [mm]', fontsize=14)
    plt.ylabel(r'$f(d)$', fontsize=14)
    plt.xlim([-5, 5])
    plt.grid(ls='--')
    plt.show()

# ISOTROPIC -- 1D
unique_mags = individual_targets['Magnitude'].unique()
unique_mags.sort()
means = []
meds = []
bars = []
all =[]
for mag in unique_mags:
    d = individual_targets[individual_targets['Magnitude'] == mag]
    stats = d[statistic]
    means.append(np.mean(stats))
    med = stats.quantile(0.5)
    meds.append(med)
    bars.append([med - stats.quantile(0.25), stats.quantile(0.75) - med])
    all.append(stats.to_numpy())
mags = np.array(unique_mags)
meds = np.array(meds)
bars = np.array(bars).T
covs = np.array(means)
all = np.array(all, dtype='object')

models = []
# x = np.array([unique_mags ** 3, unique_mags ** 2, unique_mags]).T
x = np.expand_dims(np.array(unique_mags), axis=1)
for _ in range(num_models):
    sample_coverage = np.array([])
    for i in range(all.shape[0]):
        sample_coverage = np.append(sample_coverage, np.random.choice(all[i]))
    # models.append(LinearRegression().fit(x, sample_coverage))
    # models.append(SVR(kernel='rbf', C=200, gamma=0.25).fit(x, sample_coverage))
    models.append(SVR(kernel='rbf', C=400, gamma=0.1).fit(x, sample_coverage))

# interper = interp1d(mags, covs, kind='slinear', bounds_error=False, fill_value='extrapolate')
# interper = interp1d(mags, meds, kind='quadratic', bounds_error=False, fill_value='extrapolate')

if plot_error_model2:
    example_mags = np.linspace(-5.5, 5.5, 100)
    # model_input = np.array([example_mags ** 3, example_mags ** 2, example_mags]).T
    model_input = np.expand_dims(example_mags, axis=1)
    example_covs = []
    for _ in range(1000):
        model = np.random.choice(models)
        example_covs.append(model.predict(model_input))
    example_covs = np.array(example_covs)
    example_means = np.mean(example_covs, axis=0)
    lower_example_bars = np.percentile(example_covs, 25, axis=0)
    upper_example_bars = np.percentile(example_covs, 75, axis=0)
    # print(example_means, upper_example_bars, lower_example_bars)
    # plt.plot(example_mags, example_covs, label='Model')
    fig = plt.figure(figsize=(8, 6))
    plt.plot(example_mags, example_means, color='peru', label=r'Ensemble median')
    plt.fill_between(example_mags, lower_example_bars, upper_example_bars, color='navajowhite', label=r'Ensemble IQR')
    plt.scatter(unique_mags, meds, color='seagreen', s=5, label=r'Measurement median')
    plt.errorbar(unique_mags, meds, yerr=bars, capsize=3, label=r'Measurement IQR', color='seagreen', linestyle='')

    # print('lower fill', lower_example_bars)
    # print('upper fill', upper_example_bars)
    # print(example_means)
    # print(covs)
    # plt.title('Modeling coverage as a function of displacement', fontsize=14)
    plt.grid(ls='--')
    legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.75, 0.85), fontsize=14, ncol=2)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    plt.xlabel(r'$x$ [mm]', fontsize=15)
    plt.ylabel(r'$V_{' + statnum + '\%}$', fontsize=16)
    plt.savefig('./newfigs/v{}NumModel.png'.format(statnum), dpi=300)
    plt.show()

range_sigma_dx = np.linspace(0, 3, 100)
meds = []
means = []
stds = []
bars = []
stds_bars = []
percents_covered = []
for sigma_dx in tqdm(range_sigma_dx):
    sampled_coverages_sigma_dx = []
    # for _ in range(num_samples_per_point):
    dx = np.array(np.random.normal(0, scale=sigma_dx, size=num_samples_per_point)).reshape(num_samples_per_point, 1)
    for k in range(len(models)):
        sampled_coverages_sigma_dx.append(models[k].predict(dx))
    sampled_coverages_sigma_dx = np.array(sampled_coverages_sigma_dx).flatten()
    sampled_coverages_sigma_dx[sampled_coverages_sigma_dx < 0] = 0.
    med = np.percentile(sampled_coverages_sigma_dx, 50)
    meds.append(med)
    mean = np.mean(sampled_coverages_sigma_dx)
    means.append(mean)
    std = np.std(sampled_coverages_sigma_dx)
    stds.append(std)
    stds_bars.append([mean-std, mean+std])
    bars.append([np.percentile(sampled_coverages_sigma_dx, 25),
                 np.percentile(sampled_coverages_sigma_dx, 75)])
    # plt.hist(sampled_coverages_sigma_dx[::10])
    # plt.show()
    percents_covered.append((sampled_coverages_sigma_dx > statistic_objective).sum() / np.prod(sampled_coverages_sigma_dx.shape))
percents_covered = np.array(percents_covered) * 100
meds = np.array(meds)
means = np.array(means)
stds = np.array(stds)
stds_bars = np.array(stds_bars).T
bars = np.array(bars).T


ss = np.linspace(0.01, 3, 101)
if '98' in statistic:
    exvs_analytical = [0.999243,0.996978,0.994713,0.992448,0.990183,0.987919,0.985655,0.98339,0.981127,0.978863,0.9766,0.974337,0.972074,0.969812,0.96755,0.965289,0.963028,0.960768,0.958508,0.956249,0.95399,0.951733,0.949475,0.947219,0.944963,0.942708,0.940454,0.938201,0.935949,0.933697,0.931447,0.929198,0.926949,0.924702,0.922455,0.92021,0.917966,0.915723,0.913482,0.911241,0.909002,0.906764,0.904528,0.902293,0.900059,0.897827,0.895596,0.893367,0.891139,0.888913,0.886688,0.884465,0.882244,0.880024,0.877806,0.87559,0.873376,0.871163,0.868953,0.866744,0.864537,0.862332,0.86013,0.857929,0.85573,0.853533,0.851339,0.849146,0.846956,0.844768,0.842582,0.840399,0.838218,0.836039,0.833862,0.831688,0.829517,0.827348,0.825181,0.823017,0.820855,0.818697,0.81654,0.814387,0.812236,0.810088,0.807942,0.8058,0.80366,0.801523,0.799389,0.797258,0.79513,0.793005,0.790883,0.788764,0.786648,0.784535,0.782426,0.780319,0.778216]
    vars_analytical = [3.27513e-7,5.21395e-6,0.0000159556,0.0000325511,0.000054998,0.0000832934,0.000117433,0.000157413,0.000203228,0.000254871,0.000312335,0.000375612,0.000444694,0.000519572,0.000600235,0.000686672,0.000778872,0.000876821,0.000980507,0.00108992,0.00120503,0.00132584,0.00145232,0.00158446,0.00172225,0.00186565,0.00201465,0.00216924,0.00232939,0.00249507,0.00266627,0.00284297,0.00302513,0.00321274,0.00340576,0.00360418,0.00380796,0.00401708,0.00423151,0.00445122,0.00467618,0.00490635,0.00514171,0.00538223,0.00562787,0.0058786,0.00613439,0.00639519,0.00666098,0.00693172,0.00720737,0.00748789,0.00777325,0.0080634,0.00835831,0.00865794,0.00896224,0.00927118,0.00958471,0.00990279,0.0102254,0.0105524,0.0108839,0.0112197,0.0115599,0.0119043,0.012253,0.0126058,0.0129628,0.0133239,0.013689,0.0140581,0.0144312,0.0148081,0.0151889,0.0155735,0.0159618,0.0163537,0.0167493,0.0171485,0.0175512,0.0179573,0.0183669,0.0187798,0.019196,0.0196154,0.020038,0.0204637,0.0208925,0.0213244,0.0217591,0.0221968,0.0226372,0.0230805,0.0235264,0.023975,0.0244262,0.0248799,0.0253361,0.0257947,0.0262556]
elif '95' in statistic:
    exvs_analytical = [1.,1.,1.,1.,0.999999,0.999989,0.999947,0.999847,0.999662,0.999376,0.998982,0.998476,0.99786,0.997139,0.996319,0.995405,0.994405,0.993325,0.992171,0.990949,0.989663,0.98832,0.986922,0.985476,0.983983,0.982448,0.980874,0.979264,0.97762,0.975944,0.974239,0.972507,0.970749,0.968968,0.967164,0.96534,0.963496,0.961635,0.959756,0.957861,0.955951,0.954026,0.952089,0.950139,0.948177,0.946204,0.944221,0.942228,0.940226,0.938215,0.936196,0.934169,0.932135,0.930094,0.928046,0.925992,0.923933,0.921868,0.919799,0.917724,0.915645,0.913562,0.911475,0.909385,0.907291,0.905194,0.903093,0.900991,0.898885,0.896778,0.894668,0.892556,0.890443,0.888327,0.886211,0.884093,0.881974,0.879854,0.877733,0.875612,0.87349,0.871367,0.869244,0.867121,0.864998,0.862875,0.860752,0.858629,0.856507,0.854385,0.852264,0.850143,0.848023,0.845904,0.843786,0.841668,0.839552,0.837437,0.835324,0.833211,0.831101]
    vars_analytical = [0.,0.,-3.33067e-16,1.18439e-11,1.9964e-9,3.87454e-8,2.72286e-7,1.0941e-6,3.12458e-6,7.12907e-6,0.0000139184,0.0000242733,0.0000389005,0.0000584139,0.000083331,0.000114079,0.000151006,0.000194387,0.000244443,0.000301342,0.000365213,0.000436151,0.000514223,0.000599476,0.000691935,0.000791612,0.000898508,0.00101261,0.00113391,0.00126236,0.00139795,0.00154064,0.00169039,0.00184716,0.0020109,0.00218157,0.00235912,0.00254351,0.00273467,0.00293256,0.00313713,0.00334833,0.0035661,0.00379039,0.00402115,0.00425832,0.00450184,0.00475167,0.00500774,0.00527001,0.00553841,0.00581289,0.0060934,0.00637988,0.00667226,0.0069705,0.00727454,0.00758433,0.00789979,0.00822088,0.00854754,0.00887971,0.00921733,0.00956034,0.00990868,0.0102623,0.0106211,0.0109851,0.0113542,0.0117283,0.0121074,0.0124914,0.0128803,0.0132739,0.0136723,0.0140753,0.0144829,0.0148951,0.0153117,0.0157328,0.0161582,0.0165879,0.0170218,0.0174598,0.017902,0.0183481,0.0187982,0.0192522,0.0197101,0.0201716,0.0206369,0.0211058,0.0215782,0.0220542,0.0225335,0.0230162,0.0235022,0.0239914,0.0244837,0.0249791,0.0254775]

exvs_analytical = np.array(exvs_analytical) * 100
stds_analytical = np.sqrt(np.array(vars_analytical)) * 100

if plot_coverage_by_error_model:
    fig = plt.figure(figsize=(4, 6))
    plt.xlabel('$\sigma_{x}$ [mm]', fontsize=14)
    plt.ylabel(r'$V_{' + statnum + '\%}$', fontsize=14)
    plt.grid(ls='--')
    # plt.title('Coverage for errors $\mathcal{N}(0, \sigma_{d})$', fontsize=14)

    # error bars
    # plt.errorbar(range_sigma_dx, meds, yerr=bars, capsize=2, color=colors[0])

    # region median + quartiles
    fig.tight_layout()
    if mode == 'mean':
        plt.plot(ss, np.array(exvs_analytical), color='royalblue', ls='-', label=r'Analytical $E[V_X]$')
        plt.plot(ss, exvs_analytical - stds_analytical, color='#5bedf5', ls='--',
                 label=r'Analytical $\pm \sigma_{V_X}$')
        plt.plot(ss, exvs_analytical + stds_analytical, color='#5bedf5', ls='--')
        plt.plot(range_sigma_dx, means, color='firebrick', label=r'Numerical $E[V_X]$')

        plt.fill_between(range_sigma_dx, stds_bars[0], stds_bars[1], color='lightcoral', label=r'Numerical $\pm \sigma_{V_X}$')

    else:
        plt.plot(range_sigma_dx, meds, color=colors[0], label='Numerical median')
        plt.fill_between(range_sigma_dx, bars[0], bars[1], color='plum', label='Numerical IQR')
    #
    plt.xlim([-0.08, 3.08])
    plt.ylim([50, 102])

    # legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.06, 0.5), fontsize=12)
    # legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.06, 0.06), fontsize=12)
    legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.5, -0.05), fontsize=14, ncol=1)
    frame = legend.get_frame()
    frame.set_edgecolor('black')

    #region mean + 2*std
    # plt.plot(range_sigma_dx, means, color=colors[0])
    # stds = stds / np.sqrt(num_samples_per_point)
    # plt.fill_between(range_sigma_dx, means - (2*stds), means + (2*stds), color='plum')
    if mode == 'mean':
        plt.savefig('./newfigs/v{}MeanStd.png'.format(statnum), dpi=300)
    else:
        plt.savefig('./newfigs/v{}MedianIQR.png'.format(statnum), dpi=300)
    plt.show()

if plot_fraction_of_adequate_treatment:
    # plt.title('Probability of adequate treatment', fontsize=14)
    plt.plot(range_sigma_dx, percents_covered / 100.)
    plt.ylabel('$p(V_{' + statnum + '\%} > ' + str(statistic_objective) + '\%)$', fontsize=14)
    plt.xlabel(r'$\sigma_{x}$ [mm]', fontsize=14)
    plt.grid(ls='--')
    plt.show()

np.save('./npydata/p{}g{}.npy'.format(statistic.split('(')[1][:2], statistic_objective), percents_covered / 100)


percent_interper = interp1d(range_sigma_dx, percents_covered, kind='slinear', bounds_error=True)
tester_sigmas = np.linspace(0.01, 2.99, 300)
tester_percents = percent_interper(tester_sigmas)
min_lookup = np.where(np.abs((tester_percents - percent_adequate_objective)) ==
                      np.amin(np.abs((tester_percents - percent_adequate_objective))))
desired_sigma = tester_sigmas[min_lookup]
print('In order to treat to V' + statnum + ' to ' + str(statistic_objective) + '% with probability 0.'
      + str(percent_adequate_objective) + ', our absolute translational positional error must be less than ' +
      str(np.round(desired_sigma[0]*2, 3)) + ' mm 95% of the time.')




