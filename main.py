import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import re


def freeze_formula(t_decay, b=4, d_frozen=50):
    # formula from previous ticket (I generally call it sqrt formula)
    in_sqrt = (25 * b ** 2 + 20 * d_frozen + 25 * t_decay ** 2 + 50 * b * t_decay)
    return np.sqrt(in_sqrt) / 5 - t_decay - b


def sqrt_formula_for_optimization(x, a, b, d_frozen=50):
    # sqrt formula, but with adjustable parameters
    t_decay = x[0] + a * x[1]
    in_sqrt = (20 * d_frozen + 25 * (t_decay + b) ** 2)
    return np.sqrt(in_sqrt) / 5 - t_decay - b


def exp_formula_for_optimization(x, a, c, d_frozen=50):
    # exponential decay formula with adjustable parameters
    t_decay = x[0] + a * x[1]
    decay_factor = np.exp(-c * t_decay)
    return decay_factor * freeze_formula(0, d_frozen=d_frozen)


def lin_function_for_optimization(x, slope0, slope1, d_frozen=50):
    # linear decay formula
    return freeze_formula(0, d_frozen=d_frozen) + slope0 * x[0] + slope1 * x[1]


def timestamp_to_float(timestamp):
    # regex magic, dw about it too much
    # timestamp layout xx:yy.zz
    # xx - minutes, yy - seconds, zz - fractional seconds
    second_string = re.search('[0-9]+\.[0-9]+', timestamp)  # finds out yy.zz
    minute_string = re.match('[0-9]+', timestamp)  # finds xx
    if second_string and minute_string:  # if both have matches
        return 60 * float(minute_string[0]) + float(second_string[0])  # convert to seconds
    else:
        print("no match")
        return 0


def freeze_time_calcs_from_raw(raw_data, array_mode="sequential", element_durability=40, ed_mode="exact"):
    # calculates time frozen (this iteration), time unfrozen (this iteration), time frozen cumulative(since last reset),
    # time unfrozen cumulative(since last reset), t_decay (with b = -2) (since last reset), element_durability of frozen
    # at the beginning of the reaction
    # raw_data: input of frozen start time, unfreeze start time, (aura application start time)
    # array_mode = "sequential" or "alternating"
    # element_durability - of freeze
    # ed_mode = "exact" or "estimate". Exact - will be determined by data. Estimate - is known beforehand or guessed

    if array_mode == "sequential":  # sequential data format: t_frozen, t_unfrozen in different columns, same rows
        df_length = raw_data.shape[0]
        freeze = raw_data['freeze start time [s]']
        unfreeze = raw_data['unfreeze time [s]']
        # calculate frozen and unfrozen duration
        duration_frozen = np.array(raw_data.iloc[:, 1] - raw_data.iloc[:, 0])
        duration_unfrozen = np.array([raw_data.iloc[i + 1, 0] - raw_data.iloc[i, 1] for i in range(0, df_length - 1)])
        # replace NaNs due to freezing before unfreezing
        for i in range(0, df_length - 1):
            if pd.isna(duration_frozen[i]):
                duration_frozen[i] = raw_data.iloc[i + 1, 0] - raw_data.iloc[i, 0]
                if freeze[i] < 0:  # negative duration is not possible
                    freeze[i] = float("NaN")
            if pd.isna(duration_unfrozen[i]) or duration_unfrozen[i] < 0:
                duration_unfrozen[i] = 0  # only matters for calcs
            if pd.isna(unfreeze.iloc[i]):
                duration_unfrozen[i] = 0
                unfreeze[i] = 0

        # append 0 at the last position (since we have one entry too few)
        duration_unfrozen = np.append(duration_unfrozen, 0)
        # initiate elemental durablility array
        ed_array = np.zeros_like(duration_frozen)

        if ed_mode == "exact":
            print("cannot determine exact element durability from data")
        for l in range(0, df_length):  # ed_mode = estimate - value given in function call
            ed_array[l] = element_durability

    if array_mode == "alternating":  # alternating data format: (aura application time), t_frozen, t_unfrozen in
        #                              same column, different rows. Definitely the most "spaghetti" part (but it works).
        #                              HERE BE DRAGONS

        # match element_durability:
        if ed_mode == "exact":
            if element_durability == 40:
                decay_time = 9.5
            elif element_durability == 80:
                decay_time = 12
            elif element_durability == 112.5:
                decay_time = 7.6  # swirl
            else:
                print("warning: invalid element durability")
                decay_time = 1
        else:
            decay_time = 1

        df_length = raw_data[raw_data['status'] == "frozen"].shape[0]  # length of new data frame
        # initialize arrays
        ed_array = np.zeros(df_length)  # elemental durablility
        ed_time = np.zeros(df_length)  # time at which aura is applied
        freeze = np.zeros(df_length)  # freeze start time
        unfreeze = np.zeros(df_length)  # unfreeze start time
        duration_frozen = np.zeros(df_length)  # freeze duration
        duration_unfrozen = np.zeros(df_length)  # unfreeze duration

        # k - counter for refreezes before unfreezing, l - counter for new (sequential) dataframe
        k = 0
        l = 0
        for i in range(0, raw_data.shape[0]):
            if raw_data.status.isin(("wet", "cryo"))[i]:  # aura application
                j = 1
                while j < raw_data.shape[0] - i:  # j lower than remaining rows
                    # frozen aura is determined from element aura at time of the reaction
                    if raw_data.loc[i + j, 'status'] == "frozen":
                        if ed_mode == "exact":  # determine ed from data
                            ed_time[l] = raw_data.loc[i + j, 'time'] - raw_data.loc[i, 'time']
                            ed_array[l] = max(element_durability * (1 - ed_time[l] / decay_time), 0)
                        else:  # ed_mode = estimate
                            ed_array[l] = element_durability  # take the value provided in function call
                        j = raw_data.shape[0]  # end loop kek
                    j += 1

            elif raw_data.iloc[i, 1] == "frozen":  # frozen reaction
                freeze[l] = raw_data.loc[i, 'time']  # freeze start time
                j = 1
                while j < raw_data.shape[0] - i:  # j lower than remaining rows
                    if raw_data.status.isin(("frozen", "unfrozen"))[i + j]:  # freeze ends when unfrozen or refrozen
                        duration_frozen[l] = raw_data.loc[i + j, 'time'] - raw_data.loc[i, 'time']
                        if raw_data.iloc[i + j, 1] == "frozen":
                            k += 1  # 2 frozen reactions with no unfreeze in between
                        j = raw_data.shape[0]  # end for loop
                    else:
                        j += 1
                if raw_data.shape[0] - 2 > i:  # i lower than remaining rows - 1
                    if raw_data.status.isin(("wet", "cryo"))[i + 1]:  # aura app before unfreeze
                        if raw_data.loc[i + 2, 'status'] == "frozen":
                            l += 1
                    if raw_data.loc[i + 1, 'status'] == "frozen":
                        l += 1

            elif raw_data.loc[i, 'status'] == "unfrozen":  # unfreeze
                unfreeze[l] = raw_data.loc[i, 'time']
                j = 1
                while j < raw_data.shape[0] - i:  # j lower than remaining rows
                    if raw_data.status.isin(("frozen", "unfrozen"))[i + j]:  # unfreeze end time
                        duration_unfrozen[l] = raw_data.loc[i + j, 'time'] - raw_data.loc[i, 'time']
                        j = raw_data.shape[0]  # end for loop
                    else:
                        j += 1
                if raw_data.shape[0] - 1 > i > 0: # i between 0 and rows - 1
                    if raw_data.status.isin(("wet", "cryo", "frozen"))[i + 1]:
                        l += 1

    # freeze decay reset calcs
    # freeze decay resets for sure when unfrozen for longer than 2 s
    freeze_decay_reset = np.array([(duration_unfrozen[k]) >= 2 or duration_frozen[k] <= 0 or pd.isna(duration_frozen[k])
                                   for k in range(0, df_length)], dtype=bool)
    # initialize arrays and variables
    time_frozen_cumulative = np.zeros_like(duration_unfrozen)  # since last reset
    time_unfrozen_cumulative = np.zeros_like(duration_unfrozen)  # since last reset
    time_decay = np.zeros_like(duration_unfrozen)  # since last reset
    sum_freeze = 0
    sum_unfreeze = 0

    # calculate cumulative frozen  and unfrozen time since last freeze decay reset
    for i in range(0, df_length - 1):
        if duration_frozen[i] < 0:  # duration cannot be negative (unless it's the next video logged)
            freeze_decay_reset[i] = True
            duration_frozen[i] = 0
            time_decay[i + 1] = 0
        if freeze_decay_reset[i - 1]:
            # reset means reset
            sum_freeze = 0
            sum_unfreeze = 0
        sum_freeze += duration_frozen[i]  # add this iterations' frozen time
        sum_unfreeze += duration_unfrozen[i]  # add this iteration's unfrozen time
        time_frozen_cumulative[i + 1] = sum_freeze  # write to array
        time_unfrozen_cumulative[i + 1] = sum_unfreeze  # write to array
        time_decay[i + 1] = sum_freeze - 2 * sum_unfreeze  # calculate t decay

        # calc if this iteration did reset freeze decay
        if sum_freeze - 2 * sum_unfreeze <= 0:
            freeze_decay_reset[i] = True
            time_decay[i + 1] = 0



    # put all that in dataframe
    freeze_dataframe = pd.DataFrame({'freeze': freeze,  # freeze start time (provided in function call)
                                     'unfreeze': unfreeze,  # freeze stop time (provided in function call)
                                     't_frozen': duration_frozen,  # frozen time in this iteration
                                     't_unfrozen': duration_unfrozen,  # unfrozen time in this iteration
                                     't_frozen_cumulative': time_frozen_cumulative,  # cumu frozen time since reset
                                     't_unfrozen_cumulative': time_unfrozen_cumulative,  # cumu unfrzn time since reset
                                     't_decay': time_decay,  # decay time
                                     'ed_estimate': ed_array  # elemental durability of freeze aura
                                     })
    freeze_dataframe = freeze_dataframe[pd.to_numeric(freeze_dataframe['freeze'], errors='coerce').notnull()]
    return freeze_dataframe


def fitting(data_series, ed_mode_func="exact", ed_estimate_func=40):
    # function does sqrt, exp and lin fits using scipy.optimize.curve_fit
    # cumulative frozen time and cumulative unfrozen time are considered as separate variables
    if ed_mode_func == "exact":
        # exact means determined from data
        element_durability = data_series.loc[:, "ed_estimate"]
    else:
        # estimate means given through knowledge about game (e.g. ed of rain)
        element_durability = ed_estimate_func

    # necessary to use temp functions to pass d_frozen to underlying function
    def temp_func_sqrt(x, a, b):
        return sqrt_formula_for_optimization(x, a, b, d_frozen=element_durability)

    def temp_func_exp(x, a, c):
        return exp_formula_for_optimization(x, a, c, d_frozen=element_durability)

    def temp_func_lin(x, a1, a2):
        return lin_function_for_optimization(x, a1, a2, d_frozen=element_durability)

    # convert to numpy for optimization
    data_y = data_series.loc[:, 't_frozen'].to_numpy()
    data_x = np.transpose(data_series.loc[:, 't_frozen_cumulative':'t_unfrozen_cumulative'].to_numpy())
    # fit to sqrt curve
    p_sqrt_temp = curve_fit(temp_func_sqrt, data_x, data_y, p0=(-2, 4))
    # bounds=([-5, 2.0], [-0.001, 15])
    parameters_sqrt_temp = p_sqrt_temp[0]
    var_sqrt_temp = np.diag(p_sqrt_temp[1])
    # fit to exp curve
    p_exp_temp = curve_fit(temp_func_exp, data_x, data_y, p0=(-2, 0.25), bounds=([-4, 0.001], [-0.1, 1]))
    parameters_exp_temp = p_exp_temp[0]
    var_exp_temp = np.diag(p_exp_temp[1])
    # fit to linear curve
    p_lin_temp = curve_fit(temp_func_lin, data_x, data_y, p0=(-1, 1), bounds=([-4, -10], [4, 10]))
    parameters_lin_temp = p_lin_temp[0]
    var_lin_temp = np.diag(p_lin_temp[1])

    return [[parameters_sqrt_temp, var_sqrt_temp],
            [parameters_exp_temp, var_exp_temp],
            [parameters_lin_temp, var_lin_temp]]


### Main code ###

### Read in data ###
# (ellimiku and phaZ)
freeze_data_ellimiku_raw = pd.read_csv("Freeze_extensions_ElliMiku.csv", header=0, usecols=[2, 3])
freeze_data_phaZ_raw = pd.read_csv("refreeze_test2.csv", skiprows=1, usecols=[2, 3], nrows=11)
freeze_data_phaZ_raw.columns = ("freeze start time [s]", "unfreeze time [s]")
for j in range(0, freeze_data_phaZ_raw.shape[0]):
    freeze_data_phaZ_raw.loc[j, "freeze start time [s]"] = freeze_data_phaZ_raw.loc[j, "freeze start time [s]"] / 60
    freeze_data_phaZ_raw.loc[j, "unfreeze time [s]"] = freeze_data_phaZ_raw.loc[j, "unfreeze time [s]"] / 60

# do calculations
filter_cutoff = 7

freeze_data_ellimiku = freeze_time_calcs_from_raw(freeze_data_ellimiku_raw, array_mode="sequential", ed_mode="estimate",
                                                  element_durability=50)
freeze_data_ellimiku_filtered = freeze_data_ellimiku[freeze_data_ellimiku['t_unfrozen_cumulative'] < filter_cutoff]

freeze_data_phaZ = freeze_time_calcs_from_raw(freeze_data_phaZ_raw, array_mode="sequential", ed_mode="estimate",
                                              element_durability=112.5)
freeze_data_phaZ_filtered = freeze_data_phaZ[freeze_data_phaZ['t_unfrozen_cumulative'] < filter_cutoff]

# read v1, v2, v3
freeze_data_v1_raw = pd.read_csv("Genshin_Freeze_Extension_v1.csv", header=1, skiprows=1, usecols=[3, 4])
# freeze_data_v2_raw = pd.read_csv("Genshin_Freeze_Extension_v2.csv", header=1, skiprows=1, usecols=[3, 4])
freeze_data_v3_raw = pd.read_csv("Genshin_Freeze_Extension_v3.csv", header=1, skiprows=1, usecols=[0, 1])

freeze_data_dfs = []
ed_estimate = np.array([50, 112.5, 40, 40, 50, 50, 50])
ed_modes = ["estimate", "estimate", "exact", "exact", "estimate", "estimate"]

j = 1
number = 6
for raw_data in (freeze_data_v1_raw, freeze_data_v3_raw):
    for k in range(0, raw_data.shape[0]):
        raw_data.iloc[k, 0] = timestamp_to_float(raw_data.iloc[k, 0])
    result = freeze_time_calcs_from_raw(raw_data, array_mode="alternating", ed_mode=ed_modes[j + 1])
    result_filtered = result[result['t_unfrozen_cumulative'] < filter_cutoff]
    freeze_data_dfs.append(result_filtered)
    j += 1

# isu data need some processing before I can do calcs with them - a bit messy here
freeze_data_isu_raw = pd.read_csv("Freeze_extensions_isu.csv", header=1, skiprows=10, usecols=[2, 3, 4, 5])
freeze_data_isu_raw_numpy = freeze_data_isu_raw.iloc[0:10, :].to_numpy()  # for reshaping in the next line
freeze_data_isu_raw_numpy = np.reshape(freeze_data_isu_raw_numpy, (20, 2))  # get data in the "sequential" format
freeze_data_isu_raw_numpy = freeze_data_isu_raw_numpy / 60  # divide through fps
# make a dataframe again for function
freeze_data_isu_raw_reshaped = pd.DataFrame(freeze_data_isu_raw_numpy, columns=("freeze start time [s]",
                                                                                "unfreeze time [s]"))
# initialize data frames
freeze_data_isu = pd.DataFrame(np.zeros((20, 8)), columns=('freeze', 'unfreeze', 't_frozen', 't_unfrozen',
                                                           't_frozen_cumulative', 't_unfrozen_cumulative', 't_decay',
                                                           'ed_estimate'))
partial_data = pd.DataFrame(np.zeros((2, 8)))

# videos edited together - have to split up data bc the delay is less than 2 secs in the vid between experiments
for i in range(0, 10):
    index = 2 * i
    raw_partial = freeze_data_isu_raw_reshaped.iloc[index:index + 2, :]
    partial_data = freeze_time_calcs_from_raw(raw_partial, array_mode="sequential", ed_mode="estimate",
                                              element_durability=50)
    freeze_data_isu.iloc[index:index + 2, :] = partial_data

# aloy data recorded by phaZ
freeze_data_aloy_raw = pd.read_csv("freeze_extension_aloy.csv", skiprows=1, usecols=[3, 4], nrows=76)
freeze_data_aloy_raw.columns = ("freeze start time [s]", "unfreeze time [s]")
for j in range(0, freeze_data_aloy_raw.shape[0]):
    freeze_data_aloy_raw.loc[j, "freeze start time [s]"] = freeze_data_aloy_raw.loc[j, "freeze start time [s]"] / 60
    freeze_data_aloy_raw.loc[j, "unfreeze time [s]"] = freeze_data_aloy_raw.loc[j, "unfreeze time [s]"] / 60

freeze_data_aloy = freeze_time_calcs_from_raw(freeze_data_aloy_raw, array_mode="sequential", ed_mode="estimate",
                                               element_durability=50)
freeze_data_aloy_filtered = freeze_data_aloy[freeze_data_aloy['t_unfrozen_cumulative'] < filter_cutoff]

### fitting ###

parameters_sqrt = []
var_sqrt = []
parameters_exp = []
var_exp = []
parameters_lin = []
var_lin = []

j = 0
labels = ("ellimiku", "phaZ - swirl", "v1", "v3", "isu", "phaZ - aloy")
for series in (freeze_data_ellimiku_filtered, freeze_data_phaZ, *freeze_data_dfs, freeze_data_isu, freeze_data_aloy):
    fit_results = fitting(series, ed_mode_func=ed_modes[j], ed_estimate_func=ed_estimate[j])
    # [i][k]   i = 0 corresponds to sqrt
    #          i = 1 corresponds to exp
    #          i = 2 corresponds to lin
    #          k = 0 corresponds to parameters
    #          k = 1 corresponds to said parameters variance
    parameters_sqrt.append(fit_results[0][0])
    var_sqrt.append(fit_results[0][1])
    parameters_exp.append(fit_results[1][0])
    var_exp.append(fit_results[1][1])
    parameters_lin.append(fit_results[2][0])
    var_lin.append(fit_results[2][1])
    # print results
    # python string formatting (ab)use: a +/- a_strerr, b +/- b_sterr_b
    print("Sqrt parameters {0}: a = {1:.2f} \u00B1 {3:.2f} , b = {2:.2f} \u00B1 {4:.2f}"
          .format(labels[j], *parameters_sqrt[j], *np.sqrt(var_sqrt[j])))
    print("Exp parameters {0}: a = {1:.2f} \u00B1 {3:.2f} , b = {2:.2f} \u00B1 {4:.2f}"
          .format(labels[j], *parameters_exp[j], *np.sqrt(var_exp[j])))
    print("Linear parameters {0}: a = {1:.2f} \u00B1 {3:.2f} , b = {2:.2f} \u00B1 {4:.2f}"
          .format(labels[j], *parameters_lin[j], *np.sqrt(var_lin[j])))
    print("##############################################################")
    j += 1

# inverse variance weighting
# sqrt
weights_sqrt = np.array([[1 / var_sqrt[j][i] for i in range(0, 2)] for j in range(0, number)])
best_parameters_sqrt = np.zeros((2))
var_best_parameters_sqrt = np.zeros((2))
# exp
weights_exp = np.array([[1 / var_exp[j][i] for i in range(0, 2)] for j in range(0, number)])
best_parameters_exp = np.zeros((2))
var_best_parameters_exp = np.zeros((2))
# linear
weights_lin = np.array([[1 / var_lin[j][i] for i in range(0, 2)] for j in range(0, number)])
best_parameters_lin = np.zeros((2))
var_best_parameters_lin = np.zeros((2))

# calculate weighted average and corresponding variance
for i in range(0, 2):
    best_parameters_sqrt[i] = np.average(np.array(parameters_sqrt)[:, i], axis=0, weights=weights_sqrt[:, i])
    var_best_parameters_sqrt[i] = 1 / np.sum(weights_sqrt[:, i])
    best_parameters_exp[i] = np.average(np.array(parameters_exp)[:, i], axis=0, weights=weights_exp[:, i])
    var_best_parameters_exp[i] = 1 / np.sum(weights_exp[:, i])
    best_parameters_lin[i] = np.average(np.array(parameters_lin)[:, i], axis=0, weights=weights_exp[:, i])
    var_best_parameters_lin[i] = 1 / np.sum(weights_lin[:, i])

# python string formatting (ab)use: a +/- a_strerr, b +/- b_sterr_b
print("Best overall sqrt parameters: a = {0:.2f} \u00B1 {2:.2f} , b = {1:.2f} \u00B1 {3:.2f}"
      .format(*best_parameters_sqrt, *np.sqrt(var_best_parameters_sqrt)))
print("Best overall exp parameters: a = {0:.2f} \u00B1 {2:.2f} , b = {1:.2f} \u00B1 {3:.2f}"
      .format(*best_parameters_exp, *np.sqrt(var_best_parameters_exp)))
print("Best overall linear parameters: a = {0:.2f} \u00B1 {2:.2f} , b = {1:.2f} \u00B1 {3:.2f}"
      .format(*best_parameters_lin, *np.sqrt(var_best_parameters_lin)))

# prepare arrays for plots of the fitted functions
t_decay_plot = []
t_frozen = []
t_unfrozen = []
x_for_calcs = []
array_params_t_decay = []
array_vars_decay = []
j = 0
for data_series in (freeze_data_ellimiku_filtered, freeze_data_phaZ, *freeze_data_dfs, freeze_data_isu,
                    freeze_data_aloy):

    t_frozen_max = 2 * data_series.max().loc["t_frozen_cumulative"]  # twice the max so we have large enough t_decay
    # find way to combine variables 't_frozen_cumulative' and 't_unfrozen cumulative' into one ('t_decay')
    # linregress of actual data
    params_lin_t_decay = linregress(data_series.loc[:, 't_frozen_cumulative'],
                                    data_series.loc[:, 't_unfrozen_cumulative'])
    array_params_t_decay.append(np.array([params_lin_t_decay.intercept, params_lin_t_decay.slope]))
    array_vars_decay.append(np.array([params_lin_t_decay.intercept_stderr, params_lin_t_decay.stderr]))
    # make t_unfrozen_cumulative a function of t_frozen_cumulative (based on linregress data)
    t_unfrozen_min = params_lin_t_decay.intercept
    t_unfrozen_max = params_lin_t_decay.intercept + params_lin_t_decay.slope * t_frozen_max
    t_frozen.append(np.linspace(0, t_frozen_max, 100))
    t_unfrozen.append(np.linspace(t_unfrozen_min, t_unfrozen_max, 100))
    # t_decay = t_frozen_cumulative + a * t_unfrozen_cumulative
    factor_a = parameters_sqrt[j][0]
    t_decay_plot.append(np.linspace(0, t_frozen_max + factor_a * t_unfrozen_max, 100))
    x_for_calcs.append(np.array([t_frozen[-1], t_unfrozen[-1]]))
    j += 1

y_ranges_fit_sqrt = []
y_ranges_best_fit_sqrt = []
y_ranges_fit_exp = []
y_ranges_best_fit_exp = []
y_ranges_fit_lin = []
y_ranges_best_fit_lin = []

for i in range(0, number):
    # generate points to plot the functions
    y_ranges_fit_sqrt.append(sqrt_formula_for_optimization(x_for_calcs[i], *parameters_sqrt[i],
                                                           d_frozen=ed_estimate[i]))
    y_ranges_best_fit_sqrt.append(sqrt_formula_for_optimization(x_for_calcs[i], *best_parameters_sqrt,
                                                                d_frozen=ed_estimate[i]))
    y_ranges_fit_exp.append(exp_formula_for_optimization(x_for_calcs[i], *parameters_exp[i],
                                                         d_frozen=ed_estimate[i]))
    y_ranges_best_fit_exp.append(exp_formula_for_optimization(x_for_calcs[i], *best_parameters_exp,
                                                              d_frozen=ed_estimate[i]))
    y_ranges_fit_lin.append(lin_function_for_optimization(x_for_calcs[i], *parameters_lin[i],
                                                          d_frozen=ed_estimate[i]))
    y_ranges_best_fit_lin.append(lin_function_for_optimization(x_for_calcs[i], *best_parameters_lin,
                                                               d_frozen=ed_estimate[i]))

# weighted avg for t_unfrozen_cumulative as a fcn of t_frozen_cumulative (ONLY for plotting)
weights_t_unfrozen_slope = np.array([1 / array_vars_decay[j][0] for j in range(0, number)])
# weights_t_unfrozen_intercept = np.array([1 / array_vars_decay[j] for j in range(0, number)])
slopes = np.array([array_params_t_decay[i][1] for i in range(0, number)])
slope_for_plot = np.average(slopes, axis=0, weights=weights_t_unfrozen_slope)
# generate lines corresponding to formula
t_frozen_best = np.linspace(0, 50, 100)
t_unfrozen_best = np.linspace(0, slope_for_plot * 50, 100)
a_plots = best_parameters_sqrt[0]
x_range = t_frozen_best + a_plots * t_unfrozen_best
# y_range = freeze_formula(x_range)  # y coordinate aka freeze time
# y_range_aura = freeze_formula(x_range, d_frozen=40)  # with aura tax taken into account
# y_range_2b_aura = freeze_formula(x_range, d_frozen=80)  # and with 2B and aura tax
y_range_sqrt_1A = sqrt_formula_for_optimization(np.array([t_frozen_best, t_unfrozen_best]), *best_parameters_sqrt,
                                                d_frozen=50)

y_range_sqrt_1A_aura = sqrt_formula_for_optimization(np.array([t_frozen_best, t_unfrozen_best]), *best_parameters_sqrt,
                                                     d_frozen=40)
y_range_sqrt_2B_aura = sqrt_formula_for_optimization(np.array([t_frozen_best, t_unfrozen_best]), *best_parameters_sqrt,
                                                     d_frozen=80)
y_range_sqrt_swirl_aura = sqrt_formula_for_optimization(np.array([t_frozen_best, t_unfrozen_best]),
                                                        *best_parameters_sqrt, d_frozen=112.4)
y_range_exp_1A = exp_formula_for_optimization(np.array([t_frozen_best, t_unfrozen_best]), *best_parameters_exp,
                                              d_frozen=50)
y_range_exp_1A_aura = exp_formula_for_optimization(np.array([t_frozen_best, t_unfrozen_best]), *best_parameters_exp,
                                                   d_frozen=40)
y_range_lin_1A = lin_function_for_optimization(np.array([t_frozen_best, t_unfrozen_best]), *best_parameters_lin,
                                               d_frozen=50)
y_range_lin_1A_aura = lin_function_for_optimization(np.array([t_frozen_best, t_unfrozen_best]), *best_parameters_lin,
                                                    d_frozen=40)


### Plots ###
aura_arr = ["1A", "1A"]
colors_arr = ["#4424D3", "#2CC259", "#707070", "#000000", "#2A96AB", "#C55EC3"]
fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)

# plot functions
# ax1
ax1.plot(x_range, y_range_sqrt_1A, label="Sqrt formula fit - 1A w/o aura tax", color="#EF711A", linewidth=0.5)
ax1.plot(x_range, y_range_exp_1A, label="Exp formula fit - 1A w/o aura tax", color="#202C8A", linewidth=0.5)
ax1.plot(x_range, y_range_lin_1A, label="Linear formula fit - 1A w/o aura tax", color="#27732D", linewidth=0.5)
# ax2
ax2.plot(x_range, y_range_sqrt_1A_aura, label="Sqrt formula fit - 1A with aura tax", color="#FA9F16", linewidth=0.5)
ax2.plot(x_range, y_range_sqrt_1A, label="Sqrt formula fit - 1A w/o aura tax", color="#EF711A", linewidth=0.5)
ax2.plot(x_range, y_range_sqrt_swirl_aura, label="Sqrt formula fit - swirl tests", color="#E4421E", linewidth=0.5)

# plot data
for ax in (ax1, ax2):
    ax.set_ylim((0, 5))
    ax.set_xlim((-0.1, 10.5))
    ax.plot(freeze_data_ellimiku_filtered['t_decay'], freeze_data_ellimiku_filtered['t_frozen'], linestyle="none",
            marker=".", markersize=1.2,
            label="ellimiku data - enemy in water", color=colors_arr[0])
    ax.plot(freeze_data_phaZ['t_decay'], freeze_data_phaZ['t_frozen'], linestyle="none", marker=".", markersize=1.2,
            label="phaZ data - swirl tests", color=colors_arr[1])
    j = 0
    for v in range(0, 2):
        ax.plot(freeze_data_dfs[v]['t_decay'], freeze_data_dfs[v]['t_frozen'], linestyle="none", marker=".", markersize=1.2,
                label="v{0} data - {1}".format(v, aura_arr[j]), color=colors_arr[j + 2])
        j += 1
    ax.plot(freeze_data_isu['t_decay'], freeze_data_isu['t_frozen'], linestyle="none",
            marker=".", markersize=1.2,
            label="isu data - enemy in water", color=colors_arr[4])

    ax.plot(freeze_data_aloy['t_decay'], freeze_data_aloy['t_frozen'], linestyle="none", marker=".", markersize=1.2,
            label="phaZ data - enemy in water", color=colors_arr[5])

    ax.set_xlabel("t_decay")
    ax.set_ylabel("time frozen")
i = 0
for figure in (fig1, fig2):
    plt.figure(figure.number)
    plt.legend(fontsize=8)
    plt.title("Freeze time vs t_decay")
    plt.savefig("t_frozen_vs_t_decay_{0}.png".format(i))
    i += 1


plt.show()
plt.close(fig1)
plt.close(fig2)

data_series_plot = (freeze_data_ellimiku_filtered, freeze_data_phaZ, *freeze_data_dfs, freeze_data_isu,
                    freeze_data_aloy)
labels = ("ellimiku", "phaZ_swirl", "v1", "v3", "isu", "phaZ_aloy")


for k in range(0, number):
    fig, ax = plt.subplots(1, 1)
    t_decay_data = data_series_plot[k]["t_frozen_cumulative"] + parameters_sqrt[k][0] * data_series_plot[k]["t_unfrozen_cumulative"]
    ax.plot(t_decay_data, data_series_plot[k]['t_frozen'], linestyle="none", marker=".", markersize=1.2,
            label="data", color="#707070")
    ax.plot(t_decay_plot[k], y_ranges_fit_sqrt[k], linewidth=0.5, label="individual fit sqrt", color="#FAD816")
    ax.plot(t_decay_plot[k], y_ranges_best_fit_sqrt[k], linewidth=0.5, label="best fit sqrt", color="#FA9F16")
    ax.plot(t_decay_plot[k], y_ranges_fit_exp[k], linewidth=0.5, label="individual fit exp", color="#9EA8F7")
    ax.plot(t_decay_plot[k], y_ranges_best_fit_exp[k], linewidth=0.5, label="best fit exp", color="#5968DA")
    ax.plot(t_decay_plot[k], y_ranges_fit_lin[k], linewidth=0.5, label="individual fit lin", color="#8ED89C")
    ax.plot(t_decay_plot[k], y_ranges_best_fit_lin[k], linewidth=0.5, label="best fit lin", color="#34A656")
    # set sensible y limits
    ax.set_ylim(0, data_series_plot[k].max().iloc[2] + 1)
    ax.set_xlim(0, t_decay_data.max() + 1.5)
    plt.legend(loc="upper right")
    plt.title(labels[k])
    plt.savefig("{0}.png".format(labels[k]))
    plt.show()
    plt.close()

# for k in range(0, number):
#     fig, ax = plt.subplots(1, 1)
#     t_decay_data = data_series_plot[k]["t_frozen_cumulative"] + parameters_sqrt[k][0] * data_series_plot[k][
#         "t_unfrozen_cumulative"]
#     y_data = data_series_plot[k]['t_frozen']
#     x_for_func = np.transpose(data_series_plot[k].loc[:, "t_frozen_cumulative":"t_unfrozen_cumulative"].to_numpy())
#     ed = data_series_plot[k].loc[:, "ed_estimate"]
#     y_sqrt_ind = sqrt_formula_for_optimization(x_for_func, *parameters_sqrt[k], d_frozen=ed)
#     y_sqrt_best = sqrt_formula_for_optimization(x_for_func, *best_parameters_sqrt, d_frozen=ed)
#
#     ax.plot(t_decay_data, data_series_plot[k]['t_frozen'], linestyle="none", marker=".", markersize=1.2,
#             label="data", color="#707070")
#     ax.plot(t_decay_data, y_sqrt_ind, linestyle="None", marker=".", markersize=1.2, label="individual fit sqrt",
#             color="#FAD816")
#
#     ax.plot(t_decay_data, y_sqrt_best, linestyle="None", marker=".", markersize=1.2, label="best fit sqrt",
#             color="#FA9F16")
#
#     ax.plot(t_decay_plot[k], y_ranges_fit_sqrt[k], linewidth=0.5, color="#FAD816")
#     ax.plot(t_decay_plot[k], y_ranges_best_fit_sqrt[k], linewidth=0.5, color="#FA9F16")
#     # ax.plot(t_decay_plot[k], y_ranges_best_fit_sqrt[k], linewidth=0.5, label="best fit sqrt", color="#FA9F16")
#     # ax.plot(t_decay_plot[k], y_ranges_fit_exp[k], linewidth=0.5, label="individual fit exp", color="#9EA8F7")
#     # ax.plot(t_decay_plot[k], y_ranges_best_fit_exp[k], linewidth=0.5, label="best fit exp", color="#5968DA")
#     # ax.plot(t_decay_plot[k], y_ranges_fit_lin[k], linewidth=0.5, label="individual fit lin", color="#8ED89C")
#     # ax.plot(t_decay_plot[k], y_ranges_best_fit_lin[k], linewidth=0.5, label="best fit lin", color="#34A656")
#     # set sensible y limits
#     ax.set_ylim(0, data_series_plot[k].max().iloc[2] + 1)
#     ax.set_xlim(0, t_decay_data.max() + 1.5)
#     plt.legend(loc="upper right")
#     plt.title(labels[k])
#     plt.savefig("{0}_1.png".format(labels[k]))
#     #plt.show()
#     plt.close()


### Save data ###
freeze_data_ellimiku.to_csv("freeze_data_ellimiku.csv")
