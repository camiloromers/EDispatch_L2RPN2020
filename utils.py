import pandas as pd
import numpy as np
import calendar
import os
import sys
import pypsa
from datetime import datetime, timedelta

# def format_dates(year, month):
#     # Get last of for every month
#     end_day_month = calendar.monthrange(year, month)[1]
#     from_date = f'{year}-{month}-01 00:00'
#     end_date = f'{year}-{month}-{end_day_month} 23:55'
#     # Convert them to datetime
#     FROM_DATE = datetime.strptime(from_date, '%Y-%m-%d %H:%M')
#     END_DATE = datetime.strptime(end_date, '%Y-%m-%d %H:%M')
#     return FROM_DATE, END_DATE

def reformat_gen_constraints(gen_constraints, rescaled_min, snapshots):
    for k, df in gen_constraints.items():
        if df is None:
            # Create an empty df only with index
            gen_constraints[k] = pd.DataFrame(index=snapshots)
        else:
            # Reindex df and resample according to desired mins
            gen_constraints[k].index = snapshots
            gen_constraints[k] = gen_constraints[k].resample(f'{str(rescaled_min)}min').apply(lambda x: x[0])
    return gen_constraints


# def import_data(data_path, from_date, end_date, every_min):
#     dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#     load = pd.read_csv(os.path.join(data_path, 'load_' + str(from_date.year) + '.csv.bz2'), 
#                        parse_dates=['datetime'], date_parser=dateparse)
#     solar = pd.read_csv(os.path.join(data_path, 'selected_solar.csv.bz2'), 
#                         parse_dates=['datetime'], date_parser=dateparse)
#     wind = pd.read_csv(os.path.join(data_path, 'selected_wind.csv.bz2'), 
#                        parse_dates=['datetime'], date_parser=dateparse)
#     # Set corresponding index
#     load.set_index('datetime', inplace=True)
#     solar.set_index('datetime', inplace=True)
#     wind.set_index('datetime', inplace=True)
#     # Truncate according to max dates
#     load_rscl = load.loc[from_date : end_date].resample(f'{str(every_min)}min').apply(lambda x: x[0])
#     solar_rscl = solar.loc[from_date : end_date].resample(f'{str(every_min)}min').apply(lambda x: x[0])
#     wind_rscl = wind.loc[from_date : end_date].resample(f'{str(every_min)}min').apply(lambda x: x[0])
#     load = load.loc[from_date : end_date]
#     snapshots = load.index
#     return wind_rscl, solar_rscl, load_rscl, snapshots

def rescale_gen_param(net, every_min, grid_params=5):
    # Adapt ramps according to the time 
    steps = every_min / grid_params
    net.generators.loc[:, ['ramp_limit_up', 'ramp_limit_down']] *= steps
    return net

def fill_constraints_grid(net, snapshots, trunc_load, trunc_gen_constraints):
    # Reset any previous value save it in the grid
    net.loads_t.p_set = net.loads_t.p_set.iloc[0:0, 0:0]
    net.generators_t.p_max_pu = net.generators_t.p_max_pu.iloc[0:0, 0:0]
    # Set the snapshots
    net.set_snapshots(snapshots)
    # partial_load, wind, solar = partial_load, wind.loc[snapshots], solar.loc[snapshots]
    # Set loads
    net.loads_t.p_set = pd.concat([trunc_load])
    # Set truncated gen constraints
    net.generators_t.p_max_pu = pd.concat([trunc_gen_constraints['p_max_pu']], axis=1)
    net.generators_t.p_min_pu = pd.concat([trunc_gen_constraints['p_min_pu']], axis=1)
    # Constrain nuclear power plants
    nuclear_names = net.generators[net.generators.carrier == 'nuclear'].index.tolist()
    for c in nuclear_names:
        net.generators_t.p_max_pu[c] = 1.
        net.generators_t.p_min_pu[c] = 0.4
    return net

def run_unit_commitment(net, mode, demand, gen_constraints):
    # Show info when running opf
    to_disp = {'day': demand.index.day.unique().values[0],
               'week': demand.index.week.unique().values[0],
               'month': demand.index.month.unique().values[0],
    }
    m_period = demand.index.month.unique().values[0]
    print(f'\n--> OPF single formulation by: {mode} - Analyzing {mode} # {to_disp[mode]} of month {m_period}')
    # Get new snapshots and set them up
    snapshots = demand.index
    # Truncate gen constraints
    trunc_gen_const = gen_constraints.copy()
    for k, df in trunc_gen_const.items():
        trunc_gen_const[k] = df.loc[snapshots] 
    # Prepare grid for OPF
    net = fill_constraints_grid(net, 
                                snapshots, 
                                demand, 
                                trunc_gen_const,
                               )
    # Run Linear OPF
    rel = net.lopf(net.snapshots, pyomo=False, solver_name='cbc')
    if rel[1] != 'optimal': 
        print ('** OPF failed to find a solution **')
        sys.exit()
    # Get the values
    dispatch = net.generators_t.p.copy()
    return dispatch

# def interpolate(df, ref_index=None, method='cubic'):
#     # Create dataframe with full index
#     dff = pd.DataFrame(index=ref_index, columns=df.columns)
#     # Replace values in full dataframe
#     dff.loc[df.index, :] = df
#     # Convert them to numeric columns
#     for col in dff:
#         dff[col] = pd.to_numeric(dff[col], errors='coerce')
#     interpolated_df = dff.interpolate(method=method, axis=0)
#     # Force to put zero for very samell values
#     criteria_small_value = 1e-4
#     interpolated_df[interpolated_df < criteria_small_value] = 0
#     return interpolated_df.round(2)

def add_noise_gen(df, gen_cap, noise_factor=None):
    # Get range of value per columns in df
    # stats = df.agg(['max', 'min'], axis=0)
    # range_ = stats.loc['max'] - stats.loc['min']
    variance_per_col = gen_cap * noise_factor
    for col in df:
        # Check for values greater than zero 
        # (means unit has been distpached)
        only_dispatched_steps = df[col][df[col] > 0]
        noise = np.random.normal(0, variance_per_col.loc[col], only_dispatched_steps.shape[0])
        df.loc[only_dispatched_steps.index, col] = only_dispatched_steps + noise
    return df.round(2)

# def generate_prod_voltage(vol, vol_var=1.2, ref_index=None):
#     fill_vol = pd.concat([vol] * ref_index.shape[0], axis=1)
#     prod_v = fill_vol.T
#     prod_v.index = ref_index
#     for col in prod_v:
#         noise = np.random.normal(0, vol_var, prod_v[col].shape[0])
#         prod_v[col] +=noise
#     return prod_v.round(2)

# def generate_reactive_loads(load_p, min_range=12, max_range=25):
#     # Percentage of reactive power 
#     load_q = load_p.copy()
#     q_percen = np.random.randint(min_range, max_range, load_p.shape[1])
#     for i, col in enumerate(load_q):
#         load_q[col] *= q_percen[i] / 100
#     return load_q.round(2)

# def generate_hazard_maintenance(lines_names=None, ref_index=None):
#     hazards = pd.DataFrame(0, index=ref_index, columns=lines_names)
#     maintenance = pd.DataFrame(0, index=ref_index, columns=lines_names)
#     return hazards, maintenance

# def add_noise_forecast(df, noise_factor=0.1):
#     vars_per_col = df.mean(axis=0) * noise_factor
#     for col in df:
#         noise = np.random.normal(0, vars_per_col.loc[col], df.shape[0])
#         df[col] += noise
#     return df.round(2)

# def generate_forecasts(load_p, load_q, prod_p, prod_v, maintenance):
#     load_p_f = load_p.apply(np.roll, shift=1)
#     load_p_ff = add_noise_forecast(load_p_f)
#     load_q_f = load_q.apply(np.roll, shift=1)
#     load_q_ff = add_noise_forecast(load_q_f, noise_factor=0.05)
#     prod_p_f = prod_p.apply(np.roll, shift=1)
#     prod_p_ff = add_noise_forecast(prod_p_f) 
#     prod_v_f = prod_v.apply(np.roll, shift=1)
#     maintenance_f = maintenance.apply(np.roll, shift=1)
#     return load_p_ff, load_q_ff, prod_p_ff, prod_v_f, maintenance_f

