import sys
import os
import time
import pandas as pd
import numpy as np
import pypsa
from datetime import datetime, timedelta
from utils import format_dates
from utils import import_data
from utils import prepare_grid
from utils import run_unit_commitment
from utils import interpolate
from utils import add_noise_gen
# from utils import generate_prod_voltage
# from utils import generate_reactive_loads
# from utils import generate_hazard_maintenance
# from utils import generate_forecasts

# Vars to set up...
PYPSA_CASE = './L2RPN_2020_ieee_118_pypsa_simplified'
REF_DATA_DIR = './reference_data'
DESTINATION_PATH = './OPF_rel/'
MODE_OPF = 'day'    # Can be: 'day', 'week', 'month', 'year'
RESCALED_MIN = 10
YEAR_OPF = 2007
MONTH_START = 1
MONTH_END = 1
COMPENSATE_REACTIVE = 1.025
# -------------


# Load the grid (The network already contains the
# ramps multiplied by a factor of 0.5)
net = pypsa.Network(import_name=PYPSA_CASE)

start = time.time()
for month in range(MONTH_START, MONTH_END + 1):
  # Run between dates in parallel
  FROM_DATE, END_DATE = format_dates(YEAR_OPF, month)
  # Adapt steps according the RESCALED_MIN
  net = prepare_grid(net, RESCALED_MIN)
  # Load data
  print ('Reading input data -> load, wind and solar...')
  wind_resampled , solar_resampled, load_p_resampled, tot_snapshots = import_data(REF_DATA_DIR, 
                                                                                  FROM_DATE, 
                                                                                  END_DATE, 
                                                                                  RESCALED_MIN,
                                                                                 )
  load_p_resampled *= COMPENSATE_REACTIVE

  # def run_unit_commitment(net, df, mode):
  #   # Show info when running opf
  #   to_disp = {'day': df.index.day.unique().values[0],
  #              'week': df.index.week.unique().values[0],
  #              'month': df.index.month.unique().values[0],
  #              'year': df.index.year.unique().values[0],
  #   }
  #   print(f'\n--> OPF mode: {mode} - Analyzing {mode} # {to_disp[mode]}')
  #   # Get new snapshots and set them up
  #   snapshots = df.index
  #   # Fill constrains on the grid -> wind, solar, nuclear
  #   net = fill_constrains_grid(net, snapshots, df, wind_resampled, solar_resampled,)
  #   # Run Linear OPF
  #   rel = net.lopf(net.snapshots, pyomo=False, solver_name='cbc')
  #   if rel[1] != 'optimal': 
  #     sys.exit()
  #   # Get the values
  #   full_opf_rel = net.generators_t.p.copy()
  #   return full_opf_rel

  period_dict = {'day': load_p_resampled.groupby(load_p_resampled.index.day),
                 'week': load_p_resampled.groupby(load_p_resampled.index.week),
                 'month': load_p_resampled.groupby(load_p_resampled.index.month),
                 'year': load_p_resampled.groupby(load_p_resampled.index.year)
  }

  results = []
  for _, demand_by_period in period_dict[MODE_OPF]:
    results.append(run_unit_commitment(net, 
                                       MODE_OPF, 
                                       demand_by_period,
                                       wind_resampled,
                                       solar_resampled))

  # Unpack dispatch over list to concatenate partial dfs
  opf_prod = pd.DataFrame()
  for df in results:
    opf_prod = pd.concat([opf_prod, df], axis=0)

  # Sort by datetime
  opf_prod.sort_index(inplace=True)

  # ----------------------------
  # Generate csv's
  # ----------------------------
  print ('\nGenerating csv...')
  # Create complete prod_p dataframe and interpolate missing rows
  prod_p = interpolate(opf_prod, ref_index=tot_snapshots, method='cubic')
  # prod_p = opf_prod.copy()

  # Add noise to results
  gen_cap = net.generators.p_nom
  prod_p_with_noise = add_noise_gen(prod_p, gen_cap, noise_factor=0.0015)

  # # Generate voltage set points
  # vol_vals = net.buses.loc[net.generators.bus, 'v_nom'] * 1.035
  # vol_vals.index = net.generators.index.tolist()
  # prod_v = generate_prod_voltage(vol_vals, vol_var=1., ref_index=tot_snapshots)

  # # Generate reactive power for loads
  # load_q = generate_reactive_loads(load_p, min_range=12, max_range=25)

  # # Create hazards and maintenance adapting the names accorfing to Grid2op
  # full_names = pd.read_csv('lines_names.txt', sep=';')
  # full_names = full_names.columns.tolist()
  # # new_lines_names = list(map(lambda x: x[2:], full_names))
  # hazards, maintenance = generate_hazard_maintenance(lines_names=full_names, ref_index=tot_snapshots)

  # # Create forecast dataframes
  # load_p_f, load_q_f, prod_p_f, prod_v_f, maintenance_f = generate_forecasts(load_p,
  #                                                                            load_q,
  #                                                                            prod_p_with_noise,
  #                                                                            prod_v,
  #                                                                            maintenance)

  # Write results in tree directory
  full_destination_path = os.path.join(DESTINATION_PATH, str(YEAR_OPF) + '_' +str(FROM_DATE.month))
  if not os.path.exists(full_destination_path):
      os.makedirs(full_destination_path)

  # Get price curve
  p_gens = net.generators[['marginal_cost'] * tot_snapshots.shape[0]].T
  p_gens.index = tot_snapshots
  gen_status = opf_prod.copy()
  gen_status[gen_status > 0] = 1
  totalprice = gen_status.multiply(p_gens).sum(axis=1).to_frame()
  totalprice.columns = ['price']
  
  totalprice.to_csv(os.path.join(full_destination_path, 'price.csv'), sep=';', float_format='%.2f', index=False)
  prod_p_with_noise.to_csv(os.path.join(full_destination_path, 'prod_p.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # prod_v.to_csv(os.path.join(full_destination_path, 'prod_v.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # load_p.to_csv(os.path.join(full_destination_path, 'load_p.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # load_q.to_csv(os.path.join(full_destination_path, 'load_q.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # hazards.to_csv(os.path.join(full_destination_path, 'hazards.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # maintenance.to_csv(os.path.join(full_destination_path, 'maintenance.csv.bz2'), sep=';', float_format='%.2f', index=False)

  # prod_p_f.to_csv(os.path.join(full_destination_path, 'prod_p_forecasted.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # prod_v_f.to_csv(os.path.join(full_destination_path, 'prod_v_forecasted.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # load_p_f.to_csv(os.path.join(full_destination_path, 'load_p_forecasted.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # load_q_f.to_csv(os.path.join(full_destination_path, 'load_q_forecasted.csv.bz2'), sep=';', float_format='%.2f', index=False)
  # maintenance_f.to_csv(os.path.join(full_destination_path, 'maintenance_forecasted.csv.bz2'), sep=';', float_format='%.2f', index=False)

end = time.time()
print('OPF Done......')
print('Total time {} min'.format(round((end - start)/60, 2)))