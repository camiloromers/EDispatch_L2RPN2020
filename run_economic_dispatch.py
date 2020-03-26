import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import pypsa
from datetime import datetime, timedelta

# from utils import format_dates
# from utils import import_data
from .utils import rescale_gen_param
from .utils import run_unit_commitment
# from utils import interpolate
from .utils import add_noise_gen
from .utils import reformat_gen_constraints
# from utils import generate_prod_voltage
# from utils import generate_reactive_loads
# from utils import generate_hazard_maintenance
# from utils import generate_forecasts


# Vars to set up...
PYPSA_CASE = './L2RPN_2020_ieee_118_pypsa_simplified'
REF_DATA_DIR = 'reference_data'
DESTINATION_PATH = './OPF_rel/'

MODE_OPF = 'day'    # Can be: 'day', 'week', 'month', 'year'
POSSIBLE_MODE_OPF = ['day', 'week', 'month']

RESCALED_MIN = 5    # Every timr OPF will be run it
YEAR_OPF = 2007
MONTH_START = 1     # Initial mmonth
MONTH_END = 1       # End month 

def run_disptach(pypsa_net, 
                 load,
                 mode_opf='day', 
                 rescaled_min=5,
                 gen_constraints={'p_max_pu': None, 'p_min_pu': None},
                 load_factor_q=1.025 ,
                 MONTH_START=1,
                 MONTH_END=None,
                ):

    # OPF will run until the last month registered in load or demand
    if MONTH_END is None:
        MONTH_END = load.index.month.unique()[-1]

    # Load demand
    year_data = 2007
    snapshots = pd.date_range(start=f'{year_data}-01-01', periods=load.shape[0], freq='5min')
    load.index = snapshots

    # Resample load according to RESCALED_MIN
    load_resampled = load.resample(f'{str(rescaled_min)}min').apply(lambda x: x[0])
    load_resampled *= load_factor_q

    # Validate gen constraints to be imposed in the opf
    gen_constraints = reformat_gen_constraints(gen_constraints, 
                                               rescaled_min, 
                                               snapshots)

    # Scale gen properties (ramps up/down)
    pypsa_net = rescale_gen_param(pypsa_net, rescaled_min)

    # Define period to run OPF
    period_dict = {'day': load_resampled.groupby(load_resampled.index.day),
                   'week': load_resampled.groupby(load_resampled.index.week),
                   'month': load_resampled.groupby(load_resampled.index.month),
    }

    start = time.time()
    for _ in range(MONTH_START, MONTH_END + 1):
        results = []
        for _, demand_by_period in period_dict[mode_opf]:
            results.append(run_unit_commitment(pypsa_net,
                                                mode_opf,
                                                demand_by_period,
                                                gen_constraints,
                                                )
                            )

    # Unpack dispatch over list to concatenate partial dfs
    opf_prod = pd.DataFrame()
    for df in results:
        opf_prod = pd.concat([opf_prod, df], axis=0)

    # Sort by datetime
    opf_prod.sort_index(inplace=True)

    # Create complete prod_p dataframe and interpolate missing rows
    #   prod_p = interpolate(opf_prod, ref_index=tot_snapshots, method='cubic')
    prod_p = opf_prod.copy()

    # Add noise to results
    gen_cap = pypsa_net.generators.p_nom
    prod_p_with_noise = add_noise_gen(prod_p, gen_cap, noise_factor=0.001)

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
    #   full_destination_path = os.path.join(DESTINATION_PATH, str(year_data) + '_' +str(month))
    #   full_destination_path = os.path.join(output_path)
    #   if not os.path.exists(full_destination_path):
    #       os.makedirs(full_destination_path)

    #   # Get price curve
    #   p_gens = net.generators[['marginal_cost'] * tot_snapshots.shape[0]].T
    #   p_gens.index = tot_snapshots
    #   gen_status = opf_prod.copy()
    #   gen_status[gen_status > 0] = 1
    #   totalprice = gen_status.multiply(p_gens).sum(axis=1).to_frame()
    #   totalprice.columns = ['price']

    #   totalprice.to_csv(os.path.join(full_destination_path, 'price.csv'), sep=';', float_format='%.2f', index=False)
    #   prod_p_with_noise.to_csv(os.path.join(full_destination_path, 'prod_p.csv.bz2'), sep=';', float_format='%.2f', index=False)
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
    # print('OPF Done......')
    print('Total time {} min'.format(round((end - start)/60, 2)))

    return prod_p_with_noise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch the evaluation of the Grid2Op ("Grid To Operate") code.')
    parser.add_argument('--grid_path', default=PYPSA_CASE, type=str,
                        help='PyPSA grid dir')
    parser.add_argument('--conso_full_path', default='REF_DATA_DIR/load_2007.csv.bz2', type=str,
                       help='Specify the consumption or demanda full name')
    parser.add_argument('--dest_path', default=DESTINATION_PATH, type=str,
                         help='Specify the destination dir')
    parser.add_argument('--mode_opf', default=MODE_OPF, type=str,
                        help='Optimization mode, for the opf (possible values are {})'.format(POSSIBLE_MODE_OPF))
    parser.add_argument('--rescaled_min', default=RESCALED_MIN, type=int,
                        help='Run the optimizer every "rescaled_min" minutes (default 5)'),
    parser.add_argument('--year_data', default='2007', type=str,
                        help='Year generated for consumption')

    args = parser.parse_args()
    if not args.mode_opf.lower() in POSSIBLE_MODE_OPF:
        raise RuntimeError("Please provide an argument \"mode_opf\" among {}".format(POSSIBLE_MODE_OPF))
    rescaled_min = args.rescaled_min
    try:
        rescaled_min = int(rescaled_min)
    except:
        RuntimeError("\"rescaled_min\" argument should be convertible to an integer.")

    if not rescaled_min % 5 == 0:
        raise RuntimeError("\"rescaled_min\" argument should be multiple of 5 (so 5, or 15 is ok, but not 17 nor 3)")

    # **  **  **  **  ** 
    # Load the PyPSA grid
    net = pypsa.Network(import_name=args.grid_path)

    # Load consumption data without index
    print ('Reading input data -> load...')
    demand = pd.read_csv(args.conso_full_path)
    demand.drop('datetime', axis=1, inplace=True)

    # Run Economic Dispatch
    prod_p_dispatch = run_disptach(net,
                                   demand,
                                   mode_opf=args.mode_opf.lower(), 
                                   rescaled_min=rescaled_min,
                                   year_data=args.year_data,
                                   )

    destination_path = os.path.abspath(args.dest_path)
    print (destination_path)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Save as csv
    prod_p_dispatch.to_csv(os.path.join(destination_path, 'prod_p.csv.bz2'), sep=';', float_format='%.2f', index=False)
    
    print('OPF Done......')