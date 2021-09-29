import pandas as pd
import numpy as np
import os

output_folder = '/Users/pin-yichen/Dropbox (MIT)/0_Research/2021_Fall/Code/DPS_fullsim/final_output'

commodities = pd.read_csv(os.path.join(output_folder, 'commodities.csv'))
resources = pd.read_csv(os.path.join(output_folder, 'resources.csv'))
routes = pd.read_csv(os.path.join(output_folder, 'routes.csv'))

# total AMZL, ddu, 3p flows
print(commodities[['units','amzl_flow_primal_sol','ddu_flow_primal_sol', '3p_flow_primal_sol']].sum())

# check if total flow - target <= excess, i.e. min(excess - (flow - target) >= 0 )
(resources['excess_flow_primal_sol'] - (resources['total_flow_primal_sol'] - resources['target'])).min()


# temp = commodities.groupby('ds').sum().drop(columns = ['ddu_cost','3p_cost']).reset_index()
# dscap = resources[resources['type']=='ds_resource'].groupby('destination')[['cap']].sum().rename(columns = {'cap':'ds_resource_cap'})
# scdscap = resources[resources['type']=='scds_resource'].groupby('destination')[['cap']].sum().rename(columns = {'cap':'scds_resource_cap'})
# fcdscap = resources[resources['type']=='fcds_resource'].groupby('destination')[['cap']].sum().rename(columns = {'cap':'fcds_resource_cap'})
# temp = temp.merge(dscap, how = 'left', left_on = 'ds', right_on = 'destination')
# temp = temp.merge(scdscap, how = 'left', left_on = 'ds', right_on = 'destination')
# temp = temp.merge(fcdscap, how = 'left', left_on = 'ds', right_on = 'destination')

# focus on BFL1, DPS1 commmodity 
one_commodity = commodities[(commodities.fc == 'BFL1')&(commodities.ds == 'DPS1')]
one_commodity

relevent_routes = routes[routes['index'].isin(['r18', 'r93', 'r173', 'r188', 'r278', 'r353'])]
relevent_routes

relevent_resource = resources[resources['index'].isin(['j83', 'j28','j84', 'j29'])]
relevent_resource