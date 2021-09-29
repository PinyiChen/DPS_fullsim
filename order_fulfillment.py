# import collections
import dual_generation as dg
import os 
import pandas as pd

class pre_fulfillment():
    def __init__(self, subnet, solution, OUTPUT_DIR, iter):
        self.subnet = subnet
        self.solution = solution
        self.OUTPUT_DIR = OUTPUT_DIR
        self.iter = iter
        self.create_resource_exchange_file()
        self.update_route_cost()

    def create_resource_exchange_file(self):
        self.subnet.resource_exchange = self.subnet.resource_data[['origin', 'destination', 'cap', 'index', 'soft_dual', 'hard_dual']]
        self.subnet.resource_exchange['opportunity_cost'] = self.subnet.resource_exchange['soft_dual'] + self.subnet.resource_exchange['hard_dual']
        self.subnet.resource_exchange = self.subnet.resource_exchange.drop(columns = ['soft_dual', 'hard_dual'])
        self.subnet.resource_exchange.to_csv(os.path.join(self.OUTPUT_DIR, 'resource_states_' +str(self.iter)+'.csv'))
    
    def update_route_cost(self):
        # add a new column 'route_cost' to subnet.route_data

        def route_cost_calculator(x):
            cost = 0
            for resource_index in x:
                cost += self.subnet.resource_exchange.loc[self.subnet.resource_exchange['index'] == resource_index, 'opportunity_cost'].item()
            return cost
        self.subnet.route_data['route_cost'] = self.subnet.route_data.apply(lambda x: x.base_cost + route_cost_calculator(x.resources),axis = 1)



class order_fufillment():

    def __init__(self, INPUT_DIR, subnet, start_time, end_time):
        self.subnet = subnet
        self.INPUT_DIR = INPUT_DIR
        self.start_time = start_time
        self.end_time = end_time
        self.shipment_data = pd.read_parquet(os.path.join(self.INPUT_DIR, 'shipments.parquet')).sort_values(by = 'arrival')
        # self.shipment_data = self.shipment_data[self.shipment_data.arrival <= end_time]



    def order_fulfillment(self):
        # add route assignment decision to shipment_data
        for s in range(len(self.shipment_data)):

            feasible_route_list = self.shipment_data.feasible_routes.iloc[s]
            route_table = self.subnet.route_data[self.subnet.route_data['index'].isin(feasible_route_list)].sort_values(by = 'route_cost')
            
            # check if a route is capacity feasible 
            r = 0
            chosen_route_index = 'no_feasible_route'
            while r < len(route_table):
                resource_list = route_table.resources.iloc[r]
                min_remaining_cap = self.subnet.resource_exchange[self.subnet.resource_exchange['index'].isin(resource_list)].cap.min()
                if min_remaining_cap >=1:
                    chosen_route_index = route_table['index'].iloc[r]
                    break
            self.shipment_data['assigned_route'].iloc[s] == chosen_route_index



            # for 

            # capacity state update according to the assignment decision
            resource_list = self.subnet.route_data.loc[self.subnet.route_data['index'] == chosen_route_index, 'resources'].item()
            for j in resource_list:
                self.subnet.resource_exchange.loc[self.subnet.resource_exchange['index'] == j, 'cap'] -= 1
            



# create QP
if __name__ == '__main__':

    INPUT_DIR = '/Users/pin-yichen/Dropbox (MIT)/0_Research/2021_Fall/Code/DPS_fullsim/data/simple_example_v4'
    OUTPUT_DIR = '/Users/pin-yichen/Dropbox (MIT)/0_Research/2021_Fall/Code/DPS_fullsim/final_output'
    iter = 1

    subnet = dg.subnet_initialization(INPUT_DIR) # this do not need to repeat
    solution = dg.formulate_and_solve_QP(subnet)


    dg.store_outputs(subnet, solution, OUTPUT_DIR)

     
    pre_fulfillment(subnet, solution, OUTPUT_DIR, iter)
    assigment = order_fufillment(INPUT_DIR, subnet)
    