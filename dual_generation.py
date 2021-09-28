import os
import numpy as np
import pandas as pd
import cvxpy as cp
import gurobipy
import time


# config




class subnet_initialization():

    def __init__(self, path):
        self.commodity_data, self.resource_data, self.route_data = self.dataloader(path)
        self.CV = 0.1
        self.z = 2
        self.create_sets_from_data()

    class commodity():
        def __init__(self,k, commodity_data):
            dic = commodity_data.to_dict(orient = 'index')
            self.fc = dic[k]['fc']
            self.ds = dic[k]['ds']
            self.arrival = dic[k]['arrival']
            self.promise = dic[k]['promise']
            self.dimension = dic[k]['dimension']
            self.tp_cost = dic[k]['3p_cost']
            self.ddu_cost = dic[k]['ddu_cost']
            self.feasible_routes = dic[k]['feasible_routes']
            self.units = dic[k]['units']

    class resource():
        def __init__(self,j, CV, z, resource_data):
            dic = resource_data.to_dict(orient = 'index')
            self.origin = dic[j]['origin']
            self.destination = dic[j]['destination']
            self.cpt = dic[j]['cpt']
            self.cap = dic[j]['cap']
            self.index = dic[j]['index']
            self.type = dic[j]['type']
            self.hard_or_soft = dic[j]['hard_or_soft']
            self.alternativecost = dic[j]['alternative_cost']
            # self.sigma = dic[j]['sigma']
            # self.target = dic[j]['target']
            
            # create sigma, penalty cost, target, hardcap 
            self.sigma = CV*self.cap

            if self.sigma != 0:
                self.penaltycost = 0.798*self.alternativecost/((self.sigma)*z**2)
            else:
                self.penaltycost = 0.798*self.alternativecost/((CV)*z**2)
 
            if self.hard_or_soft == 'soft':
                self.target = self.cap - z*self.sigma
                self.hardcap = float('inf')
            else:
                self.target = float('inf')
                self.hardcap = self.cap


    class route():
        def __init__(self, r, route_data):
            dic = route_data.to_dict(orient = 'index')
            self.resources = dic[r]['resources'] # put eval function here
            self.index = dic[r]['index']
            self.ds = dic[r]['ds']
            self.type = dic[r]['type']
            # self.fc = dic[r]['fc']

    def dataloader(self, path):
        return pd.read_parquet(os.path.join(path,'commodities.parquet')), pd.read_parquet(os.path.join(path,'resources.parquet')), pd.read_parquet(os.path.join(path,'routes.parquet'))


    def create_sets_from_data(self):
        self.K = [self.commodity(k, self.commodity_data) for k in range(len(self.commodity_data))] # create commodity set
        self.J = [self.resource(j,self.CV, self.z, self.resource_data) for j in range(len(self.resource_data))] # create resource set

        # breack the dataframe into two based on type
        self.L = self.P = self.R = []
        for r in range(len(self.route_data)):
            if self.route_data['type'].iloc[r] == 'ddu':
                self.L.append(self.route(r, self.route_data))
            elif self.route_data['type'].iloc[r] == '3p':
                self.P.append(self.route(r, self.route_data))
            else:
                self.R.append(self.route(r, self.route_data))
        
        def name_to_index(class_list):
            dic_name_to_index = {}
            for i in range(len(class_list)):
                dic_name_to_index[class_list[i].index] = i 
            return dic_name_to_index

        self.P_nametoindex = name_to_index(self.P)
        self.R_nametoindex = name_to_index(self.R)
        self.L_nametoindex = name_to_index(self.L)
        self.J_nametoindex = name_to_index(self.J)

        self.R_j = {j: [] for j in range(len(self.J))}
        self.L_j = {j: [] for j in range(len(self.J))}
        self.P_j = {j: [] for j in range(len(self.J))}

        for r in range(len(self.R)):
            for resource in self.R[r].resources:
                j = self.J_nametoindex[resource]
                self.R_j[j].append(r)

        for l in range(len(self.L)):
            for resource in self.L[l].resources:
                j = self.J_nametoindex[resource]
                self.L_j[j].append(l)

        for p in range(len(self.P)):
            for resource in self.P[p].resources:
                j = self.J_nametoindex[resource]
                self.P_j[j].append(p)


        self.K_r = {r: [] for r in range(len(self.R))}
        self.K_l = {l: [] for l in range(len(self.L))}
        self.K_p = {p: [] for p in range(len(self.P))}
        self.R_k = {k: [] for k in range(len(self.K))}
        self.L_k = {k: [] for k in range(len(self.K))}
        self.P_k = {k: [] for k in range(len(self.K))}
        
        for k in range(len(self.K)):
            # create R_k and L_k
            feasible_routes = self.K[k].feasible_routes
            dic_route_type = self.route_data.set_index(['index'])['type'].to_dict()

            for route in feasible_routes:
                if dic_route_type[route] == 'ddu':
                    l = self.L_nametoindex[route]
                    self.L_k[k].append(l)
                    self.K_l[l].append(k)

                elif dic_route_type[route] == '3p':
                    p = self.P_nametoindex[route]
                    self.P_k[k].append(p)
                    self.K_p[p].append(k)
                else:
                    r = self.R_nametoindex[route]
                    self.R_k[k].append(r)
                    self.K_r[r].append(k)


class formulate_and_solve_QP:
    def __init__(self, subnet):
        self.subnet = subnet
        self.build_and_solve_QP()

    def add_variables(self):
        self.y_kr = {}
        self.w_lk = {}
        self.h_kp = {}
        self.g_j = {}
        for k in range(len(self.subnet.K)):
            for r in self.subnet.R_k[k]:
                self.y_kr[(k,r)] = cp.Variable(integer=False, nonneg=True)
            for l in self.subnet.L_k[k]:
                self.w_lk[(l,k)] = cp.Variable(integer=False, nonneg=True)
            for p in self.subnet.P_k[k]:
                self.h_kp[(k,p)] = cp.Variable(integer=False, nonneg=True)              
        for j in range(len(self.subnet.J)):
            self.g_j[j] = cp.Variable(integer=False, nonneg=True)

    def add_soft_capacity_constraints(self):
        for j in range(len(self.subnet.J)):
            self.constraints.append(sum(self.h_kp[(k,p)] for p in self.subnet.P_j[j] for k in self.subnet.K_p[p])+ sum(self.w_lk[(l,k)] for l in self.subnet.L_j[j] for k in self.subnet.K_l[l])+ sum(self.y_kr[(k,r)] for r in self.subnet.R_j[j] for k in self.subnet.K_r[r]) - self.subnet.J[j].target <= self.g_j[j])
    
    def add_hard_cap_constraints(self):
        for j in range(len(self.subnet.J)):
            self.constraints.append(sum(self.h_kp[(k,p)] for p in self.subnet.P_j[j] for k in self.subnet.K_p[p])+ sum(self.w_lk[(l,k)] for l in self.subnet.L_j[j] for k in self.subnet.K_l[l])+ sum(self.y_kr[(k,r)] for r in self.subnet.R_j[j] for k in self.subnet.K_r[r]) <= self.subnet.J[j].hardcap)

    def add_demand_constraints(self):
        for k in range(len(self.subnet.K)):
            self.constraints.append(sum(self.h_kp[(k,p)] for p in self.subnet.P_k[k]) +sum(self.w_lk[(l,k)] for l in self.subnet.L_k[k]) + sum(self.y_kr[(k,r)] for r in self.subnet.R_k[k]) == self.subnet.K[k].units)

    def add_objective(self):
        tp_cost = sum((self.subnet.K[k].tp_cost)*self.h_kp[(k,p)] for k in range(len(self.subnet.K)) for p in self.subnet.P_k[k])
        ddu_cost = sum((self.subnet.K[k].ddu_cost)*self.w_lk[(l,k)] for k in range(len(self.subnet.K)) for l in self.subnet.L_k[k])
        quadratic_cost = 0.5*sum(self.subnet.J[j].penaltycost*(self.g_j[j]**2) for j in range(len(self.subnet.J)))
        self.objective  =cp.Minimize(quadratic_cost + tp_cost + ddu_cost)

    def get_solutions(self): # get primal and dual solutions
        # solve problem
        QP = cp.Problem(self.objective, self.constraints)
        QP.solve(solver=cp.GUROBI)

        # primal solution
        self.y_kr_sol = {}
        self.w_lk_sol = {}
        self.h_kp_sol = {}
        for k in range(len(self.subnet.K)):
            for r in self.subnet.R_k[k]:
                self.y_kr_sol[(k,r)] = round(self.y_kr[(k,r)].value,2)
            for l in self.subnet.L_k[k]:
                self.w_lk_sol[(l,k)] = round(self.w_lk[(l,k)].value,2)
            for p in self.subnet.P_k[k]:
                self.h_kp_sol[(k,p)] = round(self.h_kp[(k,p)].value,2)

        self.g_j_sol = []
        self.total_flow_j_sol= []
        for j in range(len(self.subnet.J)):
            self.g_j_sol.append(self.g_j[j].value)
            self.total_flow_j_sol.append(sum(self.h_kp_sol[(k,p)] for p in self.subnet.P_j[j] for k in self.subnet.K_p[p])+ sum(self.w_lk_sol[(l,k)] for l in self.subnet.L_j[j] for k in self.subnet.K_l[l])+ sum(self.y_kr_sol[(k,r)] for r in self.subnet.R_j[j] for k in self.subnet.K_r[r]))

        # dual solutions
        self.resource_soft_duals = [np.round(self.constraints[i].dual_value,2) for i in range(0, len(self.subnet.J))]
        self.resource_hard_duals = [np.round(self.constraints[i].dual_value,2) for i in range(len(self.subnet.J), 2*len(self.subnet.J))]

    def build_and_solve_QP(self):
        self.constraints = []
        self.add_variables()
        self.add_soft_capacity_constraints()
        self.add_hard_cap_constraints()
        self.add_demand_constraints()
        self.add_objective()
        self.QP = cp.Problem(self.objective, self.constraints)
        self.QP.solve(solver=cp.GUROBI)
        self.get_solutions()
    
def store_outputs(subnet, solution, output_path):
    # resource data 
    subnet.resource_data['target'] = np.abs([subnet.J[j].target for j in range(len(subnet.J))])
    subnet.resource_data['sigma'] = np.abs([subnet.J[j].sigma for j in range(len(subnet.J))])
    subnet.resource_data['soft_dual'] = np.abs(solution.resource_soft_duals)
    subnet.resource_data['hard_dual'] = np.abs(solution.resource_hard_duals)
    subnet.resource_data['excess_flow_primal_sol'] = np.abs(solution.g_j_sol)
    subnet.resource_data['total_flow_primal_sol'] = np.abs(solution.total_flow_j_sol)
    subnet.resource_data = subnet.resource_data.round(2)

    # route data 
    for rr in range(len(subnet.R)):
        if subnet.R[rr].type == '3p':
            p = subnet.P_nametoindex[subnet.R[rr].index]
            flow = sum(solution.h_kp_sol[(k,p)] for k in subnet.K_p[p])
        elif subnet.R[rr].type == 'ddu':
            l = subnet.L_nametoindex[subnet.R[rr].index]
            flow = sum(solution.w_lk_sol[(l,k)] for k in subnet.K_l[l])
        else:
            r = subnet.R_nametoindex[subnet.R[rr].index]
            flow = sum(solution.y_kr_sol[(k,r)] for k in subnet.K_r[r])
        subnet.route_data['flow_primal_sol'].iloc[r] = flow


    # subnet.route_data['total_flow'] = np.abs([subnet.J[j].target for j in range(len(subnet.J))])
    

    # commodity data
    flows_on_routes_all = []
    subnet.commodity_data['amzl_flow_primal_sol'] = subnet.commodity_data['ddu_flow_primal_sol'] = subnet.commodity_data['3p_flow_primal_sol'] = np.nan
    for k in range(len(subnet.K)):
        subnet.commodity_data['amzl_flow_primal_sol'].iloc[k] = sum(solution.y_kr_sol[(k,r)] for r in subnet.R_k[k])
        subnet.commodity_data['ddu_flow_primal_sol'].iloc[k] = sum(solution.w_lk_sol[(l,k)] for l in subnet.L_k[k])
        subnet.commodity_data['3p_flow_primal_sol'].iloc[k] = sum(solution.h_kp_sol[(k,p)] for p in subnet.P_k[k])
        
        flows_on_routes = {}
        for r in subnet.R_k[k]:
            flows_on_routes[subnet.R[r].index] = solution.y_kr_sol[(k,r)]
        for l in subnet.L_k[k]:
            flows_on_routes[subnet.L[l].index] = solution.w_lk_sol[(l,k)]
        for p in subnet.P_k[k]:
            flows_on_routes[subnet.P[p].index] = solution.h_kp_sol[(k,p)]
        flows_on_routes_all.append(flows_on_routes)
    subnet.commodity_data['flows_on_each_route'] = np.array(flows_on_routes_all)
    subnet.commodity_data = subnet.commodity_data.round(2)

    # subnet.resource_data.to_parquet('resources_with_solution.parquet', index = False)
    subnet.resource_data.to_csv(os.path.join(output_path,'resources.csv'), index = False)
    subnet.commodity_data.to_csv(os.path.join(output_path,'commodities.csv'), index = False)

    # subnet.resource_data.groupby('type').agg({'soft_dual':['mean', 'std'],'hard_dual':['mean', 'std']}).round(2)

# create QP
if __name__ == '__main__':

    INPUT_DIR = '/Users/pin-yichen/Dropbox (MIT)/0_Research/2021_Fall/Code/DPS_fullsim/data/simple_example_v3'
    QP_OUTPUT_DIR = '/Users/pin-yichen/Dropbox (MIT)/0_Research/2021_Fall/Code/DPS_fullsim/QP_output'


    before = time.time()
    subnet = subnet_initialization(INPUT_DIR) # this do not need to repeat
    after = time.time()
    print('total time for creating the network', after - before)

    before = time.time()
    solution = formulate_and_solve_QP(subnet)
    after = time.time()
    print('total time for solving QP', after - before)

    store_outputs(subnet, solution, QP_OUTPUT_DIR)



    