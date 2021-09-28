import dual_generation as dg
import pre_fulfillment
import post_fulfillment
import order_fulfillment as of


if __name__ == '__main__':

    INPUT_DIR = '/Users/pin-yichen/Dropbox (MIT)/0_Research/2021_Fall/Code/DPS_fullsim/data/simple_example_v3'
    OUTPUT_DIR = '/Users/pin-yichen/Dropbox (MIT)/0_Research/2021_Fall/Code/DPS_fullsim/final_output'

    subnet = dg.subnet_initialization(INPUT_DIR) # this do not need to repeat
    solution = dg.formulate_and_solve_QP(subnet)
    subnet = pre_fulfillment(OUTPUT_DIR, solution, subnet)
    assignment_decision = of.order_fufillment(subnet)
    subnet = post_fulfillment(OUTPUT_DIR, assignment_decision, subnet)


    