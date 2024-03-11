# HW 2
Note 0 represents empty, 1 represents red agents, 2 represents green agents


Authors: Rachael Judy, Connor Klein, Josh Smith
The `simulate_automata` function is the core part of the functionality. To run a simulation, call the function

simulate_automata(L=50, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=0, policy_parameters=[], display_flag=True)

with the desired parameters. L is the length of the square grid, alpha determines what percentage of the grid is 
populated, k sets how many neighbors are needed to be completely happy, epochs sets how many epochs to run in a trial,
trials sets the number of trials, relocation_policy selects the policy to use (currently available are 0-4), 
policy parameters is a list of the parameters for a policy in order, and display_flag determines whether to show
the visual view of the automata.

The available policies are currently as follows:
- random_move (0): searches q empty spots at random and goes to one with greatest happiness
- social_network_recommendation (1): based on a randomly selected network of n friends, take the best options from a pxp
    sized neighborhood around the friend and move there
- rachael_move (2): 
- connor_move (3):
- josh_move (4):

To add more functions, simply add to the policies dictionary and extend the code calling the policies


### Packages
- Python 3.11
- collections
- copy
- matpltlib 3.8.2
- numpy
- random
- time
