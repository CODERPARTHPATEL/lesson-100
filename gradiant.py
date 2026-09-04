from numpy import asarray
from numpy import arange
from numpy.random import rand
from matplotlib import pyplot

def objective(x):
    return x**2.0

def derivative(x):
    return x * 2.0
#gradiant descent algorithm
def gradiant_decent(objective,derivative, bounds, n_iter,step_size):
    #track all solition
    solutions,scores = list(),list()
#genaratw an internal point
    solution = bounds[:,0]+ rand(len(bounds)) *(bounds[:,1]-bounds[:,0])
#run the gradiant
    for i in range(n_iter):
        #calc gradiant
        gradiant = derivative(solution)
        #take step
        solution = solution - step_size * gradiant
        #evaluate candidate point
        solution_eval = objective(solution)
        #store solution
        solutions.append(solution)

        scores.append(solution_eval)
        #report progress
        print('>%d f(%s)=%.5f'%(i,solution,solution_eval))

    return[solution,scores]
#define range for input
bounds = asarray([[-1.0,1.0]])
#define total iterations
n_iter = 30
#define step size
step_size = 0.1
#perform the gradiant descent search
solutions,scores = gradiant_decent(objective,derivative,bounds,n_iter,step_size)
#szmple input range uniformly at 0.1 increments
inputs = arange(bounds[0,0],bounds[0,1]+0.1,0.1)
#compute targets
results = objective(inputs)

pyplot.plot(inputs,results)
pyplot.plot(solutions,scores,'.-',color='red')

pyplot.show()