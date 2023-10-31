import numpy as np

'''
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)

    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA

    return policy
'''

def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    # Policy Evaluation
    while True:
        #iter +=1
        v = V.copy()
        check = [False for _ in range(env.nS)]
        for i in range(env.nS):
            diff = 0
            Vs = 0            
            for j in range(env.nA):
                for k in env.MDP[i][j]:
                    Vs += policy[i][j] * k[0]*(k[2]+gamma*V[k[1]])
            V[i] = Vs
            diff = max(diff, abs(v[i]-V[i]))
            if diff < theta:
                check[i] = True
        if all(check) == False:
            continue
    # Policy Improvement
        old_policy = policy.copy()
        for i in range(env.nS):
            new_list = [0 for _ in range(env.nA)]
            arg = []
            for j in range(env.nA):
                Va = 0
                for k in env.MDP[i][j]:
                    Va += k[0]*(k[2]+gamma*V[k[1]])
                arg.append(Va)   
            index = arg.index(max(arg))
            new_list[index] = 1
            policy[i] = new_list
        stable = True
        for i in range(env.nS):
            for j in range(env.nA):
                if old_policy[i][j] != policy[i][j]:
                    stable = False
                    break
        if stable:
            break
    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    # Policy Evaluation
    while True:
        #iter +=1
        v = V.copy()
        check = [False for _ in range(env.nS)]
        for i in range(env.nS):
            diff = 0
            Vs = 0
            max_Vs = 0            
            for j in range(env.nA):
                for k in env.MDP[i][j]:
                    Vs += policy[i][j] * k[0]*(k[2]+gamma*V[k[1]])
                max_Vs = max(max_Vs, Vs)
            V[i] = max_Vs
            diff = max(diff, abs(v[i]-V[i]))
            if diff < theta:
                check[i] = True
        if all(check) == True:
            break
    # # Policy change
        for i in range(env.nS):
            new_list = [0 for _ in range(env.nA)]
            arg = []
            for j in range(env.nA):
                Va = 0
                for k in env.MDP[i][j]:
                    Va += k[0]*(k[2]+gamma*V[k[1]])
                arg.append(Va)    
            index = arg.index(max(arg))
            new_list[index] = 1
            policy[i] = new_list       
    return policy, V