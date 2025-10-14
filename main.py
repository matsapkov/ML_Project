from Agent import Agent
from Environment import Environment

if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print('Total reward got: %.4f' % agent.total_reward)