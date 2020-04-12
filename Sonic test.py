import retro
import neat
import cv2
import pickle
import numpy as np

SCALING_FACTOR = 8
MAX_STANGNATION_FRAMES = 250


#Setting up the game
env = retro.make("SonicTheHedgehog-Genesis","GreenHillZone.Act1")
env.reset()
input = [0,0,0,0,0,0,0,0,0,0,0,0]

# input[0] == Jump
# input[4] == "up"
# input[5] == "down"
# input[6] == "left"
# input[7] == "right"

def eval_genomes(genomes,config):

    for gen_id,genome in genomes:
        ob = env.reset()
        # rand_action = env.action_space.sample()

        img_shape = env.observation_space.shape
        img_shape = [int(pixels/SCALING_FACTOR) for pixels in img_shape] #calculate scaled down the image dimentions for neural network

        net = neat.nn.recurrent.RecurrentNetwork.create(genome,config)

        #reward function variables
        fitness_max = fitness_current = 0
        xpos_max = xpos_current = 0
        rings_max = rings_current = 0
        score_max = score_current = 0

        frame = 0
        stagnation_frames = 0

        done = False

        while not done:
            env.render()
            frame += 0
            ob = cv2.resize(ob,(img_shape[1],img_shape[0]))
            ob = cv2.cvtColor(ob,cv2.COLOR_BGR2GRAY)

            imgarray = ob.flatten()

            nn_output = net.activate(imgarray)
            action = [nn_output[0],0,0,0,nn_output[1],nn_output[2],nn_output[3],nn_output[4],0,0,0,0,0]

            ob,_,done,info = env.step(action)
            
            xpos_current = info["x"]
            rings_current = info["rings"]
            score_current = info["score"]


            #if sonic moves >1 frame to the right, increment fitness. fitness += 1
            if xpos_current > xpos_max:
                fitness_current += 1
                xpos_max = xpos_current
            
            #if sonic OBTAINS rings, increment fitness. fitness += 100
            if rings_current > rings_max:
                fitness_current += (rings_current - rings_max)*100
                rings_max = rings_current

            #if sonic LOSES rings, decrement fitness. fitness -= 100
            if rings_current < rings_max:
                fitness_current -= (rings_current - rings_max)*100
                rings_max = rings_current

            #if sonic scores, increment fitness. fitness += 50
            if score_current > score_max:
                fitness_current += (score_current - score_max)*100
                score_max = score_current

            # if current fitness went higher than previous max fitness, reset stagnation frames
            if fitness_current > fitness_max:
                fitness_max = fitness_current
                stagnation_frames = 0
            else: stagnation_frames += 1

            # if sonic reaches the screen_x_end, then fitness will be increased by 100k(fitness threshold in the config file)
            if (info["screen_x_end"] != 0) and (info["screen_x_end"] == info["screen_x"]): 
                fitness_current =+ 100000
            
            #if sonic doesn't get any fitness increment in 250 frames, reset
            if done or stagnation_frames == MAX_STANGNATION_FRAMES:
                done = True
                # print(gen_id,fitness_current)
            
            genome.fitness = fitness_current

            if genome.fitness >= 100000:
                with open("winner.pickle","wb") as f:
                    pickle.dump(net, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_neat():
    #setting up the config file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            'config-feedforward.txt')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    run_neat()

	# eval_genomes([("abs","abs")],"asd")