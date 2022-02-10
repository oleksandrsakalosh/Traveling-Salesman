from dataclasses import dataclass
import random
import math
import copy
import numpy as np
import cv2

map_range = 500

@dataclass
class Gen:
    order: list
    fitness: float
    prob: float

COUNT = 0
cities = []
POP_SIZE = 0
BEST_RES = None
mutationRate = 1

def generateFirst():
    global COUNT
    order = []
    for i in range(1, COUNT):
        order.append(i)

    order = random.sample(order,len(order))
    fitn = calcFitn(order)
    gen = Gen(order, fitn, 0)
    return gen

def calcFitn(order1):
    order = order1.copy()
    fitn = 0
    order.append(0)
    for num in range(1, len(order)):
        curr = order[num]
        past = order[num-1]
        fitn += math.sqrt(pow((cities[curr][0] - cities[past][0]), 2) + pow((cities[curr][1] - cities[past][1]), 2))

    curr = order[0]
    past = order[-1]
    fitn += math.sqrt(pow((cities[curr][0] - cities[past][0]), 2) + pow((cities[curr][1] - cities[past][1]), 2))
    return fitn

def calcProb(pop1):
    pop = pop1.copy()
    fit = []
    sum = 0
    for x in pop:
        fit.append(1 / x.fitness)
        sum += 1 / x.fitness
    i = 0
    for x in pop:
        x.prob = fit[i] / sum
        i+=1

    return pop

def pickOneRandom(population):
    pop = population.copy()
    r = random.random()
    i = 0
    while r>0:
        r -= pop[i].prob
        i+=1

    i-=1
    gen1 = pop[i]

    return gen1

def pickBest(population):
    pop = population.copy()
    best = population[0]
    for gen in population:
        if gen.fitness < best.fitness:
            best = gen

    return best

def createNew(population):
    global mutationRate
    if mutationRate > 0.15:
        mutationRate = mutationRate/1.05
    old = copy.deepcopy(population)
    population.clear()
    for i in range(POP_SIZE//4):
        gen1 = copy.deepcopy(pickOneRandom(old))
        population.append(mutate(gen1, mutationRate))

    for i in range(POP_SIZE//4, POP_SIZE//2):
        gen2 = copy.deepcopy(pickBest(old))
        population.append(mutate(gen2, mutationRate))

    for i in range(POP_SIZE//2, POP_SIZE):
        gen3 = copy.deepcopy(pickBest(old))
        gen4 = copy.deepcopy(pickOneRandom(old))
        gen = crossOver(gen3, gen4)
        population.append(mutate(gen, mutationRate))

    population = calcProb(population)

    return population

def crossOver(gen1, gen2):
    r = random.sample(gen1.order, 2)
    i1 = gen.order.index(r[0])
    i2 = gen.order.index(r[1])
    kid = Gen(gen1.order[i1:i2], 0, 0)
    for city in gen2.order:
        if city not in kid.order:
            kid.order.append(city)
    return kid

def mutate(gen, mutRate):
    global BEST_RES
    k = 0
    for i in range(COUNT):
        g = random.random()
        if g < mutRate:
            k = 1
            r1 = random.choice(gen.order)
            i1 = gen.order.index(r1)
            i2 = (i1 + 1) % (COUNT - 1)
            r2 = gen.order[i2]
            gen.order[i1] = r2
            gen.order[i2] = r1

    gen.fitness = calcFitn(gen.order)
    if gen.fitness < BEST_RES.fitness:
        BEST_RES = gen
    return gen

if __name__ == '__main__':
    COUNT = int(input('Enter count of cities: '))
    POP_SIZE = int(input('Enter size of population: '))
    #for i in range(COUNT):
        #city = []
        #city.append(int(input())) 
        #city.append(int(input()))
        #cities.append(city)

    cities = [
            [random.randrange(0, map_range), random.randrange(0, map_range)] for i in range(COUNT)
        ]
    COUNT = len(cities)

    population = []
    for i in range(POP_SIZE):
        gen = generateFirst()
        population.append(gen)
    population = calcProb(population)
    BEST_RES = pickBest(population)
    localBest = BEST_RES
    n = 1
    mult = (COUNT - 20) * 0.1 + 1
    while n != 1000 * mult:
        print(n,"\n-----------------------\n")
        
        print(localBest)

        population = createNew(population)

        print("\n-----------------------\n")
        n+=1
        order = copy.deepcopy(BEST_RES.order)
        order.append(0)
        order.append(order[0])
        image = np.ones((map_range, map_range))

        pts = np.array([cities[i] for i in order], dtype=np.int32)
        for pt in pts:
            image = cv2.circle(image, (pt[0], pt[1]), 5, (0, 255, 0), 2)

        pts = pts.reshape((-1, 1, 2))
        image = cv2.polylines(image, [pts], False, (0, 255, 0), 2)
        cv2.imshow("Map", image)
        cv2.waitKey(1)
    print("Best result:\n", BEST_RES)
    while True:
        True