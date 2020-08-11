
#      ,do  do#/reb,re,re#/mib,mi,fa,f 
# acordes 1,   2    ,3,    4  ,5, 6,    7    , 8,     9  ,10,  11   ,12
# TONALIDADES: 1 4 Y 5
# la tonalidad son las 7 notas despues de la elegida
# notas de acordes: 0,4 y 7
# crear individuo
import numpy as np
import random

# totalNotes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def create_ind():
    notes = [0]*3
    chord = random.randint(1, 12)
    notes = random.sample(range(1, 12), 3)
    notes[0] = chord
    return notes


def create_population(NumP):
    population = []
    for i in range(NumP):
        population.append(create_ind())
    return population

# totalNotes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def fitnessIndividuo(ind):
    fit = 0
    if ind[0] == 7: 
        if ind[1] != ind[0]+4: fit+=1 
        if ind[2]!= 2: fit+=1
      
    elif ind[0] == 8: 
        if ind[1] != ind[0]+4: fit+=1 
        if ind[2]!= 3: fit+=1
      
    elif ind[0] == 9: 
        if ind[1]!=1: fit+=1
        if ind[2]!= 4: fit+=1
      
    elif ind[0] == 10: 
        if ind[1]!=2: fit+=1
        if ind[2]!= 5: fit+=1
      
    elif ind[0] == 11: 
        if ind[1]!=3: fit+=1
        if ind[2]!= 6: fit+=1
      
    elif ind[0] == 12: 
        if ind[1]!=4: fit+=1
        if ind[2]!= 7: fit+=1
      
    else:
        if ind[1] != ind[0]+4: fit+=1 
        if ind[2] != ind[0]+7: fit+=1
    return fit


def fitness(inds):
    fitnessPopu = []
    for i in inds:
        fitnessPopu.append(fitnessIndividuo(i))
    return fitnessPopu

def crossOverInd(ind1,ind2):
    offspring=[]
    limit = np.random.randint(4)
    for i in range(0,limit):
        offspring.append(ind1[i])
    for j in range(limit,3):
        offspring.append(ind2[j])
    return offspring

def crossOver(popu,Pc,NumP):
    offspringPopu=[]
    while len(offspringPopu)< NumP:
        Proc = np.random.random(1)
        rand1 = np.random.randint(NumP)
        rand2 = np.random.randint(NumP)
        if Proc < Pc:
            offspringPopu.append(crossOverInd(popu[rand1],popu[rand2]))
        else:
            offspringPopu.append(popu[rand1])
    return offspringPopu

def mutationInd(ind):
    baseNote = ind.pop(0)
    np.random.shuffle(ind)
    ind.insert(0,baseNote)
    return ind

def mutation(popu,Pm,NumP):
    mutationPopu = []
    for ind in popu:
        proM = np.random.random(1)
        if proM < Pm:
            mutationPopu.append(mutationInd(ind))
        else:
            mutationPopu.append(ind)
    return mutationPopu

def elite(popu,fit):
    indexMin = np.argmin(fit)
    minFit = np.amin(fit)
    return popu[indexMin],minFit

def selectElite(popu,fit,currentElite,currentFit,NumP):
    index = np.argmin(fit)
    minFit = np.amin(fit)
    if minFit < currentFit:
        currentFit = minFit
        currentElite = popu[index]
    else:
        n = np.random.randint(NumP)
        popu[n] = currentElite
        fit[n] = currentFit
    return popu, fit,currentElite,currentFit


def selection(popu,fit,NumP):
    selectArray=[]
    while len(selectArray)<NumP:
        n1 = np.random.randint(NumP)
        n2 = np.random.randint(NumP)
        minVal = min(fit[n1],fit[n2])
        selectArray.append(popu[fit.index(minVal)])
    return selectArray

N = 1000
Pc = 0.7
Pm=0.3
G =10000

population = create_population(N)
print("pop",population)
fit = fitness(population)
ind_elite,fit_elite = elite(population,fit)
print("first fitness: ", fit_elite, "first individuo Elite: ", ind_elite)

g=0

while fit_elite > 0 and g<G:
    population = selection(population,fit,N)
    # print("popS",population)
    population = crossOver(population,Pc,N)
    # print("popC",population)
    population = mutation(population,Pm,N)
    # print("popM",population)
    fit = fitness(population)
    population,fit,ind_elite,fit_elite = selectElite(population,fit,ind_elite,fit_elite,N)

    g+=1
    if g %1000==0:
        print("Generacion: ",g,"fitness: ",fit_elite)
    # print('Generation:', g, ' fitness:', fit_elite)

print(ind_elite)


