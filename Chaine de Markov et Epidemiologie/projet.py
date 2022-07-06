import numpy as np
from random import *
import matplotlib.pyplot as plt




#### Exercice 1

# Question 1.1 (sur le notebook).




# Question 1.2 :
def estimMatriceProbaTransi(tabData):
    """
    Renvoie une matrice contenant l'estimation des probabilités de transitions à partir des données passées en paramètre

    Args :
    - tabData {np.ndarray} : Le np.ndarray contenant les données

    Returns :
    - [[float]] : La matrice contenant les probabilités estimées
    """
    matProba = []
    for state in [0, 1, 2]:
        matProba.append([0, 0, 0])
        for nextState in [0, 1, 2]:
            matProba[state]
            matProba[state][nextState] = 0

    for individualData in tabData:
        precState = int(individualData[0])  # precState contient l'indice de l'état précédent
        for state in individualData[1:len(individualData)]:
            currentState = int(state)
            matProba[precState][currentState] += 1
            precState = currentState

    for state in [0, 1, 2]:
        totalNumberOfTransition = 0
        for nextState in [0, 1, 2]:
            totalNumberOfTransition += matProba[state][nextState]
        for nextState in [0, 1, 2]:
            matProba[state][nextState] = float(matProba[state][nextState]) / totalNumberOfTransition
        totalNumberOfTransition = 0

    return matProba


#### Exercice 2

# Question 2.1(notebook) :




def isStochMat(matrix):
    """
    Renvoie Vrai si la matrice passée en paramètre est stochastique et Faux sinon

    Args :
    - matrix {[[float]]} : La matrice

    Returns :
    - {Bool} : Vrai si matrix est stochastique, Faux sinon
    """
    lenMatrix = len(matrix)

    for i in range(0, lenMatrix):
        if len(matrix[i]) != lenMatrix:  # Si la matrice n'est pas carrée on retourne faux
            return False

        sumRow = 0

        for j in range(0, len(matrix[i])):
            currentProb = matrix[i][j]
            if currentProb < 0 or currentProb > 1:
                return False
            sumRow += currentProb

        if sumRow != 1:  # Si la somme des probabilités n'est pas égale à 1 on retourne faux
            return False

    return True


#### Distribution pi(t)

# Question 1 :
def getTime1Prob(initialVector, matrix):
    """
    Renvoie le vecteur de probabilité au temps 1

    Args :
    - initialVector {dict(int,float)} : Le vecteur de probabilité initial
    - matrix {[[float]]} : La matrice de transition

    Returns :
    - tuple(float) : La vecteur de probabilité
    """
    pi1 = {0: 0.0, 1: 0.0, 2: 0.0}

    for nextState in [0, 1, 2]:
        probNextState = 0.0
        for initialState in [0, 1, 2]:
            probNextState += matrix[initialState][nextState] * initialVector[initialState]
        pi1[nextState] = probNextState

    return pi1


# Question 2 :
def getTimeTProb(initialVector, matrix, t):
    """
    Renvoie le vecteur de probabilité au temps t

    Args :
    - initialVector {dict(int,float)} : Le vecteur de probabilité initial
    - matrix {[[float]]} : La matrice de transition

    Returns :
    - {float} : La probabilité
    """
    pit = initialVector
    for _ in range(1, t):
        pit = getTime1Prob(pit, matrix)

    return pit


# Tirage aléatoire des états

def randomDraw(initialVector, matrix, T):
    """
    Renvoie une séquence de T états en suivant la chaîne de Markov décrite par matrix

    Args :
    - initialVector {dict(int,float)} : Le vecteur de probabilité initial
    - matrix {[[float]]} : La matrice de transition
    - T {int} : Le nombre d'itérations

    Returns :
    - {[int]} : La séquence d'état
    """
    sequence = [vectorDraw(initialVector)]
    for i in range(1, T):
        currentVector = matrix[sequence[i - 1]]
        sequence.append(vectorDraw(currentVector))

    return sequence


def vectorDraw(vector):
    """
    Exécute un tirage aléatoire à partir des probabilités du vecteur et renvoie l'état correspondant

    Args :
    - vector {[float]} : Le vecteur de probabilité de transition à partir d'un égtat

    Returns :
    - {int} : L'état résultat
    """
    maxProb = -1
    stateMaxProb = -1
    minProb = 1.1
    stateMinProb = -1
    for state in [0, 1, 2]:
        if vector[state] > maxProb:
            maxProb = vector[state]
            stateMaxProb = state
        if vector[state] < minProb:
            minProb = vector[state]
            stateMinProb = state

    randValue = random()
    resultState = -1
    for state in [0, 1, 2]:

        if state == stateMinProb:
            if randValue < vector[state]:
                return state

        elif state != stateMaxProb and stateMinProb != -1:
            if randValue < vector[state] + vector[stateMinProb]:
                resultState = state

    if resultState == -1:
        return stateMaxProb
    else:
        return resultState


#### Modélisation d'une population

# Question 1 :
def statsOnSequences(initialVector, matrix, T, numberOfIndiv):
    """
    Renvoie un tableau contenant le nombre d'individus dans un état à chaque pas de temps

    Args :
    - initialVector {dict(int,float)} : Le vecteur de probabilité initial
    - matrix {[[float]]} : La matrice de transition
    - T {int} : Le nombre d'itérations par individu
    - numberOfIndiv {int} : Le nombre d'individu

    Returns :
    - {[dict(int,int)]} : Un tableau où l'indice correspond au temps et contenant les dictionnaires suivants :
        - clé : L'état
        - valeur : Le nombre d'individu dans cet état
    """

    # On initialise les séquences pour chaque individu
    sequencesTab = []
    for _ in range(0, numberOfIndiv):
        sequencesTab.append(randomDraw(initialVector, matrix, T))

    # On calcule le nombre d'individu dans un état à chaque pas de temps
    statsEvolution = []
    for time in range(0, T):
        currentStats = dict()
        currentStats[0] = 0
        currentStats[1] = 0
        currentStats[2] = 0
        for i in range(0, len(sequencesTab)):
            currentStats[sequencesTab[i][time]] += 1

        statsEvolution.append(currentStats)

    return statsEvolution


# Question 2 :
def fromStatsToPercentage(statsEvolution):
    """
    Renvoie le pourcentage associé à des statistiques sur le nombre d'individu dans chaque état au temps t

    Args :
    - statsEvolution {[dict(int,int)]} : Les statistiques

    Returns :
    - {[dict(int,float)]} : Les pourcentages associés
    """

    percentageEvolution = []
    for i in range(0, len(statsEvolution)):
        numberOfIndivS = statsEvolution[i][0]
        numberOfIndivI = statsEvolution[i][1]
        numberOfIndivR = statsEvolution[i][2]
        numberOfIndiv = numberOfIndivS + numberOfIndivI + numberOfIndivR

        currentPercentage = dict()
        currentPercentage[0] = (float(numberOfIndivS) / numberOfIndiv) * 100
        currentPercentage[1] = (float(numberOfIndivI) / numberOfIndiv) * 100
        currentPercentage[2] = (float(numberOfIndivR) / numberOfIndiv) * 100

        percentageEvolution.append(currentPercentage)

    return percentageEvolution



# Question 3 (notebook) :


# Question 4 (notebook) :


# Pic de l'épidémie

# Question 1 (notebook) :


# Longueur de l'infection

# Question 1 :

# Cette fonction sera utilisé pour d'autres états que l'état I, on va donc passer l'état choisi en paramètre pour la rendre ré-utilisable
def estimAvgStateSequence(initialVector, matrix, T, numberOfIndiv, choosenState):
    """
    Retourne la longueur moyenne d'une séquence de l'état choisi en paramètre

    Args :
    - initialVector {dict(int,float)} : Le vecteur de probabilité initial
    - matrix {[[float]]} : La matrice de transition
    - T {int} : Le nombre d'itérations par individu
    - numberOfIndiv {int} : Le nombre d'individu
    - choosenState {int} : l'état dont on souhaite estimer la longueur d'une chaîne

    Returns :
    - {float} : La longueur moyenne
    """

    # On initialise les séquences pour chaque individu
    numberOfSeries = 0
    tabNumbersOfSerie = []

    for _ in range(0, numberOfIndiv):
        sequence = randomDraw(initialVector, matrix, T)
        # On calcule la longueur d'une séquence de choosenState
        CSserie = 0
        inSequence = False
        for state in sequence:
            if state == choosenState and inSequence == True:
                CSserie += 1
            elif state == choosenState and inSequence == False:
                inSequence = True
                CSserie = 1
            elif state != choosenState and inSequence == True:
                tabNumbersOfSerie.append(CSserie)
                CSserie = 0
                inSequence = False
        if(CSserie !=0):

            tabNumbersOfSerie.append(CSserie)
        CSserie = 0
    if(len(tabNumbersOfSerie) == 0):
        return 0
    return sum(tabNumbersOfSerie)/len(tabNumbersOfSerie)


# Question 2 (notebook):


#### Modèle ergodique (notebook)




#### CONFINEMENT
# Question 1 :
def statsOnSequencesLockDown(initialVector,matrix,T,numberOfIndiv):
    """
    Renvoie un tableau contenant le nombre d'individus dans un état à chaque pas de temps, dans un modèle où un confinement
    peut être instauré.

    Args :
    - initialVector {dict(int,float)} : Le vecteur de probabilité initial
    - matrix {[[float]]} : La matrice de transition
    - T {int} : Le nombre d'itérations par individu
    - numberOfIndiv {int} : Le nombre d'individu

    Returns :
    - {[dict(int,int)]} : Un tableau où l'indice correspond au temps et contenant les dictionnaires suivants :
        - clé : L'état
        - valeur : Le nombre d'individu dans cet état
    """
    tabNbJourConfinement = []
    # On initialise le tableau de séquence
    tabSequences = []
    nbJourConfinement = 0
    confinement = False
    for _ in range(0,numberOfIndiv):
        tabSequences.append([0]) # On considère que tous les individus sont sains au départ, on ne prend pas en compte initialVector
    # On initialise les séquences du tableau
    for t in range(1,T):
        iPercentage = getCurrentIPercentage(tabSequences,t - 1)

        if iPercentage >= 25: # On passe en confinement
            confinement = True
            nbJourConfinement = 0
            matrix = [[0.92,0.0,0.0],[0.0,0.93,0.07],[0.02,0.0,0.98]]

        elif iPercentage <= 10: # On retire le confinement
            matrix = [[0.92,0.08,0.0],[0.0,0.93,0.07],[0.02,0.0,0.98]]
            if(confinement):
                tabNbJourConfinement.append(nbJourConfinement)
                confinement = False
        for sequence in tabSequences:
            currentVector = matrix[sequence[t - 1]]
            sequence.append(vectorDraw(currentVector))
        nbJourConfinement += 1
    # On fait les stats sur le tableau de séquences et on retourne le résultat
    tabStats = []

    for t in range(0,T):

        tabStats.append({0 : 0, 1 : 0, 2 : 0})
        for sequence in tabSequences:
            tabStats[t][sequence[t]] += 1

    return tabStats,tabNbJourConfinement

def getCurrentIPercentage(tabSequences,currentTime):
    """
    Renvoie le pourcentage courant d'infectés

    Args :
    - tabSequences {[[int]]} : le tableau de séquences
    - currentTime {int} : Le temps t courant

    Returns :
    - {float} : Le pourcentage courant d'infectés
    """
    totalNumber = 0
    totalNumberOfI = 0
    for sequence in tabSequences:
        if sequence[currentTime] == 1:
            totalNumberOfI += 1
        totalNumber += 1

    return (float(totalNumberOfI)/totalNumber) * 100





def tracerTheoriqueGraphique(π0, A,T, confinement=False, numberOfIndiv=False):
    """Permet de tracer le graphique théorique de l'evolution de pi0 par rapport à A pendant un temps T

    """
    i = 0
    tabStateS = []  # liste des états de S
    tabStateI = []  # liste des états de I
    tabStateR = []  # liste des états de R
    πn = π0
    # Pour chaque matrice en focntion du temps, on conserve chaque Etat dans les tableaux
    while i < T:

        πn = (getTime1Prob(πn, A))
        tabStateS.append(πn[0])
        tabStateI.append(πn[1])
        tabStateR.append(πn[2])

        i += 1

    # On trace le graphique

    x = range(1, T+1)
    plt.plot(x, tabStateS, label="Etat S")
    plt.plot(x, tabStateI, label="Etat I")
    plt.plot(x, tabStateR, label="Etat R")
    plt.title("Distribution théorique des effectifs")

    plt.legend()

    plt.show()

def tracerReelGraphique(π0, A, T, nombreIndividus, confinement=False):
    """ Permet de tracer le graphique trouvés avec les simulations, avec confinements, ou non
    """
    if (confinement):

        tabStatsOnSequences,tabNbJourConfinement = statsOnSequencesLockDown(π0, A, T, nombreIndividus)
        print("En moyenne, un confinement dure t = " + str(sum(tabNbJourConfinement) / len(tabNbJourConfinement)))
    else:
        tabStatsOnSequences = fromStatsToPercentage(statsOnSequences(π0, A, T, nombreIndividus))
    tabStateI = []
    tabStateR = []
    tabStateS = []

    for t in range(0, T):
            statsTimeT = tabStatsOnSequences[t]
            tabStateS.append(statsTimeT[0])
            tabStateI.append(statsTimeT[1])
            tabStateR.append(statsTimeT[2])

    # On trace le graphique

    x = range(1, T + 1)

    plt.plot(x, tabStateS, label="Etat S")
    plt.plot(x, tabStateI, label="Etat I")
    plt.plot(x, tabStateR, label="Etat R")
    plt.title("Distribution observé sur une population de "+ str(nombreIndividus) + " individus")
    plt.legend()
    plt.show()

