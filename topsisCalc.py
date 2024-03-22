import numpy as np 


def normaliseMatrix(matrix): # returns normalised decision matrix
    #sanitize input
    if type(matrix) != np.ndarray:matrix = np.array(matrix) 

    dim = matrix.shape # get dimension of the matrix
    normalMatrix = np.empty(dim) # empty matrix for the normalised values
    n,x = dim

    for i in range(n): # row (RAT)
        for j in range(x): # col (criteria)
            col = matrix[:,j] # get column j
            m = matrix[i,j] # get the value of the matrix at (i,j)
            normalMatrix[i,j] = m/np.sqrt(np.sum(np.square(col)))
    
    #print("Normalised Matrix:")
    #print(normalMatrix)
    #print("\n")
    return normalMatrix

def normaliseWeight(weight): # returns normalised weight
    w = weight/np.sum(weight)
    #print("Normalised Weight:")
    #print(w)
    #print("\n")
    return w

def H(normalMatrix,normalWeight): # returns weighted normalised decision matrix
    H = normalMatrix * normalWeight
    #print("Weighted Normalised Matrix:")
    #print(H)
    #print("\n")
    return H 

# criteriaBenefi represent a vector with boolean value where each boolean corresponds
# with the criteria, true means is a benefit criteria and false means is a negative  
def solution(h,criteriaBenefit): # return two list one for ideal solution the other for negative ideal
    if len(criteriaBenefit)!=h.shape[1]:
        return 0

    x = len(criteriaBenefit)

    aBest = np.zeros(x) 
    aWorst = np.zeros(x)

    for j in range(x):
        col = h[:,j]
        aBest[j] = np.max(col) if criteriaBenefit[j] else np.min(col)
        aWorst[j] = np.min(col) if criteriaBenefit[j] else np.max(col)
    
    #print("A (ideal solution):")
    #print(aBest)
    #print("\n")
    #print("A (negative ideal solution):")
    #print(aWorst)
    #print("\n")
    return aBest, aWorst

def coefficient(h,solution):
    n = h.shape[0] # get the number or RAT
    d = np.zeros(n) # the ideal(positive or negative) distance for each RAT

    for i in range(n):
        row = h[i,:] # get row i
        d[i]=np.sqrt(np.sum(np.square(row-solution)))
    
    #print(d)
    return d

def closenessCoeff(posDistance,negDistance):
    return negDistance/(posDistance+negDistance)

def TOPSIS(matrix, weight, criteriaBenefit):
    n,x = matrix.shape
    closeCoeff = np.zeros(n)

    # normalised matrix and weight
    normalMatrix = normaliseMatrix(matrix)
    normalWeight = normaliseWeight(weight)

    # weighted normalised decision matrix
    normalWeightMatrix = H(normalMatrix,normalWeight)

    idealSol, negIdealSol = solution(normalWeightMatrix,criteriaBenefit)
    #print("Positive Coeffiecient:")
    posD = coefficient(normalWeightMatrix,idealSol)
    #print("\n")
    #print("Negative Coeffiecient:")
    negD = coefficient(normalWeightMatrix,negIdealSol)
    

    for i in range(n):
        closeCoeff[i]=closenessCoeff(posD[i],negD[i])
    #print("\n")
    #print("Closeness Coefficient:")
    return closeCoeff

# Example
m = np.array([[5,50,5,3,0.1],[8,20,100,5,0.15],[9,1,1000,1,0.5]])
w = np.array((8,2,5,1,6))
criteriaBenefit = np.array((False,False,True,True,False))

#print(TOPSIS(m,w,criteriaBenefit))


RATfreq = np.zeros(m.shape[0]) # RAT frequency = how many times is chosen 

for u in range(100):
    
    #w = np.random.randint(0,9,5)
    w = np.ones(5)

    # effects of reducing scale on criteria 1,2,3
    w[2]=np.random.randint(6,9)

    f = TOPSIS(m,w,criteriaBenefit)
    maxf = f==np.max(f) 

    RATfreq += maxf*1 

print(RATfreq)



