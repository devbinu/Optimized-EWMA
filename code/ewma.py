import random
import math
import matplotlib.pyplot as plt
import csv
import sys
import numpy as np
import pandas as pd
import datetime

# Parameters
m=1200
d=[]
maxI=200
npop=25
w1=1
wdamp=0.99
c1=2
c2=2
z=0.48221777744068683
maxI1=50
npop1=40
eps=sys.float_info.epsilon
e=0.0000001
g0=10
alpha=random.uniform(0,1)
alpha1  = np.random.random(1)



# Reading Data
with open('daily-minimum-temperatures-in-me.csv') as csvDataFile :
    csvReader = csv.reader(csvDataFile)
    i=1
    for row in csvReader :
    	if i==m+1 :
    		break
    	d.append(float(row[1]))
    	i=i+1

 # Cost Function of PSO
def cost(d,a,m,n) :
	mse=0
	average=0
	p=0
	w=[]
	for i in range(0,n) :
		x=(1-a)
		x=x**(n-i)
		y=a*x
		w.append(y)
		average=average+y
		p=p+w[i]*d[i]
	for i in range(0,m-n) :
		caverage=float(p/average)
		e=float(d[n+i]-caverage)
		mse=float(mse+float(e*e))
		p = float(p - float(d[i]*float(a*(float((1-a)**(n-1))))))
		p = float(p*(1-a))
		p = float(p + float((d[n+i]*w[n-1])))
	z=float(mse/(m-n))
	return z

check = []
xaxis = []
checkpso = []
checkgsa=[]
methods = ["GSA","PSO","SF","GA"]
iteration = [0,0,0,0]
computation = [0,0,0,0]

def ga(m,alpha) :
	print()
	print("USING GA")
	startTime = datetime.datetime.now()
	def getN(genes):
	        ans = 0
	        for i in range(0,32):
	            if genes[i]:
	                ans = ans + np.power(2,31-i)
	        return ans

	def generate_parent(n):
	    p = int(1)
	    q = int(0.7*n)
	    return np.random.randint(p,q+1)

	def get_fitness(n):
	    return cost(d,z,m,n)

	# Mutate
	def mutate(N):
	    childGenes = np.zeros(32,dtype=bool)
	    for i in range(0,32):
	            childGenes[i]=bool(N&(1<<(31-i)))
	    index  = random.randrange(0,32)
	    childGenes[index] = childGenes[index]^1
	    index  = random.randrange(0,32)
	    childGenes[index] = childGenes[index]^1
	    return getN(childGenes)

	def display(n,guess):
	    timeDiff = datetime.datetime.now()-startTime
	    fitness = get_fitness(guess)
	    print("Iteration: {}\t{}\t{}\t{}".format(n,guess,fitness,timeDiff))

	random.seed()
	bestParent = generate_parent(m)
	bestFitness = get_fitness(bestParent)
	# display(0,bestParent)

	hola = []

	for i in range(0,1000):
	    hola.append(bestParent)
	    print("Iteration: {}\t Best Window Size: {}\t Best Cost: {}".format(i+1,bestParent,bestFitness))
	    child = mutate(bestParent)
	    if child>0.7*m:
	        continue
	    if child<1:
	        continue
	    childFitness = get_fitness(child)
	    if bestFitness<=childFitness:
	        continue
	    # display(i+1,child)
	    bestFitness = childFitness
	    bestParent = child
	# plt.plot(hola)
	# plt.show()


def straight() :
	print()
	print("USING STRAIGHT FORWARD")
	Globalcost=2**1000
	x=int(0.7*m)
	ind=-1
	for i in range(0,m-1) :
		y=cost(d,z,m,i+1)
		if y<Globalcost and i+1<=x :
			Globalcost=y
			ind=i+1
		check.append(y)
		xaxis.append(i+1)
		if i+1<=x :
			print("Iteration: " + str(i) + " ; " + "Bestposition = " + str(ind) + "  Bestcost = " + str(Globalcost))

def pso(w1,c1,c2,maxI,npop,wdamp) :
	print()
	print("USING PSO")
	Globalcost=2**10000
	Globalpos=0
	Globalalpha=0
	class Particle:
	    def __init__(self):
	        self.position=[]
	        self.alpha=[]
	        self.velocity=[] 
	        self.velocity1=[]      
	        self.cost=[]      
	        self.bestcost=[]        
	        self.bestposition=[]
	        self.bestalpha=[]

	s1=Particle()

	for i in range(0,npop) :
	    x=random.randint(1,0.7*m)
	    z1=random.uniform(0,1)
	    s1.position.append(x)
	    s1.alpha.append(z1)
	    s1.velocity.append(0)
	    s1.velocity1.append(0)
	    y=cost(d,z1,m,x)
	    s1.cost.append(y)
	    s1.bestposition.append(x)
	    s1.bestalpha.append(z1)
	    s1.bestcost.append(y)
	    if y<Globalcost :
	        Globalcost=y
	        Globalpos=x
	        Globalalpha=z1


	# Main llop of PSO
	
	for it in range(0,maxI) :
		for i in range(0,npop) :
			r1=random.random()
			r2=random.random()
			s1.velocity[i] = w1*s1.velocity[i] + c1*r1*(s1.bestposition[i] - s1.position[i]) + c2*r2*(Globalpos - s1.position[i])
			s1.velocity1[i] = w1*s1.velocity1[i] + c1*r1*(s1.bestalpha[i] - s1.alpha[i]) + c2*r2*(Globalalpha - s1.alpha[i])
			s1.position[i] = int(s1.position[i] + s1.velocity[i])
			s1.alpha[i] = s1.alpha[i] + s1.velocity1[i]
			s1.position[i] = int(max(s1.position[i],1))
			s1.position[i] = int(min(s1.position[i],0.7*m))
			s1.alpha[i] = max(s1.alpha[i],0.0001)
			s1.alpha[i] = min(s1.alpha[i],0.9999)
			s1.cost[i] = cost(d,s1.alpha[i],m,s1.position[i])
			if s1.cost[i] < s1.bestcost[i] :
				s1.bestcost[i] = s1.cost[i]
				s1.bestposition[i] = s1.position[i]
				s1.bestalpha[i]=s1.alpha[i]
				if s1.bestcost[i] < Globalcost :
					Globalcost = s1.bestcost[i]
					Globalpos = s1.bestposition[i]
					Globalalpha=s1.bestalpha[i]
		checkpso.append(Globalcost)
		x = Globalpos
		y = Globalcost
		z1=Globalalpha
		print("Iteration: " + str(it) + " ; " + "Bestposition = " + str(x) + "  Bestcost = " + str(y) + " Bestalpha = " + str(z1))
		w1 = w1*wdamp
	return Globalcost,Globalpos,Globalalpha


#straight()
mse,n,optalpha = pso(w1,c1,c2,maxI,npop,wdamp)

prediction = []

for i in range(0,n) :
	prediction.append(0)

w=[]


mse=0
mae=0
nmse=0
nmae=0
average=0
p=0
for i in range(0,n) :
	a=optalpha
	x=(1-a)
	x=x**(n-i)
	y=a*x
	w.append(y)
	average=average+y
	p=p+w[i]*d[i]

pmin=2**1000
pmax=0
for i in range(0,m-n) :
	caverage=float(p/average)
	if caverage<pmin and i>0.7*m :
		pmin=caverage
	if caverage>pmax and i>0.7*m :
		pmax=caverage
	prediction.append(caverage)
	e=float(d[n+i]-caverage)
	if i>0.7*m :
		mae =float(mae + float(abs(e)))
		mse=float(mse+float(e*e))
		
	p = float(p - float(d[i]*float(a*(float((1-a)**(n-1))))))
	p = float(p*(1-a))
	p = float(p + float((d[n+i]*w[n-1])))



q=int(0.7*m)
for i in range(0,q) :
	prediction[i]=0

for i in range(q,m) :
	if prediction[i]<pmin :
		pmin=prediction[i]
	if prediction[i]>pmax :
		pmax=prediction[i]
	e=float(d[i]-prediction[i])
	mae =float(mae + float(abs(e)))
	mse=float(mse+float(e*e))

mse=float(mse/(0.3*m))
mae=float(mae/(0.3*m))

nmae=float(mae/(pmax-pmin))
nmse=float(mse/(pmax-pmin))
print(mse)
print(mae)
print(nmse)
print(nmae)


for i in range(0,m) :
	xaxis.append(i+1)


plt.plot(xaxis, d, label = "Actual")

plt.plot(xaxis, prediction, label = "Prediction")

plt.xlabel('x - axis')
plt.ylabel('Values')
plt.title('Actual & Predicted Data Values')
plt.legend()
plt.show()
