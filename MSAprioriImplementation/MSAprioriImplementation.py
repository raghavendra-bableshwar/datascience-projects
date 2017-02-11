import argparse
import collections
from collections import Counter
import itertools
import time

start_time = time.time()
MIS = {}
SDC = 0
cannot_be_together = []
must_have = []
transactions = []
numTransactions = 0
M = []
SC = {}
L = []
supports = {}
F = {}
FwithSupportCounts = {}

def parseMIS(param):
	keyVal = param.split("=")
	key = int(keyVal[0].strip().replace("MIS(","").replace(")",""))
	val = float(keyVal[1].strip())
	#print(key)
	#print(val)
	MIS[key] = val
	
def parseCannotBeTogether(param):
	#print(param)
	global cannot_be_together
	cannot_be_together1 = param.replace("cannot_be_together:","").split("},")
	#print(cannot_be_together1)
	for item in cannot_be_together1:
		#print(item)
		entry = item.replace("{","").replace("}","").split(",")
		itemList = []
		for i in entry:
			itemList.append(int(i.strip()))
		#print(itemList)
		cannot_be_together.append(itemList)
		#print(entry)

def readInputParameters(paramFileName):
	paramList = []
	with open(paramFileName) as f:
		paramList = f.readlines()
	paramList = [x.strip() for x in paramList] 
	#print(paramList)
	for param in paramList:
		#print(param)
		#iterate through each line and parse MIS
		if "MIS" in param:
			#print(param.split("="))
			parseMIS(param)
		elif "SDC" in param:
			#set SDC global param 
			global SDC
			SDC = float(param.split("=")[1].strip())
		elif "cannot_be_together" in param:
			#set cannot_be_together global param
			#cannot_be_together = param.replace("cannot_be_together:","").split("},")
			parseCannotBeTogether(param)
		elif "must-have" in param:
			#set must_have parameter globally
			global must_have
			must_have = param.replace("must-have:","").split("or")
			must_have = [int(x.strip()) for x in must_have] 
	#print("Printing parameters:")
	#print("MIS:"+str(MIS))
	#print(SDC)
	#print("cannot_be_together:"+str(cannot_be_together))
	#print("must_have:"+str(must_have))
	s = [(k, MIS[k]) for k in sorted(MIS, key=MIS.get, reverse=False)]
	MIS1 = {}
	for k, v in s:
		M.append(k)
		#print(str(k)+","+str(v))
	#print("M:"+str(M))
	
def readTransactions(transactionFileName):
	transactionsList = []
	with open(transactionFileName) as f:
		transactionsList = f.readlines()
	transactionsList = [x.strip() for x in transactionsList] 
	for transaction in transactionsList:
		bagOfItems = transaction.replace("{","").replace("}","").split(",")
		bagOfItems = [int(x.strip()) for x in bagOfItems]
		transactions.append(bagOfItems)
	#print("#printing transactions:")
	#print(transactions)
	global numTransactions
	numTransactions = len(transactions)
	#print(numTransactions)
	
def getSupportOfItems():
	for item in MIS:
		supports[item] = SC[item]/numTransactions
	#print("Support of all items:"+str(supports))
	
def getSupportCount():
	result = []
	[ result.extend(t) for t in transactions]
	for item in MIS:
		SC[item] = result.count(item)
	getSupportOfItems()
	#print(SC)
	
def init_pass():
	#get support count of each item
	getSupportCount()
	i=0
	for item in M:
		if supports[item] >= MIS[item]:
			#print(str(supports[item]) + "," + str(MIS[item]))
			L.append(item)
			break
		i+=1
	#print(len(MIS))
	for idx in range(i+1,len(M)):
		#print(M[idx])
		temp = SC[M[idx]]/numTransactions
		#print(str(temp) + "," + str(MIS[L[0]]))
		if temp >= MIS[L[0]]:
			L.append(M[idx])
	#print("L:"+str(L))

def level1_freqItemSet_gen():
	F = []
	for i in L:
		#print(str(SC[i]/numTransactions) + "," + str(MIS[i]))
		#global must_have
		if SC[i]/numTransactions >= MIS[i]:
			#print(str(must_have))
			F.append([i])
	return F

def level2_candidate_gen():
	C2 = []
	#print("L:"+str(L))
	for l in range(len(L)):
		if SC[L[l]]/numTransactions >= MIS[L[l]]:
			for h in range(l+1, len(L)):
				diff = abs(supports[L[h]] - supports[L[l]])
				#print("diff:"+ str(diff))
				sup = SC[L[h]]/numTransactions
				#print(SDC)
				if sup >= MIS[L[l]] and diff <= SDC:
					#print (list([L[l],L[h]]) not in cannot_be_together)
					#print([[L[l],L[h]]])
					C2.append([L[l],L[h]])
	return C2
	
def preProcessF(F):
	tempDict = {}
	sortedPrevF = []
	for i in range(0,len(F)):
		tempDict[i] = F[i][len(F[i])-1]
	s = [(k, tempDict[k]) for k in sorted(tempDict, key=tempDict.get, reverse=False)]
	for key,val in s:
		sortedPrevF.append(F[key])
	return sortedPrevF
	
def compareLists(l1, l2):
	pop1 = l1.pop()
	pop2 = l2.pop()
	retVal = Counter(l1) == Counter(l2)
	l1.append(pop1)
	l2.append(pop2)
	#print("l1:"+str(l1)+" and l2:"+str(l2)+" retVal:"+str(retVal))
	return retVal
	
def getSupportCandidates(c):
	supCountC = 0
	for t in transactions:
		if set(c).issubset(t):
			supCountC += 1
	return supCountC
	
def getSupportOfItemsets(c):
	support = 0
	support = getSupportCandidates(c) / numTransactions
	#print("support of "+str(c)+" = "+str(support))
	return support
	
def findAllSubsets(s,l):
	return list(itertools.combinations(s, l))
	#print("All subsets:"+str(list(itertools.combinations(s, l))))

def sortByMIS(c):
	temp = {}
	for i in c:
		if i in MIS:
			temp[i] = MIS[i]
	s = [(k, temp[k]) for k in sorted(temp, key=temp.get, reverse=False)]
	#print ("temp:"+str(temp))
	newC = []
	for key, val in s:
		newC.append(key)
	#print ("newC:"+str(newC))
	return newC

def MScandidate_gen(prevF):
	Ck = []
	sortedPrevF = preProcessF(prevF)
	for i in range(0,len(sortedPrevF)):
		for j in range(i+1,len(sortedPrevF)):
			diff = abs(supports[sortedPrevF[i][len(sortedPrevF[i])-1]] - supports[sortedPrevF[j][len(sortedPrevF[j])-1]])
			if sortedPrevF[i][len(sortedPrevF[i])-1] < sortedPrevF[j][len(sortedPrevF[j])-1] and compareLists(sortedPrevF[i],sortedPrevF[j]) and diff <= SDC:
				c = []
				c.extend(sortedPrevF[i])
				temp = sortedPrevF[j][len(sortedPrevF[j])-1]
				c.append(temp)
				#if checkConstraint(c) == True:
				#print("Constraint satisfied: Please insert")
				#debugging
				#c = sortByMIS(c)
				Ck.append(c)
				subsets = findAllSubsets(c,len(c)-1)
				#print("inserted:"+str(c)+" temp:"+str(temp))
				for s in subsets:
					if c[0] in list(s) or MIS[c[1]] == MIS[c[0]]:
						#print(str(list(s)) + " in " +str(sortedPrevF))
						if list(s) not in sortedPrevF:
							#print("removed:"+str(c))
							Ck.remove(c)
							break
						#print(c[0])
	#print("Ck:"+str(Ck))
	#print("prevF:"+str(prevF))
	return Ck
	
def checkMustHave(c):
	isMustHaveSatisfied = False
	for item in must_have:
		if item in c:
			isMustHaveSatisfied = True
			break;
	return isMustHaveSatisfied

def checkCannotBeTogether(c,l):
	isCannotBeTogetherStatisfied = True
	for cbt in cannot_be_together:
		#print(set(cbt).issubset(c))
		#print(findAllSubsets(cbt,l))
		l1 = l
		if l > len(cbt):
			l1 = len(cbt)
		i = 2
		while i <= l1:
			subsets = findAllSubsets(cbt,i)
			#print(subsets)
			for s in subsets:
				if set(s).issubset(c):
					isCannotBeTogetherStatisfied = False
					break
			i += 1
			if isCannotBeTogetherStatisfied == False:
				break
		if isCannotBeTogetherStatisfied == False:
			break
	return isCannotBeTogetherStatisfied
	
def applyConstraints():
	#print(F)
	global F
	newF = {}
	for key in F:
		constrainedItems = []
		for item in F[key]:
			if checkMustHave(item) == True:
				if key > 1:
					#should check for cannot be together
					#print("checkCannotBeTogether:"+str(item)+" "+str(checkCannotBeTogether(item,key)))
					if checkCannotBeTogether(item,key) == True:
						constrainedItems.append(sortByMIS(item))
				else:
					constrainedItems.append(item)
		#print (constrainedItems)
		if len(constrainedItems) != 0:
			newF[key] = constrainedItems
	F = newF
	#print(newF)
	
def calculateSupportAndTailCounts():
	global FwithSupportCounts
	for key in F:
		idx = 0
		temp = {}
		for i in F[key]:
			temp[idx] = getSupportCandidates(i)
			idx += 1
		FwithSupportCounts[key] = temp
	
	
def printOutput():
	print("OUTPUT:")
	for key in FwithSupportCounts:
		print("")
		print("Frequent "+str(key)+"-itemsets: ")
		print("")
		temp = FwithSupportCounts[key]
		s = [(k, temp[k]) for k in sorted(temp, key=temp.get, reverse=False)]
		#s = [(k, temp[k]) for k in temp]
		for k, v in s:
			flat = str(F[key][k]).replace('[','').replace(']','')
			print("\t"+str(v)+" : {"+flat+"}")
			if key > 1:
				temp1 = F[key][k].pop(0)
				print("Tailcount = "+str(getSupportCandidates(F[key][k])))
				F[key][k].insert(0,temp1)
				#print("get tail count:"+str(F[key][k]))
		print("\n\tTotal number of frequent "+str(key)+"-itemsets = "+str(len(F[key])))

def MSapriori():
	init_pass()
	k = 1
	F[1] = level1_freqItemSet_gen()
	#F[1] = [[80,30,20,60,50]]
	#print("F1:"+str(F[1]))
	k += 1
	while len(F[k-1]) != 0:
		C = []
		if k == 2:
			#Just for debugging
			#level2_candidate_gen()
			C.extend(level2_candidate_gen())
			#print("C2:"+str(C))
		else:
			C.extend(MScandidate_gen(F[k-1]))
			#print("C"+str(k)+":"+str(C))
		for c in C:
			temp = c.pop(0)
			#print("c.tailCount")
			#print(getSupportCandidates(c))
			c.insert(0,temp)
		if C == []:
			F[k] = []
		for c in C:
			#print(c)
			sup = getSupportCandidates(c)/numTransactions
			#print("c:"+str(c))
			#print(getSupportCandidates(c))
			#print("MIS:"+str(MIS[c[0]]))
			if sup >= MIS[c[0]]:
				if k in F:
					F[k].append(c)
				else:
					F[k] = [c]
		if k not in F:
			F[k] = []
		#print("F"+str(k)+":"+str(F[k]))
		k += 1
	applyConstraints()
	calculateSupportAndTailCounts()
	printOutput()
	#print("F:"+str(F))

def run(paramFile, dataFile):
	#read and parse arguments using arg parse
	readInputParameters(paramFile)
	readTransactions(dataFile)
	MSapriori()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-paramFile', required=True, help='Enter the command')
	parser.add_argument('-dataFile', required=True, help='Enter the search term')
	opts = parser.parse_args()
	#print(opts.paramFile)
	run(opts.paramFile, opts.dataFile)
	#print("--- %s seconds ---" % (time.time() - start_time))