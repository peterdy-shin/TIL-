# 집합

셋트는 mutable 함.
- set 자체가 세트의 원소가 될수 없음. immutable한 frozen set 만 가능.  


set ([1, 2, 3, 3, 2]) #중복된 숫자는 제거됨 

set ( [ [1, 2, 3], 3, 4] ) # [1, 2, 3] 과 같은 리스트는 들어갈 수 없음.


# Cardinality 

	len(A) #shows the size of the set, the number of element.
      		 #Also it can't show the size of this  { x :0 < x =< 1 }


# EG
	A1 = set ([1,2,3,4])
	A2 = set ([2,4,6])
	A3 = set ([1,2,3])
	A4 = set ([2,3,4,5,6])

# Union & Intersection

	A1.union(A2)  
	or  
	A2 | A1  #Union

result : {1,2,3,4,6}

	A3.intersection(A4) 
	or 
	A3 & A4   #Intersection

result : {2,3}

#issubset method

	A3.issubset(A1)    #A3 is subset of A1

result : True  
	or 	
	A3 <== A1

#difference (차집합)

	A1 - A2  #A1에만 있는 원소 
	or
	A1.difference(A2)

result : {1, 3}


#null set
	
	empty_set = set([])
