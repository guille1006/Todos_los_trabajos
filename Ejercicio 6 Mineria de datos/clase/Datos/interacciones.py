import itertools

 #======Iteraciones 2 a 2 de una lista=========================================
 interacciones_unicas = list(itertools.combinations(list1, 2))
 #=============================================================================


 #======Iteraciones 3 a 3 de una lista=========================================
 interacciones_unicas = list(itertools.combinations(list1, 3))
 #=============================================================================
 
 #======Iteraciones n a n de una lista=========================================
 interacciones_unicas = list(itertools.combinations(list1, n))
 #=============================================================================


# == Interacciones entre 2 listas de variables distintas ======================
# Genero interacciones:
 interacciones = list(itertools.product(list1, list2))   
# itertools.product genera duplicados
# eliminamos duplicados
 interacciones_unicas = []
for x in interacciones:
    if (sorted(x) not in [sorted(t) for t in interacciones_unicas]) and (x[0] != x[1]):
        interacciones_unicas.append(x)
# =============================================================================

# == Interacciones entre 2 listas de variables distintas + todas las ==========
# == combinaciones 2 a 2 de la segunda lista ==================================
# Genero interacciones entre listas distintas:
interacciones = list(itertools.product(list1, list2))   
# itertools.product genera duplicados
# eliminamos duplicados
 interacciones_unicas = []
 for x in interacciones:
     if (sorted(x) not in [sorted(t) for t in interacciones_unicas]) and (x[0] != x[1]):
         interacciones_unicas.append(x)
#Agrego las combinaciones 2 a 2 de la segunda lista
 interacciones_unicas = interacciones_unicas +list(itertools.combinations(list2, 2))
