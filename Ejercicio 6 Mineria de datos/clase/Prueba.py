def cambio_lista(lista):
    lista = lista[::-1]
    print(lista)
    return lista

fun = lambda x: 2*x
lista = [1,2,3,4,1,2,2,2,2,"iaosjdad", [1,1,1], fun]
print(lista[-1](2))
