def rango_puntuaciones(collection, feature, min_val=0, max_val=10, grafica=False, portero=True):
    if portero:
        resultado = collection.aggregate([
            {
            "$match": {
                "rating": {"$gte": min_val, "$lte":max_val}
                }
            },
            {
                "$group":{
                    "_id":"$position",
                    "total":{"$sum":1},
                }
            }
        ])
    
    
    else:
        resultado = collection.aggregate([
            {
            "$match": {
                "rating": {"$gte": min_val, "$lte":max_val},
                "position": {"$ne": "G"}
                }
            },
            {
                "$group":{
                    "_id":"$position",
                    "total":{"$sum":1},
                }
            }
        ])
    

    resultado = list(resultado)
    if grafica:
        pie_chart(resultado, min_val, max_val)
    return resultado