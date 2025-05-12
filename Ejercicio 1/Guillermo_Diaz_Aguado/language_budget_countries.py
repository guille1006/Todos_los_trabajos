from mrjob.job import MRJob

class MRcountry_budget(MRJob):

    def mapper(self, _, line: list[any]):
        '''
        mapper: Genera para cada linea introducida un tuple con el lenguaje
        y luego una lista que contenga el pais y el presupuesto

        Parameters
        ----------
        line:
            lista con elementos que se va a procesar
        
        Yield
        -------
        Tuple:
            Texto que especifica el idioma
            Lista con el pais y el presupuesto
        '''
        line = line.split("|")
        if not line[1] or not line[3] or int(line[4])==-1:
            return
        else:
            language, country, budget = line[1], line[3], int(line[4])
            yield language, [country, budget]



    def reducer(self, key, values):
        '''
        reducer: añade a la lista el valor del pais y aumenta en el 
        presupuesto el valor de este presupuesto

        Parameters
        ----------
        key:
            String que representa el idioma
        values:
            Lista donde se encuentra el pais y el presupuesto

        Yield
        -----
        Tuple:
            Key: idioma
            Lista con todos los paises y la suma del presupuesto
        '''
        total_budget = 0
        countries = set()
        for country, budget in values:
            total_budget += budget
            countries.add(country)
        yield key, [list(countries), total_budget]

if __name__ == '__main__':
    MRcountry_budget.run()