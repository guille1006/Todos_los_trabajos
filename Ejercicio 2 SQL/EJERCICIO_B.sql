USE prestamos_2015;

SELECT country, 
	COUNT(country) as operaciones, 
    SUM(amount) as valor_total_operaciones,
    MAX(amount) as max_operacion, 
    MIN(amount) as min_operacion
FROM orders
WHERE status NOT IN ("Delinquent", "Cancelled")
	AND amount > 100
GROUP BY country
ORDER BY operaciones DESC
LIMIT 3