USE prestamos_2015;

SELECT country, status, COUNT(*) AS operaciones, AVG(amount) AS importe_medio
FROM orders
WHERE created_at > "2015-07-01"
	AND country IN ("Espana", "Francia", "Portugal")
	AND amount BETWEEN 100 AND 1500
GROUP BY country, status
ORDER BY importe_medio DESC;


