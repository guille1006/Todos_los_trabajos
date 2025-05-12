USE prestamos_2015;

-- Ejercicio A
SELECT country, status, COUNT(*) AS operaciones, AVG(amount) AS importe_medio
FROM orders
WHERE created_at > "2015-07-01"
	AND country IN ("Espana", "Francia", "Portugal")
	AND amount BETWEEN 100 AND 1500
GROUP BY country, status
ORDER BY importe_medio DESC;

-- Ejercicio B
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
LIMIT 3;

-- Ejercicio C
SELECT name, country, COUNT(*) as pedidos
FROM orders JOIN merchants
ON orders.merchant_id = merchants.merchant_id
GROUP BY name, country;