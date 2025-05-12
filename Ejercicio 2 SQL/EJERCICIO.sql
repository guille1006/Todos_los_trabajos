USE prestamos_2015;
DESCRIBE orders;
SELECT * FROM orders;
SELECT country, status, COUNT(*) AS cuantos, AVG(amount) AS media
FROM orders
WHERE created_at > "2015-07-01"
	AND country IN ("Espana", "Francia", "Portugal")
	AND amount BETWEEN 100 AND 1500
GROUP BY country, status
ORDER BY media DESC;


