USE prestamos_2015;

SELECT name, country, COUNT(*) as pedidos
FROM orders JOIN merchants
ON orders.merchant_id = merchants.merchant_id
GROUP BY name, country;