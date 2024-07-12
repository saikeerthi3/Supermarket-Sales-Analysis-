SELECT * FROM supermarket_sales1;

SELECT COLUMN_NAME
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'sql_project' AND TABLE_NAME = 'supermarket_sales1';

ALTER TABLE supermarket_sales1
DROP COLUMN `Invoice ID`;

SELECT 'Invoice ID'
FROM supermarket_sales1;

SELECT *
FROM supermarket_sales1
WHERE Branch IS NULL;

SELECT *
FROM supermarket_sales1
WHERE City IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'Customer type' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'Product line' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'Unit price' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE Quantity IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'Tax 5%' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'Total' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'Date' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'Time' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE payment IS NULL;

SELECT *
FROM supermarket_sales1
WHERE cogs IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'gross margin percentage' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE 'gross income' IS NULL;

SELECT *
FROM supermarket_sales1
WHERE Rating IS NULL;

CREATE TABLE supermarket_2
LIKE supermarket_sales;

INSERT supermarket_2
SELECT * 
FROM supermarket_sales;

SELECT * 
FROM supermarket_2;

WITH duplicate_cte AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY BRANCH, CITY, GENDER, `DATE`, `UNIT PRICE`
           ) AS row_num
    FROM supermarket_2
)
SELECT * FROM duplicate_cte
WHERE row_num>1;

SELECT DISTINCT `Product line`
FROM supermarket_sales1;

SELECT DISTINCT `Customer type`
FROM supermarket_sales1;

SELECT DISTINCT Branch
FROM supermarket_sales1;

SELECT DISTINCT city
FROM supermarket_sales1;

SELECT DISTINCT `Gender`
FROM supermarket_sales1;

SELECT DISTINCT `Payment`
FROM supermarket_sales1;

SELECT Branch, SUM(Total) as Total_Sales
FROM supermarket_sales1
GROUP BY Branch
ORDER BY 2 DESC;

WITH ranked_sales AS (
    SELECT Branch, Gender, SUM(COGS) AS Total_Sales,
           ROW_NUMBER() OVER (PARTITION BY Branch ORDER BY SUM(COGS) DESC) AS row_num
    FROM supermarket_sales1
    GROUP BY Branch, Gender
)
SELECT Branch, Gender AS Gender_with_highest_Sales , Total_Sales
FROM ranked_sales
WHERE row_num = 1;


SELECT City, SUM(Total) AS Total_Sales
FROM supermarket_sales1
GROUP BY City;



SELECT `Product line`, SUM(Total) AS Total_Sales
FROM supermarket_sales1
GROUP BY `Product line`
ORDER BY 2 DESC;



SELECT `Product line` , AVG(Rating)
FROM supermarket_sales1
GROUP BY `Product line`
ORDER BY 2 DESC;

SELECT Payment, SUM(Total) AS Total_Sales
FROM supermarket_sales1
GROUP BY Payment 
ORDER BY Total_Sales DESC;


SELECT `Customer type`, SUM(Total) AS Total_Sales
FROM supermarket_sales1
GROUP BY `Customer type`
ORDER BY Total_Sales DESC;

SELECT *
FROM supermarket_sales1
ORDER BY Total DESC
LIMIT 5;


SELECT Payment, COUNT(*) AS Transaction_Count
FROM supermarket_sales1
GROUP BY Payment;


SELECT MONTH(Date) AS Month, SUM(Total) AS Total_Sales
FROM supermarket_sales1
GROUP BY MONTH(Date);


SELECT MONTH(Date) AS Month
FROM supermarket_sales1;

SELECT DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'supermarket_sales1'
  AND COLUMN_NAME =  'Date';
  
SELECT GENDER, COUNT(*) AS Count
FROM supermarket_sales1
GROUP BY(GENDER)
ORDER BY 2 DESC;

ALTER TABLE supermarket_sales1
ADD COLUMN new_date DATE;

UPDATE supermarket_sales1
SET new_date = STR_TO_DATE(`Date`,'%m/%d/%Y');

SELECT new_date
FROM supermarket_sales1;

SELECT DAYNAME(new_date) AS day_of_week
FROM supermarket_sales1;

SELECT DAYNAME(new_date) AS day_of_week, SUM(Total) AS Total_Sales
FROM supermarket_sales1
GROUP BY day_of_week
ORDER BY 2 DESC;




