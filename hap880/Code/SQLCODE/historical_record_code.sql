
SELECT dsysrtky, thru_dt, prncpal_dgns_cd,icd_dgns_cd1,icd_dgns_cd2, icd_dgns_cd3,icd_dgns_cd4,icd_dgns_cd5,icd_dgns_cd6,icd_dgns_cd7,icd_dgns_cd8 INTO yw_tmp_1000_stage3 FROM yw_1115_raw WHERE (prncpal_dgns_cd like '5853') or (icd_dgns_cd1 like '5853') or (icd_dgns_cd2 like '5853') or (icd_dgns_cd3 like '5853') or (icd_dgns_cd4 like '5853')or (icd_dgns_cd5 like '5853') or (icd_dgns_cd6 like '5853') or (icd_dgns_cd7 like '5853') or (icd_dgns_cd8 like '5853') GROUP BY dsysrtky LIMIT 1000;


SELECT
 film_id,
 title,
 rental_rate
FROM
 film
WHERE
 rental_rate = (
 SELECT
 MIN (rental_rate)
 FROM
 film
 );