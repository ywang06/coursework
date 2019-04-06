-- SELECT * INTO yw_diag_most_freq_list FROM (
-- SELECT * FROM yw_icd_most_freq_50 
-- UNION 
-- SELECT * FROM yw_icd_most_freq_5854_50
-- ) as tmp;





SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2011 FROM car_clm2011_dgnscd as a, car_clm2011 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2012 FROM car_clm2012_dgnscd as a, car_clm2012 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2013 FROM car_clm2013_dgnscd as a, car_clm2013 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2014 FROM car_clm2014_dgnscd as a, car_clm2014 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2015 FROM car_clm2015_dgnscd as a, car_clm2015 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT CAST (dsysrtky as varchar) as dsysrtky, dgnscd,thru_dt_list INTO yw_temp_diag_thru_dt_list_2011_m FROM yw_temp_diag_thru_dt_list_2011;

SELECT * INTO yw_diag_thru_dt_list FROM(
SELECT * FROM yw_temp_diag_thru_dt_list_2011_m
UNION
SELECT * FROM yw_temp_diag_thru_dt_list_2012
UNION
SELECT * FROM yw_temp_diag_thru_dt_list_2013
UNION
SELECT * FROM yw_temp_diag_thru_dt_list_2014
UNION
SELECT * FROM yw_temp_diag_thru_dt_list_2015
) as tmp; 

SELECT dsysrtky, dgnscd, string_agg(thru_dt_list, ',') AS thru_dt_list INTO yw_diag_thru_dt_list_all FROM yw_diag_thru_dt_list GROUP BY dsysrtky, dgnscd;


\copy yw_diag_thru_dt_list_all to '~/csv/yw_diag_thru_dt_list_all.csv' delimiters',' CSV HEADER;
