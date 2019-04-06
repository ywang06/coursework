
SELECT dsysrtky, hcpcs_cd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list INTO yw_temp_hcpcs_thru_dt_list_2011 FROM yw_hcpcs_2011 GROUP BY dsysrtky, hcpcs_cd;

SELECT dsysrtky, hcpcs_cd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list INTO yw_temp_hcpcs_thru_dt_list_2012 FROM yw_hcpcs_2012 GROUP BY dsysrtky, hcpcs_cd;

SELECT dsysrtky, hcpcs_cd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list INTO yw_temp_hcpcs_thru_dt_list_2013 FROM yw_hcpcs_2013 GROUP BY dsysrtky, hcpcs_cd;

SELECT dsysrtky, hcpcs_cd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list INTO yw_temp_hcpcs_thru_dt_list_2014 FROM yw_hcpcs_2014 GROUP BY dsysrtky, hcpcs_cd;

SELECT dsysrtky, hcpcs_cd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list INTO yw_temp_hcpcs_thru_dt_list_2015 FROM yw_hcpcs_2015 GROUP BY dsysrtky, hcpcs_cd;

SELECT * INTO yw_hcpcs_thru_dt_list FROM(
SELECT * FROM yw_temp_hcpcs_thru_dt_list_2011
UNION
SELECT * FROM yw_temp_hcpcs_thru_dt_list_2012
UNION
SELECT * FROM yw_temp_hcpcs_thru_dt_list_2013
UNION
SELECT * FROM yw_temp_hcpcs_thru_dt_list_2014
UNION
SELECT * FROM yw_temp_hcpcs_thru_dt_list_2015
) as tmp; 

SELECT dsysrtky, hcpcs_cd, string_agg(thru_dt_list, ',') AS thru_dt_list INTO yw_hcpcs_thru_dt_list_all FROM yw_hcpcs_thru_dt_list GROUP BY dsysrtky, hcpcs_cd;


\copy yw_hcpcs_thru_dt_list_all to '~/csv/yw_hcpcs_thru_dt_list_all.csv' delimiters',' CSV HEADER;


\copy yw_hcpcs_thru_dt_list to '~/csv/yw_hcpcs_thru_dt_list.csv' delimiters',' CSV HEADER;


