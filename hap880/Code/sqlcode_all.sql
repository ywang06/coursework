-- SELECT * INTO yw_diag_most_freq_list FROM (
-- SELECT * FROM yw_icd_most_freq_50 
-- UNION 
-- SELECT * FROM yw_icd_most_freq_5854_50
-- ) as tmp;


-- SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2011 FROM car_clm2011_dgnscd as a, car_clm2011 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2012 FROM car_clm2012_dgnscd as a, car_clm2012 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2013 FROM car_clm2013_dgnscd as a, car_clm2013 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2014 FROM car_clm2014_dgnscd as a, car_clm2014 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2015 FROM car_clm2015_dgnscd as a, car_clm2015 as b, yw_diag_most_freq_list as c, yw_1115_min_s3_s4_positive_6m as d WHERE a.claimno = b.claimno and a.dgnscd = c.dgnscd and CAST(a.dsysrtky as varchar) = CAST(d.dsysrtky as varchar) GROUP BY a.dsysrtky, a.dgnscd;

SELECT * INTO yw_diag_thru_dt_list FROM(
SELECT * FROM yw_temp_diag_thru_dt_list_2011
UNION
SELECT * FROM yw_temp_diag_thru_dt_list_2012
UNION
SELECT * FROM yw_temp_diag_thru_dt_list_2013
UNION
SELECT * FROM yw_temp_diag_thru_dt_list_2014
UNION
SELECT * FROM yw_temp_diag_thru_dt_list_2015
) as tmp; 

SELECT dsysrtky, hcpcs_cd, string_agg(thru_dt_list, ',') AS thru_dt_list INTO yw_diag_thru_dt_list_all FROM yw_diag_thru_dt_list GROUP BY dsysrtky, dgnscd;


\copy yw_diag_thru_dt_list_all to '~/csv/yw_diag_thru_dt_list_all.csv' delimiters',' CSV HEADER;


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


SELECT * INTO yw_car_clm_dgnscd_585_GROUP FROM (
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2007_dgnscd_585_m GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2008_dgnscd_585_m GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2009_dgnscd_585_m GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2010_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2011_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2012_dgnscd_585 GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2013_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2014_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2015_dgnscd_585 GROUP BY  dsysrtky, dgnscd
) as tmp;


SELECT claim_no, CAST(thru_dt as varchar) INTO yw_inp_thru_dt0715 FROM (
SELECT claim_no, CAST(thru_dt as varchar) FROM inp_clm2007a
UNION
SELECT claim_no, CAST(thru_dt as varchar) FROM inp_clm2008a
UNION 
SELECT claim_no, CAST(thru_dt as varchar) FROM inp_clm2009a
UNION
SELECT claim_no, CAST(thru_dt as varchar) FROM inp_clm2010a
UNION
SELECT claimno as claim_no, CAST(thru_dt as varchar) FROM inp_clm2011
UNION
SELECT claimno as claim_no, CAST(thru_dt as varchar) FROM inp_clm2012
UNION
SELECT claimno as claim_no, CAST(thru_dt as varchar) FROM inp_clm2013
UNION
SELECT claimno as claim_no, CAST(thru_dt as varchar) FROM inp_clm2014
UNION
SELECT claimno as claim_no, CAST(thru_dt as varchar) FROM inp_clm2015
) as tmp;



SELECT dsysrtky,claim_no, dgnscd, dgnscd_sequence INTO yw_inp_clm_dgnscd_0715 FROM (
SELECT dsysrtky,claim_no, dgnscd, dgnscd_sequence FROM inp_clm2007_dgnscd
UNION
SELECT dsysrtky,claim_no, dgnscd, dgnscd_sequence FROM inp_clm2008_dgnscd
UNION
SELECT dsysrtky,claim_no, dgnscd, dgnscd_sequence FROM inp_clm2009_dgnscd
UNION
SELECT dsysrtky,claim_no, dgnscd, dgnscd_sequence FROM inp_clm2010_dgnscd
UNION
SELECT CAST (dsysrtky as varchar),claimno as claim_no, dgnscd, dgnscd_sequence FROM inp_clm2011_dgnscd
UNION
SELECT CAST (dsysrtky as varchar),claimno as claim_no, dgnscd, dgnscd_sequence FROM inp_clm2012_dgnscd
UNION
SELECT CAST (dsysrtky as varchar),claimno as claim_no, dgnscd, dgnscd_sequence FROM inp_clm2013_dgnscd
UNION
SELECT CAST (dsysrtky as varchar),claimno as claim_no, dgnscd, dgnscd_sequence FROM inp_clm2014_dgnscd
UNION
SELECT CAST (dsysrtky as varchar),claimno as claim_no, dgnscd, dgnscd_sequence FROM inp_clm2015_dgnscd
) as tmp;

SELECT claim_no, dsysrtky, thru_dt, dgnscd1, dgnscd2, dgnscd3, dgnscd4 FROM yw_inp_clm_dgnscd_0715 WHERE (dgnscd1 like '5853'  or dgnscd2 like '5853' or dgnscd3 like '5853' or dgnscd4 like '5853') and (dgnscd1 like '5854'  or dgnscd2 like '5854' or dgnscd3 like '5854' or dgnscd4 like '5854')  ORDER BY dsysrtky, dgnscd1 LIMIT 100;



SELECT claim_no, dsysrtky, thru_dt, dgnscd1, dgnscd2, dgnscd3, dgnscd4, dgnscd5, dgnscd6,dgnscd7, dgnscd8, dgnscd9, dgnscd10 FROM yw_inp_clm0709a WHERE (dgnscd1 like '5853' ) or (dgnscd1 like '5854')  ORDER BY dsysrtky, claim_no LIMIT 100;


SELECT claim_no, dsysrtky, thru_dt, max(dgnscd1), min(dgnscd1) FROM yw_inp_clm0709a WHERE ((dgnscd1 like '5853' ) or (dgnscd1 like '5854')) and   GROUP BY dsysrtky ORDER BY dsysrtky, claim_no LIMIT 100;

SELECT a.claim_no, b.thru_dt, a.dsysrtky, a.dgnscd, a.dgnscd_sequence INTO yw_inp_clm_with_thru_0715 FROM yw_inp_clm_dgnscd_0715 as a, yw_inp_thru_dt0715 as b WHERE a.claim_no = b.claim_no;


SELECT dsysrtky, max(line_icd_dgns_cd), min(line_icd_dgns_cd) FROM car_line2011 WHERE line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856' GROUP BY dsysrtky LIMIT 100;



SELECT count(DISTINCT dsysrtky) as freq, hcpcs_cd INTO yw_hcpcs_count_5854_2011 FROM yw_hcpcs_585x_2011 WHERE line_icd_dgns_cd like '5854' GROUP BY hcpcs_cd order by freq DESC;

SELECT count(DISTINCT dsysrtky) as freq, hcpcs_cd INTO yw_hcpcs_count_5853_2011 FROM yw_hcpcs_585x_2011 WHERE line_icd_dgns_cd like '5853' GROUP BY hcpcs_cd order by freq DESC;

SELECT a.freq as freq_5853, b.freq as freq_5854, abs(a.freq/48258.0 - b.freq/17711.0) as diff_freq, a. hcpcs_cd INTO yw_hcpcs_comp_5853_5854_2011 FROM yw_hcpcs_count_5853_2011 as a, yw_hcpcs_count_5854_2011 as b WHERE a.hcpcs_cd = b.hcpcs_cd

SELECT count(DISTINCT dsysrtky) from yw_hcpcs_585x_2011 WHERE WHERE line_icd_dgns_cd like '5854';


SELECT freq,  hcpcs_cd FROM  yw_hcpcs_count_5854_2011 l WHERE  NOT EXISTS (
   SELECT freq 
   FROM   yw_hcpcs_count_5853_2011
   WHERE  hcpcs_cd = l.hcpcs_cd
   );


SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_585x_2011 FROM car_line2011 WHERE line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_585x_2012 FROM car_line2012 WHERE line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_585x_2013 FROM car_line2013 WHERE line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_585x_2014 FROM car_line2014 WHERE line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_585x_2015 FROM car_line2015 WHERE line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';


SELECT * INTO yw_hcpcs_585x_1115 FROM (
SELECT CAST(dsysrtky as varchar), thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 FROM yw_hcpcs_585x_2011
UNION
SELECT * FROM yw_hcpcs_585x_2012
UNION
SELECT * FROM yw_hcpcs_585x_2013
UNION
SELECT * FROM yw_hcpcs_585x_2014
UNION
SELECT * FROM yw_hcpcs_585x_2015
) as tmp;



SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_all_585x_2011 FROM car_line2011 WHERE line_icd_dgns_cd like '5851' or line_icd_dgns_cd like '5852' or line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_all_585x_2012 FROM car_line2012 WHERE line_icd_dgns_cd like '5851' or line_icd_dgns_cd like '5852' or line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_all_585x_2013 FROM car_line2013 WHERE line_icd_dgns_cd like '5851' or line_icd_dgns_cd like '5852' or line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_all_585x_2014 FROM car_line2014 WHERE line_icd_dgns_cd like '5851' or line_icd_dgns_cd like '5852' or line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_all_585x_2015 FROM car_line2015 WHERE line_icd_dgns_cd like '5851' or line_icd_dgns_cd like '5852' or line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856';


SELECT * INTO yw_hcpcs_all_585x_1115 FROM (
SELECT CAST(dsysrtky as varchar), thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 FROM yw_hcpcs_all_585x_2011
UNION
SELECT * FROM yw_hcpcs_all_585x_2012
UNION
SELECT * FROM yw_hcpcs_all_585x_2013
UNION
SELECT * FROM yw_hcpcs_all_585x_2014
UNION
SELECT * FROM yw_hcpcs_all_585x_2015
) as tmp;


SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 FROM car_line2013 WHERE (line_icd_dgns_cd like '5851' or line_icd_dgns_cd like '5852' or line_icd_dgns_cd like '5853' or line_icd_dgns_cd like '5854' or line_icd_dgns_cd like '5855'  or line_icd_dgns_cd like '5856' or line_icd_dgns_cd like '5849') and  (dsysrtky like '100001925');
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
DROP TABLE IF EXISTS yw_0710_diag_ccs1;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, b.ccs as ccs1  INTO yw_0710_diag_ccs1 FROM yw_0710_raw as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd1 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs2;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, b.ccs as ccs2  INTO yw_0710_diag_ccs2 FROM yw_0710_diag_ccs1 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd2 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs3;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, b.ccs as ccs3  INTO yw_0710_diag_ccs3 FROM yw_0710_diag_ccs2 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd3 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs4;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, b.ccs as ccs4  INTO yw_0710_diag_ccs4 FROM yw_0710_diag_ccs3 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd4 = b.icd9;

DROP TABLE IF EXISTS yw_0710_diag_ccs5;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, ccs4, b.ccs as ccs5  INTO yw_0710_diag_ccs5 FROM yw_0710_diag_ccs4 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd5 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs6;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, ccs4, ccs5, b.ccs as ccs6  INTO yw_0710_diag_ccs6 FROM yw_0710_diag_ccs5 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd6 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs7;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, ccs4, ccs5, ccs6, b.ccs as ccs7  INTO yw_0710_diag_ccs7 FROM yw_0710_diag_ccs6 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd7 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs8;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, ccs4, ccs5, ccs6, ccs7, b.ccs as ccs8  INTO yw_0710_diag_ccs8 FROM yw_0710_diag_ccs7 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd8 = b.icd9;


DROP TABLE IF EXISTS yw_0710_raw;
SELECT * INTO yw_0710_raw FROM (SELECT claim_no , dsysrtky , state_cd , thru_dt , dgnscd1 , dgnscd2 , dgnscd3 , dgnscd4 , dgnscd5 , dgnscd6 , dgnscd7 , dgnscd8 , hcpscd1 , hcpscd2 , hcpscd3 , hcpscd4 , hcpscd5 , hcpscd6 , hcpscd7 , hcpscd8 , hcpscd9 , hcpscd10 , hcpscd11 , hcpscd12 , hcpscd13  FROM car_clm2007 UNION SELECT claim_no , dsysrtky , state_cd , thru_dt , dgnscd1 , dgnscd2 , dgnscd3 , dgnscd4 , dgnscd5 , dgnscd6 , dgnscd7 , dgnscd8 , hcpscd1 , hcpscd2 , hcpscd3 , hcpscd4 , hcpscd5 , hcpscd6 , hcpscd7 , hcpscd8 , hcpscd9 , hcpscd10 , hcpscd11 , hcpscd12 , hcpscd13  FROM car_clm2008 UNION SELECT claim_no , dsysrtky , state_cd , thru_dt , dgnscd1 , dgnscd2 , dgnscd3 , dgnscd4 , dgnscd5 , dgnscd6 , dgnscd7 , dgnscd8 , hcpscd1 , hcpscd2 , hcpscd3 , hcpscd4 , hcpscd5 , hcpscd6 , hcpscd7 , hcpscd8 , hcpscd9 , hcpscd10 , hcpscd11 , hcpscd12 , hcpscd13  FROM car_clm2009 UNION SELECT claim_no , dsysrtky , state_cd , thru_dt , dgnscd1 , dgnscd2 , dgnscd3 , dgnscd4 , dgnscd5 , dgnscd6 , dgnscd7 , dgnscd8 , hcpscd1 , hcpscd2 , hcpscd3 , hcpscd4 , hcpscd5 , hcpscd6 , hcpscd7 , hcpscd8 , hcpscd9 , hcpscd10 , hcpscd11 , hcpscd12 , hcpscd13  FROM car_clm2010) as tmp;

-- DROP TABLE IF EXISTS yw_0710_diag_ccs1;
-- SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, b.ccs as ccs1  INTO yw_0710_diag_ccs1 FROM yw_0710_raw as a, icd9_ccs_single_level_dx as b WHERE a.dgnscd1 = b.icd9;
-- DROP TABLE IF EXISTS yw_0710_diag_ccs2;
-- SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, b.ccs as ccs2  INTO yw_0710_diag_ccs2 FROM yw_0710_diag_ccs1 as a, icd9_ccs_single_level_dx as b WHERE a.dgnscd2 = b.icd9;
-- DROP TABLE IF EXISTS yw_0710_diag_ccs3;
-- SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2,  b.ccs as ccs3  INTO yw_0710_diag_ccs3 FROM yw_0710_diag_ccs2 as a, icd9_ccs_single_level_dx as b WHERE a.dgnscd3 = b.icd9;
-- DROP TABLE IF EXISTS yw_0710_diag_ccs4;
-- SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2,  ccs3, b.ccs as ccs4  INTO yw_0710_diag_ccs4 FROM yw_0710_diag_ccs3 as a, icd9_ccs_single_level_dx as b WHERE a.dgnscd4 = b.icd9;




DROP TABLE IF EXISTS yw_0710_diag_ccs1;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, b.ccs as ccs1  INTO yw_0710_diag_ccs1 FROM yw_0710_raw as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd1 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs2;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, b.ccs as ccs2  INTO yw_0710_diag_ccs2 FROM yw_0710_diag_ccs1 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd2 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs3;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, b.ccs as ccs3  INTO yw_0710_diag_ccs3 FROM yw_0710_diag_ccs2 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd3 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs4;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, b.ccs as ccs4  INTO yw_0710_diag_ccs4 FROM yw_0710_diag_ccs3 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd4 = b.icd9;

DROP TABLE IF EXISTS yw_0710_diag_ccs5;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, ccs4, b.ccs as ccs5  INTO yw_0710_diag_ccs5 FROM yw_0710_diag_ccs4 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd5 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs6;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, ccs4, ccs5, b.ccs as ccs6  INTO yw_0710_diag_ccs6 FROM yw_0710_diag_ccs5 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd6 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs7;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, ccs4, ccs5, ccs6, b.ccs as ccs7  INTO yw_0710_diag_ccs7 FROM yw_0710_diag_ccs6 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd7 = b.icd9;
DROP TABLE IF EXISTS yw_0710_diag_ccs8;
SELECT  a.claim_no , a.dsysrtky , a.state_cd , a.thru_dt ,a.dgnscd1 , a.dgnscd2 , a.dgnscd3 , a.dgnscd4 , a.dgnscd5 , a.dgnscd6 , a.dgnscd7 , a.dgnscd8, ccs1, ccs2, ccs3, ccs4, ccs5, ccs6, ccs7, b.ccs as ccs8  INTO yw_0710_diag_ccs8 FROM yw_0710_diag_ccs7 as a FULL JOIN icd9_ccs_single_level_dx as b ON a.dgnscd8 = b.icd9;

SELECT dsysrtky, thurdate as yearmonth, count(DISTINCT ccs1) INTO yw_0710_group_id_dt FROM yw_0710_diag_ccs8 GROUP BY dsysrtky, yearmonth;

SELECT DISTINCT(ccs) INTO ccs_list FROM icd9_ccs_single_level_dx;
ALTER TABLE ccs_list ADD COLUMN id SERIAL PRIMARY KEY;

DROP TABLE temp_ccs_id;
SELECT ccs, id,  
case WHEN (id/32) = 8 THEN (2^(id%32))::bigint ELSE 0 end as id_1, 
case WHEN (id/32) = 7 THEN (2^(id%32))::bigint ELSE 0 end as id_2,
case WHEN (id/32) = 6 THEN (2^(id%32))::bigint ELSE 0 end as id_3,
case WHEN (id/32) = 5 THEN (2^(id%32))::bigint ELSE 0 end as id_4,
case WHEN (id/32) = 4 THEN (2^(id%32))::bigint ELSE 0 end as id_5, 
case WHEN (id/32) = 3 THEN (2^(id%32))::bigint ELSE 0 end as id_6, 
case WHEN (id/32) = 2 THEN (2^(id%32))::bigint ELSE 0 end as id_7,
case WHEN (id/32) = 1 THEN (2^(id%32))::bigint ELSE 0 end as id_8,
case WHEN (id/32) = 0 THEN (2^(id%32))::bigint ELSE 0 end as id_9  
INTO temp_ccs_id FROM ccs_list;

DROP TABLE ccs_id_binary;
SELECT ccs, id, 
id_1::bit(32) as id_1_binary,
id_2::bit(32) as id_2_binary,
id_3::bit(32) as id_3_binary,
id_4::bit(32) as id_4_binary,
id_5::bit(32) as id_5_binary,
id_6::bit(32) as id_6_binary,
id_7::bit(32) as id_7_binary,
id_8::bit(32) as id_8_binary,
id_9::bit(32) as id_9_binary
INTO ccs_id_binary FROM temp_ccs_id;

DROP TABLE yw_0710_group_id_dt;
SELECT dsysrtky, thru_dt as yearmonth, count(DISTINCT ccs1) INTO yw_0710_group_id_dt FROM yw_0710_diag_ccs8 GROUP BY dsysrtky, yearmonth;




DROP TABLE IF EXISTS yw_1115_raw;
SELECT * INTO yw_1115_raw FROM (SELECT dsysrtky::varchar  , claimno  ,  thru_dt , carr_num , prncpal_dgns_cd , prncpal_dgns_vrsn_cd , icd_dgns_cd1 , icd_dgns_vrsn_cd1 , icd_dgns_cd2 , icd_dgns_vrsn_cd2 , icd_dgns_cd3 , icd_dgns_vrsn_cd3 , icd_dgns_cd4 , icd_dgns_vrsn_cd4 , icd_dgns_cd5 , icd_dgns_vrsn_cd5 , icd_dgns_cd6, icd_dgns_vrsn_cd6 , icd_dgns_cd7 , icd_dgns_vrsn_cd7 , icd_dgns_cd8 , icd_dgns_vrsn_cd8, icd_dgns_cd9 , icd_dgns_vrsn_cd9 , icd_dgns_cd10 , icd_dgns_vrsn_cd10 , icd_dgns_cd11 ,icd_dgns_vrsn_cd11 , icd_dgns_cd12 , icd_dgns_vrsn_cd12 , dob_dt , gndr_cd , race_cd , cnty_cd , state_cd , cwf_bene_mdcr_stus_cd FROM car_clm2011 
UNION SELECT dsysrtky  , claimno  ,  thru_dt , carr_num , prncpal_dgns_cd , prncpal_dgns_vrsn_cd , icd_dgns_cd1 , icd_dgns_vrsn_cd1 , icd_dgns_cd2 , icd_dgns_vrsn_cd2 , icd_dgns_cd3 , icd_dgns_vrsn_cd3 , icd_dgns_cd4 , icd_dgns_vrsn_cd4 , icd_dgns_cd5 , icd_dgns_vrsn_cd5 , icd_dgns_cd6, icd_dgns_vrsn_cd6 , icd_dgns_cd7 , icd_dgns_vrsn_cd7 , icd_dgns_cd8 , icd_dgns_vrsn_cd8, icd_dgns_cd9 , icd_dgns_vrsn_cd9 , icd_dgns_cd10 , icd_dgns_vrsn_cd10 , icd_dgns_cd11 ,icd_dgns_vrsn_cd11 , icd_dgns_cd12 , icd_dgns_vrsn_cd12 , dob_dt , gndr_cd , race_cd , cnty_cd , state_cd , cwf_bene_mdcr_stus_cd FROM car_clm2012 
UNION SELECT dsysrtky  , claimno  ,  thru_dt , carr_num , prncpal_dgns_cd , prncpal_dgns_vrsn_cd , icd_dgns_cd1 , icd_dgns_vrsn_cd1 , icd_dgns_cd2 , icd_dgns_vrsn_cd2 , icd_dgns_cd3 , icd_dgns_vrsn_cd3 , icd_dgns_cd4 , icd_dgns_vrsn_cd4 , icd_dgns_cd5 , icd_dgns_vrsn_cd5 , icd_dgns_cd6, icd_dgns_vrsn_cd6 , icd_dgns_cd7 , icd_dgns_vrsn_cd7 , icd_dgns_cd8 , icd_dgns_vrsn_cd8, icd_dgns_cd9 , icd_dgns_vrsn_cd9 , icd_dgns_cd10 , icd_dgns_vrsn_cd10 , icd_dgns_cd11 ,icd_dgns_vrsn_cd11 , icd_dgns_cd12 , icd_dgns_vrsn_cd12 , dob_dt , gndr_cd , race_cd , cnty_cd , state_cd , cwf_bene_mdcr_stus_cd FROM car_clm2013
UNION SELECT dsysrtky  , claimno  ,  thru_dt , carr_num , prncpal_dgns_cd , prncpal_dgns_vrsn_cd , icd_dgns_cd1 , icd_dgns_vrsn_cd1 , icd_dgns_cd2 , icd_dgns_vrsn_cd2 , icd_dgns_cd3 , icd_dgns_vrsn_cd3 , icd_dgns_cd4 , icd_dgns_vrsn_cd4 , icd_dgns_cd5 , icd_dgns_vrsn_cd5 , icd_dgns_cd6, icd_dgns_vrsn_cd6 , icd_dgns_cd7 , icd_dgns_vrsn_cd7 , icd_dgns_cd8 , icd_dgns_vrsn_cd8, icd_dgns_cd9 , icd_dgns_vrsn_cd9 , icd_dgns_cd10 , icd_dgns_vrsn_cd10 , icd_dgns_cd11 ,icd_dgns_vrsn_cd11 , icd_dgns_cd12 , icd_dgns_vrsn_cd12 , dob_dt , gndr_cd , race_cd , cnty_cd , state_cd , cwf_bene_mdcr_stus_cd FROM car_clm2014
UNION SELECT dsysrtky  , claimno  ,  thru_dt , carr_num , prncpal_dgns_cd , prncpal_dgns_vrsn_cd , icd_dgns_cd1 , icd_dgns_vrsn_cd1 , icd_dgns_cd2 , icd_dgns_vrsn_cd2 , icd_dgns_cd3 , icd_dgns_vrsn_cd3 , icd_dgns_cd4 , icd_dgns_vrsn_cd4 , icd_dgns_cd5 , icd_dgns_vrsn_cd5 , icd_dgns_cd6, icd_dgns_vrsn_cd6 , icd_dgns_cd7 , icd_dgns_vrsn_cd7 , icd_dgns_cd8 , icd_dgns_vrsn_cd8, icd_dgns_cd9 , icd_dgns_vrsn_cd9 , icd_dgns_cd10 , icd_dgns_vrsn_cd10 , icd_dgns_cd11 ,icd_dgns_vrsn_cd11 , icd_dgns_cd12 , icd_dgns_vrsn_cd12 , dob_dt , gndr_cd , race_cd , cnty_cd , state_cd , cwf_bene_mdcr_stus_cd FROM car_clm2015)as tmp;


-- group data by patients and thru_dt
SELECT dsysrtky, string_agg(distinct(prncpal_dgns_cd), ', ') AS prncpal_dgns_cds, substring(thru_dt::varchar from 1 for 4) as yearno, substring(thru_dt::varchar from 6 for 2) as monthno INTO yw_grouped_diag_fixmonth_distinct FROM yw_1115_raw GROUP BY dsysrtky, yearno, monthno;
DROP TABLE yw_1115_stage3;
SELECT dsysrtky, thru_dt, prncpal_dgns_cd,icd_dgns_cd1,icd_dgns_cd2, icd_dgns_cd3,icd_dgns_cd4,icd_dgns_cd5,icd_dgns_cd6, icd_dgns_cd7,icd_dgns_cd8, icd_dgns_cd9,icd_dgns_cd10,icd_dgns_cd11,icd_dgns_cd12 INTO yw_1115_stage3 FROM yw_1115_raw WHERE (prncpal_dgns_cd like '5853') or (icd_dgns_cd1 like '5853') or (icd_dgns_cd2 like '5853') or (icd_dgns_cd3 like '5853') or (icd_dgns_cd4 like '5853')or (icd_dgns_cd5 like '5853') or (icd_dgns_cd6 like '5853') or (icd_dgns_cd7 like '5853') or (icd_dgns_cd8 like '5853')or (icd_dgns_cd9 like '5853') or (icd_dgns_cd10 like '5853') or (icd_dgns_cd11 like '5853') or (icd_dgns_cd12 like '5853');



DROP TABLE  yw_1115_minthru_stage3;

SELECT dsysrtky, MIN(thru_dt) as min_thru_dt INTO yw_1115_minthru_stage3 FROM yw_1115_stage3 GROUP BY dsysrtky;

-- DROP TABLE yw_1115_full_minthru_stage3;
-- SELECT a.dsysrtky, a.thru_dt, prncpal_dgns_cd,icd_dgns_cd1,icd_dgns_cd2, icd_dgns_cd3,icd_dgns_cd4,icd_dgns_cd5,icd_dgns_cd6, icd_dgns_cd7,icd_dgns_cd8 INTO yw_1115_full_minthru_stage3 FROM yw_1115_stage3 as a, yw_1115_minthru_stage3 as b WHERE a.dsysrtky = b.dsysrtky AND a.thru_dt = b.min_thru_dt;


DROP TABLE yw_1115_stage4;
SELECT dsysrtky, thru_dt, prncpal_dgns_cd,icd_dgns_cd1,icd_dgns_cd2, icd_dgns_cd3,icd_dgns_cd4,icd_dgns_cd5,icd_dgns_cd6, icd_dgns_cd7,icd_dgns_cd8, icd_dgns_cd9,icd_dgns_cd10,icd_dgns_cd11,icd_dgns_cd12 INTO yw_1115_stage4 FROM yw_1115_raw WHERE (prncpal_dgns_cd like '5854') or (icd_dgns_cd1 like '5854') or (icd_dgns_cd2 like '5854') or (icd_dgns_cd3 like '5854') or (icd_dgns_cd4 like '5854')or (icd_dgns_cd5 like '5854') or (icd_dgns_cd6 like '5854') or (icd_dgns_cd7 like '5854') or (icd_dgns_cd8 like '5854')or (icd_dgns_cd9 like '5854') or (icd_dgns_cd10 like '5854') or (icd_dgns_cd11 like '5854') or (icd_dgns_cd12 like '5854');


DROP TABLE  yw_1115_minthru_stage4;

SELECT dsysrtky, MIN(thru_dt) as min_thru_dt INTO yw_1115_minthru_stage4 FROM yw_1115_stage4 GROUP BY dsysrtky;


--join stage 3 and stage 4

SELECT a.dsysrtky, a.min_thru_dt as min_dt_s3, b.min_thru_dt as min_dt_s4 INTO yw_1115_min_s3_s4 FROM yw_1115_minthru_stage3 as a, yw_1115_minthru_stage4 as b WHERE a.dsysrtky = b.dsysrtky;


SELECT *, min_dt_s4-min_dt_s3 as s3_s4_duration FROM yw_1115_min_s3_s4 LIMIT 100;

SELECT *, min_dt_s4-min_dt_s3 as s3_s4_duration INTO yw_1115_min_s3_s4_positive FROM yw_1115_min_s3_s4 WHERE min_dt_s4-min_dt_s3 > 0

SELECT dsysrtky, string_agg(distinct(prncpal_dgns_cd), ', ') AS prncpal_dgns_cds, substring(thru_dt::varchar from 1 for 4) as yearno, substring(thru_dt::varchar from 6 for 2) as monthno INTO yw_grouped_diag_fixmonth_distinct FROM yw_1115_raw GROUP BY dsysrtky, yearno, monthno;

array_to_string(array_agg(projects.name), ',')) as projects
array_agg('[' || friend_id || ',' || confirmed || ']')



DROP TABLE yw_patients_s3_s4;
SELECT DISTINCT dsysrtky INTO yw_patients_s3_s4 FROM yw_1115_min_s3_s4_positive;

DROP TABLE yw_1115_record_patients_s3_s4;
SELECT a.* INTO yw_1115_record_patients_s3_s4 FROM yw_1115_raw as a, yw_patients_s3_s4 as b WHERE a.dsysrtky = b.dsysrtky;

SELECT dsysrtky, string_agg(distinct(prncpal_dgns_cd), ', ') AS prncpal_group,  string_agg(distinct(icd_dgns_cd2), ',' ) AS diags2_group, string_agg(distinct(icd_dgns_cd3), ',' ) AS diags3_group,
string_agg(distinct(icd_dgns_cd4), ',' ) AS diags4_group,
string_agg(distinct(icd_dgns_cd5), ',' ) AS diags5_group,
string_agg(distinct(icd_dgns_cd6), ',' ) AS diags6_group,
string_agg(distinct(icd_dgns_cd7), ',' ) AS diags7_group,
string_agg(distinct(icd_dgns_cd8), ',' ) AS diags8_group,
string_agg(distinct(icd_dgns_cd9), ',' ) AS diags9_group,
string_agg(distinct(icd_dgns_cd10), ',' ) AS diags10_group,
string_agg(distinct(icd_dgns_cd11), ',' ) AS diags11_group,
string_agg(distinct(icd_dgns_cd12), ',' ) AS diags12_group, 
substring(thru_dt::varchar from 1 for 4) as yearno, substring(thru_dt::varchar from 6 for 2) as monthno INTO yw_1115_grouped_diags FROM yw_1115_record_patients_s3_s4 GROUP BY dsysrtky, yearno, monthno;


\copy yw_1115_min_s3_s4_positive to '~/csv/yw_1115_min_s3_s4_positive.csv' delimiters',' CSV HEADER;

\copy yw_1115_grouped_diags to '~/csv/yw_1115_grouped_diags.csv' delimiters',' CSV HEADER;

\copy ccs_id_binary to '~/csv/ccs_id_binary.csv' delimiters',' CSV HEADER;

\copy icd9_ccs_single_level_dx to '~/csv/icd9_ccs_single_level_dx.csv' delimiters',' CSV HEADER;

-- get number of icd9 diags  in yw_1115_record_patients_s3_s4
SELECT * INTO yw_icd9_list_s3_s4 FROM (
  SELECT icd_dgns_cd1 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd2 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd3 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd4 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd5 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd6 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd7 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd8 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd9 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd10 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd11 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT icd_dgns_cd12 as icd_dgns_cd FROM yw_1115_record_patients_s3_s4
) as tmp;

SELECT DISTINCT(icd_dgns_cd) INTO yw_icd9_list_s3_s4_distinct FROM yw_icd9_list_s3_s4;

SELECT * INTO yw_icd9_list_ccs_49_50_156_157_158_161 FROM(
SELECT * FROM icd9_ccs_single_level_dx WHERE ccs like '49'
UNION
SELECT * FROM icd9_ccs_single_level_dx WHERE ccs like '50'
UNION
SELECT * FROM icd9_ccs_single_level_dx WHERE ccs like '156'
UNION
SELECT * FROM icd9_ccs_single_level_dx WHERE ccs like '157'
UNION
SELECT * FROM icd9_ccs_single_level_dx WHERE ccs like '158'
UNION
SELECT * FROM icd9_ccs_single_level_dx WHERE ccs like '161'
) as tmp;




\copy yw_icd9_list_ccs_49_50_156_157_158_161 to '~/csv/yw_icd9_list_ccs_49_50_156_157_158_161.csv' delimiters',' CSV HEADER;


SELECT * INTO yw_1115_min_s3_s4_positive_6m FROM yw_1115_min_s3_s4_positive WHERE min_dt_s3 >= '2011-06-30'::date;

\copy yw_1115_min_s3_s4_positive_6m to '~/csv/yw_1115_min_s3_s4_positive_6m.csv' delimiters',' CSV HEADER;


SELECT a.*, b.min_dt_s3, b.s3_s4_duration INTO yw_1115_grouped_diags_withbilabel FROM yw_1115_grouped_diags as a, yw_1115_min_s3_s4_positive_6m as b WHERE a.dsysrtky = b.dsysrtky;

\copy yw_1115_grouped_diags_withbilabel to '~/csv/yw_1115_grouped_diags_withbilabel.csv' delimiters',' CSV HEADER;


\copy yw_1115_record_patients_s3_s4 to '~/csv/yw_1115_record_patients_s3_s4.csv' delimiters',' CSV HEADER;



SELECT * INTO yw_pid_diags_union_demo FROM (
  SELECT dsysrtky, icd_dgns_cd1 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd2 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd3 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd4 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd5 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd6 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd7 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd8 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd9 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd10 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd11 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
UNION SELECT dsysrtky, icd_dgns_cd12 as diag, thru_dt, dob_dt, gndr_cd, race_cd,state_cd, cnty_cd FROM yw_1115_record_patients_s3_s4
) as tmp;

DROP TABLE yw_pid_diags_dict_demo;
SELECT dsysrtky, diag, min(thru_dt) as min_thru_dt, max(thru_dt) as max_thru_dt, max(dob_dt) as dob_dt, max(gndr_cd) as gndr_cd, max(race_cd) as race_cd,max(state_cd) as state_cd  INTO yw_pid_diags_dict_demo FROM yw_pid_diags_union_demo GROUP BY dsysrtky, diag;

\copy yw_pid_diags_dict_demo to '~/csv/yw_pid_diags_dict_demo.csv' delimiters',' CSV HEADER;



SELECT DISTINCT dsysrtky INTO yw_patient_list_5853_2013 FROM car_clm2013_dgnscd WHERE dgnscd like '5853

SELECT CAST(a.dsysrtky as varchar) as dsysrtky INTO yw_patient_list_5853_1112_join FROM yw_patient_list_5853_2011 as a, yw_patient_list_5853_2012 as b WHERE CAST(a.dsysrtky as varchar) = b.dsysrtky;

SELECT a.dgnscd, count(DISTINCT a.dsysrtky) as p_count INTO yw_freq_diag_of_patient_5853_12 FROM car_clm2012_dgnscd as a, yw_patient_list_5853_2012 as b WHERE (dgnscd_sequence > 0) and (a.dsysrtky = b.dsysrtky) GROUP BY dgnscd;




SELECT DISTINCT dsysrtky INTO yw_patient_list_5854_2011 FROM car_clm2011_dgnscd WHERE dgnscd like '5854';
SELECT DISTINCT dsysrtky INTO yw_patient_list_5854_2012 FROM car_clm2012_dgnscd WHERE dgnscd like '5854';

SELECT CAST(a.dsysrtky as varchar) as dsysrtky INTO yw_patient_list_5854_1112_join FROM yw_patient_list_5854_2011 as a, yw_patient_list_5854_2012 as b WHERE CAST(a.dsysrtky as varchar) = b.dsysrtky;

SELECT a.dgnscd, count(DISTINCT a.dsysrtky) as p_count INTO yw_freq_diag_of_patient_5854_12 FROM car_clm2012_dgnscd as a, yw_patient_list_5854_2012 as b WHERE (dgnscd_sequence > 0) and (a.dsysrtky = b.dsysrtky) GROUP BY dgnscd;

SELECT a.dgnscd, count(DISTINCT a.dsysrtky) as p_count INTO yw_freq_diag_of_patient_5854_11 FROM car_clm2011_dgnscd as a, yw_patient_list_5854_2011 as b WHERE (dgnscd_sequence > 0) and (a.dsysrtky = b.dsysrtky) GROUP BY dgnscd;

SELECT * FROM yw_freq_diag_of_patient_5854_11 ORDER BY p_count DESC LIMIT 100;
SELECT * FROM yw_freq_diag_of_patient_5854_12 ORDER BY p_count DESC LIMIT 100;



SELECT dgnscd INTO yw_icd_most_freq FROM yw_freq_diag_of_patient_5853_11 ORDER BY p_count DESC LIMIT 300;
\copy yw_icd_most_freq to '~/csv/yw_icd_most_freq.csv' delimiters',' CSV HEADER;

SELECT dgnscd INTO yw_icd_most_freq_5854 FROM yw_freq_diag_of_patient_5854_11 ORDER BY p_count DESC LIMIT 300;
\copy yw_icd_most_freq to '~/csv/yw_icd_most_freq_5854.csv' delimiters',' CSV HEADER;

SELECT dsysrtky, diag, min(thru_dt) as min_thru_dt, max(thru_dt) as max_thru_dt, max(dob_dt) as dob_dt, max(gndr_cd) as gndr_cd, max(race_cd) as race_cd,max(state_cd) as state_cd  INTO yw_pid_diags_dict_demo FROM yw_pid_diags_union_demo GROUP BY dsysrtky, diag;
\copy yw_pid_diags_dict_demo to '~/csv/yw_pid_diags_dict_demo.csv' delimiters',' CSV HEADER;


SELECT * INTO yw_car_line1115 FROM(

SELECT CAST(dsysrtky as varchar) as dsysrtky , thru_dt, hcpcs_cd, line_icd_dgns_cd FROM car_line2011
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky , thru_dt, hcpcs_cd, line_icd_dgns_cd FROM car_line2012 
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky , thru_dt, hcpcs_cd, line_icd_dgns_cd  FROM car_line2013
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky , thru_dt, hcpcs_cd, line_icd_dgns_cd  FROM car_line2014
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky , thru_dt, hcpcs_cd, line_icd_dgns_cd  FROM car_line2015 
) as tmp;

SELECT hcpcs_cd, line_icd_dgns_cd INTO yw_hcpcs_most_freq_5853 FROM yw_car_line1115 as a, yw_icd_most_freq as b WHERE a.line_icd_dgns_cd = b.line_icd_dgns_cd;

SELECT hcpcs_cd, line_icd_dgns_cd INTO yw_hcpcs_most_freq_5854 FROM yw_car_line1115 as a, yw_icd_most_freq_5854 as b WHERE a.line_icd_dgns_cd = b.line_icd_dgns_cd;

\copy yw_hcpcs_most_freq_5853 to '~/csv/yw_hcpcs_most_freq_5853.csv' delimiters',' CSV HEADER;

\copy yw_hcpcs_most_freq_5854 to '~/csv/yw_hcpcs_most_freq_5854.csv' delimiters',' CSV HEADER;


ldsbase=> SELECT hcpcs_cd, line_icd_dgns_cd INTO yw_hcpcs_most_freq_5854_2011 FROM car_line2011 as a, yw_icd_most_freq as b WHERE a.line_icd_dgns_cd = b.dgnscd;
SELECT 57555725
ldsbase=> SELECT hcpcs_cd, count(hcpcs_cd) as p_count INTO yw_hcpcs_group_2011 FROM yw_hcpcs_most_freq_5854_2011 GROUP BY hcpcs_cd;
ERROR:  relation "yw_hcpcs_group_2011" already exists
ldsbase=> SELECT hcpcs_cd, count(hcpcs_cd) as p_count INTO yw_hcpcs_group_2011_5854 FROM yw_hcpcs_most_freq_5854_2011 GROUP BY hcpcs_cd;
SELECT 8097
ldsbase=> SELECT hcpcs_cd INTO yw_hcpcs_most_freq_5853 FROM yw_hcpcs_group_2011_5854 ORDER BY p_count DESC LIMIT 100;

ERROR:  relation "yw_hcpcs_most_freq_5853" already exists
ldsbase=> SELECT hcpcs_cd INTO yw_hcpcs_most_freq_5854 FROM yw_hcpcs_group_2011_5854 ORDER BY p_count DESC LIMIT 100;
SELECT 100
ldsbase=> SELECT * FROM SELECT hcpcs_cd INTO yw_hcpcs_most_freq_5853 FROM yw_hcpcs_group_2011 ORDER BY p_count DESC LIMIT 100;


SELECT CAST(a.dsysrtky as varchar) as dsysrtky, thru_dt, a.hcpcs_cd INTO yw_hcpcs_2011 FROM car_line2011 as a, yw_hcpcs_most_freq_5853 as b, yw_1115_min_s3_s4_positive_6m as c WHERE a.hcpcs_cd = b.hcpcs_cd and CAST(a.dsysrtky as varchar) = c.dsysrtky;

SELECT a.* INTO yw_hcpcs_2011_befores3 FROM yw_hcpcs_2011 as a, yw_1115_min_s3_s4_positive_6m as b WHERE a.thru_dt < b.min_dt_s3 and a.dsysrtky = b.dsysrtky;

SELECT dsysrtky, hcpcs_cd, count(hcpcs_cd) INTO yw_hcpcs_2011_count FROM yw_hcpcs_2011_befores3 GROUP BY dsysrtky, hcpcs_cd;


SELECT a.dsysrtky, thru_dt, a.hcpcs_cd INTO yw_hcpcs_2012 FROM car_line2012 as a, yw_hcpcs_most_freq_5853 as b, yw_1115_min_s3_s4_positive_6m as c WHERE a.hcpcs_cd = b.hcpcs_cd and CAST(a.dsysrtky as varchar) = c.dsysrtky;

SELECT a.* INTO yw_hcpcs_2012_befores3 FROM yw_hcpcs_2012 as a, yw_1115_min_s3_s4_positive_6m as b WHERE a.thru_dt < b.min_dt_s3 and a.dsysrtky = b.dsysrtky;

SELECT dsysrtky, hcpcs_cd, count(hcpcs_cd) INTO yw_hcpcs_2012_count FROM yw_hcpcs_2012_befores3 GROUP BY dsysrtky, hcpcs_cd;


SELECT a.dsysrtky, thru_dt, a.hcpcs_cd INTO yw_hcpcs_2013 FROM car_line2013 as a, yw_hcpcs_most_freq_5853 as b, yw_1115_min_s3_s4_positive_6m as c WHERE a.hcpcs_cd = b.hcpcs_cd and CAST(a.dsysrtky as varchar) = c.dsysrtky;

SELECT a.* INTO yw_hcpcs_2013_befores3 FROM yw_hcpcs_2013 as a, yw_1115_min_s3_s4_positive_6m as b WHERE a.thru_dt < b.min_dt_s3 and a.dsysrtky = b.dsysrtky;

SELECT dsysrtky, hcpcs_cd, count(hcpcs_cd) INTO yw_hcpcs_2013_count FROM yw_hcpcs_2013_befores3 GROUP BY dsysrtky, hcpcs_cd;


SELECT a.dsysrtky, thru_dt, a.hcpcs_cd INTO yw_hcpcs_2014 FROM car_line2014 as a, yw_hcpcs_most_freq_5853 as b, yw_1115_min_s3_s4_positive_6m as c WHERE a.hcpcs_cd = b.hcpcs_cd and CAST(a.dsysrtky as varchar) = c.dsysrtky;

SELECT a.* INTO yw_hcpcs_2014_befores3 FROM yw_hcpcs_2014 as a, yw_1115_min_s3_s4_positive_6m as b WHERE a.thru_dt < b.min_dt_s3 and a.dsysrtky = b.dsysrtky;

SELECT dsysrtky, hcpcs_cd, count(hcpcs_cd) INTO yw_hcpcs_2014_count FROM yw_hcpcs_2014_befores3 GROUP BY dsysrtky, hcpcs_cd;


SELECT a.dsysrtky, thru_dt, a.hcpcs_cd INTO yw_hcpcs_2015 FROM car_line2015 as a, yw_hcpcs_most_freq_5853 as b, yw_1115_min_s3_s4_positive_6m as c WHERE a.hcpcs_cd = b.hcpcs_cd and CAST(a.dsysrtky as varchar) = c.dsysrtky;

SELECT a.* INTO yw_hcpcs_2015_befores3 FROM yw_hcpcs_2015 as a, yw_1115_min_s3_s4_positive_6m as b WHERE a.thru_dt < b.min_dt_s3 and a.dsysrtky = b.dsysrtky;

SELECT dsysrtky, hcpcs_cd, count(hcpcs_cd) INTO yw_hcpcs_2015_count FROM yw_hcpcs_2014_befores3 GROUP BY dsysrtky, hcpcs_cd;


SELECT * INTO yw_hcpcs_count_all FROM(
SELECT * FROM yw_hcpcs_2011_count
UNION
SELECT * FROM yw_hcpcs_2012_count
UNION
SELECT * FROM yw_hcpcs_2013_count
UNION
SELECT * FROM yw_hcpcs_2014_count
UNION
SELECT * FROM yw_hcpcs_2015_count
) as tmp;

SELECT dsysrtky,hcpcs_cd,sum(count) INTO yw_dict_pid_hcpcs_count FROM yw_hcpcs_count_all GROUP BY dsysrtky,hcpcs_cd;

\copy yw_dict_pid_hcpcs_count to '~/csv/yw_dict_pid_hcpcs_count.csv' delimiters',' CSV HEADER;

\copy yw_hcpcs_most_freq_5853 to '~/csv/yw_hcpcs_most_freq_5853.csv' delimiters',' CSV HEADER;


\copy yw_icd_most_freq to '~/csv/yw_icd_most_freq.csv' delimiters',' CSV HEADER;

\copy yw_icd_most_freq_5854 to '~/csv/yw_icd_most_freq_5854.csv' delimiters',' CSV HEADER;



DROP TABLE yw_car_clm_dgnscd_585_GROUP;


    
SELECT dsysrtky, claimno, dgnscd, CASE WHEN (thru_dt = 20071000) THEN to_date('20070101','YYYYMMDD') WHEN (thru_dt = 20071000) THEN to_date('20070401','YYYYMMDD') WHEN (thru_dt = 20071000) THEN to_date('20070701','YYYYMMDD') WHEN (thru_dt = 20071000) THEN to_date('20071001','YYYYMMDD') END as thru_dt INTO yw_car_clm2007_dgnscd_585_m FROM yw_car_clm2007_dgnscd_585;

SELECT dsysrtky, claimno, dgnscd, CASE WHEN (thru_dt = 20081000) THEN to_date('20080101','YYYYMMDD') WHEN (thru_dt = 20081000) THEN to_date('20080401','YYYYMMDD') WHEN (thru_dt = 20081000) THEN to_date('20080701','YYYYMMDD') WHEN (thru_dt = 20081000) THEN to_date('20081001','YYYYMMDD') END as thru_dt INTO yw_car_clm2008_dgnscd_585_m FROM yw_car_clm2008_dgnscd_585;

SELECT dsysrtky, claimno, dgnscd, CASE WHEN (thru_dt = 20091000) THEN to_date('20090101','YYYYMMDD') WHEN (thru_dt = 20091000) THEN to_date('20090401','YYYYMMDD') WHEN (thru_dt = 20091000) THEN to_date('20090701','YYYYMMDD') WHEN (thru_dt = 20091000) THEN to_date('20091001','YYYYMMDD') END as thru_dt INTO yw_car_clm2009_dgnscd_585_m FROM yw_car_clm2009_dgnscd_585;





SELECT * INTO yw_car_clm_dgnscd_585_GROUP FROM (
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2007_dgnscd_585_m GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2008_dgnscd_585_m GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2009_dgnscd_585_m GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2010_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2011_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2012_dgnscd_585 GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2013_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2014_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2015_dgnscd_585 GROUP BY  dsysrtky, dgnscd
) as tmp;

DROP TABLE  ;
SELECT dsysrtky, min(min_thru_dt) as min_thru_dt, dgnscd INTO yw_dict_patient_diag_thru FROM yw_car_clm_dgnscd_585_GROUP GROUP BY dsysrtky, dgnscd;
-- 
DROP TABLE yw_car_clm2007_dgnscd_585;
DROP TABLE yw_car_clm2008_dgnscd_585;
DROP TABLE yw_car_clm2009_dgnscd_585;
DROP TABLE yw_car_clm2010_dgnscd_585;
DROP TABLE yw_car_clm2011_dgnscd_585;
DROP TABLE yw_car_clm2012_dgnscd_585;
DROP TABLE yw_car_clm2013_dgnscd_585;
DROP TABLE yw_car_clm2014_dgnscd_585;
DROP TABLE yw_car_clm2015_dgnscd_585;


SELECT a.dsysrtky, a.claim_no as claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2007_dgnscd_585 FROM car_clm2007_dgnscd as a, car_clm2007 as b WHERE dgnscd like '585%' and a.claim_no = b.claim_no; 

SELECT a.dsysrtky, a.claim_no as claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2008_dgnscd_585 FROM car_clm2008_dgnscd as a, car_clm2008 as b WHERE dgnscd like '585%' and a.claim_no = b.claim_no; 

SELECT a.dsysrtky, a.claim_no as claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2009_dgnscd_585 FROM car_clm2009_dgnscd as a, car_clm2009 as b WHERE dgnscd like '585%' and a.claim_no = b.claim_no; 

SELECT a.dsysrtky, a.claim_no as claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2010_dgnscd_585 FROM car_clm2010_dgnscd as a, car_clm2010 as b WHERE dgnscd like '585%' and a.claim_no = b.claim_no; 

SELECT a.dsysrtky, a.claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2011_dgnscd_585 FROM car_clm2011_dgnscd as a, car_clm2011 as b WHERE dgnscd like '585%' and a.claimno = b.claim_no; 

SELECT a.dsysrtky, a.claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2012_dgnscd_585 FROM car_clm2012_dgnscd as a, car_clm2012 as b WHERE dgnscd like '585%' and a.claimno = b.claim_no; 
SELECT a.dsysrtky, a.claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2013_dgnscd_585 FROM car_clm2013_dgnscd as a, car_clm2013 as b WHERE dgnscd like '585%' and a.claimno = b.claim_no; 
SELECT a.dsysrtky, a.claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2014_dgnscd_585 FROM car_clm2014_dgnscd as a, car_clm2014 as b WHERE dgnscd like '585%' and a.claimno = b.claim_no; 
SELECT a.dsysrtky, a.claimno, a.dgnscd, b.thru_dt INTO yw_car_clm2015_dgnscd_585 FROM car_clm2015_dgnscd as a, car_clm2015 as b WHERE dgnscd like '585%' and a.claimno = b.claim_no; 

DROP TABLE yw_car_clm_dgnscd_585_GROUP;






DROP TABLE yw_car_clm2007_dgnscd_585_m;   
SELECT dsysrtky, claimno, dgnscd, CASE WHEN (thru_dt = 20071000) THEN to_date('20070101','YYYYMMDD') WHEN (thru_dt = 20072000) THEN to_date('20070401','YYYYMMDD') WHEN (thru_dt = 20073000) THEN to_date('20070701','YYYYMMDD') WHEN (thru_dt = 20074000) THEN to_date('20071001','YYYYMMDD') END as thru_dt INTO yw_car_clm2007_dgnscd_585_m FROM yw_car_clm2007_dgnscd_585;


DROP TABLE yw_car_clm2008_dgnscd_585_m;   
SELECT dsysrtky, claimno, dgnscd, CASE WHEN (thru_dt = 20081000) THEN to_date('20080101','YYYYMMDD') WHEN (thru_dt = 20082000) THEN to_date('20080401','YYYYMMDD') WHEN (thru_dt = 20083000) THEN to_date('20080701','YYYYMMDD') WHEN (thru_dt = 20084000) THEN to_date('20081001','YYYYMMDD') END as thru_dt INTO yw_car_clm2008_dgnscd_585_m FROM yw_car_clm2008_dgnscd_585;


DROP TABLE yw_car_clm2009_dgnscd_585_m;   

SELECT dsysrtky, claimno, dgnscd, CASE WHEN (thru_dt = 20091000) THEN to_date('20090101','YYYYMMDD') WHEN (thru_dt = 20092000) THEN to_date('20090401','YYYYMMDD') WHEN (thru_dt = 20093000) THEN to_date('20090701','YYYYMMDD') WHEN (thru_dt = 20094000) THEN to_date('20091001','YYYYMMDD') END as thru_dt INTO yw_car_clm2009_dgnscd_585_m FROM yw_car_clm2009_dgnscd_585;





SELECT * INTO yw_car_clm_dgnscd_585_GROUP FROM (
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2007_dgnscd_585_m GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2008_dgnscd_585_m GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2009_dgnscd_585_m GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2010_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2011_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2012_dgnscd_585 GROUP BY dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2013_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2014_dgnscd_585 GROUP BY  dsysrtky, dgnscd
UNION
SELECT CAST(dsysrtky as varchar) as dsysrtky, min(thru_dt) as min_thru_dt, dgnscd FROM yw_car_clm2015_dgnscd_585 GROUP BY  dsysrtky, dgnscd
) as tmp;

DROP TABLE yw_dict_patient_diag_thru;
SELECT dsysrtky, min(min_thru_dt) as min_thru_dt, dgnscd INTO yw_dict_patient_diag_thru FROM yw_car_clm_dgnscd_585_GROUP GROUP BY dsysrtky, dgnscd;

\copy yw_dict_patient_diag_thru to '~/csv/yw_dict_patient_diag_thru.csv' delimiters',' CSV HEADER;

\copy yw_1115_min_s3_s4_positive to '~/csv/yw_1115_min_s3_s4_positive.csv' delimiters',' CSV HEADER;

\copy yw_1115_min_s3_s4_positive_6m to '~/csv/yw_1115_min_s3_s4_positive_6m.csv' delimiters',' CSV HEADER;

\copy yw_dict_patient_diag_thru to '~/csv/yw_dict_patient_diag_thru_all.csv' delimiters',' CSV HEADER;



SELECT a.dsysrtky, a.dgnscd, string_agg(CAST(thru_dt AS varchar), ', ') AS thru_dt_list  INTO yw_temp_diag_thru_dt_list_2011 FROM car_clm2011_dgnscd as a, car_clm2011 as b WHERE a.claimno = b.claim_no GROUP BY a.dsysrtky, a.dgnscd; 




SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_icd__2011 FROM car_line2011;

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_icd__2012 FROM car_line2012;

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_icd__2013 FROM car_line2013;

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_icd__2014 FROM car_line2014;

SELECT dsysrtky, thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2 INTO yw_hcpcs_icd__2015 FROM car_line2015;


SELECT * INTO yw_hcpcs_icd__1115 FROM (
SELECT CAST(dsysrtky as varchar), thru_dt, line_icd_dgns_cd, hcpcs_cd, mdfr_cd1, mdfr_cd2  FROM yw_hcpcs_icd__2011
UNION
SELECT *  FROM yw_hcpcs_icd__2012
UNION
SELECT *  FROM yw_hcpcs_icd__2013
UNION
SELECT *  FROM yw_hcpcs_icd__2014
UNION
SELECT *  FROM yw_hcpcs_icd__2015
) AS tmp

SELECT dsysrtky, string_agg(CAST(thru_dt as varchar), ',') AS thru_dt_list, string_agg(line_icd_dgns_cd, ',') AS dgns_list INTO yw_hcpcs_thru_dngs_grouped_1115 FROM yw_hcpcs_585x_1115 GROUP BY dsysrtky;

