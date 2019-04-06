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
