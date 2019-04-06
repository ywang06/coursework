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
