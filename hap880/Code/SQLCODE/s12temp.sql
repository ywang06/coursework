

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
