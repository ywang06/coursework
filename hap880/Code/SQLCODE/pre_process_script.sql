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
