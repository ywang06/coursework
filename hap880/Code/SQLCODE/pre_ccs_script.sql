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