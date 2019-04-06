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

