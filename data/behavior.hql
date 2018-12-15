create temporary table nlp.expect_profile as
select
  a.expect_id, a.geek_id, a.l3_name, a.city, 
  a.gender, a.degree, a.fresh_graduate, a.apply_status, a.completion,
  b.cv
from
  nlp.expect_profile_category a
join
  nlp.geek_cv_token b
on
  a.geek_id = b.geek_id
where
  a.l1_name = '技术'
and
  a.is_hr = 0;

create temporary table nlp.job_profile as
select
  a.job_id, a.boss_id, a.position, a.city, a.degree, a.experience, a.area_business_name,
  a.boss_title, a.is_hr, a.stage,
  b.jd
from
  nlp.job_profile_category a
join
  nlp.job_desc_token b
on
  a.job_id = b.job_id
where
  a.l1_code = 100000;

create temporary table nlp.qyh_boss_add_friend as
  select
    actionp2 job_id, actionp3 expect_id, min(ds), 2 action
  from
    dw_bosszp.bg_action a
  join
    nlp.job_profile b
  on
    a.actionp2 = b.job_id
  where
    ds >= '2018-11-01' and ds < '2018-11-22'
  and
    action = 'detail-geek-addfriend'
  and
    bg = 1
  group by
    actionp2, actionp3
  distribute by rand()
  sort by rand()
  limit 100000;

create temporary table nlp.qyh_boss_view as
  select
    actionp3 job_id, actionp2 expect_id, min(ds), 1 action
  from
    dw_bosszp.bg_action a
  join
    nlp.job_profile b
  on
    a.actionp3 = b.job_id    
  where
    ds >= '2018-11-01' and ds < '2018-11-22'
  and
    action = 'detail-geek'
  and
    bg = 1
  group by
    actionp2, actionp3
  limit 100000;

create temporary table nlp.qyh_boss_list as
  select
    jobid job_id, expectid expect_id, min(ds), 0 action
  from
    dw_bosszp.bg_list_action a
  join
    nlp.job_profile b
  on
    a.jobid = b.job_id
  where
    ds >= '2018-11-01' and ds < '2018-11-22'
  and
    bg = 1
  group by
    jobid, expectid
  limit 100000;

create temporary table nlp.qyh_view_add_sample as
  select 
    job_id, expect_id, max(ds), max(action)
  from (
    select
      job_id, expect_id, ds, action
    from
      nlp.qyh_boss_view
    union all
    select
      job_id, expect_id, ds, action
    from
      nlp.qyh_boss_add_friend
    union all
    select
      job_id, expect_id, ds, action
    from
      nlp.qyh_boss_list
  ) a
  group by
    job_id, expect_id;

create temporary table nlp.qyh_interview_sample_train as
  select
    expect_id, job_id
  from 
    nlp.qyh_interview_sample_expect_job
  where
    ds < '2018-11-15';

create temporary table nlp.qyh_interview_sample_test as
  select
    expect_id, job_id
  from 
    nlp.qyh_interview_sample_expect_job
  where
    ds >= '2018-11-15';

insert overwrite local directory 'behavior'
  select
    a.expect_id, a.job_id
  from
    nlp.qyh_interview_sample_expect_job a
  join
    nlp.expect_profile b
  on
    a.expect_id = b.expect_id
  join
    nlp.job_profile c
  on
    a.job_id = c.job_id;

insert overwrite local directory 'train'
  select
    a.expect_id, a.job_id
  from
    nlp.qyh_interview_sample_train a
  join
    nlp.expect_profile b
  on
    a.expect_id = b.expect_id
  join
    nlp.job_profile c
  on
    a.job_id = c.job_id;

insert overwrite local directory 'test'
  select
    a.expect_id, a.job_id
  from
    nlp.qyh_interview_sample_test a
  join
    nlp.expect_profile b
  on
    a.expect_id = b.expect_id
  join
    nlp.job_profile c
  on
    a.job_id = c.job_id;

set hive.cli.print.header=true;
insert overwrite local directory 'expect_profile'
  select
    distinct(a.expect_id), 
    a.geek_id, a.l3_name, a.city, 
    a.gender, a.degree, a.fresh_graduate, 
    a.apply_status, a.completion, a.cv
  from
    nlp.expect_profile a
  join
    nlp.qyh_interview_sample_expect_job b
  on
    a.expect_id = b.expect_id;

insert overwrite local directory 'job_profile'
  select
    distinct(a.job_id), 
    a.boss_id, a.position, a.city, a.degree, 
    a.experience, a.area_business_name, 
    a.boss_title, a.is_hr, a.stage, a.jd
  from
    nlp.job_profile a
  join
    nlp.qyh_interview_sample_expect_job b
  on
    a.job_id = b.job_id;