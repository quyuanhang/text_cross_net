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
  a.geek_id = b.geek_id;

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
  a.job_id = b.job_id;

create temporary table nlp.qyh_interview_sample_geek_job as
  select
    uid geek_id, actionp2 job_id, ds
  from
    dw_bosszp.bg_action
  where
    ds > '2018-10-01' and ds <= '2018-11-01'
  and
    action = 'chat-interview-accept'
  and
    bg = 0
  group by
    uid, actionp2, ds;

create temporary table nlp.qyh_boss_add_friend as
  select
    actionp geek_id, actionp2 job_id, actionp3 expect_id
  from
    dw_bosszp.bg_action
  where
    ds > '2018-10-01' and ds <= '2018-11-01'
  and
    action = 'detail-geek-addfriend'
  and
    bg = 1
  group by
    actionp, actionp2, actionp3;

create temporary table nlp.qyh_geek_add_friend as
  select
    uid geek_id, actionp2 job_id, actionp3 expect_id
  from
    dw_bosszp.bg_action
  where
    ds > '2018-10-01' and ds <= '2018-11-01'
  and
    action = 'detail-geek-addfriend'
  and
    bg = 0
  group by
    uid, actionp2, actionp3;

create temporary table nlp.qyh_add_friend as
  select geek_id, job_id, expect_id
  from nlp.qyh_boss_add_friend
  union all
  select geek_id, job_id, expect_id
  from nlp.qyh_geek_add_friend;

create temporary table nlp.qyh_interview_sample_expect_job as
  select 
    b.expect_id, b.job_id, a.ds
  from 
    nlp.qyh_interview_sample_geek_job a
  join 
    nlp.qyh_add_friend b
  on 
    a.geek_id = b.geek_id and a.job_id = b.job_id
  join
    nlp.expect_profile c
  on
    b.expect_id = c.expect_id
  join
    nlp.job_profile d
  on
    a.job_id = d.job_id;

create temporary table nlp.qyh_interview_sample_train as
  select
    expect_id, job_id
  from 
    nlp.qyh_interview_sample_expect_job
  where
    ds < '2018-10-20'
  distribute by rand() 
  sort by rand()
  limit 100000;

create temporary table nlp.qyh_interview_sample_test as
  select
    expect_id, job_id
  from 
    nlp.qyh_interview_sample_expect_job
  where
    ds >= '2018-10-20'
  distribute by rand() 
  sort by rand()
  limit 10000;

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

insert overwrite local directory 'expect_profile'
  select
    a.*
  from
    nlp.expect_profile a
  join
    nlp.qyh_interview_sample_expect_job b
  on
    a.expect_id = b.expect_id;

insert overwrite local directory 'job_profile'
  select
    a.*
  from
    nlp.job_profile a
  join
    nlp.qyh_interview_sample_expect_job b
  on
    a.job_id = b.job_id;


    