-- Final Meridian-ready spend + revenue + feature table
-- One row per (date, dma_code)
-- This table is the direct input to BigQueryWarehouse.get_spend_tensor() and
-- get_revenue_matrix(). It is also exported as a TFRecord tensor bundle by
-- the weekly refit pipeline.
with spend as (
    select
        date,
        dma_code,
        sum(case when channel = 'meta_perf'       then spend else 0 end) as meta_perf_spend,
        sum(case when channel = 'meta_aware'       then spend else 0 end) as meta_aware_spend,
        sum(case when channel = 'google_brand'     then spend else 0 end) as google_brand_spend,
        sum(case when channel = 'google_nonbrand'  then spend else 0 end) as google_nonbrand_spend,
        sum(case when channel = 'tiktok'           then spend else 0 end) as tiktok_spend,
        sum(case when channel = 'amz_sponsored'    then spend else 0 end) as amz_sponsored_spend,
        sum(case when channel = 'email_sms'        then spend else 0 end) as email_sms_spend
    from {{ ref('int_spend_dma_daily') }}
    group by 1, 2
),
shopify_rev as (
    select
        date,
        dma_code,
        net_revenue as shopify_net_revenue
    from {{ ref('int_shopify_dma_daily') }}
    where dma_code != 'UNMAPPED'
),
amazon_rev as (
    select
        date,
        'NATIONAL' as dma_code,
        net_revenue as amazon_net_revenue
    from {{ ref('int_amazon_daily') }}
)
select
    s.date,
    s.dma_code,
    coalesce(s.meta_perf_spend,      0) as meta_perf_spend,
    coalesce(s.meta_aware_spend,     0) as meta_aware_spend,
    coalesce(s.google_brand_spend,   0) as google_brand_spend,
    coalesce(s.google_nonbrand_spend,0) as google_nonbrand_spend,
    coalesce(s.tiktok_spend,         0) as tiktok_spend,
    coalesce(s.amz_sponsored_spend,  0) as amz_sponsored_spend,
    coalesce(s.email_sms_spend,      0) as email_sms_spend,
    coalesce(r.shopify_net_revenue,  0) as shopify_net_revenue
from spend s
left join shopify_rev r using (date, dma_code)
order by s.date, s.dma_code
