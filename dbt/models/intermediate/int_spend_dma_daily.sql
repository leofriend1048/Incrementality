-- Intermediate: combine all channel spend by DMA and date.
-- Unions Meta (split into performance and awareness sub-channels),
-- Google, and TikTok spend into a single (date, dma_code, channel, spend) table.
-- TikTok state-level spend is joined to a DMA crosswalk and allocated by
-- population weight before arriving here; rows with NULL dma_code are excluded.
with meta as (
    select
        date,
        dma_code,
        'meta_perf' as channel,
        spend
    from {{ ref('stg_meta_spend') }}
    where channel = 'performance'
),
meta_aware as (
    select
        date,
        dma_code,
        'meta_aware' as channel,
        spend
    from {{ ref('stg_meta_spend') }}
    where channel = 'awareness'
),
google as (
    select
        date,
        dma_code,
        channel,
        spend
    from {{ ref('stg_google_spend') }}
),
tiktok as (
    select
        date,
        dma_code,
        channel,
        spend
    from {{ ref('stg_tiktok_spend') }}
    where dma_code is not null
),
combined as (
    select * from meta
    union all select * from meta_aware
    union all select * from google
    union all select * from tiktok
)
select
    date,
    dma_code,
    channel,
    sum(spend) as spend
from combined
group by 1, 2, 3
