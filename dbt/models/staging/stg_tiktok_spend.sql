-- Staging: TikTok Ads daily state-level spend from raw ingestion.
-- TikTok does not provide DMA-level targeting; data is ingested at US state
-- granularity and mapped to DMA codes in the intermediate layer using a
-- population-weighted state→DMA crosswalk.
with source as (
    select * from {{ source('raw', 'raw_daily_spend_tiktok') }}
),
renamed as (
    select
        loaded_at,
        date(date) as date,
        -- dma_code is populated post-mapping from state_code; may be NULL for
        -- national-only campaigns that lack geographic breakdowns.
        dma_code,
        dma_name,
        state_code,
        state_name,
        'tiktok' as channel,
        cast(spend as float64) as spend,
        cast(impressions as int64) as impressions,
        cast(video_views as int64) as video_views,
        cast(clicks as int64) as clicks,
        cast(conversions as float64) as conversions,
        cast(cpm as float64) as cpm,
        cast(cpc as float64) as cpc,
        cast(cpv as float64) as cpv
    from source
    where date >= '2022-01-01'
)
select * from renamed
