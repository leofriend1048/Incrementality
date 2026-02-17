-- Staging: Meta daily DMA spend from raw ingestion
with source as (
    select * from {{ source('raw', 'raw_daily_spend_meta') }}
),
renamed as (
    select
        loaded_at,
        date(date) as date,
        dma_code,
        dma_name,
        channel,
        cast(spend as float64) as spend,
        cast(impressions as int64) as impressions,
        cast(reach as int64) as reach,
        cast(frequency as float64) as frequency,
        cast(cpm as float64) as cpm
    from source
    where date >= '2022-01-01'
)
select * from renamed
