-- Staging: Google Ads daily DMA spend from raw ingestion.
-- Channel values: 'google_brand' for branded search campaigns,
-- 'google_nonbrand' for non-branded search, 'google_display' for display/pmax.
with source as (
    select * from {{ source('raw', 'raw_daily_spend_google') }}
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
        cast(clicks as int64) as clicks,
        cast(conversions as float64) as conversions,
        cast(conversion_value as float64) as conversion_value,
        cast(cpc as float64) as cpc,
        cast(cpm as float64) as cpm
    from source
    where date >= '2022-01-01'
)
select * from renamed
