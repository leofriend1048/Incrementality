-- Staging: Amazon SP-API daily revenue from raw ingestion.
-- Amazon revenue is available at national level only; no DMA breakdown is
-- possible from the Selling Partner API. The 'NATIONAL' pseudo-DMA code is
-- used throughout the mart layer to represent Amazon revenue.
with source as (
    select * from {{ source('raw', 'raw_amazon_revenue') }}
),
cleaned as (
    select
        loaded_at,
        date(date) as date,
        coalesce(asin, 'ALL') as asin,
        coalesce(product_name, 'unknown') as product_name,
        cast(gross_revenue as float64) as gross_revenue,
        cast(coalesce(returns, 0) as float64) as returns,
        cast(net_revenue as float64) as net_revenue,
        cast(coalesce(units, 0) as int64) as units,
        coalesce(marketplace, 'US') as marketplace
    from source
    where
        date >= '2022-01-01'
        and gross_revenue >= 0
        and marketplace = 'US'
)
select * from cleaned
