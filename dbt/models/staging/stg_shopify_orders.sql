-- Staging: Shopify order-level data from raw ingestion.
-- Each row represents one order. Net revenue = gross_revenue - discounts - returns.
-- Shipping ZIP codes are geocoded to DMA codes during ingestion via the
-- ShopifyConnector → geo.geocode_to_dma() pipeline; dma_code may be NULL for
-- PO boxes, APO/FPO addresses, or orders with missing ZIP data.
with source as (
    select * from {{ source('raw', 'raw_shopify_revenue') }}
),
cleaned as (
    select
        loaded_at,
        date(date) as date,
        order_id,
        dma_code,
        dma_name,
        cast(gross_revenue as float64) as gross_revenue,
        cast(coalesce(discounts, 0) as float64) as discounts,
        cast(coalesce(returns, 0) as float64) as returns,
        cast(net_revenue as float64) as net_revenue,
        cast(coalesce(units, 1) as int64) as units,
        cast(coalesce(new_customer, false) as bool) as new_customer,
        coalesce(product_category, 'unknown') as product_category
    from source
    where
        date >= '2022-01-01'
        and net_revenue > -10000  -- guard against erroneous mega-refunds
        and gross_revenue >= 0
)
select * from cleaned
