-- Intermediate: Shopify orders aggregated to (date, dma_code) granularity.
-- Orders with no DMA mapping (NULL dma_code) are allocated to 'UNMAPPED'
-- so they are visible in reconciliation queries but excluded from the
-- geo-level MMM tensors downstream.
with orders as (
    select
        date,
        coalesce(dma_code, 'UNMAPPED') as dma_code,
        gross_revenue,
        discounts,
        returns,
        net_revenue,
        units,
        new_customer
    from {{ ref('stg_shopify_orders') }}
),
aggregated as (
    select
        date,
        dma_code,
        sum(gross_revenue) as gross_revenue,
        sum(discounts) as total_discounts,
        sum(returns) as total_returns,
        sum(net_revenue) as net_revenue,
        sum(units) as total_units,
        count(*) as order_count,
        countif(new_customer) as new_customer_orders,
        countif(not new_customer) as returning_customer_orders,
        safe_divide(
            countif(new_customer), count(*)
        ) as new_customer_rate
    from orders
    group by 1, 2
)
select * from aggregated
