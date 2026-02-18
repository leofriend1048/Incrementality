-- Intermediate: Amazon SP-API revenue aggregated to daily national totals.
-- Because Amazon does not provide sub-national geographic data, all revenue
-- is represented at the national level. The 'NATIONAL' pseudo-DMA code is
-- used as a placeholder throughout the mart layer and is handled specially
-- in the Meridian tensor builder, which treats Amazon revenue as a
-- national-level covariate rather than a geo-level outcome.
with daily_agg as (
    select
        date,
        sum(gross_revenue) as gross_revenue,
        sum(returns) as total_returns,
        sum(net_revenue) as net_revenue,
        sum(units) as total_units,
        count(distinct asin) as distinct_asins
    from {{ ref('stg_amazon_revenue') }}
    group by 1
),
with_pseudo_geo as (
    select
        date,
        'NATIONAL' as dma_code,
        gross_revenue,
        total_returns,
        net_revenue,
        total_units,
        distinct_asins
    from daily_agg
)
select * from with_pseudo_geo
