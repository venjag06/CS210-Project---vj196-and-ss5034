-- =============================================================================
-- CS210 Rental Fairness :: Database Schema
-- =============================================================================
-- Three tables following the proposal:
--   cities      -- city-level demographics (from US Census ACS)
--   properties  -- one row per physical property (from Craigslist listings)
--   listings    -- one row per price observation (1:1 with properties for
--                  this dataset, but normalized so we *could* store multiple
--                  listings per property over time)
-- =============================================================================

-- Drop in reverse dependency order so re-runs are idempotent.
DROP TABLE IF EXISTS listings      CASCADE;
DROP TABLE IF EXISTS properties    CASCADE;
DROP TABLE IF EXISTS cities        CASCADE;

-- -----------------------------------------------------------------------------
-- cities: city / region -level demographic context
-- -----------------------------------------------------------------------------
-- The Craigslist data uses a "region" field that roughly corresponds to a
-- metro / city area (e.g., "sf bay area", "birmingham", "chicago").  We use
-- (region, state) as the composite primary key and join ACS data to it.
--
-- NOTE: We intentionally keep demographic columns nullable -- not every region
-- in the Craigslist data will have a clean ACS match on the first pass.
-- -----------------------------------------------------------------------------
CREATE TABLE cities (
    region              TEXT NOT NULL,
    state               CHAR(2) NOT NULL,
    median_income       NUMERIC(10, 2),
    population          INTEGER,
    population_density  NUMERIC(10, 2),   -- people per sq mile
    unemployment_rate   NUMERIC(5, 2),    -- percent
    pct_bachelors_plus  NUMERIC(5, 2),    -- percent of adults 25+ with BA+
    PRIMARY KEY (region, state)
);

CREATE INDEX idx_cities_state ON cities(state);

-- -----------------------------------------------------------------------------
-- properties: one row per physical listing in the Craigslist data
-- -----------------------------------------------------------------------------
CREATE TABLE properties (
    property_id              BIGINT PRIMARY KEY,  -- reuse Craigslist 'id'
    region                   TEXT        NOT NULL,
    state                    CHAR(2)     NOT NULL,
    latitude                 DOUBLE PRECISION,
    longitude                DOUBLE PRECISION,
    square_footage           INTEGER,
    bedrooms                 SMALLINT,
    bathrooms                NUMERIC(3, 1),
    property_type            TEXT,                -- apartment, house, condo, ...
    cats_allowed             BOOLEAN,
    dogs_allowed             BOOLEAN,
    smoking_allowed          BOOLEAN,
    wheelchair_access        BOOLEAN,
    electric_vehicle_charge  BOOLEAN,
    comes_furnished          BOOLEAN,
    laundry_options          TEXT,
    parking_options          TEXT,
    -- Soft FK to cities; we don't enforce it at the DB level because some
    -- regions in the Craigslist data won't have ACS matches.
    CONSTRAINT fk_properties_city
        FOREIGN KEY (region, state)
        REFERENCES cities (region, state)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_properties_region_state ON properties(region, state);
CREATE INDEX idx_properties_state        ON properties(state);
CREATE INDEX idx_properties_type         ON properties(property_type);

-- -----------------------------------------------------------------------------
-- listings: observed price for a property
-- -----------------------------------------------------------------------------
-- In the Craigslist dataset each row IS a listing, so this will be 1:1 with
-- properties. We still split it out because (a) it matches the proposal's
-- normalized design, and (b) it would let us extend to multiple listings per
-- property if we re-scraped later.
-- -----------------------------------------------------------------------------
CREATE TABLE listings (
    listing_id       BIGSERIAL PRIMARY KEY,
    property_id      BIGINT  NOT NULL REFERENCES properties(property_id)
                             ON DELETE CASCADE,
    rental_price     NUMERIC(10, 2) NOT NULL,
    listing_url      TEXT,
    CONSTRAINT chk_listings_price_positive CHECK (rental_price > 0)
);

CREATE INDEX idx_listings_property ON listings(property_id);
CREATE INDEX idx_listings_price    ON listings(rental_price);

-- -----------------------------------------------------------------------------
-- Convenience view that flattens everything for analysis.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_listings_enriched AS
SELECT
    l.listing_id,
    l.rental_price,
    p.property_id,
    p.region,
    p.state,
    p.latitude,
    p.longitude,
    p.square_footage,
    p.bedrooms,
    p.bathrooms,
    p.property_type,
    p.cats_allowed,
    p.dogs_allowed,
    p.smoking_allowed,
    p.wheelchair_access,
    p.electric_vehicle_charge,
    p.comes_furnished,
    p.laundry_options,
    p.parking_options,
    c.median_income,
    c.population,
    c.population_density,
    c.unemployment_rate,
    c.pct_bachelors_plus
FROM listings   l
JOIN properties p ON p.property_id = l.property_id
LEFT JOIN cities c ON c.region = p.region AND c.state = p.state;

-- -----------------------------------------------------------------------------
-- Sanity echo
-- -----------------------------------------------------------------------------
\echo 'Schema loaded: cities, properties, listings, v_listings_enriched.'