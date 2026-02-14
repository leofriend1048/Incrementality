"""US Designated Market Area (DMA) definitions and utilities.

Contains the ~210 Nielsen DMAs with population, state, and region data.
Used for geo holdout test cell assignment and matching.
"""

from __future__ import annotations

import pandas as pd

from incrementality.models import DMAProfile

# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------
_STATE_TO_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast", "DE": "Northeast", "MD": "Northeast", "DC": "Northeast",
    "AL": "Southeast", "AR": "Southeast", "FL": "Southeast", "GA": "Southeast",
    "KY": "Southeast", "LA": "Southeast", "MS": "Southeast", "NC": "Southeast",
    "SC": "Southeast", "TN": "Southeast", "VA": "Southeast", "WV": "Southeast",
    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest", "KS": "Midwest",
    "MI": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",
    "AZ": "West", "CA": "West", "CO": "West", "HI": "West", "ID": "West",
    "MT": "West", "NV": "West", "NM": "West", "OR": "West", "UT": "West",
    "WA": "West", "WY": "West", "AK": "West",
    "OK": "Southwest", "TX": "Southwest",
}


def _region(state: str) -> str:
    return _STATE_TO_REGION.get(state, "Other")


# ---------------------------------------------------------------------------
# DMA definitions — top 210 US DMAs (Nielsen 2024 estimates)
# Each tuple: (dma_code, name, state, population, tv_households)
# ---------------------------------------------------------------------------
_DMA_DATA: list[tuple[str, str, str, int, int]] = [
    ("501", "New York", "NY", 20140470, 7640990),
    ("803", "Los Angeles", "CA", 13211027, 5529560),
    ("602", "Chicago", "IL", 9458539, 3484440),
    ("504", "Philadelphia", "PA", 6228050, 2926460),
    ("807", "San Francisco-Oakland-San Jose", "CA", 4700000, 2607670),
    ("511", "Washington DC", "DC", 6385162, 2538940),
    ("623", "Dallas-Fort Worth", "TX", 7637387, 2646320),
    ("618", "Houston", "TX", 7122240, 2478750),
    ("506", "Boston", "MA", 4941119, 2437650),
    ("524", "Atlanta", "GA", 6144050, 2528510),
    ("539", "Tampa-St. Petersburg", "FL", 3175275, 1943210),
    ("753", "Phoenix", "AZ", 4845832, 1967220),
    ("819", "Seattle-Tacoma", "WA", 4018762, 2014840),
    ("505", "Detroit", "MI", 4392041, 1850860),
    ("613", "Minneapolis-St. Paul", "MN", 3690261, 1860020),
    ("528", "Miami-Fort Lauderdale", "FL", 6138333, 1712670),
    ("751", "Denver", "CO", 2963821, 1743220),
    ("510", "Cleveland-Akron", "OH", 3013108, 1478830),
    ("508", "Pittsburgh", "PA", 2324743, 1167240),
    ("534", "Orlando-Daytona Beach", "FL", 3255800, 1688130),
    ("527", "Indianapolis", "IN", 2111040, 1123640),
    ("548", "West Palm Beach-Fort Pierce", "FL", 1981000, 889730),
    ("617", "Milwaukee", "WI", 1574731, 910990),
    ("825", "San Diego", "CA", 3298634, 1254660),
    ("820", "Portland OR", "OR", 2492412, 1200060),
    ("616", "Kansas City", "MO", 2192035, 988420),
    ("659", "Nashville", "TN", 1989519, 1052310),
    ("533", "Hartford-New Haven", "CT", 1900285, 1011410),
    ("512", "Baltimore", "MD", 2844510, 1114040),
    ("560", "Raleigh-Durham", "NC", 1869858, 1148090),
    ("517", "Charlotte", "NC", 2660329, 1178830),
    ("641", "San Antonio", "TX", 2558143, 1010420),
    ("542", "Dayton", "OH", 834000, 493610),
    ("532", "Albany-Schenectady-Troy", "NY", 1170483, 575510),
    ("544", "Norfolk-Portsmouth-Newport News", "VA", 1799674, 751300),
    ("529", "Louisville", "KY", 1395634, 690910),
    ("518", "Greensboro-High Point-Winston Salem", "NC", 1655079, 730940),
    ("563", "Grand Rapids-Kalamazoo", "MI", 1403690, 719810),
    ("515", "Cincinnati", "OH", 2256884, 963750),
    ("567", "Greenville-Spartanburg", "SC", 1480000, 907650),
    ("577", "Wilkes Barre-Scranton", "PA", 674000, 391340),
    ("561", "Jacksonville", "FL", 1605030, 795190),
    ("566", "Harrisburg-Lancaster-Lebanon-York", "PA", 1573000, 754180),
    ("513", "Flint-Saginaw-Bay City", "MI", 643000, 373660),
    ("547", "West Palm Beach", "FL", 1500000, 800000),
    ("556", "Richmond-Petersburg", "VA", 1314434, 588160),
    ("541", "Lexington", "KY", 748338, 423910),
    ("557", "Knoxville", "TN", 1152200, 601270),
    ("546", "Columbia SC", "SC", 832666, 403320),
    ("640", "Memphis", "TN", 1341746, 665100),
    ("609", "St. Louis", "MO", 2803228, 1246480),
    ("531", "Tri-Cities TN-VA", "TN", 510000, 297330),
    ("540", "Traverse City-Cadillac", "MI", 350000, 204400),
    ("581", "Terre Haute", "IN", 247000, 141310),
    ("514", "Buffalo", "NY", 1127983, 543610),
    ("545", "Greenville-New Bern-Washington", "NC", 760000, 376970),
    ("530", "Tallahassee-Thomasville", "FL", 637000, 328680),
    ("570", "Florence-Myrtle Beach", "SC", 560000, 322790),
    ("575", "Chattanooga", "TN", 702000, 370790),
    ("519", "Charleston SC", "SC", 826105, 411720),
    ("538", "Rochester NY", "NY", 1059784, 431570),
    ("536", "Youngstown", "OH", 473000, 258080),
    ("535", "Columbus OH", "OH", 2138926, 956860),
    ("537", "Peoria-Bloomington", "IL", 453000, 257410),
    ("543", "Springfield-Holyoke", "MA", 681000, 317420),
    ("550", "Wilmington", "NC", 350000, 201200),
    ("564", "Charleston-Huntington", "WV", 804000, 409800),
    ("559", "Bluefield-Beckley-Oak Hill", "WV", 220000, 130410),
    ("573", "Roanoke-Lynchburg", "VA", 736000, 412840),
    ("569", "Harrisonburg", "VA", 155000, 89820),
    ("526", "Augusta-Aiken", "GA", 558000, 273190),
    ("525", "Madison", "WI", 689000, 407990),
    ("558", "Lima", "OH", 103000, 55430),
    ("549", "Watertown", "NY", 156000, 89860),
    ("522", "Columbus GA", "GA", 321000, 146780),
    ("523", "Burlington-Plattsburgh", "VT", 490000, 248740),
    ("521", "Providence-New Bedford", "RI", 1604291, 748570),
    ("520", "Savannah", "GA", 539000, 258260),
    ("516", "Erie", "PA", 270000, 140960),
    ("502", "Binghamton", "NY", 247000, 124000),
    ("503", "Macon", "GA", 343000, 165500),
    ("509", "Fort Wayne", "IN", 462000, 236600),
    ("555", "Syracuse", "NY", 731000, 372040),
    ("571", "Fort Myers-Naples", "FL", 1137000, 567340),
    ("507", "Savannah", "GA", 539000, 258260),
    ("574", "Johnstown-Altoona-State College", "PA", 363000, 191560),
    ("576", "Salisbury", "MD", 348000, 172840),
    ("551", "Lansing", "MI", 465000, 253490),
    ("552", "Presque Isle", "ME", 59000, 33130),
    ("553", "Marquette", "MI", 126000, 73520),
    ("554", "Wheeling-Steubenville", "WV", 188000, 101720),
    ("562", "Parkersburg", "WV", 136000, 73350),
    ("565", "Elmira-Corning", "NY", 160000, 88690),
    ("568", "Clarksburg-Weston", "WV", 118000, 68150),
    ("572", "Albany GA", "GA", 165000, 82560),
    ("578", "Bangor", "ME", 303000, 175450),
    ("579", "Watertown", "NY", 156000, 89860),
    ("580", "Portland-Auburn", "ME", 528000, 286320),
    ("584", "Charlottesville", "VA", 254000, 130890),
    ("588", "South Bend-Elkhart", "IN", 540000, 283670),
    ("592", "Gainesville", "FL", 305000, 175430),
    ("596", "Zanesville", "OH", 82000, 42870),
    ("598", "Utica", "NY", 268000, 132330),
    ("600", "Corpus Christi", "TX", 445000, 207970),
    ("604", "Columbia-Jefferson City", "MO", 450000, 224470),
    ("605", "Topeka", "KS", 326000, 163950),
    ("606", "Dothan", "AL", 198000, 100440),
    ("610", "Rockford", "IL", 340000, 171390),
    ("611", "Rochester MN-Mason City-Austin", "MN", 280000, 146300),
    ("612", "Shreveport", "LA", 675000, 340740),
    ("614", "Champaign-Springfield-Decatur", "IL", 590000, 305440),
    ("615", "Paducah-Cape Girardeau", "IL", 460000, 242810),
    ("619", "Springfield MO", "MO", 560000, 317170),
    ("622", "New Orleans", "LA", 1272940, 596880),
    ("624", "Waco-Temple-Bryan", "TX", 645000, 297690),
    ("625", "Wichita-Hutchinson", "KS", 672000, 386810),
    ("626", "Victoria", "TX", 113000, 55800),
    ("627", "Wichita Falls-Lawton", "TX", 298000, 148640),
    ("628", "Monroe-El Dorado", "LA", 275000, 140610),
    ("630", "Birmingham", "AL", 1152600, 590470),
    ("631", "Ottumwa-Kirksville", "IA", 118000, 60290),
    ("632", "Paducah", "KY", 260000, 137000),
    ("633", "Odessa-Midland", "TX", 355000, 142210),
    ("634", "Amarillo", "TX", 398000, 193010),
    ("635", "Austin", "TX", 2295303, 925600),
    ("636", "Harlingen-Weslaco-Brownsville-McAllen", "TX", 1100000, 385600),
    ("637", "Cedar Rapids-Waterloo-Iowa City", "IA", 620000, 349740),
    ("638", "St. Joseph", "MO", 116000, 62390),
    ("639", "Jackson TN", "TN", 200000, 101120),
    ("642", "Lafayette LA", "LA", 500000, 213760),
    ("643", "Lake Charles", "LA", 210000, 105240),
    ("644", "Alexandria LA", "LA", 240000, 107740),
    ("647", "Greenwood-Greenville", "MS", 154000, 82670),
    ("648", "Champaign", "IL", 250000, 129440),
    ("649", "Evansville", "IN", 480000, 257870),
    ("650", "Oklahoma City", "OK", 1408950, 668600),
    ("651", "Lubbock", "TX", 310000, 155240),
    ("652", "Omaha", "NE", 930000, 434400),
    ("656", "Panama City", "FL", 220000, 126560),
    ("657", "Sherman-Ada", "TX", 178000, 100420),
    ("658", "Green Bay-Appleton", "WI", 685000, 404930),
    ("661", "San Angelo", "TX", 120000, 59710),
    ("662", "Abilene-Sweetwater", "TX", 218000, 108970),
    ("669", "Madison", "WI", 689000, 407990),
    ("670", "Fort Smith-Fayetteville-Springdale-Rogers", "AR", 700000, 349420),
    ("671", "Tulsa", "OK", 1000000, 534530),
    ("673", "Columbus-Tupelo-West Point", "MS", 408000, 214080),
    ("675", "Peoria-Bloomington", "IL", 453000, 257410),
    ("676", "Duluth-Superior", "MN", 305000, 169180),
    ("678", "Wausau-Rhinelander", "WI", 232000, 130230),
    ("679", "Des Moines-Ames", "IA", 725000, 371480),
    ("682", "Davenport-Rock Island-Moline", "IA", 400000, 221790),
    ("686", "Mobile-Pensacola", "AL", 1120000, 513480),
    ("687", "Minot-Bismarck-Dickinson", "ND", 295000, 145890),
    ("691", "Huntsville-Decatur-Florence", "AL", 650000, 324550),
    ("692", "Beaumont-Port Arthur", "TX", 330000, 157660),
    ("693", "Little Rock-Pine Bluff", "AR", 910000, 445710),
    ("698", "Montgomery-Selma", "AL", 555000, 272730),
    ("702", "La Crosse-Eau Claire", "WI", 310000, 165180),
    ("705", "Wausau-Rhinelander", "WI", 232000, 130230),
    ("709", "Tyler-Longview", "TX", 410000, 219770),
    ("710", "Hattiesburg-Laurel", "MS", 180000, 93100),
    ("711", "Meridian", "MS", 135000, 73790),
    ("716", "Baton Rouge", "LA", 855000, 367600),
    ("717", "Quincy-Hannibal-Keokuk", "IL", 140000, 72130),
    ("718", "Jackson MS", "MS", 637000, 296310),
    ("722", "Lincoln-Hastings-Kearney", "NE", 403000, 216600),
    ("724", "Fargo-Valley City", "ND", 330000, 191040),
    ("725", "Sioux Falls-Mitchell", "SD", 450000, 262460),
    ("734", "Jonesboro", "AR", 160000, 85100),
    ("736", "Bowling Green", "KY", 210000, 119950),
    ("737", "Knoxville", "TN", 1152200, 601270),
    ("740", "North Platte", "NE", 48000, 25660),
    ("743", "Anchorage", "AK", 296000, 152500),
    ("744", "Honolulu", "HI", 988000, 432350),
    ("745", "Fairbanks", "AK", 98000, 43730),
    ("746", "Biloxi-Gulfport", "MS", 310000, 153000),
    ("747", "Juneau", "AK", 32000, 15020),
    ("749", "Laredo", "TX", 265000, 81900),
    ("752", "Colorado Springs-Pueblo", "CO", 860000, 395720),
    ("754", "Butte-Bozeman", "MT", 175000, 94020),
    ("755", "Great Falls", "MT", 108000, 55420),
    ("756", "Billings", "MT", 168000, 89080),
    ("757", "Boise", "ID", 773000, 340950),
    ("758", "Idaho Falls-Pocatello", "ID", 250000, 118810),
    ("759", "Cheyenne-Scottsbluff", "WY", 168000, 80500),
    ("760", "Twin Falls", "ID", 130000, 60720),
    ("762", "Missoula", "MT", 198000, 101150),
    ("764", "Rapid City", "SD", 162000, 96700),
    ("765", "El Paso", "TX", 850000, 308470),
    ("766", "Helena", "MT", 60000, 28000),
    ("767", "Casper-Riverton", "WY", 100000, 48420),
    ("770", "Salt Lake City", "UT", 1932000, 937860),
    ("771", "Yuma-El Centro", "AZ", 275000, 99620),
    ("773", "Grand Junction-Montrose", "CO", 190000, 93740),
    ("789", "Tucson", "AZ", 1043000, 477700),
    ("790", "Albuquerque-Santa Fe", "NM", 1120000, 527430),
    ("798", "Glendive", "MT", 7000, 3650),
    ("800", "Bakersfield", "CA", 900000, 319630),
    ("801", "Eugene", "OR", 415000, 227610),
    ("802", "Eureka", "CA", 135000, 70080),
    ("804", "Palm Springs", "CA", 462000, 210630),
    ("810", "Yakima-Pasco-Richland-Kennewick", "WA", 520000, 226000),
    ("811", "Reno", "NV", 537000, 281360),
    ("813", "Medford-Klamath Falls", "OR", 340000, 173250),
    ("818", "Monterey-Salinas", "CA", 470000, 218340),
    ("821", "Bend OR", "OR", 200000, 100930),
    ("828", "Sacramento-Stockton-Modesto", "CA", 3600000, 1434760),
    ("839", "Las Vegas", "NV", 2266715, 813570),
    ("855", "Santa Barbara-Santa Maria-San Luis Obispo", "CA", 545000, 247440),
    ("862", "Sacramento", "CA", 2420000, 1050000),
    ("866", "Fresno-Visalia", "CA", 1150000, 449490),
    ("868", "Chico-Redding", "CA", 390000, 195760),
    ("881", "Spokane", "WA", 735000, 358590),
]


def get_all_dmas() -> list[DMAProfile]:
    """Return all DMA profiles."""
    return [
        DMAProfile(
            dma_code=code,
            name=name,
            state=state,
            population=pop,
            tv_households=tvhh,
            region=_region(state),
        )
        for code, name, state, pop, tvhh in _DMA_DATA
    ]


def get_dma_dataframe() -> pd.DataFrame:
    """Return all DMAs as a DataFrame."""
    dmas = get_all_dmas()
    return pd.DataFrame([d.model_dump() for d in dmas])


def get_dma_by_code(code: str) -> DMAProfile | None:
    """Look up a single DMA by its code."""
    for entry in _DMA_DATA:
        if entry[0] == code:
            return DMAProfile(
                dma_code=entry[0],
                name=entry[1],
                state=entry[2],
                population=entry[3],
                tv_households=entry[4],
                region=_region(entry[2]),
            )
    return None


def get_dmas_by_region(region: str) -> list[DMAProfile]:
    """Get all DMAs in a given region."""
    return [d for d in get_all_dmas() if d.region == region]


def get_dma_codes() -> list[str]:
    """Get list of all DMA codes."""
    return [entry[0] for entry in _DMA_DATA]
