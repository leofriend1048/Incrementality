"""Facebook/Meta Ads data connector.

Pulls ad spend and delivery data from the Meta Marketing API,
broken down by DMA-level geo targeting.

Uses breakdowns=dma for native DMA-level reporting (requires Graph API v17+).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

from incrementality.config import FacebookConfig
from incrementality.connectors.retry import request_with_retry

logger = logging.getLogger(__name__)

_GRAPH_API_BASE = "https://graph.facebook.com/v22.0"

# Meta DMA name → Nielsen DMA code mapping (all 210 DMAs)
# Meta returns DMA names when breakdowns=dma. Names vary in format:
#   "City, ST", "City-City, ST", "City-City", or just "City"
# We include multiple variants per DMA to handle the inconsistency.
#
# Organized as (code, [name_variants]) for clarity, then flattened.
_DMA_NAMES: list[tuple[str, list[str]]] = [
    # 500 - Portland-Auburn, ME
    ("500", ["Portland-Auburn, ME", "Portland-Auburn"]),
    # 501 - New York
    ("501", ["New York, NY", "New York"]),
    # 502 - Binghamton
    ("502", ["Binghamton, NY", "Binghamton"]),
    # 503 - Macon
    ("503", ["Macon, GA", "Macon"]),
    # 504 - Philadelphia
    ("504", ["Philadelphia, PA", "Philadelphia"]),
    # 505 - Detroit
    ("505", ["Detroit, MI", "Detroit"]),
    # 506 - Boston
    ("506", ["Boston, MA-Manchester, NH", "Boston, MA (Manchester, NH)",
             "Boston, MA", "Boston"]),
    # 507 - Savannah
    ("507", ["Savannah, GA", "Savannah"]),
    # 508 - Pittsburgh
    ("508", ["Pittsburgh, PA", "Pittsburgh"]),
    # 509 - Ft. Wayne
    ("509", ["Ft. Wayne, IN", "Fort Wayne, IN", "Ft. Wayne", "Fort Wayne"]),
    # 510 - Cleveland-Akron (Canton)
    ("510", ["Cleveland-Akron (Canton), OH", "Cleveland-Akron, OH",
             "Cleveland, OH", "Cleveland-Akron (Canton)", "Cleveland-Akron",
             "Cleveland"]),
    # 511 - Washington, DC
    ("511", ["Washington, DC (Hagerstown, MD)", "Washington, DC",
             "Washington DC (Hagerstown)", "Washington DC",
             "Washington, DC (Hagrstwn)"]),
    # 512 - Baltimore
    ("512", ["Baltimore, MD", "Baltimore"]),
    # 513 - Flint-Saginaw-Bay City
    ("513", ["Flint-Saginaw-Bay City, MI", "Flint-Saginaw-Bay City",
             "Flint, MI", "Flint"]),
    # 514 - Buffalo
    ("514", ["Buffalo, NY", "Buffalo"]),
    # 515 - Cincinnati
    ("515", ["Cincinnati, OH", "Cincinnati"]),
    # 516 - Erie
    ("516", ["Erie, PA", "Erie"]),
    # 517 - Charlotte
    ("517", ["Charlotte, NC", "Charlotte"]),
    # 518 - Greensboro-High Point-Winston Salem
    ("518", ["Greensboro-High Point-Winston Salem, NC",
             "Greensboro-High Point-Winston-Salem, NC",
             "Greensboro, NC", "Greensboro-High Point-Winston Salem",
             "Greensboro-High Point-Winston-Salem", "Greensboro"]),
    # 519 - Charleston, SC (not just "Charleston" — ambiguous with WV 564)
    ("519", ["Charleston, SC"]),
    # 520 - Augusta-Aiken
    ("520", ["Augusta-Aiken, GA", "Augusta, GA", "Augusta-Aiken", "Augusta"]),
    # 521 - Providence-New Bedford
    ("521", ["Providence, RI-New Bedford, MA", "Providence-New Bedford, RI",
             "Providence, RI", "Providence-New Bedford", "Providence"]),
    # 522 - Columbus, GA (not just "Columbus" — ambiguous with OH 535 / MS 673)
    ("522", ["Columbus, GA (Opelika, AL)", "Columbus, GA",
             "Columbus-Opelika"]),
    # 523 - Burlington-Plattsburgh
    ("523", ["Burlington-Plattsburgh, VT", "Burlington, VT-Plattsburgh, NY",
             "Burlington, VT", "Burlington-Plattsburgh"]),
    # 524 - Atlanta
    ("524", ["Atlanta, GA", "Atlanta"]),
    # 525 - Albany, GA (not just "Albany" — ambiguous with NY 532)
    ("525", ["Albany, GA"]),
    # 526 - Utica
    ("526", ["Utica, NY", "Utica"]),
    # 527 - Indianapolis
    ("527", ["Indianapolis, IN", "Indianapolis"]),
    # 528 - Miami-Ft. Lauderdale
    ("528", ["Miami-Ft. Lauderdale, FL", "Miami-Fort Lauderdale, FL",
             "Miami, FL", "Miami-Ft. Lauderdale", "Miami-Fort Lauderdale",
             "Miami"]),
    # 529 - Louisville
    ("529", ["Louisville, KY", "Louisville"]),
    # 530 - Tallahassee-Thomasville
    ("530", ["Tallahassee-Thomasville, FL", "Tallahassee, FL-Thomasville, GA",
             "Tallahassee, FL", "Tallahassee-Thomasville", "Tallahassee"]),
    # 531 - Tri-Cities TN-VA
    ("531", ["Tri-Cities, TN-VA", "Tri-Cities TN-VA", "Tri-Cities"]),
    # 532 - Albany-Schenectady-Troy
    ("532", ["Albany-Schenectady-Troy, NY", "Albany-Schenectady-Troy"]),
    # 533 - Hartford & New Haven
    ("533", ["Hartford & New Haven, CT", "Hartford-New Haven, CT",
             "Hartford, CT", "Hartford & New Haven", "Hartford-New Haven",
             "Hartford"]),
    # 534 - Orlando-Daytona Beach-Melbourne
    ("534", ["Orlando-Daytona Beach-Melbourne, FL", "Orlando, FL",
             "Orlando-Daytona Beach-Melbourne", "Orlando"]),
    # 535 - Columbus, OH (not just "Columbus" — ambiguous)
    ("535", ["Columbus, OH"]),
    # 536 - Youngstown
    ("536", ["Youngstown, OH", "Youngstown"]),
    # 537 - Bangor
    ("537", ["Bangor, ME", "Bangor"]),
    # 538 - Rochester, NY (not just "Rochester" — ambiguous with MN 611)
    ("538", ["Rochester, NY"]),
    # 539 - Tampa-St. Petersburg
    ("539", ["Tampa-St. Petersburg (Sarasota), FL",
             "Tampa-St. Petersburg, FL", "Tampa, FL",
             "Tampa-St. Petersburg (Sarasota)", "Tampa-St. Petersburg",
             "Tampa"]),
    # 540 - Traverse City-Cadillac
    ("540", ["Traverse City-Cadillac, MI", "Traverse City-Cadillac",
             "Traverse City, MI", "Traverse City"]),
    # 541 - Lexington
    ("541", ["Lexington, KY", "Lexington"]),
    # 542 - Dayton
    ("542", ["Dayton, OH", "Dayton"]),
    # 543 - Springfield, MO (not just "Springfield" — ambiguous with IL 619)
    ("543", ["Springfield, MO", "Springfield MO"]),
    # 544 - Norfolk-Portsmouth-Newport News
    ("544", ["Norfolk-Portsmouth-Newport News, VA", "Norfolk, VA",
             "Norfolk-Portsmouth-Newport News", "Norfolk"]),
    # 545 - Greenville-New Bern-Washington, NC
    ("545", ["Greenville-New Bern-Washington, NC",
             "Greenville-New Bern-Washington NC",
             "Greenville-New Bern-Washington", "Greenville, NC"]),
    # 546 - Columbia, SC (not just "Columbia" — ambiguous with MO 604)
    ("546", ["Columbia, SC"]),
    # 547 - Toledo
    ("547", ["Toledo, OH", "Toledo"]),
    # 548 - West Palm Beach-Ft. Pierce
    ("548", ["West Palm Beach-Ft. Pierce, FL",
             "West Palm Beach-Fort Pierce, FL", "West Palm Beach, FL",
             "West Palm Beach-Ft. Pierce", "West Palm Beach"]),
    # 549 - Watertown
    ("549", ["Watertown, NY", "Watertown"]),
    # 550 - Wilmington
    ("550", ["Wilmington, NC", "Wilmington"]),
    # 551 - Lansing
    ("551", ["Lansing, MI", "Lansing"]),
    # 552 - Presque Isle
    ("552", ["Presque Isle, ME", "Presque Isle"]),
    # 553 - Marquette
    ("553", ["Marquette, MI", "Marquette"]),
    # 554 - Wheeling-Steubenville
    ("554", ["Wheeling-Steubenville, WV", "Wheeling, WV-Steubenville, OH",
             "Wheeling-Steubenville", "Wheeling"]),
    # 555 - Syracuse
    ("555", ["Syracuse, NY", "Syracuse"]),
    # 556 - Richmond-Petersburg
    ("556", ["Richmond-Petersburg, VA", "Richmond, VA",
             "Richmond-Petersburg", "Richmond"]),
    # 557 - Knoxville
    ("557", ["Knoxville, TN", "Knoxville"]),
    # 558 - Lima
    ("558", ["Lima, OH", "Lima"]),
    # 559 - Bluefield-Beckley-Oak Hill
    ("559", ["Bluefield-Beckley-Oak Hill, WV", "Bluefield-Beckley-Oak Hill",
             "Bluefield"]),
    # 560 - Raleigh-Durham
    ("560", ["Raleigh-Durham (Fayetteville), NC", "Raleigh-Durham, NC",
             "Raleigh-Durham (Fayetteville)", "Raleigh-Durham"]),
    # 561 - Jacksonville
    ("561", ["Jacksonville, FL", "Jacksonville"]),
    # 562 - Harrisonburg
    ("562", ["Harrisonburg, VA", "Harrisonburg"]),
    # 563 - Grand Rapids-Kalamazoo-Battle Creek
    ("563", ["Grand Rapids-Kalamazoo-Battle Creek, MI", "Grand Rapids, MI",
             "Grand Rapids-Kalamazoo-Battle Creek", "Grand Rapids"]),
    # 564 - Charleston-Huntington, WV
    ("564", ["Charleston-Huntington, WV", "Charleston, WV",
             "Charleston-Huntington"]),
    # 565 - Elmira (Corning)
    ("565", ["Elmira (Corning), NY", "Elmira, NY", "Elmira-Corning",
             "Elmira (Corning)", "Elmira"]),
    # 566 - Harrisburg-Lancaster-Lebanon-York
    ("566", ["Harrisburg-Lancaster-Lebanon-York, PA", "Harrisburg, PA",
             "Harrisburg-Lancaster-Lebanon-York", "Harrisburg"]),
    # 567 - Greenville-Spartanburg-Asheville-Anderson, SC
    ("567", ["Greenville-Spartanburg-Asheville-Anderson, SC",
             "Greenville-Spartanburg, SC",
             "Greenville-Spartanburg-Asheville-Anderson",
             "Greenville-Spartanburg-Asheville", "Greenville-Spartanburg"]),
    # 570 - Florence-Myrtle Beach
    ("570", ["Florence-Myrtle Beach, SC", "Florence-Myrtle Beach",
             "Florence, SC"]),
    # 571 - Ft. Myers-Naples
    ("571", ["Ft. Myers-Naples, FL", "Fort Myers-Naples, FL",
             "Ft. Myers-Naples", "Fort Myers-Naples"]),
    # 573 - Roanoke-Lynchburg
    ("573", ["Roanoke-Lynchburg, VA", "Roanoke-Lynchburg", "Roanoke, VA",
             "Roanoke"]),
    # 574 - Johnstown-Altoona-State College
    ("574", ["Johnstown-Altoona-State College, PA", "Johnstown-Altoona, PA",
             "Johnstown-Altoona-State College", "Johnstown-Altoona",
             "Johnstown"]),
    # 575 - Chattanooga
    ("575", ["Chattanooga, TN", "Chattanooga"]),
    # 576 - Salisbury
    ("576", ["Salisbury, MD", "Salisbury"]),
    # 577 - Wilkes Barre-Scranton-Hazleton
    ("577", ["Wilkes Barre-Scranton-Hazleton, PA",
             "Wilkes-Barre-Scranton, PA", "Wilkes Barre-Scranton-Hazleton",
             "Wilkes Barre-Scranton", "Wilkes-Barre-Scranton"]),
    # 581 - Terre Haute
    ("581", ["Terre Haute, IN", "Terre Haute"]),
    # 582 - Lafayette, IN (not just "Lafayette" — ambiguous with LA 642)
    ("582", ["Lafayette, IN"]),
    # 583 - Alpena
    ("583", ["Alpena, MI", "Alpena"]),
    # 584 - Charlottesville
    ("584", ["Charlottesville, VA", "Charlottesville"]),
    # 588 - South Bend-Elkhart
    ("588", ["South Bend-Elkhart, IN", "South Bend-Elkhart",
             "South Bend, IN", "South Bend"]),
    # 592 - Gainesville
    ("592", ["Gainesville, FL", "Gainesville"]),
    # 596 - Zanesville
    ("596", ["Zanesville, OH", "Zanesville"]),
    # 597 - Parkersburg
    ("597", ["Parkersburg, WV", "Parkersburg"]),
    # 598 - Clarksburg-Weston
    ("598", ["Clarksburg-Weston, WV", "Clarksburg-Weston", "Clarksburg"]),
    # 600 - Corpus Christi
    ("600", ["Corpus Christi, TX", "Corpus Christi"]),
    # 602 - Chicago
    ("602", ["Chicago, IL", "Chicago"]),
    # 603 - Joplin-Pittsburg
    ("603", ["Joplin-Pittsburg, MO", "Joplin-Pittsburg", "Joplin, MO",
             "Joplin"]),
    # 604 - Columbia-Jefferson City, MO
    ("604", ["Columbia-Jefferson City, MO", "Columbia-Jefferson City",
             "Columbia, MO"]),
    # 605 - Topeka
    ("605", ["Topeka, KS", "Topeka"]),
    # 606 - Dothan
    ("606", ["Dothan, AL", "Dothan"]),
    # 609 - St. Louis
    ("609", ["St. Louis, MO", "Saint Louis, MO", "St. Louis",
             "Saint Louis"]),
    # 610 - Rockford
    ("610", ["Rockford, IL", "Rockford"]),
    # 611 - Rochester-Mason City-Austin, MN
    ("611", ["Rochester-Mason City-Austin, MN",
             "Rochester, MN-Mason City, IA-Austin, MN",
             "Rochester-Mason City-Austin", "Rochester, MN"]),
    # 612 - Shreveport
    ("612", ["Shreveport, LA", "Shreveport"]),
    # 613 - Minneapolis-St. Paul
    ("613", ["Minneapolis-St. Paul, MN", "Minneapolis-Saint Paul, MN",
             "Minneapolis, MN", "Minneapolis-St. Paul",
             "Minneapolis-Saint Paul", "Minneapolis"]),
    # 616 - Kansas City
    ("616", ["Kansas City, MO", "Kansas City"]),
    # 617 - Milwaukee
    ("617", ["Milwaukee, WI", "Milwaukee"]),
    # 618 - Houston
    ("618", ["Houston, TX", "Houston"]),
    # 619 - Springfield-Decatur, IL
    ("619", ["Springfield-Decatur, IL", "Springfield-Decatur IL",
             "Springfield, IL", "Springfield-Decatur"]),
    # 622 - New Orleans
    ("622", ["New Orleans, LA", "New Orleans"]),
    # 623 - Dallas-Ft. Worth
    ("623", ["Dallas-Ft. Worth, TX", "Dallas-Fort Worth, TX", "Dallas, TX",
             "Dallas-Ft. Worth", "Dallas-Fort Worth", "Dallas"]),
    # 624 - Sioux City
    ("624", ["Sioux City, IA", "Sioux City"]),
    # 625 - Waco-Temple-Bryan
    ("625", ["Waco-Temple-Bryan, TX", "Waco-Temple-Bryan", "Waco, TX",
             "Waco"]),
    # 626 - Victoria
    ("626", ["Victoria, TX", "Victoria"]),
    # 627 - Wichita Falls & Lawton
    ("627", ["Wichita Falls-Lawton, OK", "Wichita Falls & Lawton",
             "Wichita Falls-Lawton", "Wichita Falls, TX",
             "Wichita Falls"]),
    # 628 - Monroe-El Dorado
    ("628", ["Monroe-El Dorado, LA", "Monroe-El Dorado",
             "Monroe, LA"]),
    # 630 - Birmingham
    ("630", ["Birmingham (Anniston and Tuscaloosa), AL", "Birmingham, AL",
             "Birmingham (Anniston and Tuscaloosa)",
             "Birmingham-Anniston-Tuscaloosa", "Birmingham"]),
    # 631 - Ottumwa-Kirksville
    ("631", ["Ottumwa-Kirksville, IA", "Ottumwa-Kirksville", "Ottumwa"]),
    # 632 - Paducah-Cape Girardeau-Harrisburg-Mt Vernon
    ("632", ["Paducah-Cape Girardeau-Harrisburg-Mt Vernon, IL",
             "Paducah-Cape Girardeau-Harrisburg, IL",
             "Paducah-Cape Girardeau-Harrisburg-Mt Vernon",
             "Paducah-Cape Girardeau-Harrisburg",
             "Paducah, KY", "Paducah"]),
    # 633 - Odessa-Midland
    ("633", ["Odessa-Midland, TX", "Odessa-Midland", "Odessa"]),
    # 634 - Amarillo
    ("634", ["Amarillo, TX", "Amarillo"]),
    # 635 - Austin
    ("635", ["Austin, TX", "Austin"]),
    # 636 - Harlingen-Weslaco-Brownsville-McAllen
    ("636", ["Harlingen-Weslaco-Brownsville-McAllen, TX",
             "Harlingen-Weslaco-Brownsville-McAllen",
             "Harlingen, TX", "Harlingen"]),
    # 637 - Cedar Rapids-Waterloo-Iowa City-Dubuque
    ("637", ["Cedar Rapids-Waterloo-Iowa City-Dubuque, IA",
             "Cedar Rapids-Waterloo-Iowa City-Dubuque",
             "Cedar Rapids-Waterloo & Dubuque",
             "Cedar Rapids, IA", "Cedar Rapids"]),
    # 638 - St. Joseph
    ("638", ["St. Joseph, MO", "Saint Joseph, MO", "St. Joseph",
             "Saint Joseph"]),
    # 639 - Jackson, TN (not just "Jackson" — ambiguous with MS 718)
    ("639", ["Jackson, TN"]),
    # 640 - Memphis
    ("640", ["Memphis, TN", "Memphis"]),
    # 641 - San Antonio
    ("641", ["San Antonio, TX", "San Antonio"]),
    # 642 - Lafayette, LA (not just "Lafayette" — ambiguous with IN 582)
    ("642", ["Lafayette, LA"]),
    # 643 - Lake Charles
    ("643", ["Lake Charles, LA", "Lake Charles"]),
    # 644 - Alexandria, LA
    ("644", ["Alexandria, LA", "Alexandria"]),
    # 647 - Greenwood-Greenville, MS
    ("647", ["Greenwood-Greenville, MS", "Greenwood-Greenville MS",
             "Greenwood-Greenville"]),
    # 648 - Champaign & Springfield-Decatur
    ("648", ["Champaign & Springfield-Decatur, IL",
             "Champaign-Springfield-Decatur",
             "Champaign & Springfield-Decatur",
             "Champaign, IL", "Champaign"]),
    # 649 - Evansville
    ("649", ["Evansville, IN", "Evansville"]),
    # 650 - Oklahoma City
    ("650", ["Oklahoma City, OK", "Oklahoma City"]),
    # 651 - Lubbock
    ("651", ["Lubbock, TX", "Lubbock"]),
    # 652 - Omaha
    ("652", ["Omaha, NE", "Omaha"]),
    # 656 - Panama City
    ("656", ["Panama City, FL", "Panama City"]),
    # 657 - Sherman-Ada
    ("657", ["Sherman-Ada, TX", "Sherman-Ada", "Sherman, TX"]),
    # 658 - Green Bay-Appleton
    ("658", ["Green Bay-Appleton, WI", "Green Bay-Appleton",
             "Green Bay, WI", "Green Bay"]),
    # 659 - Nashville
    ("659", ["Nashville, TN", "Nashville"]),
    # 661 - San Angelo
    ("661", ["San Angelo, TX", "San Angelo"]),
    # 662 - Abilene-Sweetwater
    ("662", ["Abilene-Sweetwater, TX", "Abilene-Sweetwater",
             "Abilene, TX", "Abilene"]),
    # 669 - Madison
    ("669", ["Madison, WI", "Madison"]),
    # 670 - Ft. Smith-Fayetteville-Springdale-Rogers
    ("670", ["Ft. Smith-Fayetteville-Springdale-Rogers, AR",
             "Fort Smith-Fayetteville-Springdale-Rogers, AR",
             "Ft. Smith-Fayetteville-Springdale-Rogers",
             "Fort Smith-Fayetteville-Springdale-Rogers",
             "Ft. Smith, AR", "Fort Smith"]),
    # 671 - Tulsa
    ("671", ["Tulsa, OK", "Tulsa"]),
    # 673 - Columbus-Tupelo-West Point
    ("673", ["Columbus-Tupelo-West Point, MS",
             "Columbus-Tupelo-West Point"]),
    # 675 - Peoria-Bloomington
    ("675", ["Peoria-Bloomington, IL", "Peoria-Bloomington",
             "Peoria, IL", "Peoria"]),
    # 676 - Duluth-Superior
    ("676", ["Duluth-Superior, MN", "Duluth-Superior",
             "Duluth, MN", "Duluth"]),
    # 678 - Wichita-Hutchinson Plus
    ("678", ["Wichita-Hutchinson Plus, KS", "Wichita-Hutchinson, KS",
             "Wichita, KS", "Wichita-Hutchinson Plus",
             "Wichita-Hutchinson"]),
    # 679 - Des Moines-Ames
    ("679", ["Des Moines-Ames, IA", "Des Moines, IA",
             "Des Moines-Ames", "Des Moines"]),
    # 682 - Davenport-Rock Island-Moline
    ("682", ["Davenport-Rock Island-Moline, IL",
             "Davenport-Rock Island-Moline",
             "Davenport, IA", "Davenport"]),
    # 686 - Mobile-Pensacola
    ("686", ["Mobile-Pensacola (Ft. Walton Beach), FL",
             "Mobile-Pensacola (Ft Walton Beach)",
             "Mobile-Pensacola, FL", "Mobile, AL",
             "Mobile-Pensacola", "Mobile"]),
    # 687 - Minot-Bismarck-Dickinson
    ("687", ["Minot-Bismarck-Dickinson, ND",
             "Minot-Bismarck-Dickinson(Williston)",
             "Minot-Bismarck-Dickinson", "Minot, ND"]),
    # 691 - Huntsville-Decatur (Florence)
    ("691", ["Huntsville-Decatur (Florence), AL",
             "Huntsville-Decatur, AL", "Huntsville-Decatur (Florence)",
             "Huntsville-Decatur", "Huntsville, AL", "Huntsville"]),
    # 692 - Beaumont-Port Arthur
    ("692", ["Beaumont-Port Arthur, TX", "Beaumont-Port Arthur",
             "Beaumont, TX", "Beaumont"]),
    # 693 - Little Rock-Pine Bluff
    ("693", ["Little Rock-Pine Bluff, AR", "Little Rock, AR",
             "Little Rock-Pine Bluff", "Little Rock"]),
    # 698 - Montgomery-Selma
    ("698", ["Montgomery-Selma, AL", "Montgomery, AL",
             "Montgomery-Selma", "Montgomery"]),
    # 702 - La Crosse-Eau Claire
    ("702", ["La Crosse-Eau Claire, WI", "La Crosse-Eau Claire",
             "La Crosse, WI", "La Crosse"]),
    # 705 - Wausau-Rhinelander
    ("705", ["Wausau-Rhinelander, WI", "Wausau-Rhinelander",
             "Wausau, WI", "Wausau"]),
    # 709 - Tyler-Longview
    ("709", ["Tyler-Longview (Lufkin & Nacogdoches), TX",
             "Tyler-Longview, TX", "Tyler-Longview (Lufkin & Nacogdoches)",
             "Tyler-Longview", "Tyler, TX", "Tyler"]),
    # 710 - Hattiesburg-Laurel
    ("710", ["Hattiesburg-Laurel, MS", "Hattiesburg-Laurel",
             "Hattiesburg, MS", "Hattiesburg"]),
    # 711 - Meridian
    ("711", ["Meridian, MS", "Meridian"]),
    # 716 - Baton Rouge
    ("716", ["Baton Rouge, LA", "Baton Rouge"]),
    # 717 - Quincy-Hannibal-Keokuk
    ("717", ["Quincy-Hannibal-Keokuk, IL", "Quincy-Hannibal-Keokuk",
             "Quincy, IL", "Quincy"]),
    # 718 - Jackson, MS (not just "Jackson" — ambiguous with TN 639)
    ("718", ["Jackson, MS"]),
    # 722 - Lincoln & Hastings-Kearney
    ("722", ["Lincoln & Hastings-Kearney, NE", "Lincoln-Hastings-Kearney",
             "Lincoln & Hastings-Kearney", "Lincoln, NE"]),
    # 724 - Fargo-Valley City
    ("724", ["Fargo-Valley City, ND", "Fargo-Valley City",
             "Fargo, ND", "Fargo"]),
    # 725 - Sioux Falls (Mitchell)
    ("725", ["Sioux Falls (Mitchell), SD", "Sioux Falls, SD",
             "Sioux Falls (Mitchell)", "Sioux Falls"]),
    # 734 - Jonesboro
    ("734", ["Jonesboro, AR", "Jonesboro"]),
    # 736 - Bowling Green
    ("736", ["Bowling Green, KY", "Bowling Green"]),
    # 740 - North Platte
    ("740", ["North Platte, NE", "North Platte"]),
    # 743 - Anchorage
    ("743", ["Anchorage, AK", "Anchorage"]),
    # 744 - Honolulu
    ("744", ["Honolulu, HI", "Honolulu"]),
    # 745 - Fairbanks
    ("745", ["Fairbanks, AK", "Fairbanks"]),
    # 746 - Biloxi-Gulfport
    ("746", ["Biloxi-Gulfport, MS", "Biloxi-Gulfport",
             "Biloxi, MS", "Biloxi"]),
    # 747 - Juneau
    ("747", ["Juneau, AK", "Juneau"]),
    # 751 - Denver
    ("751", ["Denver, CO", "Denver"]),
    # 752 - Colorado Springs-Pueblo
    ("752", ["Colorado Springs-Pueblo, CO", "Colorado Springs, CO",
             "Colorado Springs-Pueblo", "Colorado Springs"]),
    # 753 - Phoenix
    ("753", ["Phoenix (Prescott), AZ", "Phoenix, AZ",
             "Phoenix (Prescott)", "Phoenix"]),
    # 754 - Butte-Bozeman
    ("754", ["Butte-Bozeman, MT", "Butte-Bozeman", "Butte, MT", "Butte"]),
    # 755 - Great Falls
    ("755", ["Great Falls, MT", "Great Falls"]),
    # 756 - Billings
    ("756", ["Billings, MT", "Billings"]),
    # 757 - Boise
    ("757", ["Boise, ID", "Boise"]),
    # 758 - Idaho Falls-Pocatello
    ("758", ["Idaho Falls-Pocatello, ID", "Idaho Falls-Pocatello (Jackson)",
             "Idaho Falls-Pocatello", "Idaho Falls, ID", "Idaho Falls"]),
    # 759 - Cheyenne-Scottsbluff
    ("759", ["Cheyenne-Scottsbluff, WY", "Cheyenne-Scottsbluff",
             "Cheyenne, WY", "Cheyenne"]),
    # 760 - Twin Falls
    ("760", ["Twin Falls, ID", "Twin Falls"]),
    # 762 - Missoula
    ("762", ["Missoula, MT", "Missoula"]),
    # 764 - Rapid City
    ("764", ["Rapid City, SD", "Rapid City"]),
    # 765 - El Paso
    ("765", ["El Paso (Las Cruces), TX", "El Paso, TX",
             "El Paso (Las Cruces)", "El Paso"]),
    # 766 - Helena
    ("766", ["Helena, MT", "Helena"]),
    # 770 - Salt Lake City
    ("770", ["Salt Lake City, UT", "Salt Lake City"]),
    # 771 - Yuma-El Centro
    ("771", ["Yuma-El Centro, AZ", "Yuma-El Centro", "Yuma, AZ", "Yuma"]),
    # 773 - Grand Junction-Montrose
    ("773", ["Grand Junction-Montrose, CO", "Grand Junction-Montrose",
             "Grand Junction, CO", "Grand Junction"]),
    # 789 - Tucson
    ("789", ["Tucson (Sierra Vista), AZ", "Tucson, AZ",
             "Tucson (Sierra Vista)", "Tucson"]),
    # 790 - Albuquerque-Santa Fe
    ("790", ["Albuquerque-Santa Fe, NM", "Albuquerque, NM",
             "Albuquerque-Santa Fe", "Albuquerque"]),
    # 798 - Glendive
    ("798", ["Glendive, MT", "Glendive"]),
    # 800 - Bakersfield
    ("800", ["Bakersfield, CA", "Bakersfield"]),
    # 801 - Eugene
    ("801", ["Eugene, OR", "Eugene"]),
    # 802 - Eureka
    ("802", ["Eureka, CA", "Eureka"]),
    # 803 - Los Angeles
    ("803", ["Los Angeles, CA", "Los Angeles"]),
    # 804 - Palm Springs
    ("804", ["Palm Springs, CA", "Palm Springs"]),
    # 807 - San Francisco-Oakland-San Jose
    ("807", ["San Francisco-Oakland-San Jose, CA", "San Francisco, CA",
             "San Francisco-Oakland-San Jose", "San Francisco"]),
    # 810 - Yakima-Pasco-Richland-Kennewick
    ("810", ["Yakima-Pasco-Richland-Kennewick, WA",
             "Yakima-Pasco-Richland-Kennewick", "Yakima, WA", "Yakima"]),
    # 811 - Reno
    ("811", ["Reno, NV", "Reno"]),
    # 813 - Medford-Klamath Falls
    ("813", ["Medford-Klamath Falls, OR", "Medford-Klamath Falls",
             "Medford, OR", "Medford"]),
    # 819 - Seattle-Tacoma
    ("819", ["Seattle-Tacoma, WA", "Seattle, WA", "Seattle-Tacoma",
             "Seattle"]),
    # 820 - Portland, OR (not just "Portland" — ambiguous with ME 500)
    ("820", ["Portland, OR"]),
    # 821 - Bend
    ("821", ["Bend, OR", "Bend"]),
    # 825 - San Diego
    ("825", ["San Diego, CA", "San Diego"]),
    # 828 - Monterey-Salinas
    ("828", ["Monterey-Salinas, CA", "Monterey-Salinas",
             "Monterey, CA", "Monterey"]),
    # 839 - Las Vegas
    ("839", ["Las Vegas, NV", "Las Vegas"]),
    # 855 - Santa Barbara-Santa Maria-San Luis Obispo
    ("855", ["Santa Barbara-Santa Maria-San Luis Obispo, CA",
             "Santa Barbara, CA",
             "Santa Barbara-Santa Maria-San Luis Obispo",
             "Santa Barbara"]),
    # 862 - Sacramento-Stockton-Modesto
    ("862", ["Sacramento-Stockton-Modesto, CA", "Sacramento, CA",
             "Sacramento-Stockton-Modesto", "Sacramento"]),
    # 866 - Fresno-Visalia
    ("866", ["Fresno-Visalia, CA", "Fresno, CA", "Fresno-Visalia",
             "Fresno"]),
    # 868 - Chico-Redding
    ("868", ["Chico-Redding, CA", "Chico-Redding", "Chico, CA", "Chico"]),
    # 881 - Spokane
    ("881", ["Spokane, WA", "Spokane"]),

    # --- Missing DMA entries (not in original list) ---
    # 737 - Mankato
    ("737", ["Mankato, MN", "Mankato"]),
    # 749 - Laredo
    ("749", ["Laredo, TX", "Laredo"]),
    # 767 - Casper-Riverton
    ("767", ["Casper-Riverton, WY", "Casper-Riverton", "Casper, WY", "Casper"]),

    # --- Meta abbreviated name variants (from actual API responses) ---
    # Meta truncates long DMA names; these are the exact strings the API returns.
    ("506", ["Boston (Manchester)"]),
    ("518", ["Greensboro-H.Point-W.Salem"]),
    ("534", ["Orlando-Daytona Bch-Melbrn"]),
    ("539", ["Tampa-St. Pete (Sarasota)"]),
    ("544", ["Norfolk-Portsmth-Newpt Nws"]),
    ("545", ["Greenville-N.Bern-Washngtn"]),
    ("560", ["Raleigh-Durham (Fayetvlle)"]),
    ("563", ["Grand Rapids-Kalmzoo-B.Crk"]),
    ("566", ["Harrisburg-Lncstr-Leb-York"]),
    ("567", ["Greenvll-Spart-Ashevll-And"]),
    ("570", ["Myrtle Beach-Florence"]),
    ("574", ["Johnstown-Altoona-St Colge"]),
    ("577", ["Wilkes Barre-Scranton-Hztn"]),
    ("630", ["Birmingham (Ann And Tusc)"]),
    ("632", ["Paducah-Cape Girard-Harsbg"]),
    ("636", ["Harlingen-Wslco-Brnsvl-Mca"]),
    ("637", ["Cedar Rapids-Wtrlo-Iwc&Dub"]),
    ("648", ["Champaign&Sprngfld-Decatur"]),
    ("670", ["Ft. Smith-Fay-Sprngdl-Rgrs"]),
    ("673", ["Columbus-Tupelo-W Pnt-Hstn"]),
    ("682", ["Davenport-R.Island-Moline"]),
    ("686", ["Mobile-Pensacola (Ft Walt)"]),
    ("687", ["Minot-Bsmrck-Dcknsn(Wlstn)"]),
    ("691", ["Huntsville-Decatur (Flor)"]),
    ("709", ["Tyler-Longview(Lfkn&Ncgd)"]),
    ("722", ["Lincoln & Hastings-Krny"]),
    ("725", ["Sioux Falls(Mitchell)"]),
    ("758", ["Idaho Fals-Pocatllo(Jcksn)"]),
    ("807", ["San Francisco-Oak-San Jose"]),
    ("810", ["Yakima-Pasco-Rchlnd-Knnwck"]),
    ("855", ["Santabarbra-Sanmar-Sanluob"]),
    ("862", ["Sacramnto-Stkton-Modesto"]),

    # Springfield-Holyoke (MA) — Nielsen DMA 543
    ("543", ["Springfield-Holyoke"]),
]

# Build flat lookup dict from the definition list
_META_DMA_TO_NIELSEN: dict[str, str] = {}
for _code, _names in _DMA_NAMES:
    for _name in _names:
        _META_DMA_TO_NIELSEN[_name] = _code

# Build case-insensitive normalized lookup for fallback matching
_META_DMA_NORMALIZED: dict[str, str] = {
    k.lower().strip(): v for k, v in _META_DMA_TO_NIELSEN.items()
}


class FacebookConnector:
    """Connects to Meta Marketing API for ad spend by DMA."""

    DEFAULT_TIMEOUT = 30  # seconds per request

    def __init__(self, config: FacebookConfig):
        self.config = config
        self.session = requests.Session()
        self.session.params = {"access_token": config.access_token}  # type: ignore

    def _get(self, url_or_path: str, params: dict[str, Any] | None = None) -> dict:
        # If it's already a full URL (e.g. pagination next link), use it directly
        if url_or_path.startswith("https://"):
            url = url_or_path
        else:
            url = f"{_GRAPH_API_BASE}/{url_or_path}"
        resp = request_with_retry(
            self.session, "GET", url,
            params=params or {}, timeout=self.DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict[str, Any] | None = None) -> dict:
        url = f"{_GRAPH_API_BASE}/{path}"
        resp = request_with_retry(
            self.session, "POST", url,
            json=data or {}, timeout=self.DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_campaigns(self) -> pd.DataFrame:
        """Fetch all campaigns in the ad account.

        Returns columns: campaign_id, name, status, objective, daily_budget
        """
        data = self._get(
            f"{self.config.ad_account_id}/campaigns",
            params={
                "fields": "id,name,status,objective,daily_budget",
                "limit": 500,
            },
        )
        records = []
        for c in data.get("data", []):
            records.append({
                "campaign_id": c["id"],
                "name": c.get("name", ""),
                "status": c.get("status", ""),
                "objective": c.get("objective", ""),
                "daily_budget": float(c.get("daily_budget", 0)) / 100,
            })
        return pd.DataFrame(records)

    def fetch_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily ad spend broken down by DMA.

        Uses the Marketing API insights endpoint with breakdowns=dma
        for native DMA-level reporting.

        Returns columns: date, dma_code, dma_name, spend, impressions, clicks
        """
        params: dict[str, Any] = {
            "fields": "spend,impressions,clicks",
            "breakdowns": "dma",
            "time_range": json.dumps({
                "since": str(start_date),
                "until": str(end_date),
            }),
            "time_increment": 1,  # Daily granularity
            "limit": 5000,
        }
        if campaign_ids:
            filtering = [
                {"field": "campaign.id", "operator": "IN", "value": campaign_ids}
            ]
            params["filtering"] = json.dumps(filtering)

        all_records = []
        unmapped_dmas: set[str] = set()
        total_rows = 0
        path = f"{self.config.ad_account_id}/insights"

        while path:
            data = self._get(path, params)
            for row in data.get("data", []):
                total_rows += 1
                dma_name = row.get("dma", "")
                dma_code = self._dma_name_to_code(dma_name)
                if dma_code:
                    all_records.append({
                        "date": pd.Timestamp(row["date_start"]).date(),
                        "dma_code": dma_code,
                        "dma_name": dma_name,
                        "spend": float(row.get("spend", 0)),
                        "impressions": int(row.get("impressions", 0)),
                        "clicks": int(row.get("clicks", 0)),
                    })
                else:
                    unmapped_dmas.add(dma_name)
            # Pagination — Meta returns a full URL for the next page
            paging = data.get("paging", {})
            next_url = paging.get("next")
            if next_url:
                path = next_url  # Pass full URL; _get() handles it
                params = {}  # All params are embedded in the URL
            else:
                path = None  # type: ignore

        # Log mapping coverage
        if total_rows > 0:
            n_mapped = len(all_records)
            coverage_pct = n_mapped / total_rows
            if unmapped_dmas:
                logger.warning(
                    f"Facebook DMA mapping: {n_mapped}/{total_rows} rows mapped "
                    f"({coverage_pct:.0%}). {len(unmapped_dmas)} DMA names unmapped: "
                    f"{sorted(unmapped_dmas)}"
                )
            else:
                logger.info(
                    f"Facebook: {n_mapped} rows, all DMAs mapped successfully"
                )

        return pd.DataFrame(all_records)

    def get_campaign_spend_by_dma(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str],
    ) -> pd.DataFrame:
        """Convenience: fetch spend for specific campaigns by DMA."""
        return self.fetch_spend_by_dma(start_date, end_date, campaign_ids)

    @staticmethod
    def _dma_name_to_code(dma_name: str) -> str | None:
        """Map a Meta DMA name to a Nielsen DMA code.

        Meta returns DMA names when breakdowns=dma. We map these
        to the standard Nielsen DMA codes used throughout the platform.

        Tries: exact match → case-insensitive match → strip state suffix.
        """
        if not dma_name:
            return None

        # Direct lookup (exact match)
        code = _META_DMA_TO_NIELSEN.get(dma_name)
        if code:
            return code

        # Case-insensitive / whitespace-normalized lookup
        normalized = dma_name.lower().strip()
        code = _META_DMA_NORMALIZED.get(normalized)
        if code:
            return code

        # Try without state suffix: "City, ST" → look up "City"
        if "," in dma_name:
            base_name = dma_name.split(",")[0].strip()
            code = _META_DMA_TO_NIELSEN.get(base_name)
            if code:
                return code
            code = _META_DMA_NORMALIZED.get(base_name.lower().strip())
            if code:
                return code

        logger.debug(f"No DMA code mapping for: {dma_name}")
        return None

    # ------------------------------------------------------------------
    # Targeting deployment (execute / revert)
    # ------------------------------------------------------------------

    def get_campaign_targeting(self, campaign_id: str) -> dict:
        """Read current geo-targeting for a campaign.

        Returns the full targeting spec including geo_locations and
        excluded_geo_locations so it can be restored after the test.
        """
        data = self._get(campaign_id, params={"fields": "targeting,name,status"})
        return data.get("targeting", {})

    def get_adset_targeting(self, adset_id: str) -> dict:
        """Read current geo-targeting for an ad set."""
        data = self._get(adset_id, params={"fields": "targeting,name,status"})
        return data.get("targeting", {})

    def get_active_adsets(self, campaign_id: str) -> list[dict]:
        """Get all active ad sets for a campaign.

        Facebook targeting lives at the ad set level, not campaign level.
        We need to modify each ad set to apply DMA exclusions.
        """
        data = self._get(
            f"{campaign_id}/adsets",
            params={
                "fields": "id,name,status,targeting",
                "limit": 500,
                "filtering": json.dumps([
                    {"field": "effective_status", "operator": "IN",
                     "value": ["ACTIVE", "PAUSED"]}
                ]),
            },
        )
        return data.get("data", [])

    def get_all_campaign_ids(self) -> list[str]:
        """Get all active campaign IDs in the ad account."""
        data = self._get(
            f"{self.config.ad_account_id}/campaigns",
            params={
                "fields": "id,name,status",
                "limit": 500,
                "filtering": json.dumps([
                    {"field": "effective_status", "operator": "IN",
                     "value": ["ACTIVE", "PAUSED"]}
                ]),
            },
        )
        return [c["id"] for c in data.get("data", [])]

    def deploy_holdout(
        self,
        holdout_dma_codes: list[str],
        campaign_ids: list[str] | None = None,
    ) -> dict[str, dict]:
        """Deploy DMA holdout exclusions to Facebook ad sets.

        Saves original targeting for each ad set before modifying, so it
        can be reverted later. For channel-level tests, applies to ALL
        active campaigns. For campaign-level, applies only to specified campaigns.

        Args:
            holdout_dma_codes: Nielsen DMA codes to exclude from ad delivery.
            campaign_ids: Specific campaign IDs (None = all active campaigns).

        Returns:
            Dict mapping ad set IDs to their original targeting.
        """
        if campaign_ids is None:
            campaign_ids = self.get_all_campaign_ids()
            logger.info(f"Channel-level test: found {len(campaign_ids)} active campaigns")

        original_targeting: dict[str, dict] = {}
        exclusion_spec = get_exclusion_targeting_spec(holdout_dma_codes)
        n_updated = 0

        for campaign_id in campaign_ids:
            adsets = self.get_active_adsets(campaign_id)
            logger.info(
                f"Campaign {campaign_id}: {len(adsets)} ad sets to update"
            )

            for adset in adsets:
                adset_id = adset["id"]
                current_targeting = adset.get("targeting", {})

                # Save original targeting state
                original_targeting[adset_id] = {
                    "targeting": current_targeting.copy(),
                    "campaign_id": campaign_id,
                    "adset_name": adset.get("name", ""),
                }

                # Merge holdout exclusions into existing targeting
                updated_targeting = current_targeting.copy()
                existing_excluded = updated_targeting.get("excluded_geo_locations", {})
                existing_markets = existing_excluded.get("geo_markets", [])

                # Add holdout DMA exclusions (avoid duplicates)
                existing_keys = {m.get("key") for m in existing_markets}
                new_markets = [
                    m for m in exclusion_spec["excluded_geo_locations"]["geo_markets"]
                    if m["key"] not in existing_keys
                ]
                all_markets = existing_markets + new_markets

                updated_targeting["excluded_geo_locations"] = {
                    "geo_markets": all_markets,
                }

                # Apply the update
                self._post(adset_id, data={
                    "targeting": json.dumps(updated_targeting),
                })
                n_updated += 1
                logger.info(
                    f"  Updated ad set {adset_id} ({adset.get('name', '')}): "
                    f"excluded {len(new_markets)} holdout DMAs"
                )

        logger.info(
            f"Deployed holdout to {n_updated} ad sets across "
            f"{len(campaign_ids)} campaigns. "
            f"Excluding {len(holdout_dma_codes)} DMAs."
        )
        return original_targeting

    def deploy_holdout_update(
        self,
        holdout_dma_codes: list[str],
        already_excluded_adset_ids: set[str],
        campaign_ids: list[str] | None = None,
    ) -> tuple[dict[str, dict], int]:
        """Scan ad sets and apply holdout exclusions to any new ones.

        Only processes ad sets NOT already in already_excluded_adset_ids.
        This catches new campaigns/ad sets created after the initial deploy.

        Args:
            holdout_dma_codes: Nielsen DMA codes to exclude.
            already_excluded_adset_ids: Ad set IDs already tracked from deploy.
            campaign_ids: Specific campaigns to scan (None = all active campaigns).

        Returns:
            Tuple of (new original_targeting entries, count of newly updated ad sets).
        """
        all_campaign_ids = campaign_ids or self.get_all_campaign_ids()
        exclusion_spec = get_exclusion_targeting_spec(holdout_dma_codes)
        new_targeting: dict[str, dict] = {}
        n_new = 0

        for campaign_id in all_campaign_ids:
            adsets = self.get_active_adsets(campaign_id)
            for adset in adsets:
                adset_id = adset["id"]
                if adset_id in already_excluded_adset_ids:
                    continue  # Already managed

                current_targeting = adset.get("targeting", {})

                # Check if holdout DMAs are already excluded
                existing_excluded = current_targeting.get("excluded_geo_locations", {})
                existing_keys = {
                    m.get("key") for m in existing_excluded.get("geo_markets", [])
                }
                holdout_set = set(holdout_dma_codes)
                if holdout_set.issubset(existing_keys):
                    continue  # Already has all exclusions

                # Save original and apply exclusions
                new_targeting[adset_id] = {
                    "targeting": current_targeting.copy(),
                    "campaign_id": campaign_id,
                    "adset_name": adset.get("name", ""),
                }

                updated_targeting = current_targeting.copy()
                existing_markets = existing_excluded.get("geo_markets", [])
                new_markets = [
                    m for m in exclusion_spec["excluded_geo_locations"]["geo_markets"]
                    if m["key"] not in existing_keys
                ]
                updated_targeting["excluded_geo_locations"] = {
                    "geo_markets": existing_markets + new_markets,
                }

                self._post(adset_id, data={
                    "targeting": json.dumps(updated_targeting),
                })
                n_new += 1
                logger.info(
                    f"  NEW ad set {adset_id} ({adset.get('name', '')}): "
                    f"excluded {len(new_markets)} holdout DMAs"
                )

        logger.info(
            f"Holdout update: scanned {len(all_campaign_ids)} campaigns, "
            f"found {n_new} new ad sets to exclude."
        )
        return new_targeting, n_new

    def revert_holdout(self, original_targeting: dict[str, dict]) -> int:
        """Revert ad sets back to their pre-test targeting.

        Args:
            original_targeting: Dict from deploy_holdout() mapping ad set IDs
                                to their original targeting state.

        Returns:
            Number of ad sets successfully reverted.
        """
        n_reverted = 0
        n_failed = 0

        for adset_id, state in original_targeting.items():
            try:
                self._post(adset_id, data={
                    "targeting": json.dumps(state["targeting"]),
                })
                n_reverted += 1
                logger.info(
                    f"Reverted ad set {adset_id} ({state.get('adset_name', '')})"
                )
            except Exception as e:
                n_failed += 1
                logger.error(
                    f"FAILED to revert ad set {adset_id}: {e}. "
                    f"Manual revert required!"
                )

        logger.info(
            f"Reverted {n_reverted}/{len(original_targeting)} ad sets. "
            f"{n_failed} failures."
        )
        if n_failed > 0:
            logger.error(
                f"WARNING: {n_failed} ad sets could not be reverted automatically. "
                f"Check Facebook Ads Manager and restore geo targeting manually."
            )
        return n_reverted


def get_dma_geo_targeting_spec(dma_codes: list[str]) -> dict:
    """Build a Meta geo_locations targeting spec for specific DMAs.

    Use this when setting up holdout tests to exclude DMAs from ad delivery.
    """
    return {
        "geo_locations": {
            "geo_markets": [
                {"key": code, "market_type": "dma"}
                for code in dma_codes
            ],
        },
    }


def get_exclusion_targeting_spec(holdout_dma_codes: list[str]) -> dict:
    """Build an exclusion targeting spec to hold out specific DMAs.

    Apply this to campaigns during the test period to create the holdout.
    """
    return {
        "excluded_geo_locations": {
            "geo_markets": [
                {"key": code, "market_type": "dma"}
                for code in holdout_dma_codes
            ],
        },
    }
