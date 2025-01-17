
This archive contains programs for downloading and analyzing solar wind data from multiple spacecraft.
Detailed science information may be found at https://sites.google.com/a/cfa.harvard.edu/jakub-prchlik/home/shock-finding.
In addition, more coding information may be found at the associated `READTHEDOCS <https://solar-wind-matching.readthedocs.io/en/latest/index.html>`_ page.
The directory structure is as follows:

cdf
---
Database of cdf files downloaded from ftp://spdf.gsfc.nasa.gov/pub/data/. Note that SOHO/CELIAS data has its own program because it is not available at cdaw.

code
----
Programs that analyze solar Wind discontinuities. 
The primary programs have the following functionality:
Find discontinuities given jumps in plasma parameters,
match time offsets for a single discontinuity between four L1 spacecraft,
and match time offsets continously for four L1 spacecraft and predict planar arrival time at THEMIS/ARTEMIS spacecraft.

comb_data
---------
Old, contained merged Wind, ACE, and SOHO data sets. Time resolution too low to be interesting.

html_files
---------
Files containing summaries of spacecraft event discontinuities observed in wind and the time offsets with the other 3 L1 spacecraft.
Use https://rawgit.com/ to view the html files over the internet (e.g. https://cdn.rawgit.com/jprchlik/solar_wind_jets/25aca910/html_files/wind_level_0999_full_res.html).

plots
-----
Directory of plots linked in html_files.