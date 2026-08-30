# Preview and review

Status: **complete**.

Preview generation creates bounded source/distorted side-by-side JPEGs,
optional difference images, an HTML table, and Markdown index beneath the
external artifact preview root. Labels include source page, generated page,
profile, and severity. The default limit is 10 and the hard limit is 100.

Preview artifacts are ignored by Git and may be lifecycle-managed with the
dry-run-first `cleanup-preview` command.

