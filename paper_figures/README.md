# Figure 3: total work and longest coherent chain

From the archive root, install the figure-only dependencies in your chosen
Python environment and run:

```sh
python -m pip install -r paper_figures/requirements-figures.txt
python paper_figures/render_resource_frontier.py --data-root . --output rendered_figure3
```

The script reads the archive-root `controls.csv` and `resources.csv` and exports
PDF, SVG and PNG figures plus `figure_values.json`. The distributed JSON records
the source hashes and exact selected resource integers. The raw response name
`nodes` denotes the cubic response used in the manuscript; the data dictionary
retains this mapping. No scientific data or selected parameters are changed.

Panel (a) displays complete public T counts in millions. Panel (b) displays
maximum consecutive oracle calls. Both axes begin at zero. The logical widths
586, 587, 586 and 588 remain in manuscript Table 2 and in the JSON; they are not
a third panel. The comparison uses the principal n=68 cubic-response schedules:
three-run standard sign QPE, one-run sine QPE, one-run four-shift randomized-grid
QPE and one-run fourfold zero-padded measurement.

This is plotting-source reproduction, separate from certificate or gate-resource
replay. The optional `--qa-tools` argument accepts an external panel-alignment
auditor; it is not needed to regenerate the figure. Dependencies and fonts are
installed separately and are not bundled here.
