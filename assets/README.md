# assets/ — sample sky photos

Full-sky photos used as one-click **examples** in the app (`gr.Examples`, wired in
`app.py` via `EXAMPLE_PHOTOS`) and for the demo video. They ship inside the Space.

- Full-sky scenes (clouds against blue) so MobileSAM has something to isolate when
  you click — not tight single-cloud crops.
- Resized to **1600px** longest edge to keep the Space repo light
  (`sips -Z 1600 -s formatOptions 80 *.jpeg`).
- Drop more in here and they appear automatically (`.jpg` / `.jpeg` / `.png`).
