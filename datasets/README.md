# Data for Tidewatch

The default simulation uses generated sonar contacts and hydrophone audio. No real recordings, sonar images or trained weights are bundled. The sources below were checked on 11 October 2026. Public access and permission to reuse are separate questions; the notes describe what the source actually states.

## Sources found

| Source | What it contains | How it fits | Access and limits |
| --- | --- | --- | --- |
| [NOAA passive acoustic archive](https://www.ncei.noaa.gov/products/passive-acoustic-data) | Underwater audio, environmental noise, biological sounds and human-made sources | Start with a short original-speed WAV to test acoustic replay and STFT features | Free archive access; retain the individual dataset's README, citation and terms. NOAA also distributes partner data, so the whole archive should not be assumed to have one license. |
| [DeepShip](https://github.com/irfankamboh/DeepShip) | Real recordings of cargo ships, tankers, passenger ships and tugs | Vessel-noise evaluation after checking recording provenance and splits | The author's repository provides part of the data and an email route for the remainder. I did not find an explicit reuse license in the repository; no recordings have been copied here. |
| [ShipsEar](https://underwaternoise.atlanttic.uvigo.es/) | Shallow-water vessel recordings with acquisition metadata | A second environment for acoustic evaluation | The site offers samples and asks users to request full access. It does not give an unrestricted redistribution license on the landing page. |
| [Marine Debris FLS](https://github.com/mvaldenegro/marine-debris-fls-datasets) | ARIS forward-looking sonar images from tank, turntable and quarry settings | Future sonar image detection/segmentation adapter | The [authors' paper](https://arxiv.org/html/2503.22880v1) specifies noncommercial/share-alike licensing. The current scalar sonar classifier cannot consume these images directly. These are debris examples, not shark labels. |
| [NOAA ETOPO 2022](https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.html) | Global bathymetric/topographic elevation | Regional mission planning and later terrain import | The provider permits free use and redistribution. Its 15 arc-second grid is much coarser than our small local obstacle scene; it cannot validate metre-scale diver clearance. |
| [NOAA CO-OPS API](https://api.tidesandcurrents.noaa.gov/api/dev) | Measured current speed/direction and separately identified current predictions | Recorded current scenarios after importing a station, bin and time range | Use `product=currents` for measurements. Metric current speed is **cm/s**, so convert to m/s. Preserve station/bin depth, time zone and data flags. Wind must not be substituted for underwater current. |

NOAA audio is the first candidate for a small replay experiment. DeepShip and ShipsEar can add vessel examples once their intended-use terms are settled. The sonar images are a separate future input; they are not silently mapped into simulated echo features.

## Replay a local recording

Choose a clip you are allowed to use and keep its source URL, dataset citation, recording date, hydrophone details and license beside your experiment notes. Use original-speed audio. Some archive examples are sped up for listening and are unsuitable for frequency analysis without reversing that transformation.

Tidewatch accepts an uncompressed PCM16 WAV at the configured sample rate, normally 8000 Hz. Clips are limited to 60 seconds. If needed, convert a local original with FFmpeg:

```bash
ffmpeg -i /path/to/original.wav -t 60 -ac 1 -ar 8000 \
  -c:a pcm_s16le datasets/hydrophone.wav
source /opt/ros/lyrical/setup.bash
python3 scripts/run_simulation.py --hydrophone-recording datasets/hydrophone.wav
```

Or set these fields in `config/mission.yaml`:

```yaml
perception:
  sample_rate_hz: 8000
  hydrophone_recording: /absolute/path/to/hydrophone.wav
  hydrophone_recording_loop: false
```

Playback ends in silence unless looping is explicitly enabled. The dashboard identifies recorded audio separately from synthetic sonar. The same replay is scene-level background input for the three virtual hydrophones; it does not simulate range-dependent propagation or establish which sonar contact made a sound. Audio remains local and never enters acoustic mesh packets. PCM amplitude is not a calibrated sound-pressure level.

The recording loader is implemented and tested, but no internet recording has been downloaded or activated in this checkout. This environment can browse source descriptions but cannot open the network sockets needed to fetch data files or run the local dashboard. The default remains synthetic.

## Classifier interface

Live edge processing uses `perception.EdgePerception`. The default NumPy baseline analyzes a 1024-sample frame with a 512-sample Hann window and a 128-sample hop. Optional acoustic TorchScript input is `[batch, 1, frequency, time]`; it uses log power relative to the frame maximum, clipped to -80..0 dB and scaled to -1..1.

The optional sonar model takes ten numerical features in `SONAR_FEATURE_NAMES` order:

| Feature | Divisor |
| --- | --- |
| `range_m` | 30 |
| `echo_strength` | 1 |
| `size_m` | 3 |
| `speed_m_s` | 2 |
| `elongation` | 8 |
| `specular_ratio` | 1 |
| `echo_roughness` | 1 |
| `micro_doppler_hz` | 50 |
| `echo_spread_m` | 3 |
| `measurement_quality` | 1 |

Normalized sonar inputs are clipped to 0..2. Outputs are class logits ordered `unknown`, `diver`, `marine_life`, `vehicle`, `obstacle`. Model paths are configured with `acoustic_model_path` and `sonar_model_path`; null keeps the baseline. Real sonar needs a calibrated extractor or an image model, not synthetic numbers reconstructed from class labels. The older `acoustics.DualPipelineClassifier` uses a different interface and is not the live ROS classifier.

Before training, split by recording session, physical source and site, then cut audio windows. Windows from one recording must stay in the same split. Keep an unfamiliar-source test set and report false alerts, class precision/recall and unknown rejection alongside accuracy. Recorded noise alone does not create a trained detector or prove real-world protection.
